import streamlit as st
import google.generativeai as genai
import psycopg2
import pandas as pd
import numpy as np
import logging
import chromadb
import uuid
import re
from typing import Tuple, Optional, Dict, Any

# ==============================================================================
# --- ⚙ PAGE CONFIGURATION & HEADER ---
# ==============================================================================
st.set_page_config(page_title="AI Data Explorer with RAG", page_icon="🧠", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_header():
    """Creates a consistent header with logo and navigation buttons."""
    # Center the logo at the top
    st.image("logo.png", width=350)
    st.caption(f"Displaying real-time data from **July 2025 to August 2025**.")

    # UPDATED: Now uses 5 columns for the 5 buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Use type="primary" to indicate this is the current (home) page
        if st.button("Dashboard"):
            st.switch_page("app.py")

    with col2:
        if st.button("Closest Buoy Finder"):
            st.switch_page("pages/page1.py")

    with col3:
        if st.button("Chat AI RAG", type="primary"):
            st.switch_page("pages/page2.py")
    
    with col4:
        if st.button("Learn More Videos"):
            st.switch_page("pages/page3.py")

create_header()

st.title("🧠 AI Conversational Data Explorer with RAG")
st.markdown(
    """
    This chatbot has a persistent memory of the data you've queried.
    Ask for data, and it will be summarized and stored. Your follow-up questions will retrieve the most relevant context.
    """
)

# ==============================================================================
# --- 🔐 API & DATABASE CONNECTION ---
# ==============================================================================
try:
    genai.configure(api_key=st.secrets["gemini_api_key"])
except (FileNotFoundError, KeyError):
    st.error("Gemini API key not found. Please add it to your Streamlit secrets.")
    st.stop()

@st.cache_resource
def init_connection():
    try:
        conn = psycopg2.connect(**st.secrets["postgres"])
        logger.info("Database connection established.")
        return conn
    except Exception as e:
        st.error(f"Error connecting to PostgreSQL: {e}")
        st.stop()

conn = init_connection()

# ==============================================================================
# --- 🧠 RAG SETUP (Vector DB & Embeddings) ---
# ==============================================================================
@st.cache_resource
def setup_rag_system():
    """Initializes ChromaDB client and a collection."""
    try:
        # ChromaDB 1.x: use PersistentClient(path=...) for on-disk persistence.
        # Older 0.4.x: use Client(Settings(..., persist_directory=...)).
        # Fallback: ephemeral in-memory Client() if persistence isn't available.
        try:
            # Preferred path for persistence (cross-platform, relative to project)
            client = chromadb.PersistentClient(path="/mnt/chroma_blob")
            logger.info("Worked")
        except AttributeError:
            # chromadb < 1.0 fallback
            try:
                import importlib
                cfg_mod = importlib.import_module("chromadb.config")
                Settings = getattr(cfg_mod, "Settings", None)
                if Settings is None:
                    raise ImportError("chromadb.config.Settings not available")
                client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="/home/chroma_db"))
            except Exception:
                # Last resort: in-memory (non-persistent)
                client = chromadb.Client()
        collection = client.get_or_create_collection(name="data_summaries_rag")
        logger.info("ChromaDB client and collection initialized.")
        return collection
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        st.stop()

rag_collection = setup_rag_system()

# ==============================================================================
# --- 🧠 CORE HELPER FUNCTIONS ---
# ==============================================================================

# --- Part 1: SQL Generation & Execution ---
@st.cache_data(ttl=600)
def get_db_schema() -> Optional[str]:
    # (This function is unchanged)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';")
            tables = [row[0] for row in cur.fetchall()]
            schema_parts = []
            for table in tables:
                schema_parts.append(f"-- Table: {table}")
                cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = '{table}';")
                columns = [f"    - {row[0]} ({row[1]})" for row in cur.fetchall()]
                schema_parts.append("\n".join(columns))
            return "\n\n".join(schema_parts)
    except Exception as e:
        st.error(f"Could not retrieve database schema: {e}")
        return None

def generate_sql_with_gemini(user_prompt: str, schema: str) -> str:
    # (This function is unchanged)
    system_instruction = """You are an expert PostgreSQL data analyst. Your task is to write a single, valid PostgreSQL SQL query based on the user's request and the provided database schema.

RULES:
- ONLY use the tables and columns listed in the provided schema.
- When the user asks for observational data (like temperature, salinity, etc.), you MUST join observations (o) with profiles (p) to include location and time. Join condition: o.platform_number = p.float_id AND o.cycle_number = p.cycle_number.
- ALWAYS include a LIMIT clause (e.g., LIMIT 10000).
- If the user's request seems impossible based on the schema, return "ERROR: I cannot answer that question with the available data schema."
- Respond with ONLY the raw SQL query.
"""
    prompt_with_schema = f"Database schema:\n{schema}\n\nUser prompt: {user_prompt}"

    try:
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction)
        response = model.generate_content(prompt_with_schema, generation_config=genai.types.GenerationConfig(temperature=0.0))
        sql_query = response.text.strip().replace("sql", "").replace("", "")
        return clean_sql(sql_query).strip()
    except Exception as e:
        return f"ERROR: Failed to generate SQL due to an API error: {e}"

def clean_sql(sql_text: str) -> str:
    sql_text = sql_text.strip()
    if sql_text.startswith("'''") and sql_text.endswith("'''"):
        sql_text = sql_text[3:-3].strip()
    if sql_text.startswith("```") and sql_text.endswith("```"):
        sql_text = sql_text[3:-3].strip()
    return sql_text


def _sanitize_sql(sql: str) -> str:
    """Remove BOM/zero-width/nbsp and trim."""
    return re.sub(r'[\u200B-\u200D\uFEFF\u00A0]', ' ', sql).strip()

def _fix_pg_group_by_aliases(sql: str) -> str:
    """Replace GROUP BY aliases with positional indexes if we detect lat/lon aliasing."""
    s = sql
    if re.search(r'AS\s+lat_grid', s, re.IGNORECASE) and re.search(r'AS\s+lon_grid', s, re.IGNORECASE):
        s = re.sub(r'GROUP\s+BY\s+lat_grid\s*,\s*lon_grid', 'GROUP BY 1, 2', s, flags=re.IGNORECASE)
    return s

# Add this function near your other helpers
def sanitize_sql_query(sql: str) -> str:
    sql = sql.replace("`", "")  # Remove backticks
    sql = sql.replace('"', "")
    sql = sql.replace("sql", "")
    sql = sql.strip()
    sql = "\n".join([line for line in sql.splitlines() if line.strip()])
    return sql

@st.cache_data(show_spinner=False)
def run_query(query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Execute SQL and return (DataFrame, error)."""
    try:
        q = sanitize_sql_query(query)
        print("Executing SQL:", q)  # For debugging
        try:
            df = pd.read_sql_query(q, conn)
            return df, None
        except Exception:
            # Retry with GROUP BY alias fix for Postgres
            q2 = _fix_pg_group_by_aliases(q)
            if q2 != q:
                df = pd.read_sql_query(q2, conn)
                return df, None
            raise
    except Exception as e:
        return None, f"Database error: {e}"

# --- Part 2: RAG Ingestion (Creating and Storing Summaries) ---
def generate_and_store_summary(df: pd.DataFrame, sql_query: str):
    """Generates a detailed text summary of a DataFrame and stores it in ChromaDB."""
    if df.empty:
        return

    # 1. Create the summary text document
    numeric_stats = df.select_dtypes(include=np.number).describe().to_string()
    schema_info = df.info(verbose=False, buf=None)
    schema_str = str(schema_info) if schema_info else "No schema info available."


    summary_prompt = f"""
You are a data analyst. Create a concise, human-readable summary paragraph for the following dataset.
The data was retrieved using this SQL query:
{sql_query}

Here are the column names and data types:
{schema_str}

Here are the summary statistics for the numeric columns:
{numeric_stats}

Based on all this information, write a single paragraph that describes the key aspects of this dataset.
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(summary_prompt)
        natural_language_summary = response.text

        # Combine everything into a single document for embedding
        full_document = f"""
Context Summary for a Dataset:
Query: {sql_query}
Summary: {natural_language_summary}
Schema:
{schema_str}
Statistics:
{numeric_stats}
"""
        # 2. Generate embedding for the document
        embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=full_document,
            task_type="retrieval_document"
        )
        embedding = embedding_result['embedding']
        doc_id = f"doc_{uuid.uuid4()}"

        # 3. Store in ChromaDB
        rag_collection.add(
            embeddings=[embedding],
            documents=[full_document],
            ids=[doc_id]
        )
        logger.info(f"Stored document {doc_id} in ChromaDB.")
        st.toast(f"✅ Context stored in memory!")

    except Exception as e:
        logger.error(f"Failed during RAG ingestion: {e}")
        st.warning(f"Could not store data summary in RAG memory: {e}")


# --- Part 3: RAG Retrieval & Generation (Answering Follow-ups) ---
def answer_with_rag(question: str) -> str:
    """Answers a follow-up question using the RAG system."""
    try:
        # 1. Embed the user's question
        question_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=question,
            task_type="retrieval_query"
        )['embedding']

        # 2. Query ChromaDB to find the most relevant context
        results = rag_collection.query(
            query_embeddings=[question_embedding],
            n_results=2 # Retrieve the top 2 most relevant summaries
        )
        retrieved_documents = "\n---\n".join(results['documents'][0])

        # 3. Generate the final answer with the retrieved context
        rag_prompt = f"""You are a helpful and precise data assistant.
Answer the user's question based ONLY on the provided context below.
If the context is insufficient to answer the question, state that clearly. Do not use any external knowledge.

[CONTEXT FROM PREVIOUSLY QUERIED DATA]
{retrieved_documents}

[USER'S QUESTION]
{question}

Answer:
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        final_response = model.generate_content(rag_prompt)
        return final_response.text

    except Exception as e:
        logger.error(f"Failed during RAG retrieval/generation: {e}")
        return f"An error occurred while answering your question: {e}"


# --- Visualization, Classification, etc. (Mostly unchanged) ---
def get_visualization_suggestion(sql_query: str) -> str: # Unchanged
    """
    Sends the SQL query to a Gemini model to get a visualization suggestion.
    """
    system_instruction = """
You are a data visualization expert. Your task is to analyze a SQL query and its potential output, then suggest the best way to visualize it in Streamlit.

You know about the following tables:
floats: Contains float metadata like deployment_date, platform_type, deployment_lat, deployment_lon. Best visualized with st.map, st.bar_chart, st.line_chart.
profiles: Contains individual float cycle data like time, latitude, longitude. Best visualized with st.pydeck_chart, st.line_chart, st.bar_chart, st.map.
observations: Contains sensor readings like temp, psal, doxy. Best visualized with st.plotly_chart, st.line_chart, st.scatter_chart, st.bar_chart, st.dataframe, st.metric.

Based on the user's SQL query, you must respond with a SINGLE LINE in the following format:
visualization_function x_column y_column [optional_color_column]

### RULES:
1.  *Analyze the SQL SELECT statement*: The columns selected will determine the best chart.
2.  *Bar Chart*: For categorical data. st.bar_chart categorical_column numerical_column
3.  *Line Chart*: For time-series or sequential data. st.line_chart date_or_sequence_column numerical_column
4.  *Scatter Chart*: For comparing two numerical columns. st.scatter_chart numerical_column_1 numerical_column_2 [optional_categorical_color_column]
5.  *Map: If the query selects geographical coordinates. The column names for latitude and longitude *must be returned. st.map latitude_column longitude_column
6.  *Metric: If the query returns a single numerical value (e.g., COUNT(), AVG(temp)). st.metric single_value_column None
7.  *Dataframe*: If the result is a simple list or doesn't fit other charts well. st.dataframe None None
8.  *NO EXTRA TEXT*: Your response must be ONLY the single line.

### Examples:
- User SQL: SELECT platform_type, COUNT(*) as count FROM floats GROUP BY platform_type;
- Your Response: st.bar_chart platform_type count

- User SQL: SELECT deployment_date, float_id FROM floats ORDER BY deployment_date;
- Your Response: st.line_chart deployment_date float_id

- User SQL: SELECT temp, psal, platform_number FROM observations WHERE temp > 25 LIMIT 100;
- Your Response: st.scatter_chart temp psal platform_number

- User SQL: SELECT deployment_lat, deployment_lon FROM floats LIMIT 100;
- Your Response: st.map deployment_lat deployment_lon

- User SQL: SELECT COUNT(*) as total FROM observations;
- Your Response: st.metric total None
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction)
        generation_config = genai.types.GenerationConfig(temperature=0.5)
        response = model.generate_content(sql_query, generation_config=generation_config)
        return response.text.strip()
    except Exception as e: 
        logger.error(f"Gemini viz suggestion error: {e}")
        return "st.dataframe"

def render_chart_from_payload(payload: Dict[str, Any]): # Unchanged
    # (Logic for rendering charts remains the same)
    try:
        viz_type = payload.get("type", "dataframe")
        df = payload.get("data")
        params = payload.get("params", {})

        # Handle map explicitly: st.map does not take lat_col/lon_col kwargs
        if viz_type == "map":
            # DataFrame already renamed to 'lat'/'lon' in render_chart_from_suggestion
            st.map(df)
            return

        viz_func = getattr(st, viz_type, st.dataframe)
        # Filter out any map-only params if present
        safe_params = {k: v for k, v in params.items() if k not in ("lat_col", "lon_col")}
        viz_func(df, **safe_params)
    except Exception as e:
        st.error(f"Failed to render visualization: {e}")

def render_chart_from_suggestion(df: pd.DataFrame, suggestion: str) -> Optional[Dict[str, Any]]:
    # Auto-detect latitude/longitude columns (supports lat_grid/lon_grid, deployment_lat/deployment_lon, etc.)
    def _detect_lat_lon(columns: list[str]) -> Tuple[Optional[str], Optional[str]]:
        cols_l = [c.lower() for c in columns]
        lat_syns = {"lat", "latitude", "deployment_lat", "lat_grid", "latdeg", "lat_deg", "latbin", "lat_bin"}
        lon_syns = {"lon", "longitude", "deployment_lon", "lon_grid", "londeg", "lon_deg", "lonbin", "lon_bin"}
        lat = next((c for c in columns if c.lower() in lat_syns), None)
        lon = next((c for c in columns if c.lower() in lon_syns), None)
        return lat, lon

    try:
        parts = suggestion.split()
        viz_func_name = parts[0].replace("st.", "") if parts else "dataframe"
        payload = {"type": viz_func_name, "data": df, "params": {}}

        if viz_func_name == "map":
            # Try suggestion args first
            lat_col = parts[1] if len(parts) >= 2 else None
            lon_col = parts[2] if len(parts) >= 3 else None

            if not lat_col or lat_col not in df.columns or not lon_col or lon_col not in df.columns:
                # Auto-detect if suggestion columns are missing/mismatched
                lat_col, lon_col = _detect_lat_lon(df.columns.tolist())

            if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
                map_df = df[[lat_col, lon_col]].rename(columns={lat_col: "lat", lon_col: "lon"})
                st.map(map_df)
                # Return payload without unsupported kwargs
                return {"type": "map", "data": map_df, "params": {}}
            else:
                st.warning(
                    f"Could not detect latitude/longitude columns for map. "
                    f"Available columns: {', '.join(df.columns)}. Showing table instead."
                )
                st.dataframe(df)
                return {"type": "dataframe", "data": df, "params": {}}

        elif viz_func_name == "bar_chart" and len(parts) >= 3:
            x, y = parts[1], parts[2]
            if x in df.columns and y in df.columns:
                st.bar_chart(df, x=x, y=y)
                payload["params"] = {"x": x, "y": y}
                return payload

        elif viz_func_name == "line_chart" and len(parts) >= 3:
            x, y = parts[1], parts[2]
            if x in df.columns and y in df.columns:
                st.line_chart(df, x=x, y=y)
                payload["params"] = {"x": x, "y": y}
                return payload

        elif viz_func_name == "scatter_chart" and len(parts) >= 3:
            params = {"x": parts[1], "y": parts[2]}
            if len(parts) >= 4 and parts[3] in df.columns:
                params["color"] = parts[3]
            if params["x"] in df.columns and params["y"] in df.columns:
                st.scatter_chart(df, **params)
                return {"type": "scatter_chart", "data": df, "params": params}

        # Fallback
        st.dataframe(df)
        return {"type": "dataframe", "data": df, "params": {}}

    except Exception as e:
        st.error(f"Failed to render visualization from suggestion '{suggestion}': {e}")
        return {"type": "dataframe", "data": df, "params": {}}

# def classify_prompt(prompt: str) -> str:
#     # Simplified: If there are documents in the DB, assume it might be a follow-up.
#     # A more robust solution might still use an LLM call here.
#     if rag_collection.count() > 0:
#         # A simple keyword-based classifier can be a good-enough starting point
#         follow_up_keywords = ['what is', 'explain', 'average', 'compare', 'which', 'how many', 'summarize']
#         if any(keyword in prompt.lower() for keyword in follow_up_keywords):
#             return "FOLLOW_UP"
#     return "NEW_QUERY"

def classify_prompt(prompt: str, context: str) -> str:
    """Uses Gemini to decide if a prompt is a new query or a follow-up."""
    if not context: # If no context, it must be a new query
        return "NEW_QUERY"

    system_instruction = """You are a classification model. Your task is to determine if a user's prompt is asking for NEW data or is a FOLLOW-UP question about the data they are currently viewing.
- *NEW_QUERY*: The user wants different information, a new chart, or a completely different slice of data. Examples: "Show me all floats", "What about pressure data?", "Graph salinity vs temperature".
- *FOLLOW_UP*: The user is asking for clarification, summary, or calculation based on the data ALREADY SHOWN. Examples: "What's the average of the temp column?", "Which platform had the highest value?", "Can you explain these results?".

Respond with ONLY the string "NEW_QUERY" or "FOLLOW_UP". Do not add any other text.
"""
    full_prompt = f"Current data context: {context}\n\nUser's new prompt: \"{prompt}\"\n\nClassification:"
    try:
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction)
        response = model.generate_content(full_prompt, generation_config=genai.types.GenerationConfig(temperature=0))
        return response.text.strip()
    except Exception as e:
        logger.error(f"Prompt classification error: {e}")
        return "NEW_QUERY" # Default to new query on error

# ==============================================================================
# --- 🎈 STREAMLIT UI & SESSION STATE ---
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you explore the data?", "type": "text"}]
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text": st.markdown(message["content"])
        elif message["type"] == "sql_viz":
            if "sql" in message: st.code(message["sql"], language="sql")
            if "suggestion" in message: st.caption(f"Viz Suggestion: {message['suggestion']}")
            if "payload" in message and message["payload"]: render_chart_from_payload(message["payload"])
            if "df_preview" in message: st.dataframe(message["df_preview"])

if prompt := st.chat_input("Ask about data..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            classification = classify_prompt(prompt, st.session_state.conversation_context)

            if classification == "NEW_QUERY":
                db_schema = get_db_schema()
                if not db_schema: st.stop()

                sql_query = generate_sql_with_gemini(prompt, db_schema)
                if sql_query.startswith("ERROR:"):
                    st.error(sql_query)
                    st.session_state.messages.append({"role": "assistant", "content": sql_query, "type": "text"})
                else:
                    df, err = run_query(sql_query)
                    if err:
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err, "type": "text"})
                    elif df is None or df.empty:
                        st.warning("Query ran successfully but returned no data.")
                        st.session_state.messages.append({"role": "assistant", "content": "The query returned no data.", "type": "text"})
                        st.session_state.conversation_context = "The last query returned no data."
                    else:
                        st.success(f"Query returned {len(df)} rows.")
                        # RAG INGESTION
                        summary = generate_and_store_summary(df, sql_query)
                        if summary:
                            st.session_state.conversation_context = summary
                        suggestion = get_visualization_suggestion(sql_query)
                        payload = render_chart_from_suggestion(df, suggestion)
                        st.dataframe(df.head())
                        st.session_state.messages.append({
                            "role": "assistant", "type": "sql_viz", "sql": sql_query,
                            "suggestion": suggestion, "payload": payload, "df_preview": df.head()
                        })

            else: # classification == "FOLLOW_UP"
                answer = answer_with_rag(prompt) # RAG RETRIEVAL
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "type": "text"})

    st.rerun()