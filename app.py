import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import pydeck as pdk

# --- Page Configuration ---
st.set_page_config(
    page_title="ARGO Fleet Data Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Reusable Header and Navigation ---
def create_header():
    """Creates a consistent header with logo and navigation buttons."""
    # Center the logo at the top
    st.image("logo.png", width=350)
    st.caption(f"Displaying real-time data from **July 2025 to August 2025**.")

    # UPDATED: Now uses 5 columns for the 5 buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Use type="primary" to indicate this is the current (home) page
        if st.button("Dashboard", type="primary"):
            st.switch_page("app.py")

    with col2:
        if st.button("Closest Buoy Finder"):
            st.switch_page("pages/page1.py")

    with col3:
        if st.button("Chat AI RAG"):
            st.switch_page("pages/page2.py")
    
    with col4:
        if st.button("Learn More Videos"):
            st.switch_page("pages/page3.py")

create_header()


# --- Database Connection ---
@st.cache_resource
def init_connection():
    """Initializes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**st.secrets["postgres"])
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Error connecting to PostgreSQL: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unknown error occurred during connection: {e}")
        st.stop()

conn = init_connection()

@st.cache_data(ttl=600) # Cache data for 10 minutes
def run_query(query):
    """Executes a SQL query and returns the result as a Pandas DataFrame."""
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            if cur.description is None:
                return pd.DataFrame()
            
            colnames = [desc[0] for desc in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=colnames)
            return df
    except Exception as e:
        return f"Database Error: {e}"

# --- Dashboard UI ---



# --- Key Performance Indicators (KPIs) ---
st.header("📊 Fleet at a Glance", divider='blue')

kpi_queries = {
    "total_floats": """
        SELECT COUNT(DISTINCT float_id) 
        FROM profiles 
        WHERE time BETWEEN '2025-07-01' AND '2025-08-31';
    """,
    "total_profiles": "SELECT COUNT(*) FROM profiles WHERE time BETWEEN '2025-07-01' AND '2025-08-31';",
    "latest_update": "SELECT MAX(time) as last_update FROM profiles WHERE time BETWEEN '2025-07-01' AND '2025-08-31';"
}

# Fetch KPI data
total_floats_df = run_query(kpi_queries["total_floats"])
total_profiles_df = run_query(kpi_queries["total_profiles"])
latest_update_df = run_query(kpi_queries["latest_update"])

# Display KPIs in columns
col1, col2, col3 = st.columns(3)
with col1:
    if isinstance(total_floats_df, pd.DataFrame):
        st.metric(label="**Active Floats (Jul-Aug 2025)**", value=f"🛰️ {total_floats_df.iloc[0,0]}")
    else:
        st.error("Could not load Total Floats.")
with col2:
    if isinstance(total_profiles_df, pd.DataFrame):
        st.metric(label="**Profiles Collected (Jul-Aug 2025)**", value=f"📥 {total_profiles_df.iloc[0,0]:,}")
    else:
        st.error("Could not load Total Profiles.")
with col3:
    if isinstance(latest_update_df, pd.DataFrame) and not latest_update_df.empty:
        if 'last_update' in latest_update_df.columns and pd.notna(latest_update_df['last_update'].iloc[0]):
             st.metric(label="**Latest Data Transmission**", value=f"📡 {latest_update_df['last_update'].dt.strftime('%Y-%m-%d').iloc[0]}")
        else:
             st.metric(label="**Latest Data Transmission**", value="N/A")
    else:
        st.error("Could not load Latest Update.")

st.subheader("📍 Latest Known Float Positions")
map_query = """
SELECT DISTINCT ON (float_id) float_id, latitude, longitude 
FROM profiles 
WHERE time BETWEEN '2025-07-01' AND '2025-08-31'
ORDER BY float_id, time DESC;
"""
latest_positions_df = run_query(map_query)

if isinstance(latest_positions_df, pd.DataFrame) and not latest_positions_df.empty:
    latest_positions_df.dropna(subset=['latitude', 'longitude'], inplace=True)

    view_state = pdk.ViewState(
        latitude=latest_positions_df['latitude'].mean(),
        longitude=latest_positions_df['longitude'].mean(),
        zoom=1.8,
        pitch=0
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=latest_positions_df,
        get_position='[longitude, latitude]',
        get_color='[0, 100, 255, 160]',
        get_radius=50000,
        pickable=True, 
        tooltip={"html": "<b>Float ID:</b> {float_id}"}
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
    st.caption("This map shows the last reported position for each active ARGO float between July and August 2025. Each blue dot represents a single float.")

else:
    st.warning("No position data available to display on the map.")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("🛠️ Fleet Composition by Model")
    composition_query = """
    SELECT platform_type, COUNT(*) as number_of_floats 
    FROM floats 
    GROUP BY platform_type 
    ORDER BY number_of_floats DESC;
    """
    composition_df = run_query(composition_query)
    if isinstance(composition_df, pd.DataFrame) and not composition_df.empty:
        # --- NEW: Replaced st.bar_chart with Plotly for axis labels ---
        fig = px.bar(
            composition_df, 
            x='platform_type', 
            y='number_of_floats',
            labels={
                "platform_type": "Float Model",
                "number_of_floats": "Number of Floats"
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart shows the total count of floats for each model type.")
    else:
        st.warning("No fleet composition data available.")
        
with chart_col2:
    st.subheader("📈 Profile Collection Rate")
    rate_query = """
    SELECT DATE_TRUNC('week', time) as week, COUNT(*) as profile_count 
    FROM profiles 
    WHERE time BETWEEN '2025-07-01' AND '2025-08-31'
    GROUP BY week 
    ORDER BY week;
    """
    rate_df = run_query(rate_query)
    if isinstance(rate_df, pd.DataFrame) and not rate_df.empty:
        # --- NEW: Replaced st.area_chart with Plotly for axis labels ---
        fig = px.area(
            rate_df,
            x='week',
            y='profile_count',
            labels={
                "week": "Week",
                "profile_count": "Profiles Collected"
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart shows the trend of profiles collected each week.")
    else:
        st.warning("No profile rate data available.")

st.markdown("---")

# --- Interactive Deep Dive Section ---
st.header("🔬 Deep Dive into a Single Float", divider='blue')

floats_list_df = run_query("""
    SELECT DISTINCT float_id FROM profiles 
    WHERE time BETWEEN '2025-07-01' AND '2025-08-31' 
    ORDER BY float_id;
""")

if isinstance(floats_list_df, pd.DataFrame) and not floats_list_df.empty:
    
    options_list = floats_list_df['float_id'].tolist()
    default_float_str = '1901910' 
    default_index = 0
    if default_float_str in options_list:
        default_index = options_list.index(default_float_str)
    else:
        try:
            numeric_options = [float(f) for f in options_list]
            if 1901910.0 in numeric_options:
                default_index = numeric_options.index(1901910.0)
        except (ValueError, TypeError):
            pass
    
    float_to_inspect = st.selectbox(
        "**Select a Float to inspect its journey and latest sensor readings:**",
        options=options_list,
        index=default_index,
        help="Choose a float from this list to see its detailed data below."
    )

    if float_to_inspect:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🗺️ Trajectory for Float #{float_to_inspect}")
            trajectory_query = f"""
            SELECT latitude, longitude, cycle_number, to_char(time, 'YYYY-MM-DD') as date
            FROM profiles 
            WHERE float_id = '{float_to_inspect}' 
            AND time BETWEEN '2025-07-01' AND '2025-08-31'
            ORDER BY time;
            """
            trajectory_df = run_query(trajectory_query)

            if isinstance(trajectory_df, pd.DataFrame) and not trajectory_df.empty:
                view_state = pdk.ViewState(
                    latitude=trajectory_df['latitude'].mean(),
                    longitude=trajectory_df['longitude'].mean(),
                    zoom=4,
                    pitch=20
                )

                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=trajectory_df,
                    get_position='[longitude, latitude]',
                    get_color='[0, 100, 255, 160]',
                    get_radius=15000,
                    pickable=True,
                    tooltip={"html": "<b>Cycle:</b> {cycle_number}<br/><b>Date:</b> {date}"}
                )

                line_layer = pdk.Layer(
                    "PathLayer",
                    data=trajectory_df,
                    get_path='[longitude, latitude]',
                    get_color='[200, 30, 0, 160]',
                    width_min_pixels=2
                )

                st.pydeck_chart(pdk.Deck(
                    layers=[line_layer, scatter_layer], 
                    initial_view_state=view_state
                ))
                st.caption("This map shows the path of the selected float. Each blue dot is a recorded position.")

            else:
                st.warning("No trajectory data available for this float in the selected period.")

        with col2:
            st.subheader(f"🌡️ Latest Sensor Profile")
            latest_obs_query = f"""
            WITH latest_cycle AS (
                SELECT MAX(cycle_number) as max_cycle
                FROM profiles
                WHERE float_id = '{float_to_inspect}'
            )
            SELECT pres, temp, psal 
            FROM observations 
            WHERE platform_number = '{float_to_inspect}' 
            AND cycle_number = (SELECT max_cycle FROM latest_cycle);
            """
            latest_obs_df = run_query(latest_obs_query)
            
            if isinstance(latest_obs_df, pd.DataFrame) and not latest_obs_df.empty:
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Temperature Profile", "Salinity Profile"),
                    shared_yaxes=True
                )

                fig.add_trace(
                    go.Scatter(x=latest_obs_df['temp'], y=latest_obs_df['pres'], mode='lines', name='Temperature'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=latest_obs_df['psal'], y=latest_obs_df['pres'], mode='lines', name='Salinity'),
                    row=1, col=2
                )

                fig.update_layout(
                    title_text=f"Latest Profile for Float #{float_to_inspect}",
                    yaxis_title="Pressure (dbar)",
                    xaxis_title="Temperature (°C)",
                    xaxis2_title="Salinity (PSU)",
                    showlegend=False
                )
                
                fig.update_yaxes(autorange="reversed")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No observation data available for the latest profile.")
else:
    st.warning("Could not load the list of available floats for the selected time period.")