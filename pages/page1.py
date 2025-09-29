import streamlit as st
import pandas as pd
import psycopg2
import requests
import pydeck as pdk

# --- Page Configuration ---
st.set_page_config(page_title="ARGO Closest Points Finder", layout="wide", initial_sidebar_state="collapsed")

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
        if st.button("Dashboard"):
            st.switch_page("app.py")

    with col2:
        if st.button("Closest Buoy Finder", type="primary"):
            st.switch_page("pages/page1.py")

    with col3:
        if st.button("Chat AI RAG"):
            st.switch_page("pages/page2.py")
    
    with col4:
        if st.button("Learn More Videos"):
            st.switch_page("pages/page3.py")

create_header()

# =========================
# Database Connection
# =========================
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

@st.cache_data(ttl=600)
def run_query(query, params=None):
    """Executes a SQL query with parameters and returns the result as a Pandas DataFrame."""
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description is None:
                return pd.DataFrame()
            
            colnames = [desc[0] for desc in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=colnames)
            return df
    except Exception as e:
        conn.rollback()
        st.error(f"Database Query Error: {e}")
        return pd.DataFrame()

# =========================
# Get user location from IP
# =========================
@st.cache_data(ttl=3600)
def get_ip_location():
    """Fetches the user's approximate location from their IP address."""
    try:
        resp = requests.get("https://ipinfo.io/json", timeout=5)
        if resp.status_code == 200:
            loc = resp.json().get("loc", None)
            if loc:
                lat_str, lon_str = loc.split(",")
                return float(lat_str), float(lon_str)
    except Exception:
        pass
    return None, None

# =========================
# Streamlit UI (Main Page)
# =========================
st.header("🛰️ ARGO Closest Points Finder", divider='blue')
st.markdown("Discover the last known locations of ARGO buoys closest to any coordinate on Earth.")

ip_lat, ip_lon = get_ip_location()

# --- Set default coordinates ---
default_lat = 16.5062  # Amaravati
default_lon = 80.6480

# --- Check for IP-based location and update defaults if available ---
if ip_lat is not None and ip_lon is not None:
    st.info(f"🌍 Your estimated location: {ip_lat}, {ip_lon}")
    default_lat = ip_lat
    default_lon = ip_lon
else:
    st.warning("⚠️ Could not determine your location. Please enter coordinates manually.")

# --- Use columns for a cleaner layout on the main page ---
col1, col2, col3 = st.columns(3)

with col1:
    target_lat = st.number_input("Enter Latitude", value=default_lat, format="%.6f")
with col2:
    target_lon = st.number_input("Enter Longitude", value=default_lon, format="%.6f")
with col3:
    # --- Default value changed to 3 ---
    n = st.number_input("How many buoys to find?", min_value=1, value=3, step=1)

# The query is hardcoded to find the last known location from the 'profiles' table.
closest_points_query = """
WITH LastKnownLocations AS (
    SELECT DISTINCT ON (float_id) * FROM profiles ORDER BY float_id, time DESC
)
SELECT 
    *, 
    (6371 * acos(cos(radians(%s)) * cos(radians(latitude)) * cos(radians(longitude) - radians(%s)) + sin(radians(%s)) * sin(radians(latitude)))) AS distance_km
FROM 
    LastKnownLocations
WHERE 
    latitude IS NOT NULL AND longitude IS NOT NULL
ORDER BY 
    distance_km
LIMIT %s;
"""

with st.spinner("Querying for the closest buoys and calculating distances..."):
    closest_df = run_query(closest_points_query, (target_lat, target_lon, target_lat, n))

if closest_df.empty:
    st.warning("No data found for the selected criteria.")
else:
    st.write(f"✅ Top {n} Closest Buoys (Last Known Locations)")
    st.dataframe(closest_df)

    st.header("🗺️ Map of Target and Closest Buoys", divider='blue')
    
    # Prepare dataframes for Pydeck
    closest_map_df = closest_df[['latitude', 'longitude', 'float_id']]
    target_df = pd.DataFrame([{"latitude": target_lat, "longitude": target_lon}])

    # Define the view state, centered on the target
    view_state = pdk.ViewState(
        latitude=target_lat,
        longitude=target_lon,
        zoom=5,
        pitch=0 # Ensures a 2D view
    )

    # Layer for the closest buoys (blue)
    buoy_layer = pdk.Layer(
        "ScatterplotLayer",
        data=closest_map_df,
        get_position='[longitude, latitude]',
        get_color='[0, 100, 255, 160]', # Blue color
        get_radius=10000,
        pickable=True,
        tooltip={"html": "<b>Float ID:</b> {float_id}"}
    )

    # Layer for the user's target location (green)
    target_layer = pdk.Layer(
        "ScatterplotLayer",
        data=target_df,
        get_position='[longitude, latitude]',
        get_color='[0, 255, 0, 160]', # Green color
        get_radius=10000,
        pickable=True,
        tooltip={"html": "<b>Your Target Location</b>"}
    )

    # Render the map using pydeck
    st.pydeck_chart(pdk.Deck(
        layers=[target_layer, buoy_layer],
        initial_view_state=view_state,
        map_style=None # Set to None for a plain background
    ))
    # --- UPDATED CAPTION ---
    st.caption("Your location is green 🟢, and the buoys are blue 🔵.")
    
    st.header("🌟 Fun Fact", divider='blue')
    closest_record = closest_df.iloc[0]
    float_id = closest_record.get("float_id", "Unknown ID")
    distance = closest_record["distance_km"]
    st.info(f"🎯 The closest buoy to your location is **{float_id}**, approximately **{distance:,.0f} kilometers** away!")

    # --- TRAJECTORY MAP ---
    if 'float_id' in closest_df.columns:
        float_to_inspect = closest_df.iloc[0]["float_id"]
        st.subheader(f"🧭 Trajectory for Closest Buoy: {float_to_inspect}")
        
        # Updated query to get data for tooltips
        trajectory_query = """
        SELECT latitude, longitude, cycle_number, to_char(time, 'YYYY-MM-DD') as date
        FROM profiles 
        WHERE float_id = %s
        ORDER BY time;
        """
        
        with st.spinner("Fetching trajectory data..."):
            trajectory_df = run_query(trajectory_query, (float_to_inspect,))

        if not trajectory_df.empty:
            view_state_traj = pdk.ViewState(
                latitude=trajectory_df['latitude'].mean(),
                longitude=trajectory_df['longitude'].mean(),
                zoom=6,
                pitch=0 # Keep it 2D
            )

            # Layer for the points on the trajectory
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=trajectory_df,
                get_position='[longitude, latitude]',
                get_color='[0, 100, 255, 160]', # Blue
                get_radius=5000,
                pickable=True,
                tooltip={"html": "<b>Cycle:</b> {cycle_number}<br/><b>Date:</b> {date}"}
            )
            
            # Initialize layers list with the scatter plot
            trajectory_layers = [scatter_layer]

            # Only add the PathLayer if there are enough points to draw a line
            if len(trajectory_df) > 1:
                line_layer = pdk.Layer(
                    "PathLayer",
                    data=trajectory_df,
                    get_path='[longitude, latitude]',
                    get_color='[0, 100, 255, 160]', # Blue
                    width_min_pixels=2
                )
                trajectory_layers.append(line_layer)

            # Render the trajectory map
            st.pydeck_chart(pdk.Deck(
                layers=trajectory_layers,
                initial_view_state=view_state_traj,
                map_style=None
            ))
            # --- UPDATED CAPTION ---
            st.caption("This map shows the historical path of the buoy.")
        else:
            st.warning(f"🤷 No trajectory data available for buoy {float_to_inspect}.")