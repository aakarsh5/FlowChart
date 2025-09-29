import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="ARGO Videos",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Reusable Header and Navigation ---
def create_header(page_name="default"):
    """Creates a consistent header with logo and navigation buttons."""
    # Center the logo at the top
    st.image("logo.png", width=350)
    st.caption(f"Displaying real-time data from **July 2025 to August 2025**.")

    # 4 columns for navigation buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Dashboard", key=f"dashboard_{page_name}"):
            st.switch_page("app.py")

    with col2:
        if st.button("Closest Buoy Finder", key=f"finder_{page_name}"):
            st.switch_page("pages/page1.py")

    with col3:
        if st.button("Chat AI RAG", key=f"chat_{page_name}"):
            st.switch_page("pages/page2.py")
    
    with col4:
        if st.button("Learn More Videos", key=f"videos_{page_name}", type="primary"):
            st.switch_page("pages/page3.py")


# --- Call the header function at the top of the page ---
create_header()

# --- Page Content ---
st.header("🎥 Learn More: Videos from the Field", divider='blue')
st.markdown("""
The ARGO program is a massive global effort involving numerous countries and institutions. 
Below are some excellent videos that explain what ARGO floats are, how they work, and why the data they collect is so vital for understanding our planet's oceans and climate.
""")

st.markdown("---")

# --- Video Section 1 ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Argo Floats : How do we measure the ocean?")
    st.video("http://www.youtube.com/watch?v=WGbanFvBX38")
    st.info("""
    **Source:** IMOS5395
    
    Learn about the global Argo fleet, how floats measure temperature, salinity, and current speed, and how their data is used to predict weather and climate, and aid industries like shipping and fishing.
    """)

with col2:
    st.subheader("Argo float animation")
    st.video("http://www.youtube.com/watch?v=PzHZdwaBr_Q")
    st.info("""
    **Source:** argoproject
    
    An animation demonstrating how Argo floats dive, drift, ascend, and transmit vital ocean data for long-term monitoring, revealing subsurface ocean conditions for climate change and El Niño research.
    """)

st.markdown("---")

# --- Video Section 2 ---
col3, col4 = st.columns(2, gap="large")

with col3:
    st.subheader("Scientists launch 'Argo float' to gather ocean data in Antarctica")
    st.video("http://www.youtube.com/watch?v=YofPLdjgu2Q")
    st.info("""
    **Source:** CBC News
    
    Witness the deployment of an Argo float in the Drake Passage as scientists gather crucial ocean temperature and salinity data in Antarctica. This data is vital for understanding ocean warming and improving climate models.
    """)

with col4:
    st.subheader("The cycle of an Argo Float")
    st.video("http://www.youtube.com/watch?v=YkctZlQgU0g")
    st.info("""
    **Source:** Bureau of Meteorology
    
    A short, animated explanation of the typical operational cycle of an Argo float, illustrating its descent, drifting at depth, ascent while collecting data, and transmission via satellite.
    """)