import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import os
import pytz
from datetime import datetime
import folium
from streamlit_folium import folium_static
from geospatial_utils import create_station_map

class RealTimeDashboard:
    # ... (keep the entire existing RealTimeDashboard class unchanged)

def plot_historical_trends(historical_data):
    # ... (keep the existing plot_historical_trends function unchanged)

def main():
    st.set_page_config(page_title="Flood Monitoring Dashboard", layout="wide")
    st.title("Comprehensive Flood Monitoring Dashboard")
    
    # Initialize dashboard
    dashboard = RealTimeDashboard()
    
    # Create tabs - Add a fourth tab for Geospatial View
    tab1, tab2, tab3, tab4 = st.tabs([
        "Real-Time Monitoring", 
        "Historical Trends", 
        "Station Details", 
        "Geospatial View"
    ])
    
    # Load data
    data = dashboard.load_data()
    
    with tab1:
        # ... (keep existing tab1 code)

    with tab2:
        # ... (keep existing tab2 code)

    with tab3:
        # ... (keep existing tab3 code)
    
    # New Geospatial Tab
    with tab4:
        st.header("River Monitoring Stations Map")
        
        # Create the map
        station_map = create_station_map()
        
        # Display the map in Streamlit
        if station_map:
            folium_static(station_map)
        else:
            st.error("Failed to generate station map")
        
        # Additional geospatial information
        st.subheader("Station Coordinates")
        stations = {
            'Rochdale': {'lat': 53.611067, 'lon': -2.178685},
            'Manchester': {'lat': 53.499526, 'lon': -2.271756},
            'Bury': {'lat': 53.598766, 'lon': -2.305182}
        }
        
        # Create a DataFrame for station coordinates
        coord_df = pd.DataFrame.from_dict(stations, orient='index')
        coord_df.columns = ['Latitude', 'Longitude']
        coord_df.index.name = 'Station'
        
        st.dataframe(coord_df)

    # Update query params
    st.query_params.update(refresh=True)

if __name__ == '__main__':
    main()