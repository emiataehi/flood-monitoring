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
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

# Station Coordinates (Global definition)
STATIONS = {
    'Rochdale': {
        'lat': 53.611067, 
        'lon': -2.178685,
        'description': 'Located in the northern part of Greater Manchester',
        'river': 'River Roch'
    },
    'Manchester': {
        'lat': 53.499526, 
        'lon': -2.271756,
        'description': 'Central Manchester river monitoring station',
        'river': 'River Irwell'
    },
    'Bury': {
        'lat': 53.598766, 
        'lon': -2.305182,
        'description': 'Bury metropolitan borough monitoring station',
        'river': 'River Irwell'
    }
}

class RealTimeDashboard:
    # ... (previous implementation remains the same)

def create_interactive_map():
    """
    Create an interactive map using Plotly
    """
    # Prepare data for map
    map_data = pd.DataFrame.from_dict(STATIONS, orient='index')
    map_data.reset_index(inplace=True)
    map_data.columns = ['Station', 'Latitude', 'Longitude', 'Description', 'River']

    # Create interactive map
    fig = px.scatter_mapbox(
        map_data, 
        lat='Latitude', 
        lon='Longitude', 
        hover_name='Station',
        hover_data=['Description', 'River'],
        color='Station',
        zoom=9, 
        height=600,
        mapbox_style="open-street-map"
    )
    
    return fig

def enhanced_historical_trends(historical_data):
    """
    Create more comprehensive historical trend visualizations
    """
    # Ensure sufficient historical data
    if len(historical_data) < 2:
        st.warning("Insufficient historical data for meaningful visualization")
        return None

    # Prepare data for multi-station, multi-feature visualization
    plt.figure(figsize=(15, 10))

    # River Level Subplot
    plt.subplot(2, 1, 1)
    for station in historical_data['location_name'].unique():
        station_data = historical_data[historical_data['location_name'] == station]
        plt.plot(
            station_data['river_timestamp'], 
            station_data['river_level'], 
            label=f'{station} River Level'
        )
    
    plt.title('Historical River Levels Comparison')
    plt.xlabel('Timestamp')
    plt.ylabel('River Level (m)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Rainfall Subplot
    plt.subplot(2, 1, 2)
    for station in historical_data['location_name'].unique():
        station_data = historical_data[historical_data['location_name'] == station]
        plt.plot(
            station_data['river_timestamp'], 
            station_data['rainfall'], 
            label=f'{station} Rainfall'
        )
    
    plt.title('Historical Rainfall Comparison')
    plt.xlabel('Timestamp')
    plt.ylabel('Rainfall (mm)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return plt

def main():
    # Configure Streamlit page
    st.set_page_config(page_title="Flood Monitoring Dashboard", layout="wide")
    st.title("Comprehensive Flood Monitoring Dashboard")
    
    # Initialize dashboard
    dashboard = RealTimeDashboard()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Real-Time Monitoring", 
        "Historical Trends", 
        "Station Details", 
        "Geospatial Overview",
        "River System Insights"
    ])
    
    # Load data
    data = dashboard.load_data()
    
    # Existing tabs remain the same as in previous implementation
    
    # Geospatial Overview Tab
    with tab4:
        st.header("Geospatial Station Overview")
        
        # Interactive Map
        st.subheader("River Monitoring Stations Map")
        map_fig = create_interactive_map()
        st.plotly_chart(map_fig, use_container_width=True)
        
        # Detailed Station Information
        st.subheader("Station Details")
        for station, details in STATIONS.items():
            st.markdown(f"""
            ### {station} Station
            - **Latitude**: {details['lat']}
            - **Longitude**: {details['lon']}
            - **Description**: {details['description']}
            - **Primary River**: {details['river']}
            """)
    
    # River System Insights Tab
    with tab5:
        st.header("River System Insights")
        
        # Station Comparative Analysis
        st.subheader("Comparative Station Analysis")
        
        # Basic statistical comparison
        if data is not None:
            comparison_df = data.groupby('location_name').agg({
                'river_level': ['mean', 'min', 'max'],
                'rainfall': ['mean', 'min', 'max']
            })
            st.dataframe(comparison_df)
        
        # Distance between stations
        st.subheader("Inter-Station Distances")
        distances = {
            'Rochdale-Manchester': round(np.sqrt(
                (STATIONS['Rochdale']['lat'] - STATIONS['Manchester']['lat'])**2 + 
                (STATIONS['Rochdale']['lon'] - STATIONS['Manchester']['lon'])**2
            ) * 111, 2),
            'Rochdale-Bury': round(np.sqrt(
                (STATIONS['Rochdale']['lat'] - STATIONS['Bury']['lat'])**2 + 
                (STATIONS['Rochdale']['lon'] - STATIONS['Bury']['lon'])**2
            ) * 111, 2),
            'Manchester-Bury': round(np.sqrt(
                (STATIONS['Manchester']['lat'] - STATIONS['Bury']['lat'])**2 + 
                (STATIONS['Manchester']['lon'] - STATIONS['Bury']['lon'])**2
            ) * 111, 2)
        }
        st.write(distances)

    # Update query params
    st.query_params.update(refresh=True)

if __name__ == '__main__':
    main()