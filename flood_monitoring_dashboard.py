import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import os
import pytz
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
from geospatial_utils import create_station_map
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

# Global Station Configuration
STATION_CONFIG = {
    'Rochdale': {
        'full_name': 'Rochdale River Monitoring Station',
        'latitude': 53.611067,
        'longitude': -2.178685,
        'river': 'River Roch',
        'description': 'Monitoring station in northern Greater Manchester',
        'risk_level': 'Moderate'
    },
    'Manchester Racecourse': {
        'full_name': 'Manchester Racecourse River Station',
        'latitude': 53.499526,
        'longitude': -2.271756,
        'river': 'River Irwell',
        'description': 'Central Manchester river monitoring location',
        'risk_level': 'High'
    },
    'Bury Ground': {
        'full_name': 'Bury Ground River Monitoring Point',
        'latitude': 53.598766,
        'longitude': -2.305182,
        'river': 'River Irwell',
        'description': 'Monitoring station in Bury metropolitan area',
        'risk_level': 'Low'
    }
}

class FloodMonitoringDashboard:
    def __init__(self):
        """
        Initialize Supabase client for data retrieval
        """
        try:
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_KEY')
            self.supabase = create_client(supabase_url, supabase_key)
        except Exception as e:
            st.error(f"Failed to initialize Supabase client: {e}")
            self.supabase = None

    def fetch_river_data(self, days_back=7):
        """
        Retrieve river monitoring data
        
        Args:
            days_back (int): Number of days to retrieve data for
        
        Returns:
            pd.DataFrame: River monitoring data
        """
        try:
            # Calculate date range
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=days_back)

            # Fetch data from Supabase
            response = self.supabase.table('river_data')\
                .select('*')\
                .gte('river_timestamp', start_date.isoformat())\
                .lte('river_timestamp', end_date.isoformat())\
                .order('river_timestamp', desc=True)\
                .execute()

            # Convert to DataFrame
            if response.data:
                df = pd.DataFrame(response.data)
                df['river_timestamp'] = pd.to_datetime(df['river_timestamp'], utc=True)
                return df
            else:
                st.warning("No recent river data found")
                return None

        except Exception as e:
            st.error(f"Data retrieval error: {e}")
            return None

def create_station_map():
    """
    Generate interactive map of monitoring stations
    
    Returns:
        plotly Figure: Interactive station map
    """
    # Prepare station data
    stations_df = pd.DataFrame.from_dict(STATION_CONFIG, orient='index')
    stations_df.reset_index(inplace=True)
    stations_df.columns = ['Station', 'Full Name', 'Latitude', 'Longitude', 'River', 'Description', 'Risk Level']

    # Create map
    fig = px.scatter_mapbox(
        stations_df, 
        lat='Latitude', 
        lon='Longitude',
        hover_name='Station',
        hover_data=['Full Name', 'River', 'Description', 'Risk Level'],
        color='Risk Level',
        color_discrete_map={
            'Low': 'green', 
            'Moderate': 'yellow', 
            'High': 'red'
        },
        zoom=9,
        height=600
    )
    fig.update_layout(mapbox_style="open-street-map")
    
    return fig

def plot_river_trends(data):
    """
    Generate river level and rainfall trend plots
    
    Args:
        data (pd.DataFrame): River monitoring data
    
    Returns:
        matplotlib Figure: Trend visualization
    """
    plt.figure(figsize=(15, 10))

    # River Levels
    plt.subplot(2, 1, 1)
    for station in data['location_name'].unique():
        station_data = data[data['location_name'] == station]
        plt.plot(
            station_data['river_timestamp'], 
            station_data['river_level'], 
            label=station,
            marker='o'
        )
    plt.title('River Levels Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('River Level (m)')
    plt.legend()
    plt.xticks(rotation=45)

    # Rainfall
    plt.subplot(2, 1, 2)
    for station in data['location_name'].unique():
        station_data = data[data['location_name'] == station]
        plt.plot(
            station_data['river_timestamp'], 
            station_data['rainfall'], 
            label=station,
            marker='o'
        )
    plt.title('Rainfall Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Rainfall (mm)')
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    return plt

def main():
    # Page configuration
    st.set_page_config(
        page_title="Flood Monitoring Dashboard", 
        layout="wide"
    )
    st.title("Comprehensive Flood Monitoring Dashboard")

    # Initialize dashboard
    dashboard = FloodMonitoringDashboard()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Real-Time Monitoring", 
        "Historical Trends", 
        "Station Details", 
        "Geospatial View"
    ])

    # Fetch river data
    river_data = dashboard.fetch_river_data()

    # Real-Time Monitoring Tab
    with tab1:
        st.header("Current Station Metrics")
        if river_data is not None:
            cols = st.columns(3)
            for i, station in enumerate(river_data['location_name'].unique()):
                with cols[i]:
                    station_data = river_data[river_data['location_name'] == station]
                    river_level = station_data['river_level'].values[0]
                    
                    risk_text = 'Low Risk'
                    delta_color = 'normal'
                    
                    if river_level > 0.7:
                        risk_text = 'High Risk'
                        delta_color = 'inverse'
                    elif river_level > 0.4:
                        risk_text = 'Moderate Risk'
                    
                    st.metric(
                        station, 
                        f"River Level: {river_level:.3f} m", 
                        f"Risk: {risk_text}",
                        delta_color=delta_color
                    )

    # Historical Trends Tab
    with tab2:
        st.header("Historical Data Analysis")
        if river_data is not None:
            # Trend visualization
            historical_plot = plot_river_trends(river_data)
            st.pyplot(historical_plot)

    # Station Details Tab
    with tab3:
        st.header("Station Information")
        selected_station = st.selectbox(
            "Select Station", 
            list(STATION_CONFIG.keys())
        )
        
        # Display station details
        station_info = STATION_CONFIG[selected_station]
        st.write(f"**Full Name:** {station_info['full_name']}")
        st.write(f"**River:** {station_info['river']}")
        st.write(f"**Description:** {station_info['description']}")
        st.write(f"**Risk Level:** {station_info['risk_level']}")
        st.write(f"**Coordinates:** {station_info['latitude']}, {station_info['longitude']}")

    # Geospatial View Tab
    with tab4:
        st.header("Station Geographic Distribution")
        station_map = create_station_map()
        st.plotly_chart(station_map, use_container_width=True)

    # Optional: Update query parameters
    st.query_params.update(refresh=True)

if __name__ == '__main__':
    main()