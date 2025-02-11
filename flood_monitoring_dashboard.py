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

# Station Coordinates with Enhanced Metadata
STATIONS = {
    'Rochdale': {
        'lat': 53.611067, 
        'lon': -2.178685,
        'description': 'Northern Greater Manchester monitoring station',
        'river': 'River Roch',
        'watershed_area': '68.3 sq km',
        'flood_risk_level': 'Moderate'
    },
    'Manchester': {
        'lat': 53.499526, 
        'lon': -2.271756,
        'description': 'Central Manchester river monitoring station',
        'river': 'River Irwell',
        'watershed_area': '129.5 sq km',
        'flood_risk_level': 'High'
    },
    'Bury': {
        'lat': 53.598766, 
        'lon': -2.305182,
        'description': 'Bury metropolitan borough monitoring station',
        'river': 'River Irwell',
        'watershed_area': '95.7 sq km',
        'flood_risk_level': 'Low'
    }
}

class RealTimeDashboard:
    def __init__(self):
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = create_client(supabase_url, supabase_key)
    
    def load_data(self):
        """
        Load the most recent river data with extended timeframe
        """
        try:
            # Fetch data from the last 7 days
            seven_days_ago = datetime.now(pytz.UTC) - timedelta(days=7)
            
            response = self.supabase.table('river_data')\
                .select('*')\
                .gte('river_timestamp', seven_days_ago.isoformat())\
                .order('river_timestamp', desc=True)\
                .execute()
            
            if response.data:
                data = pd.DataFrame(response.data)
                data['river_timestamp'] = pd.to_datetime(data['river_timestamp'], utc=True)
                return data
            else:
                st.warning("No recent data found")
                return None
        
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

def create_comprehensive_map():
    """
    Create a more detailed interactive map
    """
    # Prepare data for map
    map_data = pd.DataFrame.from_dict(STATIONS, orient='index')
    map_data.reset_index(inplace=True)
    map_data.columns = ['Station', 'Latitude', 'Longitude', 'Description', 'River', 'Watershed', 'Flood Risk']

    # Create interactive map with more details
    fig = px.scatter_mapbox(
        map_data, 
        lat='Latitude', 
        lon='Longitude', 
        hover_name='Station',
        hover_data={
            'Description': True,
            'River': True,
            'Watershed': True,
            'Flood Risk': True,
            'Latitude': ':.4f',
            'Longitude': ':.4f'
        },
        color='Flood Risk',
        color_discrete_map={
            'Low': 'green',
            'Moderate': 'yellow',
            'High': 'red'
        },
        zoom=9, 
        height=600,
        mapbox_style="open-street-map"
    )
    
    return fig

def generate_river_system_insights(data):
    """
    Generate comprehensive river system insights
    """
    if data is None:
        return None
    
    # Create insights dataframe
    insights = []
    
    for station in data['location_name'].unique():
        station_data = data[data['location_name'] == station]
        
        insight = {
            'Station': station,
            'Avg River Level': station_data['river_level'].mean(),
            'Max River Level': station_data['river_level'].max(),
            'Min River Level': station_data['river_level'].min(),
            'Avg Rainfall': station_data['rainfall'].mean(),
            'Max Rainfall': station_data['rainfall'].max(),
            'Risk Level': STATIONS[station]['flood_risk_level'],
            'Watershed Area': STATIONS[station]['watershed_area']
        }
        insights.append(insight)
    
    return pd.DataFrame(insights)

def plot_advanced_historical_trends(historical_data):
    """
    Create more comprehensive historical trend visualization
    """
    plt.figure(figsize=(15, 10))
    
    # River Level Trends
    plt.subplot(2, 1, 1)
    for station in historical_data['location_name'].unique():
        station_data = historical_data[historical_data['location_name'] == station]
        plt.plot(
            station_data['river_timestamp'], 
            station_data['river_level'], 
            label=f'{station} River Level',
            marker='o',
            markersize=4
        )
    
    plt.title('Detailed River Levels Over Time', fontsize=15)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('River Level (m)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Rainfall Trends
    plt.subplot(2, 1, 2)
    for station in historical_data['location_name'].unique():
        station_data = historical_data[historical_data['location_name'] == station]
        plt.plot(
            station_data['river_timestamp'], 
            station_data['rainfall'], 
            label=f'{station} Rainfall',
            marker='o',
            markersize=4
        )
    
    plt.title('Detailed Rainfall Trends', fontsize=15)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Rainfall (mm)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def main():
    st.set_page_config(page_title="Advanced Flood Monitoring Dashboard", layout="wide")
    st.title("Comprehensive Flood Monitoring Dashboard")
    
    # Initialize dashboard
    dashboard = RealTimeDashboard()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Real-Time Monitoring", 
        "Historical Trends", 
        "Station Details", 
        "River System Analysis"
    ])
    
    # Load data
    data = dashboard.load_data()
    
    # Real-Time Monitoring Tab
    with tab1:
        if data is not None:
            st.header("Current Station Metrics")
            cols = st.columns(3)
            
            for i, station in enumerate(data['location_name'].unique()):
                with cols[i]:
                    station_data = data[data['location_name'] == station]
                    river_level = station_data['river_level'].values[0]
                    
                    # Risk assessment
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
        st.header("Comprehensive Historical Data Analysis")
        
        historical_data = dashboard.load_data()
        
        if historical_data is not None:
            # Data Overview
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(historical_data))
            
            with col2:
                st.metric("Date Range", 
                          f"{historical_data['river_timestamp'].min().date()} to {historical_data['river_timestamp'].max().date()}")
            
            with col3:
                st.metric("Stations", ", ".join(historical_data['location_name'].unique()))
            
            # Detailed Trend Visualization
            st.subheader("Detailed Trend Analysis")
            historical_plot = plot_advanced_historical_trends(historical_data)
            st.pyplot(historical_plot)
    
    # Station Details Tab
    with tab3:
        st.header("Detailed Station Information")
        
        if data is not None:
            selected_station = st.selectbox(
                "Select Station", 
                data['location_name'].unique()
            )
            
            # Detailed station information with metadata
            st.subheader(f"{selected_station} Comprehensive Details")
            
            # Display station-specific details from STATIONS dictionary
            station_details = STATIONS[selected_station]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Description:** {station_details['description']}")
                st.write(f"**Primary River:** {station_details['river']}")
                st.write(f"**Watershed Area:** {station_details['watershed_area']}")
            
            with col2:
                st.write(f"**Flood Risk Level:** {station_details['flood_risk_level']}")
                st.write(f"**Latitude:** {station_details['lat']}")
                st.write(f"**Longitude:** {station_details['lon']}")
    
    # River System Analysis Tab
    with tab4:
        st.header("Comprehensive River System Insights")
        
        # Interactive Map
        st.subheader("Monitoring Stations Geographic Distribution")
        map_fig = create_comprehensive_map()
        st.plotly_chart(map_fig, use_container_width=True)
        
        # River System Insights
        st.subheader("Comparative Station Analysis")
        if data is not None:
            river_insights = generate_river_system_insights(data)
            if river_insights is not None:
                st.dataframe(river_insights)
            
            # Additional Analysis Visualizations
            st.subheader("Station Comparison")
            
            # River Level Comparison
            river_level_fig = px.bar(
                river_insights, 
                x='Station', 
                y='Avg River Level', 
                title='Average River Levels by Station',
                labels={'Avg River Level': 'Average River Level (m)'}
            )
            st.plotly_chart(river_level_fig, use_container_width=True)
            
            # Rainfall Comparison
            rainfall_fig = px.bar(
                river_insights, 
                x='Station', 
                y='Avg Rainfall', 
                title='Average Rainfall by Station',
                labels={'Avg Rainfall': 'Average Rainfall (mm)'}
            )
            st.plotly_chart(rainfall_fig, use_container_width=True)

    # Update query params
    st.query_params.update(refresh=True)

if __name__ == '__main__':
    main()