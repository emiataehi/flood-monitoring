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
    def __init__(self):
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = create_client(supabase_url, supabase_key)
    
    def load_data(self):
        try:
            # Fetch latest data from Supabase
            response = self.supabase.table('river_data')\
                .select('*')\
                .order('river_timestamp', desc=True)\
                .limit(3)\
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
    
    def collect_historical_data(self):
        try:
            # Fetch historical data from Supabase
            response = self.supabase.table('river_data')\
                .select('*')\
                .order('river_timestamp')\
                .execute()
            
            if response.data:
                historical_data = pd.DataFrame(response.data)
                historical_data['river_timestamp'] = pd.to_datetime(historical_data['river_timestamp'], utc=True)
                return historical_data
            else:
                st.warning("No historical data found")
                return None
        
        except Exception as e:
            st.error(f"Error collecting historical data: {e}")
            return None

# Function moved outside the class definition
def plot_historical_trends(historical_data):
    """
    Create plots for historical river levels and rainfall
    """
    # Group by location and timestamp
    grouped = historical_data.groupby(['location_name', 'river_timestamp']).agg({
        'river_level': 'mean',
        'rainfall': 'mean'
    }).reset_index()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot river levels
    for station in grouped['location_name'].unique():
        station_data = grouped[grouped['location_name'] == station]
        ax1.plot(station_data['river_timestamp'], station_data['river_level'], 
                 label=station, marker='o')
    
    ax1.set_title('Historical River Levels')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('River Level (m)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot rainfall
    for station in grouped['location_name'].unique():
        station_data = grouped[grouped['location_name'] == station]
        ax2.plot(station_data['river_timestamp'], station_data['rainfall'], 
                 label=station, marker='o')
    
    ax2.set_title('Historical Rainfall')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

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
        if data is not None:
            # Display station metrics
            st.header("Station Metrics")
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
    
    with tab2:
        st.header("Historical Data Trends")
        
        # Collect historical data
        historical_data = dashboard.collect_historical_data()
        
        if historical_data is not None:
            # Display basic statistics
            st.subheader("Historical Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(historical_data))
            
            with col2:
                st.metric("Date Range", 
                          f"{historical_data['river_timestamp'].min().date()} to {historical_data['river_timestamp'].max().date()}")
            
            with col3:
                st.metric("Stations", ", ".join(historical_data['location_name'].unique()))
            
            # Plot historical trends
            st.subheader("Trend Visualization")
            historical_plot = plot_historical_trends(historical_data)
            st.pyplot(historical_plot)
            
            # Station-specific summary
            st.subheader("Station-wise Summary")
            station_summary = historical_data.groupby('location_name').agg({
                'river_level': ['mean', 'min', 'max'],
                'rainfall': ['mean', 'min', 'max']
            })
            st.dataframe(station_summary)
    
    with tab3:
        st.header("Detailed Station Information")
        
        if data is not None:
            selected_station = st.selectbox(
                "Select Station", 
                data['location_name'].unique()
            )
            
            station_data = data[data['location_name'] == selected_station]
            
            st.subheader(f"{selected_station} Station Details")
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.write(f"**River Station ID:** {station_data['river_station_id'].values[0]}")
                st.write(f"**Current River Level:** {station_data['river_level'].values[0]:.3f} m")
            
            with detail_col2:
                st.write(f"**Rainfall Station ID:** {station_data['rainfall_station_id'].values[0]}")
                st.write(f"**Current Rainfall:** {station_data['rainfall'].values[0]:.3f} mm")
    
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