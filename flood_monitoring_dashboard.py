import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pytz
import time
import glob
from datetime import datetime  # Add this line

class RealTimeDashboard:
    def __init__(self, data_directory):
        self.data_directory = data_directory
    
    def get_latest_csv(self):
        try:
            # Get all CSV files in the directory
            csv_files = [f for f in os.listdir(self.data_directory) 
                         if f.startswith('combined_data_') and f.endswith('.csv')]
            
            if not csv_files:
                st.error(f"No CSV files found in {self.data_directory}")
                return None
            
            # Sort files by modification time, most recent first
            csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.data_directory, f)), reverse=True)
            
            # Select the most recent file
            latest_file = csv_files[0]
            full_path = os.path.join(self.data_directory, latest_file)
            
            return full_path
        
        except Exception as e:
            st.error(f"Error finding latest CSV: {e}")
            return None
    
    def load_data(self):
        latest_file = self.get_latest_csv()
        if latest_file:
            try:
                # Read the CSV
                data = pd.read_csv(latest_file)
                
                # Convert timestamp and ensure UTC timezone
                data['river_timestamp'] = pd.to_datetime(data['river_timestamp'], utc=True)
                
                return data
            except Exception as e:
                st.error(f"Error loading data from {latest_file}: {e}")
                return None
        return None
    
    def collect_historical_data(self):
        """
        Collect and merge historical CSV files
        """
        csv_files = glob.glob(os.path.join(self.data_directory, 'combined_data_*.csv'))
        csv_files.sort()
        
        historical_dataframes = []
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df['file_timestamp'] = os.path.basename(file).replace('combined_data_', '').replace('.csv', '')
                historical_dataframes.append(df)
            except Exception as e:
                st.warning(f"Could not read file {file}: {e}")
        
        if historical_dataframes:
            historical_data = pd.concat(historical_dataframes, ignore_index=True)
            historical_data['river_timestamp'] = pd.to_datetime(historical_data['river_timestamp'])
            return historical_data
        else:
            st.error("No historical data found")
            return None

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
    # Page configuration
    st.set_page_config(page_title="Flood Monitoring Dashboard", layout="wide")
    
    # Dashboard Title
    st.title("Comprehensive Flood Monitoring Dashboard")
    
    # Initialize dashboard
    dashboard = RealTimeDashboard('C:/Users/Administrator/NEWPROJECT/combined_data')
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Real-Time Monitoring", "Historical Trends", "Station Details"])
    
    # Real-Time Monitoring Tab
    with tab1:
        # Load latest data
        data = dashboard.load_data()
        
        if data is not None:
            # Sidebar with file and collection information
            file_name = os.path.basename(dashboard.get_latest_csv())
            st.sidebar.info(f"Data File: {file_name}")
            
            # Current time for comparison (in UTC)
            current_time = datetime.now(pytz.UTC)
            st.sidebar.write("Current System Time (UTC):", current_time)
            
            # Data timestamp
            data_timestamp = data['river_timestamp'].max()
            st.sidebar.info(f"Data Collected: {data_timestamp}")
            
            # Time difference
            time_diff = current_time - data_timestamp
            st.sidebar.write("Time Since Data Collection:", time_diff)
            
            # Station Overview
            st.header("Station Metrics")
            cols = st.columns(3)
            
            for i, station in enumerate(data['location_name'].unique()):
                with cols[i]:
                    station_data = data[data['location_name'] == station]
                    
                    # Risk Assessment
                    river_level = station_data['river_level'].values[0]
                    risk_text = 'Low Risk'
                    delta_color = 'normal'
                    
                    if river_level > 0.7:
                        risk_text = 'High Risk'
                        delta_color = 'inverse'
                    elif river_level > 0.4:
                        risk_text = 'Moderate Risk'
                        delta_color = 'normal'
                    
                    st.metric(
                        station, 
                        f"River Level: {river_level:.3f} m", 
                        f"Risk: {risk_text}",
                        delta_color=delta_color
                    )
    
    # Historical Trends Tab
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
    
    # Station Details Tab
    with tab3:
        st.header("Detailed Station Information")
        selected_station = st.selectbox(
            "Select Station", 
            data['location_name'].unique() if data is not None else []
        )
        
        if data is not None and selected_station:
            station_data = data[data['location_name'] == selected_station]
            
            # Station Details
            st.subheader(f"{selected_station} Station Details")
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.write(f"**River Station ID:** {station_data['river_station_id'].values[0]}")
                st.write(f"**Current River Level:** {station_data['river_level'].values[0]:.3f} m")
            
            with detail_col2:
                st.write(f"**Rainfall Station ID:** {station_data['rainfall_station_id'].values[0]}")
                st.write(f"**Current Rainfall:** {station_data['rainfall'].values[0]:.3f} mm")
    
    # Refresh button
    if st.sidebar.button('Refresh Data'):
        st.experimental_rerun()
    
    # Automatic refresh
    st.experimental_set_query_params(refresh=True)
    time.sleep(30)  # Wait 30 seconds before refreshing
    st.experimental_rerun()

if __name__ == '__main__':
    main()