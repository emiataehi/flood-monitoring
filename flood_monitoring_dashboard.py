import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from supabase import create_client
import os
import pytz
from datetime import datetime, timedelta
from prediction_utils import FloodPredictor

# Station Configuration
STATION_CONFIG = {
    'Rochdale': {
        'full_name': 'Rochdale River Monitoring Station',
        'latitude': 53.611067,
        'longitude': -2.178685,
        'river': 'River Roch',
        'description': 'Monitoring station in northern Greater Manchester',
        'warning_level': 0.3
    },
    'Manchester Racecourse': {
        'full_name': 'Manchester Racecourse River Station',
        'latitude': 53.499526,
        'longitude': -2.271756,
        'river': 'River Irwell',
        'description': 'Central Manchester river monitoring location',
        'warning_level': 1.1
    },
    'Bury Ground': {
        'full_name': 'Bury Ground River Monitoring Point',
        'latitude': 53.598766,
        'longitude': -2.305182,
        'river': 'River Irwell',
        'description': 'Monitoring station in Bury metropolitan area',
        'warning_level': 0.4
    }
}

class FloodMonitoringDashboard:
    def __init__(self):
        """Initialize dashboard"""
        self.setup_supabase()

    def setup_supabase(self):
        """Connect to Supabase"""
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
            self.supabase = create_client(url, key)
        except Exception as e:
            st.error(f"Failed to connect to database: {str(e)}")
            self.supabase = None

    def get_data(self, days=7):
        """Get river monitoring data"""
        try:
            # Calculate date range
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=days)
            
            # Get data from Supabase
            response = self.supabase.table('river_data')\
                .select('*')\
                .gte('river_timestamp', start_date.isoformat())\
                .lte('river_timestamp', end_date.isoformat())\
                .execute()
                
            if response.data:
                df = pd.DataFrame(response.data)
                df['river_timestamp'] = pd.to_datetime(df['river_timestamp'])
                return df.sort_values('river_timestamp')
            return None
        except Exception as e:
            st.error(f"Error getting data: {str(e)}")
            return None

    def create_map(self):
        """Create station map"""
        # Create dataframe from station config
        stations_df = pd.DataFrame.from_dict(STATION_CONFIG, orient='index')
        stations_df.reset_index(inplace=True)
        stations_df.columns = ['Station', 'Full Name', 'Latitude', 'Longitude', 
                             'River', 'Description', 'Warning Level']
        
        # Create map
        fig = px.scatter_mapbox(
            stations_df,
            lat='Latitude',
            lon='Longitude',
            hover_name='Station',
            hover_data=['River', 'Description'],
            zoom=9,
            height=600
        )
        
        fig.update_layout(mapbox_style="open-street-map")
        return fig

    def create_trend_plot(self, data):
        """Create river level trends plot"""
        fig = px.line(
            data,
            x='river_timestamp',
            y='river_level',
            color='location_name',
            title='River Levels Over Time',
            labels={'river_timestamp': 'Time', 
                   'river_level': 'River Level (m)',
                   'location_name': 'Station'}
        )
        return fig

def main():
    # Page setup
    st.set_page_config(page_title="Flood Monitoring", layout="wide")
    st.title("Flood Monitoring Dashboard")

    # Initialize dashboard
    dashboard = FloodMonitoringDashboard()

    # Data timeframe selector
    days = st.sidebar.slider("Select Days of Data", 1, 30, 7)

    # Get data
    data = dashboard.get_data(days=days)

    if data is not None:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Current Status", "Trends", "Map"])

        # Tab 1: Current Status
        with tab1:
            st.header("Current River Levels")
            # Show current levels for each station
            cols = st.columns(len(STATION_CONFIG))
            
            for i, (station, info) in enumerate(STATION_CONFIG.items()):
                with cols[i]:
                    station_data = data[data['location_name'] == station]
                    if not station_data.empty:
                        current_level = station_data['river_level'].iloc[-1]
                        warning_level = info['warning_level']
                        
                        # Determine risk
                        if current_level > warning_level:
                            risk = "High"
                            delta_color = "inverse"
                        else:
                            risk = "Normal"
                            delta_color = "normal"
                        
                        st.metric(
                            label=station,
                            value=f"{current_level:.3f}m",
                            delta=risk,
                            delta_color=delta_color
                        )

        # Tab 2: Trends
        with tab2:
            st.header("River Level Trends")
            trend_plot = dashboard.create_trend_plot(data)
            st.plotly_chart(trend_plot, use_container_width=True)
            
            # Show statistics
            st.subheader("Statistics")
            stats = data.groupby('location_name').agg({
                'river_level': ['mean', 'min', 'max'],
                'rainfall': ['mean', 'sum']
            }).round(3)
            st.dataframe(stats)

        # Tab 3: Map
        with tab3:
            st.header("Station Locations")
            map_fig = dashboard.create_map()
            st.plotly_chart(map_fig, use_container_width=True)

    else:
        st.error("Could not load data. Please try again later.")

if __name__ == "__main__":
    main()