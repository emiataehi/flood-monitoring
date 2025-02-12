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
        'normal_range': {'min': 0.15, 'max': 0.30}
    },
    'Manchester Racecourse': {
        'full_name': 'Manchester Racecourse River Station',
        'latitude': 53.499526,
        'longitude': -2.271756,
        'river': 'River Irwell',
        'description': 'Central Manchester river monitoring location',
        'normal_range': {'min': 0.80, 'max': 1.10}
    },
    'Bury Ground': {
        'full_name': 'Bury Ground River Monitoring Point',
        'latitude': 53.598766,
        'longitude': -2.305182,
        'river': 'River Irwell',
        'description': 'Monitoring station in Bury metropolitan area',
        'normal_range': {'min': 0.25, 'max': 0.40}
    }
}

class FloodMonitoringDashboard:
    def __init__(self):
        """Initialize dashboard components"""
        self.setup_supabase()
        self.predictor = FloodPredictor()
        
    def setup_supabase(self):
        """Initialize Supabase connection"""
        try:
            url = os.getenv('SUPABASE_URL')
            key = os.getenv('SUPABASE_KEY')
            self.supabase = create_client(url, key)
        except Exception as e:
            st.error(f"Supabase connection error: {str(e)}")
            self.supabase = None

    def fetch_data(self, days=7):
        """Fetch recent monitoring data"""
        try:
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=days)
            
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
            st.error(f"Data fetch error: {str(e)}")
            return None

    def create_map(self, data):
        """Create interactive station map"""
        stations_df = pd.DataFrame.from_dict(STATION_CONFIG, orient='index')
        stations_df.reset_index(inplace=True)
        stations_df.columns = ['Station', 'Full Name', 'Latitude', 'Longitude', 
                             'River', 'Description', 'Normal_Range_Min', 'Normal_Range_Max']
        
        # Add current levels if available
        if data is not None:
            current_levels = data.groupby('location_name')['river_level'].last()
            stations_df['Current_Level'] = stations_df['Station'].map(current_levels)
            stations_df['Risk_Level'] = stations_df.apply(
                lambda x: self.calculate_risk_level(x['Current_Level'], x['Station']), axis=1)
        
        fig = px.scatter_mapbox(
            stations_df,
            lat='Latitude',
            lon='Longitude',
            hover_name='Station',
            hover_data=['Current_Level', 'River', 'Description'],
            color='Risk_Level',
            color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'},
            zoom=9,
            height=600
        )
        fig.update_layout(mapbox_style="open-street-map")
        return fig

    def calculate_risk_level(self, level, station):
        """Calculate risk level based on river level"""
        if level is None:
            return 'Unknown'
            
        normal_range = STATION_CONFIG[station]['normal_range']
        if level > normal_range['max'] * 1.5:
            return 'High'
        elif level > normal_range['max']:
            return 'Medium'
        return 'Low'

    def create_trend_plot(self, data):
        """Create interactive trend plots"""
        fig = go.Figure()
        
        for station in data['location_name'].unique():
            station_data = data[data['location_name'] == station]
            
            # River levels
            fig.add_trace(go.Scatter(
                x=station_data['river_timestamp'],
                y=station_data['river_level'],
                name=f"{station} - Level",
                mode='lines+markers'
            ))
            
            # Add normal range
            normal_range = STATION_CONFIG[station]['normal_range']
            fig.add_hline(y=normal_range['max'], line_dash="dash", 
                         annotation_text=f"{station} Max Normal",
                         line_color="red", opacity=0.3)
            
        fig.update_layout(
            title="River Levels Over Time",
            xaxis_title="Time",
            yaxis_title="River Level (m)",
            height=500
        )
        return fig

    def display_predictions(self, data):
        """Display predictions for each station"""
        st.subheader("Predictions")
        cols = st.columns(len(STATION_CONFIG))
        
        for i, station in enumerate(STATION_CONFIG.keys()):
            with cols[i]:
                station_data = data[data['location_name'] == station]
                if not station_data.empty:
                    prediction, error = self.predictor.predict_next_level(station, station_data)
                    if prediction is not None:
                        risk_level = self.predictor.get_risk_level(prediction, station)
                        st.metric(
                            label=f"{station} Prediction",
                            value=f"{prediction:.3f}m",
                            delta=f"Risk: {risk_level}"
                        )
                    else:
                        st.warning(f"Could not predict for {station}: {error}")

def main():
    st.set_page_config(page_title="Flood Monitoring Dashboard", layout="wide")
    st.title("Flood Early Warning System")
    
    # Initialize dashboard
    dashboard = FloodMonitoringDashboard()
    
    # Data timeframe selector
    days_back = st.sidebar.slider("Select Data Timeframe (Days)", 1, 30, 7)
    
    # Fetch data
    data = dashboard.fetch_data(days=days_back)
    
    if data is not None:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Real-Time Monitoring",
            "Predictions",
            "Historical Trends",
            "Geospatial View"
        ])
        
        # Tab 1: Real-Time Monitoring
        with tab1:
            st.header("Current Station Metrics")
            cols = st.columns(len(STATION_CONFIG))
            
            for i, (station, info) in enumerate(STATION_CONFIG.items()):
                with cols[i]:
                    station_data = data[data['location_name'] == station]
                    if not station_data.empty:
                        current_level = station_data['river_level'].iloc[-1]
                        risk_level = dashboard.calculate_risk_level(current_level, station)
                        
                        st.metric(
                            label=station,
                            value=f"{current_level:.3f}m",
                            delta=f"Risk: {risk_level}",
                            delta_color="inverse" if risk_level == "High" else "normal"
                        )
        
        # Tab 2: Predictions
        with tab2:
            st.header("Flood Risk Predictions")
            dashboard.display_predictions(data)
        
        # Tab 3: Historical Trends
        with tab3:
            st.header("Historical Trends")
            trend_plot = dashboard.create_trend_plot(data)
            st.plotly_chart(trend_plot, use_container_width=True)
            
            # Station statistics
            st.subheader("Station Statistics")
            stats = data.groupby('location_name').agg({
                'river_level': ['mean', 'min', 'max'],
                'rainfall': ['mean', 'sum']
            }).round(3)
            st.dataframe(stats)
        
        # Tab 4: Geospatial View
        with tab4:
            st.header("Station Locations")
            map_fig = dashboard.create_map(data)
            st.plotly_chart(map_fig, use_container_width=True)
    
    else:
        st.error("Unable to fetch monitoring data. Please check your connection.")

if __name__ == "__main__":
    main()