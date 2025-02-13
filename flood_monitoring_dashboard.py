import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import os
import pytz
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objs as go
from prediction_utils import FloodPredictor
from watershed_utils import WatershedAnalysis

# Add prediction system class
class FloodPredictionSystem:
    def __init__(self):
        self.warning_levels = {
            'Rochdale': 0.3,
            'Manchester Racecourse': 1.1,
            'Bury Ground': 0.4
        }
    
    def analyze_trend(self, data, window=24):
        recent_data = data.tail(window)
        level_trend = recent_data['river_level'].diff().mean()
        stability = recent_data['river_level'].std()
        confidence = 1 - min(1, stability * 10)
        
        if abs(level_trend) < 0.0001:
            trend_direction = "Stable"
        elif level_trend > 0:
            trend_direction = "Rising"
        else:
            trend_direction = "Falling"
            
        return trend_direction, level_trend, confidence
    
    def get_risk_level(self, current_level, station):
        warning_level = self.warning_levels[station]
        if current_level > warning_level:
            return "HIGH", "red"
        elif current_level > warning_level * 0.8:
            return "MODERATE", "yellow"
        return "LOW", "green"

# Add to your imports at the top
from watershed_utils import WatershedAnalysis

# In your FloodMonitoringDashboard class, add this method:
def show_watershed_analysis(self, data):
    """Show watershed analysis tab"""
    st.header("Watershed Analysis")
    
    if data is not None:
        # Initialize watershed analysis
        watershed = WatershedAnalysis()
        
        # Get current levels and rainfall
        current_levels = {
            row['location_name']: row['river_level'] 
            for _, row in data.groupby('location_name').last().iterrows()
        }
        rainfall_data = {
            row['location_name']: row['rainfall']
            for _, row in data.groupby('location_name').last().iterrows()
        }
        
        # Get watershed summary
        summary = watershed.get_watershed_summary()
        
        # Display watershed overview
        st.subheader("Watershed Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Catchment Area", f"{summary['total_area']:.1f} km²")
        with col2:
            st.metric("Elevation Range", 
                     f"{summary['elevation_range']['min']}m - {summary['elevation_range']['max']}m")
        with col3:
            st.metric("Number of Flow Paths", len(summary['flow_paths']))
        
        # Show flow paths
        st.subheader("Water Flow Paths")
        for path in summary['flow_paths']:
            st.write(f"- {path}")
        
        # Get and display impact analysis
        st.subheader("Station Analysis")
        impact = watershed.analyze_flow_impact(current_levels, rainfall_data)
        
        # Display results in columns
        cols = st.columns(3)
        for i, (station, analysis) in enumerate(impact.items()):
            with cols[i]:
                st.write(f"**{station}**")
                st.write(f"Catchment Area: {analysis['catchment_area']} km²")
                st.write(f"Elevation: {analysis['elevation']}m")
                st.write(f"Risk Score: {analysis['risk_score']:.1f}%")
                st.write(f"Downstream Impact: {analysis['downstream_impact']}")

# In your main() function, update your tabs:
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Real-Time Monitoring",
    "Predictions",
    "Historical Trends",
    "Station Details",
    "Watershed Analysis"  # New tab
])

# And add the watershed tab:
with tab5:
    dashboard.show_watershed_analysis(river_data)


class FloodPredictionSystem:
    def __init__(self):
        # Updated thresholds from our analysis
        self.thresholds = {
            'Rochdale': {
                'warning': 0.168,
                'alert': 0.169,
                'critical': 0.170
            },
            'Manchester Racecourse': {
                'warning': 0.938,
                'alert': 0.944,
                'critical': 0.950
            },
            'Bury Ground': {
                'warning': 0.314,
                'alert': 0.317,
                'critical': 0.320
            }
        }
    
    def analyze_trend(self, data, window=24):
        recent_data = data.tail(window)
        level_trend = recent_data['river_level'].diff().mean()
        stability = recent_data['river_level'].std()
        confidence = 1 - min(1, stability * 10)
        
        if abs(level_trend) < 0.0001:
            trend_direction = "Stable"
        elif level_trend > 0:
            trend_direction = "Rising"
        else:
            trend_direction = "Falling"
            
        return trend_direction, level_trend, confidence
    
    def get_risk_level(self, current_level, station):
        """Get risk level based on analyzed thresholds"""
        thresholds = self.thresholds[station]
        
        if current_level > thresholds['critical']:
            return "HIGH", "red"
        elif current_level > thresholds['alert']:
            return "MODERATE", "yellow"
        return "LOW", "green"

def show_predictions(self, data):
    """Display predictions tab"""
    st.header("River Level Predictions")
    if data is not None:
        cols = st.columns(3)
        for i, station in enumerate(data['location_name'].unique()):
            with cols[i]:
                station_data = data[data['location_name'] == station].copy()
                current_level = station_data['river_level'].iloc[0]
                
                trend_direction, trend_rate, confidence = self.predictor.analyze_trend(station_data)
                risk_level, risk_color = self.predictor.get_risk_level(current_level, station)
                
                with st.expander(f"{station} Prediction Details", expanded=True):
                    st.metric(
                        "Current Level",
                        f"{current_level:.3f}m",
                        f"Risk: {risk_level}",
                        delta_color="inverse" if risk_level == "HIGH" else "normal"
                    )
                    
                    st.write(f"**Trend:** {trend_direction}")
                    st.write(f"**Rate of Change:** {trend_rate:.6f}m/hour")
                    st.write(f"**Prediction Confidence:** {confidence:.1%}")
                    
                    thresholds = self.predictor.thresholds[station]
                    st.write("**Warning Thresholds:**")
                    st.write(f"- Warning: {thresholds['warning']:.3f}m")
                    st.write(f"- Alert: {thresholds['alert']:.3f}m")
                    st.write(f"- Critical: {thresholds['critical']:.3f}m")
                    
                    st.markdown(
                        f"<div style='padding: 10px; background-color: {risk_color}; "
                        f"color: black; border-radius: 5px; text-align: center;'>"
                        f"Risk Level: {risk_level}</div>",
                        unsafe_allow_html=True
                    )
        
        # Trend visualization
        st.subheader("Recent Trends")
        fig = go.Figure()
        
        for station in data['location_name'].unique():
            station_data = data[data['location_name'] == station].head(24)
            fig.add_trace(go.Scatter(
                x=station_data['river_timestamp'],
                y=station_data['river_level'],
                name=station,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="River Levels - Last 6 Hours",
            xaxis_title="Time",
            yaxis_title="River Level (m)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

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
        """Initialize dashboard components"""
        try:
            supabase_url = st.secrets["SUPABASE_URL"]
            supabase_key = st.secrets["SUPABASE_KEY"]
            self.supabase = create_client(supabase_url, supabase_key)
            self.predictor = FloodPredictionSystem()
        except Exception as e:
            st.error(f"Failed to initialize dashboard: {e}")
            self.supabase = None

    def fetch_river_data(self, days_back=30):
        """Fetch river monitoring data"""
        try:
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=days_back)

            response = self.supabase.table('river_data')\
                .select('*')\
                .gte('river_timestamp', start_date.isoformat())\
                .lte('river_timestamp', end_date.isoformat())\
                .order('river_timestamp', desc=True)\
                .execute()

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

    def show_real_time_monitoring(self, data):
        """Display real-time monitoring tab"""
        st.header("Current Station Metrics")
        if data is not None:
            cols = st.columns(3)
            for i, station in enumerate(data['location_name'].unique()):
                with cols[i]:
                    station_data = data[data['location_name'] == station]
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

    def show_predictions(self, data):
        """Display predictions tab"""
        st.header("River Level Predictions")
        if data is not None:
            cols = st.columns(3)
            for i, station in enumerate(data['location_name'].unique()):
                with cols[i]:
                    station_data = data[data['location_name'] == station].copy()
                    current_level = station_data['river_level'].iloc[0]
                    
                    trend_direction, trend_rate, confidence = self.predictor.analyze_trend(station_data)
                    risk_level, risk_color = self.predictor.get_risk_level(current_level, station)
                    
                    with st.expander(f"{station} Prediction Details", expanded=True):
                        st.metric(
                            "Current Level",
                            f"{current_level:.3f}m",
                            f"Risk: {risk_level}",
                            delta_color="inverse" if risk_level == "HIGH" else "normal"
                        )
                        
                        st.write(f"**Trend:** {trend_direction}")
                        st.write(f"**Rate of Change:** {trend_rate:.6f}m/hour")
                        st.write(f"**Prediction Confidence:** {confidence:.1%}")
                        
                        st.markdown(
                            f"<div style='padding: 10px; background-color: {risk_color}; "
                            f"color: black; border-radius: 5px; text-align: center;'>"
                            f"Risk Level: {risk_level}</div>",
                            unsafe_allow_html=True
                        )
            
            # Trend visualization
            st.subheader("Recent Trends")
            fig = go.Figure()
            
            for station in data['location_name'].unique():
                station_data = data[data['location_name'] == station].head(24)
                fig.add_trace(go.Scatter(
                    x=station_data['river_timestamp'],
                    y=station_data['river_level'],
                    name=station,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title="River Levels - Last 6 Hours",
                xaxis_title="Time",
                yaxis_title="River Level (m)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def show_historical_trends(self, data):
        """Display historical trends tab"""
        st.header("Historical Data Analysis")
        if data is not None:
            # Basic statistics overview
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(data))
            
            with col2:
                st.metric("Date Range", 
                          f"{data['river_timestamp'].min().date()} to {data['river_timestamp'].max().date()}")
            
            with col3:
                st.metric("Stations", ", ".join(data['location_name'].unique()))

            # Trend visualization
            st.subheader("Trends Visualization")
            fig = go.Figure()
            
            # River Levels
            for station in data['location_name'].unique():
                station_data = data[data['location_name'] == station]
                fig.add_trace(go.Scatter(
                    x=station_data['river_timestamp'],
                    y=station_data['river_level'],
                    name=f"{station} - Level",
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title="River Levels Over Time",
                xaxis_title="Time",
                yaxis_title="River Level (m)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Station summary
            st.subheader("Station-wise Summary")
            station_summary = data.groupby('location_name').agg({
                'river_level': ['mean', 'min', 'max', 'count'],
                'rainfall': ['mean', 'min', 'max', 'count'],
                'river_timestamp': ['min', 'max']
            })

            station_summary.columns = [
                'Avg River Level', 'Min River Level', 'Max River Level', 'River Level Readings',
                'Avg Rainfall', 'Min Rainfall', 'Max Rainfall', 'Rainfall Readings',
                'First Timestamp', 'Last Timestamp'
            ]

            numeric_cols = [
                'Avg River Level', 'Min River Level', 'Max River Level',
                'Avg Rainfall', 'Min Rainfall', 'Max Rainfall'
            ]
            station_summary[numeric_cols] = station_summary[numeric_cols].round(3)

            st.dataframe(station_summary)

    def show_station_details(self, data):
        """Display station details tab"""
        st.header("Station Information")
        selected_station = st.selectbox(
            "Select Station", 
            list(STATION_CONFIG.keys())
        )
        
        station_info = STATION_CONFIG[selected_station]
        st.write(f"**Full Name:** {station_info['full_name']}")
        st.write(f"**River:** {station_info['river']}")
        st.write(f"**Description:** {station_info['description']}")
        st.write(f"**Risk Level:** {station_info['risk_level']}")
        st.write(f"**Coordinates:** {station_info['latitude']}, {station_info['longitude']}")

        if data is not None:
            st.subheader("Latest Readings")
            station_data = data[data['location_name'] == selected_station].iloc[0]
            st.write(f"**River Level:** {station_data['river_level']:.3f}m")
            st.write(f"**Rainfall:** {station_data['rainfall']:.3f}mm")
            st.write(f"**Timestamp:** {station_data['river_timestamp']}")

    def show_geospatial_view(self, data):
        """Display geospatial view tab"""
        st.header("Station Geographic Distribution")
        
        # Create station data for map
        stations_df = pd.DataFrame.from_dict(STATION_CONFIG, orient='index')
        stations_df.reset_index(inplace=True)
        stations_df.columns = ['Station', 'Full Name', 'Latitude', 'Longitude', 'River', 'Description', 'Risk Level']

        if data is not None:
            # Add current levels to stations
            current_levels = data.groupby('location_name')['river_level'].first()
            stations_df['Current Level'] = stations_df['Station'].map(current_levels)

        # Create map
        fig = px.scatter_mapbox(
            stations_df, 
            lat='Latitude', 
            lon='Longitude',
            hover_name='Station',
            hover_data=['Full Name', 'River', 'Description', 'Risk Level', 'Current Level'],
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
        st.plotly_chart(fig, use_container_width=True)


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Real-Time Monitoring",
        "Predictions",
        "Historical Trends",
        "Station Details",
        "Geospatial View"
    ])

    # Fetch river data
    river_data = dashboard.fetch_river_data()

    # Display tabs
    with tab1:
        dashboard.show_real_time_monitoring(river_data)
    
    with tab2:
        dashboard.show_predictions(river_data)
    
    with tab3:
        dashboard.show_historical_trends(river_data)
    
    with tab4:
        dashboard.show_station_details(river_data)
    
    with tab5:
        dashboard.show_geospatial_view(river_data)

    # Optional: Update query parameters
    st.query_params.update(refresh=True)

if __name__ == '__main__':
    main()