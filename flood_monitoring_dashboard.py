
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
            self.watershed = WatershedAnalysis()
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

    def show_watershed_analysis(self, data):
        """Display watershed analysis tab"""
        st.header("Watershed Analysis")
        
        if data is not None:
            # Get current levels for each station
            current_levels = data.groupby('location_name')['river_level'].last()
            
            # Display flow paths
            st.subheader("Water Flow Network")
            cols = st.columns(3)
            
            for i, (station, level) in enumerate(current_levels.items()):
                with cols[i]:
                    st.write(f"**{station}**")
                    info = self.watershed.get_station_info(station)
                    risk = self.watershed.calculate_risk_score(station, level)
                    flow = self.watershed.get_flow_path(station)
                    
                    # Display station info
                    st.metric(
                        "Current Level",
                        f"{level:.3f}m",
                        f"Risk: {risk:.1f}%"
                    )
                    
                    # Station details
                    st.write(f"Elevation: {info['elevation']}m")
                    st.write(f"Catchment Area: {info['catchment_area']} km²")
                    
                    if flow:
                        st.write(f"Flows to: {flow['next_station']}")
                        st.write(f"Elevation difference: {flow['elevation_diff']}m")
                    
                    # Risk color coding
                    if risk >= 80:
                        st.error(f"High Risk: {risk:.1f}%")
                    elif risk >= 50:
                        st.warning(f"Moderate Risk: {risk:.1f}%")
                    else:
                        st.success(f"Low Risk: {risk:.1f}%")
            
            # Add flow visualization
            st.subheader("Flow Network Visualization")
            
            # Prepare data for visualization
            stations_list = list(current_levels.index)
            elevations = [self.watershed.station_info[s]['elevation'] for s in stations_list]
            water_levels = [level + self.watershed.station_info[s]['elevation'] 
                          for s, level in current_levels.items()]
            
            # Create figure
            fig = go.Figure()
            
            # Add elevation profile
            fig.add_trace(go.Scatter(
                x=stations_list,
                y=elevations,
                name='Elevation Profile',
                mode='lines+markers',
                line=dict(color='blue'),
                marker=dict(size=10)
            ))
            
            # Add water levels
            fig.add_trace(go.Scatter(
                x=stations_list,
                y=water_levels,
                name='Water Level',
                mode='lines+markers',
                line=dict(color='red', dash='dash'),
                marker=dict(size=10)
            ))
            
            # Update layout
            fig.update_layout(
                title='Station Elevations and Water Levels',
                xaxis_title='Stations',
                yaxis_title='Height (meters)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Add summary section
            st.subheader("Network Summary")
            summary_cols = st.columns(3)
            
            total_catchment = sum(info['catchment_area'] 
                                for info in self.watershed.station_info.values())
            elevation_range = (max(info['elevation'] 
                                 for info in self.watershed.station_info.values()) - 
                             min(info['elevation'] 
                                 for info in self.watershed.station_info.values()))
            avg_risk = (sum(self.watershed.calculate_risk_score(s, l) 
                          for s, l in current_levels.items()) / 
                       len(current_levels))
            
            with summary_cols[0]:
                st.metric("Total Catchment Area", f"{total_catchment:.1f} km²")
            
            with summary_cols[1]:
                st.metric("Elevation Range", f"{elevation_range}m")
            
            with summary_cols[2]:
                st.metric("Average Network Risk", f"{avg_risk:.1f}%")
    
    
    def show_alerts(self, data):
        """Display alerts tab"""
        st.header("Flood Alert System")
        
        if data is not None:
            # Current Alerts Section
            st.subheader("Current Alert Status")
            alert_cols = st.columns(3)
            
            # Initialize thresholds
            thresholds = {
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
            
            for i, station in enumerate(data['location_name'].unique()):
                with alert_cols[i]:
                    station_data = data[data['location_name'] == station].iloc[0]
                    current_level = station_data['river_level']
                    
                    # Get trend
                    station_history = data[data['location_name'] == station]
                    trend = "Stable"
                    if len(station_history) > 1:
                        level_change = station_history['river_level'].diff().mean()
                        if abs(level_change) < 0.0001:
                            trend = "Stable"
                        elif level_change > 0:
                            trend = "Rising"
                        else:
                            trend = "Falling"
                    
                    # Determine alert status
                    station_thresholds = thresholds[station]
                    if current_level > station_thresholds['critical']:
                        status = 'CRITICAL'
                        status_color = 'red'
                        message = 'Immediate action required'
                    elif current_level > station_thresholds['alert']:
                        status = 'ALERT'
                        status_color = 'orange'
                        message = 'Prepare for potential flooding'
                    elif current_level > station_thresholds['warning']:
                        status = 'WARNING'
                        status_color = 'yellow'
                        message = 'Monitor conditions closely'
                    else:
                        status = 'NORMAL'
                        status_color = 'green'
                        message = 'Normal conditions'
                    
                    # Display alert
                    st.write(f"**{station}**")
                    st.metric(
                        "Current Level",
                        f"{current_level:.3f}m",
                        f"Trend: {trend}"
                    )
                    
                    st.markdown(
                        f"<div style='padding: 10px; background-color: {status_color}; "
                        f"color: black; border-radius: 5px; text-align: center;'>"
                        f"Status: {status}</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.write(f"**Message:** {message}")
    
    
    def show_advanced_analytics(self, data):
        """Display advanced analytics tab"""
        st.header("Advanced Analytics")
        
        if data is not None:
            analytics = AdvancedAnalytics()
            
            # Station Analysis
            st.subheader("Station Analysis")
            for station in data['location_name'].unique():
                with st.expander(f"{station} Analysis", expanded=True):
                    analysis = analytics.analyze_station(data, station)
                    
                    # Current Status
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Current Level",
                            f"{analysis['current_level']:.3f}m",
                            f"{analysis['deviation']:.3f}m vs baseline"
                        )
                    with col2:
                        st.metric(
                            "Trend",
                            analysis['trend'],
                            f"{analysis['trend_rate']:.6f}m/hour"
                        )
                    with col3:
                        st.metric(
                            "Average Level",
                            f"{analysis['average_level']:.3f}m"
                        )
                    
                    # Daily Pattern
                    st.write("**Daily Pattern:**")
                    st.write(f"Peak levels typically at: {analysis['peak_hour']}:00")
                    st.write(f"Lowest levels typically at: {analysis['low_hour']}:00")
                    
                    # Forecast
                    st.write("**24-Hour Forecast:**")
                    forecast = analytics.get_forecast(data, station)
                    
                    # Plot forecast
                    fig = go.Figure()
                    
                    # Historical data
                    recent_data = data[data['location_name'] == station].tail(24)
                    fig.add_trace(go.Scatter(
                        x=recent_data['river_timestamp'],
                        y=recent_data['river_level'],
                        name='Historical',
                        mode='lines+markers'
                    ))
                    
                    # Forecast data
                    forecast_times = pd.date_range(
                        start=recent_data['river_timestamp'].iloc[-1],
                        periods=25,
                        freq='H'
                    )
                    fig.add_trace(go.Scatter(
                        x=forecast_times,
                        y=[recent_data['river_level'].iloc[-1]] + forecast['levels'],
                        name='Forecast',
                        mode='lines',
                        line=dict(dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"{station} - 24 Hour Forecast",
                        xaxis_title="Time",
                        yaxis_title="River Level (m)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(f"Forecast Confidence: {forecast['confidence']}")
    
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Real-Time Monitoring",
        "Predictions",
        "Historical Trends",
        "Station Details",
        "Geospatial View",
        "Watershed Analysis",
        "Alerts",
        "Advanced Analytics"  # New tab
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
        
    with tab6:
        dashboard.show_watershed_analysis(river_data)
    
    with tab7:
        dashboard.show_alerts(river_data)
    
    with tab8:
        dashboard.show_advanced_analytics(river_data)  # New advanced analytics tab

    # Optional: Update query parameters
    st.query_params.update(refresh=True)

if __name__ == '__main__':
    main()