import streamlit as st
import requests 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from supabase import create_client
import os
import pytz
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objs as go
from prediction_utils import FloodPredictor
from watershed_utils import WatershedAnalysis
from dotenv import load_dotenv
from notification_system import NotificationSystem
from alert_config import AlertConfiguration
from alert_history import AlertHistoryTracker
from alert_system import AlertSystem  
from anomaly_detector import AnomalyDetector
from confidence_intervals import ConfidenceIntervalCalculator
from enhanced_alert_system import EnhancedAlertSystem

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

class FloodPredictionSystem:
    def __init__(self):
        # FIXED: Realistic thresholds based on UK Environment Agency data
        self.thresholds = {
            'Rochdale': {
                'warning': 1.0,      # Typical range: 0.098m to 1.8m
                'alert': 1.4,
                'critical': 1.8
            },
            'Manchester Racecourse': {
                'warning': 2.5,
                'alert': 3.0,
                'critical': 3.5
            },
            'Bury Ground': {
                'warning': 1.5,
                'alert': 2.0,
                'critical': 2.5
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
        """Get risk level based on analyzed thresholds with more granularity"""
        thresholds = self.thresholds[station]
        
        # MODIFIED: Added buffer zone between alert and critical levels
        if current_level >= thresholds['critical']:
            return "HIGH", "red"
        # Added buffer category to reduce immediate jumps to HIGH
        elif current_level >= (thresholds['critical'] - 0.01):
            return "ELEVATED", "orange"
        elif current_level >= thresholds['alert']:
            return "MODERATE", "orange"
        elif current_level >= thresholds['warning']:
            return "LOW", "yellow"
        else:
            return "NORMAL", "green"

class EnhancedAlertSystem:
    def __init__(self):
        # You can modify this section if needed to change alert behavior
        pass
        
    def generate_alert(self, station, current_level):
        # This is a placeholder - your actual implementation would need to be adjusted
        # based on the modified thresholds in FloodPredictionSystem
        predictor = FloodPredictionSystem()
        risk_level, _ = predictor.get_risk_level(current_level, station)
        
        # Map risk levels to alert levels
        alert_level_map = {
            "HIGH": "CRITICAL",
            "ELEVATED": "HIGH",  # New buffer category
            "MODERATE": "WARNING",
            "LOW": "MONITOR",
            "NORMAL": "NORMAL"
        }
        
        alert_level = alert_level_map.get(risk_level, "MONITOR")
        
        # Create description based on alert level
        descriptions = {
            "CRITICAL": "Immediate action required. Critical flood risk detected.",
            "HIGH": "High flood risk. Prepare for possible evacuation.",
            "WARNING": "Moderate flood risk. Monitor conditions closely.",
            "MONITOR": "Low flood risk. Stay informed of changing conditions.",
            "NORMAL": "Normal conditions. No flood risk at this time."
        }
        
        return {
            "station": station,
            "alert_level": alert_level,
            "current_level": current_level,
            "description": descriptions.get(alert_level, "Unknown risk level"),
            "timestamp": datetime.now(pytz.UTC)
        }
    
    def send_notifications(self, alert):
        # This is a placeholder for your notification logic
        # Modify if needed based on new alert thresholds
        channels = []
        
        if alert['alert_level'] == 'CRITICAL':
            channels = ['email', 'sms', 'app', 'emergency_services']
        elif alert['alert_level'] == 'HIGH':
            channels = ['email', 'sms', 'app']
        elif alert['alert_level'] == 'WARNING':
            channels = ['email', 'app']
        elif alert['alert_level'] == 'MONITOR':
            channels = ['app']
        
        return channels
    
    def get_recent_alerts(self, days=7):
        # This is a placeholder - in a real system, you'd fetch from a database
        return pd.DataFrame({
            'station': [],
            'alert_level': [],
            'current_level': [],
            'description': [],
            'timestamp': []
        })

# The rest of the code remains unchanged
# ...

class AdvancedAnalytics:
    def __init__(self):
        self.baseline_levels = {
            'Rochdale': 0.168,
            'Manchester Racecourse': 0.928,
            'Bury Ground': 0.311
        }
    
    def analyze_station(self, data, station_name):
        """Detailed analysis for a single station"""
        station_data = data[data['location_name'] == station_name].copy()
        station_data = station_data.sort_values('river_timestamp')
        
        # Basic statistics
        current_level = station_data['river_level'].iloc[-1]
        avg_level = station_data['river_level'].mean()
        baseline = self.baseline_levels[station_name]
        deviation = current_level - baseline
        
        # Trend analysis (last 24 hours)
        recent_data = station_data.tail(24)
        trend = recent_data['river_level'].diff().mean()
        
        if abs(trend) < 0.0001:
            trend_direction = "Stable"
        elif trend > 0:
            trend_direction = "Rising"
        else:
            trend_direction = "Falling"
        
        hourly_pattern = station_data.groupby(
            station_data['river_timestamp'].dt.hour
        )['river_level'].mean()
        
        peak_hour = hourly_pattern.idxmax()
        low_hour = hourly_pattern.idxmin()
        
        return {
            'current_level': current_level,
            'average_level': avg_level,
            'baseline': baseline,
            'deviation': deviation,
            'trend': trend_direction,
            'trend_rate': trend,
            'peak_hour': peak_hour,
            'low_hour': low_hour,
            'hourly_pattern': hourly_pattern.to_dict()
        }
    
    def get_forecast(self, data, station_name, hours_ahead=24):
        """Generate forecast for a station"""
        station_data = data[data['location_name'] == station_name].copy()
        current_level = station_data['river_level'].iloc[-1]
        trend = station_data['river_level'].diff().mean()
        
        forecast = []
        for hour in range(hours_ahead):
            predicted_level = current_level + (trend * hour)
            predicted_level = max(0, predicted_level)
            forecast.append(predicted_level)
        
        return {
            'levels': forecast,
            'confidence': 'High' if abs(trend) < 0.001 else 'Medium'
        }

class FloodMonitoringDashboard:
    def __init__(self):
        """Initialize dashboard components"""
        try:
            # Load environment variables
            load_dotenv()
            
            # Debug: Print if secrets are loaded
            if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
                st.sidebar.success("Supabase credentials loaded successfully")
            else:
                st.sidebar.error("Supabase credentials not found")
                
            # Initialize Supabase client
            supabase_url = st.secrets["SUPABASE_URL"]
            supabase_key = st.secrets["SUPABASE_KEY"]
            self.supabase = create_client(supabase_url, supabase_key)
            
            # Test the connection
            test_response = self.supabase.table('river_data').select('*').limit(1).execute()
            if test_response.data:
                st.sidebar.success("Supabase connection successful")
            else:
                st.sidebar.warning("Connected to Supabase but no data found")
            
            # Initialize components
            self.predictor = FloodPredictionSystem()
            self.watershed = WatershedAnalysis()
            self.analytics = AdvancedAnalytics()
            
            # Initialize alert system components
            self.alert_system = AlertSystem()
            self.notification_system = NotificationSystem()
            self.alert_config = AlertConfiguration()
            self.alert_history = AlertHistoryTracker()
            
            # Add anomaly detector
            self.anomaly_detector = AnomalyDetector()
            
            # Add this line
            self.ci_calculator = ConfidenceIntervalCalculator()
        
            self.enhanced_alert_system = EnhancedAlertSystem()
        
        
        except Exception as e:
            st.error(f"Failed to initialize dashboard: {e}")
            self.supabase = None

def fetch_river_data(self, days_back=7):
    """Fetch river monitoring data from real API"""
    try:
        # Try UK Environment Agency API with correct measure IDs
        stations = {
            'Rochdale': '690203-level-stage-i-15_min-m',
            'Manchester Racecourse': '690510-level-stage-i-15_min-m',
            'Bury Ground': '690160-level-stage-i-15_min-m'
        }
        
        all_data = []
        
        for station_name, measure_id in stations.items():
            url = f"https://environment.data.gov.uk/flood-monitoring/id/measures/{measure_id}/readings?_sorted&_limit=500"
            
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    readings = data.get('items', [])
                    
                    for reading in readings:
                        all_data.append({
                            'river_timestamp': pd.to_datetime(reading.get('dateTime')),
                            'location_name': station_name,
                            'river_level': reading.get('value', 0),
                            'rainfall': 0,
                            'rainfall_timestamp': pd.to_datetime(reading.get('dateTime'))
                        })
            except Exception as e:
                continue
        
        if all_data:
            df = pd.DataFrame(all_data)
            df = df.sort_values('river_timestamp', ascending=False)
            return df
                    
    except Exception:
        pass
    
    # Fallback to simulated data
    return self._generate_sample_data(days_back)

    def _generate_sample_data(self, days_back=90):
        """Generate sample river monitoring data with current timestamps"""
        # Use current time as the end date
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=days_back)
        
        # Generate hourly timestamps for the entire period
        dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq='H'
        )
        
        # Ensure all stations are included
        stations = list(STATION_CONFIG.keys())
        base_levels = {
            'Rochdale': 0.173,
            'Manchester Racecourse': 0.927,
            'Bury Ground': 0.311
        }
        
        data = []
        for station in stations:
            base_level = base_levels.get(station, 0.2)  # Default level if not in dictionary
            for date in dates:
                # Daily variation
                hour_effect = 0.02 * np.sin(2 * np.pi * date.hour / 24)
                # Monthly variation
                day_effect = 0.03 * np.sin(2 * np.pi * date.day / 30)
                # Random variation
                random_effect = np.random.normal(0, 0.005)
                
                # Combine effects
                level = max(0, base_level + hour_effect + day_effect + random_effect)
                
                data.append({
                    'river_timestamp': date,
                    'location_name': station,
                    'river_level': level,
                    'rainfall': np.random.uniform(0, 0.2) if np.random.random() < 0.3 else 0,
                    'rainfall_timestamp': date
                })
        
        # Sort data by timestamp to ensure most recent data is first
        df = pd.DataFrame(data)
        return df.sort_values('river_timestamp', ascending=False)
    
    
    def show_real_time_monitoring(self, data):
        """Display real-time monitoring tab"""
        st.header("Current Station Metrics")
        if data is not None:
            unique_stations = data['location_name'].unique()
            num_stations = len(unique_stations)
            
            # Dynamically create the correct number of columns
            cols = st.columns(num_stations)
            
            for i, station in enumerate(unique_stations):
                with cols[i % num_stations]:  # Use modulo to ensure we don't exceed column count
                    station_data = data[data['location_name'] == station]
                    if len(station_data) > 0:  # Check if we have data for this station
                        river_level = station_data['river_level'].values[0]
                        
                        # Get risk thresholds from predictor
                        thresholds = self.predictor.thresholds.get(station, {
                            'warning': 0.4, 'alert': 0.6, 'critical': 0.8
                        })
                        
                        # Determine risk level
                        risk_text = 'Low Risk'
                        delta_color = 'normal'
                        
                        if river_level > thresholds['critical']:
                            risk_text = 'High Risk'
                            delta_color = 'inverse'
                        elif river_level > thresholds['alert']:
                            risk_text = 'Moderate Risk'
                            delta_color = 'off'
                        
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
            # Sort data by most recent timestamp and get the current time
            data = data.sort_values('river_timestamp', ascending=False)
            current_time = pd.Timestamp.now(tz='UTC')
            
            cols = st.columns(3)
            for i, station in enumerate(data['location_name'].unique()):
                with cols[i]:
                    # Get most recent station data
                    station_data = data[data['location_name'] == station].copy()
                    current_level = station_data['river_level'].iloc[0]
                    
                    # Analyze trend on recent data
                    recent_data = station_data.head(24)  # Last 24 hours
                    trend_direction, trend_rate, confidence = self.predictor.analyze_trend(recent_data)
                    risk_level, risk_color = self.predictor.get_risk_level(current_level, station)
                    
                    # Calculate Confidence Interval
                    ci_data = self.ci_calculator.calculate_interval(data, station)
                    
                    with st.expander(f"{station} Prediction Details", expanded=True):
                        st.metric(
                            "Current Level",
                            f"{current_level:.3f}m",
                            f"Risk: {risk_level}",
                            delta_color="inverse" if risk_level == "HIGH" else "normal"
                        )
                        
                        # Confidence Interval Metric
                        st.metric(
                            "Confidence Interval",
                            f"{ci_data['confidence_percentage']:.1f}% CI",
                            f"±{ci_data['margin_of_error']:.3f}m"
                        )
                        
                        # Additional confidence interval details
                        st.write(f"**Lower Bound:** {ci_data['lower_bound']:.3f}m")
                        st.write(f"**Upper Bound:** {ci_data['upper_bound']:.3f}m")
                        
                        
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
                # Get last 24 hours of data for each station
                station_data = data[data['location_name'] == station].head(24)
                fig.add_trace(go.Scatter(
                    x=station_data['river_timestamp'],
                    y=station_data['river_level'],
                    name=station,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title=f"River Levels - Last 24 Hours (as of {current_time.strftime('%Y-%m-%d %H:%M')})",
                xaxis_title="Time",
                yaxis_title="River Level (m)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
	
            st.subheader("Anomaly Detection")
            anomaly_summary = {}

            for station in data['location_name'].unique():
                anomalies = self.anomaly_detector.detect_anomalies(data, station)
                
                if not anomalies.empty:
                    # Store summary for each station
                    anomaly_summary[station] = {
                        'total_count': len(anomalies),
                        'high_level': len(anomalies[anomalies['anomaly_type'] == 'High Level']),
                        'low_level': len(anomalies[anomalies['anomaly_type'] == 'Low Level'])
                    }

            # Display anomaly summary
            if anomaly_summary:
                col1, col2, col3 = st.columns(3)
                
                for i, (station, summary) in enumerate(anomaly_summary.items()):
                    with [col1, col2, col3][i]:
                        st.metric(
                            station, 
                            f"Total Anomalies: {summary['total_count']}",
                            delta=f"High: {summary['high_level']} | Low: {summary['low_level']}"
                        )
                
                # Detailed view toggle
                if st.checkbox("Show Detailed Anomalies"):
                    for station, anomalies in anomaly_summary.items():
                        st.write(f"### {station} Anomalies")
                        detailed_anomalies = self.anomaly_detector.detect_anomalies(data, station)
                        st.dataframe(detailed_anomalies[['river_timestamp', 'river_level', 'anomaly_type', 'z_score']])
            else:
                st.success("No significant anomalies detected across stations")           
    
    
    def show_historical_trends(self, data):
        """Display historical trends tab with proper date range"""
        st.header("Historical Data Analysis")
        
        if data is not None:
            # Force data regeneration for 90 days
            data = self._generate_sample_data(days_back=90)
            
            # Date range selection
            st.subheader("Select Date Range")
            min_date = data['river_timestamp'].min().date()
            max_date = data['river_timestamp'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=max_date - timedelta(days=7),
                    min_value=min_date,
                    max_value=max_date
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )

            # Filter data based on selected dates
            mask = (data['river_timestamp'].dt.date >= start_date) & (data['river_timestamp'].dt.date <= end_date)
            filtered_data = data[mask]

            # Display data overview
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(filtered_data))
            
            with col2:
                date_range = f"{filtered_data['river_timestamp'].min().date()} to {filtered_data['river_timestamp'].max().date()}"
                st.metric("Date Range", date_range)
            
            with col3:
                st.metric("Stations", ", ".join(filtered_data['location_name'].unique()))

            # Trend visualization
            st.subheader("Trends Visualization")
            fig = go.Figure()
            
            for station in filtered_data['location_name'].unique():
                station_data = filtered_data[filtered_data['location_name'] == station]
                fig.add_trace(go.Scatter(
                    x=station_data['river_timestamp'],
                    y=station_data['river_level'],
                    name=station,
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
            station_summary = filtered_data.groupby('location_name').agg({
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
            filtered_data = data[data['location_name'] == selected_station]
            if not filtered_data.empty:  # If there is data
                station_data = filtered_data.iloc[0]
                st.write(f"**River Level:** {station_data['river_level']:.3f}m")
                st.write(f"**Rainfall:** {station_data['rainfall']:.3f}mm")
                st.write(f"**Timestamp:** {station_data['river_timestamp']}")
            else:  # If there is NO data
                st.warning(f"No data available for {selected_station}.")


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

            # Add Network Summary
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
        """Display flood alerts tab with enhanced alert system"""
        st.header("Flood Alerts and Warnings")
        if data is None:
            st.warning("No data available for generating alerts")
            return

        # Create tabs for current alerts and history
        current_tab, history_tab = st.tabs(["Current Alerts", "Alert History"])
        
        with current_tab:
            # Generate and display alerts for each station
            for station in data['location_name'].unique():
                station_data = data[data['location_name'] == station]
                current_level = station_data['river_level'].iloc[0]
                
                # Use enhanced alert system to generate alert
                alert = self.enhanced_alert_system.generate_alert(station, current_level)
                
                # Send notifications
                notification_channels = self.enhanced_alert_system.send_notifications(alert)
                
                # Display alert with styling based on alert level
                alert_styling = {
                    'CRITICAL': (st.error, "🚨", "red"),
                    'HIGH': (st.error, "🚨", "red"),
                    'WARNING': (st.warning, "⚠️", "orange"),
                    'MONITOR': (st.info, "ℹ️", "yellow"),
                    'NORMAL': (st.success, "✅", "green")
                }
                
                # Get appropriate styling
                alert_display, icon, color = alert_styling.get(
                    alert['alert_level'], 
                    (st.info, "ℹ️", "blue")
                )
                
                # Display alert
                alert_display(f"{icon} **{alert['alert_level']} FLOOD RISK** at {station}")
                
                # Metrics and additional information
                st.metric(
                    label=f"{station} Current Level",
                    value=f"{current_level:.3f}m",
                    delta=f"{alert['alert_level']} Risk"
                )
                
                # Additional alert details
                with st.expander("Alert Details"):
                    st.write(f"**Description:** {alert['description']}")
                    st.write(f"**Notification Channels:** {', '.join(notification_channels) or 'No channels notified'}")
        
        with history_tab:
            # Retrieve and display recent alerts
            recent_alerts = self.enhanced_alert_system.get_recent_alerts(days=7)
            
            if not recent_alerts.empty:
                st.subheader("Recent Alerts (Last 7 Days)")
                
                # Format the dataframe for display
                display_df = recent_alerts.copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                display_df['current_level'] = display_df['current_level'].round(3)
                
                # Highlight rows based on alert level
                def color_alert_levels(row):
                    if row['alert_level'] == 'CRITICAL':
                        return ['background-color: #ffcccc']*len(row)
                    elif row['alert_level'] == 'HIGH':
                        return ['background-color: #ffeeee']*len(row)
                    elif row['alert_level'] == 'WARNING':
                        return ['background-color: #fff2cc']*len(row)
                    return [''] * len(row)
                
                # Display dataframe with conditional formatting
                st.dataframe(
                    display_df.style.apply(color_alert_levels, axis=1),
                    hide_index=True
                )
            else:
                st.info("No alerts in the past 7 days")
            
            # Emergency guidance section
            st.subheader("Emergency Guidance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### For Low to Moderate Risk
                - Stay informed
                - Monitor local news
                - Prepare emergency kit
                - Check local flood warnings
                """)
            
            with col2:
                st.markdown("""
                ### For High to Critical Risk
                - Prepare for immediate evacuation
                - Move to higher ground
                - Protect critical belongings
                - Follow official emergency guidance
                """)

    def show_advanced_analytics(self, data):
        """Display advanced analytics tab"""
        st.header("Advanced Analytics")
        
        if data is not None:
            for station in data['location_name'].unique():
                with st.expander(f"{station} Analysis", expanded=True):
                    analysis = self.analytics.analyze_station(data, station)
                    
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
                    
                    st.write("**Daily Pattern:**")
                    st.write(f"Peak levels typically at: {analysis['peak_hour']}:00")
                    st.write(f"Lowest levels typically at: {analysis['low_hour']}:00")
                    
                    # Forecast visualization
                    forecast = self.analytics.get_forecast(data, station)
                    recent_data = data[data['location_name'] == station].tail(24)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=recent_data['river_timestamp'],
                        y=recent_data['river_level'],
                        name='Historical',
                        mode='lines+markers'
                    ))
                    
                    forecast_times = pd.date_range(
                        start=recent_data['river_timestamp'].iloc[-1],
                        periods=25,
                        freq='h'
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
                    
    def generate_report(self, data):
        """Generate comprehensive flood monitoring report"""
        st.header("Flood Monitoring Report")
        
        if data is None:
            st.warning("No data available for report generation")
            return

        # Add timestamp
        st.markdown(f"*Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        # Current Status Overview
        st.subheader("Current Status Overview")
        status_df = []
        for station in data['location_name'].unique():
            station_data = data[data['location_name'] == station]
            current_level = station_data['river_level'].iloc[0]
            risk_level, _ = self.predictor.get_risk_level(current_level, station)
            
            status_df.append({
                'Station': station,
                'Current Level': f"{current_level:.3f}m",
                'Risk Level': risk_level,
                'Status': 'Above Threshold' if risk_level in ['MODERATE', 'HIGH'] else 'Normal'
            })
        
        st.dataframe(pd.DataFrame(status_df), hide_index=True)

        # Trend Analysis
        st.subheader("24-Hour Trend Analysis")
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
            title="River Levels - Last 24 Hours",
            xaxis_title="Time",
            yaxis_title="River Level (m)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Risk Assessment Summary
        st.subheader("Risk Assessment Summary")
        risk_cols = st.columns(3)
        with risk_cols[0]:
            high_risk = sum(1 for s in status_df if s['Risk Level'] == 'HIGH')
            st.metric("High Risk Stations", high_risk, 
                     delta="Critical" if high_risk > 0 else "Normal")
        
        with risk_cols[1]:
            moderate_risk = sum(1 for s in status_df if s['Risk Level'] == 'MODERATE')
            st.metric("Moderate Risk Stations", moderate_risk,
                     delta="Warning" if moderate_risk > 0 else "Normal")
        
        with risk_cols[2]:
            low_risk = sum(1 for s in status_df if s['Risk Level'] == 'LOW')
            st.metric("Low Risk Stations", low_risk,
                     delta="Normal" if low_risk == len(status_df) else "Some Risks Present")

        # Recent Alerts
        st.subheader("Recent Alert History")
        recent_alerts = self.alert_system.get_recent_alerts(days=1)
        if not recent_alerts.empty:
            recent_alerts['timestamp'] = pd.to_datetime(recent_alerts['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_alerts)
        else:
            st.info("No alerts in the past 24 hours")

        # Export Options
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Data as CSV"):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"flood_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv'
                )
        
        with col2:
            if st.button("📄 Export Summary Report"):
                # Create summary report
                report = f"""Flood Monitoring Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Current Status:
    {"="*50}"""
                for station in status_df:
                    report += f"\n{station['Station']}:"
                    report += f"\n- Current Level: {station['Current Level']}"
                    report += f"\n- Risk Level: {station['Risk Level']}"
                    report += f"\n- Status: {station['Status']}\n"
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"flood_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime='text/plain'
                )
    def generate_report(self, data):
        """Generate flood monitoring report"""
        st.header("Flood Monitoring Report")
        
        if data is None:
            st.warning("No data available for report generation")
            return

        # Add timestamp
        st.markdown(f"*Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        # Current Status Overview
        st.subheader("Current Status Overview")
        status_df = []
        for station in data['location_name'].unique():
            station_data = data[data['location_name'] == station]
            current_level = station_data['river_level'].iloc[0]
            risk_level, _ = self.predictor.get_risk_level(current_level, station)
            
            status_df.append({
                'Station': station,
                'Current Level': f"{current_level:.3f}m",
                'Risk Level': risk_level,
                'Status': 'Above Threshold' if risk_level in ['MODERATE', 'HIGH'] else 'Normal'
            })
        
        st.dataframe(pd.DataFrame(status_df), hide_index=True)

        # Trend Analysis
        st.subheader("24-Hour Trend Analysis")
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
            title="River Levels - Last 24 Hours",
            xaxis_title="Time",
            yaxis_title="River Level (m)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Risk Assessment Summary
        st.subheader("Risk Assessment Summary")
        risk_cols = st.columns(3)
        with risk_cols[0]:
            high_risk = sum(1 for s in status_df if s['Risk Level'] == 'HIGH')
            st.metric("High Risk Stations", high_risk, 
                     delta="Normal" if high_risk == 0 else "Critical")
        
        with risk_cols[1]:
            moderate_risk = sum(1 for s in status_df if s['Risk Level'] == 'MODERATE')
            st.metric("Moderate Risk Stations", moderate_risk,
                     delta="Normal" if moderate_risk == 0 else "Warning")
        
        with risk_cols[2]:
            low_risk = sum(1 for s in status_df if s['Risk Level'] == 'LOW')
            st.metric("Low Risk Stations", low_risk,
                     delta="Normal" if low_risk == len(status_df) else "Some Risks Present")

        # Recent Alerts
        st.subheader("Recent Alert History")
        recent_alerts = self.alert_system.get_recent_alerts(days=1)
        if not recent_alerts.empty:
            st.dataframe(recent_alerts)
        else:
            st.info("No alerts in the past 24 hours")

        # Export Options
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Raw Data"):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"flood_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv'
                )
        
        with col2:
            if st.button("📑 Export Summary Report"):
                summary = f"""Flood Monitoring Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Current Status:
    {'-'*50}"""
                for station in status_df:
                    summary += f"\n{station['Station']}:"
                    summary += f"\n- Current Level: {station['Current Level']}"
                    summary += f"\n- Risk Level: {station['Risk Level']}"
                    summary += f"\n- Status: {station['Status']}\n"
                
                st.download_button(
                    label="Download Report",
                    data=summary,
                    file_name=f"flood_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime='text/plain'
                )

    def show_mobile_dashboard(self, data):
        """
        Render mobile-specific dashboard view
        
        Args:
            data (pd.DataFrame): Flood monitoring data
        """
        st.header("📱 Mobile Quick View")
        
        if data is None:
            st.warning("No data available")
            return
        
        # Track overall system risk
        system_risks = []
        
        # Quick status overview
        for station in data['location_name'].unique():
            station_data = data[data['location_name'] == station]
            current_level = station_data['river_level'].iloc[0]
            
            # Use precise thresholds for risk determination
            thresholds = self.predictor.thresholds[station]
            
            # Calculate proximity to thresholds
            distance_to_warning = thresholds['warning'] - current_level
            total_threshold_range = thresholds['critical'] - thresholds['warning']
            proximity_percentage = max(0, 1 - (distance_to_warning / total_threshold_range)) * 100
            
            # Precise risk categorization with proximity to thresholds
            if current_level >= thresholds['critical']:
                risk_level = "HIGH RISK"
                risk_color = "red"
                icon = "🚨"
            elif current_level >= thresholds['alert']:
                risk_level = "HIGH RISK"
                risk_color = "red"
                icon = "🚨"
            elif current_level >= thresholds['warning']:
                risk_level = "MODERATE RISK"
                risk_color = "orange"
                icon = "⚠️"
            elif proximity_percentage > 50:  # Closer to warning threshold
                risk_level = "CAUTION"
                risk_color = "yellow"
                icon = "⚠️"
            else:
                risk_level = "LOW RISK"
                risk_color = "green"
                icon = "ℹ️"
            
            # Track system risk for overall assessment
            system_risks.append(risk_level)
            
            # Mobile-friendly card
            st.markdown(f"""
            <div style='
                background-color: {risk_color}15; 
                border: 2px solid {risk_color}; 
                border-radius: 10px; 
                padding: 15px; 
                margin-bottom: 10px;
            '>
                <h3 style='margin: 0; color: {risk_color};'>
                    {icon} {station}
                </h3>
                <div style='display: flex; justify-content: space-between; margin-top: 10px;'>
                    <p><strong>Current Level:</strong></p>
                    <p>{current_level:.3f}m</p>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <p><strong>Risk Level:</strong></p>
                    <p style='color: {risk_color}; font-weight: bold;'>{risk_level}</p>
                </div>
                <div style='margin-top: 10px; font-size: 0.8em; color: #666;'>
                    <p><strong>Proximity Details:</strong></p>
                    <p>Warning: {thresholds['warning']:.3f}m | Alert: {thresholds['alert']:.3f}m | Critical: {thresholds['critical']:.3f}m</p>
                    <p>Proximity to Warning: {proximity_percentage:.1f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # River Level Trends
        st.subheader("River Level Trends")
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
            height=350,  # Smaller height for mobile
            margin=dict(l=40, r=20, t=40, b=40),  # Tight margins
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Emergency Guidance Section
        st.subheader("Emergency Guidance")
        
        # Determine overall system risk
        risk_priority = {
            "HIGH RISK": 3,
            "MODERATE RISK": 2,
            "CAUTION": 1,
            "LOW RISK": 0
        }
        
        overall_risk = max(system_risks, key=lambda x: risk_priority[x])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Low Risk Areas
            - Stay informed
            - Monitor local news
            - Prepare emergency kit
            - Check local flood warnings
            """)
        
        with col2:
            st.markdown("""
            ### High Risk Areas
            - Prepare for immediate evacuation
            - Move to higher ground
            - Follow official guidance
            - Protect critical belongings
            """)
        
        # Conditional Emergency Information
        if overall_risk == "HIGH RISK":
            st.error("""
            ### 🚨 CRITICAL ALERT
            High-risk conditions detected. 
            Immediate action may be required.
            """)
        elif overall_risk == "MODERATE RISK":
            st.warning("""
            ### ⚠️ FLOOD WARNING
            Moderate risk conditions present. 
            Stay alert and prepared.
            """)
        elif overall_risk == "CAUTION":
            st.warning("""
            ### ⚠️ CAUTION LEVEL
            Potential flood risk detected.
            Monitor conditions closely.
            """)
        
        # Emergency Contacts
        st.markdown("""
        ### Emergency Contacts
        - **Emergency Services:** 999
        - **Flood Helpline:** 0345 988 1188
        - **Local Council:** Contact your local authority
        """)
   
def completely_hide_sidebar():
    """Hide the sidebar completely using CSS"""
    st.markdown("""
    <style>
        /* Hide the sidebar completely */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Ensure main content takes full width */
        .main .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
   
def main():
    # Page configuration
    st.set_page_config(
        page_title="Flood Monitoring Dashboard",
        layout="wide"
    )
    
    # Apply the function to completely hide the sidebar
    completely_hide_sidebar()
    
    # Display title
    st.title("Comprehensive Flood Monitoring Dashboard")
    
    # Add critical controls to the main area in a collapsible section if needed
    with st.expander("Dashboard Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            use_real_data = st.checkbox("Use real-time data", value=True)
            days_back = st.slider("Days of data to show", 1, 30, 7)
        with col2:
            all_stations = list(STATION_CONFIG.keys())
            selected_stations = st.multiselect(
                "Select stations to display",
                all_stations,
                default=all_stations
            )
    
    # Initialize dashboard
    dashboard = FloodMonitoringDashboard()
    
    # Get data based on selection
    if use_real_data:
        river_data = dashboard.fetch_river_data(days_back=days_back)
    else:
        river_data = dashboard._generate_sample_data(days_back=days_back)

    # Filter data for selected stations if needed
    if selected_stations and len(selected_stations) < len(all_stations):
        river_data = river_data[river_data['location_name'].isin(selected_stations)]

    # Show quick stats at the top
    if river_data is not None:
        stations_count = river_data['location_name'].nunique()
        readings_count = len(river_data)
        latest_update = river_data['river_timestamp'].max().strftime('%Y-%m-%d %H:%M')
        
        cols = st.columns(3)
        cols[0].metric("Monitoring Stations", stations_count)
        cols[1].metric("Total Readings", f"{readings_count:,}")
        cols[2].metric("Latest Update", latest_update)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Real-Time Monitoring",
        "Predictions",
        "Historical Trends",
        "Station Details",
        "Geospatial View",
        "Watershed Analysis",
        "Alerts",
        "Advanced Analytics",
        "Reports",
        "Mobile View"
    ])

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
        dashboard.show_advanced_analytics(river_data)
    with tab9:
        dashboard.generate_report(river_data)
    with tab10:
        dashboard.show_mobile_dashboard(river_data)

    # Optional: Update query parameters
    st.query_params.update(refresh=True)

if __name__ == '__main__':
    main()	