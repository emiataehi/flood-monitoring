import streamlit as st
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
            
            # Initialize Supabase client
            supabase_url = st.secrets["SUPABASE_URL"]
            supabase_key = st.secrets["SUPABASE_KEY"]
            self.supabase = create_client(supabase_url, supabase_key)
            
            # Initialize components
            self.predictor = FloodPredictionSystem()
            self.watershed = WatershedAnalysis()
            self.analytics = AdvancedAnalytics()
            
            # Initialize alert system components
            self.alert_system = AlertSystem()  # Add this line
            self.notification_system = NotificationSystem()
            self.alert_config = AlertConfiguration()
            self.alert_history = AlertHistoryTracker()
            
        except Exception as e:
            st.error(f"Failed to initialize dashboard: {e}")
            self.supabase = None

    def fetch_river_data(self, days_back=30):
        """Fetch river monitoring data with fallback"""
        try:
            # Check if Supabase client is initialized
            if self.supabase is None:
                st.warning("Database connection not available. Using simulated data.")
                return self._generate_sample_data()

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
                st.warning("No data found in database. Using simulated data.")
                return self._generate_sample_data()

        except Exception as e:
            st.error(f"Data retrieval error: {e}")
            st.info("Falling back to simulated data for demonstration.")
            return self._generate_sample_data()

    def _generate_sample_data(self):
        """Generate sample river monitoring data"""
        current_time = datetime.now(pytz.UTC)
        dates = pd.date_range(end=current_time, periods=48, freq='H')
        
        stations = ['Rochdale', 'Manchester Racecourse', 'Bury Ground']
        base_levels = {
            'Rochdale': 0.173,
            'Manchester Racecourse': 0.927,
            'Bury Ground': 0.311
        }
        
        data = []
        for station in stations:
            base_level = base_levels[station]
            for date in dates:
                # Add some random variation to base levels
                variation = np.random.normal(0, 0.002)
                level = max(0, base_level + variation)
                
                data.append({
                    'river_timestamp': date,
                    'location_name': station,
                    'river_level': level,
                    'rainfall': np.random.uniform(0, 0.1),
                    'rainfall_timestamp': date
                })
        
        df = pd.DataFrame(data)
        return df

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
        """Display historical trends tab with dynamic date range"""
        st.header("Historical Data Analysis")
        if data is not None:
            # Date range selection
            st.subheader("Select Date Range")
            col1, col2 = st.columns(2)
            with col1:
                min_date = data['river_timestamp'].min().date()
                max_date = data['river_timestamp'].max().date()
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

            # Filter data based on selected date range
            mask = (data['river_timestamp'].dt.date >= start_date) & (data['river_timestamp'].dt.date <= end_date)
            filtered_data = data[mask]

            # Basic statistics overview
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(filtered_data))
            
            with col2:
                st.metric("Date Range", 
                         f"{filtered_data['river_timestamp'].min().date()} to {filtered_data['river_timestamp'].max().date()}")
            
            with col3:
                st.metric("Stations", len(filtered_data['location_name'].unique()))

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

            # Station summary with filtered data
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
        """Display flood alerts tab with enhanced alert logging"""
        st.header("Flood Alerts and Warnings")

        if data is None:
            st.warning("No data available for generating alerts")
            return

        # Create tabs for current alerts and history
        current_tab, history_tab = st.tabs(["Current Alerts", "Alert History"])
        
        with current_tab:
            alerts_recorded = False
            for station in data['location_name'].unique():
                station_data = data[data['location_name'] == station]
                current_level = station_data['river_level'].iloc[0]
                risk_level, risk_color = self.predictor.get_risk_level(current_level, station)
                
                # Log alert in history
                self.alert_system.history.log_alert(
                    station=station,
                    river_level=current_level,
                    alert_type=risk_level,
                    notification_sent=True
                )
                alerts_recorded = True
                
                # Display alert with appropriate styling
                if risk_level == "HIGH":
                    st.error(f"🚨 **CRITICAL FLOOD RISK** at {station}")
                elif risk_level == "MODERATE":
                    st.warning(f"⚠️ **FLOOD WARNING** at {station}")
                else:
                    st.info(f"ℹ️ **Normal Conditions** at {station}")
                
                st.metric(
                    label=f"{station} Current Level",
                    value=f"{current_level:.3f}m",
                    delta=f"Risk Level: {risk_level}"
                )
        
        with history_tab:
            recent_alerts = self.alert_system.get_recent_alerts(days=7)
            if not recent_alerts.empty:
                st.subheader("Recent Alerts (Last 7 Days)")
                
                # Format timestamps for display
                display_df = recent_alerts.copy()
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['river_level'] = display_df['river_level'].round(3)
                
                st.dataframe(
                    display_df[['timestamp', 'station', 'alert_type', 'river_level']],
                    column_config={
                        "timestamp": "Time",
                        "station": "Station",
                        "alert_type": "Alert Level",
                        "river_level": "River Level (m)"
                    },
                    hide_index=True
                )
            else:
                if alerts_recorded:
                    st.info("New alerts have been recorded. Refresh to see updates.")
                else:
                    st.info("No alerts in the past 7 days")

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Real-Time Monitoring",
        "Predictions",
        "Historical Trends",
        "Station Details",
        "Geospatial View",
        "Watershed Analysis",
        "Alerts",
        "Advanced Analytics",
        "Reports"  # New tab
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
        dashboard.show_advanced_analytics(river_data)
    with tab9:
        dashboard.generate_report(river_data)

    # Optional: Update query parameters
    st.query_params.update(refresh=True)

if __name__ == '__main__':
    main()	