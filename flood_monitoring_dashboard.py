import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import os
import pytz
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objs as go

# Enhanced Prediction System
class EnhancedFloodPredictor:
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
        
        # Station relationships from our analysis
        self.correlations = {
            ('Manchester Racecourse', 'Bury Ground'): {'correlation': 0.916, 'lag': 0},
            ('Rochdale', 'Manchester Racecourse'): {'correlation': -0.643, 'lag': 10},
            ('Rochdale', 'Bury Ground'): {'correlation': -0.686, 'lag': 12}
        }

    def analyze_station(self, station_name, current_data):
        """Analyze station data and provide predictions"""
        if len(current_data) < 2:
            return {
                'status': 'insufficient_data',
                'risk_level': 'unknown'
            }

        current_level = current_data['river_level'].iloc[-1]
        level_change = current_data['river_level'].diff().iloc[-1]
        
        # Determine risk level
        thresholds = self.thresholds[station_name]
        if current_level > thresholds['critical']:
            risk_level = 'CRITICAL'
            risk_color = 'red'
        elif current_level > thresholds['alert']:
            risk_level = 'HIGH'
            risk_color = 'orange'
        elif current_level > thresholds['warning']:
            risk_level = 'MODERATE'
            risk_color = 'yellow'
        else:
            risk_level = 'LOW'
            risk_color = 'green'

        return {
            'status': 'ok',
            'current_level': current_level,
            'level_change': level_change,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'threshold_warning': thresholds['warning'],
            'threshold_alert': thresholds['alert'],
            'threshold_critical': thresholds['critical']
        }

# Global Station Configuration
STATION_CONFIG = {
    'Rochdale': {
        'full_name': 'Rochdale River Monitoring Station',
        'latitude': 53.611067,
        'longitude': -2.178685,
        'river': 'River Roch',
        'description': 'Monitoring station in northern Greater Manchester'
    },
    'Manchester Racecourse': {
        'full_name': 'Manchester Racecourse River Station',
        'latitude': 53.499526,
        'longitude': -2.271756,
        'river': 'River Irwell',
        'description': 'Central Manchester river monitoring location'
    },
    'Bury Ground': {
        'full_name': 'Bury Ground River Monitoring Point',
        'latitude': 53.598766,
        'longitude': -2.305182,
        'river': 'River Irwell',
        'description': 'Monitoring station in Bury metropolitan area'
    }
}

class FloodMonitoringDashboard:
    def __init__(self):
        """Initialize dashboard"""
        try:
            supabase_url = st.secrets["SUPABASE_URL"]
            supabase_key = st.secrets["SUPABASE_KEY"]
            self.supabase = create_client(supabase_url, supabase_key)
            self.predictor = EnhancedFloodPredictor()
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
                .execute()

            if response.data:
                df = pd.DataFrame(response.data)
                df['river_timestamp'] = pd.to_datetime(df['river_timestamp'])
                return df
            return None
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    def show_predictions_tab(self, data):
        """Display enhanced predictions tab"""
        st.header("Enhanced Flood Predictions")
        
        if data is not None:
            # Create three columns for station predictions
            cols = st.columns(3)
            
            for i, station in enumerate(STATION_CONFIG.keys()):
                with cols[i]:
                    station_data = data[data['location_name'] == station].copy()
                    analysis = self.predictor.analyze_station(station, station_data)
                    
                    if analysis['status'] == 'ok':
                        # Create expander for detailed information
                        with st.expander(f"{station} Prediction Details", expanded=True):
                            # Current Level and Risk
                            st.metric(
                                "Current Level",
                                f"{analysis['current_level']:.3f}m",
                                f"Change: {analysis['level_change']:.4f}m"
                            )
                            
                            # Risk Level Indicator
                            st.markdown(
                                f"<div style='padding: 10px; background-color: {analysis['risk_color']}; "
                                f"color: black; border-radius: 5px; text-align: center;'>"
                                f"Risk Level: {analysis['risk_level']}</div>",
                                unsafe_allow_html=True
                            )
                            
                            # Thresholds
                            st.write("**Warning Thresholds:**")
                            st.write(f"- Warning: {analysis['threshold_warning']:.3f}m")
                            st.write(f"- Alert: {analysis['threshold_alert']:.3f}m")
                            st.write(f"- Critical: {analysis['threshold_critical']:.3f}m")
            
            # Add trend visualization
            st.subheader("Recent Trends")
            fig = go.Figure()
            
            for station in STATION_CONFIG.keys():
                station_data = data[data['location_name'] == station].copy()
                fig.add_trace(go.Scatter(
                    x=station_data['river_timestamp'],
                    y=station_data['river_level'],
                    name=f"{station} Level",
                    mode='lines+markers'
                ))
                
                # Add threshold lines
                thresholds = self.predictor.thresholds[station]
                for level, value in thresholds.items():
                    fig.add_hline(
                        y=value,
                        line_dash="dash",
                        line_color="red" if level == 'critical' else "orange" if level == 'alert' else "yellow",
                        annotation_text=f"{station} {level}",
                        opacity=0.3
                    )
            
            fig.update_layout(
                title="River Levels with Warning Thresholds",
                xaxis_title="Time",
                yaxis_title="River Level (m)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Page configuration
    st.set_page_config(page_title="Flood Monitoring Dashboard", layout="wide")
    st.title("Enhanced Flood Monitoring Dashboard")

    # Initialize dashboard
    dashboard = FloodMonitoringDashboard()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Current Status", "Enhanced Predictions"])
    
    # Fetch data
    data = dashboard.fetch_river_data(days_back=7)  # Last 7 days of data
    
    if data is not None:
        # Current Status Tab
        with tab1:
            st.header("Current River Levels")
            cols = st.columns(3)
            
            for i, station in enumerate(STATION_CONFIG.keys()):
                with cols[i]:
                    station_data = data[data['location_name'] == station]
                    if not station_data.empty:
                        current_level = station_data['river_level'].iloc[0]
                        st.metric(
                            station,
                            f"{current_level:.3f}m",
                            f"Latest Reading"
                        )
        
        # Enhanced Predictions Tab
        with tab2:
            dashboard.show_predictions_tab(data)
    else:
        st.error("Unable to fetch monitoring data")

if __name__ == "__main__":
    main()