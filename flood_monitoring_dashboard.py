import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client
import os
import pytz
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RealTimeDashboard:
    def __init__(self):
        # Explicitly print and validate environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        # Detailed error checking
        if not supabase_url:
            st.error("Supabase URL is missing. Please set the SUPABASE_URL environment variable.")
            raise ValueError("SUPABASE_URL is required")
        
        if not supabase_key:
            st.error("Supabase Key is missing. Please set the SUPABASE_KEY environment variable.")
            raise ValueError("SUPABASE_KEY is required")
        
        # Print for debugging (remove in production)
        st.write(f"Supabase URL: {supabase_url[:10]}...")
        
        try:
            self.supabase = create_client(supabase_url, supabase_key)
        except Exception as e:
            st.error(f"Failed to create Supabase client: {e}")
            raise
    
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

def main():
    st.set_page_config(page_title="Flood Monitoring Dashboard", layout="wide")
    st.title("Comprehensive Flood Monitoring Dashboard")
    
    # Add environment variable debugging
    st.write("Environment Variables:")
    st.write(f"SUPABASE_URL: {os.getenv('SUPABASE_URL', 'Not Set')}")
    
    try:
        # Initialize dashboard
        dashboard = RealTimeDashboard()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Real-Time Monitoring", "Historical Trends", "Station Details"])
        
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
        
        # Update query params
        st.query_params.update(refresh=True)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)

if __name__ == '__main__':
    main()