# alert_system.py
import streamlit as st
import pandas as pd
from datetime import datetime

class AlertSystem:
    def __init__(self):
        """Initialize alert system"""
        self.alert_history = pd.DataFrame(columns=[
            'timestamp',
            'station',
            'river_level',
            'risk_level'
        ])
    
    def process_alert(self, station, river_level):
        """Process new alert"""
        # Determine risk level based on thresholds
        if river_level > 0.9:
            risk_level = "HIGH"
        elif river_level > 0.5:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        # Log alert
        new_alert = pd.DataFrame({
            'timestamp': [datetime.now()],
            'station': [station],
            'river_level': [river_level],
            'risk_level': [risk_level]
        })
        
        self.alert_history = pd.concat([self.alert_history, new_alert], ignore_index=True)
        return True, risk_level
    
    def get_recent_alerts(self, days=7):
        """Get recent alerts"""
        if self.alert_history.empty:
            return pd.DataFrame()
            
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        return self.alert_history[
            self.alert_history['timestamp'] > cutoff_date
        ].sort_values('timestamp', ascending=False)