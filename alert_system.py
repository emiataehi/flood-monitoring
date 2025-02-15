# alert_system.py
from alert_config import AlertConfiguration
from notification_system import NotificationSystem
from alert_history import AlertHistoryTracker
import pandas as pd

class AlertSystem:
    def __init__(self):
        """Initialize the complete alert system"""
        self.config = AlertConfiguration()
        self.notifier = NotificationSystem()
        self.history = AlertHistoryTracker()
    
    def process_alert(self, station, river_level):
        """Process and generate alerts based on current river levels"""
        try:
            # Get station configuration
            station_config = self.config.get_alert_configuration(station)
            
            # Determine alert type
            if river_level >= station_config['critical_level']:
                alert_type = 'CRITICAL'
            elif river_level >= station_config['alert_level']:
                alert_type = 'ALERT'
            elif river_level >= station_config['warning_level']:
                alert_type = 'WARNING'
            else:
                alert_type = None
            
            if alert_type:
                # Log the alert
                self.history.log_alert(
                    station=station,
                    river_level=river_level,
                    alert_type=alert_type,
                    notification_sent=True
                )
                return True, alert_type
            
            return False, None
            
        except Exception as e:
            st.error(f"Error processing alert: {e}")
            return False, None
    
    def get_recent_alerts(self, days=7):
        """Get recent alert history"""
        return self.history.get_recent_alerts(days)