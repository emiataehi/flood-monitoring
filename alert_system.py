
from alert_config import AlertConfiguration
from notification_system import NotificationSystem
from alert_history import AlertHistoryTracker
import pandas as pd
from datetime import datetime

class AlertSystem:
    def __init__(self):
        """Initialize the complete alert system"""
        self.config = AlertConfiguration()
        self.notifier = NotificationSystem()
        self.history = AlertHistoryTracker('flood_alerts.csv')
    
    def process_alert(self, station, river_level):
        """Process and generate alerts based on current river levels"""
        # Get station configuration
        station_config = self.config.get_alert_configuration(station)
        
        # Determine alert type based on thresholds
        alert_type = self._determine_alert_type(river_level, station_config)
        
        if alert_type:  # If alert needed
            # Check if this is a new or changed alert
            latest_alert = self.history.get_latest_alert(station)
            
            if self._should_send_alert(latest_alert, alert_type):
                # Log the alert
                self.history.log_alert(
                    station=station,
                    river_level=river_level,
                    alert_type=alert_type,
                    notification_sent=True
                )
                
                return True, alert_type
        
        return False, None
    
    def _determine_alert_type(self, river_level, station_config):
        """Determine the type of alert based on thresholds"""
        if river_level >= station_config['critical_level']:
            return 'CRITICAL'
        elif river_level >= station_config['alert_level']:
            return 'ALERT'
        elif river_level >= station_config['warning_level']:
            return 'WARNING'
        return None
    
    def _should_send_alert(self, latest_alert, new_alert_type):
        """Determine if a new alert should be sent"""
        if not latest_alert:
            return True
            
        # Send if alert type has changed or if it's been more than 1 hour
        time_since_last = pd.Timestamp.now() - pd.Timestamp(latest_alert['timestamp'])
        return (latest_alert['alert_type'] != new_alert_type or 
                time_since_last.total_seconds() > 3600)
    
    def get_recent_alerts(self, days=7):
        """Get recent alert history"""
        return self.history.get_recent_alerts(days)
    
    def get_station_alerts(self, station):
        """Get alerts for specific station"""
        return self.history.get_station_alerts(station)