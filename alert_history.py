
import pandas as pd
from datetime import datetime

class AlertHistoryTracker:
    def __init__(self):
        self.alert_history = pd.DataFrame(columns=[
            'timestamp', 
            'station', 
            'previous_status', 
            'current_status', 
            'river_level', 
            'notification_sent'
        ])
    
    def log_alert(self, station, previous_status, current_status, river_level, notification_sent=False):
        """Log alert event"""
        new_entry = pd.DataFrame({
            'timestamp': [datetime.now()],
            'station': [station],
            'previous_status': [previous_status],
            'current_status': [current_status],
            'river_level': [river_level],
            'notification_sent': [notification_sent]
        })
        self.alert_history = pd.concat([self.alert_history, new_entry], ignore_index=True)
    
    def get_recent_alerts(self, days=7):
        """Retrieve recent alert history"""
        recent_date = datetime.now() - pd.Timedelta(days=days)
        return self.alert_history[self.alert_history['timestamp'] > recent_date]
    
    def export_history(self, filename='alert_history.csv'):
        """Export alert history to CSV"""
        self.alert_history.to_csv(filename, index=False)