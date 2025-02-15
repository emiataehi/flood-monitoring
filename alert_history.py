# alert_history.py
import pandas as pd
from datetime import datetime
import os

class AlertHistoryTracker:
    def __init__(self):
        """Initialize alert history tracking"""
        self.history_file = 'alert_history.csv'
        
        # Define columns
        self.columns = [
            'timestamp',
            'station',
            'river_level',
            'alert_type',
            'notification_sent',
            'notification_type',
            'recipients'
        ]
        
        # Create or load history file
        if not os.path.exists(self.history_file):
            self.alert_history = pd.DataFrame(columns=self.columns)
            self.alert_history.to_csv(self.history_file, index=False)
        else:
            self.alert_history = pd.read_csv(self.history_file)
            self.alert_history['timestamp'] = pd.to_datetime(self.alert_history['timestamp'])
    
    def log_alert(self, station, river_level, alert_type, notification_sent=False, 
                  notification_type=None, recipients=None):
        """Log a new alert event"""
        new_alert = pd.DataFrame({
            'timestamp': [datetime.now()],
            'station': [station],
            'river_level': [river_level],
            'alert_type': [alert_type],
            'notification_sent': [notification_sent],
            'notification_type': [notification_type if notification_type else ''],
            'recipients': [recipients if recipients else '']
        })
        
        self.alert_history = pd.concat([self.alert_history, new_alert], ignore_index=True)
        self.alert_history.to_csv(self.history_file, index=False)
        return True
    
    def get_recent_alerts(self, days=7):
        """Get alerts from the last X days"""
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        return self.alert_history[self.alert_history['timestamp'] > cutoff_date]