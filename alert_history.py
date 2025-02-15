# alert_history.py
import pandas as pd
from datetime import datetime
import os

class AlertHistoryTracker:
    def __init__(self):  # Remove history_file parameter
        """Initialize alert history tracking"""
        self.history_file = 'alert_history.csv'  # Set default filename
        
        # Create or load history file
        if not os.path.exists(self.history_file):
            self.alert_history = pd.DataFrame(columns=[
                'timestamp',
                'station',
                'river_level',
                'alert_type',
                'notification_sent',
                'notification_type',
                'recipients'
            ])
            # Save empty DataFrame
            self.alert_history.to_csv(self.history_file, index=False)
        else:
            # Load existing history
            self.alert_history = pd.read_csv(self.history_file)
            # Convert timestamp strings to datetime
            self.alert_history['timestamp'] = pd.to_datetime(self.alert_history['timestamp'])