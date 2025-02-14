# alert_utils.py
from datetime import datetime
import pandas as pd

class FloodAlertSystem:
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
        self.alert_history = []

    def check_alert_conditions(self, station, current_level, trend=None):
        """Check if alert conditions are met"""
        station_thresholds = self.thresholds[station]
        timestamp = datetime.now()
        
        if current_level > station_thresholds['critical']:
            status = 'CRITICAL'
            message = 'Immediate action required'
        elif current_level > station_thresholds['alert']:
            status = 'ALERT'
            message = 'Prepare for potential flooding'
        elif current_level > station_thresholds['warning']:
            status = 'WARNING'
            message = 'Monitor conditions closely'
        else:
            status = 'NORMAL'
            message = 'Normal conditions'
        
        alert = {
            'timestamp': timestamp,
            'station': station,
            'level': current_level,
            'status': status,
            'message': message,
            'trend': trend
        }
        
        self.alert_history.append(alert)
        return alert