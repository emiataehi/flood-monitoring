# enhanced_alert_system.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import random

class EnhancedAlertSystem:
    def __init__(self):
        """
        Initialize Enhanced Alert System with more sensitive thresholds
        """
        # More granular and sensitive alert thresholds
        self.alert_thresholds = {
            'Bury Ground': {
                'normal': 0.290,      # Baseline normal level
                'low_risk': 0.300,    # Start of mild concern
                'warning': 0.310,     # Increased monitoring
                'high_risk': 0.315,   # Serious concern
                'critical': 0.320     # Immediate action required
            },
            'Manchester Racecourse': {
                'normal': 0.890,
                'low_risk': 0.900,
                'warning': 0.930,
                'high_risk': 0.940,
                'critical': 0.950
            },
            'Rochdale': {
                'normal': 0.150,
                'low_risk': 0.160,
                'warning': 0.165,
                'high_risk': 0.168,
                'critical': 0.170
            }
        }
        
        # Alert history storage
        self.alert_history = []
        
        # Simulated notification contacts
        self.notification_contacts = {
            'email': [
                'emergency_services@localcouncil.gov',
                'flood_response@citycouncil.org'
            ],
            'sms': [
                '+441234567890',
                '+447890123456'
            ],
            'mobile_push': [
                'emergency_app_admin',
                'flood_alert_system'
            ]
        }
        
        # Logging setup
        logging.basicConfig(
            filename='flood_alerts.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
    
    def determine_alert_level(self, station, current_level):
        """
        Determine the current alert level for a station with more nuanced categorization
        
        Args:
        - station: Name of the monitoring station
        - current_level: Current river level
        
        Returns:
        - Alert level and description
        """
        thresholds = self.alert_thresholds.get(station, {})
        
        if current_level >= thresholds.get('critical', float('inf')):
            return 'CRITICAL', 'Immediate evacuation required'
        elif current_level >= thresholds.get('high_risk', float('inf')):
            return 'HIGH', 'Severe flood risk, prepare for evacuation'
        elif current_level >= thresholds.get('warning', float('inf')):
            return 'WARNING', 'Elevated flood risk, increased monitoring needed'
        elif current_level >= thresholds.get('low_risk', float('inf')):
            return 'MODERATE', 'Water levels above normal, stay alert'
        elif current_level >= thresholds.get('normal', float('inf')):
            return 'LOW', 'Slightly elevated water levels'
        else:
            return 'NORMAL', 'Water levels within expected range'
    
    def generate_alert(self, station, current_level):
        """
        Generate a comprehensive alert for a station
        
        Args:
        - station: Name of the monitoring station
        - current_level: Current river level
        
        Returns:
        - Alert details dictionary
        """
        # Determine alert level
        alert_level, description = self.determine_alert_level(station, current_level)
        
        # Decide whether to send notifications based on alert level
        send_notifications = alert_level in ['WARNING', 'HIGH', 'CRITICAL']
        
        # Create alert record
        alert = {
            'timestamp': datetime.now(),
            'station': station,
            'current_level': current_level,
            'alert_level': alert_level,
            'description': description,
            'notified_channels': self.send_notifications(alert_level) if send_notifications else []
        }
        
        # Log the alert
        logging.info(f"{station} Alert: {alert_level} - {description}")
        
        # Store in alert history
        self.alert_history.append(alert)
        
        return alert
    
    def send_notifications(self, alert_level):
        """
        Send notifications for serious alert levels
        
        Args:
        - alert_level: Current alert level
        
        Returns:
        - List of notification channels used
        """
        notification_channels = []
        
        # More aggressive notification for serious alerts
        if alert_level == 'WARNING':
            # Lower probability for warning level
            if random.random() < 0.4:
                notification_channels.extend(['email'])
        
        elif alert_level == 'HIGH':
            # Higher probability for high-risk alerts
            notification_channels.extend(['email', 'sms'])
        
        elif alert_level == 'CRITICAL':
            # Maximum notification for critical alerts
            notification_channels.extend(['email', 'sms', 'mobile_push'])
        
        return notification_channels
    
    def get_recent_alerts(self, days=7):
        """
        Retrieve recent alerts within specified days
        
        Args:
        - days: Number of days to retrieve alerts for
        
        Returns:
        - DataFrame of recent alerts
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] >= cutoff_date
        ]
        
        return pd.DataFrame(recent_alerts)