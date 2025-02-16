# enhanced_alert_system.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import random

class EnhancedAlertSystem:
    def __init__(self):
        """
        Initialize Enhanced Alert System with more realistic notification handling
        """
        # Alert thresholds for each station
        self.alert_thresholds = {
            'Bury Ground': {
                'low_risk': 0.300,    # Normal monitoring
                'warning': 0.314,     # First alert level
                'high_risk': 0.317,   # Serious concern
                'critical': 0.320     # Immediate action required
            },
            'Manchester Racecourse': {
                'low_risk': 0.900,
                'warning': 0.938,
                'high_risk': 0.944,
                'critical': 0.950
            },
            'Rochdale': {
                'low_risk': 0.160,
                'warning': 0.168,
                'high_risk': 0.169,
                'critical': 0.170
            }
        }
        
        # Alert history storage
        self.alert_history = []
        
        # Notification contacts (simulated)
        self.notification_contacts = {
            'email': ['admin@floodmonitoring.com', 'emergency@localcouncil.gov'],
            'sms': ['+441234567890', '+447890123456'],
            'mobile_push': ['device1', 'device2']
        }
        
        # Logging setup
        logging.basicConfig(
            filename='flood_alerts.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
    
    def determine_alert_level(self, station, current_level):
        """
        Determine the current alert level for a station
        
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
            return 'WARNING', 'Potential flood risk, stay alert'
        elif current_level >= thresholds.get('low_risk', float('inf')):
            return 'MONITOR', 'Elevated water levels, monitor closely'
        else:
            return 'NORMAL', 'No immediate flood risk'
    
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
        Selectively send notifications based on alert level
        
        Args:
        - alert_level: Current alert level
        
        Returns:
        - List of notification channels used
        """
        # Only send notifications for serious alerts
        if alert_level not in ['WARNING', 'HIGH', 'CRITICAL']:
            return []
        
        # Simulate realistic notification selection
        notification_channels = []
        
        # Email notification (70% chance for serious alerts)
        if random.random() < 0.7 and self.notification_contacts['email']:
            notification_channels.append('email')
        
        # SMS notification (50% chance for high-risk alerts)
        if alert_level in ['HIGH', 'CRITICAL'] and random.random() < 0.5 and self.notification_contacts['sms']:
            notification_channels.append('sms')
        
        # Mobile push notification (30% chance)
        if random.random() < 0.3 and self.notification_contacts['mobile_push']:
            notification_channels.append('mobile_push')
        
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