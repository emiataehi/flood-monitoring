# enhanced_alert_system.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class EnhancedAlertSystem:
    def __init__(self):
        """
        Initialize Enhanced Alert System
        
        Key Features:
        - Track multiple alert levels
        - Store alert history
        - Support different notification channels
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
        
        # Notification channels
        self.notification_channels = {
            'email': [],
            'sms': [],
            'mobile_push': []
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
        
        # Create alert record
        alert = {
            'timestamp': datetime.now(),
            'station': station,
            'current_level': current_level,
            'alert_level': alert_level,
            'description': description,
            'notified_channels': []
        }
        
        # Log the alert
        logging.info(f"{station} Alert: {alert_level} - {description}")
        
        # Store in alert history
        self.alert_history.append(alert)
        
        return alert
    
    def send_notifications(self, alert):
        """
        Send notifications through different channels
        
        Args:
        - alert: Alert details dictionary
        
        Returns:
        - List of successful notification channels
        """
        successful_channels = []
        
        # Email Notification
        if self._send_email_alert(alert):
            successful_channels.append('email')
        
        # SMS Notification
        if self._send_sms_alert(alert):
            successful_channels.append('sms')
        
        # Mobile Push Notification
        if self._send_mobile_push_alert(alert):
            successful_channels.append('mobile_push')
        
        # Update alert with notified channels
        alert['notified_channels'] = successful_channels
        
        return successful_channels
    
    def _send_email_alert(self, alert):
        """Simulate email notification"""
        try:
            # Placeholder for actual email sending logic
            print(f"Email Alert Sent: {alert['station']} - {alert['alert_level']}")
            return True
        except Exception as e:
            logging.error(f"Email Alert Failed: {str(e)}")
            return False
    
    def _send_sms_alert(self, alert):
        """Simulate SMS notification"""
        try:
            # Placeholder for actual SMS sending logic
            print(f"SMS Alert Sent: {alert['station']} - {alert['alert_level']}")
            return True
        except Exception as e:
            logging.error(f"SMS Alert Failed: {str(e)}")
            return False
    
    def _send_mobile_push_alert(self, alert):
        """Simulate mobile push notification"""
        try:
            # Placeholder for actual mobile push notification logic
            print(f"Mobile Push Alert Sent: {alert['station']} - {alert['alert_level']}")
            return True
        except Exception as e:
            logging.error(f"Mobile Push Alert Failed: {str(e)}")
            return False
    
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