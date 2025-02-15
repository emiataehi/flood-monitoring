
c# alert_config.py
class AlertConfiguration:
    def __init__(self):
        # Predefined alert thresholds with customization options
        self.thresholds = {
            'Rochdale': {
                'warning_level': 0.168,
                'alert_level': 0.169,
                'critical_level': 0.170,
                'custom_contacts': [],
                'notification_channels': ['dashboard', 'email']
            },
            'Manchester Racecourse': {
                'warning_level': 0.938,
                'alert_level': 0.944,
                'critical_level': 0.950,
                'custom_contacts': [],
                'notification_channels': ['dashboard', 'sms']
            },
            'Bury Ground': {
                'warning_level': 0.314,
                'alert_level': 0.317,
                'critical_level': 0.320,
                'custom_contacts': [],
                'notification_channels': ['dashboard']
            }
        }
    
    def update_threshold(self, station, threshold_type, new_value):
        """Update alert threshold for a specific station"""
        if station in self.thresholds and threshold_type in ['warning_level', 'alert_level', 'critical_level']:
            self.thresholds[station][threshold_type] = new_value
            return True
        return False
    
    def add_custom_contact(self, station, contact_info):
        """Add a custom contact for alerts"""
        if station in self.thresholds:
            contact = {
                'name': contact_info.get('name', ''),
                'email': contact_info.get('email', ''),
                'phone': contact_info.get('phone', '')
            }
            self.thresholds[station]['custom_contacts'].append(contact)
            return True
        return False
    
    def get_alert_configuration(self, station):
        """Get current alert configuration for a station"""
        return self.thresholds.get(station, {})

    def get_notification_channels(self, station):
        """Get configured notification channels for a station"""
        station_config = self.thresholds.get(station, {})
        return station_config.get('notification_channels', ['dashboard'])