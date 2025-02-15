# notification_system.py
import streamlit as st
from datetime import datetime

class NotificationSystem:
    def __init__(self):
        """Initialize notification system"""
        self.notification_log = []
    
    def send_email(self, recipient, subject, message):
        """Simulate email notification"""
        try:
            # Log the notification
            notification_entry = {
                'type': 'email',
                'recipient': recipient,
                'subject': subject,
                'message': message,
                'timestamp': datetime.now()
            }
            self.notification_log.append(notification_entry)
            
            # Show notification in Streamlit
            st.toast(f"📧 Email notification sent to {recipient}")
            return True
            
        except Exception as e:
            st.error(f"Failed to send email: {e}")
            return False