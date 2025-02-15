
import smtplib
from twilio.rest import Client

class NotificationSystem:
    def __init__(self):
        # Email configuration
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'your_flood_alert_email@gmail.com',
            'sender_password': 'your_email_password'
        }
        
        # SMS configuration (Twilio)
        self.sms_config = {
            'account_sid': 'your_twilio_account_sid',
            'auth_token': 'your_twilio_auth_token',
            'twilio_number': 'your_twilio_phone_number'
        }
    
    def send_email(self, recipient, subject, message):
        """Send email notification"""
        try:
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.sendmail(self.email_config['sender_email'], recipient, f"Subject: {subject}\n\n{message}")
            return True
        except Exception as e:
            print(f"Email sending failed: {e}")
            return False
    
    def send_sms(self, recipient_number, message):
        """Send SMS notification"""
        try:
            client = Client(self.sms_config['account_sid'], self.sms_config['auth_token'])
            message = client.messages.create(
                body=message,
                from_=self.sms_config['twilio_number'],
                to=recipient_number
            )
            return True
        except Exception as e:
            print(f"SMS sending failed: {e}")
            return False