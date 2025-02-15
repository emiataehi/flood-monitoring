
class AlertSystem:
    def __init__(self):
        """Initialize the complete alert system"""
        self.config = AlertConfiguration()
        self.notifier = NotificationSystem()
        self.history = AlertHistoryTracker()  # Remove filename parameter