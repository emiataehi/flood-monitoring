# advanced_analytics.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AdvancedAnalytics:
    def __init__(self):
        self.baseline_levels = {
            'Rochdale': 0.168,
            'Manchester Racecourse': 0.928,
            'Bury Ground': 0.311
        }
    
    def analyze_station(self, data, station_name):
        """Detailed analysis for a single station"""
        station_data = data[data['location_name'] == station_name].copy()
        station_data = station_data.sort_values('river_timestamp')
        
        # Basic statistics
        current_level = station_data['river_level'].iloc[-1]
        avg_level = station_data['river_level'].mean()
        baseline = self.baseline_levels[station_name]
        deviation = current_level - baseline
        
        # Trend analysis (last 24 hours)
        recent_data = station_data.tail(24)
        trend = recent_data['river_level'].diff().mean()
        
        if abs(trend) < 0.0001:
            trend_direction = "Stable"
        elif trend > 0:
            trend_direction = "Rising"
        else:
            trend_direction = "Falling"
        
        # Pattern detection
        hourly_pattern = station_data.groupby(
            station_data['river_timestamp'].dt.hour
        )['river_level'].mean()
        
        peak_hour = hourly_pattern.idxmax()
        low_hour = hourly_pattern.idxmin()
        
        return {
            'current_level': current_level,
            'average_level': avg_level,
            'baseline': baseline,
            'deviation': deviation,
            'trend': trend_direction,
            'trend_rate': trend,
            'peak_hour': peak_hour,
            'low_hour': low_hour,
            'hourly_pattern': hourly_pattern.to_dict()
        }
    
    def get_forecast(self, data, station_name, hours_ahead=24):
        """Generate forecast for a station"""
        station_data = data[data['location_name'] == station_name].copy()
        current_level = station_data['river_level'].iloc[-1]
        trend = station_data['river_level'].diff().mean()
        
        forecast = []
        for hour in range(hours_ahead):
            predicted_level = current_level + (trend * hour)
            # Ensure prediction doesn't go below 0
            predicted_level = max(0, predicted_level)
            forecast.append(predicted_level)
        
        return {
            'levels': forecast,
            'confidence': 'High' if abs(trend) < 0.001 else 'Medium'
        }