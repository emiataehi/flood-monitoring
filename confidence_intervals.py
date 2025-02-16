# confidence_intervals.py
import numpy as np
import pandas as pd
from scipy import stats

class ConfidenceIntervalCalculator:
    def __init__(self):
        """
        Initialize Confidence Interval Calculator
        """
        self.confidence_levels = {
            'low': 0.68,  # ~1 standard deviation
            'medium': 0.95,  # ~2 standard deviations
            'high': 0.99  # ~3 standard deviations
        }
    
    def calculate_interval(self, data, station, confidence_level='medium'):
        """
        Calculate confidence interval for river levels
        
        Args:
        - data: DataFrame with river level data
        - station: Station name
        - confidence_level: Confidence level ('low', 'medium', 'high')
        
        Returns:
        - Dictionary with confidence interval details
        """
        # Filter data for specific station
        station_data = data[data['location_name'] == station]
        
        # Calculate key statistics
        current_level = station_data['river_level'].iloc[0]
        mean_level = station_data['river_level'].mean()
        std_level = station_data['river_level'].std()
        
        # Get z-score based on confidence level
        confidence = self.confidence_levels.get(confidence_level, 0.95)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Calculate interval
        margin_of_error = z_score * std_level
        lower_bound = max(0, current_level - margin_of_error)
        upper_bound = current_level + margin_of_error
        
        return {
            'current_level': current_level,
            'mean_level': mean_level,
            'std_dev': std_level,
            'confidence_level': confidence_level,
            'confidence_percentage': confidence * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'margin_of_error': margin_of_error
        }
    
    def predict_future_levels(self, data, station, hours_ahead=24):
        """
        Predict future river levels with confidence intervals
        
        Args:
        - data: DataFrame with river level data
        - station: Station name
        - hours_ahead: Number of hours to predict ahead
        
        Returns:
        - DataFrame with predicted levels and confidence intervals
        """
        # Filter data for specific station
        station_data = data[data['location_name'] == station]
        
        # Calculate trend and variability
        recent_data = station_data.head(24)
        trend = recent_data['river_level'].diff().mean()
        std_dev = recent_data['river_level'].std()
        current_level = station_data['river_level'].iloc[0]
        
        # Generate predictions
        predictions = []
        for hour in range(hours_ahead):
            # Predicted level with trend
            predicted_level = current_level + (trend * hour)
            
            # Calculate confidence interval
            z_score = stats.norm.ppf(0.95)  # 95% confidence
            margin_of_error = z_score * std_dev
            
            predictions.append({
                'hour': hour,
                'predicted_level': max(0, predicted_level),
                'lower_bound': max(0, predicted_level - margin_of_error),
                'upper_bound': predicted_level + margin_of_error
            })
        
        return pd.DataFrame(predictions)