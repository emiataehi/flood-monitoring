# anomaly_detector.py
import numpy as np
import pandas as pd
from scipy import stats

class AnomalyDetector:
    def detect_anomalies(self, data, station, z_threshold=1.5):  # Lowered from 2.5
        """
        Detect anomalies in river levels with more sensitivity
        
        Args:
        - data: Full dataset
        - station: Station name
        - z_threshold: Sensitivity of anomaly detection
        
        Returns:
        - DataFrame of anomalies
        """
        # Filter data for specific station
        station_data = data[data['location_name'] == station].copy()
        
        # Calculate z-scores based on recent data (last 24 hours)
        recent_data = station_data.tail(24)
        z_scores = np.abs(stats.zscore(recent_data['river_level']))
        
        # Flag anomalies
        recent_data['z_score'] = z_scores
        anomalies = recent_data[z_scores > z_threshold]
        
        # Classify anomaly type
        if not anomalies.empty:
            anomalies['anomaly_type'] = np.where(
                anomalies['river_level'] > recent_data['river_level'].mean(), 
                'High Level', 
                'Low Level'
            )
        
        return anomalies