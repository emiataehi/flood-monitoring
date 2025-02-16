# anomaly_detector.py
import numpy as np
import pandas as pd
from scipy import stats

class AnomalyDetector:
    def detect_anomalies(self, data, station, z_threshold=2.5):
        """
        Detect anomalies in river levels for a specific station
        
        Args:
        - data: Full dataset
        - station: Station name
        - z_threshold: Sensitivity of anomaly detection
        
        Returns:
        - DataFrame of anomalies
        """
        # Filter data for specific station
        station_data = data[data['location_name'] == station].copy()
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(station_data['river_level']))
        
        # Flag anomalies
        station_data['z_score'] = z_scores
        anomalies = station_data[z_scores > z_threshold]
        
        # Classify anomaly type
        anomalies['anomaly_type'] = np.where(
            anomalies['river_level'] > station_data['river_level'].mean(), 
            'High Level', 
            'Low Level'
        )
        
        return anomalies