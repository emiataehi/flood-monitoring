# anomaly_detector.py  (note the change)
import numpy as np
import pandas as pd
from scipy import stats

class AnomalyDetector:
    @staticmethod
    def detect_anomalies(data, station, z_threshold=2.5):
        """
        Detect anomalies in river level data
        
        Args:
        - data: Pandas DataFrame with river data
        - station: Station name
        - z_threshold: Z-score threshold for anomaly detection
        
        Returns:
        - DataFrame with anomaly flags
        """
        station_data = data[data['location_name'] == station]
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(station_data['river_level']))
        
        # Flag anomalies
        anomalies = z_scores > z_threshold
        
        station_data['is_anomaly'] = anomalies
        
        return station_data[station_data['is_anomaly']]