import pandas as pd
import numpy as np
import joblib

class FloodPredictor:
    def __init__(self):
        # Try to load the trained models
        try:
            self.models = joblib.load('station_models.joblib')
            print("Models loaded successfully")
        except:
            print("No trained models found")
            self.models = {}

    def prepare_features(self, df):
        """Prepare features for prediction"""
        df = df.copy()
        
        # Time-based features
        df['hour'] = df['river_timestamp'].dt.hour
        df['day_of_week'] = df['river_timestamp'].dt.dayofweek
        df['month'] = df['river_timestamp'].dt.month
        
        # Calculate rolling features
        df['level_6h_mean'] = df['river_level'].rolling(window=24, min_periods=1).mean()
        df['rainfall_6h_sum'] = df['rainfall'].rolling(window=24, min_periods=1).sum()
        
        # Calculate rate of change
        df['level_change'] = df['river_level'].diff().fillna(0)
        
        return df

    def predict_next_level(self, station_name, current_data):
        """Predict next river level"""
        try:
            if station_name not in self.models:
                return None, f"No model available for {station_name}"
            
            # Prepare features
            features_df = self.prepare_features(current_data)
            
            # Get the latest row of features
            latest_features = features_df.iloc[-1:]
            
            # Select required features
            model_features = ['hour', 'day_of_week', 'month', 'level_6h_mean', 
                            'rainfall_6h_sum', 'level_change', 'rainfall']
            X = latest_features[model_features]
            
            # Make prediction
            prediction = self.models[station_name]['model'].predict(X)[0]
            
            return prediction, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

    def get_risk_level(self, predicted_level, station_name):
        """Determine risk level based on predicted level"""
        risk_thresholds = {
            'Rochdale': {'high': 0.3, 'medium': 0.2},
            'Manchester Racecourse': {'high': 1.1, 'medium': 0.9},
            'Bury Ground': {'high': 0.4, 'medium': 0.3}
        }
        
        if station_name not in risk_thresholds:
            return "Unknown"
            
        thresholds = risk_thresholds[station_name]
        
        if predicted_level > thresholds['high']:
            return "High"
        elif predicted_level > thresholds['medium']:
            return "Medium"
        else:
            return "Low"