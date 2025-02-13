import pandas as pd
import numpy as np

class WatershedAnalysis:
    def __init__(self):
        # Station information with elevation and catchment area
        self.station_info = {
            'Rochdale': {
                'elevation': 150,  # meters above sea level
                'catchment_area': 12.5,  # km²
                'flow_to': 'Manchester Racecourse'
            },
            'Manchester Racecourse': {
                'elevation': 25,
                'catchment_area': 15.3,
                'flow_to': 'Bury Ground'
            },
            'Bury Ground': {
                'elevation': 75,
                'catchment_area': 18.7,
                'flow_to': None
            }
        }
    
    def analyze_flow_impact(self, current_levels, rainfall_data):
        """Analyze how changes might impact downstream stations"""
        impact_analysis = {}
        
        for station in self.station_info:
            # Get current station data
            station_data = {
                'current_level': current_levels.get(station, 0),
                'elevation': self.station_info[station]['elevation'],
                'catchment_area': self.station_info[station]['catchment_area'],
                'rainfall': rainfall_data.get(station, 0)
            }
            
            # Calculate basic risk score (0-100)
            risk_score = self._calculate_risk_score(station_data)
            
            # Get downstream impact
            downstream = self.station_info[station]['flow_to']
            downstream_impact = "None" if not downstream else f"May affect {downstream}"
            
            impact_analysis[station] = {
                'risk_score': risk_score,
                'downstream_impact': downstream_impact,
                'catchment_area': station_data['catchment_area'],
                'elevation': station_data['elevation']
            }
        
        return impact_analysis
    
    def _calculate_risk_score(self, station_data):
        """Calculate risk score based on current conditions"""
        # Normalize factors to 0-1 scale
        level_factor = min(station_data['current_level'] * 2, 1)
        rain_factor = min(station_data['rainfall'] * 0.5, 1)
        area_factor = station_data['catchment_area'] / 20  # Normalize by max area
        
        # Combined weighted score
        risk_score = (level_factor * 0.4 + 
                     rain_factor * 0.3 + 
                     area_factor * 0.3) * 100
        
        return min(100, max(0, risk_score))  # Ensure 0-100 range

    def get_watershed_summary(self):
        """Get a summary of watershed characteristics"""
        return {
            'total_area': sum(s['catchment_area'] for s in self.station_info.values()),
            'elevation_range': {
                'min': min(s['elevation'] for s in self.station_info.values()),
                'max': max(s['elevation'] for s in self.station_info.values())
            },
            'flow_paths': [
                f"{station} → {info['flow_to']}" 
                for station, info in self.station_info.items() 
                if info['flow_to']
            ]
        }