# watershed_utils.py

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
    
    def calculate_risk_score(self, station_name, current_level):
        """Calculate risk score for a station"""
        station = self.station_info[station_name]
        
        # Factor in elevation (lower elevation = higher base risk)
        elevation_factor = 1 - (station['elevation'] / 200)  # Normalize to 0-1
        
        # Factor in current water level
        level_factor = current_level * 2  # Simple scaling
        
        # Combine factors (weighted average)
        risk_score = (elevation_factor * 0.6 + level_factor * 0.4) * 100
        return min(100, max(0, risk_score))  # Ensure 0-100 range
    
    def get_flow_path(self, station_name):
        """Get downstream flow path for a station"""
        if self.station_info[station_name]['flow_to']:
            next_station = self.station_info[station_name]['flow_to']
            elevation_diff = (self.station_info[station_name]['elevation'] - 
                            self.station_info[next_station]['elevation'])
            return {
                'next_station': next_station,
                'elevation_diff': elevation_diff
            }
        return None
    
    def get_station_info(self, station_name):
        """Get all information for a station"""
        return {
            'elevation': self.station_info[station_name]['elevation'],
            'catchment_area': self.station_info[station_name]['catchment_area'],
            'flow_to': self.station_info[station_name]['flow_to']
        }