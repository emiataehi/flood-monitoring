
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import streamlit as st

class ReportGenerator:
    def __init__(self):
        self.report_types = {
            'incident': self.generate_incident_report,
            'daily': self.generate_daily_report,
            'historical': self.generate_historical_comparison
        }
    
    def generate_incident_report(self, data, station):
        """Generate a report for a specific incident"""
        station_data = data[data['location_name'] == station].copy()
        current_level = station_data['river_level'].iloc[0]
        
        report = {
            'timestamp': datetime.now(),
            'station': station,
            'current_level': current_level,
            'data': station_data.to_dict('records')
        }
        
        return report
    
    def generate_daily_report(self, data):
        """Generate a daily summary report"""
        daily_summary = []
        
        for station in data['location_name'].unique():
            station_data = data[data['location_name'] == station]
            daily_summary.append({
                'station': station,
                'avg_level': station_data['river_level'].mean(),
                'max_level': station_data['river_level'].max(),
                'min_level': station_data['river_level'].min(),
                'readings_count': len(station_data)
            })
        
        return {
            'timestamp': datetime.now(),
            'summary': daily_summary,
            'data': data.to_dict('records')
        }
    
    def generate_historical_comparison(self, data, days_back=30):
        """Generate historical comparison report"""
        comparison = []
        
        for station in data['location_name'].unique():
            station_data = data[data['location_name'] == station]
            current_level = station_data['river_level'].iloc[0]
            historical_avg = station_data['river_level'].mean()
            
            comparison.append({
                'station': station,
                'current_level': current_level,
                'historical_avg': historical_avg,
                'deviation': current_level - historical_avg
            })
        
        return {
            'timestamp': datetime.now(),
            'comparison': comparison,
            'data': data.to_dict('records')
        }
    
    def export_csv(self, data, filename=None):
        """Export data to CSV"""
        if filename is None:
            filename = f"flood_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        return data.to_csv(index=False), filename
    
    def create_visualization(self, data, report_type):
        """Create visualization for report"""
        fig = go.Figure()
        
        for station in data['location_name'].unique():
            station_data = data[data['location_name'] == station]
            fig.add_trace(go.Scatter(
                x=station_data['river_timestamp'],
                y=station_data['river_level'],
                name=station,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title=f"{report_type.title()} Report Visualization",
            xaxis_title="Time",
            yaxis_title="River Level (m)",
            height=400
        )
        
        return fig