import folium
import numpy as np

# Station Coordinates (match your existing coordinate definitions)
stations = {
    'Rochdale': {'lat': 53.611067, 'lon': -2.178685},
    'Manchester': {'lat': 53.499526, 'lon': -2.271756},
    'Bury': {'lat': 53.598766, 'lon': -2.305182}
}

def create_station_map(stations=stations):
    """
    Create an interactive map of river monitoring stations
    
    Args:
        stations (dict): Dictionary of station coordinates
    
    Returns:
        folium.Map: Interactive map object
    """
    try:
        # Create a map centered on the mean latitude and longitude
        center_lat = np.mean([coord['lat'] for coord in stations.values()])
        center_lon = np.mean([coord['lon'] for coord in stations.values()])
        
        # Create base map
        river_map = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=10,
            tiles='OpenStreetMap'  # More detailed map tiles
        )
        
        # Color coding for stations
        colors = ['red', 'blue', 'green']
        
        # Add markers for each station with additional information
        for (station, coords), color in zip(stations.items(), colors):
            folium.Marker(
                location=[coords['lat'], coords['lon']],
                popup=f"Station: {station}\nLatitude: {coords['lat']}\nLongitude: {coords['lon']}",
                tooltip=station,
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(river_map)
        
        # Save the map
        river_map.save('station_locations_map.html')
        print("Station map saved as 'station_locations_map.html'")
        
        return river_map
    
    except Exception as e:
        print(f"Error creating station map: {e}")
        return None

# Optional: Function to run if script is executed directly
def generate_station_map():
    map_obj = create_station_map()
    if map_obj:
        print("Station map generation complete.")

# This allows the script to be run independently
if __name__ == "__main__":
    generate_station_map()