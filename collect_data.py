import requests
import os
from supabase import create_client
from datetime import datetime

# Get Supabase credentials from environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

def collect_and_store():
    """Fetch latest readings and store in Supabase"""
    
    # Initialize Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Station measure IDs
    stations = {
        'Rochdale': '690203-level-stage-i-15_min-m',
        'Manchester Racecourse': '690510-level-stage-i-15_min-m',
        'Bury Ground': '690160-level-stage-i-15_min-m'
    }
    
    print(f"Starting data collection at {datetime.now()}")
    
    for station_name, measure_id in stations.items():
        url = f"https://environment.data.gov.uk/flood-monitoring/id/measures/{measure_id}/readings?_sorted&_limit=1"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                readings = data.get('items', [])
                
                if readings:
                    reading = readings[0]
                    
                    # Prepare data for insertion
                    record = {
                        'river_timestamp': reading.get('dateTime'),
                        'location_name': station_name,
                        'river_level': reading.get('value', 0),
                        'rainfall': 0,
                        'rainfall_timestamp': reading.get('dateTime')
                    }
                    
                    # Insert into Supabase
                    result = supabase.table('river_data').insert(record).execute()
                    
                    print(f"✓ Stored {station_name}: {reading.get('value')}m at {reading.get('dateTime')}")
                else:
                    print(f"✗ No readings available for {station_name}")
            else:
                print(f"✗ API error for {station_name}: Status {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error collecting {station_name}: {str(e)}")
    
    print(f"Data collection completed at {datetime.now()}")

if __name__ == '__main__':
    collect_and_store()
