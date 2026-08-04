import requests
import math
import psycopg2
import os

conn = psycopg2.connect(
    dbname='mydb',
    user=os.environ.get('USER'),
    password=os.environ.get('PASSWORD'),
    host=os.environ.get('DB_HOST', 'localhost'),
    port=os.environ.get('DB_PORT', '4321')
)

cursor = conn.cursor()

def get_tms_coordinates(sensors):
    url = "https://tie.digitraffic.fi/api/tms/v1/stations"
    headers = {"Digitraffic-User": "CityBikePredictor_UniversityOfHelsinki_Student/1.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    tms_stations = {}
    features = response.json().get("features", [])
    
    for feature in features:
        station_id = feature["properties"]["id"]
        if station_id in sensors:
            lon, lat = feature["geometry"]["coordinates"][:2]
            tms_stations[station_id] = {"lat": lat, "lon": lon}
        
    return tms_stations

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the distance in meters between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0)**2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def map_nearest_sensors(bike_stations, tms_stations):
    """Maps each bike station to its 3 nearest TMS sensors."""
    mapping = {}
    
    for bike_id, bike_coords in bike_stations.items():
        distances = []
        for tms_id, tms_coords in tms_stations.items():
            dist = haversine(
                bike_coords["lat"], bike_coords["lon"],
                tms_coords["lat"], tms_coords["lon"]
            )
            distances.append((tms_id, dist))

        distances.sort(key=lambda x: x[1])
        nearest_3 = [x[0] for x in distances[:3]]
        
        mapping[bike_id] = nearest_3
        
    return mapping

def backfill():
    bike_query = """
    SELECT station_id AS id, lat, lon
    FROM bike_stations
    """

    bike_result = cursor.execute(bike_query)
    bike_result = cursor.fetchall()
    bike_stations = {}
    for row in bike_result:
        bike_stations[row[0]] = {'lat': row[1], 'lon': row[2]}

    distinct_sensors = set()
    sensor_result = cursor.execute("SELECT DISTINCT sensor_id FROM traffic_readings")
    sensor_result = cursor.fetchall()
    for row in sensor_result:
        distinct_sensors.add(row[0])

    print(f'Bike stations: {len(list(bike_stations.keys()))}')
    tms_stations = get_tms_coordinates(distinct_sensors)
    print(f'tms_stations: {len(list(tms_stations.keys()))}')

    mapping = map_nearest_sensors(bike_stations, tms_stations)
    print(f'mapping length: {len(list(mapping.keys()))}')

    for id, nearest_3 in mapping.items():
        cursor.execute("""
        INSERT INTO station_mapping(bike_station_id, tms_sensor_1, tms_sensor_2, tms_sensor_3) VALUES(%s, %s, %s, %s)
        """, (id, nearest_3[0], nearest_3[1], nearest_3[2]))

    conn.commit()

if __name__ == '__main__':
    print('Starting backfill')
    backfill()
    print('Backfill complete!')
    
