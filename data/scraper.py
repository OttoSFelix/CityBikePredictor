import requests
import psycopg2
import xml.etree.ElementTree as ET
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
import os
import json


DIGITRANSIT_API_KEY = os.environ.get('API_KEY')
if os.path.exists('.env'):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Error loading .env file")

conn = psycopg2.connect(
    dbname='mydb',
    user=os.environ.get('USER'),
    password=os.environ.get('PASSWORD'),
    host=os.environ.get('DB_HOST', 'localhost'),
    port=os.environ.get('DB_PORT', '4321')
)

cursor = conn.cursor()

def initialize_database():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bike_stations (
    station_id VARCHAR(50) PRIMARY KEY UNIQUE,
    name VARCHAR(100) UNIQUE,
    lat DOUBLE PRECISION UNIQUE,
    lon DOUBLE PRECISION UNIQUE
);
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bike_readings (
    timestamp TIMESTAMP NOT NULL,
    station_id VARCHAR(50) REFERENCES bike_stations(station_id),
    bikes INTEGER NOT NULL,
    PRIMARY KEY (timestamp, station_id)
);
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS traffic_readings (
    timestamp TIMESTAMP NOT NULL,
    sensor_id INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (timestamp, sensor_id)
);
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weather_readings (
    timestamp TIMESTAMP PRIMARY KEY,
    rainfall_mm REAL NOT NULL
);
    """)

    conn.commit()


def fetch_bike_stations():
    url = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"
    headers = {
        "Content-Type": "application/json",
        "Digitransit-Subscription-Key": DIGITRANSIT_API_KEY
    }
    query = """
    {
      vehicleRentalStations {
        stationId
        name
        lat
        lon
      }
    }
    """
    response = requests.post(url, json={"query": query}, headers=headers, timeout=10)
    response.raise_for_status()

    stations = response.json()["data"]["vehicleRentalStations"]
    cleaned_stations = []

    for s in stations:


        cleaned_stations.append({
            "station_id": s["stationId"],
            "name": s["name"],
            "lat": s['lat'],
            'lon': s['lon']
        })

    for station in cleaned_stations:
        cursor.execute("""
        INSERT INTO bike_stations(station_id, name, lat, lon)
        VALUES(%s, %s, %s, %s) ON CONFLICT (station_id) DO NOTHING;
        """, (station['station_id'], station['name'], station['lat'], station['lon']))

    conn.commit()


def fetch_city_bikes():
    """Fetches live station status from Digitransit GraphQL API."""
    url = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"
    headers = {
        "Content-Type": "application/json",
        "Digitransit-Subscription-Key": DIGITRANSIT_API_KEY
    }
    query = """
    {
      vehicleRentalStations {
        stationId
        availableVehicles { byType { count } }
      }
    }
    """
    response = requests.post(url, json={"query": query}, headers=headers, timeout=10)
    response.raise_for_status()

    stations = response.json()["data"]["vehicleRentalStations"]
    cleaned_stations = []

    for s in stations:
        bikes_info = s.get('availableVehicles')
        if bikes_info is not None:
            by_type = bikes_info.get('byType')
            if isinstance(by_type, list):
                total_bikes = sum(item.get('count', 0) for item in by_type if isinstance(item, dict))
            else:
                total_bikes = 0
            cleaned_stations.append({
                "station_id": s["stationId"],
                "bikes": total_bikes,
            })

    return cleaned_stations

def fetch_traffic():
    """Fetches live traffic counts from Digitraffic TMS API."""
    url = "https://tie.digitraffic.fi/api/tms/v1/stations/data"
    headers = {"Digitraffic-User": "CityBikePredictor_UniversityOfHelsinki_Student/1.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    stations = response.json().get("stations", [])
    helsinki_traffic = []

    for station in stations:
        station_id = station.get("id")

        for sensor in station.get("sensorValues", []):
            if sensor['id'] == 5016:
                helsinki_traffic.append({
                    "station_id": station_id,
                    "volume": sensor.get("value", 0)
                })
                break

    return helsinki_traffic

def fetch_rainfall():
    """Fetches rainfall in Helsinki over the last hour using FMI WFS."""
    url = (
        "https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0"
        "&request=getFeature&storedquery_id=fmi::observations::weather::simple"
        "&place=helsinki&parameters=ri_10min"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    namespaces = {'BsWfs': 'http://xml.fmi.fi/schema/wfs/2.0'}

    values = root.findall('.//BsWfs:ParameterValue', namespaces)
    if values:
        try:
            intensity_mmh = float(values[-1].text)
            amount_10m = intensity_mmh / 6.0
            return round(amount_10m, 2)
        except ValueError:
            return 0.0
    return 0.0

def poll_and_store():
    timestamp = datetime.now()
    print(f"\n[{timestamp.strftime('%H:%M:%S')}] Fetching snapshot...")

    try:
        bikes = fetch_city_bikes()
        traffic = fetch_traffic()
        rainfall_mm = fetch_rainfall()

        print(f"Success: Fetched {len(bikes)} bike stations, {len(traffic)} traffic cameras, {rainfall_mm}mm rain.")

        for bike in bikes:
            cursor.execute("""
            INSERT INTO bike_readings(timestamp, station_id, bikes)
            VALUES(%s, %s, %s)""", (timestamp, bike['station_id'], bike['bikes']))

        for t in traffic:
            cursor.execute("""
            INSERT INTO traffic_readings(timestamp, sensor_id, volume)
            VALUES(%s, %s, %s)""", (timestamp, t['station_id'], t['volume']))

        cursor.execute("""
        INSERT INTO weather_readings(timestamp, rainfall_mm)
        VALUES(%s, %s)""", (timestamp, rainfall_mm))

        conn.commit()

    except Exception as e:
        print(f"Error during polling: {e}")
        exit()

if __name__ == "__main__":
    print('Initializing database...')
    initialize_database()
    fetch_bike_stations()

    print("Initializing API Watcher...")

    scheduler = BlockingScheduler()
    scheduler.add_job(poll_and_store, 'cron', minute='*/5')

    scheduler.start()

