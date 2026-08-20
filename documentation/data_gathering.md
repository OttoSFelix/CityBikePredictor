# Data gathered for the model

The data gathered for the model is gathered from digitransit's GraphQL API, Digitraffic's TMS API and the Finnish Meteorological Institute's open data API. 
[scraper](docker_services.md#scraper) handles the data gathering entirely.

Here are the datapoints being gathered:
- station id for each bike station
- lat and lon coordinates of each bike station
- bike amount of each bike station
- sensor id for each traffic sensor
- traffic volume from each traffic sensor
- rainfall (in mm/h for the past 10 minutes)

These values are stored into a PostgreSQL database with 5 different tables described below:

### bike_stations
A table to store the names and coordinates of each bike station

`schema`: \
station_id VARCHAR(50) PRIMARY KEY UNIQUE, \
    name VARCHAR(100) UNIQUE, \
    lat DOUBLE PRECISION UNIQUE, \
    lon DOUBLE PRECISION UNIQUE 

### bike_readings
A table to store the bike readings for each station at a given time

`schema`: \
timestamp TIMESTAMP NOT NULL, \
    station_id VARCHAR(50) REFERENCES bike_stations(station_id), \
    bikes INTEGER NOT NULL, 

### traffic_readings
A table to store the traffic volume for each traffic sensor at a given time

`schema`: \
timestamp TIMESTAMP NOT NULL, \
    sensor_id INTEGER NOT NULL, \
    volume INTEGER NOT NULL, 

### weather_readings
A table to store the rainfall at Helsinki at a given time

`schema`: \
timestamp TIMESTAMP PRIMARY KEY, \
    rainfall_mm REAL NOT NULL 

### station_mapping
A table used for quick lookups for the nearest 3 traffic sensors to each station

`schema`: \
bike_station_id VARCHAR(50) PRIMARY KEY, \
    tms_sensor_1 INTEGER, \
    tms_sensor_2 INTEGER, \
    tms_sensor_3 INTEGER 

