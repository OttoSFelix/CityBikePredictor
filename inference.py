import math
import torch
import joblib
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from time import sleep
import os
import psycopg2
from apscheduler.schedulers.background import BackgroundScheduler
from neuralnet import PytorchModel

class Inference:
    def __init__(self):
        self._weight_path = f'./parameters/model_weights.pth'
        self._state_dict = torch.load(self._weight_path, weights_only=True)
        self._scaler = joblib.load(f'./parameters/model_scaler.joblib')
        self._encoder = joblib.load(f'./parameters/model_encoder.joblib')
        self._num_stations = self._state_dict['embedding.weight'].shape[0]
        self._model = PytorchModel(num_stations=self._num_stations)
        self._model.eval()

        self.conn = psycopg2.connect(
        dbname='mydb',
        user=os.environ.get('USER'),
        password=os.environ.get('PASSWORD'),
        host=os.environ.get('DB_HOST', 'localhost'),
        port=os.environ.get('DB_PORT', '4321')
        )
        self.api_key = os.environ.get('API_KEY')

        self.cursor = self.conn.cursor()
        self.predictions = {}
        self.station_lookup = {}

        self.outflow_treshold = 0.363
        self.inflow_treshold = 0.35

        self.tensors = []

        self.initialize_lookup_map()

    def initialize_lookup_map(self):
        self.cursor.execute('SELECT * FROM station_mapping;')
        result = self.cursor.fetchall()

        for row in result:
            self.station_lookup[row[0]] = [row[1], row[2], row[3]]

    def parse_time(self, timestamp, utc: bool = True, date: bool = True):
        timedata = timestamp
        if date:
            stripped = timedata.split('T')
            stripped_time = stripped[1][:-1]
            stripped_date = stripped[0]
            timedata = stripped_time
            datedata = stripped_date
        time = timedata.split(':')
        h = int(time[0])
        if utc:
            h += 3
        min = int(time[1])
        sec = float(time[2])
        total_time = int((h * 60 * 60) + (min * 60) + sec)
        if date:
            return datedata, total_time
        return total_time

    def get_rainfall(self):
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

    def get_nearest_traffic(self):
        url = "https://tie.digitraffic.fi/api/tms/v1/stations/data"
        headers = {"Digitraffic-User": "CityBikePredictor_UniversityOfHelsinki_Student/1.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        traffic_map = {}
        stations = response.json().get("stations", [])

        for station in stations:
            for sensor in station.get("sensorValues", []):
                if sensor.get('id') == 5016: 
                    sensor_id = station.get("id") 
                    traffic_map[sensor_id] = sensor.get("value", 0)
        
        return traffic_map

    def get_bike_amount(self):
        url = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"
        headers = {
            "Content-Type": "application/json",
            "Digitransit-Subscription-Key": self.api_key
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
                    total_bikes = sum(item.get('count', None) for item in by_type if isinstance(item, dict))
                else:
                    total_bikes = None
                if total_bikes:
                    cleaned_stations.append({
                        "station_id": s["stationId"],
                        "bikes": total_bikes,
                    })

        return cleaned_stations

    def perform_inference(self, payload):
        sid_raw = payload['station_id']
        if isinstance(sid_raw, str):
            clean_id = int(sid_raw.replace('v', '').split(':')[-1])
        else:
            clean_id = int(sid_raw)
        station_id = self._encoder.transform([clean_id])[0]

        time_sin = math.sin(payload['time'] * (2 * math.pi / 86400))
        time_cos = math.cos(payload['time'] * (2 * math.pi / 86400))
        day_sin = math.sin(payload['weekday'] * (2 * math.pi / 7))
        day_cos = math.cos(payload['weekday'] * (2 * math.pi / 7))
        traffic = payload['traffic']
        mean_volume = sum(traffic) // len(traffic) if traffic else 0

        tensor = torch.tensor([[time_sin, time_cos, day_sin, day_cos, mean_volume, payload['bikes'], payload['rainfall'], station_id]], dtype=torch.float32)
        tensor[:, 4:7] = torch.tensor(self._scaler.transform(tensor[:, 4:7]), dtype=torch.float32)

        with torch.no_grad():
            prediction = self._model(tensor)
            probabilities = torch.nn.functional.softmax(prediction, dim=1)

        self.tensors.append(probabilities)
        return torch.argmax(prediction, dim=1).item()

        
        if probabilities[0][0].item() >= self.outflow_treshold:
            return 0
        if probabilities[0][1].item() >= self.inflow_treshold:
            return 1
        return 2

    def measure_inference(self):
        time = datetime.now()
        print(f'Measuring at {time}...')
        weekday = time.weekday()
        seconds = self.parse_time(str(time.time()), date=False)

        bike_data = self.get_bike_amount()
        rainfall = self.get_rainfall()
        global_traffic_map = self.get_nearest_traffic()

        predictions = {}
        for station in bike_data:
            station_id = station['station_id']
            nearest_sensors = self.station_lookup.get(station_id, [])
            station_traffic = [global_traffic_map.get(s_id, 0) for s_id in nearest_sensors]

            payload = {}
            payload['station_id'] = station_id
            payload['bikes'] = station['bikes']
            payload['rainfall'] = rainfall
            payload['traffic'] = station_traffic
            payload['time'] = seconds
            payload['weekday'] = weekday


            prediction = self.perform_inference(payload)
            predictions[station_id] = {
                'bikes': station['bikes'],
                'prediction': prediction
            }

        n = len(self.predictions)
        self.predictions[n] = predictions
        print(f'Predictions done for {time}!')

    def start_live_inference(self, duration):
        scheduler = BackgroundScheduler()
        scheduler.add_job(self.measure_inference, 'cron', minute='*/5')

        scheduler.start()
        sleep(duration)
        scheduler.shutdown()

        correct = 0
        total = 0
        pred_distribution = {1: [0, 0], 0: [0, 0], 2: [0, 0]}

        num_snapshots = len(self.predictions)
        for n in range(num_snapshots - 6):
            start_snapshot = self.predictions.get(n, {})
            end_snapshot = self.predictions.get(n + 6, {})

            for station_id, start_data in start_snapshot.items():
                if station_id in end_snapshot:
                    start_bikes = start_data['bikes']
                    end_bikes = end_snapshot[station_id]['bikes']
                    actual_diff = end_bikes - start_bikes

                    if actual_diff < -3:
                        actual_class = 0
                    elif actual_diff > 3:
                        actual_class = 1
                    else:
                        actual_class = 2

                    predicted_class = start_data['prediction']
                    if isinstance(predicted_class, torch.Tensor):
                        predicted_class = torch.argmax(predicted_class).item()

                    pred_distribution[predicted_class][1] += 1
                    if predicted_class == actual_class:
                        correct += 1
                        pred_distribution[predicted_class][0] += 1
                    total += 1


        if total > 0:
            out_correct, out_total = pred_distribution[0][0], pred_distribution[0][1]
            in_correct, in_total = pred_distribution[1][0], pred_distribution[1][1]
            stable_correct, stable_total = pred_distribution[2][0], pred_distribution[2][1]
            accuracy = correct / total
            out_accuracy = out_correct / out_total if out_total > 0 else 0
            in_accuracy = in_correct / in_total if in_total > 0 else 0
            stable_accuracy = stable_correct / stable_total if stable_total > 0 else 0
            print(f"Total Accuracy: {accuracy:.4f} ({correct}/{total} predictions correct)")
            print(f"Outflow Accuracy: {out_accuracy:.4f} ({out_correct}/{out_total} predictions correct)")
            print(f"Inflow Accuracy: {in_accuracy:.4f} ({in_correct}/{in_total} predictions correct)")
            print(f"Stable Accuracy: {stable_accuracy:.4f} ({stable_correct}/{stable_total} predictions correct)")


            return accuracy
        else:
            print("Not enough predictions to measure accuracy (requires at least 7 5-minute snapshots / 30 minutes).")
            return 0.0


if __name__ == '__main__':
    inf = Inference()
    inf.start_live_inference(4000)



