import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import psycopg2
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import math
import datetime
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, EpochScoring, EarlyStopping
from skorch.dataset import ValidSplit
import matplotlib.pyplot as plt
import joblib
import numpy as np
import signal

class PytorchModel(nn.Module):
    def __init__(self, num_stations):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_stations, embedding_dim=16)

        self.main_network = nn.Sequential(
            nn.Linear(23, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 3)
        )

    def forward(self, X):
        continuous_features = X[:, :7]
        station_ids = X[:, 7].long()
        embedded_ids = self.embedding(station_ids)
        combined_features = torch.cat((continuous_features, embedded_ids), dim=1)
        return self.main_network(combined_features)


class NeuralNetwork:
    def __init__(self):
        self.conn = psycopg2.connect(
        dbname='mydb',
        user=os.environ.get('USER'),
        password=os.environ.get('PASSWORD'),
        host=os.environ.get('DB_HOST', 'localhost'),
        port=os.environ.get('DB_PORT', '4321')
        )

        self.cursor = self.conn.cursor()

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

    def apply_custom_thresholds(self, probs, threshold_outflow=0.30, threshold_inflow=0.30):
        y_pred = np.full(probs.shape[0], 2)
        p_out = probs[:, 0]
        p_in = probs[:, 1]
        trigger_mask = (p_out >= threshold_outflow) | (p_in >= threshold_inflow)
        y_pred[trigger_mask] = np.argmax(probs[trigger_mask, :2], axis=1)
        
        return y_pred

    def train(self, graph=False, verbose=True):
        print('Fetching data...')
        query = """
        WITH traffic_aggregated AS (
        SELECT
            date_trunc('minute', tr.timestamp) as time,
            sm.bike_station_id,
            AVG(tr.volume) as mean_volume
        FROM traffic_readings tr
        JOIN station_mapping sm
            ON tr.sensor_id IN (sm.tms_sensor_1, sm.tms_sensor_2, sm.tms_sensor_3)
        GROUP BY 1, 2
    ),
    ordered_snapshots AS (
        SELECT
            br.station_id,
            date_trunc('minute', br.timestamp) as time,
            br.bikes,
            ta.mean_volume,
            wr.rainfall_mm,
            LEAD(br.bikes, 6) OVER (PARTITION BY br.station_id ORDER BY br.timestamp) as future_bikes
        FROM bike_readings br
        JOIN weather_readings wr
            ON date_trunc('minute', br.timestamp) = date_trunc('minute', wr.timestamp)
        JOIN traffic_aggregated ta
            ON ta.time = date_trunc('minute', br.timestamp)
            AND ta.bike_station_id = br.station_id
    )
    SELECT * FROM ordered_snapshots
    WHERE future_bikes IS NOT NULL
    ORDER BY station_id, time;
        """
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        print('Data fetched')

        X_data = []
        y_data = []

        self.encoder = LabelEncoder()
        self.cursor.execute('SELECT station_id from bike_stations;')
        unique_stations = [int(row[0].split(':')[1]) for row in self.cursor.fetchall()]
        self.encoder.fit(unique_stations)
        num_stations = len(self.encoder.classes_)
        station_map = {station: self.encoder.transform([station])[0] for station in unique_stations}

        for row in data:
            station_id, time_obj, bikes, mean_volume, rainfall, future_bikes = row

            station_id = int(station_id.split(':')[1])
            time_sec = self.parse_time(str(time_obj.time()), date=False)
            time_sin = math.sin(time_sec * (2 * math.pi / 86400))
            time_cos = math.cos(time_sec * (2 * math.pi / 86400))
            weekday = time_obj.weekday()
            day_sin = math.sin(weekday * (2 * math.pi / 7))
            day_cos = math.cos(weekday * (2 * math.pi / 7))

            encoded_id = station_map[station_id]

            X_data.append([time_sin, time_cos, day_sin, day_cos, mean_volume, bikes, rainfall, encoded_id])

            delta = future_bikes - bikes
            if delta < -3:
                y_data.append(0)
            elif delta > 3:
                y_data.append(1)
            else:
                y_data.append(2)


        self.scaler = StandardScaler()

        X = torch.tensor(X_data, dtype=torch.float32)
        y = torch.tensor(y_data, dtype=torch.long)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=10)
        X_train_scaled = X_train.clone()
        X_test_scaled = X_test.clone()
        X_train_scaled[:, 4:7] = torch.tensor(self.scaler.fit_transform(X_train[:, 4:7]), dtype=torch.float32)
        X_test_scaled[:, 4:7] = torch.tensor(self.scaler.transform(X_test[:, 4:7]), dtype=torch.float32)

        self.model = PytorchModel(num_stations)

        lr = 0.0003
        max_epochs = 60
        batch_size = 256
        lr_patience = 3
        lr_factor = 0.3
        stop_patience = 10

        self.net = NeuralNetClassifier(
            module=self.model,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_split=ValidSplit(0.2),
            callbacks=[
                LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=lr_patience, factor=lr_factor),
                EpochScoring(scoring='f1_macro', name='valid_f1_macro', lower_is_better=False),
                EarlyStopping(monitor='valid_loss', patience=stop_patience, lower_is_better=True)
            ]
        )

        try:
            self.net.fit(X_train_scaled, y_train)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

        try:
            losses = self.net.history[:, 'train_loss']
        except (KeyError, IndexError):
            losses = []

        raw_accuracy = self.net.score(X_test_scaled, y_test)
        print(f"Overall raw accuracy: {raw_accuracy:.4f}")

        y_probs = self.net.predict_proba(X_test_scaled)
        y_pred = self.apply_custom_thresholds(y_probs)
        y_test_np = y_test.numpy()

        print("\n--- Detailed Classification Report ---")
        target_names = ['High Outflow (0)', 'High Inflow (1)', 'Stable (2)']
        print(classification_report(y_test_np, y_pred, target_names=target_names, zero_division=0))

        print("--- Confusion Matrix ---")
        print(confusion_matrix(y_test_np, y_pred))
        if verbose:
            print(f'Learning rate: {lr}, batch_size: {batch_size}')
            print(f'LR patience: {lr_patience}, LR factor: {lr_factor}')
            print(f'Early stopping patience: {stop_patience}')
            print('Arch:')
            for name, layer in self.model.named_modules():
                if isinstance(layer, nn.Linear):
                    print(layer.in_features)


        if graph:
            plt.figure(figsize=(8, 4))
            plt.plot(losses, color='blue', linewidth=2)
            plt.title("Training Loss over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Logistic Loss")
            plt.grid(True, linestyle='--', alpha=0.6)
            save_path = os.path.join(os.path.dirname(__file__), 'training_loss.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            print(f'Saved plot to {save_path}')

    def save_weights(self):
        torch.save(self.net.module_.state_dict(), f'./parameters/model_weights.pth')
        joblib.dump(self.scaler, f'./parameters/model_scaler.joblib')
        joblib.dump(self.encoder, f'./parameters/model_encoder.joblib')
        print(f'Parameters saved.')

    def stop_training(self, signum=None, frame=None):
        print("\nKeyboard interrupt received. Stopping training...")
        raise KeyboardInterrupt


    def testi(self):
        tensor = torch.tensor([3], dtype=torch.float32)
        sig = torch.sigmoid_(tensor)
        print(sig.item())








if __name__ == '__main__':
    network = NeuralNetwork()
    signal.signal(signal.SIGINT, network.stop_training)
    network.train(graph=True)
    # network.save_weights()

