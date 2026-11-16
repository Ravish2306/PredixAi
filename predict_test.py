import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

print("Loading model...")

model = load_model("output/rul_model.h5", compile=False)

print("Loading test dataset...")

columns = ['engine_id','cycle','op1','op2','op3']

for i in range(1,22):
    columns.append(f'sensor{i}')

data = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None)
data.columns = columns

print("Preparing sequences...")

sensor_cols = [col for col in data.columns if "sensor" in col]

sequence_length = 30

X_test = []

for engine in data['engine_id'].unique():

    engine_data = data[data['engine_id'] == engine]

    sensors = engine_data[sensor_cols].values

    if len(sensors) >= sequence_length:
        X_test.append(sensors[-sequence_length:])

X_test = np.array(X_test)

print("Making predictions...")

predictions = model.predict(X_test)

print("\nPredicted RUL for test engines:")

for i, pred in enumerate(predictions[:10]):
    print(f"Engine {i+1} Predicted RUL: {pred[0]:.2f}")