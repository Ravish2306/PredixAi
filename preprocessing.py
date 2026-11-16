import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("Loading Dataset...")

columns = ['engine_id','cycle','op1','op2','op3']
for i in range(1,22):
    columns.append(f'sensor{i}')

data = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None)
data.columns = columns

max_cycle = data.groupby('engine_id')['cycle'].max()
data = data.merge(max_cycle.to_frame(name='max_cycle'), on='engine_id')

data['RUL'] = data['max_cycle'] - data['cycle']
data.drop('max_cycle', axis=1, inplace=True)

print("RUL Column Added")

sensor_cols = [col for col in data.columns if 'sensor' in col]

scaler = MinMaxScaler()
data[sensor_cols] = scaler.fit_transform(data[sensor_cols])

print("Sensor Data Normalized")

sequence_length = 30
X=[]
y=[]

for engine in data['engine_id'].unique():

    engine_data = data[data['engine_id']==engine]

    sensors = engine_data[sensor_cols].values
    rul = engine_data['RUL'].values

    for i in range(len(engine_data)-sequence_length):

        X.append(sensors[i:i+sequence_length])
        y.append(rul[i+sequence_length])

X=np.array(X)
y=np.array(y)

print("Sequences Generated")
print("X shape:",X.shape)
print("y shape:",y.shape)

np.save("output/X_train.npy",X)
np.save("output/y_train.npy",y)

print("Milestone 1 Completed Successfully")
