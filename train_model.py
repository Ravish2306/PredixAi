import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("Loading training data...")

X = np.load("output/X_train.npy")
y = np.load("output/y_train.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

model = Sequential()

model.add(LSTM(64, input_shape=(30,21), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

print("Starting training...")

history = model.fit(
    X,
    y,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

model.save("output/rul_model.h5")

print("Model training completed and saved.")
