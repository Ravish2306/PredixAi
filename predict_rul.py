import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

print("Loading trained model...")

model = load_model("output/rul_model.h5", compile=False)

print("Loading data...")

X = np.load("output/X_train.npy")
y = np.load("output/y_train.npy")

print("Predicting RUL...")

predictions = model.predict(X)

print("\nSample Predictions:")

for i in range(10):
    print(f"Actual: {y[i]:.2f}  Predicted: {predictions[i][0]:.2f}")

plt.plot(y[:200], label="Actual RUL")
plt.plot(predictions[:200], label="Predicted RUL")

plt.legend()
plt.title("Actual vs Predicted RUL")

plt.show()