import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.title("Engine Remaining Useful Life Prediction")

model = load_model("output/rul_model.h5", compile=False)

X = np.load("output/X_train.npy")
y = np.load("output/y_train.npy")

predictions = model.predict(X)

st.write("Sample Prediction Results")

for i in range(5):
    st.write(f"Actual: {y[i]:.2f} | Predicted: {predictions[i][0]:.2f}")

fig, ax = plt.subplots()

ax.plot(y[:200], label="Actual RUL")
ax.plot(predictions[:200], label="Predicted RUL")

ax.set_title("Actual vs Predicted RUL")
ax.legend()

st.pyplot(fig)