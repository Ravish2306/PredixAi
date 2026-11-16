import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("OutPut/rul_model.h5", compile=False)

# Load data
X = np.load("OutPut/X_train.npy")

print("Running bulk predictions...\n")

# Loop through first 10 engines
for i in range(10):
    sample = X[i:i+1]
    pred = model.predict(sample)
    
    print(f"Engine {i+1} Predicted RUL: {pred[0][0]:.2f}")