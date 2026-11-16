import numpy as np
import pandas as pd
import math
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading model...")
model = load_model("Output/rul_model.h5", compile=False)

print("Loading test data...")

# Column names
cols = ['engine_id','cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]

# Load test data
df_test = pd.read_csv("data/test_FD001.txt", sep=" ", header=None)
df_test = df_test.dropna(axis=1)
df_test.columns = cols

# =========================
# 🔥 CREATE SEQUENCES (IMPORTANT FIX)
# =========================

sequence_length = 30
X_test = []

for engine_id in df_test["engine_id"].unique():
    df_engine = df_test[df_test["engine_id"] == engine_id]

    if len(df_engine) >= sequence_length:
        seq = df_engine[[f's{i}' for i in range(1,22)]].iloc[-sequence_length:] # skip engine_id, cycle
    else:
        padding = np.zeros((sequence_length - len(df_engine), df_engine.shape[1]-2))
        seq = np.vstack((padding, df_engine.iloc[:, 2:].values))

    X_test.append(seq)

X_test = np.array(X_test)

print("X_test shape:", X_test.shape)  # DEBUG

# =========================
# Load actual RUL
# =========================

print("Loading actual RUL...")
rul = pd.read_csv("data/RUL_FD001.txt", header=None)
y_true = rul[0].values

# =========================
# Prediction
# =========================

print("Making predictions...")
pred = model.predict(X_test).flatten()

# =========================
# 📊 Metrics
# =========================

rmse = math.sqrt(mean_squared_error(y_true, pred))
mae = mean_absolute_error(y_true, pred)
r2 = r2_score(y_true, pred)

print("\n📊 FINAL TEST RESULT")
print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)

# =========================
# 🔥 NASA Score
# =========================

def nasa_score(y_true, y_pred):
    score = 0
    for i in range(len(y_true)):
        d = y_pred[i] - y_true[i]
        if d < 0:
            score += math.exp(-d/13) - 1
        else:
            score += math.exp(d/10) - 1
    return score

nasa = nasa_score(y_true, pred)

print("NASA Score:", nasa)
# =========================
# 💾 Save predictions
# =========================

output_df = pd.DataFrame({
    "engine_id": range(1, len(pred)+1),
    "predicted_RUL": pred
})

output_df.to_csv("Output/predictions.csv", index=False)

print("✅ Predictions saved to Output/predictions.csv")