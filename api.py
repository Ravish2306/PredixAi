import psycopg2
from fastapi import FastAPI
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# 🔹 DB connection
def connect_db():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="@manish900yadav",
        host="localhost",
        port="5432"
    )
    return conn

# 🔹 TABLE CREATE (ADD THIS)
def create_table():
    conn = connect_db()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            engine_id INT,
            predicted_rul FLOAT
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

# 🔹 MODEL LOAD
print("Loading model...")
model = load_model("OutPut/rul_model.h5", compile=False)

# 🔥 IMPORTANT (ADD THIS LINE)
create_table()

# 🔹 HOME API
@app.get("/")
def home():
    return {"message": "PrognosAI API Running 🚀"}

# 🔹 PREDICT API
@app.get("/predict")
def predict():
    X = np.load("OutPut/X_train.npy")
    pred = model.predict(X[:1])

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO predictions (engine_id, predicted_rul) VALUES (%s, %s)",
        (1, float(pred[0][0]))
    )

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "engine_id": 1,
        "predicted_rul": float(pred[0][0])
    }