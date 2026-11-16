CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    engine_id INT,
    predicted_rul FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);