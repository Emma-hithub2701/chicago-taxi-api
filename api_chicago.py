from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi.responses import FileResponse
app = FastAPI(title="Chicago Taxi Demand Predictor")

# Allow the dashboard to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and helper files
with open('taxi_model_chicago.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names_chicago.pkl', 'rb') as f:
    feature_names = pickle.load(f)

recent = pd.read_csv('recent_data_chicago.csv')
recent_rides = recent['rides'].tolist()

@app.get("/")
def home():
    return {
        "app": "Chicago Taxi Demand Predictor",
        "model": "XGBoost trained on 2022 data",
        "features": feature_names
    }

@app.get("/predict/tomorrow")
def predict_tomorrow():
    tomorrow = datetime.now() + timedelta(days=1)

    features = {
        'dayofweek': tomorrow.weekday(),
        'month': tomorrow.month,
        'is_weekend': 1 if tomorrow.weekday() >= 5 else 0,
        'rides_lag_1': recent_rides[-1],
        'rides_lag_7': recent_rides[-7] if len(recent_rides) >= 7 else recent_rides[-1],
        'rides_lag_14': recent_rides[-14] if len(recent_rides) >= 14 else recent_rides[-1],
        'rolling_mean_7d': np.mean(recent_rides[-7:]),
        'rolling_mean_14d': np.mean(recent_rides[-14:]) if len(recent_rides) >= 14 else np.mean(recent_rides)
    }

    features_df = pd.DataFrame([features])[feature_names]
    prediction = model.predict(features_df)[0]

    return {
        "date": tomorrow.strftime("%Y-%m-%d"),
        "day": tomorrow.strftime("%A"),
        "predicted_rides": int(prediction),
        "range": {
            "low": int(prediction * 0.85),
            "high": int(prediction * 1.15)
        }
    }
@app.get("/predict/date")
def predict_date(target_date: str):
    from datetime import datetime as dt
    try:
        target_dt = dt.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    
    # For simplicity, use the same logic as tomorrow but with the target date's features
    # In a real scenario, you'd recalculate features based on recent data up to target_date-1
    features = {
        'dayofweek': target_dt.weekday(),
        'month': target_dt.month,
        'is_weekend': 1 if target_dt.weekday() >= 5 else 0,
        'rides_lag_1': recent_rides[-1],
        'rides_lag_7': recent_rides[-7] if len(recent_rides) >= 7 else recent_rides[-1],
        'rides_lag_14': recent_rides[-14] if len(recent_rides) >= 14 else recent_rides[-1],
        'rolling_mean_7d': np.mean(recent_rides[-7:]),
        'rolling_mean_14d': np.mean(recent_rides[-14:]) if len(recent_rides) >= 14 else np.mean(recent_rides)
    }
    features_df = pd.DataFrame([features])[feature_names]
    prediction = model.predict(features_df)[0]
    
    return {
        "date": target_date,
        "day": target_dt.strftime("%A"),
        "predicted_rides": int(prediction),
        "range": {"low": int(prediction * 0.85), "high": int(prediction * 1.15)}
    }

@app.get("/health")
def health():
    return {"status": "healthy", "city": "Chicago"}
@app.get("/dashboard")
def serve_dashboard():
    return FileResponse("dashboard.html")