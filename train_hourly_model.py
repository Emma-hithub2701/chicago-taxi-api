import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle

print("=== Training Hourly Taxi Demand Model ===")

# Load hourly data
df = pd.read_csv('hourly_rides_2022.csv')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
df = df.sort_values('datetime').reset_index(drop=True)
print(f"Total hourly records: {len(df)}")

# Feature engineering
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Lag features (shift in hours)
df['rides_lag_1h'] = df['rides'].shift(1)      # previous hour
df['rides_lag_2h'] = df['rides'].shift(2)
df['rides_lag_24h'] = df['rides'].shift(24)    # same hour yesterday
df['rides_lag_168h'] = df['rides'].shift(168)  # same hour last week

# Rolling averages
df['rolling_mean_3h'] = df['rides'].shift(1).rolling(3).mean()
df['rolling_mean_24h'] = df['rides'].shift(1).rolling(24).mean()

# Drop rows with NaN
df = df.dropna().reset_index(drop=True)
print(f"Rows after dropna: {len(df)}")

# Features and target
feature_columns = [
    'hour', 'dayofweek', 'month', 'is_weekend',
    'rides_lag_1h', 'rides_lag_2h', 'rides_lag_24h', 'rides_lag_168h',
    'rolling_mean_3h', 'rolling_mean_24h'
]

# Chronological split
split_idx = int(len(df) * 0.85)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

X_train = train[feature_columns]
y_train = train['rides']
X_test = test[feature_columns]
y_test = test['rides']

print(f"Train hours: {len(train)}, Test hours: {len(test)}")

# Train XGBoost
model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
avg_rides = y_test.mean()
print(f"Test MAE: {mae:.1f} rides per hour")
print(f"Average hourly rides: {avg_rides:.1f}")
print(f"MAPE: {mae/avg_rides*100:.1f}%")

# Save the model and helper files
with open('taxi_model_hourly.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('feature_names_hourly.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# Save last 7 days (168 hours) of recent data for API lag features
recent = df.tail(168)[['datetime', 'rides']].copy()
recent.to_csv('recent_data_hourly.csv', index=False)

print("✅ Hourly model saved: taxi_model_hourly.pkl, feature_names_hourly.pkl, recent_data_hourly.csv")