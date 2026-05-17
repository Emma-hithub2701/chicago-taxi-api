import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import matplotlib.pyplot as plt

# ---------------------------------------------
# Load and prepare the daily rides data
# ---------------------------------------------
df = pd.read_csv('chicago_daily_rides_2022.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

print(f"Loaded {len(df)} days of data")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ---------------------------------------------
# Feature engineering
# ---------------------------------------------
df['dayofweek'] = df['date'].dt.dayofweek          # Monday=0, Sunday=6
df['month'] = df['date'].dt.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Lags
df['rides_lag_1'] = df['rides'].shift(1)           # Yesterday
df['rides_lag_7'] = df['rides'].shift(7)           # Same day last week
df['rides_lag_14'] = df['rides'].shift(14)         # Two weeks ago

# Rolling means (using only past values – shift before rolling)
df['rolling_mean_7d'] = df['rides'].shift(1).rolling(7).mean()
df['rolling_mean_14d'] = df['rides'].shift(1).rolling(14).mean()

# Drop rows with NaN (first few days won't have complete history)
df = df.dropna().reset_index(drop=True)
print(f"After dropping NaN: {len(df)} days")

# ---------------------------------------------
# Define features and split chronologically
# ---------------------------------------------
feature_columns = [
    'dayofweek', 'month', 'is_weekend',
    'rides_lag_1', 'rides_lag_7', 'rides_lag_14',
    'rolling_mean_7d', 'rolling_mean_14d'
]

# Use the first 85% for training, last 15% for testing
split_idx = int(len(df) * 0.85)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

X_train, y_train = train[feature_columns], train['rides']
X_test, y_test = test[feature_columns], test['rides']

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---------------------------------------------
# Train XGBoost model
# ---------------------------------------------
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------------------------
# Evaluate on test set
# ---------------------------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
avg_rides = y_test.mean()
print(f"\nTest MAE: {mae:.0f} rides")
print(f"Average daily rides: {avg_rides:.0f}")
print(f"Error percentage: {(mae / avg_rides) * 100:.1f}%")

# ---------------------------------------------
# Save model and required files
# ---------------------------------------------
with open('taxi_model_chicago.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('feature_names_chicago.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# Save recent data for the API (last 30 days)
recent_data = df.tail(30)[['date', 'rides']]
recent_data.to_csv('recent_data_chicago.csv', index=False)

# ---------------------------------------------
# Optional: plot actual vs predicted
# ---------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(test['date'], y_test, label='Actual', alpha=0.7)
plt.plot(test['date'], preds, label='Predicted', alpha=0.7)
plt.legend()
plt.title('Chicago Taxi Rides – Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Rides')
plt.tight_layout()
plt.savefig('chicago_predictions.png')
plt.show()  # If a display is available; otherwise it will just save
print("\n✅ Model and files saved. Ready to start the API.")