import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import json

# Alpha Vantage API key
API_KEY = "AELOSC3OP7F0I708"

def fetch_forex_data(from_currency, to_currency):
    base_url = "https://www.alphavantage.co/query"
    function = "FX_DAILY"
    
    url = f"{base_url}?function={function}&from_symbol={from_currency}&to_symbol={to_currency}&apikey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    if "Time Series FX (Daily)" not in data:
        print(f"Error fetching data for {from_currency}/{to_currency}. Response: {data}")
        return None
    
    df = pd.DataFrame(data["Time Series FX (Daily)"]).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = ['open', 'high', 'low', 'close']
    return df

def fetch_all_forex_data():
    currency_pairs = [("EUR", "USD"), ("GBP", "USD"), ("JPY", "USD")]
    forex_data = {}
    
    for base, quote in currency_pairs:
        print(f"Fetching {base}/{quote} data...")
        forex_data[f"{base}/{quote}"] = fetch_forex_data(base, quote)
        
    return forex_data

def prepare_data(forex_data):
    combined_data = pd.DataFrame()
    
    for pair, data in forex_data.items():
        if data is not None:
            combined_data[f"{pair}_close"] = data['close']
    
    combined_data.dropna(inplace=True)
    returns = combined_data.pct_change()
    
    X = returns.iloc[:-1]
    y = returns.iloc[1:]
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    return model

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    return joblib.load(filename)

def make_prediction(model, latest_data):
    prediction = model.predict(latest_data.reshape(1, -1))
    return prediction[0]

def save_prediction(prediction, actual, timestamp):
    predictions_file = 'predictions.json'
    
    if os.path.exists(predictions_file):
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
    else:
        predictions = []
    
    predictions.append({
        'timestamp': timestamp.strftime('%Y-%m-%d'),
        'prediction': prediction.tolist(),
        'actual': actual.tolist() if actual is not None else None
    })
    
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)

def evaluate_predictions():
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)
    
    actual_values = [p['actual'] for p in predictions if p['actual'] is not None]
    predicted_values = [p['prediction'] for p in predictions if p['actual'] is not None]
    
    if len(actual_values) > 0:
        mse = mean_squared_error(actual_values, predicted_values)
        print(f"Mean Squared Error of predictions: {mse}")
    else:
        print("Not enough data to evaluate predictions yet.")

def main():
    model_filename = 'forex_prediction_model.joblib'
    
    # Fetch forex data
    forex_data = fetch_all_forex_data()
    
    # Prepare data for modeling
    X, y = prepare_data(forex_data)
    
    # Check if model exists, if not, train a new one
    if os.path.exists(model_filename):
        model = load_model(model_filename)
        print("Loaded existing model.")
    else:
        model = train_model(X, y)
        save_model(model, model_filename)
    
    # Make a prediction for the next day
    latest_data = X.iloc[-1].values
    prediction = make_prediction(model, latest_data)
    
    # Save the prediction
    save_prediction(prediction, None, datetime.now())
    
    print(f"Prediction for next day's returns: {prediction}")
    
    # Evaluate previous predictions
    evaluate_predictions()
    
    print("Prediction made and saved. Run this script daily to make new predictions and update the model.")

if __name__ == "__main__":
    main()