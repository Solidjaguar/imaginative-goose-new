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
import pandas_ta as ta
import logging

# Set up logging
logging.basicConfig(filename='forex_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Alpha Vantage API key
API_KEY = "AELOSC3OP7F0I708"

def fetch_forex_data(from_currency, to_currency):
    base_url = "https://www.alphavantage.co/query"
    function = "FX_DAILY"
    
    url = f"{base_url}?function={function}&from_symbol={from_currency}&to_symbol={to_currency}&apikey={API_KEY}&outputsize=full"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series FX (Daily)" not in data:
            logging.error(f"Error fetching data for {from_currency}/{to_currency}. Response: {data}")
            return None
        
        df = pd.DataFrame(data["Time Series FX (Daily)"]).T
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df.columns = ['open', 'high', 'low', 'close']
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data for {from_currency}/{to_currency}: {str(e)}")
        return None

def add_technical_indicators(df):
    df['SMA_10'] = ta.sma(df['close'], length=10)
    df['SMA_30'] = ta.sma(df['close'], length=30)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df

def fetch_all_data():
    currency_pairs = [("EUR", "USD"), ("GBP", "USD"), ("JPY", "USD")]
    
    data = {}
    
    for base, quote in currency_pairs:
        logging.info(f"Fetching {base}/{quote} data...")
        pair_data = fetch_forex_data(base, quote)
        if pair_data is not None:
            data[f"{base}/{quote}"] = add_technical_indicators(pair_data)
    
    return data

def prepare_data(data):
    combined_data = pd.DataFrame()
    
    for key, df in data.items():
        if df is not None:
            for col in df.columns:
                combined_data[f"{key}_{col}"] = df[col]
    
    combined_data.dropna(inplace=True)
    
    forex_returns = combined_data[[f"{pair}_close" for pair in data.keys()]].pct_change()
    features = pd.concat([forex_returns, combined_data.drop([f"{pair}_close" for pair in data.keys()], axis=1)], axis=1)
    features.dropna(inplace=True)
    
    X = features.iloc[:-1]
    y = forex_returns.iloc[1:]
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Mean Squared Error: {mse}")
    
    return model

def save_model(model, filename):
    joblib.dump(model, filename)
    logging.info(f"Model saved to {filename}")

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
        logging.info(f"Mean Squared Error of predictions: {mse}")
    else:
        logging.info("Not enough data to evaluate predictions yet.")

def main():
    try:
        model_filename = 'forex_prediction_model.joblib'
        
        # Fetch all data
        data = fetch_all_data()
        
        # Prepare data for modeling
        X, y = prepare_data(data)
        
        # Check if model exists, if not, train a new one
        if os.path.exists(model_filename):
            model = load_model(model_filename)
            logging.info("Loaded existing model.")
        else:
            model = train_model(X, y)
            save_model(model, model_filename)
        
        # Make a prediction for the next day
        latest_data = X.iloc[-1].values
        prediction = make_prediction(model, latest_data)
        
        # Save the prediction
        save_prediction(prediction, None, datetime.now())
        
        logging.info(f"Prediction for next day's returns: {prediction}")
        
        # Evaluate previous predictions
        evaluate_predictions()
        
        logging.info("Prediction made and saved. Run this script daily to make new predictions and update the model.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()