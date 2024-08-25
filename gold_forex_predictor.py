import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import io
import base64
import logging
import os
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from sklearn.ensemble import RandomForestRegressor
import joblib

nltk.download('vader_lexicon', quiet=True)

logging.basicConfig(filename='gold_forex_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# API keys
ALPHA_VANTAGE_API_KEY = "PIFHGHQNBWL37L0T"
CURRENTS_API_KEY = "IEfpA5hCrH6Xh4E9f7R0jEOHcEjxSI8k6s71NwcYXRPqtohR"
EXCHANGE_RATES_API_KEY = "977aa5b8e6b88d6e1d0c82ce1aabe665"

# Function definitions (fetch_all_data, fetch_forex_data, fetch_economic_data, fetch_crypto_data, fetch_news_sentiment, add_advanced_features, prepare_data) remain the same as in the previous version

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_predictions(model, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    return predictions

def save_files():
    # Save the main Python script
    with open('gold_forex_predictor.py', 'r') as file:
        script_content = file.read()
    
    with open('gold_forex_predictor_backup.py', 'w') as file:
        file.write(script_content)
    
    logging.info("Main script backed up as 'gold_forex_predictor_backup.py'")

    # Save the trained model
    if os.path.exists('trained_model.joblib'):
        joblib.dump(model, 'trained_model_backup.joblib')
        logging.info("Trained model backed up as 'trained_model_backup.joblib'")

    # Save the scaler
    if os.path.exists('scaler.joblib'):
        joblib.dump(scaler, 'scaler_backup.joblib')
        logging.info("Scaler backed up as 'scaler_backup.joblib'")

    # Save the latest predictions
    if os.path.exists('latest_predictions.csv'):
        predictions_df = pd.read_csv('latest_predictions.csv')
        predictions_df.to_csv('latest_predictions_backup.csv', index=False)
        logging.info("Latest predictions backed up as 'latest_predictions_backup.csv'")

    logging.info("All files have been saved and backed up.")

def run_predictor():
    global model, scaler  # Make these global so they can be accessed in save_files()

    # Fetch and prepare data
    start_date = datetime.now() - timedelta(days=365*5)  # 5 years of data
    end_date = datetime.now()
    
    data = fetch_all_data(start_date, end_date)
    news_sentiment = fetch_news_sentiment(start_date, end_date)
    
    df = prepare_data(data, news_sentiment)
    
    # Prepare features and target
    features = df.drop(['Close'], axis=1)
    target = df['Close']
    
    # Split the data
    split = int(len(df) * 0.8)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = target[:split], target[split:]
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Make predictions
    predictions = make_predictions(model, scaler, X_test)
    
    # Create a DataFrame with dates and predictions
    prediction_df = pd.DataFrame({
        'Date': X_test.index,
        'Predicted_Price': predictions,
        'Actual_Price': y_test
    })
    
    # Save predictions to CSV
    prediction_df.to_csv('latest_predictions.csv', index=False)
    logging.info("Predictions saved to 'latest_predictions.csv'")
    
    # Save the trained model and scaler
    joblib.dump(model, 'trained_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    logging.info("Model and scaler saved.")

    # Save all files
    save_files()

if __name__ == "__main__":
    run_predictor()