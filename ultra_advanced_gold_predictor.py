import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import optuna
from skopt import BayesSearchCV
import requests
from textblob import TextBlob
import talib
from datetime import datetime, timedelta
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pyod.models.iforest import IForest
import logging
from typing import List, Dict, Tuple
from river import linear_model, preprocessing, compose
from scipy.stats import norm
import joblib
from multiprocessing import Pool, cpu_count
from dask import dataframe as dd
from dask.distributed import Client
import fredapi
from newsapi import NewsApiClient
import schedule
import time
import os
from ratelimit import limits, sleep_and_retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize APIs
fred = fredapi.Fred(api_key='b908a65514252b6083d034c389db6ad9')

# Set the News API key as an environment variable
os.environ['NEWS_API_KEY'] = '92b4a1aad8a04c6bb909892b99202d91'

# Initialize NewsApiClient with the key from the environment variable
newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

# Rate limiting decorators
@sleep_and_retry
@limits(calls=95, period=86400)  # 95 calls per day (leaving some margin)
def rate_limited_newsapi_call(func, *args, **kwargs):
    return func(*args, **kwargs)

def fetch_economic_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch economic data from FRED."""
    logger.info(f"Fetching economic data from {start_date} to {end_date}")
    indicators = {
        'CPIAUCSL': 'Inflation_Rate',
        'FEDFUNDS': 'Interest_Rate',
        'DTWEXBGS': 'USD_Index',
        'GDP': 'GDP',
        'UNRATE': 'Unemployment_Rate'
    }
    data = {}
    for series_id, name in indicators.items():
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        data[name] = series
    return pd.DataFrame(data)

def fetch_sentiment_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch and analyze sentiment from financial news."""
    logger.info(f"Fetching sentiment data from {start_date} to {end_date}")
    articles = rate_limited_newsapi_call(
        newsapi.get_everything,
        q='gold OR "precious metals"',
        from_param=start_date,
        to=end_date,
        language='en',
        sort_by='publishedAt'
    )
    
    sentiments = []
    dates = []
    for article in articles['articles']:
        sentiment = TextBlob(article['title'] + ' ' + article['description']).sentiment.polarity
        sentiments.append(sentiment)
        dates.append(article['publishedAt'])
    
    sentiment_df = pd.DataFrame({'Date': dates, 'Sentiment': sentiments})
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
    return sentiment_df.groupby('Date').mean().reset_index()

def fetch_geopolitical_events(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch major geopolitical events."""
    logger.info(f"Fetching geopolitical events from {start_date} to {end_date}")
    events = rate_limited_newsapi_call(
        newsapi.get_everything,
        q='geopolitical OR war OR "trade conflict"',
        from_param=start_date,
        to=end_date,
        language='en',
        sort_by='relevancy'
    )
    
    event_data = []
    for event in events['articles']:
        event_data.append({
            'Date': event['publishedAt'],
            'Event': event['title'],
            'Impact': TextBlob(event['title'] + ' ' + event['description']).sentiment.polarity
        })
    
    event_df = pd.DataFrame(event_data)
    event_df['Date'] = pd.to_datetime(event_df['Date']).dt.date
    return event_df.groupby('Date').agg({'Event': lambda x: ', '.join(x), 'Impact': 'mean'}).reset_index()

# ... (rest of the functions remain the same)

def fetch_data_for_model(days: int = 30):
    """Fetch all necessary data for the model using a rolling window."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    gold_data = yf.download("GC=F", start=start_date, end=end_date)
    economic_data = fetch_economic_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    sentiment_data = fetch_sentiment_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    geopolitical_events = fetch_geopolitical_events(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    related_assets = {
        'Silver': yf.download("SI=F", start=start_date, end=end_date),
        'Oil': yf.download("CL=F", start=start_date, end=end_date),
        'SP500': yf.download("^GSPC", start=start_date, end=end_date)
    }
    
    return gold_data, economic_data, sentiment_data, geopolitical_events, related_assets

if __name__ == "__main__":
    # Check for NEWS_API_KEY
    if 'NEWS_API_KEY' not in os.environ:
        logger.error("NEWS_API_KEY environment variable is not set. Please set it before running the script.")
        exit(1)

    # Set up distributed computing
    client = Client()  # Set up Dask client for distributed computing
    
    # Fetch data for the last 30 days
    gold_data, economic_data, sentiment_data, geopolitical_events, related_assets = fetch_data_for_model(30)
    
    # Create features
    df = create_features(gold_data, economic_data, sentiment_data, related_assets, geopolitical_events)
    
    # Use Dask for large dataframes
    ddf = dd.from_pandas(df, npartitions=4)
    
    # Prepare features and target
    X = ddf.drop('Close', axis=1)
    y = ddf['Close']
    
    # Robust scaling
    scaler = RobustScaler()
    X_scaled = ddf.map_partitions(lambda df: pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index))
    
    # Detect anomalies
    anomalies = X_scaled.map_partitions(detect_anomalies).compute()
    logger.info(f"Detected {sum(anomalies == -1)} anomalies")
    
    # Feature selection
    rfe = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=30)
    X_selected = X_scaled.map_partitions(lambda df: pd.DataFrame(rfe.fit_transform(df, y), columns=df.columns[rfe.support_], index=df.index))
    
    # Split data
    train_size = int(len(df) * 0.8)
    X_train, X_test = X_selected.iloc[:train_size], X_selected.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Optimize and train models in parallel
    model_names = ['rf', 'xgb', 'lgbm', 'elasticnet']
    best_params = optimize_hyperparameters_parallel(X_train.compute(), y_train.compute(), model_names)
    
    with Pool(processes=cpu_count()) as pool:
        models = pool.map(train_model_parallel, [(X_train.compute(), y_train.compute(), model_name, params) for model_name, params in best_params.items()])
    
    # Create and train stacking ensemble
    stacking_model = create_stacking_ensemble(list(zip(model_names, models)))
    stacking_model.fit(X_train.compute(), y_train.compute())
    
    # Train LSTM model
    X_lstm_train, y_lstm_train = prepare_lstm_data(X_train.compute(), y_train.compute())
    lstm_model = train_lstm(X_lstm_train, y_lstm_train)
    
    # Make predictions
    stacking_predictions = stacking_model.predict(X_test.compute())
    X_lstm_test, _ = prepare_lstm_data(X_test.compute(), y_test.compute())
    lstm_predictions = lstm_model.predict(X_lstm_test).flatten()
    
    # Combine predictions (simple average)
    combined_predictions = (stacking_predictions + lstm_predictions) / 2
    
    # Calculate prediction intervals
    prediction_intervals = calculate_prediction_intervals(combined_predictions, y_test.compute()[-len(combined_predictions):])
    
    # Evaluate models
    logger.info("Stacking Ensemble Performance:")
    evaluate_model(y_test.compute(), stacking_predictions)
    logger.info("LSTM Model Performance:")
    evaluate_model(y_test.compute()[-len(lstm_predictions):], lstm_predictions)
    logger.info("Combined Model Performance:")
    evaluate_model(y_test.compute()[-len(combined_predictions):], combined_predictions)
    
    # Monitor performance
    monitor_performance(y_test.compute()[-len(combined_predictions):], combined_predictions, 'Combined_Model')
    
    # Save checkpoint
    checkpoint_data = {
        'stacking_model': stacking_model,
        'lstm_model': lstm_model,
        'scaler': scaler,
        'rfe': rfe
    }
    save_checkpoint(checkpoint_data, 'model_checkpoint.joblib')
    
    # Schedule retraining
    import threading
    retraining_thread = threading.Thread(target=schedule_retraining)
    retraining_thread.start()
    
    logger.info("Analysis complete!")

print("\nNote: This script now implements rate limiting for News API calls and uses a rolling 30-day window for data. Please ensure you have the 'ratelimit' library installed (pip install ratelimit) before running the script.")