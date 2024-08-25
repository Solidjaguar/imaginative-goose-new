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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
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
import json
import hashlib
import functools
from config import *

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize APIs
fred = fredapi.Fred(api_key=FRED_API_KEY)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Caching decorator
def cache_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = hashlib.md5(f"{func.__name__}{str(args)}{str(kwargs)}".encode()).hexdigest()
        cache_file = f"{CACHE_DIR}/{key}.json"
        
        if os.path.exists(cache_file) and time.time() - os.path.getmtime(cache_file) < CACHE_EXPIRY:
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        result = func(*args, **kwargs)
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        
        return result
    return wrapper

# Rate limiting decorators
@sleep_and_retry
@limits(calls=95, period=86400)  # 95 calls per day (leaving some margin)
@cache_result
def rate_limited_newsapi_call(func, *args, **kwargs):
    return func(*args, **kwargs)

def fetch_economic_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch economic data from FRED."""
    logger.info(f"Fetching economic data from {start_date} to {end_date}")
    data = {}
    for series_id, name in FRED_INDICATORS.items():
        try:
            series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            data[name] = series
        except Exception as e:
            logger.error(f"Error fetching {name} from FRED: {str(e)}")
            data[name] = pd.Series(dtype=float)  # Empty series as placeholder
    return pd.DataFrame(data)

@cache_result
def fetch_sentiment_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch and analyze sentiment from financial news."""
    logger.info(f"Fetching sentiment data from {start_date} to {end_date}")
    try:
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
    except Exception as e:
        logger.error(f"Error fetching sentiment data: {str(e)}")
        return pd.DataFrame(columns=['Date', 'Sentiment'])

@cache_result
def fetch_geopolitical_events(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch major geopolitical events."""
    logger.info(f"Fetching geopolitical events from {start_date} to {end_date}")
    try:
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
    except Exception as e:
        logger.error(f"Error fetching geopolitical events: {str(e)}")
        return pd.DataFrame(columns=['Date', 'Event', 'Impact'])

def create_features(gold_data: pd.DataFrame, economic_data: pd.DataFrame, sentiment_data: pd.DataFrame, 
                    related_assets: Dict[str, pd.DataFrame], geopolitical_events: pd.DataFrame) -> pd.DataFrame:
    """Create features for the gold price prediction model."""
    df = gold_data.copy()
    
    # Technical indicators
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(df['Close'])
    
    # Merge economic data
    df = df.merge(economic_data, left_index=True, right_index=True, how='left')
    
    # Merge sentiment data
    df = df.merge(sentiment_data, left_index=True, right_on='Date', how='left')
    
    # Add related assets data
    for asset, asset_data in related_assets.items():
        df[f'{asset}_Close'] = asset_data['Close']
        df[f'{asset}_Volume'] = asset_data['Volume']
    
    # Add geopolitical events
    df = df.merge(geopolitical_events, left_index=True, right_on='Date', how='left')
    
    # Fill missing values
    df = df.ffill().bfill()
    
    return df

def detect_anomalies(X: pd.DataFrame) -> np.ndarray:
    """Detect anomalies in the feature set."""
    clf = IForest(contamination=ANOMALY_CONTAMINATION, random_state=42)
    return clf.fit_predict(X)

def optimize_hyperparameters_parallel(X: pd.DataFrame, y: pd.Series, model_names: List[str]) -> Dict[str, Dict]:
    """Optimize hyperparameters for multiple models in parallel."""
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(optimize_model, [(X, y, model_name) for model_name in model_names])
    return dict(zip(model_names, results))

def optimize_model(X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
    """Optimize hyperparameters for a single model."""
    if model_name == 'rf':
        param_space = {
            'n_estimators': (10, 200),
            'max_depth': (2, 30),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10)
        }
        estimator = RandomForestRegressor(random_state=42)
    elif model_name == 'xgb':
        param_space = {
            'n_estimators': (10, 200),
            'max_depth': (2, 30),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'subsample': (0.5, 1.0, 'uniform'),
            'colsample_bytree': (0.5, 1.0, 'uniform')
        }
        estimator = XGBRegressor(random_state=42)
    elif model_name == 'lgbm':
        param_space = {
            'n_estimators': (10, 200),
            'max_depth': (2, 30),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'subsample': (0.5, 1.0, 'uniform'),
            'colsample_bytree': (0.5, 1.0, 'uniform')
        }
        estimator = LGBMRegressor(random_state=42)
    elif model_name == 'elasticnet':
        param_space = {
            'alpha': (0.0001, 1.0, 'log-uniform'),
            'l1_ratio': (0.0, 1.0, 'uniform')
        }
        estimator = ElasticNet(random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    opt = BayesSearchCV(estimator, param_space, n_iter=50, cv=TimeSeriesSplit(n_splits=5), random_state=42)
    opt.fit(X, y)
    return opt.best_params_

def train_model_parallel(args: Tuple[pd.DataFrame, pd.Series, str, Dict]) -> object:
    """Train a single model with optimized hyperparameters."""
    X, y, model_name, params = args
    if model_name == 'rf':
        model = RandomForestRegressor(random_state=42, **params)
    elif model_name == 'xgb':
        model = XGBRegressor(random_state=42, **params)
    elif model_name == 'lgbm':
        model = LGBMRegressor(random_state=42, **params)
    elif model_name == 'elasticnet':
        model = ElasticNet(random_state=42, **params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    model.fit(X, y)
    return model

def create_stacking_ensemble(models: List[Tuple[str, object]]) -> StackingRegressor:
    """Create a stacking ensemble from the trained models."""
    estimators = [(name, model) for name, model in models]
    return StackingRegressor(estimators=estimators, final_estimator=ElasticNet(random_state=42))

def train_lstm(X: np.ndarray, y: np.ndarray) -> Sequential:
    """Train an LSTM model."""
    model = Sequential([
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
        Dropout(LSTM_DROPOUT),
        Dense(LSTM_DENSE_UNITS, activation='relu'),
        Dropout(LSTM_DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LSTM_LEARNING_RATE), loss='mse')
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5)
    model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=0)
    return model

def prepare_lstm_data(X: pd.DataFrame, y: pd.Series, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for LSTM model."""
    data = np.column_stack((X, y))
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), :-1])
        y.append(data[i + lookback, -1])
    return np.array(X), np.array(y)

def calculate_prediction_intervals(predictions: np.ndarray, y_true: np.ndarray, confidence: float = PREDICTION_INTERVAL_CONFIDENCE) -> np.ndarray:
    """Calculate prediction intervals."""
    errors = y_true - predictions
    std_error = np.std(errors)
    z_score = norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * std_error
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return np.column_stack((lower_bound, predictions, upper_bound))

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Evaluate model performance."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R2 Score: {r2:.4f}")

def monitor_performance(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Monitor model performance over time."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS performance
                 (timestamp TEXT, model TEXT, mse REAL, mae REAL, r2 REAL)''')
    c.execute("INSERT INTO performance VALUES (?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), model_name, mse, mae, r2))
    conn.commit()
    conn.close()

def save_checkpoint(data: Dict, filename: str = CHECKPOINT_FILE) -> None:
    """Save model checkpoint."""
    joblib.dump(data, filename)

def load_checkpoint(filename: str = CHECKPOINT_FILE) -> Dict:
    """Load model checkpoint."""
    return joblib.load(filename)

def schedule_retraining() -> None:
    """Schedule model retraining."""
    def retrain():
        logger.info("Retraining models...")
        # Add retraining logic here
    
    schedule.every().day.at(RETRAIN_TIME).do(retrain)
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Sleep for an hour between checks

def fetch_data_for_model(start_date: str, end_date: str):
    """Fetch all necessary data for the model."""
    try:
        gold_data = yf.download(GOLD_SYMBOL, start=start_date, end=end_date)
    except Exception as e:
        logger.error(f"Error fetching gold data: {str(e)}")
        gold_data = pd.DataFrame()
    
    economic_data = fetch_economic_data(start_date, end_date)
    sentiment_data = fetch_sentiment_data(start_date, end_date)
    geopolitical_events = fetch_geopolitical_events(start_date, end_date)
    
    related_assets = {}
    for asset, symbol in RELATED_ASSETS:
        try:
            related_assets[asset] = yf.download(symbol, start=start_date, end=end_date)
        except Exception as e:
            logger.error(f"Error fetching {asset} data: {str(e)}")
            related_assets[asset] = pd.DataFrame()
    
    return gold_data, economic_data, sentiment_data, geopolitical_events, related_assets

if __name__ == "__main__":
    # Check for API keys
    if not NEWS_API_KEY or not FRED_API_KEY:
        logger.error("NEWS_API_KEY or FRED_API_KEY is not set. Please set them before running the script.")
        exit(1)

    try:
        # Set up distributed computing
        client = Client()  # Set up Dask client for distributed computing
        
        # Fetch data for the last LOOKBACK_DAYS days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS)
        gold_data, economic_data, sentiment_data, geopolitical_events, related_assets = fetch_data_for_model(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Check if we have enough data to proceed
        if len(gold_data) < 5:  # Arbitrary threshold, adjust as needed
            logger.error("Not enough gold price data to proceed. Exiting.")
            exit(1)
        
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
        rfe = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=N_FEATURES_TO_SELECT)
        X_selected = X_scaled.map_partitions(lambda df: pd.DataFrame(rfe.fit_transform(df, y), columns=df.columns[rfe.support_], index=df.index))
        
        # Split data
        train_size = int(len(df) * TRAIN_TEST_SPLIT)
        X_train, X_test = X_selected.iloc[:train_size], X_selected.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Optimize and train models in parallel
        best_params = optimize_hyperparameters_parallel(X_train.compute(), y_train.compute(), MODEL_NAMES)
        
        with Pool(processes=cpu_count()) as pool:
            models = pool.map(train_model_parallel, [(X_train.compute(), y_train.compute(), model_name, params) for model_name, params in best_params.items()])
        
        # Create and train stacking ensemble
        stacking_model = create_stacking_ensemble(list(zip(MODEL_NAMES, models)))
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
        save_checkpoint(checkpoint_data)
        
        # Schedule retraining
        import threading
        retraining_thread = threading.Thread(target=schedule_retraining)
        retraining_thread.start()
        
        logger.info("Analysis complete!")
    
    except Exception as e:
        logger.error(f"An error occurred during script execution: {str(e)}")
        raise

print("\nNote: This script now uses configuration parameters from config.py and implements caching for API calls. Please ensure you have sufficient disk space for caching. The cache is stored in a 'cache' directory and is valid for 24 hours.")