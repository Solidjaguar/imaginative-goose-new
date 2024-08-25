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
from typing import List, Dict, Tuple, Any
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
from scipy.stats import norm

# Existing functions (fetch_economic_data, fetch_sentiment_data, etc.) remain unchanged

def fetch_all_data(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Fetch all necessary data for the model."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=config['lookback_days'])
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    gold_data = yf.download(config['symbols']['gold'], start=start_date_str, end=end_date_str)
    economic_data = fetch_economic_data(start_date_str, end_date_str)
    sentiment_data = fetch_sentiment_data(start_date_str, end_date_str)
    geopolitical_events = fetch_geopolitical_events(start_date_str, end_date_str)
    
    related_assets = {}
    for asset, symbol in config['symbols']['related_assets'].items():
        related_assets[asset] = yf.download(symbol, start=start_date_str, end=end_date_str)
    
    return {
        'gold': gold_data,
        'economic': economic_data,
        'sentiment': sentiment_data,
        'geopolitical': geopolitical_events,
        'related_assets': related_assets
    }

def prepare_data(data: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare data for the gold price prediction model."""
    df = create_features(data['gold'], data['economic'], data['sentiment'], data['related_assets'], data['geopolitical'])
    
    X = df.drop('Close', axis=1)
    y = df['Close']
    
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    anomalies = detect_anomalies(X_scaled)
    logger.info(f"Detected {sum(anomalies == -1)} anomalies")
    
    rfe = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=config['feature_selection']['n_features'])
    X_selected = pd.DataFrame(rfe.fit_transform(X_scaled, y), columns=X_scaled.columns[rfe.support_], index=X_scaled.index)
    
    train_size = int(len(df) * config['train_test_split'])
    X_train, X_test = X_selected.iloc[:train_size], X_selected.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train ensemble and LSTM models."""
    best_params = optimize_hyperparameters_parallel(X_train, y_train, config['model_names'])
    
    with Pool(processes=cpu_count()) as pool:
        models = pool.map(train_model_parallel, [(X_train, y_train, model_name, params) for model_name, params in best_params.items()])
    
    stacking_model = create_stacking_ensemble(list(zip(config['model_names'], models)))
    stacking_model.fit(X_train, y_train)
    
    X_lstm_train, y_lstm_train = prepare_lstm_data(X_train, y_train)
    lstm_model = train_lstm(X_lstm_train, y_lstm_train)
    
    return {
        'stacking_model': stacking_model,
        'lstm_model': lstm_model,
        'best_params': best_params
    }

def make_predictions(models: Dict[str, Any], X_test: pd.DataFrame) -> np.ndarray:
    """Make predictions using trained models."""
    stacking_predictions = models['stacking_model'].predict(X_test)
    X_lstm_test, _ = prepare_lstm_data(X_test, pd.Series(np.zeros(len(X_test))))
    lstm_predictions = models['lstm_model'].predict(X_lstm_test).flatten()
    
    # Combine predictions (simple average)
    return (stacking_predictions + lstm_predictions[:len(stacking_predictions)]) / 2

def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate model predictions."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def save_predictions(y_true: pd.Series, y_pred: np.ndarray, config: Dict[str, Any]) -> None:
    """Save predictions to a file."""
    results = pd.DataFrame({
        'true_values': y_true,
        'predictions': y_pred
    })
    results.to_csv(config['paths']['predictions'], index=True)

def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, config: Dict[str, Any]) -> None:
    """Plot true values vs predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label='True Values')
    plt.plot(y_true.index, y_pred, label='Predictions')
    plt.title('Gold Price: True Values vs Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(config['paths']['plots'] + 'predictions_plot.png')
    plt.close()

def plot_feature_importance(model: StackingRegressor, feature_names: List[str]) -> None:
    """Plot feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(config['paths']['plots'] + 'feature_importance_plot.png')
    plt.close()

def plot_correlation_matrix(data: pd.DataFrame) -> None:
    """Plot correlation matrix."""
    corr = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(config['paths']['plots'] + 'correlation_matrix.png')
    plt.close()

# Main execution remains unchanged