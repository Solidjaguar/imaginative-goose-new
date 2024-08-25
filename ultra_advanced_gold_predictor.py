import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import requests
from textblob import TextBlob
import talib
from datetime import datetime, timedelta
import time
import joblib
from collections import defaultdict
import multiprocessing
from river import linear_model, preprocessing, compose
import json
from scipy import stats
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import shap
from pyod.models.iforest import IForest
import sqlite3

# ... (previous code remains unchanged)

def analyze_past_predictions(window_days=365):
    """Analyze past predictions to improve the model, using a moving window."""
    conn = sqlite3.connect('predictions.db')
    current_date = datetime.now().date()
    window_start = current_date - timedelta(days=window_days)
    
    query = f"""
    SELECT * FROM predictions 
    WHERE actual_price IS NOT NULL 
    AND date >= '{window_start}'
    ORDER BY date DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        mse = mean_squared_error(df['actual_price'], df['predicted_price'])
        mae = mean_absolute_error(df['actual_price'], df['predicted_price'])
        r2 = r2_score(df['actual_price'], df['predicted_price'])
        
        print(f"Historical Model Performance (Last {window_days} days):")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Analyze error patterns
        df['error'] = df['actual_price'] - df['predicted_price']
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['error'])
        plt.title(f"Prediction Error Over Time (Last {window_days} days)")
        plt.xlabel("Date")
        plt.ylabel("Error")
        plt.show()
        
        return df
    else:
        print("No historical predictions available for analysis.")
        return None

def adjust_model_based_on_history(model, historical_data, max_adjustment_percent=5):
    """Adjust the model based on historical prediction performance with safeguards."""
    if historical_data is not None and len(historical_data) > 0:
        # Calculate weighted mean error with exponential decay
        days = (datetime.now().date() - historical_data['date'].min().date()).days
        weights = np.exp(-np.arange(len(historical_data)) / (days / 2))  # Half-life of half the period
        weighted_mean_error = np.average(historical_data['error'], weights=weights)
        
        # Cap the adjustment
        max_adjustment = np.mean(historical_data['actual_price']) * (max_adjustment_percent / 100)
        capped_adjustment = np.clip(weighted_mean_error, -max_adjustment, max_adjustment)
        
        print(f"Applied adjustment: {capped_adjustment:.4f}")
        
        def adjusted_predict(X):
            original_prediction = model.predict(X)
            return original_prediction + capped_adjustment
        
        return adjusted_predict
    else:
        return model.predict

if __name__ == "__main__":
    # ... (previous main code)
    
    # Analyze past predictions with a 365-day window
    historical_data = analyze_past_predictions(window_days=365)
    
    # Adjust model based on historical performance, with a 5% max adjustment
    adjusted_predict = adjust_model_based_on_history(stacking_model, historical_data, max_adjustment_percent=5)
    
    # Make new predictions using the adjusted model
    adjusted_predictions = adjusted_predict(X_test)
    
    print("\nAdjusted Model Performance:")
    evaluate_model(y_test, adjusted_predictions)
    
    # ... (rest of the main code)

print("\nNote: This ultra-advanced model now incorporates learning from past predictions with safeguards against overfitting to recent history. However, it should still be used cautiously for actual trading decisions.")