import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

# Add the Currents API key
CURRENTS_API_KEY = "FkEEwNLACnLEfCoJ09fFe3yrVaPGZ76u-PKi8-yHqGRJ6hd8"

# Add an Economic Calendar API key (you'll need to sign up for a service)
ECONOMIC_CALENDAR_API_KEY = "your_economic_calendar_api_key_here"

def fetch_data(start_date, end_date):
    # ... (previous fetch_data function remains unchanged)

def fetch_news_sentiment(start_date, end_date, max_requests=600):
    # ... (previous fetch_news_sentiment function remains unchanged)

def fetch_recent_news(days=7):
    """Fetch recent news articles related to gold."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.currentsapi.services/v1/search?keywords=gold&language=en&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&apiKey={CURRENTS_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get('news', [])
    else:
        print(f"Failed to fetch recent news")
        return []

def fetch_economic_calendar(start_date, end_date):
    """Fetch economic calendar events."""
    # Note: You'll need to replace this with an actual economic calendar API
    # This is a placeholder function
    url = f"https://api.example.com/economic_calendar?start_date={start_date}&end_date={end_date}&api_key={ECONOMIC_CALENDAR_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        calendar_data = response.json()
        return calendar_data
    else:
        print(f"Failed to fetch economic calendar data")
        return []

def create_features(df, recent_news, economic_calendar):
    # ... (previous feature creation code)

    # Add features based on recent news
    recent_news_sentiment = np.mean([TextBlob(article['title'] + ' ' + article['description']).sentiment.polarity for article in recent_news])
    df['Recent_News_Sentiment'] = recent_news_sentiment

    # Add features based on economic calendar
    # This is a placeholder - you'll need to adapt this based on the actual economic calendar data structure
    important_events = ['Fed Rate Decision', 'GDP Release', 'Inflation Report']
    for event in important_events:
        df[f'{event}_Upcoming'] = economic_calendar.get(event, 0)

    df.dropna(inplace=True)
    return df

def train_model(model_name, X_train, y_train, params=None):
    # ... (previous train_model function remains unchanged)

def predict(model, model_name, X):
    # ... (previous predict function remains unchanged)

def evaluate_model(y_true, y_pred):
    # ... (previous evaluate_model function remains unchanged)

def plot_predictions(y_true, y_pred, title):
    # ... (previous plot_predictions function remains unchanged)

def ensemble_predict(predictions):
    # ... (previous ensemble_predict function remains unchanged)

if __name__ == "__main__":
    # Fetch and prepare data
    df = fetch_data("2010-01-01", "2023-05-31")
    news_sentiment = fetch_news_sentiment("2010-01-01", "2023-05-31")
    df['News_Sentiment'] = news_sentiment
    
    # Fetch recent news and economic calendar data
    recent_news = fetch_recent_news()
    economic_calendar = fetch_economic_calendar("2023-05-31", "2023-06-07")  # Fetch one week ahead
    
    df = create_features(df, recent_news, economic_calendar)
    
    # ... (rest of the main block remains unchanged)

    # After making predictions, analyze recent news and upcoming economic events
    print("\nRecent News Analysis:")
    for article in recent_news[:5]:  # Print top 5 recent news articles
        print(f"Title: {article['title']}")
        print(f"Sentiment: {TextBlob(article['title'] + ' ' + article['description']).sentiment.polarity}")
        print("---")

    print("\nUpcoming Economic Events:")
    for event, value in economic_calendar.items():
        print(f"{event}: {value}")

print("\nNote: This ultra-advanced model now incorporates historical and recent news sentiment data, as well as economic calendar information. However, it should still be used cautiously for actual trading decisions.")