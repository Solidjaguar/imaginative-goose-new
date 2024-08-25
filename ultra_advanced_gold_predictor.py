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

# API keys
CURRENTS_API_KEY = "FkEEwNLACnLEfCoJ09fFe3yrVaPGZ76u-PKi8-yHqGRJ6hd8"
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your actual Alpha Vantage API key

def fetch_data(start_date, end_date):
    # ... (previous fetch_data function remains unchanged)

def fetch_news_sentiment(start_date, end_date, max_requests=600):
    # ... (previous fetch_news_sentiment function remains unchanged)

def fetch_recent_news(days=7):
    # ... (previous fetch_recent_news function remains unchanged)

def fetch_economic_calendar(start_date, end_date):
    """Fetch economic calendar events from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=ECONOMIC_CALENDAR&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        calendar_data = response.json()
        if 'error' in calendar_data:
            print(f"Error fetching economic calendar: {calendar_data['error']}")
            return []
        
        # Filter events within the specified date range
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        filtered_events = [
            event for event in calendar_data.get('economic_calendar', [])
            if start_date <= datetime.strptime(event['date'], "%Y-%m-%d") <= end_date
        ]
        return filtered_events
    else:
        print(f"Failed to fetch economic calendar data")
        return []

def create_features(df, recent_news, economic_calendar):
    # ... (previous feature creation code)

    # Add features based on recent news
    recent_news_sentiment = np.mean([TextBlob(article['title'] + ' ' + article['description']).sentiment.polarity for article in recent_news])
    df['Recent_News_Sentiment'] = recent_news_sentiment

    # Add features based on economic calendar
    important_countries = ['United States', 'China', 'European Union', 'Japan']
    important_events = ['GDP', 'Inflation Rate', 'Interest Rate Decision', 'Non-Farm Payrolls']
    
    for country in important_countries:
        for event in important_events:
            event_count = sum(1 for e in economic_calendar if e['country'] == country and event.lower() in e['event'].lower())
            df[f'{country}_{event}_Count'] = event_count

    # Add a feature for the overall number of high-impact events
    high_impact_count = sum(1 for e in economic_calendar if e['impact'] == 'High')
    df['High_Impact_Events_Count'] = high_impact_count

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
    for event in economic_calendar[:10]:  # Print top 10 upcoming events
        print(f"Date: {event['date']}")
        print(f"Country: {event['country']}")
        print(f"Event: {event['event']}")
        print(f"Impact: {event['impact']}")
        print("---")

print("\nNote: This ultra-advanced model now incorporates historical and recent news sentiment data, as well as economic calendar information from Alpha Vantage. However, it should still be used cautiously for actual trading decisions.")