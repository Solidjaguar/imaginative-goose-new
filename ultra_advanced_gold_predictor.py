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

def fetch_data(start_date, end_date):
    # ... (previous fetch_data function remains unchanged)

def fetch_news_sentiment(start_date, end_date, max_requests=600):
    sentiments = []
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    total_days = (end_date - start_date).days + 1
    requests_per_day = max(1, min(max_requests // total_days, 1))  # Ensure at least 1 request per day
    days_to_skip = max(1, total_days // max_requests)  # Skip days if necessary
    
    current_date = start_date
    requests_made = 0
    
    while current_date <= end_date and requests_made < max_requests:
        # Format the date for the API request
        formatted_date = current_date.strftime("%Y-%m-%d")
        
        # Make API request
        url = f"https://api.currentsapi.services/v1/search?keywords=gold&language=en&start_date={formatted_date}&end_date={formatted_date}&apiKey={CURRENTS_API_KEY}"
        response = requests.get(url)
        requests_made += 1
        
        if response.status_code == 200:
            news_data = response.json()
            
            # Calculate average sentiment for the day
            daily_sentiment = 0
            for article in news_data.get('news', []):
                title = article.get('title', '')
                description = article.get('description', '')
                full_text = title + ' ' + description
                sentiment = TextBlob(full_text).sentiment.polarity
                daily_sentiment += sentiment
            
            if news_data.get('news'):
                daily_sentiment /= len(news_data['news'])
            
            sentiments.append((current_date, daily_sentiment))
        else:
            print(f"Failed to fetch news for {formatted_date}")
            sentiments.append((current_date, 0))  # Neutral sentiment if fetch fails
        
        # Move to the next date, skipping days if necessary
        current_date += timedelta(days=days_to_skip)
        
        # Sleep to avoid hitting rate limits
        time.sleep(1)
    
    # Create a complete date range and fill in missing dates with interpolation
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment_series = pd.Series(dict(sentiments))
    sentiment_series = sentiment_series.reindex(date_range).interpolate()
    
    return sentiment_series

def create_features(df):
    # ... (previous create_features function remains unchanged)

# ... (rest of the script remains unchanged)

if __name__ == "__main__":
    # Fetch and prepare data
    df = fetch_data("2010-01-01", "2023-05-31")
    news_sentiment = fetch_news_sentiment("2010-01-01", "2023-05-31")
    df['News_Sentiment'] = news_sentiment
    df = create_features(df)
    
    # ... (rest of the main block remains unchanged)

print("\nNote: This ultra-advanced model now incorporates real news sentiment data from the Currents API, respecting the daily request limit, but should still be used cautiously for actual trading decisions.")