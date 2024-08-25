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
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange

nltk.download('vader_lexicon', quiet=True)

logging.basicConfig(filename='gold_forex_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your actual API keys
ALPHA_VANTAGE_API_KEY = "PIFHGHQNBWL37L0T"
CURRENTS_API_KEY = "YOUR_CURRENTS_API_KEY"

def fetch_all_data(start_date=None, end_date=None):
    # ... (rest of the function remains the same)

def fetch_economic_indicators(start_date, end_date):
    # ... (rest of the function remains the same)

def fetch_news_sentiment(start_date, end_date):
    # Use Currents API to fetch relevant news articles
    url = f"https://api.currentsapi.services/v1/search?keywords=forex,gold&language=en&apiKey={CURRENTS_API_KEY}"
    response = requests.get(url)
    news_data = response.json()

    # Perform sentiment analysis on the news articles
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    dates = []

    for article in news_data['news']:
        sentiment = sia.polarity_scores(article['title'] + ' ' + article['description'])
        sentiments.append(sentiment['compound'])
        dates.append(article['published'][:10])  # Extract date from datetime string

    sentiment_df = pd.DataFrame({'date': dates, 'sentiment': sentiments})
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df = sentiment_df.groupby('date').mean()

    # Reindex to include all dates in the range
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment_df = sentiment_df.reindex(date_range)
    sentiment_df = sentiment_df.fillna(method='ffill')  # Forward fill missing values

    return sentiment_df

def add_advanced_features(df):
    # ... (rest of the function remains the same)

def prepare_data(data, economic_indicators, news_sentiment):
    # ... (rest of the function remains the same)

def train_model():
    # ... (rest of the function remains the same)

def make_predictions(model):
    # ... (rest of the function remains the same)

def update_actual_values():
    # ... (rest of the function remains the same)

def run_predictor():
    model = train_model()
    make_predictions(model)
    update_actual_values()

if __name__ == "__main__":
    run_predictor()