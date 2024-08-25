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

# API keys
ALPHA_VANTAGE_API_KEY = "PIFHGHQNBWL37L0T"
CURRENTS_API_KEY = "IEfpA5hCrH6Xh4E9f7R0jEOHcEjxSI8k6s71NwcYXRPqtohR"
EXCHANGE_RATES_API_KEY = "977aa5b8e6b88d6e1d0c82ce1aabe665"
FMP_API_KEY = "3667d5a89d6b4d5a66cc0301258ba80c"  # Keeping this for economic calendar data

def fetch_all_data(start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*2)
    if end_date is None:
        end_date = datetime.now()

    gold = yf.Ticker("GC=F")
    gold_data = gold.history(start=start_date, end=end_date)

    # Fetch forex data from exchangeratesapi.io
    fx_data = fetch_forex_data(start_date, end_date)

    return {
        'Gold': gold_data['Close'],
        'EUR/USD': fx_data['EUR'],
        'GBP/USD': fx_data['GBP'],
        'JPY/USD': 1 / fx_data['JPY'],
        'Additional_FX': fx_data
    }

def fetch_forex_data(start_date, end_date):
    base_url = "http://api.exchangeratesapi.io/v1/"
    params = {
        "access_key": EXCHANGE_RATES_API_KEY,
        "base": "USD",
        "symbols": "EUR,GBP,JPY"
    }
    
    forex_data = {}
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        response = requests.get(f"{base_url}{date_str}", params=params)
        if response.status_code == 200:
            data = response.json()
            forex_data[date_str] = data['rates']
        else:
            logging.error(f"Failed to fetch forex data for {date_str}: {response.status_code}")
        current_date += timedelta(days=1)
    
    return pd.DataFrame.from_dict(forex_data, orient='index')

def fetch_economic_indicators(start_date, end_date):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    fx = ForeignExchange(key=ALPHA_VANTAGE_API_KEY)

    # Fetch GDP growth rate (quarterly)
    gdp_data, _ = ts.get_global_quote('GDP')
    
    # Fetch inflation rate (monthly)
    cpi_data, _ = ts.get_global_quote('CPI')
    
    # Fetch interest rates (daily)
    interest_rate_data, _ = fx.get_currency_exchange_daily('USD', 'EUR')  # Using EUR/USD as a proxy for interest rate differentials

    # Combine and resample data to daily frequency
    indicators = pd.concat([gdp_data, cpi_data, interest_rate_data], axis=1)
    indicators = indicators.resample('D').ffill()
    
    # Fetch economic calendar data
    calendar_data = fetch_economic_calendar(start_date, end_date)
    indicators = indicators.join(calendar_data)

    # Trim to the specified date range
    indicators = indicators.loc[start_date:end_date]

    return indicators

def fetch_economic_calendar(start_date, end_date):
    url = f"https://financialmodelingprep.com/api/v3/economic_calendar"
    params = {
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "apikey": FMP_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['impact'] = df['impact'].map({'High': 3, 'Medium': 2, 'Low': 1})
        return df.groupby(df.index).agg({
            'impact': 'sum',
            'actual': 'count'
        }).rename(columns={'actual': 'event_count'})
    else:
        logging.error(f"Failed to fetch economic calendar: {response.status_code}")
        return pd.DataFrame()

def fetch_news_sentiment(start_date, end_date):
    # Use Currents API to fetch relevant news articles
    url = f"https://api.currentsapi.services/v1/search"
    params = {
        "keywords": "forex,gold,currency,economy",
        "language": "en",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "apiKey": CURRENTS_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
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
    else:
        logging.error(f"Failed to fetch news sentiment: {response.status_code}")
        return pd.DataFrame()

# The rest of the functions (add_advanced_features, prepare_data, train_model, make_predictions, update_actual_values, run_predictor) remain the same

if __name__ == "__main__":
    run_predictor()