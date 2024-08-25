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
from alpha_vantage.cryptocurrencies import CryptoCurrencies

nltk.download('vader_lexicon', quiet=True)

logging.basicConfig(filename='gold_forex_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# API keys
ALPHA_VANTAGE_API_KEY = "PIFHGHQNBWL37L0T"
CURRENTS_API_KEY = "IEfpA5hCrH6Xh4E9f7R0jEOHcEjxSI8k6s71NwcYXRPqtohR"
EXCHANGE_RATES_API_KEY = "977aa5b8e6b88d6e1d0c82ce1aabe665"

def fetch_all_data(start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*2)
    if end_date is None:
        end_date = datetime.now()

    gold = yf.Ticker("GC=F")
    gold_data = gold.history(start=start_date, end=end_date)

    # Fetch forex data from Alpha Vantage
    fx_data = fetch_forex_data(start_date, end_date)

    # Fetch additional economic data from Alpha Vantage
    economic_data = fetch_economic_data()

    # Fetch crypto data as an additional indicator
    crypto_data = fetch_crypto_data()

    return {
        'Gold': gold_data['Close'],
        'Forex': fx_data,
        'Economic': economic_data,
        'Crypto': crypto_data
    }

def fetch_forex_data(start_date, end_date):
    fx = ForeignExchange(key=ALPHA_VANTAGE_API_KEY)
    currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    fx_data = {}

    for pair in currency_pairs:
        data, _ = fx.get_currency_exchange_daily(from_symbol=pair[:3], to_symbol=pair[3:], outputsize='full')
        fx_data[pair] = data['4. close'].astype(float)
        fx_data[pair].index = pd.to_datetime(fx_data[pair].index)
        fx_data[pair] = fx_data[pair].loc[start_date:end_date]

    return pd.DataFrame(fx_data)

def fetch_economic_data():
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    indicators = ['GDP', 'REAL_GDP', 'CPI', 'INFLATION', 'RETAIL_SALES', 'DURABLES', 'UNEMPLOYMENT', 'NONFARM_PAYROLL']
    economic_data = {}

    for indicator in indicators:
        data, _ = ts.get_economic_indicator(indicator)
        economic_data[indicator] = data['value'].astype(float)
        economic_data[indicator].index = pd.to_datetime(economic_data[indicator].index)

    return pd.DataFrame(economic_data)

def fetch_crypto_data():
    cc = CryptoCurrencies(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    cryptos = ['BTC', 'ETH']
    crypto_data = {}

    for crypto in cryptos:
        data, _ = cc.get_digital_currency_daily(symbol=crypto, market='USD')
        crypto_data[crypto] = data['4a. close (USD)'].astype(float)
        crypto_data[crypto].index = pd.to_datetime(crypto_data[crypto].index)

    return pd.DataFrame(crypto_data)

def fetch_news_sentiment(start_date, end_date):
    # Use Currents API to fetch relevant news articles
    url = f"https://api.currentsapi.services/v1/search"
    params = {
        "keywords": "forex,gold,currency,economy,cryptocurrency",
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

def add_advanced_features(df):
    # Add technical indicators
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    rsi = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi.rsi()
    
    bollinger = BollingerBands(close=df['Close'])
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    
    # Add lagged features
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_7'] = df['Close'].shift(7)
    
    # Add return features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    return df

def prepare_data(data, news_sentiment):
    df = pd.DataFrame(data['Gold'])
    df = df.join(data['Forex'])
    df = df.join(data['Economic'])
    df = df.join(data['Crypto'])
    df = df.join(news_sentiment)
    
    df = add_advanced_features(df)
    df = df.dropna()
    
    return df

# The rest of the functions (train_model, make_predictions, update_actual_values, run_predictor) remain the same

if __name__ == "__main__":
    run_predictor()