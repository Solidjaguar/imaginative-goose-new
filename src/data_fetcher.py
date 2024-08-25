import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

def fetch_all_data(config):
    start_date = datetime.now() - timedelta(days=365*config['data']['lookback_years'])
    end_date = datetime.now()

    gold_data = fetch_gold_data(start_date, end_date)
    forex_data = fetch_forex_data(config, start_date, end_date)
    economic_data = fetch_economic_data(config)
    crypto_data = fetch_crypto_data(config)
    news_sentiment = fetch_news_sentiment(config, start_date, end_date)

    return {
        'Gold': gold_data,
        'Forex': forex_data,
        'Economic': economic_data,
        'Crypto': crypto_data,
        'Sentiment': news_sentiment
    }

def fetch_gold_data(start_date, end_date):
    gold = yf.Ticker("GC=F")
    gold_data = gold.history(start=start_date, end=end_date)
    return gold_data['Close']

def fetch_forex_data(config, start_date, end_date):
    fx = ForeignExchange(key=config['api_keys']['alpha_vantage'])
    currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    fx_data = {}

    for pair in currency_pairs:
        data, _ = fx.get_currency_exchange_daily(from_symbol=pair[:3], to_symbol=pair[3:], outputsize='full')
        fx_data[pair] = data['4. close'].astype(float)
        fx_data[pair].index = pd.to_datetime(fx_data[pair].index)
        fx_data[pair] = fx_data[pair].loc[start_date:end_date]

    return pd.DataFrame(fx_data)

def fetch_economic_data(config):
    ts = TimeSeries(key=config['api_keys']['alpha_vantage'], output_format='pandas')
    indicators = ['GDP', 'REAL_GDP', 'CPI', 'INFLATION', 'RETAIL_SALES', 'DURABLES', 'UNEMPLOYMENT', 'NONFARM_PAYROLL']
    economic_data = {}

    for indicator in indicators:
        data, _ = ts.get_economic_indicator(indicator)
        economic_data[indicator] = data['value'].astype(float)
        economic_data[indicator].index = pd.to_datetime(economic_data[indicator].index)

    return pd.DataFrame(economic_data)

def fetch_crypto_data(config):
    cc = CryptoCurrencies(key=config['api_keys']['alpha_vantage'], output_format='pandas')
    cryptos = ['BTC', 'ETH']
    crypto_data = {}

    for crypto in cryptos:
        data, _ = cc.get_digital_currency_daily(symbol=crypto, market='USD')
        crypto_data[crypto] = data['4a. close (USD)'].astype(float)
        crypto_data[crypto].index = pd.to_datetime(crypto_data[crypto].index)

    return pd.DataFrame(crypto_data)

def fetch_news_sentiment(config, start_date, end_date):
    url = f"https://api.currentsapi.services/v1/search"
    params = {
        "keywords": "forex,gold,currency,economy,cryptocurrency",
        "language": "en",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "apiKey": config['api_keys']['currents']
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        news_data = response.json()
        sia = SentimentIntensityAnalyzer()
        sentiments = []
        dates = []

        for article in news_data['news']:
            sentiment = sia.polarity_scores(article['title'] + ' ' + article['description'])
            sentiments.append(sentiment['compound'])
            dates.append(article['published'][:10])

        sentiment_df = pd.DataFrame({'date': dates, 'sentiment': sentiments})
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df = sentiment_df.groupby('date').mean()

        date_range = pd.date_range(start=start_date, end=end_date)
        sentiment_df = sentiment_df.reindex(date_range)
        sentiment_df = sentiment_df.fillna(method='ffill')

        return sentiment_df
    else:
        raise Exception(f"Failed to fetch news sentiment: {response.status_code}")