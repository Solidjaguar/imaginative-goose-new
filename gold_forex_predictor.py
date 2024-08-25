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

    # Trim to the specified date range
    indicators = indicators.loc[start_date:end_date]

    return indicators

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

def prepare_data(data, economic_indicators, news_sentiment):
    df = pd.DataFrame(data['Gold'])
    df = df.join(economic_indicators)
    df = df.join(news_sentiment)
    
    # Add forex data
    for currency, values in data.items():
        if currency != 'Gold':
            df[currency] = values
    
    df = add_advanced_features(df)
    df = df.dropna()
    
    return df

def train_model():
    start_date = datetime.now() - timedelta(days=365*5)  # 5 years of data
    end_date = datetime.now()
    
    data = fetch_all_data(start_date, end_date)
    economic_indicators = fetch_economic_indicators(start_date, end_date)
    news_sentiment = fetch_news_sentiment(start_date, end_date)
    
    df = prepare_data(data, economic_indicators, news_sentiment)
    
    # Prepare features and target
    features = df.drop(['Close'], axis=1)
    target = df['Close']
    
    # Split the data
    split = int(len(df) * 0.8)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = target[:split], target[split:]
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model (using a simple model for demonstration)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    logging.info(f"Model trained. Train score: {train_score:.4f}, Test score: {test_score:.4f}")
    
    return model, scaler

def make_predictions(model, scaler):
    # Fetch the latest data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Fetch last 30 days of data
    
    data = fetch_all_data(start_date, end_date)
    economic_indicators = fetch_economic_indicators(start_date, end_date)
    news_sentiment = fetch_news_sentiment(start_date, end_date)
    
    df = prepare_data(data, economic_indicators, news_sentiment)
    
    # Prepare features
    features = df.drop(['Close'], axis=1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    # Create a DataFrame with dates and predictions
    prediction_df = pd.DataFrame({
        'Date': df.index,
        'Predicted_Price': predictions,
        'Actual_Price': df['Close']
    })
    
    logging.info(f"Predictions made for dates: {prediction_df['Date'].min()} to {prediction_df['Date'].max()}")
    
    return prediction_df

def update_actual_values():
    # This function would typically update a database or file with actual observed values
    # For demonstration, we'll just log a message
    logging.info("Actual values would be updated here.")

def run_predictor():
    model, scaler = train_model()
    predictions = make_predictions(model, scaler)
    update_actual_values()
    
    # Log some sample predictions
    logging.info(f"Sample predictions:\n{predictions.tail().to_string()}")
    
    # Here you could save the predictions to a file or database
    predictions.to_csv('latest_predictions.csv', index=False)
    logging.info("Predictions saved to 'latest_predictions.csv'")

if __name__ == "__main__":
    run_predictor()