import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import io
import base64
import logging
from trading_strategy import calculate_indicators
from ensemble_model import EnsembleModel, train_ensemble_model
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests

nltk.download('vader_lexicon')

logging.basicConfig(filename='gold_forex_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_all_data(start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*2)
    if end_date is None:
        end_date = datetime.now()

    gold = yf.Ticker("GC=F")
    gold_data = gold.history(start=start_date, end=end_date)

    currency_pairs = ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X']
    fx_data = {}

    for pair in currency_pairs:
        currency = yf.Ticker(pair)
        fx_data[pair] = currency.history(start=start_date, end=end_date)

    return {
        'Gold': gold_data['Close'],
        'EUR/USD': fx_data['EURUSD=X']['Close'],
        'GBP/USD': fx_data['GBPUSD=X']['Close'],
        'JPY/USD': 1 / fx_data['JPYUSD=X']['Close']
    }

def fetch_economic_indicators(start_date, end_date):
    # This is a placeholder function. In a real-world scenario, you would use an API
    # to fetch economic indicators like GDP, inflation rates, interest rates, etc.
    # For this example, we'll generate random data
    date_range = pd.date_range(start=start_date, end=end_date)
    indicators = pd.DataFrame({
        'GDP_growth': np.random.normal(2, 0.5, len(date_range)),
        'Inflation_rate': np.random.normal(2, 0.3, len(date_range)),
        'Interest_rate': np.random.normal(1, 0.2, len(date_range)),
    }, index=date_range)
    return indicators

def fetch_news_sentiment(start_date, end_date):
    # This is a placeholder function. In a real-world scenario, you would use a news API
    # to fetch relevant news articles and perform sentiment analysis
    # For this example, we'll generate random sentiment scores
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment = pd.DataFrame({
        'News_sentiment': np.random.uniform(-1, 1, len(date_range)),
    }, index=date_range)
    return sentiment

def add_advanced_features(df):
    # Technical indicators
    for column in ['Gold', 'EUR/USD', 'GBP/USD', 'JPY/USD']:
        # MACD
        macd = MACD(close=df[column])
        df[f'{column}_MACD'] = macd.macd()
        df[f'{column}_MACD_signal'] = macd.macd_signal()
        df[f'{column}_MACD_diff'] = macd.macd_diff()

        # SMA
        sma = SMAIndicator(close=df[column], window=14)
        df[f'{column}_SMA'] = sma.sma_indicator()

        # RSI
        rsi = RSIIndicator(close=df[column])
        df[f'{column}_RSI'] = rsi.rsi()

        # Bollinger Bands
        bb = BollingerBands(close=df[column])
        df[f'{column}_BB_high'] = bb.bollinger_hband()
        df[f'{column}_BB_low'] = bb.bollinger_lband()

    # Lagged features
    for column in df.columns:
        df[f'{column}_lag1'] = df[column].shift(1)
        df[f'{column}_lag2'] = df[column].shift(2)

    # Return features
    for column in ['Gold', 'EUR/USD', 'GBP/USD', 'JPY/USD']:
        df[f'{column}_return'] = df[column].pct_change()

    return df

def prepare_data(data, economic_indicators, news_sentiment):
    df = pd.DataFrame(data)
    df = add_advanced_features(df)
    
    # Add economic indicators and news sentiment
    df = df.join(economic_indicators)
    df = df.join(news_sentiment)

    df.dropna(inplace=True)

    X = df.drop(['EUR/USD', 'GBP/USD', 'JPY/USD'], axis=1)
    y = df[['EUR/USD', 'GBP/USD', 'JPY/USD']].pct_change().shift(-1).dropna()

    X = X[:-1]  # Remove the last row to align with y
    
    return X, y

def train_model():
    data = fetch_all_data()
    economic_indicators = fetch_economic_indicators(data.index[0], data.index[-1])
    news_sentiment = fetch_news_sentiment(data.index[0], data.index[-1])
    X, y = prepare_data(data, economic_indicators, news_sentiment)

    model = train_ensemble_model(X, y)

    # Generate feature importance plot
    feature_importance = model.rf_model.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 6))
    plt.bar(features, feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_str = base64.b64encode(img.getvalue()).decode()

    with open('feature_importance.txt', 'w') as f:
        f.write(img_str)

    return model

def make_predictions(model):
    data = fetch_all_data(start_date=datetime.now() - timedelta(days=30))
    economic_indicators = fetch_economic_indicators(data.index[0], data.index[-1])
    news_sentiment = fetch_news_sentiment(data.index[0], data.index[-1])
    X, _ = prepare_data(data, economic_indicators, news_sentiment)
    X_latest = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(X_latest)[0]

    timestamp = datetime.now().isoformat()
    prediction_data = {
        'timestamp': timestamp,
        'prediction': prediction.tolist(),
        'actual': None  # This will be updated later when actual data is available
    }

    # Load existing predictions
    try:
        with open('predictions.json', 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        predictions = []

    # Add new prediction
    predictions.append(prediction_data)

    # Save updated predictions
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

    logging.info(f"Made prediction at {timestamp}: {prediction.tolist()}")

def update_actual_values():
    # Load existing predictions
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)

    # Fetch the latest data
    latest_data = fetch_all_data(start_date=datetime.now() - timedelta(days=30))
    economic_indicators = fetch_economic_indicators(latest_data.index[0], latest_data.index[-1])
    news_sentiment = fetch_news_sentiment(latest_data.index[0], latest_data.index[-1])
    _, y = prepare_data(latest_data, economic_indicators, news_sentiment)

    # Update actual values for previous predictions
    for i, pred in enumerate(predictions):
        if pred['actual'] is None:
            timestamp = datetime.fromisoformat(pred['timestamp'])
            actual_data = y.loc[y.index.date == timestamp.date()]
            if not actual_data.empty:
                predictions[i]['actual'] = actual_data.iloc[0].tolist()

    # Save updated predictions
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

def run_predictor():
    model = train_model()
    make_predictions(model)
    update_actual_values()

if __name__ == "__main__":
    run_predictor()