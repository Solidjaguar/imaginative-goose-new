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

# Add the Currents API key
CURRENTS_API_KEY = "FkEEwNLACnLEfCoJ09fFe3yrVaPGZ76u-PKi8-yHqGRJ6hd8"

def fetch_data(start_date, end_date):
    # Fetch gold price data
    gold = yf.download("GC=F", start=start_date, end=end_date)
    
    # Fetch additional features
    usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)["Close"]
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
    oil = yf.download("CL=F", start=start_date, end=end_date)["Close"]
    vix = yf.download("^VIX", start=start_date, end=end_date)["Close"]
    interest_rates = yf.download("^TNX", start=start_date, end=end_date)["Close"]
    inflation = yf.download("CPI", start=start_date, end=end_date)["Close"]
    
    # Fetch cryptocurrency data (Bitcoin as a proxy)
    btc = yf.download("BTC-USD", start=start_date, end=end_date)["Close"]
    
    # Combine all features
    df = pd.DataFrame({
        "Gold_Price": gold["Close"],
        "USD_Index": usd_index,
        "SP500": sp500,
        "Oil_Price": oil,
        "VIX": vix,
        "Interest_Rate": interest_rates,
        "Inflation": inflation,
        "Bitcoin": btc
    })
    
    # Forward fill missing data
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def fetch_news_sentiment(start_date, end_date):
    sentiments = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_date <= end_date:
        # Format the date for the API request
        formatted_date = current_date.strftime("%Y-%m-%d")
        
        # Make API request
        url = f"https://api.currentsapi.services/v1/search?keywords=gold&language=en&start_date={formatted_date}&end_date={formatted_date}&apiKey={CURRENTS_API_KEY}"
        response = requests.get(url)
        
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
        
        current_date += timedelta(days=1)
    
    return pd.Series(dict(sentiments))

def create_features(df):
    # Basic features
    df['MA7'] = df['Gold_Price'].rolling(window=7).mean()
    df['MA30'] = df['Gold_Price'].rolling(window=30).mean()
    df['Volatility'] = df['Gold_Price'].rolling(window=30).std()
    
    # Percentage changes
    for col in df.columns:
        df[f'{col}_Change'] = df[col].pct_change()
    
    # Technical indicators
    df['RSI'] = talib.RSI(df['Gold_Price'])
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Gold_Price'])
    df['ATR'] = talib.ATR(df['Gold_Price'], df['Gold_Price'], df['Gold_Price'])
    
    # Fourier transforms for cyclical patterns
    for period in [30, 90, 365]:
        fourier = np.fft.fft(df['Gold_Price'])
        frequencies = np.fft.fftfreq(len(df['Gold_Price']))
        indices = np.argsort(frequencies)
        top_indices = indices[-period:]
        restored_sig = np.fft.ifft(fourier[top_indices])
        df[f'Fourier_{period}'] = np.real(restored_sig)
    
    # Lag features
    for lag in [1, 7, 30]:
        df[f'Gold_Price_Lag_{lag}'] = df['Gold_Price'].shift(lag)
    
    df.dropna(inplace=True)
    return df

# The rest of the script remains the same...

if __name__ == "__main__":
    # Fetch and prepare data
    df = fetch_data("2010-01-01", "2023-05-31")
    news_sentiment = fetch_news_sentiment("2010-01-01", "2023-05-31")
    df['News_Sentiment'] = news_sentiment
    df = create_features(df)
    
    # The rest of the main block remains the same...

print("\nNote: This ultra-advanced model now incorporates real news sentiment data from the Currents API, but should still be used cautiously for actual trading decisions.")