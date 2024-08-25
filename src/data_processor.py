import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def prepare_data(data, config):
    df = pd.DataFrame(data['Gold'])
    df = df.join(data['Forex'])
    df = df.join(data['Economic'])
    df = df.join(data['Crypto'])
    df = df.join(data['Sentiment'])
    
    df = add_technical_indicators(df, config['features']['technical_indicators'])
    df = add_lagged_features(df, config['features']['lagged_features'])
    df = add_return_features(df, config['features']['return_features'])
    
    df = df.dropna()
    
    features = df.drop(['Close'], axis=1)
    target = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def add_technical_indicators(df, indicators):
    if 'SMA' in indicators:
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    if 'EMA' in indicators:
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    if 'MACD' in indicators:
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
    if 'RSI' in indicators:
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    if 'BollingerBands' in indicators:
        bollinger = BollingerBands(close=df['Close'])
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()
    return df

def add_lagged_features(df, lags):
    for lag in lags:
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    return df

def add_return_features(df, returns):
    if 'Returns' in returns:
        df['Returns'] = df['Close'].pct_change()
    if 'LogReturns' in returns:
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
    return df