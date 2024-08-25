import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

def prepare_data(data):
    prepared_data = {}
    for market, prices in data.items():
        # Remove any missing values
        df = prices.to_frame(name='price').dropna()
        
        # Calculate returns
        df['returns'] = df['price'].pct_change()
        
        # Basic moving averages
        df['ma_50'] = SMAIndicator(close=df['price'], window=50).sma_indicator()
        df['ma_200'] = SMAIndicator(close=df['price'], window=200).sma_indicator()
        
        # Exponential moving averages
        df['ema_12'] = EMAIndicator(close=df['price'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['price'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=df['price'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['price']).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['price'], low=df['price'], close=df['price'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['price'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mavg'] = bollinger.bollinger_mavg()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=30).std()
        
        # Market-specific features
        if 'USD' in market:
            # For USD pairs, add Dollar Index
            df['usd_index'] = data['USD Index'].loc[df.index]
        
        if market == 'Gold':
            # For Gold, add S&P 500 as a feature (inverse relationship)
            df['sp500'] = data['S&P 500'].loc[df.index]
        
        prepared_data[market] = df.dropna()
    
    return prepared_data

if __name__ == "__main__":
    from data_fetcher import fetch_all_data
    
    raw_data = fetch_all_data()
    prepared_data = prepare_data(raw_data)
    
    for market, data in prepared_data.items():
        print(f"{market}: {len(data)} prepared data points")
        print(data.head())
        print("\n")