import pandas as pd
import numpy as np

def prepare_data(data):
    prepared_data = {}
    for market, prices in data.items():
        # Remove any missing values
        prices = prices.dropna()
        
        # Calculate returns
        returns = prices.pct_change()
        
        # Calculate moving averages
        ma_50 = prices.rolling(window=50).mean()
        ma_200 = prices.rolling(window=200).mean()
        
        # Calculate volatility
        volatility = returns.rolling(window=30).std()
        
        # Combine features
        prepared_data[market] = pd.DataFrame({
            'price': prices,
            'returns': returns,
            'ma_50': ma_50,
            'ma_200': ma_200,
            'volatility': volatility
        }).dropna()
    
    return prepared_data

if __name__ == "__main__":
    from data_fetcher import fetch_all_data
    
    raw_data = fetch_all_data()
    prepared_data = prepare_data(raw_data)
    
    for market, data in prepared_data.items():
        print(f"{market}: {len(data)} prepared data points")
        print(data.head())
        print("\n")