import yfinance as yf
import pandas as pd

def fetch_market_data(symbol, start_date='2010-01-01'):
    data = yf.download(symbol, start=start_date)
    return data['Close']

def fetch_all_data():
    markets = {
        'Gold': 'GC=F',
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'USDJPY=X',
        'AUD/USD': 'AUDUSD=X',
        'USD/CAD': 'USDCAD=X'
    }
    
    data = {}
    for market, symbol in markets.items():
        data[market] = fetch_market_data(symbol)
    
    return data

if __name__ == "__main__":
    data = fetch_all_data()
    for market, prices in data.items():
        print(f"{market}: {len(prices)} data points")