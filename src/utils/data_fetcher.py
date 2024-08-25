import yfinance as yf
import pandas as pd
import requests
from fredapi import Fred

def fetch_market_data(symbol, start_date='2010-01-01'):
    data = yf.download(symbol, start=start_date)
    return data['Close']

def fetch_economic_data(indicator, start_date='2010-01-01'):
    fred = Fred(api_key='YOUR_FRED_API_KEY')  # Replace with your actual FRED API key
    data = fred.get_series(indicator, observation_start=start_date)
    return data

def fetch_all_data():
    markets = {
        'Gold': 'GC=F',
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'USDJPY=X',
        'AUD/USD': 'AUDUSD=X',
        'USD/CAD': 'USDCAD=X',
        'USD Index': 'DX-Y.NYB',
        'S&P 500': '^GSPC'
    }
    
    data = {}
    for market, symbol in markets.items():
        data[market] = fetch_market_data(symbol)
    
    # Fetch economic indicators
    indicators = {
        'US_GDP': 'GDP',
        'US_CPI': 'CPIAUCSL',
        'US_Unemployment': 'UNRATE',
        'EU_GDP': 'CLVMNACSCAB1GQEA19',
        'EU_CPI': 'CP0000EZ19M086NEST',
        'EU_Unemployment': 'LRHUTTTTEZM156S',
        'UK_GDP': 'UKNGDP',
        'UK_CPI': 'GBRCPIALLMINMEI',
        'UK_Unemployment': 'LMUNRRTTGBM156S'
    }
    
    for name, indicator in indicators.items():
        data[name] = fetch_economic_data(indicator)
    
    return data

if __name__ == "__main__":
    data = fetch_all_data()
    for market, prices in data.items():
        print(f"{market}: {len(prices)} data points")