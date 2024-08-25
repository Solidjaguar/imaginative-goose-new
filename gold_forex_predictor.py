import requests
import pandas as pd
from datetime import datetime
import os

# Alpha Vantage API key - you'll need to sign up for a free API key
API_KEY = "YOUR_API_KEY_HERE"

def fetch_forex_data(from_currency, to_currency):
    base_url = "https://www.alphavantage.co/query"
    function = "FX_DAILY"
    
    url = f"{base_url}?function={function}&from_symbol={from_currency}&to_symbol={to_currency}&apikey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    if "Time Series FX (Daily)" not in data:
        print(f"Error fetching data: {data.get('Note', 'Unknown error')}")
        return None
    
    df = pd.DataFrame(data["Time Series FX (Daily)"]).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = ['open', 'high', 'low', 'close']
    return df

# Fetch Gold (XAU) to USD data
gold_usd_data = fetch_forex_data("XAU", "USD")

if gold_usd_data is not None:
    print(gold_usd_data.head())
    
    # Save data to CSV
    if not os.path.exists('data'):
        os.makedirs('data')
    gold_usd_data.to_csv('data/gold_usd_historical.csv')
    print("Data saved to data/gold_usd_historical.csv")
else:
    print("Failed to fetch Gold/USD data")

print("Initial setup complete. Next steps: Add more data sources and implement prediction model.")