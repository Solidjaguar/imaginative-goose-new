import requests
import pandas as pd
from datetime import datetime
import os

# Alpha Vantage API key
API_KEY = "AELOSC3OP7F0I708"

def fetch_forex_data(from_currency, to_currency):
    base_url = "https://www.alphavantage.co/query"
    function = "FX_DAILY"
    
    url = f"{base_url}?function={function}&from_symbol={from_currency}&to_symbol={to_currency}&apikey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    if "Time Series FX (Daily)" not in data:
        print(f"Error fetching data. Response: {data}")
        return None
    
    df = pd.DataFrame(data["Time Series FX (Daily)"]).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = ['open', 'high', 'low', 'close']
    return df

# Fetch EUR to USD data
eur_usd_data = fetch_forex_data("EUR", "USD")

if eur_usd_data is not None:
    print("Successfully fetched EUR/USD data:")
    print(eur_usd_data.head())
    
    # Save data to CSV
    if not os.path.exists('data'):
        os.makedirs('data')
    eur_usd_data.to_csv('data/eur_usd_historical.csv')
    print("Data saved to data/eur_usd_historical.csv")
else:
    print("Failed to fetch EUR/USD data")

print("Initial setup complete. Next steps: Add more data sources and implement prediction model.")