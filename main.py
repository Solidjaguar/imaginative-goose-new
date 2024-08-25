import urllib.request
import json
from datetime import datetime, timedelta

API_KEY = "demo"  # Replace with your Alpha Vantage API key for more frequent updates
BASE_URL = "https://www.alphavantage.co/query"

def fetch_gold_data():
    url = f"{BASE_URL}?function=GOLD&apikey={API_KEY}"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        
        if 'data' not in data:
            print(f"Unexpected API response: {data}")
            return []
        
        return data['data']
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return []

def calculate_sma(data, window):
    if len(data) < window:
        return None
    return sum(float(d['price']) for d in data[:window]) / window

def calculate_ema(data, window):
    if len(data) < window:
        return None
    
    prices = [float(d['price']) for d in data[:window]]
    ema = sum(prices) / window
    multiplier = 2 / (window + 1)
    
    for price in prices[window:]:
        ema = (price - ema) * multiplier + ema
    
    return ema

def predict_price(data):
    if len(data) < 5:
        return None
    
    sma_5 = calculate_sma(data, 5)
    ema_5 = calculate_ema(data, 5)
    
    if sma_5 is None or ema_5 is None:
        return None
    
    last_price = float(data[0]['price'])
    prediction = (sma_5 + ema_5) / 2
    
    return round(prediction, 2)

def main():
    gold_data = fetch_gold_data()
    
    if not gold_data:
        return json.dumps({"error": "Failed to fetch gold data"})
    
    prediction = predict_price(gold_data)
    
    result = {
        'latest_price': float(gold_data[0]['price']),
        'latest_date': gold_data[0]['date'],
        'prediction': prediction,
        'historical_data': [{'date': d['date'], 'price': float(d['price'])} for d in gold_data]
    }
    
    return json.dumps(result)

if __name__ == "__main__":
    print(main())