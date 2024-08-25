import urllib.request
import json
from datetime import datetime, timedelta

API_KEY = "9V0G4JNKUKQ56QSB"
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
    if len(data) < 10:
        return None
    
    last_price = float(data[0]['price'])
    sma_5 = calculate_sma(data, 5)
    sma_10 = calculate_sma(data, 10)
    ema_5 = calculate_ema(data, 5)
    ema_10 = calculate_ema(data, 10)
    
    predictions = {
        'SMA_5': round(sma_5, 2) if sma_5 else None,
        'SMA_10': round(sma_10, 2) if sma_10 else None,
        'EMA_5': round(ema_5, 2) if ema_5 else None,
        'EMA_10': round(ema_10, 2) if ema_10 else None,
    }
    
    # Calculate average prediction
    valid_predictions = [p for p in predictions.values() if p is not None]
    avg_prediction = sum(valid_predictions) / len(valid_predictions) if valid_predictions else None
    
    # Calculate win percentage
    up_indicators = sum(1 for p in valid_predictions if p > last_price)
    win_percentage = (up_indicators / len(valid_predictions)) * 100 if valid_predictions else 50
    
    # Calculate take profit and stop loss
    if avg_prediction:
        take_profit = round(avg_prediction * 1.01, 2)  # 1% above prediction
        stop_loss = round(avg_prediction * 0.99, 2)    # 1% below prediction
    else:
        take_profit = stop_loss = None
    
    return {
        'predictions': predictions,
        'avg_prediction': round(avg_prediction, 2) if avg_prediction else None,
        'win_percentage': round(win_percentage, 2),
        'take_profit': take_profit,
        'stop_loss': stop_loss
    }

def main():
    gold_data = fetch_gold_data()
    
    if not gold_data:
        return json.dumps({"error": "Failed to fetch gold data"})
    
    prediction_data = predict_price(gold_data)
    
    result = {
        'latest_price': float(gold_data[0]['price']),
        'latest_date': gold_data[0]['date'],
        'prediction_data': prediction_data,
    }
    
    return json.dumps(result)

if __name__ == "__main__":
    print(main())