import urllib.request
import json
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

def prepare_data(data):
    X = []
    y = []
    for i in range(len(data) - 5):
        X.append([float(data[j]['price']) for j in range(i, i+5)])
        y.append(float(data[i+5]['price']))
    return np.array(X), np.array(y)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_price(model, data):
    last_5_prices = np.array([[float(data[i]['price']) for i in range(5)]])
    prediction = model.predict(last_5_prices)[0]
    
    last_price = float(data[0]['price'])
    change_percentage = ((prediction - last_price) / last_price) * 100
    
    win_percentage = 60 if change_percentage > 0 else 40
    
    take_profit = round(prediction * 1.01, 2)
    stop_loss = round(prediction * 0.99, 2)
    
    return {
        'prediction': round(prediction, 2),
        'change_percentage': round(change_percentage, 2),
        'win_percentage': win_percentage,
        'take_profit': take_profit,
        'stop_loss': stop_loss
    }

def main():
    gold_data = fetch_gold_data()
    
    if not gold_data:
        return json.dumps({"error": "Failed to fetch gold data"})
    
    X, y = prepare_data(gold_data)
    model = train_model(X, y)
    prediction_data = predict_price(model, gold_data)
    
    result = {
        'latest_price': float(gold_data[0]['price']),
        'latest_date': gold_data[0]['date'],
        'prediction_data': prediction_data,
    }
    
    return json.dumps(result)

if __name__ == "__main__":
    print(main())