from datetime import datetime, timedelta
import random
import json

def generate_simulated_data(days=30):
    end_date = datetime.now()
    data = []
    price = 1800  # Starting price around $1800 per ounce
    
    for i in range(days):
        date = end_date - timedelta(days=i)
        price += random.uniform(-10, 10)  # Random daily change between -$10 and $10
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': round(price, 2)
        })
    
    return data

def simple_prediction(data):
    if len(data) < 2:
        return None
    
    last_price = float(data[0]['price'])
    prev_price = float(data[1]['price'])
    
    if last_price > prev_price:
        return round(last_price * 1.01, 2)  # Predict 1% increase
    else:
        return round(last_price * 0.99, 2)  # Predict 1% decrease

def main():
    gold_data = generate_simulated_data()
    prediction = simple_prediction(gold_data)
    
    result = {
        'latest_price': gold_data[0]['price'],
        'latest_date': gold_data[0]['date'],
        'prediction': prediction,
        'historical_data': gold_data
    }
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()