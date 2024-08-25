from datetime import datetime, timedelta
import random

def generate_simulated_data(days=30):
    end_date = datetime.now()
    data = []
    price = 1800  # Starting price around $1800 per ounce
    
    for i in range(days):
        date = end_date - timedelta(days=i)
        price += random.uniform(-10, 10)  # Random daily change between -$10 and $10
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': f'{price:.2f}'
        })
    
    return data

def simple_prediction(data):
    if len(data) < 2:
        return None
    
    last_price = float(data[0]['price'])
    prev_price = float(data[1]['price'])
    
    if last_price > prev_price:
        return last_price * 1.01  # Predict 1% increase
    else:
        return last_price * 0.99  # Predict 1% decrease

def main():
    print("Generating simulated gold price data...")
    gold_data = generate_simulated_data()
    
    print(f"Latest simulated gold price: ${gold_data[0]['price']} USD (as of {gold_data[0]['date']})")
    
    prediction = simple_prediction(gold_data)
    
    if prediction:
        print(f"Simple prediction for next price: ${prediction:.2f} USD")
    else:
        print("Unable to make a prediction.")

if __name__ == "__main__":
    main()