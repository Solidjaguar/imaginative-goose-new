import random
from datetime import datetime, timedelta

def generate_dummy_data(start_date, days):
    data = []
    price = 1000  # Starting price
    for i in range(days):
        date = start_date + timedelta(days=i)
        change = random.uniform(-10, 10)  # Random daily change between -$10 and $10
        price += change
        data.append((date, price))
    return data

def simple_moving_average(data, window):
    return sum(data[-window:]) / window

def predict_next_price(data, sma_window):
    prices = [price for _, price in data]
    sma = simple_moving_average(prices, sma_window)
    last_price = prices[-1]
    predicted_change = sma - last_price
    return last_price + predicted_change

def evaluate_model(actual, predicted):
    error = abs(actual - predicted)
    percent_error = (error / actual) * 100
    return error, percent_error

if __name__ == "__main__":
    start_date = datetime(2023, 1, 1)
    days = 100
    sma_window = 7

    print("Generating dummy data...")
    data = generate_dummy_data(start_date, days)

    print(f"\nLast 5 days of data:")
    for date, price in data[-5:]:
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

    actual_price = data[-1][1]
    predicted_price = predict_next_price(data, sma_window)

    print(f"\nActual last price: ${actual_price:.2f}")
    print(f"Predicted next price: ${predicted_price:.2f}")

    error, percent_error = evaluate_model(actual_price, predicted_price)
    print(f"\nPrediction Error: ${error:.2f}")
    print(f"Percent Error: {percent_error:.2f}%")

    print("\nNote: This is a very simple model and should not be used for actual trading decisions.")