import random
from datetime import datetime, timedelta
import math

def generate_dummy_data(start_date, days):
    data = []
    price = 1000  # Starting price
    trend = 0.1  # Slight upward trend
    annual_cycle = 50  # $50 annual cycle amplitude
    for i in range(days):
        date = start_date + timedelta(days=i)
        trend_component = i * trend
        seasonal_component = annual_cycle * math.sin(2 * math.pi * i / 365)
        noise = random.uniform(-10, 10)
        price = 1000 + trend_component + seasonal_component + noise
        data.append((date, price))
    return data

def simple_moving_average(data, window):
    return sum(data[-window:]) / window

def exponential_moving_average(data, window):
    alpha = 2 / (window + 1)
    ema = data[0]
    for price in data[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema

def calculate_rsi(data, window):
    deltas = [b - a for a, b in zip(data[:-1], data[1:])]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def extract_features(data, sma_window, ema_window, rsi_window):
    prices = [price for _, price in data]
    features = {
        'last_price': prices[-1],
        'sma': simple_moving_average(prices, sma_window),
        'ema': exponential_moving_average(prices, ema_window),
        'rsi': calculate_rsi(prices, rsi_window),
        'trend': (prices[-1] - prices[0]) / len(prices),
        'day_of_year': data[-1][0].timetuple().tm_yday,
    }
    return features

def predict_next_price(features):
    # This is a very simplistic prediction model
    predicted_change = (
        (features['sma'] - features['last_price']) * 0.3 +
        (features['ema'] - features['last_price']) * 0.3 +
        (50 - features['rsi']) * 0.2 +  # RSI: above 70 overbought, below 30 oversold
        features['trend'] * 10 +  # Trend component
        math.sin(2 * math.pi * features['day_of_year'] / 365) * 5  # Seasonal component
    )
    return features['last_price'] + predicted_change

def evaluate_model(actual, predicted):
    error = abs(actual - predicted)
    percent_error = (error / actual) * 100
    return error, percent_error

if __name__ == "__main__":
    start_date = datetime(2023, 1, 1)
    days = 365
    sma_window = 14
    ema_window = 14
    rsi_window = 14

    print("Generating dummy data...")
    data = generate_dummy_data(start_date, days)

    print(f"\nLast 5 days of data:")
    for date, price in data[-5:]:
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

    features = extract_features(data, sma_window, ema_window, rsi_window)
    predicted_price = predict_next_price(features)

    actual_price = data[-1][1]
    print(f"\nFeatures used for prediction:")
    for feature, value in features.items():
        print(f"{feature}: {value:.2f}")

    print(f"\nActual last price: ${actual_price:.2f}")
    print(f"Predicted next price: ${predicted_price:.2f}")

    error, percent_error = evaluate_model(actual_price, predicted_price)
    print(f"\nPrediction Error: ${error:.2f}")
    print(f"Percent Error: {percent_error:.2f}%")

    print("\nNote: This is a simplified model and should not be used for actual trading decisions.")