import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from datetime import datetime, timedelta

def fetch_gold_data():
    gold = yf.Ticker("GC=F")
    data = gold.history(period="1y")
    return data[['Close']].reset_index()

def prepare_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    return data['Close']

def train_model(data):
    model = ARIMA(data, order=(1,1,1))
    return model.fit()

def predict_price(model, data, confidence_threshold=0.7):
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7)
    forecast = model.get_forecast(steps=7)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int(alpha=0.32)  # ~1 standard deviation

    predictions = []
    for date, mean, lower, upper in zip(future_dates, mean_forecast, confidence_intervals['lower Close'], confidence_intervals['upper Close']):
        std_dev = (upper - lower) / 2
        z_score = abs(mean - data.iloc[-1]) / std_dev
        confidence = 1 - 2 * (1 - norm.cdf(z_score))
        
        if confidence >= confidence_threshold:
            predictions.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Predicted_Price': mean,
                'Confidence': confidence
            })

    return predictions

if __name__ == "__main__":
    data = fetch_gold_data()
    prepared_data = prepare_data(data)
    model = train_model(prepared_data)
    predictions = predict_price(model, prepared_data)
    print(predictions)