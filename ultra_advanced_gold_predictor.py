import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from datetime import datetime, timedelta

def fetch_gold_data(interval='15m', period='1d'):
    gold = yf.Ticker("GC=F")
    data = gold.history(interval=interval, period=period)
    return data[['Close']].reset_index()

def prepare_data(data):
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
    return data['Close']

def train_model(data):
    model = ARIMA(data, order=(1,1,1))
    return model.fit()

def predict_price(model, data, steps=4, confidence_threshold=0.7):
    last_datetime = data.index[-1]
    future_datetimes = pd.date_range(start=last_datetime + pd.Timedelta(minutes=15), periods=steps, freq='15T')
    forecast = model.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int(alpha=0.32)  # ~1 standard deviation

    predictions = []
    for datetime, mean, lower, upper in zip(future_datetimes, mean_forecast, confidence_intervals['lower Close'], confidence_intervals['upper Close']):
        std_dev = (upper - lower) / 2
        z_score = abs(mean - data.iloc[-1]) / std_dev
        confidence = 1 - 2 * (1 - norm.cdf(z_score))
        
        if confidence >= confidence_threshold:
            predictions.append({
                'Datetime': datetime.strftime('%Y-%m-%d %H:%M:%S'),
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
