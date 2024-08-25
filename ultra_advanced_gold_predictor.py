import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
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

def predict_price(model, data):
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7)
    forecast = model.forecast(steps=7)
    prediction_data = pd.DataFrame({'Date': future_dates, 'Predicted_Price': forecast})
    return prediction_data.to_dict(orient='records')

if __name__ == "__main__":
    data = fetch_gold_data()
    prepared_data = prepare_data(data)
    model = train_model(prepared_data)
    predictions = predict_price(model, prepared_data)
    print(predictions)