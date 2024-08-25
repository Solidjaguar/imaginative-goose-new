import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_gold_data(interval='1d', period='1mo'):
    gold = yf.Ticker("GC=F")
    data = gold.history(interval=interval, period=period)
    return data[['Close']].reset_index()

def prepare_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    return data['Close']

def train_arima_model(data):
    model = ARIMA(data, order=(1,1,1))
    return model.fit()

def train_random_forest_model(data):
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, mean_squared_error(y_test, model.predict(X_test))

def ensemble_predict(arima_model, rf_model, data, steps=7):
    arima_forecast = arima_model.forecast(steps=steps)
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, steps+1)]
    rf_forecast = rf_model.predict(np.array(range(len(data), len(data)+steps)).reshape(-1, 1))
    ensemble_forecast = (arima_forecast + rf_forecast) / 2
    return pd.Series(ensemble_forecast, index=future_dates)

def plot_predictions(data, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, label='Historical Data')
    plt.plot(predictions.index, predictions.values, label='Predictions', color='red')
    plt.title('Gold Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('gold_predictions.png')
    plt.close()

def predict_price(data, steps=7):
    arima_model = train_arima_model(data)
    rf_model, rf_mse = train_random_forest_model(data)
    predictions = ensemble_predict(arima_model, rf_model, data, steps)
    plot_predictions(data, predictions)
    return predictions

if __name__ == "__main__":
    data = fetch_gold_data()
    prepared_data = prepare_data(data)
    predictions = predict_price(prepared_data)
    print(predictions)