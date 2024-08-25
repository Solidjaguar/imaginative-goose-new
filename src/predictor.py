import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

class StackingEnsembleModel:
    def __init__(self, lstm_units=50, dropout_rate=0.2):
        self.rf_model = None
        self.lstm_model = None
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler = StandardScaler()

    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_lstm_data(self, X, y, lookback=10):
        X_lstm, y_lstm = [], []
        for i in range(len(X) - lookback):
            X_lstm.append(X[i:(i + lookback)])
            y_lstm.append(y[i + lookback])
        return np.array(X_lstm), np.array(y_lstm)

    def fit(self, X, y, market):
        # Load Random Forest model
        self.rf_model = joblib.load(f'best_model_{market.replace("/", "_")}.joblib')

        # LSTM
        X_scaled = self.scaler.fit_transform(X)
        X_lstm, y_lstm = self.prepare_lstm_data(X_scaled, y)

        self.lstm_model = self.create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
        self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)

    def predict(self, X, steps=7):
        predictions = []
        current_X = X[-10:].copy()  # Use the last 10 data points

        for _ in range(steps):
            # Random Forest prediction
            rf_pred = self.rf_model.predict(current_X[-1:])

            # LSTM prediction
            X_scaled = self.scaler.transform(current_X)
            X_lstm = X_scaled.reshape((1, 10, X_scaled.shape[1]))
            lstm_pred = self.lstm_model.predict(X_lstm).flatten()

            # Combine predictions (simple average)
            final_pred = (rf_pred + lstm_pred) / 2

            predictions.append(final_pred[0])

            # Update current_X for the next prediction
            new_row = current_X[-1:].copy()
            new_row['price'] = final_pred[0]
            current_X = pd.concat([current_X[1:], new_row])

        return np.array(predictions)

def predict_prices(data, steps=7):
    predictions = {}
    for market, market_data in data.items():
        print(f"Making predictions for {market}...")
        X = market_data.drop('price', axis=1)
        y = market_data['price']

        ensemble = StackingEnsembleModel()
        ensemble.fit(X, y, market)

        future_dates = [market_data.index[-1] + timedelta(days=i) for i in range(1, steps+1)]
        predictions[market] = pd.Series(ensemble.predict(X, steps=steps), index=future_dates)

    return predictions

if __name__ == "__main__":
    from data_fetcher import fetch_all_data
    from data_processor import prepare_data
    
    raw_data = fetch_all_data()
    prepared_data = prepare_data(raw_data)
    predictions = predict_prices(prepared_data)
    
    for market, pred in predictions.items():
        print(f"\n{market} predictions:")
        print(pred)