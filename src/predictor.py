import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

class StackingEnsembleModel:
    def __init__(self, lstm_units=50, dropout_rate=0.2):
        self.rf_model = joblib.load('best_model.joblib')
        self.lstm_model = None
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler = StandardScaler()
        self.meta_model = LinearRegression()

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

    def fit(self, X, y):
        # Random Forest prediction
        rf_pred = self.rf_model.predict(X)

        # LSTM
        X_scaled = self.scaler.fit_transform(X)
        X_lstm, y_lstm = self.prepare_lstm_data(X_scaled, y)

        self.lstm_model = self.create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
        self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)

        lstm_pred = self.lstm_model.predict(X_lstm).flatten()

        # Prepare meta-model input
        meta_features = np.column_stack((rf_pred[10:], lstm_pred))

        # Train meta-model
        self.meta_model.fit(meta_features, y[10:])

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

            # Combine predictions for meta-model
            meta_features = np.column_stack((rf_pred, lstm_pred))

            # Meta-model prediction
            final_pred = self.meta_model.predict(meta_features)

            predictions.append(final_pred[0])

            # Update current_X for the next prediction
            current_X = np.vstack((current_X[1:], final_pred))

        return np.array(predictions)

def predict_price(models, data, steps=7):
    X = data.index.astype(int).values.reshape(-1, 1)
    y = data.values

    ensemble = StackingEnsembleModel()
    ensemble.fit(X, y)

    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, steps+1)]
    predictions = ensemble.predict(X, steps=steps)

    return pd.Series(predictions, index=future_dates)