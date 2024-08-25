import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

class EnsembleModel:
    def __init__(self, n_estimators=100, lstm_units=50, dropout_rate=0.2):
        self.rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
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
            Dense(3)  # 3 output units for EUR/USD, GBP/USD, JPY/USD
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
        # Random Forest
        self.rf_model.fit(X, y)

        # LSTM
        X_scaled = self.scaler.fit_transform(X)
        X_lstm, y_lstm = self.prepare_lstm_data(X_scaled, y)
        self.lstm_model = self.create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
        self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    def predict(self, X):
        # Random Forest prediction
        rf_pred = self.rf_model.predict(X)

        # LSTM prediction
        X_scaled = self.scaler.transform(X)
        X_lstm, _ = self.prepare_lstm_data(X_scaled, np.zeros((len(X), 3)))  # Dummy y values
        lstm_pred = self.lstm_model.predict(X_lstm)

        # Ensemble prediction (simple average)
        ensemble_pred = (rf_pred + lstm_pred[-1]) / 2
        return ensemble_pred

def train_ensemble_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ensemble = EnsembleModel()
    ensemble.fit(X_train, y_train)
    
    # Evaluate the model
    train_pred = ensemble.predict(X_train)
    test_pred = ensemble.predict(X_test)
    
    train_mse = np.mean((y_train - train_pred) ** 2)
    test_mse = np.mean((y_test - test_pred) ** 2)
    
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    
    return ensemble

if __name__ == "__main__":
    # For testing purposes
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 3)
    model = train_ensemble_model(X, y)