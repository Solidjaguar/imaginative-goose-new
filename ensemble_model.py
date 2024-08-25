import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class StackingEnsembleModel:
    def __init__(self, n_estimators=100, lstm_units=50, dropout_rate=0.2):
        self.rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_train_pred = self.rf_model.predict(X_train)
        rf_val_pred = self.rf_model.predict(X_val)

        # LSTM
        X_scaled = self.scaler.fit_transform(X)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        X_lstm_train, y_lstm_train = self.prepare_lstm_data(X_train_scaled, y_train)
        X_lstm_val, y_lstm_val = self.prepare_lstm_data(X_val_scaled, y_val)

        self.lstm_model = self.create_lstm_model((X_lstm_train.shape[1], X_lstm_train.shape[2]))
        self.lstm_model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=32, validation_data=(X_lstm_val, y_lstm_val), verbose=0)

        lstm_train_pred = self.lstm_model.predict(X_lstm_train)
        lstm_val_pred = self.lstm_model.predict(X_lstm_val)

        # Prepare meta-model input
        meta_train_features = np.column_stack((rf_train_pred, lstm_train_pred))
        meta_val_features = np.column_stack((rf_val_pred, lstm_val_pred))

        # Train meta-model
        self.meta_model.fit(meta_train_features, y_train[len(y_train)-len(lstm_train_pred):])

        # Evaluate the stacking ensemble
        meta_val_pred = self.meta_model.predict(meta_val_features)
        mse = mean_squared_error(y_val[len(y_val)-len(lstm_val_pred):], meta_val_pred)
        print(f"Stacking Ensemble Validation MSE: {mse}")

    def predict(self, X):
        # Random Forest prediction
        rf_pred = self.rf_model.predict(X)

        # LSTM prediction
        X_scaled = self.scaler.transform(X)
        X_lstm, _ = self.prepare_lstm_data(X_scaled, np.zeros((len(X), 3)))  # Dummy y values
        lstm_pred = self.lstm_model.predict(X_lstm)

        # Combine predictions for meta-model
        meta_features = np.column_stack((rf_pred, lstm_pred))

        # Meta-model prediction
        final_pred = self.meta_model.predict(meta_features)

        return final_pred

def train_stacking_ensemble_model(X, y):
    ensemble = StackingEnsembleModel()
    ensemble.fit(X, y)
    return ensemble

if __name__ == "__main__":
    # For testing purposes
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 3)
    model = train_stacking_ensemble_model(X, y)