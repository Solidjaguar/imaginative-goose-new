from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from src.utils.logger import app_logger
from src.utils.model_versioner import ModelVersioner

class ModelTrainer:
    def __init__(self):
        self.model_versioner = ModelVersioner('models')

    def prepare_data(self, forex_data, gdp_data, inflation_data):
        # Implement data preparation logic here
        # This is a placeholder implementation
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_random_forest(self, forex_data, gdp_data, inflation_data):
        app_logger.info("Training Random Forest model")
        X_train, X_test, y_train, y_test = self.prepare_data(forex_data, gdp_data, inflation_data)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {'mse': mse, 'r2': r2}
        self.model_versioner.save_model(model, 'RandomForest', metrics)
        
        app_logger.info(f"Random Forest model trained. MSE: {mse}, R2: {r2}")
        return model

    def train_lstm(self, forex_data, gdp_data, inflation_data):
        app_logger.info("Training LSTM model")
        X_train, X_test, y_train, y_test = self.prepare_data(forex_data, gdp_data, inflation_data)
        
        # Reshape input for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, verbose=0)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {'mse': mse, 'r2': r2}
        self.model_versioner.save_model(model, 'LSTM', metrics)
        
        app_logger.info(f"LSTM model trained. MSE: {mse}, R2: {r2}")
        return model