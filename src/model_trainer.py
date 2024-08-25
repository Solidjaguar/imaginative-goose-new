from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import joblib

class WeightedEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.average(predictions, axis=1, weights=self.weights)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_models(data):
    models = {}
    for market, market_data in data.items():
        print(f"Training models for {market}...")
        X = market_data.drop('price', axis=1)
        y = market_data['price']

        # Prepare data for LSTM
        X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))

        # Define models
        rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
        lasso = Lasso(alpha=0.1)
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        lstm = create_lstm_model((1, X.shape[1]))

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = {model.__class__.__name__: [] for model in [rf, lasso, svr, xgb, lgbm, lstm]}

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train_lstm, X_test_lstm = X_lstm[train_index], X_lstm[test_index]

            for model in [rf, lasso, svr, xgb, lgbm]:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                model_scores[model.__class__.__name__].append(mse)

            lstm.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
            predictions = lstm.predict(X_test_lstm).flatten()
            mse = mean_squared_error(y_test, predictions)
            model_scores['LSTM'].append(mse)

        # Calculate weights based on average MSE
        total_score = sum([1/np.mean(scores) for scores in model_scores.values()])
        weights = [1/np.mean(scores)/total_score for scores in model_scores.values()]

        # Create weighted ensemble
        ensemble = WeightedEnsemble([rf, lasso, svr, xgb, lgbm, lstm], weights)
        ensemble.fit(X, y)

        # Save the ensemble
        joblib.dump(ensemble, f'models/ensemble_{market.replace("/", "_")}.joblib')
        models[market] = ensemble

    return models

if __name__ == "__main__":
    from data_fetcher import fetch_all_data
    from data_processor import prepare_data
    
    raw_data = fetch_all_data()
    prepared_data = prepare_data(raw_data)
    models = train_models(prepared_data)
    
    for market, model in models.items():
        print(f"{market}: Weighted Ensemble with {len(model.models)} models")