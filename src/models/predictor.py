import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import joblib

class AdaptivePredictor:
    def __init__(self, market):
        self.market = market
        self.ensemble = joblib.load(f'models/ensemble_{market.replace("/", "_")}.joblib')
        self.scaler = StandardScaler()
        self.learning_rate = 0.01
        self.performance_history = []

    def predict(self, X, steps=7):
        X_scaled = self.scaler.fit_transform(X)
        predictions = []
        current_X = X_scaled[-1:].copy()

        for _ in range(steps):
            pred = self.ensemble.predict(current_X)
            predictions.append(pred[0])

            # Update current_X for the next prediction
            current_X = np.roll(current_X, -1, axis=0)
            current_X[0, -1] = pred[0]

        return np.array(predictions)

    def update(self, actual, predicted):
        error = np.mean((actual - predicted) ** 2)
        self.performance_history.append(error)

        # Adjust learning rate based on recent performance
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-10:])
            if recent_performance > np.mean(self.performance_history):
                self.learning_rate *= 0.9  # Decrease learning rate
            else:
                self.learning_rate *= 1.1  # Increase learning rate

        # Update model weights
        for model in self.ensemble.models:
            if hasattr(model, 'learning_rate'):
                model.learning_rate = self.learning_rate

def predict_prices(data, steps=7):
    predictions = {}
    predictors = {}
    for market, market_data in data.items():
        print(f"Making predictions for {market}...")
        X = market_data.drop('price', axis=1)
        y = market_data['price']

        predictor = AdaptivePredictor(market)
        future_dates = [market_data.index[-1] + timedelta(days=i) for i in range(1, steps+1)]
        predictions[market] = pd.Series(predictor.predict(X, steps=steps), index=future_dates)
        predictors[market] = predictor

    return predictions, predictors

if __name__ == "__main__":
    from data_fetcher import fetch_all_data
    from data_processor import prepare_data
    
    raw_data = fetch_all_data()
    prepared_data = prepare_data(raw_data)
    predictions, _ = predict_prices(prepared_data)
    
    for market, pred in predictions.items():
        print(f"\n{market} predictions:")
        print(pred)