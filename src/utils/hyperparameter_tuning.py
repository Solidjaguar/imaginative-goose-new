from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

class HyperparameterTuner:
    def tune_random_forest(self, X, y):
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        rf = RandomForestRegressor()

        random_search = RandomizedSearchCV(rf, param_distributions=param_dist, 
                                           n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

        random_search.fit(X, y)
        return random_search.best_params_

    def create_lstm_model(self, input_shape, units=50, learning_rate=0.001):
        model = Sequential()
        model.add(LSTM(units, input_shape=input_shape))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def tune_lstm(self, X, y):
        model = KerasRegressor(build_fn=self.create_lstm_model, verbose=0)

        param_dist = {
            'units': [32, 64, 128],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 150],
            'learning_rate': [0.001, 0.01, 0.1]
        }

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                           n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

        random_search.fit(X, y)
        return random_search.best_params_

# Usage:
# tuner = HyperparameterTuner()
# best_rf_params = tuner.tune_random_forest(X, y)
# best_lstm_params = tuner.tune_lstm(X, y)