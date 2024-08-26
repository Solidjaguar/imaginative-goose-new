import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

class HyperparameterTuner:
    def __init__(self, n_trials=100):
        self.n_trials = n_trials

    async def tune_random_forest(self, X, y):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 10, 100),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            rf = RandomForestRegressor(**params)
            return np.mean(cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error'))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    async def tune_lstm(self, X, y):
        def create_model(trial):
            model = Sequential()
            model.add(LSTM(trial.suggest_int('lstm_units', 32, 256), 
                           input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(trial.suggest_float('dropout', 0.1, 0.5)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', 
                          optimizer=Adam(learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)))
            return model

        def objective(trial):
            model = create_model(trial)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            epochs = trial.suggest_int('epochs', 50, 200)
            
            history = model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
            return -np.min(history.history['val_loss'])  # We want to minimize the validation loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

# Usage:
# tuner = HyperparameterTuner(n_trials=50)
# best_rf_params = await tuner.tune_random_forest(X, y)
# best_lstm_params = await tuner.tune_lstm(X, y)