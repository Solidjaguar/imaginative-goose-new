from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class ModelTrainer:
    async def train_random_forest(self, X, y, **params):
        model = RandomForestRegressor(**params)
        model.fit(X, y)
        return model

    async def train_lstm(self, X, y, **params):
        model = Sequential([
            LSTM(params['lstm_units'], input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['lstm_units'] // 2),
            Dropout(params['dropout']),
            Dense(1)
        ])
        
        model.compile(loss='mean_squared_error', 
                      optimizer=Adam(learning_rate=params['learning_rate']))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X, y, 
            epochs=params['epochs'], 
            batch_size=params['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return model, history

    def prepare_lstm_data(self, X, y, lookback=60):
        X_lstm, y_lstm = [], []
        for i in range(len(X) - lookback):
            X_lstm.append(X[i:(i + lookback)])
            y_lstm.append(y[i + lookback])
        return np.array(X_lstm), np.array(y_lstm)

# Usage:
# trainer = ModelTrainer()
# rf_model = await trainer.train_random_forest(X, y, **best_rf_params)
# X_lstm, y_lstm = trainer.prepare_lstm_data(X, y)
# lstm_model, history = await trainer.train_lstm(X_lstm, y_lstm, **best_lstm_params)