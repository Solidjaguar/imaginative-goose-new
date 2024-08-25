import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump
import tensorflow as tf
from config import USE_GPU, NUM_CPU_CORES, RANDOM_FOREST_MODEL_PATH, LASSO_MODEL_PATH, SVR_MODEL_PATH, XGB_MODEL_PATH, LGBM_MODEL_PATH, LSTM_MODEL_PATH

def train_models(X_train, y_train):
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, n_jobs=NUM_CPU_CORES)
    rf_model.fit(X_train, y_train)
    dump(rf_model, RANDOM_FOREST_MODEL_PATH)

    # Lasso
    lasso_model = MultiOutputRegressor(Lasso())
    lasso_model.fit(X_train, y_train)
    dump(lasso_model, LASSO_MODEL_PATH)

    # SVR
    svr_model = MultiOutputRegressor(SVR())
    svr_model.fit(X_train, y_train)
    dump(svr_model, SVR_MODEL_PATH)

    # XGBoost
    xgb_model = MultiOutputRegressor(XGBRegressor(n_jobs=NUM_CPU_CORES, gpu_id=0 if USE_GPU else None, tree_method='gpu_hist' if USE_GPU else 'hist'))
    xgb_model.fit(X_train, y_train)
    dump(xgb_model, XGB_MODEL_PATH)

    # LightGBM
    lgbm_model = MultiOutputRegressor(LGBMRegressor(n_jobs=NUM_CPU_CORES, device='gpu' if USE_GPU else 'cpu'))
    lgbm_model.fit(X_train, y_train)
    dump(lgbm_model, LGBM_MODEL_PATH)

    # LSTM
    if USE_GPU:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(y_train.shape[1])
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    lstm_model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, verbose=1)
    lstm_model.save(LSTM_MODEL_PATH)

    return rf_model, lasso_model, svr_model, xgb_model, lgbm_model, lstm_model

# You'll need to call this function with your training data
# Example: train_models(X_train, y_train)