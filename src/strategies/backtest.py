import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ultra_advanced_gold_predictor import (fetch_data_for_model, create_features, 
                                           detect_anomalies, RobustScaler, RFE, 
                                           RandomForestRegressor, prepare_lstm_data,
                                           train_lstm, create_stacking_ensemble,
                                           optimize_hyperparameters_parallel,
                                           train_model_parallel)
from config import *
import logging
from typing import Tuple, List
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def prepare_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for backtesting."""
    gold_data, economic_data, sentiment_data, geopolitical_events, related_assets = fetch_data_for_model(start_date, end_date)
    df = create_features(gold_data, economic_data, sentiment_data, related_assets, geopolitical_events)
    
    X = df.drop('Close', axis=1)
    y = df['Close']
    
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    anomalies = detect_anomalies(X_scaled)
    logger.info(f"Detected {sum(anomalies == -1)} anomalies")
    
    rfe = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=N_FEATURES_TO_SELECT)
    X_selected = pd.DataFrame(rfe.fit_transform(X_scaled, y), columns=X_scaled.columns[rfe.support_], index=X_scaled.index)
    
    return X_selected, y

def train_models(X: pd.DataFrame, y: pd.Series) -> Tuple[object, object]:
    """Train ensemble and LSTM models."""
    best_params = optimize_hyperparameters_parallel(X, y, MODEL_NAMES)
    
    with Pool(processes=cpu_count()) as pool:
        models = pool.map(train_model_parallel, [(X, y, model_name, params) for model_name, params in best_params.items()])
    
    stacking_model = create_stacking_ensemble(list(zip(MODEL_NAMES, models)))
    stacking_model.fit(X, y)
    
    X_lstm, y_lstm = prepare_lstm_data(X, y)
    lstm_model = train_lstm(X_lstm, y_lstm)
    
    return stacking_model, lstm_model

def backtest(start_date: str, end_date: str, window_size: int = 90, step_size: int = 30) -> pd.DataFrame:
    """Perform backtesting."""
    X, y = prepare_data(start_date, end_date)
    results = []
    
    for i in range(0, len(X) - window_size, step_size):
        train_end = i + window_size
        test_end = min(train_end + step_size, len(X))
        
        X_train, y_train = X.iloc[i:train_end], y.iloc[i:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        
        stacking_model, lstm_model = train_models(X_train, y_train)
        
        stacking_pred = stacking_model.predict(X_test)
        X_lstm_test, _ = prepare_lstm_data(X_test, y_test)
        lstm_pred = lstm_model.predict(X_lstm_test).flatten()
        
        combined_pred = (stacking_pred + lstm_pred[:len(stacking_pred)]) / 2
        
        mse = mean_squared_error(y_test, combined_pred)
        mae = mean_absolute_error(y_test, combined_pred)
        r2 = r2_score(y_test, combined_pred)
        
        results.append({
            'start_date': X_test.index[0],
            'end_date': X_test.index[-1],
            'mse': mse,
            'mae': mae,
            'r2': r2
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = "2023-05-31"
    backtest_results = backtest(start_date, end_date)
    print(backtest_results)
    
    # You can add more analysis here, such as plotting the results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['end_date'], backtest_results['mse'], label='MSE')
    plt.plot(backtest_results['end_date'], backtest_results['mae'], label='MAE')
    plt.title('Backtest Results: Error Metrics Over Time')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('backtest_results.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['end_date'], backtest_results['r2'], label='R2 Score')
    plt.title('Backtest Results: R2 Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.savefig('backtest_r2.png')
    plt.close()

    logger.info("Backtesting completed. Results saved to backtest_results.png and backtest_r2.png")