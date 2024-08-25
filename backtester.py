import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
import json
import logging
from gold_forex_predictor import fetch_all_data, prepare_data, add_technical_indicators

logging.basicConfig(filename='backtester.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def split_data(X, y, train_size):
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2, y_pred

def plot_backtest_results(dates, actual, predicted, currency_pair):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predicted, label='Predicted')
    plt.title(f'Backtest Results for {currency_pair}')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

def run_backtest(train_size=0.8, window_size=30):
    data = fetch_all_data()
    X, y = prepare_data(data)
    
    results = []
    plots = {}
    
    for i, currency_pair in enumerate(['EUR/USD', 'GBP/USD', 'JPY/USD']):
        X_train, X_test, y_train, y_test = split_data(X, y.iloc[:, i], train_size)
        
        test_dates = y_test.index
        actual_returns = y_test.values
        predicted_returns = []
        
        for j in range(0, len(X_test) - window_size, window_size):
            window_X_train = pd.concat([X_train, X_test.iloc[:j]])
            window_y_train = pd.concat([y_train, y_test.iloc[:j]])
            
            model = train_model(window_X_train, window_y_train)
            
            window_X_test = X_test.iloc[j:j+window_size]
            window_predictions = model.predict(window_X_test)
            predicted_returns.extend(window_predictions)
        
        # Evaluate the entire test set
        mse, mae, r2, _ = evaluate_model(model, X_test, y_test)
        
        results.append({
            'currency_pair': currency_pair,
            'mse': mse,
            'mae': mae,
            'r2': r2
        })
        
        # Generate plot
        plot = plot_backtest_results(test_dates, actual_returns, predicted_returns, currency_pair)
        plots[currency_pair] = plot
        
        logging.info(f"Backtest results for {currency_pair}: MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")
    
    # Save results and plots
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f)
    
    with open('backtest_plots.json', 'w') as f:
        json.dump(plots, f)

if __name__ == "__main__":
    run_backtest()