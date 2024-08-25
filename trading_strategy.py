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

logging.basicConfig(filename='trading_strategy.log', level=logging.INFO,
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

def simple_trading_strategy(predictions, threshold=0.0001):
    """
    A simple trading strategy:
    - If predicted return > threshold, buy (1)
    - If predicted return < -threshold, sell (-1)
    - Otherwise, hold (0)
    """
    return np.where(predictions > threshold, 1, np.where(predictions < -threshold, -1, 0))

def calculate_returns(prices, positions):
    """Calculate returns based on positions and price changes"""
    price_changes = np.diff(prices) / prices[:-1]
    return positions[:-1] * price_changes

def backtest_strategy(model, X_test, y_test, currency_pair):
    predictions = model.predict(X_test)
    positions = simple_trading_strategy(predictions)
    
    # Assuming y_test contains actual returns, we need to convert it to prices
    initial_price = 1.0
    prices = initial_price * (1 + y_test.cumsum())
    
    strategy_returns = calculate_returns(prices, positions)
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    
    buy_and_hold_returns = (1 + y_test).cumprod() - 1
    
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, cumulative_returns, label='Trading Strategy')
    plt.plot(y_test.index, buy_and_hold_returns, label='Buy and Hold')
    plt.title(f'Trading Strategy vs Buy and Hold for {currency_pair}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    plot = base64.b64encode(img.getvalue()).decode()
    
    return {
        'currency_pair': currency_pair,
        'final_return': cumulative_returns[-1],
        'sharpe_ratio': sharpe_ratio,
        'plot': plot
    }

def run_trading_strategy(train_size=0.8):
    data = fetch_all_data()
    X, y = prepare_data(data)
    
    results = []
    
    for i, currency_pair in enumerate(['EUR/USD', 'GBP/USD', 'JPY/USD']):
        X_train, X_test, y_train, y_test = split_data(X, y.iloc[:, i], train_size)
        
        model = train_model(X_train, y_train)
        
        result = backtest_strategy(model, X_test, y_test, currency_pair)
        results.append(result)
        
        logging.info(f"Trading strategy results for {currency_pair}: Final Return={result['final_return']:.4f}, Sharpe Ratio={result['sharpe_ratio']:.4f}")
    
    # Save results and plots
    with open('trading_strategy_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_trading_strategy()