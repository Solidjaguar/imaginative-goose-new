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

def calculate_indicators(prices):
    """Calculate technical indicators"""
    df = pd.DataFrame(prices, columns=['close'])
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['close'])
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def advanced_trading_strategy(row, prev_position):
    """
    A more advanced trading strategy using technical indicators:
    - If price > SMA_50 and SMA_20 > SMA_50 and RSI < 70, buy (1)
    - If price < SMA_50 and SMA_20 < SMA_50 and RSI > 30, sell (-1)
    - Otherwise, hold (0)
    """
    if row['close'] > row['SMA_50'] and row['SMA_20'] > row['SMA_50'] and row['RSI'] < 70:
        return 1
    elif row['close'] < row['SMA_50'] and row['SMA_20'] < row['SMA_50'] and row['RSI'] > 30:
        return -1
    else:
        return prev_position

def apply_risk_management(position, entry_price, current_price, stop_loss_pct=0.02, take_profit_pct=0.04):
    """Apply stop-loss and take-profit"""
    if position != 0:
        stop_loss = entry_price * (1 - stop_loss_pct) if position == 1 else entry_price * (1 + stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct) if position == 1 else entry_price * (1 - take_profit_pct)
        
        if (position == 1 and current_price <= stop_loss) or (position == -1 and current_price >= stop_loss):
            return 0  # Stop-loss triggered
        elif (position == 1 and current_price >= take_profit) or (position == -1 and current_price <= take_profit):
            return 0  # Take-profit triggered
    
    return position

def calculate_returns(prices, positions, transaction_cost=0.0001, slippage=0.0001):
    """Calculate returns based on positions and price changes, including transaction costs and slippage"""
    price_changes = np.diff(prices) / prices[:-1]
    transaction_costs = np.abs(np.diff(positions)) * (transaction_cost + slippage)
    returns = positions[:-1] * price_changes - transaction_costs
    return returns

def backtest_strategy(model, X_test, y_test, currency_pair):
    predictions = model.predict(X_test)
    
    # Convert y_test (returns) to prices
    initial_price = 1.0
    prices = initial_price * (1 + y_test.cumsum())
    
    # Calculate technical indicators
    indicators = calculate_indicators(prices)
    
    # Initialize positions and entry prices
    positions = np.zeros(len(prices))
    entry_prices = np.zeros(len(prices))
    
    # Apply trading strategy with risk management
    for i in range(1, len(prices)):
        suggested_position = advanced_trading_strategy(indicators.iloc[i], positions[i-1])
        positions[i] = apply_risk_management(suggested_position, entry_prices[i-1], prices[i])
        
        if positions[i] != positions[i-1]:
            entry_prices[i] = prices[i]
        else:
            entry_prices[i] = entry_prices[i-1]
    
    strategy_returns = calculate_returns(prices, positions)
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    
    buy_and_hold_returns = (1 + y_test).cumprod() - 1
    
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
    
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
        'max_drawdown': max_drawdown,
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
        
        logging.info(f"Trading strategy results for {currency_pair}: Final Return={result['final_return']:.4f}, "
                     f"Sharpe Ratio={result['sharpe_ratio']:.4f}, Max Drawdown={result['max_drawdown']:.4f}")
    
    # Save results and plots
    with open('trading_strategy_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_trading_strategy()