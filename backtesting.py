import pandas as pd
import numpy as np
from trading_strategy import calculate_indicators, sma_crossover_strategy, rsi_strategy, macd_strategy, bollinger_bands_strategy, combined_strategy
from performance_metrics import calculate_performance_metrics

class Backtester:
    def __init__(self, data, initial_balance=100000, transaction_cost=0.0001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

    def run_backtest(self, strategy_func):
        balance = self.initial_balance
        position = 0
        trades = []
        portfolio_values = [balance]

        for i in range(1, len(self.data)):
            current_price = self.data.iloc[i]
            previous_price = self.data.iloc[i-1]
            indicators = calculate_indicators(self.data.iloc[:i+1])

            if strategy_func == combined_strategy:
                action = strategy_func(indicators.iloc[-1], position, current_price)
            else:
                action = strategy_func(indicators.iloc[-1], position)

            if action == 1 and position <= 0:  # Buy
                shares_to_buy = balance / current_price
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if balance >= cost:
                    balance -= cost
                    position += shares_to_buy
                    trades.append(('buy', current_price, shares_to_buy, cost))
            elif action == -1 and position >= 0:  # Sell
                shares_to_sell = abs(position)
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                balance += revenue
                position -= shares_to_sell
                trades.append(('sell', current_price, shares_to_sell, revenue))

            portfolio_value = balance + position * current_price
            portfolio_values.append(portfolio_value)

        return pd.Series(portfolio_values, index=self.data.index)

    def compare_strategies(self):
        strategies = {
            'SMA Crossover': sma_crossover_strategy,
            'RSI': rsi_strategy,
            'MACD': macd_strategy,
            'Bollinger Bands': bollinger_bands_strategy,
            'Combined': combined_strategy
        }

        results = {}
        for name, strategy in strategies.items():
            portfolio_values = self.run_backtest(strategy)
            returns = portfolio_values.pct_change().dropna()
            metrics = calculate_performance_metrics(returns, self.data['EUR/USD'].pct_change().dropna())
            results[name] = metrics

        return pd.DataFrame(results).T

def run_backtests(data):
    backtester = Backtester(data)
    results = backtester.compare_strategies()
    
    # Save results to a file
    results.to_csv('backtest_results.csv')
    
    print("Backtest results:")
    print(results)
    
    return results

if __name__ == "__main__":
    # For testing purposes
    data = pd.DataFrame({
        'EUR/USD': np.random.randn(1000).cumsum() + 100,
        'GBP/USD': np.random.randn(1000).cumsum() + 150,
        'JPY/USD': np.random.randn(1000).cumsum() + 1,
    }, index=pd.date_range(start='2020-01-01', periods=1000))
    
    run_backtests(data)