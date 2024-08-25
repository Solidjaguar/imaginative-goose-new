import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import time
from gold_forex_predictor import fetch_all_data, prepare_data, add_technical_indicators
from trading_strategy import calculate_indicators, combined_strategy, optimize_strategy, apply_risk_management, calculate_position_size
from performance_metrics import generate_performance_report

logging.basicConfig(filename='paper_trader.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class PaperTrader:
    def __init__(self, initial_balance=100000, transaction_cost=0.0001, slippage=0.0001):
        self.balance = initial_balance
        self.positions = {'EUR/USD': 0, 'GBP/USD': 0, 'JPY/USD': 0}
        self.entry_prices = {'EUR/USD': 0, 'GBP/USD': 0, 'JPY/USD': 0}
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.trade_history = []
        self.portfolio_values = []
        self.strategy = combined_strategy
        self.optimization_interval = 30  # Optimize strategy every 30 days
        self.performance_report_interval = 7  # Generate performance report every 7 days

    def execute_trade(self, currency_pair, action, amount, price):
        cost = amount * price * (1 + self.transaction_cost + self.slippage)
        if action == 'buy':
            if self.balance >= cost:
                self.balance -= cost
                self.positions[currency_pair] += amount
                self.entry_prices[currency_pair] = price
                self.trade_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'currency_pair': currency_pair,
                    'action': action,
                    'amount': amount,
                    'price': price,
                    'cost': cost
                })
                logging.info(f"Bought {amount} {currency_pair} at {price}")
            else:
                logging.warning(f"Insufficient balance to buy {amount} {currency_pair}")
        elif action == 'sell':
            if self.positions[currency_pair] >= amount:
                self.balance += amount * price * (1 - self.transaction_cost - self.slippage)
                self.positions[currency_pair] -= amount
                self.trade_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'currency_pair': currency_pair,
                    'action': action,
                    'amount': amount,
                    'price': price,
                    'cost': amount * price * (self.transaction_cost + self.slippage)
                })
                logging.info(f"Sold {amount} {currency_pair} at {price}")
            else:
                logging.warning(f"Insufficient {currency_pair} to sell {amount}")

    def get_portfolio_value(self, current_prices):
        portfolio_value = self.balance
        for currency_pair, amount in self.positions.items():
            portfolio_value += amount * current_prices[currency_pair]
        return portfolio_value

    def optimize_strategy(self, historical_data):
        self.strategy = optimize_strategy(historical_data, self.trade_history)
        logging.info("Strategy optimized based on historical data and paper trading results")

    def generate_performance_report(self, historical_data):
        portfolio_values = pd.DataFrame(self.portfolio_values)
        portfolio_values['timestamp'] = pd.to_datetime(portfolio_values['timestamp'])
        portfolio_values.set_index('timestamp', inplace=True)
        portfolio_returns = portfolio_values['value'].pct_change().dropna()

        benchmark_returns = historical_data['EUR/USD_returns']  # Using EUR/USD as benchmark
        benchmark_returns = benchmark_returns[benchmark_returns.index.isin(portfolio_returns.index)]

        report = generate_performance_report(portfolio_returns, benchmark_returns)
        
        with open('performance_report.json', 'w') as f:
            json.dump(report.to_dict(), f)
        
        logging.info("Performance report generated and saved")

    def run_paper_trading(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Use 1 year of historical data
        
        data = fetch_all_data(start_date, end_date)
        X, y = prepare_data(data)
        
        last_optimization = datetime.now()
        last_performance_report = datetime.now()
        
        while True:
            current_time = datetime.now()
            
            # Optimize strategy every 30 days
            if (current_time - last_optimization).days >= self.optimization_interval:
                self.optimize_strategy(pd.concat([X, y], axis=1))
                last_optimization = current_time
            
            # Generate performance report every 7 days
            if (current_time - last_performance_report).days >= self.performance_report_interval:
                self.generate_performance_report(pd.concat([X, y], axis=1))
                last_performance_report = current_time
            
            current_prices = {
                'EUR/USD': data['EUR/USD'].iloc[-1],
                'GBP/USD': data['GBP/USD'].iloc[-1],
                'JPY/USD': data['JPY/USD'].iloc[-1]
            }
            
            indicators = calculate_indicators(pd.Series(current_prices))
            
            for currency_pair in ['EUR/USD', 'GBP/USD', 'JPY/USD']:
                suggested_position = self.strategy(indicators.iloc[-1], self.positions[currency_pair], current_prices[currency_pair])
                position = apply_risk_management(suggested_position, self.entry_prices[currency_pair], current_prices[currency_pair])
                
                if position != self.positions[currency_pair]:
                    trade_amount = calculate_position_size(self.balance)
                    if position > self.positions[currency_pair]:
                        self.execute_trade(currency_pair, 'buy', trade_amount, current_prices[currency_pair])
                    else:
                        self.execute_trade(currency_pair, 'sell', trade_amount, current_prices[currency_pair])
            
            portfolio_value = self.get_portfolio_value(current_prices)
            self.portfolio_values.append({
                'timestamp': current_time.isoformat(),
                'value': portfolio_value
            })
            
            logging.info(f"Current portfolio value: {portfolio_value}")
            
            # Save the current state
            self.save_state()
            
            # Wait for 1 hour before the next iteration
            time.sleep(3600)
            
            # Fetch new data
            new_data = fetch_all_data(end_date, current_time)
            data = pd.concat([data, new_data])
            X_new, y_new = prepare_data(new_data)
            X = pd.concat([X, X_new])
            y = pd.concat([y, y_new])
            end_date = current_time

    def save_state(self):
        state = {
            'balance': self.balance,
            'positions': self.positions,
            'entry_prices': self.entry_prices,
            'trade_history': self.trade_history,
            'portfolio_values': self.portfolio_values
        }
        with open('paper_trading_state.json', 'w') as f:
            json.dump(state, f)

    def load_state(self):
        try:
            with open('paper_trading_state.json', 'r') as f:
                state = json.load(f)
            self.balance = state['balance']
            self.positions = state['positions']
            self.entry_prices = state['entry_prices']
            self.trade_history = state['trade_history']
            self.portfolio_values = state['portfolio_values']
            logging.info("Loaded previous paper trading state")
        except FileNotFoundError:
            logging.info("No previous state found, starting with initial values")

def run_paper_trading():
    paper_trader = PaperTrader()
    paper_trader.load_state()  # Load previous state if available
    paper_trader.run_paper_trading()

if __name__ == "__main__":
    run_paper_trading()