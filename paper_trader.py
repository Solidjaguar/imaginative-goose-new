import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import time
from gold_forex_predictor import fetch_all_data, prepare_data, add_advanced_features
from ensemble_model import StackingEnsembleModel, train_stacking_ensemble_model
from risk_management import apply_risk_management, calculate_var, calculate_cvar, maximum_drawdown, sharpe_ratio
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
        self.model = None
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

    def train_model(self, X, y):
        self.model = train_stacking_ensemble_model(X, y)
        logging.info("Stacking Ensemble Model trained")

    def generate_performance_report(self, historical_data):
        portfolio_values = pd.DataFrame(self.portfolio_values)
        portfolio_values['timestamp'] = pd.to_datetime(portfolio_values['timestamp'])
        portfolio_values.set_index('timestamp', inplace=True)
        portfolio_returns = portfolio_values['value'].pct_change().dropna()

        benchmark_returns = historical_data['EUR/USD'].pct_change().dropna()
        benchmark_returns = benchmark_returns[benchmark_returns.index.isin(portfolio_returns.index)]

        report = generate_performance_report(portfolio_returns, benchmark_returns)
        
        # Add advanced risk metrics
        report['Value at Risk (95%)'] = calculate_var(portfolio_returns)
        report['Conditional Value at Risk (95%)'] = calculate_cvar(portfolio_returns)
        report['Maximum Drawdown'] = maximum_drawdown(portfolio_returns)
        report['Sharpe Ratio'] = sharpe_ratio(portfolio_returns)
        
        with open('performance_report.json', 'w') as f:
            json.dump(report.to_dict(), f)
        
        logging.info("Performance report generated and saved")

    def run_paper_trading(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Use 1 year of historical data
        
        data = fetch_all_data(start_date, end_date)
        X, y = prepare_data(data)
        
        self.train_model(X, y)
        
        last_optimization = datetime.now()
        last_performance_report = datetime.now()
        
        while True:
            current_time = datetime.now()
            
            # Retrain model every 30 days
            if (current_time - last_optimization).days >= self.optimization_interval:
                data = fetch_all_data(current_time - timedelta(days=365), current_time)
                X, y = prepare_data(data)
                self.train_model(X, y)
                last_optimization = current_time
            
            # Generate performance report every 7 days
            if (current_time - last_performance_report).days >= self.performance_report_interval:
                self.generate_performance_report(data)
                last_performance_report = current_time
            
            current_prices = {
                'EUR/USD': data['EUR/USD'].iloc[-1],
                'GBP/USD': data['GBP/USD'].iloc[-1],
                'JPY/USD': data['JPY/USD'].iloc[-1]
            }
            
            latest_features = X.iloc[-1].values.reshape(1, -1)
            predictions = self.model.predict(latest_features)[0]
            
            for i, currency_pair in enumerate(['EUR/USD', 'GBP/USD', 'JPY/USD']):
                prediction = predictions[i]
                current_price = current_prices[currency_pair]
                entry_price = self.entry_prices[currency_pair]
                
                # Determine suggested position based on prediction
                if prediction > 0.001:  # Bullish
                    suggested_position = 1
                elif prediction < -0.001:  # Bearish
                    suggested_position = -1
                else:  # Neutral
                    suggested_position = 0
                
                # Calculate ATR for risk management
                atr = data[currency_pair].diff().abs().rolling(window=14).mean().iloc[-1]
                
                # Apply risk management
                position_size, stop_loss = apply_risk_management(
                    suggested_position, entry_price, current_price, self.balance, atr
                )
                
                # Execute trade
                if position_size > 0 and self.positions[currency_pair] <= 0:
                    self.execute_trade(currency_pair, 'buy', position_size, current_price)
                elif position_size < 0 and self.positions[currency_pair] >= 0:
                    self.execute_trade(currency_pair, 'sell', abs(position_size), current_price)
                
                # Check for stop loss
                if (self.positions[currency_pair] > 0 and current_price <= stop_loss) or \
                   (self.positions[currency_pair] < 0 and current_price >= stop_loss):
                    self.execute_trade(currency_pair, 'sell' if self.positions[currency_pair] > 0 else 'buy',
                                       abs(self.positions[currency_pair]), current_price)
            
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