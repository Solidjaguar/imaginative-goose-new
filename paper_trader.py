import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import time
from gold_forex_predictor import fetch_all_data, prepare_data, add_technical_indicators, train_model
from trading_strategy import advanced_trading_strategy, apply_risk_management, calculate_indicators

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

    def run_paper_trading(self):
        model = train_model()  # Train the initial model
        
        while True:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)  # Fetch the last day of data
            
            data = fetch_all_data(start_date, end_date)
            X, y = prepare_data(data)
            
            predictions = model.predict(X)
            
            current_prices = {
                'EUR/USD': data['EUR/USD'].iloc[-1],
                'GBP/USD': data['GBP/USD'].iloc[-1],
                'JPY/USD': data['JPY/USD'].iloc[-1]
            }
            
            indicators = calculate_indicators(pd.Series(current_prices))
            
            for i, currency_pair in enumerate(['EUR/USD', 'GBP/USD', 'JPY/USD']):
                suggested_position = advanced_trading_strategy(indicators.iloc[-1], self.positions[currency_pair])
                position = apply_risk_management(suggested_position, self.entry_prices[currency_pair], current_prices[currency_pair])
                
                if position != self.positions[currency_pair]:
                    if position > self.positions[currency_pair]:
                        self.execute_trade(currency_pair, 'buy', position - self.positions[currency_pair], current_prices[currency_pair])
                    else:
                        self.execute_trade(currency_pair, 'sell', self.positions[currency_pair] - position, current_prices[currency_pair])
            
            portfolio_value = self.get_portfolio_value(current_prices)
            self.portfolio_values.append({
                'timestamp': datetime.now().isoformat(),
                'value': portfolio_value
            })
            
            logging.info(f"Current portfolio value: {portfolio_value}")
            
            # Save the current state
            self.save_state()
            
            # Wait for 1 hour before the next iteration
            time.sleep(3600)
            
            # Retrain the model every 24 hours
            if len(self.portfolio_values) % 24 == 0:
                model = train_model()

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