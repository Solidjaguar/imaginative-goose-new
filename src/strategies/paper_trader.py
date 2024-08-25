import json
from datetime import datetime, timedelta

class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.gold_holdings = 0
        self.trades = []
        self.load_state()

    def load_state(self):
        try:
            with open('paper_trader_state.json', 'r') as f:
                state = json.load(f)
                self.balance = state['balance']
                self.gold_holdings = state['gold_holdings']
                self.trades = state['trades']
        except FileNotFoundError:
            self.save_state()

    def save_state(self):
        state = {
            'balance': self.balance,
            'gold_holdings': self.gold_holdings,
            'trades': self.trades
        }
        with open('paper_trader_state.json', 'w') as f:
            json.dump(state, f)

    def buy(self, price, amount):
        cost = price * amount
        if cost <= self.balance:
            self.balance -= cost
            self.gold_holdings += amount
            self.trades.append({
                'type': 'buy',
                'price': price,
                'amount': amount,
                'date': datetime.now().isoformat()
            })
            self.save_state()
            return True
        return False

    def sell(self, price, amount):
        if amount <= self.gold_holdings:
            self.balance += price * amount
            self.gold_holdings -= amount
            self.trades.append({
                'type': 'sell',
                'price': price,
                'amount': amount,
                'date': datetime.now().isoformat()
            })
            self.save_state()
            return True
        return False

    def get_portfolio_value(self, current_price):
        return self.balance + (self.gold_holdings * current_price)

    def get_recent_trades(self, hours=24):
        cutoff_date = datetime.now() - timedelta(hours=hours)
        return [trade for trade in self.trades if datetime.fromisoformat(trade['date']) > cutoff_date]

paper_trader = PaperTrader()