import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedTradingSystem:
    def __init__(self, data):
        self.data = data
        self.strategies = {
            'moving_average_crossover': self.moving_average_crossover,
            'rsi_strategy': self.rsi_strategy,
            'bollinger_bands_strategy': self.bollinger_bands_strategy
        }

    def moving_average_crossover(self, short_window=10, long_window=50):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0

        signals['short_mavg'] = self.data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = self.data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                    > signals['long_mavg'][short_window:], 1.0, 0.0)   
        signals['positions'] = signals['signal'].diff()

        return signals

    def rsi_strategy(self, rsi_window=14, overbought=70, oversold=30):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0

        rsi = self.calculate_rsi(self.data['Close'], rsi_window)
        signals['rsi'] = rsi

        signals['signal'] = np.where(rsi > overbought, -1.0, np.where(rsi < oversold, 1.0, 0.0))
        signals['positions'] = signals['signal'].diff()

        return signals

    def bollinger_bands_strategy(self, window=20, num_std=2):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0

        rolling_mean = self.data['Close'].rolling(window=window).mean()
        rolling_std = self.data['Close'].rolling(window=window).std()
        
        signals['upper_band'] = rolling_mean + (rolling_std * num_std)
        signals['lower_band'] = rolling_mean - (rolling_std * num_std)

        signals['signal'] = np.where(self.data['Close'] > signals['upper_band'], -1.0, 
                                     np.where(self.data['Close'] < signals['lower_band'], 1.0, 0.0))
        signals['positions'] = signals['signal'].diff()

        return signals

    @staticmethod
    def calculate_rsi(prices, window=14):
        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum()/window
        down = -seed[seed < 0].sum()/window
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100./(1. + rs)

        for i in range(window, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(window - 1) + upval)/window
            down = (down*(window - 1) + downval)/window
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)

        return rsi

    def backtest_strategy(self, signals, initial_capital=100000.0):
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions['Asset'] = 100 * signals['signal']
        portfolio = positions.multiply(self.data['Close'], axis=0)

        pos_diff = positions.diff()
        portfolio['holdings'] = (positions.multiply(self.data['Close'], axis=0)).sum(axis=1)
        portfolio['cash'] = initial_capital - (pos_diff.multiply(self.data['Close'], axis=0)).sum(axis=1).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()

        return portfolio

    def optimize_strategy(self, strategy_name, param_grid):
        best_sharpe = -np.inf
        best_params = None

        for params in ParameterGrid(param_grid):
            signals = self.strategies[strategy_name](**params)
            portfolio = self.backtest_strategy(signals)
            sharpe = self.calculate_sharpe_ratio(portfolio['returns'])

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        return best_params, best_sharpe

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
        return np.sqrt(252) * (returns.mean() - risk_free_rate) / returns.std()

    def calculate_performance_metrics(self, portfolio):
        metrics = {}
        returns = portfolio['returns']

        metrics['Total Return'] = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
        metrics['Annualized Return'] = (1 + metrics['Total Return']) ** (252 / len(returns)) - 1
        metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)
        metrics['Sharpe Ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['Max Drawdown'] = (portfolio['total'] / portfolio['total'].cummax() - 1).min()
        metrics['Skewness'] = skew(returns)
        metrics['Kurtosis'] = kurtosis(returns)
        metrics['Value at Risk (95%)'] = np.percentile(returns, 5)
        metrics['Conditional Value at Risk (95%)'] = returns[returns <= metrics['Value at Risk (95%)']].mean()

        return metrics

    def plot_performance(self, portfolio, signals):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot asset price and signals
        ax1.plot(self.data.index, self.data['Close'])
        ax1.plot(signals.loc[signals['positions'] == 1.0].index, 
                 self.data['Close'][signals['positions'] == 1.0], 
                 '^', markersize=10, color='g')
        ax1.plot(signals.loc[signals['positions'] == -1.0].index, 
                 self.data['Close'][signals['positions'] == -1.0], 
                 'v', markersize=10, color='r')
        ax1.set_title('Asset Price and Trading Signals')
        ax1.set_ylabel('Price')

        # Plot portfolio value
        ax2.plot(portfolio.index, portfolio['total'])
        ax2.set_title('Portfolio Value')
        ax2.set_ylabel('Value')

        # Plot returns distribution
        sns.histplot(portfolio['returns'].dropna(), kde=True, ax=ax3)
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Returns')

        plt.tight_layout()
        plt.savefig('trading_performance.png')
        plt.close()

def run_advanced_trading():
    # Load data (replace this with your actual data loading logic)
    data = pd.read_csv('your_data.csv', index_col='Date', parse_dates=True)

    trading_system = AdvancedTradingSystem(data)

    # Optimize and run the moving average crossover strategy
    ma_param_grid = {'short_window': range(5, 30, 5), 'long_window': range(30, 100, 10)}
    best_ma_params, best_ma_sharpe = trading_system.optimize_strategy('moving_average_crossover', ma_param_grid)
    
    ma_signals = trading_system.moving_average_crossover(**best_ma_params)
    ma_portfolio = trading_system.backtest_strategy(ma_signals)
    ma_metrics = trading_system.calculate_performance_metrics(ma_portfolio)

    # Optimize and run the RSI strategy
    rsi_param_grid = {'rsi_window': range(10, 30, 2), 'overbought': range(65, 85, 5), 'oversold': range(15, 35, 5)}
    best_rsi_params, best_rsi_sharpe = trading_system.optimize_strategy('rsi_strategy', rsi_param_grid)
    
    rsi_signals = trading_system.rsi_strategy(**best_rsi_params)
    rsi_portfolio = trading_system.backtest_strategy(rsi_signals)
    rsi_metrics = trading_system.calculate_performance_metrics(rsi_portfolio)

    # Optimize and run the Bollinger Bands strategy
    bb_param_grid = {'window': range(10, 30, 5), 'num_std': [1.5, 2, 2.5]}
    best_bb_params, best_bb_sharpe = trading_system.optimize_strategy('bollinger_bands_strategy', bb_param_grid)
    
    bb_signals = trading_system.bollinger_bands_strategy(**best_bb_params)
    bb_portfolio = trading_system.backtest_strategy(bb_signals)
    bb_metrics = trading_system.calculate_performance_metrics(bb_portfolio)

    # Plot performance for the best strategy (assuming it's the one with the highest Sharpe ratio)
    best_portfolio = max([ma_portfolio, rsi_portfolio, bb_portfolio], key=lambda x: trading_system.calculate_sharpe_ratio(x['returns']))
    best_signals = max([ma_signals, rsi_signals, bb_signals], key=lambda x: trading_system.calculate_sharpe_ratio(trading_system.backtest_strategy(x)['returns']))
    trading_system.plot_performance(best_portfolio, best_signals)

    # Print results
    print("Moving Average Crossover Strategy:")
    print(f"Best parameters: {best_ma_params}")
    print(f"Performance metrics: {ma_metrics}")
    print("\nRSI Strategy:")
    print(f"Best parameters: {best_rsi_params}")
    print(f"Performance metrics: {rsi_metrics}")
    print("\nBollinger Bands Strategy:")
    print(f"Best parameters: {best_bb_params}")
    print(f"Performance metrics: {bb_metrics}")

if __name__ == "__main__":
    run_advanced_trading()