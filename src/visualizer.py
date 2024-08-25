import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def plot_predictions(data, predictions):
    for market in data.keys():
        plt.figure(figsize=(12, 6))
        plt.plot(data[market].index, data[market]['price'], label='Historical Data')
        plt.plot(predictions[market].index, predictions[market].values, label='Predictions', color='red')
        plt.title(f'{market} Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f'static/{market.replace("/", "_")}_predictions.png')
        plt.close()

def plot_performance(data, signals, portfolio):
    for market in data.keys():
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot asset price and signals
        ax1.plot(data[market].index, data[market]['price'])
        ax1.plot(signals[market].loc[signals[market]['positions'] == 1.0].index, 
                 data[market].loc[signals[market]['positions'] == 1.0, 'price'], 
                 '^', markersize=10, color='g')
        ax1.plot(signals[market].loc[signals[market]['positions'] == -1.0].index, 
                 data[market].loc[signals[market]['positions'] == -1.0, 'price'], 
                 'v', markersize=10, color='r')
        ax1.set_title(f'{market} Price and Trading Signals')
        ax1.set_ylabel('Price')

        # Plot portfolio value
        ax2.plot(portfolio[market].index, portfolio[market]['total'])
        ax2.set_title(f'{market} Portfolio Value')
        ax2.set_ylabel('Value')

        # Plot returns distribution
        sns.histplot(portfolio[market]['returns'].dropna(), kde=True, ax=ax3)
        ax3.set_title(f'{market} Returns Distribution')
        ax3.set_xlabel('Returns')

        plt.tight_layout()
        plt.savefig(f'static/{market.replace("/", "_")}_trading_performance.png')
        plt.close()

def calculate_performance_metrics(portfolio):
    metrics = {}
    for market, market_portfolio in portfolio.items():
        returns = market_portfolio['returns']

        market_metrics = {}
        market_metrics['Total Return'] = (market_portfolio['total'].iloc[-1] / market_portfolio['total'].iloc[0]) - 1
        market_metrics['Annualized Return'] = (1 + market_metrics['Total Return']) ** (252 / len(returns)) - 1
        market_metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)
        market_metrics['Sharpe Ratio'] = (returns.mean() - 0.01) / returns.std() * np.sqrt(252)
        market_metrics['Max Drawdown'] = (market_portfolio['total'] / market_portfolio['total'].cummax() - 1).min()
        market_metrics['Skewness'] = skew(returns)
        market_metrics['Kurtosis'] = kurtosis(returns)
        market_metrics['Value at Risk (95%)'] = np.percentile(returns, 5)
        market_metrics['Conditional Value at Risk (95%)'] = returns[returns <= market_metrics['Value at Risk (95%)']].mean()

        metrics[market] = market_metrics

    return metrics

if __name__ == "__main__":
    from data_fetcher import fetch_all_data
    from data_processor import prepare_data
    from predictor import predict_prices
    
    raw_data = fetch_all_data()
    prepared_data = prepare_data(raw_data)
    predictions = predict_prices(prepared_data)
    
    plot_predictions(prepared_data, predictions)
    print("Prediction plots saved in the static directory.")
    
    # Note: You'll need to implement the trading strategies and portfolio calculation
    # before you can use plot_performance and calculate_performance_metrics