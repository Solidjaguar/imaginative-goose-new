import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def plot_predictions(data, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, label='Historical Data')
    plt.plot(predictions.index, predictions.values, label='Predictions', color='red')
    plt.title('Gold Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('static/gold_predictions.png')
    plt.close()

def plot_performance(data, signals, portfolio):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Plot asset price and signals
    ax1.plot(data.index, data.values)
    ax1.plot(signals.loc[signals['positions'] == 1.0].index, 
             data[signals['positions'] == 1.0], 
             '^', markersize=10, color='g')
    ax1.plot(signals.loc[signals['positions'] == -1.0].index, 
             data[signals['positions'] == -1.0], 
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
    plt.savefig('static/trading_performance.png')
    plt.close()

def calculate_performance_metrics(portfolio):
    metrics = {}
    returns = portfolio['returns']

    metrics['Total Return'] = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
    metrics['Annualized Return'] = (1 + metrics['Total Return']) ** (252 / len(returns)) - 1
    metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)
    metrics['Sharpe Ratio'] = (returns.mean() - 0.01) / returns.std() * np.sqrt(252)
    metrics['Max Drawdown'] = (portfolio['total'] / portfolio['total'].cummax() - 1).min()
    metrics['Skewness'] = skew(returns)
    metrics['Kurtosis'] = kurtosis(returns)
    metrics['Value at Risk (95%)'] = np.percentile(returns, 5)
    metrics['Conditional Value at Risk (95%)'] = returns[returns <= metrics['Value at Risk (95%)']].mean()

    return metrics