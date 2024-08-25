import numpy as np
import pandas as pd
from scipy import stats

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe ratio of a strategy.
    :param returns: pandas Series of strategy returns
    :param risk_free_rate: annualized risk-free rate
    :return: Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days in a year
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    """
    Calculate the Sortino ratio of a strategy.
    :param returns: pandas Series of strategy returns
    :param risk_free_rate: annualized risk-free rate
    :param target_return: target return (usually 0)
    :return: Sortino ratio
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation

def calculate_max_drawdown(returns):
    """
    Calculate the maximum drawdown of a strategy.
    :param returns: pandas Series of strategy returns
    :return: Maximum drawdown
    """
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()

def calculate_calmar_ratio(returns, period=36):
    """
    Calculate the Calmar ratio of a strategy.
    :param returns: pandas Series of strategy returns
    :param period: number of months to consider (default is 3 years)
    :return: Calmar ratio
    """
    total_return = (1 + returns).prod() - 1
    max_drawdown = calculate_max_drawdown(returns)
    return (total_return / period) / abs(max_drawdown)

def calculate_omega_ratio(returns, risk_free_rate=0.02, target_return=0):
    """
    Calculate the Omega ratio of a strategy.
    :param returns: pandas Series of strategy returns
    :param risk_free_rate: annualized risk-free rate
    :param target_return: target return (usually risk-free rate)
    :return: Omega ratio
    """
    excess_returns = returns - risk_free_rate / 252
    positive_returns = excess_returns[excess_returns > target_return]
    negative_returns = excess_returns[excess_returns <= target_return]
    return positive_returns.sum() / abs(negative_returns.sum())

def calculate_var(returns, confidence_level=0.95):
    """
    Calculate the Value at Risk (VaR) of a strategy.
    :param returns: pandas Series of strategy returns
    :param confidence_level: confidence level for VaR calculation
    :return: Value at Risk
    """
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculate the Conditional Value at Risk (CVaR) of a strategy.
    :param returns: pandas Series of strategy returns
    :param confidence_level: confidence level for CVaR calculation
    :return: Conditional Value at Risk
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_beta(returns, market_returns):
    """
    Calculate the beta of a strategy relative to a market benchmark.
    :param returns: pandas Series of strategy returns
    :param market_returns: pandas Series of market benchmark returns
    :return: Beta
    """
    covariance = np.cov(returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

def calculate_alpha(returns, market_returns, risk_free_rate=0.02):
    """
    Calculate the alpha of a strategy relative to a market benchmark.
    :param returns: pandas Series of strategy returns
    :param market_returns: pandas Series of market benchmark returns
    :param risk_free_rate: annualized risk-free rate
    :return: Alpha
    """
    beta = calculate_beta(returns, market_returns)
    strategy_return = returns.mean() * 252
    market_return = market_returns.mean() * 252
    return strategy_return - (risk_free_rate + beta * (market_return - risk_free_rate))

def calculate_information_ratio(returns, benchmark_returns):
    """
    Calculate the Information Ratio of a strategy.
    :param returns: pandas Series of strategy returns
    :param benchmark_returns: pandas Series of benchmark returns
    :return: Information Ratio
    """
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(252)
    return (excess_returns.mean() * 252) / tracking_error

def calculate_performance_metrics(returns, benchmark_returns, risk_free_rate=0.02):
    """
    Calculate all performance metrics for a strategy.
    :param returns: pandas Series of strategy returns
    :param benchmark_returns: pandas Series of benchmark returns
    :param risk_free_rate: annualized risk-free rate
    :return: Dictionary of performance metrics
    """
    metrics = {
        'Total Return': (1 + returns).prod() - 1,
        'Annualized Return': (1 + returns).prod() ** (252 / len(returns)) - 1,
        'Annualized Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'Sortino Ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'Max Drawdown': calculate_max_drawdown(returns),
        'Calmar Ratio': calculate_calmar_ratio(returns),
        'Omega Ratio': calculate_omega_ratio(returns, risk_free_rate),
        'Value at Risk (95%)': calculate_var(returns),
        'Conditional Value at Risk (95%)': calculate_cvar(returns),
        'Beta': calculate_beta(returns, benchmark_returns),
        'Alpha': calculate_alpha(returns, benchmark_returns, risk_free_rate),
        'Information Ratio': calculate_information_ratio(returns, benchmark_returns)
    }
    return metrics

def generate_performance_report(returns, benchmark_returns, risk_free_rate=0.02):
    """
    Generate a comprehensive performance report for a strategy.
    :param returns: pandas Series of strategy returns
    :param benchmark_returns: pandas Series of benchmark returns
    :param risk_free_rate: annualized risk-free rate
    :return: pandas DataFrame with performance metrics
    """
    metrics = calculate_performance_metrics(returns, benchmark_returns, risk_free_rate)
    report = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    report.index.name = 'Metric'
    return report