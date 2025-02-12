import numpy as np
import scipy.stats as stats
import pandas as pd

def calculate_var(data, confidence=0.95):
    """ Calcule la VaR paramétrique pour chaque actif. """
    return {ticker: data[ticker].mean() + stats.norm.ppf(1 - confidence) * data[ticker].std() for ticker in data.columns}

def monte_carlo_var(data, confidence=0.95, simulations=10000):
    """ Calcule la VaR Monte Carlo. """
    simulated_returns = np.random.choice(data.values.flatten(), (simulations, len(data.columns)))
    portfolio_losses = np.sum(simulated_returns, axis=1)
    return np.percentile(portfolio_losses, (1 - confidence) * 100)

def calculate_cvar(data, confidence=0.95):
    """ Calcule la CVaR pour chaque actif. """
    var = calculate_var(data, confidence)
    return {ticker: data[ticker][data[ticker] <= var[ticker]].mean() for ticker in data.columns}

def calculate_drawdown(prices):
    """ Calcule le drawdown pour chaque actif. """
    peak = prices.cummax()
    return (prices - peak) / peak

def max_drawdown(prices):
    """ Calcule le maximum drawdown pour chaque actif. """
    return calculate_drawdown(prices).min().to_dict()

