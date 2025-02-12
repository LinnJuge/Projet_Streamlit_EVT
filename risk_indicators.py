import numpy as np
import scipy.stats as stats
import pandas as pd

def calculate_var(data, confidence=0.95):
    """
    Calcule la Value at Risk (VaR) paramétrique pour chaque actif.
    Retourne un dictionnaire {ticker: VaR_value}.
    """
    var_results = {}
    for ticker in data.columns:
        mean_return = data[ticker].mean()
        std_dev = data[ticker].std()

        if pd.isnull(std_dev) or std_dev == 0:
            var_results[ticker] = np.nan
        else:
            var_results[ticker] = float(stats.norm.ppf(1 - confidence, mean_return, std_dev))

    return var_results  

def monte_carlo_var(data, confidence=0.95, simulations=10000):
    """
    Calcule la Value at Risk (VaR) via simulation Monte Carlo.
    """
    simulated_returns = np.random.choice(data.values.flatten(), (simulations, len(data.columns)))
    portfolio_losses = np.sum(simulated_returns, axis=1)
    var_mc = np.percentile(portfolio_losses, (1 - confidence) * 100)
    return var_mc

def calculate_cvar(data, confidence=0.95):
    """
    Calcule la Conditional Value at Risk (CVaR) ou Expected Shortfall.
    """
    var = calculate_var(data, confidence)
    cvar_results = {ticker: data[ticker][data[ticker] <= var[ticker]].mean() for ticker in data.columns}
    return cvar_results

def calculate_drawdown(prices):
    """
    Calcule le drawdown de chaque actif.
    """
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown

def max_drawdown(prices):
    """
    Calcule le Max Drawdown (perte maximale enregistrée).
    """
    drawdown = calculate_drawdown(prices)
    return drawdown.min().to_dict()

def calculate_volatility(data):
    """
    Calcule la volatilité classique et la volatilité exponentielle EWMA.
    """
    vol_results = data.std()  # Volatilité simple
    ewma_vol_results = data.ewm(span=30).std().iloc[-1]  # EWMA avec lambda=0.94

    return vol_results.to_dict(), ewma_vol_results.to_dict()

def ewma_var(data, confidence=0.95, lambda_=0.94):
    """
    Calcule le VaR en utilisant l'estimateur de volatilité EWMA.
    """
    ewma_vol = data.ewm(span=30).std().iloc[-1]
    mean_return = data.mean()
    var_results = {ticker: stats.norm.ppf(1 - confidence, mean_return[ticker], ewma_vol[ticker]) for ticker in data.columns}
    return var_results
