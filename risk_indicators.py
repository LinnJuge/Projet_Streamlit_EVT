import numpy as np
import pandas as pd
import scipy.stats as stats

def get_portfolio_returns(returns, weights=None):
    """
    Calcule les rendements d'un portefeuille si des poids sont fournis.
    """
    if weights is not None:
        weights = np.array(weights).reshape(-1)
        if isinstance(returns, pd.DataFrame):
            return returns.dot(weights)
        elif isinstance(returns, pd.Series):
            return returns
    return returns

def var_historique(data, confidence=0.95, weights=None):
    """
    Calcule la VaR Historique pour chaque actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    if isinstance(data, pd.Series):
        return abs(np.percentile(data.dropna(), (1 - confidence) * 100))
    return {ticker: abs(np.percentile(data[ticker].dropna(), (1 - confidence) * 100)) for ticker in data.columns}

def calculate_var(data, confidence=0.95, weights=None):
    """
    VaR paramétrique pour chaque actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    if isinstance(data, pd.Series):
        return abs(float(stats.norm.ppf(1 - confidence, data.mean(), data.std())))
    return {ticker: abs(float(stats.norm.ppf(1 - confidence, data[ticker].mean(), data[ticker].std()))) for ticker in data.columns}

def var_monte_carlo(data, confidence=0.95, simulations=10000, weights=None):
    """
    Calcule la VaR Monte Carlo pour chaque actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    mu, sigma = data.mean(), data.std()
    P0 = 1
    
    if isinstance(data, pd.Series):
        Z = np.random.normal(0, 1, simulations)
        P_t1 = P0 * np.exp((mu - 0.5 * sigma**2) + sigma * Z)
        returns_mc = np.log(P_t1 / P0)
        return abs(np.percentile(returns_mc, (1 - confidence) * 100))
    
    return {ticker: abs(np.percentile(np.log(P0 * np.exp((mu[ticker] - 0.5 * sigma[ticker]**2) + sigma[ticker] * np.random.normal(0, 1, simulations)) / P0), (1 - confidence) * 100)) for ticker in data.columns}

def calculate_cvar(data, confidence=0.95, weights=None):
    """
    Calcule le Conditional VaR (CVaR).
    """
    data = get_portfolio_returns(data, weights)
    var = calculate_var(data, confidence)
    
    var = -var if isinstance(var, (int, float)) else {ticker: -value for ticker, value in var.items()}
    if isinstance(data, pd.Series):
        return abs(data[data <= var].mean())
    return {ticker: abs(data[ticker][data[ticker] <= var[ticker]].mean()) for ticker in data.columns}

def semi_deviation(data, weights=None):
    """
    Calcule la semi-déviation (volatilité des pertes) pour chaque actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    negative_returns = data[data < 0]
    if isinstance(data, pd.Series):
        return negative_returns.std()
    return {ticker: negative_returns[ticker].std() for ticker in data.columns}

def annual_volatility(data, trading_days=252, weights=None):
    """
    Calcule la volatilité annualisée pour chaque actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    if isinstance(data, pd.Series):
        return data.std() * np.sqrt(trading_days)
    return {ticker: data[ticker].std() * np.sqrt(trading_days) for ticker in data.columns}

def ewma_volatility(data, lambda_=0.94, weights=None):
    """
    Calcule la volatilité EWMA pour chaque actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    if isinstance(data, pd.Series):
        squared_returns = data ** 2
        ewma_vol = [squared_returns.iloc[0]]
        for r2 in squared_returns[1:]:
            ewma_vol.append(lambda_ * ewma_vol[-1] + (1 - lambda_) * r2)
        return np.sqrt(ewma_vol[-1])
    return {ticker: ewma_volatility(data[ticker]) for ticker in data.columns}

def calculate_drawdown(prices, weights=None):
    """
    Calcule le drawdown pour chaque actif ou un portefeuille.
    """
    if weights is not None:
        prices = (prices * weights).sum(axis=1)
    peak = prices.cummax()
    return (prices - peak) / peak

def max_drawdown(prices, weights=None):
    """
    Calcule le max drawdown pour chaque actif ou un portefeuille.
    """
    drawdown = calculate_drawdown(prices, weights)
    return drawdown.min()
