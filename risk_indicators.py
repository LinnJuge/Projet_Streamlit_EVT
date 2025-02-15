import numpy as np
import pandas as pd
import scipy.stats as stats

def get_portfolio_returns(returns, weights=None):
    """
    Calcule les rendements d'un portefeuille pondéré.
    """
    if weights is not None:
        weights = np.array(weights).reshape(-1)
        if len(weights) != returns.shape[1]:
            raise ValueError(f"Erreur : {returns.shape[1]} actifs ≠ {len(weights)} poids")
        return returns.dot(weights)
    
    return returns  # Retourne directement les rendements bruts

def calculate_var(data, confidence=0.95, weights=None):
    """Calcule la VaR Paramétrique."""
    data = get_portfolio_returns(data, weights).dropna()
    
    if data.empty:
        return 0.0  

    mean, std = data.mean(), data.std()
    
    if isinstance(std, (int, float, np.number)):  
        if std == 0:
            return abs(mean)
        return abs(stats.norm.ppf(1 - confidence, mean, std))
    
    return {asset: abs(stats.norm.ppf(1 - confidence, mean[asset], std[asset])) for asset in data.columns}

def var_historique(data, confidence=0.95, weights=None):
    """Calcule la VaR Historique."""
    data = get_portfolio_returns(data, weights).dropna()
    
    if data.empty:
        return 0.0  

    return abs(np.percentile(data, (1 - confidence) * 100))

def var_monte_carlo(data, confidence=0.95, simulations=10000, weights=None):
    """Calcule la VaR Monte Carlo."""
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:
        return 0.0  

    mu, sigma = data.mean(), data.std()
    
    if isinstance(sigma, (int, float, np.number)):  
        if sigma == 0:
            return abs(mu)
        simulated_returns = np.random.normal(mu, sigma, simulations)
        return abs(np.percentile(simulated_returns, (1 - confidence) * 100))
    
    return {asset: abs(np.percentile(np.random.normal(mu[asset], sigma[asset], simulations), (1 - confidence) * 100))
            for asset in data.columns}

def calculate_cvar(data, confidence=0.95, weights=None):
    """Calcule le CVaR."""
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:
        return 0.0  

    var = calculate_var(data, confidence)
    
    if isinstance(var, (int, float, np.number)):  
        losses = data[data <= -var]
        return abs(losses.mean()) if not losses.empty else 0.0

    return {asset: abs(data[asset][data[asset] <= -var[asset]].mean()) if not data[asset][data[asset] <= -var[asset]].empty else 0.0
            for asset in data.columns}
