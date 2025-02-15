import numpy as np
import pandas as pd
import scipy.stats as stats

def var_historique(data, confidence=0.95):
    """Calcule la VaR Historique."""
    return abs(np.percentile(data.dropna(), (1 - confidence) * 100)) if not data.empty else 0.0

def calculate_var(data, confidence=0.95):
    """Calcule la VaR Paramétrique."""
    data = data.dropna()
    if data.empty: return 0.0
    mean, std = data.mean(), data.std()
    return abs(stats.norm.ppf(1 - confidence, mean, std)) if std != 0 else abs(mean)

def var_monte_carlo(data, confidence=0.95, simulations=10000):
    """Calcule la VaR Monte Carlo."""
    data = data.dropna()
    if data.empty: return 0.0
    mu, sigma = data.mean(), data.std()
    return abs(np.percentile(np.random.normal(mu, sigma, simulations), (1 - confidence) * 100)) if sigma != 0 else abs(mu)

def calculate_cvar(data, confidence=0.95):
    """Calcule le CVaR."""
    data = data.dropna()
    if data.empty: return 0.0
    var = calculate_var(data, confidence)
    losses = data[data <= -var]
    return abs(losses.mean()) if not losses.empty else 0.0

