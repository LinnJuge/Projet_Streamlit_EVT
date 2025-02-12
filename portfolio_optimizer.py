import numpy as np
import pandas as pd
from scipy.optimize import minimize

def equal_weight_allocation(n_assets):
    """ Retourne une allocation équipondérée. """
    return np.ones(n_assets) / n_assets

def min_variance_allocation(returns):
    """ Retourne l'allocation qui minimise la variance du portefeuille. """
    cov_matrix = returns.cov()

    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    n_assets = len(returns.columns)
    init_guess = np.ones(n_assets) / n_assets
    bounds = [(0, 1) for _ in range(n_assets)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(portfolio_volatility, init_guess, bounds=bounds, constraints=constraints)
    return result.x if result.success else equal_weight_allocation(n_assets)
