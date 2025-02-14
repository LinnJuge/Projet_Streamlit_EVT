import numpy as np
from scipy.optimize import minimize

def equal_weighted_portfolio(returns):
    """
    Calcule un portefeuille équipondéré.
    """
    n = len(returns.columns)
    return np.ones(n) / n  

def min_variance_portfolio(returns):
    """
    Optimisation d'un portefeuille à variance minimale.
    """
    n = len(returns.columns)
    initial_guess = np.ones(n) / n  
    cov_matrix = returns.cov()

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.05, 0.95) for _ in range(n))

    result = minimize(portfolio_volatility, initial_guess, constraints=constraints, bounds=bounds)
    return result.x if result.success else initial_guess 
