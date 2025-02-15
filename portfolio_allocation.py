import numpy as np
from scipy.optimize import minimize

def equal_weighted_portfolio(returns):
    """
    Crée un portefeuille équipondéré.
    - Retourne un vecteur de poids égal (somme = 1).
    """
    n = returns.shape[1]
    return np.ones(n) / n  # Poids égaux pour chaque actif

def min_variance_portfolio(returns):
    """
    Optimise un portefeuille à variance minimale.
    - Retourne un vecteur de poids optimisé (somme = 1).
    """
    n = returns.shape[1]
    initial_guess = np.ones(n) / n  # Départ avec allocation équipondérée
    cov_matrix = returns.cov()  # Matrice de covariance

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # √(w'Σw)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Somme des poids = 1
    bounds = tuple((0.05, 0.95) for _ in range(n))  # Contraintes sur les poids

    result = minimize(portfolio_volatility, initial_guess, constraints=constraints, bounds=bounds)
    return result.x if result.success else initial_guess  # Si optimisation échoue, retourne équipondéré

