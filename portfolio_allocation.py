import numpy as np
import pandas as pd
from scipy.optimize import minimize

def equal_weighted_portfolio(returns):
    """
    Crée un portefeuille équipondéré.
    - Retourne un vecteur de poids égal (somme = 1).
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("🚨 Erreur : `returns` doit être un DataFrame contenant les rendements des actifs du portefeuille.")
    
    n = returns.shape[1]
    weights = np.ones(n) / n  # Poids égaux

    print(f"✅ Poids équipondérés calculés : {weights}")  # Debugging
    return weights

def min_variance_portfolio(returns):
    """
    Optimise un portefeuille à variance minimale.
    - Retourne un vecteur de poids optimisé (somme = 1).
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("🚨 Erreur : `returns` doit être un DataFrame contenant les rendements des actifs du portefeuille.")

    n = returns.shape[1]
    initial_guess = np.ones(n) / n  # Allocation de départ
    cov_matrix = returns.cov()  # Matrice de covariance

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # √(w'Σw)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Somme des poids = 1
    bounds = tuple((0.0, 1.0) for _ in range(n))  # Contraintes sur les poids (0% à 100%)

    result = minimize(portfolio_volatility, initial_guess, constraints=constraints, bounds=bounds)
    
    weights = result.x if result.success else initial_guess  # Si optimisation échoue, retourne équipondéré
    
    print(f"✅ Poids MinVariance calculés : {weights}")  # Debugging
    return weights


def get_portfolio_returns(returns, weights):
    """
    Calcule les rendements du portefeuille pondéré.
    - Retourne une **Series** représentant les rendements agrégés du portefeuille.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("🚨 Erreur : `returns` doit être un DataFrame contenant les rendements des actifs du portefeuille.")

    if weights is None or len(weights) != returns.shape[1]:
        raise ValueError(f"🚨 Erreur : Nombre d'actifs ({returns.shape[1]}) ≠ Nombre de poids ({len(weights)})")

    weights = np.array(weights).reshape(-1)  # Assurer un tableau 1D
    return returns.dot(weights)  # Appliquer les poids

