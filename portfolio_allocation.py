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


def get_portfolio_returns(returns, weights):
    """
    Calcule les rendements du portefeuille pondéré.
    - Retourne une **Series** si un seul actif, sinon un **DataFrame** bien formaté.
    """
    if weights is not None:
        weights = np.array(weights).reshape(-1)  # Assurer un tableau 1D

        # Cas normal : Plusieurs actifs → DataFrame
        if isinstance(returns, pd.DataFrame):
            if len(weights) != returns.shape[1]:
                raise ValueError(f"🚨 Erreur : Nombre d'actifs ({returns.shape[1]}) ≠ Nombre de poids ({len(weights)})")
            return returns.dot(weights)  # Appliquer les poids

        # Cas particulier : Un seul actif → Convertir en DataFrame avant dot()
        elif isinstance(returns, pd.Series):
            return returns.to_frame().dot(weights)[0]  # Convertir en DataFrame puis extraire le scalaire

    return returns  # Si pas de pondération, retourner directement les rendements

