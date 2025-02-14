import numpy as np
import pandas as pd
import scipy.stats as stats

def get_portfolio_returns(returns, weights=None):
    """
    Calcule les rendements d'un portefeuille pondéré.
    - Retourne un DataFrame ou une Series bien formatée.
    """
    if weights is not None:
        weights = np.array(weights).reshape(-1)

        # Vérifier si returns est un DataFrame
        if isinstance(returns, pd.DataFrame):
            if len(weights) != returns.shape[1]:
                raise ValueError(f"🚨 Erreur : Nombre d'actifs ({returns.shape[1]}) ≠ Nombre de poids ({len(weights)})")
            return returns.dot(weights)  # Appliquer les poids
    
    return returns.squeeze()  # Assurer qu'on retourne une Series si un seul actif


def var_historique(data, confidence=0.95, weights=None):
    """Calcule la VaR Historique"""
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:  # Vérifier si les rendements sont vides
        return 0.0

    return abs(np.percentile(data, (1 - confidence) * 100))


def calculate_var(data, confidence=0.95, weights=None):
    """Calcule la VaR Paramétrique"""
    data = get_portfolio_returns(data, weights).dropna()
    
    if data.empty:  # Vérification si les rendements sont vides
        return 0.0
    
    mean, std = data.mean(), data.std()
    
    if std == 0:  # Gérer le cas où la volatilité est nulle
        return abs(mean)  # Dans ce cas, on retourne la moyenne absolue

    return abs(float(stats.norm.ppf(1 - confidence, mean, std)))



def var_monte_carlo(data, confidence=0.95, simulations=10000, weights=None):
    """Calcule la VaR Monte Carlo"""
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:
        return 0.0  # Retourner 0 si aucun rendement disponible

    mu, sigma = data.mean(), data.std()

    if sigma == 0:  # Gérer le cas où la volatilité est nulle
        return abs(mu)  # Retourner la moyenne absolue

    simulated_returns = np.random.normal(mu, sigma, simulations)
    return abs(np.percentile(simulated_returns, (1 - confidence) * 100))

def calculate_cvar(data, confidence=0.95, weights=None):
    """Calcule le Conditional VaR (CVaR)"""
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:
        return 0.0  # Si pas de rendements valides, retour 0

    var = calculate_var(data, confidence)

    losses = data[data <= -var]  # Sélectionner uniquement les pertes extrêmes

    if losses.empty:  # Si aucune perte sous la VaR, retour 0
        return 0.0

    return abs(losses.mean())  # Moyenne des pertes sous la VaR


# 📌 Fonction pour la semi-déviation (volatilité des pertes uniquement)
def semi_deviation(data, weights=None):
    """
    Calcule la semi-déviation (volatilité des pertes) pour un actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    negative_returns = data[data < 0]  # Sélection des rendements négatifs

    if isinstance(data, pd.Series):  # Portefeuille ou actif unique
        return negative_returns.std() if not negative_returns.empty else 0.0

    return {ticker: negative_returns[ticker].std() if not negative_returns[ticker].dropna().empty else 0.0
            for ticker in data.columns}


    return {ticker: negative_returns[ticker].std() for ticker in data.columns}

# 📌 Fonction pour la volatilité annualisée
def annual_volatility(data, trading_days=252, weights=None):
    """
    Calcule la volatilité annualisée pour un actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    vol = data.std() * np.sqrt(trading_days)  # Multiplication par √252 pour annualiser

    if isinstance(data, pd.Series):  # Portefeuille ou actif unique
        return vol

    return {ticker: vol[ticker] for ticker in data.columns}  # Renvoie un dict pour plusieurs actifs

# 📌 Fonction pour la volatilité EWMA
def ewma_volatility(data, lambda_=0.94, weights=None):
    """
    Calcule la volatilité EWMA (Exponentially Weighted Moving Average).
    """
    data = get_portfolio_returns(data, weights)
    
    # Vérification si un seul actif ou plusieurs actifs
    if isinstance(data, pd.Series):  
        squared_returns = data ** 2
        weights_vector = (1 - lambda_) * lambda_ ** np.arange(len(squared_returns))[::-1]
        ewma_vol = np.sqrt(np.sum(weights_vector * squared_returns) / np.sum(weights_vector))
        return ewma_vol
    
    return {ticker: ewma_volatility(data[ticker], lambda_) for ticker in data.columns}  # Cas multi-actifs

def calculate_drawdown(prices, weights=None):
    """Calcule le Drawdown"""
    prices = get_portfolio_returns(prices, weights)
    peak = prices.cummax()
    return (prices - peak) / peak

def max_drawdown(prices, weights=None):
    """Calcule le Max Drawdown"""
    return calculate_drawdown(prices, weights).min()
