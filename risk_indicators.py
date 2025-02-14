import numpy as np
import pandas as pd
import scipy.stats as stats

def get_portfolio_returns(returns, weights=None):
    """
    Calcule les rendements d'un portefeuille si des poids sont fournis.
    """
    if weights is not None:
        weights = np.array(weights).reshape(-1)  # ✅ S'assurer que weights est un tableau 1D
        
        # 🔍 Debug pour vérifier que la taille des poids correspond au nombre d'actifs
        if isinstance(returns, pd.DataFrame) and weights.shape[0] != returns.shape[1]:
            raise ValueError(f"🚨 Erreur : Nombre d'actifs ({returns.shape[1]}) ≠ Nombre de poids ({weights.shape[0]})")
        
        if isinstance(returns, pd.DataFrame):  # ✅ Cas normal : plusieurs actifs
            return returns.dot(weights)
        elif isinstance(returns, pd.Series):  # ✅ Cas spécial : un seul actif ou portefeuille déjà pondéré
            return returns  # ✅ Ne pas appliquer weights une seconde fois
    return returns  # ✅ Retourne directement si pas de weights

def var_historique(data, confidence=0.95, weights=None):
    """
    Calcule la VaR Historique pour chaque actif ou un portefeuille.
    - Trie explicitement les rendements avant d'extraire le quantile.
    """
    data = get_portfolio_returns(data, weights)

    if isinstance(data, pd.Series):
        sorted_returns = np.sort(data.dropna())  # ✅ Tri explicite des rendements
        return abs(np.percentile(sorted_returns, (1 - confidence) * 100))

    return {ticker: abs(np.percentile(np.sort(data[ticker].dropna()), (1 - confidence) * 100)) for ticker in data.columns}

def calculate_var(data, confidence=0.95, weights=None):
    """
    VaR paramétrique pour chaque actif ou un portefeuille.
    """
    if weights is not None:
        data = (data * weights).sum(axis=1)

    if isinstance(data, pd.Series):
        return abs(float(stats.norm.ppf(1 - confidence, data.mean(), data.std())))

    return {ticker: abs(float(stats.norm.ppf(1 - confidence, data[ticker].mean(), data[ticker].std())))
            for ticker in data.columns}

def var_monte_carlo(data, confidence=0.95, simulations=10000, weights=None):
    """
    Calcule la VaR Monte Carlo pour chaque actif ou un portefeuille.
    - Utilise un modèle de diffusion géométrique brownien pour simuler les prix futurs.
    """
    data = get_portfolio_returns(data, weights)

    # Calcul des paramètres estimés
    mu = data.mean()  # Moyenne des rendements
    sigma = data.std()  # Volatilité des rendements
    P0 = 1  # On normalise le prix initial à 1 pour simplifier

    if isinstance(data, pd.Series):  # Cas d'un seul actif
        Z = np.random.normal(0, 1, simulations)  # Variables normales centrées réduites
        P_t1 = P0 * np.exp((mu - 0.5 * sigma**2) + sigma * Z)  # Simulation des prix futurs
        returns_mc = np.log(P_t1 / P0)  # Calcul des rendements simulés
        returns_mc.sort()
        return abs(np.percentile(returns_mc, (1 - confidence) * 100))  # Extraction de la VaR Monte Carlo

    # Cas où plusieurs actifs sont analysés individuellement
    var_results = {}
    for ticker in data.columns:
        Z = np.random.normal(0, 1, simulations)
        P_t1 = P0 * np.exp((mu[ticker] - 0.5 * sigma[ticker]**2) + sigma[ticker] * Z)
        returns_mc = np.log(P_t1 / P0)
        returns_mc.sort()
        var_results[ticker] = abs(np.percentile(returns_mc, (1 - confidence) * 100))

    return var_results

def calculate_cvar(data, confidence=0.95, weights=None):
    """
    Calcule le Conditional Value at Risk (CVaR ou Expected Shortfall) pour chaque actif ou un portefeuille.
    - data : dataframe des rendements
    - confidence : niveau de confiance (ex: 0.95 pour un CVaR à 95%)
    - weights : vecteur de poids pour un portefeuille
    """
    data = get_portfolio_returns(data, weights)  # ✅ Assure que data est bien pondéré

    # Vérifier que `calculate_var()` est bien définie
    if "calculate_var" not in globals():
        raise NameError("⚠️ La fonction calculate_var() n'est pas définie !")

    # Calcul de la VaR
    var = calculate_var(data, confidence)

    # 🔹 Correction pour éviter TypeError : appliquer - uniquement sur les valeurs numériques
    var = -var if isinstance(var, (int, float)) else {ticker: -value for ticker, value in var.items()}

    if isinstance(data, pd.Series):  # ✅ Cas d'un portefeuille ou d'un actif unique
        return abs(data[data <= var].mean())  # ✅ Moyenne des pertes sous la VaR

    # ✅ Cas de plusieurs actifs (individuellement)
    return {ticker: abs(data[ticker][data[ticker] <= var[ticker]].mean()) for ticker in data.columns}

# 📌 Fonction pour la semi-déviation (volatilité des pertes uniquement)
def semi_deviation(data, weights=None):
    """
    Calcule la semi-déviation (volatilité des pertes) pour chaque actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    negative_returns = data[data < 0]  # On ne garde que les rendements négatifs

    if isinstance(data, pd.Series):
        return negative_returns.std()

    return {ticker: negative_returns[ticker].std() for ticker in data.columns}

# 📌 Fonction pour la volatilité annualisée
def annual_volatility(data, trading_days=252, weights=None):
    """
    Calcule la volatilité annualisée pour chaque actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)

    if isinstance(data, pd.Series):
        return data.std() * np.sqrt(trading_days)

    return {ticker: data[ticker].std() * np.sqrt(trading_days) for ticker in data.columns}

# 📌 Fonction pour la volatilité EWMA
def ewma_volatility(data, lambda_=0.94, weights=None):
    """
    Calcule la volatilité EWMA pour chaque actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)

    if isinstance(data, pd.Series):
        squared_returns = data ** 2
        ewma_vol = [squared_returns.iloc[0]]  # Initialisation

        for r2 in squared_returns[1:]:
            ewma_vol.append(lambda_ * ewma_vol[-1] + (1 - lambda_) * r2)

        return np.sqrt(ewma_vol[-1])  # Dernière valeur = volatilité EWMA actuelle

    return {ticker: ewma_volatility(data[ticker]) for ticker in data.columns}


def calculate_drawdown(prices, weights=None):
    """
    Drawdown pour chaque actif ou un portefeuille.
    """
    if weights is not None and isinstance(prices, pd.DataFrame):
        prices = prices.dot(weights)  # ✅ Applique correctement les poids

    peak = prices.cummax()
    drawdown = (prices - peak) / peak

    return drawdown
    
def max_drawdown(prices, weights=None):
    """
    Max Drawdown pour chaque actif ou un portefeuille.
    """
    drawdown = calculate_drawdown(prices, weights)
    return drawdown.min()
