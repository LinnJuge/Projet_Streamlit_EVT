import numpy as np
import scipy.stats as stats
import pandas as pd

def calculate_var(data, confidence=0.95):
    """
    Calcule la Value at Risk (VaR) paramétrique pour chaque actif.
    Si plusieurs actifs sont sélectionnés, retourne un dictionnaire avec les valeurs.
    """
    var_results = {}

    # Calculer VaR pour chaque colonne (chaque actif sélectionné)
    for ticker in data.columns:
        mean_return = data[ticker].mean()
        std_dev = data[ticker].std()

        # Vérification : éviter les erreurs si std_dev est 0
        if pd.isnull(std_dev) or std_dev == 0:
            var_results[ticker] = np.nan
        else:
            var_results[ticker] = float(stats.norm.ppf(1 - confidence, mean_return, std_dev))

    return var_results  # Retourne un dictionnaire {ticker: var_value}

def monte_carlo_var(data, confidence=0.95, simulations=10000):
    """
    Calcule la Value at Risk (VaR) via simulation Monte Carlo
    :param data: DataFrame des rendements
    :param confidence: Niveau de confiance
    :param simulations: Nombre de simulations Monte Carlo
    :return: VaR Monte Carlo
    """
    simulated_returns = np.random.choice(data.values.flatten(), (simulations, len(data.columns)))
    portfolio_losses = np.sum(simulated_returns, axis=1)
    var_mc = np.percentile(portfolio_losses, (1 - confidence) * 100)
    return var_mc

def calculate_cvar(data, confidence=0.95):
    """
    Calcule la Conditional Value at Risk (CVaR) ou Expected Shortfall
    :param data: DataFrame des rendements
    :param confidence: Niveau de confiance
    :return: CVaR pour chaque actif
    """
    var = calculate_var(data, confidence)
    cvar = data[data <= var].mean()
    return cvar

def calculate_drawdown(prices):
    """
    Calcule le drawdown de chaque actif
    :param prices: DataFrame des prix ajustés
    :return: DataFrame des drawdowns
    """
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown

def max_drawdown(prices):
    """
    Calcule le Max Drawdown (perte maximale enregistrée)
    :param prices: DataFrame des prix ajustés
    :return: Max Drawdown pour chaque actif
    """
    drawdown = calculate_drawdown(prices)
    return drawdown.min()
