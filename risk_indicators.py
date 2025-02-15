import numpy as np
import scipy.stats as stats

def var_historique(portfolio_returns, confidence=0.95):
    """
    Calcule la VaR Historique :
    - Si un seul actif / portefeuille : retourne un float
    - Si plusieurs actifs : retourne un dict {ticker: VaR}
    """
    if isinstance(portfolio_returns, pd.Series):  # Un seul actif ou un portefeuille
        return abs(np.percentile(portfolio_returns.dropna(), (1 - confidence) * 100))
    
    # Plusieurs actifs → Appliquer à chaque colonne
    return {ticker: abs(np.percentile(portfolio_returns[ticker].dropna(), (1 - confidence) * 100))
            for ticker in portfolio_returns.columns}


def calculate_var(portfolio_returns, confidence=0.95):
    """
    Calcule la VaR Paramétrique :
    - Si un seul actif / portefeuille : retourne un float
    - Si plusieurs actifs : retourne un dict {ticker: VaR}
    """
    if isinstance(portfolio_returns, pd.Series):  # Un seul actif ou un portefeuille
        mean, std = portfolio_returns.mean(), portfolio_returns.std()
        return abs(float(stats.norm.ppf(1 - confidence, mean, std)))
    
    # Plusieurs actifs → Appliquer à chaque colonne
    return {ticker: abs(float(stats.norm.ppf(1 - confidence, portfolio_returns[ticker].mean(), portfolio_returns[ticker].std())))
            for ticker in portfolio_returns.columns}


def var_monte_carlo(portfolio_returns, confidence=0.95, simulations=10000):
    """
    Calcule la VaR Monte Carlo :
    - Si un seul actif / portefeuille : retourne un float
    - Si plusieurs actifs : retourne un dict {ticker: VaR}
    """
    if isinstance(portfolio_returns, pd.Series):  # Un seul actif ou un portefeuille
        mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
        simulated_returns = np.random.normal(mu, sigma, simulations)
        return abs(np.percentile(simulated_returns, (1 - confidence) * 100))
    
    # Plusieurs actifs → Appliquer à chaque colonne
    return {ticker: abs(np.percentile(np.random.normal(portfolio_returns[ticker].mean(), 
                                                        portfolio_returns[ticker].std(), 
                                                        simulations), 
                                      (1 - confidence) * 100))
            for ticker in portfolio_returns.columns}


def calculate_cvar(portfolio_returns, confidence=0.95):
    """
    Calcule le Conditional VaR (CVaR) :
    - Si un seul actif / portefeuille : retourne un float
    - Si plusieurs actifs : retourne un dict {ticker: CVaR}
    """
    if isinstance(portfolio_returns, pd.Series):  # Un seul actif ou un portefeuille
        var = calculate_var(portfolio_returns, confidence)
        losses = portfolio_returns[portfolio_returns <= -var]  # Sélection des pertes extrêmes
        return abs(losses.mean()) if not losses.empty else 0.0
    
    # Plusieurs actifs → Appliquer à chaque colonne
    return {ticker: abs(portfolio_returns[ticker][portfolio_returns[ticker] <= -calculate_var(portfolio_returns[ticker], confidence)].mean())
            if not portfolio_returns[ticker][portfolio_returns[ticker] <= -calculate_var(portfolio_returns[ticker], confidence)].empty 
            else 0.0
            for ticker in portfolio_returns.columns}

