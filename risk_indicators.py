import numpy as np
import scipy.stats as stats
import pandas as pd

def var_historique(portfolio_returns, confidence=0.95):
    """
    Calcule la VaR Historique :
    - Si un seul actif / portefeuille : retourne un float
    - Si plusieurs actifs : retourne un dict {ticker: VaR}
    """
    if isinstance(portfolio_returns, pd.Series):  # Un seul actif ou un portefeuille
        return abs(np.percentile(portfolio_returns.dropna(), (1 - confidence) * 100))
        #Retourne un float
    # Plusieurs actifs → Appliquer à chaque colonne
    return {ticker: abs(np.percentile(portfolio_returns[ticker].dropna(), (1 - confidence) * 100))
            for ticker in portfolio_returns.columns} 
    #Retourne un dict {ticker : valeur}


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





def annual_volatility(portfolio_returns, trading_days=252):
    """
    Calcule la volatilité annualisée :
    - Si un seul actif / portefeuille → retourne un float
    - Si plusieurs actifs → retourne un dict {ticker: volatilité}
    """
    if isinstance(portfolio_returns, pd.Series):  # Cas d'un actif ou portefeuille
        return portfolio_returns.std() * np.sqrt(trading_days)
    
    # Cas de plusieurs actifs → Appliquer à chaque colonne
    return {ticker: portfolio_returns[ticker].std() * np.sqrt(trading_days)
            for ticker in portfolio_returns.columns}



def ewma_volatility(portfolio_returns, lambda_=0.94):
    """
    Calcule la volatilité EWMA :
    - Si un seul actif / portefeuille → retourne un float
    - Si plusieurs actifs → retourne un dict {ticker: volatilité}
    """
    if isinstance(portfolio_returns, pd.Series):  # Cas d'un actif ou portefeuille
        squared_returns = portfolio_returns ** 2
        weights = (1 - lambda_) * lambda_ ** np.arange(len(squared_returns))[::-1]
        ewma_vol = np.sqrt(np.sum(weights * squared_returns) / np.sum(weights))
        return ewma_vol

    # Cas de plusieurs actifs → Appliquer à chaque colonne
    return {ticker: ewma_volatility(portfolio_returns[ticker], lambda_) for ticker in portfolio_returns.columns}


def semi_deviation(portfolio_returns):
    """
    Calcule la semi-déviation (volatilité des pertes) :
    - Si un seul actif / portefeuille → retourne un float
    - Si plusieurs actifs → retourne un dict {ticker: semi-deviation}
    """
    negative_returns = portfolio_returns[portfolio_returns < 0]  # Filtrer les rendements négatifs

    if isinstance(portfolio_returns, pd.Series):  # Cas d'un actif ou portefeuille
        return negative_returns.std() if not negative_returns.empty else 0.0

    # Cas de plusieurs actifs → Appliquer à chaque colonne
    return {ticker: negative_returns[ticker].std() if not negative_returns[ticker].dropna().empty else 0.0
            for ticker in portfolio_returns.columns}




def calculate_drawdown(prices):
    """
    Calcule le drawdown à partir des prix.
    - Si un seul actif → retourne une Series
    - Si plusieurs actifs → applique à chaque colonne et retourne un DataFrame
    """
    if isinstance(prices, pd.Series):  # Cas d'un seul actif
        peak = prices.cummax()  # Valeur maximale atteinte
        drawdown = (prices - peak) / peak
        return drawdown

    # Plusieurs actifs → Appliquer sur chaque colonne
    return prices.apply(lambda x: (x - x.cummax()) / x.cummax(), axis=0)


def max_drawdown(prices):
    """
    Calcule le Max Drawdown (drawdown le plus important atteint).
    - Si un seul actif → retourne un float
    - Si plusieurs actifs → applique à chaque colonne et retourne un dictionnaire {ticker: valeur}
    """
    drawdowns = calculate_drawdown(prices)

    if isinstance(prices, pd.Series):  # Cas d'un seul actif
        return drawdowns.min()

    # Plusieurs actifs → Appliquer sur chaque colonne
    return {ticker: drawdowns[ticker].min() for ticker in prices.columns}





import matplotlib.pyplot as plt
import seaborn as sns

def plot_var_cvar_graph(data, confidence, title="Distribution des Rendements & VaR/CVaR"):
    """
    Affiche un graphique de distribution des rendements avec la VaR et CVaR.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data, kde=True, bins=50, ax=ax, color="blue", label="Rendements")

    var_value = calculate_var(data, confidence)
    cvar_value = calculate_cvar(data, confidence)

    ax.axvline(-var_value, color='red', linestyle='dashed', label=f"VaR {confidence*100:.0f}%")
    ax.axvline(-cvar_value, color='orange', linestyle='dashed', label=f"CVaR {confidence*100:.0f}%")

    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def plot_correlation_heatmap(data, title="Matrice de Corrélation des Actifs"):
    """
    Affiche une heatmap de corrélation des actifs sélectionnés.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_comparison_graphs(data, title="Comparaison des Indicateurs", ylabel="Valeur", color_palette="Blues"):
    """
    Affiche un graphique comparatif des indicateurs de risque pour chaque actif.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=data.index, y=data.values, ax=ax, palette=color_palette)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)

