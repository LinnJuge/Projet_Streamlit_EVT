import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
import yfinance as yf


def get_data(tickers, start, end):
    """
    Récupère les prix de clôture et les rendements log des actifs sélectionnés.
    - Gère les cas d'un seul ou plusieurs actifs correctement.
    - Vérifie et retourne `None, None` si les données sont vides.
    """
    df = yf.download(tickers, start=start, end=end)["Close"]  # 🔹 Téléchargement des prix
    
    if df.empty:
        print("⚠️ Aucune donnée récupérée. Vérifiez les tickers et la période sélectionnée.")
        return None, None

    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers)  # Convertir en DataFrame avec un nom explicite pour éviter les erreurs
    
    df.dropna(inplace=True)  # 🔹 Suppression des valeurs manquantes
    returns = np.log(df / df.shift(1)).dropna()  # 🔹 Calcul des rendements log

    return df, returns  # Retourne les prix et les rendements


def equal_weighted_portfolio(returns):
    """Crée un portefeuille équipondéré."""
    n = returns.shape[1]
    return np.ones(n) / n  

def min_variance_portfolio(returns):
    """Optimisation d'un portefeuille à variance minimale."""
    n = returns.shape[1]
    initial_guess = np.ones(n) / n  
    cov_matrix = returns.cov()

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.05, 0.95) for _ in range(n))

    result = minimize(portfolio_volatility, initial_guess, constraints=constraints, bounds=bounds)
    return result.x if result.success else initial_guess





import scipy.stats as stats

def get_portfolio_returns(returns, weights=None):
    """
    Calcule les rendements d'un portefeuille pondéré.
    - Retourne une **Series** si un seul actif, sinon un **DataFrame** bien formaté.
    - Gère le cas où `weights` est fourni pour un seul actif.
    """
    if returns is None or returns.empty:
        return None  # Si aucun retour n'est dispo, retour `None` pour éviter les erreurs

    if weights is not None:
        weights = np.array(weights).reshape(-1)  # Assurer un tableau 1D

        # Cas normal : Plusieurs actifs → DataFrame
        if isinstance(returns, pd.DataFrame):
            if len(weights) != returns.shape[1]:
                raise ValueError(f"🚨 Erreur : Nombre d'actifs ({returns.shape[1]}) ≠ Nombre de poids ({len(weights)})")
            return returns.dot(weights)  # Appliquer les poids

        # Cas particulier : Un seul actif → Convertir en DataFrame avant dot()
        elif isinstance(returns, pd.Series):
            return returns.to_frame().dot(weights)[0]  # Convertir en DataFrame puis extraire le scalair

    return returns  # Si pas de pondération, retourner directement les rendements


def var_historique(data, confidence=0.95, weights=None):
    """Calcule la VaR Historique"""
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:  # Vérifier si les rendements sont vides
        return 0.0

    return abs(np.percentile(data, (1 - confidence) * 100))


def calculate_var(data, confidence=0.95, weights=None):
    """Calcule la VaR Paramétrique"""
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:
        return 0.0  # Retourner 0 si aucun rendement valide

    mean = data.mean()
    std = data.std()

    # Cas où data est une Series (un seul actif ou portefeuille agrégé)
    if isinstance(std, (int, float, np.number)):  
        if std == 0:
            return abs(mean)  # Si volatilité nulle, retour moyenne absolue
        return abs(float(stats.norm.ppf(1 - confidence, mean, std)))

    # Cas où data est un DataFrame (plusieurs actifs)
    var_results = {}
    for asset in data.columns:
        if std[asset] == 0:
            var_results[asset] = abs(mean[asset])  # Gérer le cas de volatilité nulle
        else:
            var_results[asset] = abs(float(stats.norm.ppf(1 - confidence, mean[asset], std[asset])))

    return var_results  # Retourne un dictionnaire avec la VaR pour chaque actif


def var_monte_carlo(data, confidence=0.95, simulations=10000, weights=None):
    """Calcule la VaR Monte Carlo"""
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:
        return 0.0  # Retourner 0 si aucun rendement disponible

    mu, sigma = data.mean(), data.std()

    # Cas où data est une Series (portefeuille pondéré ou un seul actif)
    if isinstance(sigma, (int, float, np.number)):  
        if sigma == 0:
            return abs(mu)  # Si volatilité nulle, retourner la moyenne absolue

        simulated_returns = np.random.normal(mu, sigma, simulations)
        return abs(np.percentile(simulated_returns, (1 - confidence) * 100))

    # Cas où data est un DataFrame (plusieurs actifs)
    var_results = {}
    for asset in data.columns:
        if sigma[asset] == 0:
            var_results[asset] = abs(mu[asset])  # Si volatilité nulle, retour mean absolu
        else:
            simulated_returns = np.random.normal(mu[asset], sigma[asset], simulations)
            var_results[asset] = abs(np.percentile(simulated_returns, (1 - confidence) * 100))

    return var_results  # Retourne un dictionnaire avec la VaR pour chaque actif





def calculate_cvar(data, confidence=0.95, weights=None):
    """Calcule le Conditional VaR (CVaR)"""
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:
        return 0.0  # Si pas de rendements valides, retour 0

    var = calculate_var(data, confidence)

    # Cas où data est une Series (un seul actif ou portefeuille agrégé)
    if isinstance(var, (int, float, np.number)):  
        losses = data[data <= -var]  # Sélectionner uniquement les pertes extrêmes
        return abs(losses.mean()) if not losses.empty else 0.0

    # Cas où data est un DataFrame (plusieurs actifs)
    cvar_results = {}
    for asset in data.columns:
        losses = data[asset][data[asset] <= -var[asset]]  # Sélectionner uniquement les pertes sous la VaR
        cvar_results[asset] = abs(losses.mean()) if not losses.empty else 0.0  # Gérer le cas sans pertes sous la VaR

    return cvar_results  # Retourne un dictionnaire avec le CVaR pour chaque actif






# 📌 Fonction pour la semi-déviation (volatilité des pertes uniquement)
def semi_deviation(data, weights=None):
    """
    Calcule la semi-déviation (volatilité des pertes) pour un actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)
    negative_returns = data[data < 0]  # Sélection des rendements négatifs

    if isinstance(data, pd.Series):  # Cas d'un portefeuille ou actif unique
        return negative_returns.std() if not negative_returns.empty else 0.0

    # Cas où `data` est un DataFrame (plusieurs actifs)
    return {ticker: negative_returns[ticker].std() if not negative_returns[ticker].empty else 0.0
            for ticker in data.columns}




# 📌 Fonction pour la volatilité annualisée
def annual_volatility(data, trading_days=252, weights=None):
    """
    Calcule la volatilité annualisée pour un actif ou un portefeuille.
    """
    data = get_portfolio_returns(data, weights)

    if data.empty:  # Cas où il n'y a pas de données valides
        return 0.0 if isinstance(data, pd.Series) else {ticker: 0.0 for ticker in data.columns}

    vol = data.std() * np.sqrt(trading_days)  # Multiplication par √252 pour annualiser

    if isinstance(data, pd.Series):  # Cas d'un portefeuille ou actif unique
        return vol

    # Cas où `data` est un DataFrame (plusieurs actifs)
    return vol.to_dict()




# 📌 Fonction pour la volatilité EWMA
def ewma_volatility(data, lambda_=0.94, weights=None):
    """
    Calcule la volatilité EWMA (Exponentially Weighted Moving Average).
    """
    data = get_portfolio_returns(data, weights).dropna()

    if data.empty:
        return 0.0 if isinstance(data, pd.Series) else {ticker: 0.0 for ticker in data.columns}

    # Cas d'un actif unique (Series)
    if isinstance(data, pd.Series):  
        squared_returns = data ** 2
        weights_vector = (1 - lambda_) * lambda_ ** np.arange(len(squared_returns))[::-1]
        
        if np.sum(weights_vector) == 0:
            return 0.0  # Evite une division par zéro
        
        ewma_vol = np.sqrt(np.sum(weights_vector * squared_returns) / np.sum(weights_vector))
        return ewma_vol
    
    # Cas de plusieurs actifs
    return {ticker: ewma_volatility(data[ticker], lambda_) for ticker in data.columns}






def calculate_drawdown(prices, weights=None):
    """
    Calcule le Drawdown pour un actif ou un portefeuille.
    """
    prices = get_portfolio_returns(prices, weights).dropna()

    if prices.empty:
        return 0.0 if isinstance(prices, pd.Series) else {ticker: 0.0 for ticker in prices.columns}

    peak = prices.cummax()
    drawdown = (prices - peak) / peak

    return drawdown.fillna(0.0)  # Remplace les NaN par 0 (cas où aucun drawdown)




def max_drawdown(prices, weights=None):
    """
    Calcule le Maximum Drawdown (perte max depuis un sommet).
    """
    drawdowns = calculate_drawdown(prices, weights)

    if isinstance(drawdowns, pd.Series):
        return drawdowns.min() if not drawdowns.empty else 0.0

    return {ticker: drawdowns[ticker].min() if not drawdowns[ticker].empty else 0.0 for ticker in drawdowns.columns}
























































st.set_page_config(page_title="Dashboard de Risque", layout="wide")

st.sidebar.header("Paramètres de l'étude")
tickers = st.sidebar.multiselect("Sélectionner un ou plusieurs actifs", ["AAPL", "GOOGL", "MSFT", "SPY"])
mode = st.sidebar.radio("Mode d'analyse", ["Comparaison", "Portefeuille"])

start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("2023-12-31"))
confidence = st.sidebar.slider("Niveau de confiance pour la VaR", 0.90, 0.99, 0.95, step=0.01)

if tickers:
    prices, returns = get_data(tickers, start_date, end_date)

    weights = None
    if mode == "Portefeuille":
        allocation_type = st.sidebar.radio("Choix de l'allocation", ["Équipondérée", "MinVariance", "Définir moi-même"])
        if allocation_type == "Équipondérée":
            weights = equal_weighted_portfolio(returns)
        elif allocation_type == "MinVariance":
            weights = min_variance_portfolio(returns)

    portfolio_returns = get_portfolio_returns(returns, weights)

    st.subheader("📉 Indicateurs de Risque")
    st.write("VaR Paramétrique:", calculate_var(portfolio_returns, confidence))
    st.write("CVaR:", calculate_cvar(portfolio_returns, confidence))

    drawdowns = calculate_drawdown(prices, weights)
    st.line_chart(drawdowns)
