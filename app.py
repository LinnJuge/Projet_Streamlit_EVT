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
