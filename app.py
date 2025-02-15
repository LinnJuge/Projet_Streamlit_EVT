import streamlit as st
import pandas as pd
import numpy as np
from data_loader import get_data
from risk_indicators import var_historique, calculate_var, var_monte_carlo, calculate_cvar
from portfolio_allocation import equal_weighted_portfolio, min_variance_portfolio

# Configuration de l'application
st.set_page_config(page_title="Dashboard de Gestion des Risques", layout="wide")

# Sidebar - Sélection des actifs
st.sidebar.header("Paramètres de l'étude")
tickers = st.sidebar.multiselect("Sélectionner un ou plusieurs actifs", ["AAPL", "GOOGL", "MSFT", "SPY"])

# Sélection des dates
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("2023-12-31"))

# Mode d'analyse
mode = None
if len(tickers) > 1:
    mode = st.sidebar.radio("Mode d'analyse", ["Comparer", "Créer un portefeuille"])

# Chargement des données
if tickers:
    prices, returns = get_data(tickers, start_date, end_date)

    if mode == "Créer un portefeuille":
        allocation = st.sidebar.radio("Allocation", ["Équipondérée", "MinVariance", "Définir moi-même"])
        if allocation == "Définir moi-même":
            weights = np.array([st.sidebar.number_input(f"Poids {t}", 0.0, 1.0, 1/len(tickers), 0.01) for t in tickers])
            weights /= weights.sum()
        elif allocation == "Équipondérée":
            weights = equal_weighted_portfolio(returns)
        else:
            weights = min_variance_portfolio(returns)

        portfolio_returns = returns.dot(weights)
    else:
        portfolio_returns = returns  # Mode comparaison

    # Affichage des métriques
    st.write(f"**VaR Historique** : {var_historique(portfolio_returns)}")
    st.write(f"**VaR Paramétrique** : {calculate_var(portfolio_returns)}")
    st.write(f"**VaR Monte Carlo** : {var_monte_carlo(portfolio_returns)}")
    st.write(f"**CVaR** : {calculate_cvar(portfolio_returns)}")

