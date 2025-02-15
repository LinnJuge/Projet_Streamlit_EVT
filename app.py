import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from risk_indicators import *  # Import des fonctions de risk_indicators.py
from portfolio_optimization import *  # Import des fonctions de portfolio_optimization.py

# Configuration de l'application
st.set_page_config(page_title="Dashboard de Risque", layout="wide")

# Sidebar - Sélection des actifs
st.sidebar.header("Paramètres de l'étude")
tickers = st.sidebar.multiselect("Sélectionner un ou plusieurs actifs", ["AAPL", "GOOGL", "MSFT", "SPY"])

# Choix du mode (Comparaison ou Portefeuille)
mode = st.sidebar.radio("Mode d'analyse", ["Comparaison", "Portefeuille"])

# Allocation du portefeuille
if mode == "Portefeuille":
    allocation_type = st.sidebar.radio("Choix de l'allocation", ["Équipondérée", "MinVariance", "Définir moi-même"])
    
    if allocation_type == "Définir moi-même":
        user_weights = {}
        for ticker in tickers:
            user_weights[ticker] = st.sidebar.number_input(f"Poids de {ticker}", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        user_weights = np.array(list(user_weights.values()))
        user_weights /= user_weights.sum()  # Normalisation pour que la somme fasse 1
    else:
        user_weights = None

# Sélection des dates
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("2023-12-31"))

# Niveau de confiance
confidence = st.sidebar.slider("Niveau de confiance pour la VaR", 0.90, 0.99, 0.95, step=0.01)

# Chargement des données
if tickers:
    prices, returns = get_data(tickers, start_date, end_date)
    
    if mode == "Portefeuille":
        if allocation_type == "Équipondérée":
            weights = equal_weighted_portfolio(returns)
            portfolio_returns = get_portfolio_returns(returns, weights)
        elif allocation_type == "MinVariance":
            weights = min_variance_portfolio(returns)
            portfolio_returns = get_portfolio_returns(returns, weights)
        elif allocation_type == "Définir moi-même":
            weights = user_weights
            portfolio_returns = get_portfolio_returns(returns, weights)
    else:
        portfolio_returns = returns
    
    # Tabs pour afficher les différentes sections
    # Tabs pour afficher les différentes sections
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Indicateurs de Risque", "📊 Volatilité", "📈 Rendements & VaR", "📉 Drawdowns"])

    with tab1:
    st.subheader("📉 Indicateurs de Risque")

    # 🎯 SECTION VaR
    with st.expander("🔍 Value at Risk (VaR)"):
        # 🔹 Calcul des VaR
        var_param = calculate_var(portfolio_returns, confidence)
        var_hist = var_historique(portfolio_returns, confidence)
        var_mc = var_monte_carlo(portfolio_returns, confidence)
        cvar = calculate_cvar(portfolio_returns, confidence)

        # ✅ SI UN SEUL ACTIF / PORTEFEUILLE : Affichage simple
        if isinstance(var_param, dict):  # Plusieurs actifs (dict)
            for ticker in portfolio_returns.columns:
                with st.expander(f"📌 {ticker}"):
                    st.write(f"**VaR Paramétrique**: {var_param[ticker]:.4f}")
                    st.write(f"**VaR Historique**: {var_hist[ticker]:.4f}")
                    st.write(f"**VaR Monte Carlo**: {var_mc[ticker]:.4f}")
                    st.write(f"**CVaR (Conditional VaR)**: {cvar[ticker]:.4f}")
        else:  # Un seul actif (float)
            st.write(f"**VaR Paramétrique**: {var_param:.4f}")
            st.write(f"**VaR Historique**: {var_hist:.4f}")
            st.write(f"**VaR Monte Carlo**: {var_mc:.4f}")
            st.write(f"**CVaR (Conditional VaR)**: {cvar:.4f}")

    # 🎯 SECTION Volatilité
    with st.expander("📊 Volatilité"):
        # 🔹 Calcul des indicateurs de volatilité
        annual_vol = annual_volatility(portfolio_returns)
        ewma_vol = ewma_volatility(portfolio_returns)
        semi_dev = semi_deviation(portfolio_returns)

        # ✅ SI UN SEUL ACTIF / PORTEFEUILLE : Affichage simple
        if isinstance(annual_vol, dict):  # Plusieurs actifs
            for ticker in portfolio_returns.columns:
                with st.expander(f"📌 {ticker}"):
                    st.write(f"**Volatilité Annualisée**: {annual_vol[ticker]:.4f}")
                    st.write(f"**Volatilité EWMA**: {ewma_vol[ticker]:.4f}")
                    st.write(f"**Semi-Deviation**: {semi_dev[ticker]:.4f}")
        else:  # Un seul actif
            st.write(f"**Volatilité Annualisée**: {annual_vol:.4f}")
            st.write(f"**Volatilité EWMA**: {ewma_vol:.4f}")
            st.write(f"**Semi-Deviation**: {semi_dev:.4f}")
