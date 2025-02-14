import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from data_loader import get_data
import seaborn as sns
from risk_indicators import get_portfolio_returns, var_historique, var_monte_carlo, calculate_var, calculate_cvar, semi_deviation, annual_volatility, ewma_volatility, calculate_drawdown, max_drawdown  # Import des fonctions de risk_indicators.py
from portfolio_allocation import *  # Import des fonctions de portfolio_optimization.py

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
        elif allocation_type == "MinVariance":
            weights = min_variance_portfolio(returns)
        elif allocation_type == "Définir moi-même":
            weights = user_weights
        else:
            weights = None
        portfolio_returns = get_portfolio_returns(returns, weights)
    else:
        portfolio_returns = returns
    
    # Tabs pour afficher les différentes sections
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Indicateurs de Risque", "📊 Volatilité", "📈 Rendements & VaR", "📉 Drawdowns"])
    
    with tab1:
        st.subheader("Indicateurs de Risque")
        st.write("### VaR")
        st.write("VaR Historique :", var_historique(portfolio_returns, confidence, weights))
        st.write("VaR Paramétrique :", calculate_var(portfolio_returns, confidence, weights))
        st.write("VaR Monte Carlo :", var_monte_carlo(portfolio_returns, confidence, weights=weights))
        
        st.write("### CVaR")
        st.write("CVaR :", calculate_cvar(portfolio_returns, confidence, weights))
    
    with tab2:
        st.subheader("Volatilité")
        st.write("Volatilité Annualisée :", annual_volatility(portfolio_returns, weights=weights))
        st.write("Volatilité EWMA :", ewma_volatility(portfolio_returns, weights=weights))
        st.write("Semi-Deviation :", semi_deviation(portfolio_returns, weights=weights))
    
    with tab3:
        st.subheader("Visualisation des rendements et VaR")
        show_var_param = st.checkbox("Afficher VaR Paramétrique", value=True)
        show_var_hist = st.checkbox("Afficher VaR Historique", value=True)
        show_var_mc = st.checkbox("Afficher VaR Monte Carlo", value=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(portfolio_returns, bins=50, kde=True, ax=ax, color="blue")
        
        if show_var_param:
            ax.axvline(-calculate_var(portfolio_returns, confidence, weights), color='purple', linestyle='--', label='VaR Paramétrique')
        if show_var_hist:
            ax.axvline(-var_historique(portfolio_returns, confidence, weights), color='red', linestyle='--', label='VaR Historique')
        if show_var_mc:
            ax.axvline(-var_monte_carlo(portfolio_returns, confidence, weights=weights), color='green', linestyle='--', label='VaR Monte Carlo')
        
        ax.axvline(-calculate_cvar(portfolio_returns, confidence, weights), color='black', linestyle='-', linewidth=2, label='CVaR')
        ax.legend()
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Super Visualisation des Drawdowns")
        drawdowns = calculate_drawdown(prices, weights)
        fig, ax = plt.subplots(figsize=(12, 6))
        drawdowns.plot(ax=ax, color='red')
        ax.set_title("Drawdown Historique")
        ax.set_ylabel("Drawdown (%)")
        st.pyplot(fig)
else:
    st.write("## Veuillez sélectionner au moins un actif pour commencer l'analyse.")
