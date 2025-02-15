import streamlit as st
st.cache_data.clear()
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import get_data
from risk_indicators import *  # Import des fonctions de risk_indicators.py
from portfolio_allocation import *  # Import des fonctions de portfolio_optimization.py

# Configuration de l'application
st.set_page_config(page_title="Risk Management Dashboard", layout="wide")

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

    # DEBUG: Vérifier le contenu des données avant calculs
    #st.write("DEBUG - prices:", prices)
    #st.write("DEBUG - returns:", returns)
    #st.write("DEBUG - portfolio_returns:", portfolio_returns)

    st.title("📉 Risk Management Dashboard  ")

    # Tabs pour afficher les différentes sections
    tab1, tab2, tab3 = st.tabs(["📉 Risk Indicators", "📈 EVT", "⚠️ Stress Tests"])

    
    ########################################### TAB 1##############################################
    with tab1:
        st.subheader("📉 Risk Indicators")

        with st.expander("📊 Visualizations"):
            # 📌 **Si un seul actif ou un portefeuille**
            if isinstance(portfolio_returns, pd.Series):  
                st.write("🔹 **Rendements avec VaR & CVaR**")
                plot_var_cvar_graph(portfolio_returns, confidence)
                
                if len(tickers) > 1:  # Afficher la heatmap SEULEMENT si plusieurs actifs
                    st.write("🔥 **Matrice de Corrélation des Actifs**")
                    plot_correlation_heatmap(returns)  # Utiliser `returns` pour le portefeuille
                # 🔹 Affichage des poids du portefeuille
                if mode == "Portefeuille" and weights is not None:
                    weight_df = pd.DataFrame({"Actifs": tickers, "Poids": weights})
                    st.write("⚖️ **Répartition des Poids dans le Portefeuille**")
                    st.dataframe(weight_df.style.format({"Poids": "{:.2%}"}))
            else:  # 📌 **Si plusieurs actifs en comparaison**
                for ticker in portfolio_returns.columns:
                    st.write(f"📊 **{ticker} : Rendements avec VaR & CVaR**")
                    plot_var_cvar_graph(portfolio_returns[ticker], confidence, title=f"VaR et CVaR pour {ticker}")
                if len(portfolio_returns.columns) > 1:  # 🔥 Heatmap de corrélation entre actifs
                    st.write("🔥 **Matrice de Corrélation entre Actifs**")
                    plot_correlation_heatmap(portfolio_returns)  # Utiliser `portfolio_returns` ici
                    
                

        # 🎯 SECTION VaR
        with st.expander("💰 Value at Risk "):
            # 🔹 Calcul des VaR
            var_param = calculate_var(portfolio_returns, confidence)
            var_hist = var_historique(portfolio_returns, confidence)
            var_mc = var_monte_carlo(portfolio_returns, confidence)
            cvar = calculate_cvar(portfolio_returns, confidence)

            # DEBUG: Vérification des valeurs calculées
            #st.write("DEBUG - var_param:", var_param)
            #st.write("DEBUG - var_hist:", var_hist)
            #st.write("DEBUG - var_mc:", var_mc)
            #st.write("DEBUG - cvar:", cvar)

            # ✅ SI UN SEUL ACTIF / PORTEFEUILLE : Affichage simple
            if isinstance(var_param, dict):  # Plusieurs actifs (dict)
                for ticker in portfolio_returns.columns:
                    st.subheader(f"📌 {ticker}")
                    st.write(f"**Parametric VaR**: {var_param[ticker] * 100:.2f} %")
                    st.write(f"**Historical VaR**: {var_hist[ticker] * 100:.2f} %")
                    st.write(f"**Monte Carlo VaR**: {var_mc[ticker] * 100:.2f} %")
                    st.write(f"**CVaR (Conditional VaR)**: {cvar[ticker] * 100:.2f} %")
            else:  # Un seul actif (float)
                st.write(f"**Parametric VaR**: {var_param * 100:.2f} %")
                st.write(f"**Historical VaR**: {var_hist * 100:.2f} %")
                st.write(f"**Monte Carlo VaR**: {var_mc * 100:.2f} %")
                st.write(f"**CVaR (Conditional VaR)**: {cvar * 100:.2f} %")

        # 🎯 SECTION Volatilité
        with st.expander("🎢 Volatility"):
            # 🔹 Calcul des indicateurs de volatilité
            annual_vol = annual_volatility(portfolio_returns)
            ewma_vol = ewma_volatility(portfolio_returns)
            semi_dev = semi_deviation(portfolio_returns)

            # DEBUG: Vérification des valeurs calculées
            #st.write("DEBUG - annual_vol:", annual_vol)
            #st.write("DEBUG - ewma_vol:", ewma_vol)
            #st.write("DEBUG - semi_dev:", semi_dev)

            # ✅ SI UN SEUL ACTIF / PORTEFEUILLE : Affichage simple
            if isinstance(annual_vol, dict):  # Plusieurs actifs
                for ticker in portfolio_returns.columns:
                    st.subheader(f"📌 {ticker}")
                    st.write(f"**Annual Volatility**: {annual_vol[ticker] * 100:.2f} %")
                    st.write(f"**EWMA Volatility**: {ewma_vol[ticker]*100:.2f} %")
                    st.write(f"**Semi-Deviation**: {semi_dev[ticker]*100:.2f} %")
            else:  # Un seul actif
                st.write(f"**Annual Volatility**: {annual_vol * 100:.2f} %")
                st.write(f"**EWMA Volatility**: {ewma_vol * 100:.2f} %")
                st.write(f"**Semi-Deviation**: {semi_dev * 100:.2f} %")

        # 🎯 SECTION Drawdowns
        with st.expander("🔻 Drawdowns"):
            if isinstance(prices, pd.Series):  # Un seul actif ou portefeuille global
                drawdowns = calculate_drawdown(prices)
                max_dd = max_drawdown(prices)

                # 📈 Graphique drawdown unique
                st.line_chart(drawdowns)
                st.write(f"**Max Drawdown**: {max_dd * 100:.2f} %")

            elif mode == "Comparaison":  # Plusieurs actifs séparés
                if isinstance(prices, pd.DataFrame):  
                    drawdowns = {ticker: calculate_drawdown(prices[ticker]) for ticker in prices.columns}
                    max_dd = {ticker: max_drawdown(prices[ticker]) for ticker in prices.columns}

                    for ticker in prices.columns:
                        st.subheader(f"📌 {ticker}")
                        st.line_chart(drawdowns[ticker])
                        st.write(f"**Max Drawdown**: {max_dd[ticker] * 100:.2f} %")

            elif mode == "Portefeuille":  # Drawdown pondéré pour un portefeuille
                if isinstance(prices, pd.DataFrame):  
                    drawdowns = {ticker: calculate_drawdown(prices[ticker]) for ticker in prices.columns}
                    max_dd = {ticker: max_drawdown(prices[ticker]) for ticker in prices.columns}

                    # Calcul du drawdown pondéré
                    portfolio_drawdown = sum(drawdowns[ticker] * weights[i] for i, ticker in enumerate(prices.columns))
                    portfolio_max_dd = sum(max_dd[ticker] * weights[i] for i, ticker in enumerate(prices.columns))

                    # 📈 Affichage du drawdown pondéré du portefeuille
                    st.subheader("📉 Drawdown du Portefeuille")
                    st.line_chart(portfolio_drawdown)
                    st.write(f"**Max Drawdown du Portefeuille**: {portfolio_max_dd * 100:.2f} %")

                    # 🔹 Affichage des drawdowns des actifs du portefeuille
                    st.subheader("🔍 Détail des actifs")
                    for ticker in prices.columns:
                        st.subheader(f"📌 {ticker}")
                        st.line_chart(drawdowns[ticker])
                        st.write(f"**Max Drawdown**: {max_dd[ticker] * 100:.2f} %")

           ########################################### TAB 2##############################################
