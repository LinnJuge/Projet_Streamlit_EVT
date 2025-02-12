import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import streamlit as st
import data_loader as dl
import risk_indicators as ri

# 📌 Titre du Dashboard
st.title("📊 Dashboard de Gestion du Risque Extrême & EVT")

# 📌 Barre latérale pour les entrées utilisateur
st.sidebar.header("🔍 Paramètres de l'analyse")
tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA"]
selected_tickers = st.sidebar.multiselect("📌 Sélectionnez les actifs à analyser", tickers, default=["SPY"])
start_date = st.sidebar.date_input("📅 Date de début", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("📅 Date de fin", value=pd.to_datetime("2024-01-01"))

# 📌 Sélection du niveau de confiance
confidence_level = st.sidebar.slider("🔧 Niveau de Confiance (%)", 90, 99, 95) / 100  # Convertir en décimal

# 📌 Chargement des données si l'utilisateur a fait une sélection
if selected_tickers:
    prices_data, returns_data = dl.get_data(selected_tickers, start_date, end_date)

    if returns_data.empty:
        st.error("⚠️ Les rendements sont vides, vérifiez les dates et les tickers sélectionnés.")
    else:
        st.success("📈 Données chargées avec succès !")

        # 📌 Visualisation des rendements
        st.subheader("📈 Évolution des Rendements")
        fig, ax = plt.subplots(figsize=(12, 5))
        for ticker in returns_data.columns:
            sns.lineplot(x=returns_data.index, y=returns_data[ticker], label=ticker)
        plt.xlabel("Date")
        plt.ylabel("Rendements")
        plt.title("Évolution des Rendements des Actifs")
        plt.legend()
        st.pyplot(fig)

        # 📌 Onglets du Dashboard
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Indicateurs de Risque", "📈 EVT", "📉 Stress Testing", "📌 Visualisations"])

        # 🟢 Onglet 1 : Indicateurs de Risque
        with tab1:
            st.subheader("📊 Indicateurs de Risque")

            # 📌 Calcul des indicateurs
            var_param = ri.calculate_var(returns_data, confidence_level)
            var_mc = ri.monte_carlo_var(returns_data, confidence_level)
            cvar = ri.calculate_cvar(returns_data, confidence_level)
            drawdown = ri.calculate_drawdown(prices_data)
            max_dd = ri.max_drawdown(prices_data) 

            # 🔹 Récupération des résultats pour chaque actif
            selected_assets = returns_data.columns  # Liste des actifs sélectionnés

            # 📊 Indicateurs Clés de Risque
            st.subheader("📊 Indicateurs Clés de Risque")
            for ticker in selected_assets:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(f"📉 {ticker} - VaR Param.", f"{var_param[ticker]:.4f}")
                col2.metric(f"📉 {ticker} - VaR Monte Carlo", f"{var_mc:.4f}")
                col3.metric(f"📉 {ticker} - CVaR", f"{cvar[ticker]:.4f}")
                col4.metric(f"📉 {ticker} - Max Drawdown", f"{max_dd:.4f}")

            # 📊 Visualisation du Drawdown
            st.subheader("📉 Évolution du Drawdown")
            st.line_chart(drawdown)

            # 📊 Distribution des Rendements avec VaR et CVaR
            st.subheader("📊 Distribution des Rendements et Risques")
            fig, ax = plt.subplots(figsize=(10, 5))
            for ticker in returns_data.columns:
                sns.histplot(returns_data[ticker], bins=50, kde=True, label=ticker, alpha=0.6)
            plt.axvline(var_param, color='red', linestyle='dashed', linewidth=2, label=f'VaR ({confidence_level*100}%)')
            plt.axvline(cvar, color='green', linestyle='dashed', linewidth=2, label=f'CVaR ({confidence_level*100}%)')
            plt.legend()
            plt.xlabel("Rendements")
            plt.ylabel("Fréquence")
            plt.title("Distribution des Rendements avec Indicateurs de Risque")
            st.pyplot(fig)
