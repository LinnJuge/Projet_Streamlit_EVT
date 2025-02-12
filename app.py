
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

# 📌 Sélection des Données (actifs et période)
tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA"]
selected_tickers = st.multiselect("Sélectionnez les actifs à analyser", tickers, default=["SPY"])
start_date = st.date_input("Date de début")
end_date = st.date_input("Date de fin")

# 📌 Sélection du niveau de confiance
confidence_level = st.slider("Niveau de Confiance (%)", 90, 99, 95) / 100  # Convertir en décimal

# 📌 Chargement des données si l'utilisateur a fait une sélection
if selected_tickers:
    prices_data, returns_data = dl.get_data(selected_tickers, start_date, end_date)  # ✅ Correction ici

    if returns_data.empty:
        st.error("⚠️ Les rendements sont vides, vérifiez les dates et les tickers sélectionnés.")
    else:
        st.write("📈 Données chargées avec succès !")
        st.write("📌 Aperçu des rendements :", returns_data.head())

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

            # 📌 Affichage des résultats
            st.write(f"📌 **VaR Paramétrique ({confidence_level*100}%)** : ", var_param)
            st.write(f"📌 **VaR Monte Carlo ({confidence_level*100}%)** : ", var_mc)
            st.write(f"📌 **CVaR ({confidence_level*100}%)** : ", cvar)
            st.write(f"📌 **Max Drawdown** : ", max_dd)

            # 📊 Visualisation du Drawdown
            st.subheader("📉 Évolution du Drawdown")
            st.line_chart(drawdown)

            # 📊 Visualisation de la distribution des rendements avec VaR et CVaR
            st.subheader("📊 Distribution des Rendements et Risques")
            fig, ax = plt.subplots(figsize=(10,5))
            for ticker in returns_data.columns:
                plt.hist(returns_data[ticker], bins=50, alpha=0.6, label=ticker)

            plt.axvline(var_param.min(), color='red', linestyle='dashed', linewidth=2, label=f'VaR ({confidence_level*100}%)')
            plt.axvline(cvar.min(), color='green', linestyle='dashed', linewidth=2, label=f'CVaR ({confidence_level*100}%)')
            plt.legend()
            st.pyplot(fig)
