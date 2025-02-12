import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from data_loader import get_data, TICKERS_LIST
import risk_indicators as ri

# 📌 Titre du Dashboard
st.title("📊 Dashboard de Gestion du Risque Extrême & EVT")

# 📌 Sélection des actifs et période
selected_tickers = st.sidebar.multiselect("📌 Sélectionnez les actifs", TICKERS_LIST, default=["SPY"])
start_date = st.sidebar.date_input("📅 Date de début", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("📅 Date de fin", value=pd.to_datetime("2024-01-01"))

# 📌 Sélection de la méthode d'allocation
allocation_method = st.sidebar.radio("🔄 Méthode d'allocation", ["Équipondérée", "Min Variance", "Manuelle"])

if allocation_method == "Manuelle":
    user_weights = {}
    for ticker in selected_tickers:
        user_weights[ticker] = st.sidebar.slider(f"🔧 Poids de {ticker}", 0.0, 1.0, 1.0/len(selected_tickers))
    weights = np.array(list(user_weights.values()))
else:
    weights = np.array([1/len(selected_tickers)] * len(selected_tickers))  # Équipondérée par défaut

# 📌 Chargement des données
prices_data, returns_data = get_data(selected_tickers, start_date, end_date)

if returns_data.empty:
    st.error("⚠️ Les rendements sont vides, vérifiez les dates et les tickers sélectionnés.")
else:
    st.success("📈 Données chargées avec succès !")

    # 📌 Calcul des indicateurs
    var_param = ri.calculate_var(returns_data)
    var_mc = ri.monte_carlo_var(returns_data)
    cvar = ri.calculate_cvar(returns_data)
    drawdown = ri.calculate_drawdown(prices_data)
    max_dd = ri.max_drawdown(prices_data)
    vol, ewma_vol = ri.calculate_volatility(returns_data)
    ewma_var = ri.ewma_var(returns_data)

    # 📊 Affichage des métriques
    st.subheader("📊 Indicateurs Clés de Risque")
    for ticker in selected_tickers:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric(f"{ticker} - VaR Param.", f"{var_param[ticker] * 100:.2f}%")
        col2.metric(f"{ticker} - VaR Monte Carlo", f"{var_mc * 100:.2f}%")
        col3.metric(f"{ticker} - CVaR", f"{cvar[ticker] * 100:.2f}%")
        col4.metric(f"{ticker} - Max Drawdown", f"{max_dd[ticker] * 100:.2f}%")
        col5.metric(f"{ticker} - Volatilité", f"{vol[ticker] * 100:.2f}%")
        col6.metric(f"{ticker} - EWMA VaR", f"{ewma_var[ticker] * 100:.2f}%")

    st.subheader("📉 Évolution du Drawdown")
    st.line_chart(drawdown)
