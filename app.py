import streamlit as st
from data_loader import get_data, TICKERS_LIST
import risk_indicators as ri
import portfolio_optimizer as po
import pandas as pd
import numpy as np

st.title("📊 Dashboard de Gestion du Risque & EVT")

# 📌 Sélection des actifs
tickers = TICKERS_LIST
selected_tickers = st.sidebar.multiselect("Sélectionnez les actifs", tickers, default=["SPY"])
start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("2024-01-01"))

# 📌 Choix du mode d'allocation
if len(selected_tickers) > 1:
    allocation_method = st.sidebar.radio("Méthode d'Allocation", ["Manuelle", "Équipondérée", "Min Variance"])
    weights = []

    if allocation_method == "Manuelle":
        for ticker in selected_tickers:
            weight = st.sidebar.slider(f"⚖ Poids de {ticker}", 0.0, 1.0, 1.0 / len(selected_tickers))
            weights.append(weight)
        weights = np.array(weights) / np.sum(weights)

if selected_tickers:
    prices_data, returns_data = dl.get_data(selected_tickers, start_date, end_date)

    if len(selected_tickers) > 1:
        if allocation_method == "Équipondérée":
            weights = po.equal_weight_allocation(len(selected_tickers))
        elif allocation_method == "Min Variance":
            weights = po.min_variance_allocation(returns_data)

        returns_portfolio = returns_data @ weights
    else:
        returns_portfolio = returns_data

    # 📌 Calcul des Indicateurs
    var_param = ri.calculate_var(returns_portfolio)
    var_mc = ri.monte_carlo_var(returns_portfolio)
    cvar = ri.calculate_cvar(returns_portfolio)
    drawdown = ri.calculate_drawdown(prices_data)
    max_dd = ri.max_drawdown(prices_data)

    # 📌 Affichage des Indicateurs
    st.subheader("📊 Indicateurs Clés de Risque")
    for ticker in selected_tickers:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"📉 {ticker} - VaR Param.", f"{var_param[ticker] * 100:.2f}%")
        col2.metric(f"📉 {ticker} - VaR Monte Carlo", f"{var_mc * 100:.2f}%")
        col3.metric(f"📉 {ticker} - CVaR", f"{cvar[ticker] * 100:.2f}%")
        col4.metric(f"📉 {ticker} - Max Drawdown", f"{max_dd[ticker] * 100:.2f}%")

    st.subheader("📉 Évolution du Drawdown")
    st.line_chart(drawdown)
