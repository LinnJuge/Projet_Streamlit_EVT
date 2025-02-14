import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import get_data
from risk_indicators import *
from portfolio_allocation import *

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
