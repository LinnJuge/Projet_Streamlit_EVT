import streamlit as st
from data_loader import get_data
from portfolio_allocation import equal_weighted_portfolio, min_variance_portfolio
from risk_indicators import calculate_var, var_historique, var_monte_carlo, calculate_cvar

st.set_page_config(page_title="Dashboard de Risque", layout="wide")

# Sidebar - Sélection des actifs
st.sidebar.header("Paramètres de l'étude")
tickers = st.sidebar.multiselect("Sélectionner un ou plusieurs actifs", ["AAPL", "GOOGL", "MSFT", "SPY"])
start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("2023-12-31"))

mode = st.sidebar.radio("Mode d'analyse", ["Comparer", "Créer un portefeuille"])

if tickers:
    prices, returns = get_data(tickers, start_date, end_date)

    if mode == "Créer un portefeuille":
        allocation_type = st.sidebar.radio("Choix de l'allocation", ["Équipondérée", "MinVariance"])
        weights = equal_weighted_portfolio(returns) if allocation_type == "Équipondérée" else min_variance_portfolio(returns)
        portfolio_returns = returns.dot(weights)
    else:
        portfolio_returns = returns

    st.write(f"**VaR Paramétrique** : {calculate_var(portfolio_returns, 0.95)}")
    st.write(f"**VaR Historique** : {var_historique(portfolio_returns, 0.95)}")
    st.write(f"**VaR Monte Carlo** : {var_monte_carlo(portfolio_returns, 0.95)}")
    st.write(f"**CVaR** : {calculate_cvar(portfolio_returns, 0.95)}")
