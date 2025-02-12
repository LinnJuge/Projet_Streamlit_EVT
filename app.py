
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import streamlit as st
import data_loader as dl

# 📌 Titre du Dashboard
st.title("📊 Dashboard de Gestion du Risque Extrême & EVT")

# 📌 Sélection des Données (actifs et période)
tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA"]
selected_tickers = st.multiselect("Sélectionnez les actifs à analyser", tickers, default=["SPY"])
start_date = st.date_input("Date de début")
end_date = st.date_input("Date de fin")

# 📌 Chargement des données si l'utilisateur a fait une sélection
if selected_tickers:
    returns_data = dl.get_data(selected_tickers, start_date, end_date)
    st.write("📈 Données chargées avec succès !")
    st.write("📌 Aperçu des rendements :", returns_data.head())

# 📌 Onglets du Dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📊 Indicateurs de Risque", "📈 EVT", "📉 Stress Testing", "📌 Visualisations"])

# 🟢 Onglet 1 : Indicateurs de Risque
with tab1:
    st.subheader("📊 Indicateurs de Risque")
    var_values = ri.calculate_var(returns_data)
    cvar_values = ri.calculate_cvar(returns_data)
    st.write("📌 VaR : ", var_values)
    st.write("📌 CVaR : ", cvar_values)
