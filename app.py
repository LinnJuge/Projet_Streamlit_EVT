import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

st.title("📊 Analyse des Rendements et EVT")

# 📌 L’utilisateur sélectionne les actifs et les dates
tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA"]
selected_tickers = st.multiselect("Sélectionnez les actifs à analyser", tickers, default=["SPY"])
start_date = st.date_input("Date de début", value=pd.to_datetime("2010-01-01"))
end_date = st.date_input("Date de fin", value=pd.to_datetime("2024-01-01"))

# 📌 Fonction pour récupérer les données
def get_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)["Close"]
    df = df.pct_change().dropna()
    return df

if selected_tickers:
    returns_data = get_data(selected_tickers, start_date, end_date)

    # 📌 Affichage des statistiques descriptives
    st.write("Statistiques descriptives :", returns_data.describe())

    # 📌 Histogramme des rendements
    st.subheader("📈 Distribution des Rendements")
    fig, ax = plt.subplots(figsize=(12,6))
    for ticker in selected_tickers:
        sns.histplot(returns_data[ticker], bins=50, kde=True, label=ticker, alpha=0.6)
    plt.legend()
    st.pyplot(fig)

    # 📌 QQ-Plot
    st.subheader("📊 QQ-Plot pour chaque actif")
    fig, axes = plt.subplots(1, len(selected_tickers), figsize=(15, 5))
    for i, ticker in enumerate(selected_tickers):
        stats.probplot(returns_data[ticker], dist="norm", plot=axes[i])
        axes[i].set_title(f"QQ-Plot - {ticker}")
    st.pyplot(fig)
