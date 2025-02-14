import yfinance as yf
import pandas as pd
import numpy as np

# 📌 Liste élargie d'actifs disponibles
TICKERS_LIST = [
    "SPY", "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX", "BRK-B",
    "JPM", "V", "PG", "JNJ", "UNH", "DIS", "KO", "PEP", "PFE", "XOM", 
    "IBM", "CSCO", "BA", "MCD", "GS", "CAT", "CVX", "T", "INTC", "WMT",
    "QQQ", "DIA", "IWM", "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLU",
    "GLD", "SLV", "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "JPYUSD=X"
]


def get_data(tickers, start, end):
    """
    Récupère les prix de clôture des actifs sélectionnés depuis Yahoo Finance.
    
    :param tickers: Liste des tickers des actifs
    :param start: Date de début
    :param end: Date de fin
    :return: DataFrame des prix de clôture et DataFrame des rendements log
    """
    df = yf.download(tickers, start=start, end=end)["Close"]  # 🔹 Prix de clôture
    df.dropna(inplace=True)  # 🔹 Supprimer les valeurs manquantes
    returns = np.log(df / df.shift(1)).dropna()  # 🔹 Calcul des rendements log
    return df, returns
