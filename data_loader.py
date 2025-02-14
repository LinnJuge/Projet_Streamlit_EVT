import yfinance as yf
import pandas as pd
import numpy as np



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
