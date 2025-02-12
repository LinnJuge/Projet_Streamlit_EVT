import yfinance as yf
import pandas as pd

def get_data(tickers, start, end):
    """
    Télécharge les données boursières de Yahoo Finance et calcule les rendements logarithmiques.
    :param tickers: Liste des tickers (actions)
    :param start: Date de début (format YYYY-MM-DD)
    :param end: Date de fin (format YYYY-MM-DD)
    :return: DataFrame avec les rendements logarithmiques
    """
    df = yf.download(tickers, start=start, end=end)["Close"]

    # Vérification que toutes les actions ont bien des données
    df.dropna(inplace=True)

    # Calcul des rendements logarithmiques
    returns = df.pct_change().dropna()

    return returns
