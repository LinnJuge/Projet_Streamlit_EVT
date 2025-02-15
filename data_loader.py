import yfinance as yf
import pandas as pd
import numpy as np

def get_data(tickers, start, end):
    """
    Télécharge les prix de clôture de Yahoo Finance et calcule les log-retours.
    Gère un seul actif ou plusieurs actifs.
    """
    df = yf.download(tickers, start=start, end=end)["Close"]

    if df.empty:
        print("⚠️ Aucune donnée récupérée. Vérifiez les tickers et la période sélectionnée.")
        return None, None

    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers)  # Convertir en DataFrame si un seul actif

    df.dropna(inplace=True)  
    returns = np.log(df / df.shift(1)).dropna()

    return df, returns  # Retourne les prix et les rendements


