import yfinance as yf
import pandas as pd
import numpy as np

def get_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)["Close"]

    if df.empty:
        print("⚠️ Aucune donnée récupérée, vérifie tes dates et tickers !")
        return pd.DataFrame(), pd.DataFrame()  # Retourne deux DataFrames vides

    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Remplace les infinis par NaN
    df.dropna(inplace=True)  # Supprime les valeurs NaN
    df = df[df > 0]  # Supprime les valeurs <= 0

    # Calcul des rendements logarithmiques
    returns = df.pct_change().dropna()

    if returns.empty:
        print("⚠️ Les rendements sont vides après calcul, vérifie les données.")
    
    return df, returns  # ✅ Renvoie PRIX + RENDEMENTS
