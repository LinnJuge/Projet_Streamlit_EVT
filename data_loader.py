import yfinance as yf
import pandas as pd
import numpy as np

def get_data(tickers, start, end):
    """
    Récupère les prix de clôture et calcule les rendements log pour les actifs sélectionnés.
    - Retourne `prices` et `returns`
    - Gère le cas d'un seul actif ou plusieurs actifs
    """
    df = yf.download(tickers, start=start, end=end)["Close"]

    if df.empty:
        print("⚠️ Aucune donnée récupérée. Vérifiez les tickers et la période sélectionnée.")
        return None, None

    # Cas d’un seul actif → Convertir en DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers)

    df.dropna(inplace=True)  # Suppression des valeurs manquantes
    returns = np.log(df / df.shift(1)).dropna()  # Rendements log

    return df, returns  # Retourne les prix et les rendements

