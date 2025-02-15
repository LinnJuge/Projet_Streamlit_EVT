import yfinance as yf
import pandas as pd
import numpy as np

def get_data(tickers, start, end):
    """
    Récupère les prix de clôture et calcule les rendements log des actifs sélectionnés.
    """
    df = yf.download(tickers, start=start, end=end)["Close"]
    
    if df.empty:
        print("⚠️ Aucune donnée récupérée. Vérifiez les tickers et la période.")
        return None, None

    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers)  # Convertir en DataFrame pour homogénéiser

    df.dropna(inplace=True)  # Suppression des valeurs manquantes
    returns = np.log(df / df.shift(1)).dropna()  # Rendements log

    return df, returns  # Retourne les prix et rendements

