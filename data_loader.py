import yfinance as yf
import pandas as pd
import numpy as np

def get_data(tickers, start, end):
    """
    Récupère les prix de clôture et les rendements log des actifs sélectionnés.
    - Gère les cas d'un seul ou plusieurs actifs correctement.
    - Vérifie et retourne `None, None` si les données sont vides.
    """
    df = yf.download(tickers, start=start, end=end)["Close"]  # 🔹 Téléchargement des prix
    
    if df.empty:
        print("⚠️ Aucune donnée récupérée. Vérifiez les tickers et la période sélectionnée.")
        return None, None

    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers)  # Convertir en DataFrame avec un nom explicite pour éviter les erreurs
    
    df.dropna(inplace=True)  # 🔹 Suppression des valeurs manquantes
    returns = np.log(df / df.shift(1)).dropna()  # 🔹 Calcul des rendements log

    return df, returns  # Retourne les prix et les rendements

