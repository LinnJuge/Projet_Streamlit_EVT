import yfinance as yf
import pandas as pd

def get_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)["Close"]

    if df.empty:
        print("⚠️ Aucune donnée récupérée, vérifie tes dates et tickers !")
        return pd.DataFrame()  # Retourne un DataFrame vide

    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Remplace les infinis par NaN
    df.dropna(inplace=True)  # Supprime les valeurs NaN

    # 📌 Vérification et suppression des zéros pour éviter les infinis
    df = df[df > 0]

    # Calcul des rendements logarithmiques
    returns = df.pct_change().dropna()

    if returns.empty:
        print("⚠️ Les rendements sont vides après calcul, vérifie les données.")
    
    return returns
