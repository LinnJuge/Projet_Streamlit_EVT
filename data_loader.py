import yfinance as yf
import pandas as pd
import numpy as np


import yfinance as yf
import pandas as pd
import numpy as np

def get_data(tickers, start, end):
    """
    Récupère les prix de clôture et les rendements log des actifs sélectionnés.
    - Gère le cas où un seul actif est sélectionné (conversion en DataFrame)
    """
    df = yf.download(tickers, start=start, end=end)["Close"]  # 🔹 Prix de clôture
    if isinstance(df, pd.Series):
        df = df.to_frame()  # Convertir en DataFrame si un seul actif est sélectionné
    df.dropna(inplace=True)  # 🔹 Supprimer les valeurs manquantes
    
    returns = np.log(df / df.shift(1)).dropna()  # 🔹 Calcul des rendements log
    return df, returns
