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
    df = yf.download(tickers, start=start, end=end)["Adj Close"]

    if df.empty:
        print("⚠️ Aucune donnée récupérée, vérifie tes dates et tickers !")
        return pd.DataFrame(), pd.DataFrame()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[df > 0]

    # Calcul des rendements logarithmiques
    returns = np.log(df / df.shift(1)).dropna()

    if returns.empty:
        print("⚠️ Les rendements sont vides après calcul, vérifie les données.")
    
    return df, returns
