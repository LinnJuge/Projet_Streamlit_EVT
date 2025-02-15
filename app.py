import streamlit as st
import datetime

# Configuration du dashboard
st.set_page_config(page_title="Dashboard de Gestion des Risques", layout="wide")

# Sidebar - Paramètres de l’étude
st.sidebar.header("📊 Paramètres de l’étude")

# Sélection des actifs
tickers = st.sidebar.multiselect(
    "Sélectionner un ou plusieurs actifs",
    options=["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"],
    default=["MSFT"]
)

# Sélection des dates
start_date = st.sidebar.date_input("Date de début", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("Date de fin", datetime.date(2023, 12, 31))

# Mode d’analyse
mode = st.sidebar.radio("Mode d'analyse", ["Comparer", "Créer un portefeuille"])

# Allocation si portefeuille choisi
user_weights = None
if mode == "Créer un portefeuille":
    allocation_type = st.sidebar.radio("Choix de l’allocation", ["Équipondérée", "MinVariance", "Définir moi-même"])
    
    if allocation_type == "Définir moi-même":
        user_weights = {}
        for ticker in tickers:
            user_weights[ticker] = st.sidebar.number_input(f"Poids de {ticker}", min_value=0.0, max_value=1.0, step=0.01)
        total_weight = sum(user_weights.values())
        if total_weight != 1:
            st.sidebar.warning("⚠️ La somme des poids doit être égale à 1 !")

# Affichage des paramètres sélectionnés
st.write("### ✅ Paramètres sélectionnés")
st.write(f"📌 Actifs sélectionnés : {tickers}")
st.write(f"📆 Période d'analyse : {start_date} → {end_date}")
st.write(f"📊 Mode d'analyse : {mode}")
if mode == "Créer un portefeuille":
    st.write(f"📈 Allocation choisie : {allocation_type}")
    if allocation_type == "Définir moi-même":
        st.write(f"📊 Poids des actifs : {user_weights}")

