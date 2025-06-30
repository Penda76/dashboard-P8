from pathlib import Path

# Version corrigée avec exclusion explicite de AGE pour éviter l'erreur XGBoost
# dashboard_fixed_age_code = """
import streamlit as st
st.set_page_config(page_title="Dashboard Crédit", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import shap
import joblib

st.title("📊 Analyse complète du client")

@st.cache_data
def load_data():
    return pd.read_csv("app/data_sample/X_test_clean.csv")

@st.cache_data
def fetch_predictions(df):
    predictions = []
    for client_id in df["SK_ID_CURR"]:
        try:
            r = requests.get(f"https://api-credit-scoring-v2.onrender.com/predict/{client_id}")
            if r.status_code == 200:
                data = r.json()
                predictions.append({
                    "SK_ID_CURR": client_id,
                    "prediction": data.get("prediction"),
                    "probability": data.get("probability")
                })
            else:
                predictions.append({"SK_ID_CURR": client_id, "prediction": None, "probability": None})
        except:
            predictions.append({"SK_ID_CURR": client_id, "prediction": None, "probability": None})
    return pd.DataFrame(predictions)

def get_similar_clients(df, client_row):
    age_tolerance = 3 * 365
    return df[
        (abs(df["DAYS_BIRTH"] - client_row["DAYS_BIRTH"]) < age_tolerance) &
        (abs(df["DAYS_EMPLOYED_PERC"] - client_row["DAYS_EMPLOYED_PERC"]) < 0.05) &
        (df["CODE_GENDER"] == client_row["CODE_GENDER"])
    ]

# Chargement
df = load_data()
df["AGE"] = (-df["DAYS_BIRTH"] / 365).round(1)
df_pred = fetch_predictions(df)
df = df.merge(df_pred, on="SK_ID_CURR")
client_ids = df["SK_ID_CURR"].tolist()
client_id = st.sidebar.selectbox("📌 Sélectionnez un identifiant client :", client_ids)
client_row = df[df["SK_ID_CURR"] == client_id].iloc[0]
similar_clients = get_similar_clients(df, client_row)

# Résumé client
st.subheader("🧾 Résumé du client")
col1, col2 = st.columns(2)
col1.metric("Décision", "✅ Accepté" if client_row["prediction"] == 0 else "❌ Refusé")
col2.metric("Probabilité de défaut", f"{client_row['probability']:.2f}" if pd.notna(client_row['probability']) else "N/A")

# Graphique SHAP local
st.subheader("🔍 Variables ayant influencé la décision (SHAP)")
model = joblib.load("app/model/XGBoost_auc_0.741_cout_33940_trial_1.joblib")
explainer = shap.TreeExplainer(model)
X = df.set_index("SK_ID_CURR").drop(columns=["prediction", "probability"], errors="ignore")
X.columns = X.columns.astype(str)

client_data = X.loc[[client_id]].drop(columns=["AGE"], errors="ignore")
shap_values = explainer.shap_values(client_data)

explanation = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=client_data.values[0],
    feature_names=client_data.columns
)

fig = plt.gcf()
shap.plots.waterfall(explanation, show=False)
st.pyplot(fig)

# Barres comparatives
st.subheader("📊 Comparaison du client avec différents groupes")

top_features = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AGE", "DAYS_EMPLOYED_PERC"
]

for i in range(0, len(top_features), 3):
    cols = st.columns(3)
    for j, feature in enumerate(top_features[i:i+3]):
        if feature in df.columns:
            valeurs = {
                "Client": client_row[feature],
                "Globale": df[feature].mean(),
                "Acceptés": df[df["prediction"] == 0][feature].mean(),
                "Refusés": df[df["prediction"] == 1][feature].mean(),
                "Similaires": similar_clients[feature].mean()
            }

            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.bar(valeurs.keys(), valeurs.values(), color=["blue", "gray", "green", "red", "purple"])
            ax.set_title("Âge (en années)" if feature == "AGE" else feature, fontsize=10)
            ax.tick_params(axis='x', labelrotation=15, labelsize=8)
            ax.set_ylabel("Valeur", fontsize=9)
            cols[j].pyplot(fig)

# Légende unique
st.markdown("ℹ️ Comparaison sur 5 groupes : Client, Moyenne globale, Acceptés, Refusés, Similaires.")


file_path = Path("/mnt/data/dashboard_age_corrected.py")
#file_path.write_text(dashboard_fixed_age_code)

file_path.name
