from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd


# 🔧 Corriger le chemin racine du projet
PROJECT_ROOT = os.getcwd()

# 📁 Chemins vers les fichiers
model_path = os.path.join(PROJECT_ROOT, "model", "XGBoost_auc_0.748_cout_33136_trial_9.joblib")
seuil_path = os.path.join(PROJECT_ROOT, "data_sample", "seuil_optimal.txt")
x_train_path = os.path.join(PROJECT_ROOT, "data_sample", "X_test_sample.csv")


# === Application FastAPI ===
app = FastAPI(title="Credit Scoring API")

# === Chargement des fichiers ===
model = joblib.load(model_path)

with open(seuil_path, "r") as f:
    seuil_metier = float(f.read())

df_train = pd.read_csv(x_train_path)

# ✅ Utiliser SK_ID_CURR comme index si dispo
if "SK_ID_CURR" in df_train.columns:
    df_train.set_index("SK_ID_CURR", inplace=True)

columns = df_train.columns.tolist()
sample = df_train.iloc[0].to_dict()

# === Pydantic Model (structure d'entrée) ===
class ClientData(BaseModel):
    __annotations__ = {col: float for col in columns}

    class Config:
        schema_extra = {
            "example": sample
        }

# === Endpoint général (données JSON) ===
@app.post("/predict")
def predict(data: ClientData):
    try:
        df = pd.DataFrame([data.dict()])
        proba = model.predict_proba(df)[0][1]
        prediction = int(proba >= seuil_metier)
        decision = "Refusé" if prediction == 1 else "Accepté"

        return {
            "probability": round(float(proba), 4),
            "prediction": prediction,
            "decision": decision,
            "seuil_metier": seuil_metier
        }

    except Exception as e:
        return {"error": str(e)}

# === Endpoint de démonstration ===
@app.get("/predict_demo")
def predict_demo():
    df = pd.DataFrame([sample])
    proba = model.predict_proba(df)[0][1]
    prediction = int(proba >= seuil_metier)
    decision = "Refusé" if prediction == 1 else "Accepté"

    return {
        "probability": round(float(proba), 4),
        "prediction": prediction,
        "decision": decision,
        "seuil_metier": seuil_metier
    }

# === Endpoint basé sur un ID client ===
@app.get("/predict/{client_id}")
def predict_by_id(client_id: int):
    try:
        if client_id not in df_train.index:
            return {"error": f"Client {client_id} introuvable dans les données."}

        client_data = df_train.loc[client_id]
        df = pd.DataFrame([client_data])

        proba = model.predict_proba(df)[0][1]
        prediction = int(proba >= seuil_metier)
        decision = "Refusé" if prediction == 1 else "Accepté"

        return {
            "client_id": client_id,
            "probability": round(float(proba), 4),
            "prediction": prediction,
            "decision": decision,
            "seuil_metier": seuil_metier
        }

    except Exception as e:
        return {"error": str(e)}

# === Endpoint santé ===
@app.get("/health")
def health_check():
    return {
        "status": "✅ API opérationnelle",
        "model_loaded": isinstance(model, object),
        "seuil_metier": seuil_metier
    }