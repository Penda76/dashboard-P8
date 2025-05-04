from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# === RÉPERTOIRE DE TRAVAIL ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # remonte à la racine du dépôt

# === CHEMINS DES FICHIERS ===
model_path = os.path.join(ROOT_DIR, "model", "XGBoost_auc_0.741_cout_33940_trial_1.joblib")
seuil_path = os.path.join(ROOT_DIR, "data_sample", "seuil_optimal.txt")
x_train_path = os.path.join(ROOT_DIR, "data_sample", "X_test_clean.csv")

# === LANCEMENT DE L’API ===
app = FastAPI(title="Credit Scoring API")

# === CHARGEMENT DU MODÈLE & DES DONNÉES ===
model = joblib.load(model_path)

with open(seuil_path, "r") as f:
    seuil_metier = float(f.read())

df_train = pd.read_csv(x_train_path)

# Exclure SK_ID_CURR des features utilisées pour l'entraînement
columns = [col for col in df_train.columns if col != "SK_ID_CURR"]
sample = df_train[columns].iloc[0].to_dict()  # Exemple Swagger sans SK_ID_CURR

# === SCHÉMA D’ENTRÉE ===
class ClientData(BaseModel):
    __annotations__ = {col: float for col in columns}

    class Config:
        schema_extra = {
            "example": sample
        }

# === ENDPOINT POST /predict ===
@app.post("/predict")
def predict(data: ClientData):
    try:
        df = pd.DataFrame([data.dict()])

        # Par sécurité, même si SK_ID_CURR n'est plus attendu ici
        if "SK_ID_CURR" in df.columns:
            df = df.drop(columns=["SK_ID_CURR"])

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

# === ENDPOINT GET /predict_demo ===
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

# === ENDPOINT GET /health ===
@app.get("/health")
def health_check():
    return {
        "status": "✅ API opérationnelle",
        "model_loaded": isinstance(model, object),
        "seuil_metier": seuil_metier
    }

# === ENDPOINT GET /predict/{client_id} ===
@app.get("/predict/{client_id}")
def predict_by_id(client_id: int):
    try:
        if "SK_ID_CURR" not in df_train.columns:
            return {"error": "Colonne 'SK_ID_CURR' absente dans les données."}

        client_data = df_train[df_train["SK_ID_CURR"] == client_id]

        if client_data.empty:
            return {"error": f"Client {client_id} introuvable dans les données."}

        input_data = client_data.drop(columns=["SK_ID_CURR"])
        proba = model.predict_proba(input_data)[0][1]
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
