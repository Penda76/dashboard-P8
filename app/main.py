from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# === RÉPERTOIRE DE TRAVAIL ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # remonte à la racine du dépôt

# === CHEMINS DES FICHIERS ===
model_path = os.path.join(ROOT_DIR, "model", "XGBoost_auc_0.748_cout_33136_trial_9.joblib")
seuil_path = os.path.join(ROOT_DIR, "data_sample", "seuil_optimal.txt")
x_train_path = os.path.join(ROOT_DIR, "data_sample", "X_test_sample.csv")

# === LANCEMENT DE L’API ===
app = FastAPI(title="Credit Scoring API")

# === CHARGEMENT DU MODÈLE & DES DONNÉES ===
model = joblib.load(model_path)

with open(seuil_path, "r") as f:
    seuil_metier = float(f.read())

df_train = pd.read_csv(x_train_path)
columns = df_train.columns.tolist()
sample = df_train.iloc[0].to_dict()  # Exemple Swagger

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
