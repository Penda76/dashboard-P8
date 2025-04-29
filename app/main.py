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

# === CHEMINS RELATIFS ROBUSTES ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_DIR, "model", "XGBoost_auc_0.748_cout_33136_trial_9.joblib")
seuil_path = os.path.join(CURRENT_DIR, "data_sample", "seuil_optimal.txt")
x_test_path = os.path.join(CURRENT_DIR, "data_sample", "X_test_sample_id.csv")

# === LANCEMENT DE L’API ===
app = FastAPI(title="Credit Scoring API")

# === CHARGEMENTS DES RESSOURCES ===
model = joblib.load(model_path)

with open(seuil_path, "r") as f:
    seuil_metier = float(f.read())

df_test = pd.read_csv(x_test_path)

# ✅ S'assurer que SK_ID_CURR est l’index
if "SK_ID_CURR" in df_test.columns:
    df_test.set_index("SK_ID_CURR", inplace=True)

columns = df_test.columns.tolist()
sample = df_test.iloc[0].to_dict()


# === CLASSE D’ENTRÉE POUR SWAGGER ===
class ClientData(BaseModel):
    __annotations__ = {col: float for col in columns}

    class Config:
        schema_extra = {
            "example": sample
        }


# === ENDPOINT : prédiction par contenu JSON ===
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


# === ENDPOINT : prédiction par identifiant client ===
@app.get("/predict/{client_id}")
def predict_by_id(client_id: int):
    try:
        if client_id not in df_test.index:
            return {"error": f"Client {client_id} introuvable."}

        client_data = df_test.loc[client_id]
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


# === ENDPOINT : test de l’API ===
@app.get("/health")
def health_check():
    return {
        "status": "✅ API opérationnelle",
        "model_loaded": isinstance(model, object),
        "seuil_metier": seuil_metier
    }
