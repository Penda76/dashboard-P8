from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# === CHEMINS DES FICHIERS ===
model_path = os.path.join("app","model", "XGBoost_auc_0.741_cout_33940_trial_1.joblib")
seuil_path = os.path.join("app","data_sample", "seuil_optimal.txt")
x_test_path = os.path.join("app","data_sample", "X_test_clean.csv")

# === LANCEMENT DE L’API ===
app = FastAPI(title="Credit Scoring API")

# === CHARGEMENT DU MODÈLE & DES DONNÉES ===
model = joblib.load(model_path)

with open(seuil_path, "r") as f:
    seuil_metier = float(f.read())

df_test = pd.read_csv(x_test_path)
columns = df_test.columns.tolist()
sample = df_test.iloc[0].to_dict()  # Exemple Swagger

# === SCHÉMA D’ENTRÉE ===
class ClientData(BaseModel):
    __annotations__ = {col: float for col in columns}

    model_config = {
        "json_schema_extra": {
            "example": sample
        }
    }

# === ENDPOINT POST /predict ===
@app.post("/predict")
def predict(data: ClientData):
    try:
        df = pd.DataFrame([data.model_dump()])
        if "SK_ID_CURR" in df.columns:
            df = df.drop(columns=["SK_ID_CURR"])  # L’ID n’est pas utilisé par le modèle

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
        if "SK_ID_CURR" not in df_test.columns:
            return {"error": "Colonne 'SK_ID_CURR' absente dans les données."}

        client_data = df_test[df_test["SK_ID_CURR"] == client_id]

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
