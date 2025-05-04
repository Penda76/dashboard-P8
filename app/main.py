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
        if "SK_ID_CURR" in df.columns:
            df = df.drop(columns=["SK_ID_CURR"])  # On ne passe pas l'ID au modèle

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