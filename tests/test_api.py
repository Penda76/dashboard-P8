import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Charger les données de test une seule fois
df_test = pd.read_csv("app/data_sample/X_test_clean.csv")

# Prendre le premier client comme exemple
sample = df_test.iloc[0].to_dict()
client_id = sample["SK_ID_CURR"]

def test_post_predict():
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    res_json = response.json()
    assert "prediction" in res_json
    assert "probability" in res_json
    assert res_json["decision"] in ["Accepté", "Refusé"]

def test_get_predict_by_id_existing():
    response = client.get(f"/predict/{int(client_id)}")
    assert response.status_code == 200
    res_json = response.json()
    assert "prediction" in res_json
    assert res_json["client_id"] == int(client_id)

def test_get_predict_by_id_not_found():
    response = client.get("/predict/99999999")  # ID fictif
    assert response.status_code == 200
    assert "error" in response.json()
