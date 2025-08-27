from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
import os, json
import mlflow
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="Risk-Demand-Lab API", version="0.1.0")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Carrega modelo e metadados
MODEL_DIR = Path("models/credit/latest")
MODEL_PATH = MODEL_DIR / "model.pkl"
META_PATH = MODEL_DIR / "metadata.json"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {"features": []}
FEATURES = meta.get("features", [])

class CreditRequest(BaseModel):
    # Envie {"features": {"renda": 3500, "idade": 42, ...}}
    features: dict[str, float]

@app.get("/")
def index():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_loaded": MODEL_PATH.exists(),
        "n_features": len(FEATURES),
    }

@app.post("/predict/credit")
def predict_credit(req: CreditRequest):
    assert model is not None, "Modelo não carregado. Treine com make train."
    # ordena as features conforme o treino
    row = [req.features.get(f, 0.0) for f in FEATURES]
    x = np.array([row], dtype=float)
    prob = float(model.predict_proba(x)[:, 1][0])
    return {"default_probability": prob, "used_features": FEATURES}
