from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartDiseaseInput

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

model = None
model_metadata = None


@app.on_event("startup")
def load_model():
    global model, model_metadata
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run the training script before starting the API."
        )
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    model_metadata = artifact


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/info")
def info():
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "model_type": model_metadata["model_type"],
        "features": model_metadata["features"],
    }


@app.post("/predict")
def predict(payload: HeartDiseaseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_data = pd.DataFrame([payload.model_dump()])
    prediction = int(model.predict(input_data)[0])
    return {"heart_disease": bool(prediction)}
