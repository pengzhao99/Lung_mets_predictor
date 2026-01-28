from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load model
try:
    with open('lung_model.pkl', 'rb') as f:
        model = pickle.load(f)

    if hasattr(model, 'model'):
        model.model.device = 'cpu'
        model.model.n_estimators = 1

except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Initialize FastAPI app with English metadata
app = FastAPI(
    title="Lung Cancer Metastasis Risk Prediction System",
    description="A TabPFN-based predictor for lung cancer metastasis probability"
)


# Define input data schema (field names converted: spaces/hyphens → underscores)
class PatientInput(BaseModel):
    Histologic_type: int  # "Histologic type" → Histologic_type
    NSE: float
    LDH: float
    CEA: float
    Nsclc_21_1: float  # "Nsclc-21-1" → Nsclc_21_1
    NK_count: float    # "NK-count" → NK_count
    CREA: float
    T_count: float     # "T-count" → T_count
    L_count: float     # "L-count" → L_count


@app.on_event("startup")
async def startup_event():
    print("Warming up model...")
    dummy = pd.DataFrame([[1, 10.0, 200.0, 5.0, 5.5, 100, 80.0, 1500.0, 1.0]],
                         columns=[f for f in PatientInput.__annotations__])
    _ = model.predict_proba(dummy)
    print("Warmup complete!")


# Prediction endpoint
@app.post("/predict")
def predict(data: PatientInput):
    try:
        # Use DataFrame to match training format
        feature_names = [
            "Histologic_type",
            "NSE",
            "LDH",
            "CEA",
            "Nsclc_21_1",
            "NK_count",
            "CREA",
            "T_count",
            "L_count"
        ]
        features_df = pd.DataFrame(
            [[data.Histologic_type, data.NSE, data.LDH, data.CEA, data.Nsclc_21_1, data.NK_count, data.CREA,
              data.T_count, data.L_count]], columns=feature_names)

        prob = model.predict_proba(features_df)[0][1]
        return {
            "metastasis_probability": round(float(prob), 4),
            "risk_level": "High Risk" if prob > 0.7 else "Medium Risk" if prob > 0.3 else "Low Risk"
        }
    except Exception as e:
        print(f"🚨 Prediction error: {e}")  # This will appear in Render logs!
        raise HTTPException(status_code=500, detail=str(e))


# Homepage: serve frontend HTML
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()