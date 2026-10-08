from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src_pipeline import add_engineered_features, load_bundle

ROOT = Path(__file__).resolve().parent
BUNDLE = load_bundle(ROOT / "models" / "xgboost_failure_risk.joblib")
MODEL = BUNDLE["pipeline"]
THRESHOLD = float(BUNDLE["threshold"])

app = FastAPI(
    title="Predictive Maintenance Risk API",
    version="1.0.0",
    description=(
        "Educational predictive-maintenance prototype. The returned score is a model risk score, "
        "not a calibrated probability and not an autonomous maintenance decision."
    ),
)

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


class EquipmentReading(BaseModel):
    type: str = Field(default="L", pattern="^[LMH]$")
    air_temperature_k: float = Field(default=300.0, ge=250, le=350)
    process_temperature_k: float = Field(default=310.0, ge=250, le=380)
    rotational_speed_rpm: int = Field(default=1500, ge=500, le=4000)
    torque_nm: float = Field(default=40.0, ge=0, le=150)
    tool_wear_min: int = Field(default=100, ge=0, le=500)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": BUNDLE["metadata"]["model"]}


@app.post("/predict")
def predict(reading: EquipmentReading) -> dict:
    frame = pd.DataFrame(
        [
            {
                "Type": reading.type,
                "Air temperature [K]": reading.air_temperature_k,
                "Process temperature [K]": reading.process_temperature_k,
                "Rotational speed [rpm]": reading.rotational_speed_rpm,
                "Torque [Nm]": reading.torque_nm,
                "Tool wear [min]": reading.tool_wear_min,
            }
        ]
    )
    engineered = add_engineered_features(frame)
    score = float(MODEL.predict_proba(engineered)[:, 1][0])
    status = "FAILURE_RISK" if score >= THRESHOLD else "NORMAL"
    return {
        "status": status,
        "risk_score": round(score, 4),
        "decision_threshold": round(THRESHOLD, 4),
        "derived_features": {
            "temperature_gap_k": round(float(engineered["Temperature gap [K]"].iloc[0]), 3),
            "mechanical_power_w": round(float(engineered["Mechanical power [W]"].iloc[0]), 1),
            "wear_x_torque": round(float(engineered["Wear × torque"].iloc[0]), 1),
        },
        "warning": "Educational prototype: risk score is not a calibrated probability. Human review is required.",
    }
