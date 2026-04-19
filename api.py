"""
Part 4 - FastAPI Backend
API server untuk inference model klasifikasi & regresi.
Jalankan: uvicorn api:app --reload --port 8000
Swagger UI: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -- Schemas --------------------------------------------------------------
class StudentInput(BaseModel):
    gender: str = Field(..., examples=["Male"])
    branch: str = Field(..., examples=["CSE"])
    cgpa: float = Field(..., ge=0.0, le=10.0, examples=[8.5])
    tenth_percentage: float = Field(..., ge=0.0, le=100.0, examples=[85.0])
    twelfth_percentage: float = Field(..., ge=0.0, le=100.0, examples=[88.0])
    backlogs: int = Field(..., ge=0, examples=[0])
    study_hours_per_day: float = Field(..., ge=0.0, le=24.0, examples=[5.0])
    attendance_percentage: float = Field(..., ge=0.0, le=100.0, examples=[85.0])
    projects_completed: int = Field(..., ge=0, examples=[6])
    internships_completed: int = Field(..., ge=0, examples=[2])
    coding_skill_rating: int = Field(..., ge=1, le=5, examples=[4])
    communication_skill_rating: int = Field(..., ge=1, le=5, examples=[4])
    aptitude_skill_rating: int = Field(..., ge=1, le=5, examples=[4])
    hackathons_participated: int = Field(..., ge=0, examples=[3])
    certifications_count: int = Field(..., ge=0, examples=[4])
    sleep_hours: float = Field(..., ge=0.0, le=24.0, examples=[7.0])
    stress_level: int = Field(..., ge=1, le=10, examples=[5])
    part_time_job: str = Field(..., examples=["No"])
    family_income_level: str = Field(..., examples=["Medium"])
    city_tier: str = Field(..., examples=["Tier 2"])
    internet_access: str = Field(..., examples=["Yes"])
    extracurricular_involvement: str = Field(..., examples=["Medium"])


class ClassificationOutput(BaseModel):
    placement_status: str
    probability_placed: float
    probability_not_placed: float


class RegressionOutput(BaseModel):
    salary_lpa: float


class PredictionOutput(BaseModel):
    placement_status: str
    probability_placed: float
    salary_lpa: Optional[float] = None
    note: str


# -- Model loading --------------------------------------------------------
MODELS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODELS["classifier"] = joblib.load("models/placement_classifier.pkl")
    MODELS["regressor"]  = joblib.load("models/salary_regressor.pkl")
    print("Models loaded.")
    yield
    MODELS.clear()


app = FastAPI(
    title="UTS MODEP - Placement & Salary API",
    description="Decoupled inference API untuk prediksi placement dan estimasi gaji.",
    version="1.0",
    lifespan=lifespan,
)


# -- Helpers --------------------------------------------------------------
def to_dataframe(inp: StudentInput) -> pd.DataFrame:
    row = inp.model_dump()
    row["academic_score"] = (row["cgpa"]*10 + row["tenth_percentage"] + row["twelfth_percentage"]) / 3
    row["skill_avg"]      = (row["coding_skill_rating"]
                             + row["communication_skill_rating"]
                             + row["aptitude_skill_rating"]) / 3
    return pd.DataFrame([row])


# -- Endpoints ------------------------------------------------------------
@app.get("/", tags=["health"])
def root():
    return {"message": "UTS MODEP API is running", "docs": "/docs"}


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "classifier_loaded": "classifier" in MODELS,
        "regressor_loaded":  "regressor" in MODELS,
    }


@app.post("/predict/classification", response_model=ClassificationOutput, tags=["predict"])
def predict_classification(data: StudentInput):
    """Prediksi status penempatan (Placed / Not Placed) beserta probabilitasnya."""
    model = MODELS.get("classifier")
    if model is None:
        raise HTTPException(503, "Classifier not loaded")

    X = to_dataframe(data)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)

    return ClassificationOutput(
        placement_status=pred,
        probability_placed=float(proba[classes.index("Placed")]),
        probability_not_placed=float(proba[classes.index("Not Placed")]),
    )


@app.post("/predict/regression", response_model=RegressionOutput, tags=["predict"])
def predict_regression(data: StudentInput):
    """Prediksi estimasi gaji (LPA). Catatan: model hanya dilatih di siswa Placed."""
    model = MODELS.get("regressor")
    if model is None:
        raise HTTPException(503, "Regressor not loaded")

    X = to_dataframe(data)
    salary = max(0.0, float(model.predict(X)[0]))
    return RegressionOutput(salary_lpa=salary)


@app.post("/predict", response_model=PredictionOutput, tags=["predict"])
def predict_combined(data: StudentInput):
    """
    Endpoint two-stage:
    1. Klasifikasi placement.
    2. Jika Placed, jalankan regresi gaji. Jika Not Placed, salary = None.
    """
    clf = MODELS.get("classifier")
    reg = MODELS.get("regressor")
    if clf is None or reg is None:
        raise HTTPException(503, "Models not loaded")

    X = to_dataframe(data)
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    prob_placed = float(proba[classes.index("Placed")])

    if pred == "Placed":
        salary = max(0.0, float(reg.predict(X)[0]))
        return PredictionOutput(
            placement_status=pred,
            probability_placed=prob_placed,
            salary_lpa=salary,
            note="Prediksi gaji tersedia karena siswa diprediksi Placed.",
        )

    return PredictionOutput(
        placement_status=pred,
        probability_placed=prob_placed,
        salary_lpa=None,
        note="Prediksi gaji tidak dijalankan karena status Not Placed.",
    )
