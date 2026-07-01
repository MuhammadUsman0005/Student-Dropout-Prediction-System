# /* uvicorn app:app --reload --port 8000 */
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
import pickle
import os

app = FastAPI(title="Student Dropout Warning System API")

# Setup CORS so front-end can talk to back-end securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Data schema validation
class StudentDataInput(BaseModel):
    attendance: float = Field(..., ge=0.0, le=100.0)
    cgpa: float = Field(..., ge=0.0, le=4.0)
    failed_courses: int = Field(..., ge=0, le=10)
    fee_paid: int = Field(..., ge=0, le=1)
    part_time_job: int = Field(..., ge=0, le=1)
    mental_health: int = Field(..., ge=1, le=10)
    distance: float = Field(..., ge=0.0, le=200.0)
    family_income: int = Field(..., ge=0, le=2)

# Global holder for pipeline model
ml_pipeline = None

@app.on_event("startup")
def load_trained_model():
    global ml_pipeline
    model_path = 'model/dropout_pipeline.pkl'
    if not os.path.exists(model_path):
        raise RuntimeError(f"Trained model not found at {model_path}. Please execute train.py first!")
    with open(model_path, 'rb') as f:
        ml_pipeline = pickle.load(f)
    print("Production ML model pipeline is active and loaded.")

@app.post("/api/predict-dropout")
def predict_student_status(data: StudentDataInput):
    try:
        # Features map array format for pipeline input
        features_vector = np.array([[
            data.attendance,
            data.cgpa,
            data.failed_courses,
            data.fee_paid,
            data.part_time_job,
            data.mental_health,
            data.distance,
            data.family_income
        ]])
        
        # Calculate true matrix probability distribution
        probabilities = ml_pipeline.predict_proba(features_vector)[0]
        dropout_chance = float(probabilities[1]) * 100  # Probability of class 1 (Dropout)
        
        # Generate dynamic feature-wise insights from data vectors
        contributing_factors = []
        if data.attendance < 60:
            contributing_factors.append({"text": f"Attendance ({data.attendance}%) is critically low.", "level": "danger"})
        if data.cgpa < 2.0:
            contributing_factors.append({"text": f"CGPA ({data.cgpa}) is in the failing zone.", "level": "danger"})
        if data.failed_courses >= 2:
            contributing_factors.append({"text": f"Failed in {data.failed_courses} active courses.", "level": "danger"})
        if data.fee_paid == 0:
            contributing_factors.append({"text": "Pending university fee dues.", "level": "warn"})
        if data.mental_health < 4:
            contributing_factors.append({"text": f"Low structural psychological wellness score ({data.mental_health}/10).", "level": "danger"})

        if not contributing_factors:
            contributing_factors.append({"text": "All parameters are working normally within expected optimal limits.", "level": "ok"})

        return {
            "status": "success",
            "dropout_probability": round(dropout_chance, 2),
            "verdict": "HIGH DROPOUT RISK" if dropout_chance >= 50.0 else "LOW RISK — Likely to Complete",
            "factors": contributing_factors[:4]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")

# Mount the static directory to host index.html directly
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")