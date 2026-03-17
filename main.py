from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import xgboost as xgb
import shap

app = FastAPI(title="Chronic Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Models & Scalers ─────────────────────────────────────
heart_model = xgb.XGBClassifier()
heart_model.load_model("models/Heart_xgboost_model.json")
heart_scaler = joblib.load("models/Heart_scaler.joblib")

diabetes_model = xgb.XGBClassifier()
diabetes_model.load_model("models/Diabetes_xgboost_model.json")
diabetes_scaler = joblib.load("models/Diabetes_scaler.joblib")

hypertension_model = xgb.XGBClassifier()
hypertension_model.load_model("models/hypertension_dataset_xgboost_model.json")
# No scaler for hypertension


# ─── Schemas ──────────────────────────────────────────────────

# Shared features (appear in 2+ models)
class SharedFeatures(BaseModel):
    Age: float
    BMI: float
    Glucose: float

class HeartInput(BaseModel):
    # Heart-specific (no overlap with shared page)
    Age: float
    Gender: float           # 0 = Female, 1 = Male
    ChestPainType: float    # 0-3
    MaxHeartRate: float
    ExerciseAngina: float   # 0 = No, 1 = Yes
    ST_Depression: float
    ST_Slope: float         # 0-2
    MajorVessels: float     # 0-3
    Thalassemia: float      # 0-3

class DiabetesInput(BaseModel):
    # Shared
    Age: float
    BMI: float
    Glucose: float
    # Diabetes-specific
    Pregnancies: float
    Insulin: float
    DiabetesPedigreeFunction: float

class HypertensionInput(BaseModel):
    # Shared
    Age: float
    BMI: float
    Glucose: float
    # Hypertension-specific
    Cholesterol: float
    Systolic_BP: float
    Diastolic_BP: float
    Smoking_Status: float          # 0 = No, 1 = Yes
    Physical_Activity_Level: float # 0 = Low, 1 = Medium, 2 = High
    Diabetes: float                # 0 = No, 1 = Yes


# ─── SHAP Helper ──────────────────────────────────────────────
def get_shap(model, arr, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(arr)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        pairs = dict(zip(feature_names, [round(float(v), 4) for v in shap_values[0]]))
        # Return top 3 by absolute value
        top3 = sorted(pairs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        return dict(top3)
    except Exception as e:
        return {}


# ─── Endpoints ────────────────────────────────────────────────

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    feature_names = [
        "Age", "Gender", "ChestPainType", "MaxHeartRate",
        "ExerciseAngina", "ST_Depression", "ST_Slope",
        "MajorVessels", "Thalassemia"
    ]
    features = [
        data.Age, data.Gender, data.ChestPainType, data.MaxHeartRate,
        data.ExerciseAngina, data.ST_Depression, data.ST_Slope,
        data.MajorVessels, data.Thalassemia
    ]
    arr = np.array([features])
    arr_scaled = heart_scaler.transform(arr)

    prediction  = int(heart_model.predict(arr_scaled)[0])
    probability = float(heart_model.predict_proba(arr_scaled)[0][1])
    shap_vals   = get_shap(heart_model, arr_scaled, feature_names)

    return {
        "disease": "Heart Disease",
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "shap_values": shap_vals
    }


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    feature_names = [
        "Pregnancies", "Glucose", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    features = [
        data.Pregnancies, data.Glucose, data.Insulin,
        data.BMI, data.DiabetesPedigreeFunction, data.Age
    ]
    arr = np.array([features])
    arr_scaled = diabetes_scaler.transform(arr)

    prediction  = int(diabetes_model.predict(arr_scaled)[0])
    probability = float(diabetes_model.predict_proba(arr_scaled)[0][1])
    shap_vals   = get_shap(diabetes_model, arr_scaled, feature_names)

    return {
        "disease": "Diabetes",
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "shap_values": shap_vals
    }


@app.post("/predict/hypertension")
def predict_hypertension(data: HypertensionInput):
    feature_names = [
        "Age", "BMI", "Cholesterol", "Systolic_BP", "Diastolic_BP",
        "Smoking_Status", "Physical_Activity_Level", "Diabetes", "Glucose"
    ]
    features = [
        data.Age, data.BMI, data.Cholesterol, data.Systolic_BP, data.Diastolic_BP,
        data.Smoking_Status, data.Physical_Activity_Level, data.Diabetes, data.Glucose
    ]
    arr = np.array([features])
    # No scaler for hypertension model

    prediction  = int(hypertension_model.predict(arr)[0])
    probability = float(hypertension_model.predict_proba(arr)[0][1])
    shap_vals   = get_shap(hypertension_model, arr, feature_names)

    return {
        "disease": "Hypertension",
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "shap_values": shap_vals
    }


@app.get("/")
def root():
    return {"status": "Chronic Disease Prediction API is running ✅"}
