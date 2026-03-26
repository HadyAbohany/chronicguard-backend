import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import xgboost as xgb
import shap
import google.generativeai as genai

app = FastAPI(title="Chronic Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── تحميل الموديلز والـ Scalers ─────────────────────────────
heart_model = xgb.XGBClassifier()
heart_model.load_model("Heart_xgboost_model.json")
heart_scaler = joblib.load("Heart_scaler.joblib")

diabetes_model = xgb.XGBClassifier()
diabetes_model.load_model("Diabetes_xgboost_model.json")
diabetes_scaler = joblib.load("Diabetes_scaler.joblib")

hypertension_model = xgb.XGBClassifier()
hypertension_model.load_model("hypertension_dataset_xgboost_model.json")


# ─── Schemas ──────────────────────────────────────────────────

class HeartInput(BaseModel):
    Age: float
    Gender: float
    ChestPainType: float
    MaxHeartRate: float
    ExerciseAngina: float
    ST_Depression: float
    ST_Slope: float
    MajorVessels: float
    Thalassemia: float

class DiabetesInput(BaseModel):
    Age: float
    BMI: float
    Glucose: float
    Pregnancies: float
    Insulin: float
    DiabetesPedigreeFunction: float

class HypertensionInput(BaseModel):
    Age: float
    BMI: float
    Glucose: float
    Cholesterol: float
    Systolic_BP: float
    Diastolic_BP: float
    Smoking_Status: float
    Physical_Activity_Level: float
    Diabetes: float

# استخدام List من typing بدل list[] عشان يشتغل على كل Python versions
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    message: str
    history: List[ChatMessage]  # ← التعديل هنا
    results: Optional[dict] = {}  # ← Optional عشان لو الفرونت مابعتهوش


# ─── SHAP Helper ──────────────────────────────────────────────

def get_shap(model, arr, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(arr)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        pairs = dict(zip(feature_names, [round(float(v), 4) for v in shap_values[0]]))
        top3 = sorted(pairs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        return dict(top3)
    except:
        return {}


# ─── Endpoints التوقع ─────────────────────────────────────────

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
    prediction  = int(hypertension_model.predict(arr)[0])
    probability = float(hypertension_model.predict_proba(arr)[0][1])
    shap_vals   = get_shap(hypertension_model, arr, feature_names)
    return {
        "disease": "Hypertension",
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "shap_values": shap_vals
    }


# ─── Endpoint الـ Chatbot ─────────────────────────────────────

@app.post("/chat")
def chat(data: ChatInput):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"reply": "API key not configured."}

        genai.configure(api_key=api_key)

        heart        = data.results.get("heart", {}) if data.results else {}
        diabetes     = data.results.get("diabetes", {}) if data.results else {}
        hypertension = data.results.get("hypertension", {}) if data.results else {}

        system_prompt = f"""
You are a friendly and professional medical AI assistant for ChronicGuard.

The user's health risk results:
- Heart Disease Risk: {heart.get("probability", "N/A")}%
  Top factors: {heart.get("shap_values", {})}

- Diabetes Risk: {diabetes.get("probability", "N/A")}%
  Top factors: {diabetes.get("shap_values", {})}

- Hypertension Risk: {hypertension.get("probability", "N/A")}%
  Top factors: {hypertension.get("shap_values", {})}

Risk levels: Low = below 30%, Moderate = 30-60%, High = above 60%

Rules:
- Answer questions about the user's results only
- Explain risk factors in simple non-technical language
- Give practical lifestyle advice
- Always recommend consulting a real doctor
- Keep answers under 120 words
- Be warm and supportive
- Respond in the SAME LANGUAGE the user writes in (Arabic or English)
- Never diagnose — you assess risk only
"""

        # تحويل الـ history للشكل الصح
        gemini_history = []
        for msg in data.history:
            gemini_role = "user" if msg.role == "user" else "model"
            gemini_history.append({
                "role": gemini_role,
                "parts": [msg.content]
            })

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt
        )

        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(data.message)

        return {"reply": response.text}

    except Exception as e:
        # بنرجع الـ error الحقيقي عشان نعرف المشكلة
        return {"reply": f"Error: {str(e)}"}


# ─── Root ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Chronic Disease Prediction API is running ✅"}
