import joblib
import pandas as pd
import os

# Load model and feature names from models folder
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "crop_choice_xgboost_model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "features_names.joblib")

print("📌 MODEL PATH:", MODEL_PATH)
print("📌 FEATURES PATH:", FEATURES_PATH)

# Load model + feature names
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

print("✅ Model loaded")
print("✅ Feature names loaded:", feature_names)


def predict_crop_choice(input_json: dict):
    """
    Input: JSON dict from frontend
    Output: prediction with class and probability
    """

    # Check for missing fields
    missing = [f for f in feature_names if f not in input_json]
    if missing:
        return {
            "success": False,
            "error": "Missing required fields",
            "missing": missing
        }

    # Build row in correct order
    row = {f: input_json[f] for f in feature_names}
    df = pd.DataFrame([row])

    # Prediction
    pred_class = int(model.predict(df)[0])
    label = "GOOD" if pred_class == 1 else "BAD"

    # Probability
    prob_good = prob_bad = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        prob_bad = float(proba[0])
        prob_good = float(proba[1])

    return {
        "success": True,
        "prediction": {
            "label": label,
            "class": pred_class,
            "prob_good": prob_good,
            "prob_bad": prob_bad
        }
    }
