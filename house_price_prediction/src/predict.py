"""Load the saved model and predict prices."""
import joblib
import pandas as pd

from src.config import CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)


def predict_price(house: dict) -> float:
    model = load_model()
    df = pd.DataFrame([house])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return float(model.predict(df)[0])
