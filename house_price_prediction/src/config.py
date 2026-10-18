"""Central configuration for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "house_data.csv"
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"

TARGET = "price"
NUMERIC_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces"]
CATEGORICAL_FEATURES = ["location"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
