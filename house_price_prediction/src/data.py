"""Data loading. Uses your CSV if present, otherwise generates sample data."""
import numpy as np
import pandas as pd

from src.config import DATA_PATH, TARGET


def generate_sample_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic housing dataset (for demo/testing)."""
    rng = np.random.default_rng(seed)
    locations = ["city_center", "suburb", "rural", "waterfront"]
    loc_premium = {"city_center": 80_000, "suburb": 30_000, "rural": -20_000, "waterfront": 150_000}

    df = pd.DataFrame({
        "area_sqft": rng.integers(500, 4500, n),
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.integers(1, 5, n),
        "age_years": rng.integers(0, 60, n),
        "garage_spaces": rng.integers(0, 4, n),
        "location": rng.choice(locations, n),
    })
    df[TARGET] = (
        50_000
        + df["area_sqft"] * 120
        + df["bedrooms"] * 8_000
        + df["bathrooms"] * 12_000
        - df["age_years"] * 900
        + df["garage_spaces"] * 6_000
        + df["location"].map(loc_premium)
        + rng.normal(0, 20_000, n)
    ).round(0)
    return df


def load_data() -> pd.DataFrame:
    """Load data/house_data.csv, or create it from sample data if missing."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"No dataset found. Sample data saved to {DATA_PATH}")
    return df
