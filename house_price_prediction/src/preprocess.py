"""Cleaning, splitting and feature pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE,
                        TARGET, TEST_SIZE)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna(subset=[TARGET])
    return df[df[TARGET] > 0]


def split(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])
