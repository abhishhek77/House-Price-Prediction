"""Model definitions and training."""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocess import build_preprocessor


def get_candidates() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def make_pipeline(regressor) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", regressor)])


def select_best(X_train, y_train):
    """Pick the model with the best 5-fold CV R², then fit it on all training data."""
    best_name, best_score = None, float("-inf")
    for name, reg in get_candidates().items():
        score = cross_val_score(make_pipeline(reg), X_train, y_train, cv=5, scoring="r2").mean()
        print(f"  {name:<18} CV R² = {score:.4f}")
        if score > best_score:
            best_name, best_score = name, score
    pipe = make_pipeline(get_candidates()[best_name]).fit(X_train, y_train)
    return best_name, pipe
