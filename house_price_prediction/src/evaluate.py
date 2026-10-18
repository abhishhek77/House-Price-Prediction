"""Evaluation metrics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": r2_score(y_test, pred),
    }
