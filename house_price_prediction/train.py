"""Train, evaluate and save the model:  python train.py"""
import joblib

from src.config import MODEL_PATH
from src.data import load_data
from src.evaluate import evaluate
from src.model import select_best
from src.preprocess import clean, split


def main():
    df = clean(load_data())
    print(f"Loaded {len(df)} rows")
    X_train, X_test, y_train, y_test = split(df)

    print("Cross-validating models:")
    name, model = select_best(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print(f"\nBest model: {name}")
    print(f"  MAE  : {metrics['MAE']:,.0f}")
    print(f"  RMSE : {metrics['RMSE']:,.0f}")
    print(f"  R²   : {metrics['R2']:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
