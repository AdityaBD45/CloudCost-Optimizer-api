import os
import pickle
from pathlib import Path
import pandas as pd

from data_preparation.generate_data import generate_time_series_data
from ml.cost_predictor import CostPredictor


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "sample_generated.csv"
MODEL_PATH = Path(__file__).resolve().parent / "cost_predictor.pkl"


def load_or_generate_data():
    """Load existing CSV or generate synthetic dataset."""
    if DATA_PATH.exists():
        print("[INFO] Loading sample_generated.csv...")
        df = pd.read_csv(DATA_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    print("[INFO] sample_generated.csv not found. Generating synthetic dataset...")
    df = generate_time_series_data(days=30)
    df.to_csv(DATA_PATH, index=False)
    print("[INFO] Saved new sample_generated.csv")
    return df


def save_model_pickle(model, path):
    """Save model using Python pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved (pickle) → {path}")


def main():
    print("[INFO] Preparing dataset...")
    df = load_or_generate_data()

    print("[INFO] Training CostPredictor...")
    model = CostPredictor()
    results = model.train(df)
    print("[INFO] Training completed:", results)

    print("[INFO] Saving trained model with pickle...")
    save_model_pickle(model, MODEL_PATH)


if __name__ == "__main__":
    main()
