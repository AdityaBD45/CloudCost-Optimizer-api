"""
Evaluate the LSTM forecaster on held-out test data, at the same daily
granularity the API actually returns to users (cost summed per day,
cpu/memory averaged per day).

Writes ml/metrics.json - this is the receipt for "high accuracy" claims:
a real number measured on data the model never saw during training.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.generate_data import generate_dataset, chronological_split
from ml.lstm_forecaster import (
    Scaler, add_calendar_features, ALL_FEATURE_COLS, FEATURE_COLS,
    INPUT_LEN, OUTPUT_LEN, make_pinball_loss,
)

MODEL_DIR = Path(__file__).resolve().parent


def load_scaler():
    with open(MODEL_DIR / "scaler.json") as f:
        d = json.load(f)
    s = Scaler()
    s.mean = np.array(d["mean"], dtype=np.float32)
    s.std = np.array(d["std"], dtype=np.float32)

    t = Scaler()
    t.mean = np.array(d["target_mean"], dtype=np.float32)
    t.std = np.array(d["target_std"], dtype=np.float32)
    return s, t


def make_eval_windows(df: pd.DataFrame, scaler: Scaler, stride: int = 24):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_feat = add_calendar_features(df)
    scaled_feat = scaler.transform(df_feat[ALL_FEATURE_COLS].values.astype(np.float32))
    raw_targets = df[FEATURE_COLS].values.astype(np.float32)

    X, Y = [], []
    n = len(df)
    last_start = n - (INPUT_LEN + OUTPUT_LEN)
    for start in range(0, last_start + 1, stride):
        in_end = start + INPUT_LEN
        out_end = in_end + OUTPUT_LEN
        X.append(scaled_feat[start:in_end])
        Y.append(raw_targets[in_end:out_end])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def daily_agg(hourly_cost, hourly_cpu, hourly_mem):
    """hourly_* shape (168,) -> 7 daily values (cost summed, cpu/mem averaged)."""
    cost_d = hourly_cost.reshape(7, 24).sum(axis=1)
    cpu_d = hourly_cpu.reshape(7, 24).mean(axis=1)
    mem_d = hourly_mem.reshape(7, 24).mean(axis=1)
    return cost_d, cpu_d, mem_d


def metrics(true, pred):
    true, pred = np.array(true), np.array(pred)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = float(np.mean(np.abs((true - pred) / np.clip(true, 1e-6, None))) * 100)
    r2 = r2_score(true, pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2), "R2": round(r2, 4)}


def main():
    print("Regenerating data (same seed -> same test set)...")
    profile_dfs = generate_dataset(days_per_profile=150)
    test_dfs = {}
    for name, df in profile_dfs.items():
        _, _, test = chronological_split(df)
        test_dfs[name] = test

    scaler, target_scaler = load_scaler()

    model = keras.models.load_model(
        MODEL_DIR / "lstm_forecaster.keras",
        custom_objects={"loss_fn": make_pinball_loss()},
        compile=False,
    )

    cost_idx = FEATURE_COLS.index("cost_per_hour")
    util_idx = [FEATURE_COLS.index(c) for c in ["cpu_usage", "memory_usage", "disk_usage"]]

    true_cost_all, pred_cost_all = [], []
    true_cpu_all, pred_cpu_all = [], []
    true_mem_all, pred_mem_all = [], []

    n_windows = 0
    for name, df in test_dfs.items():
        X, Y = make_eval_windows(df, scaler, stride=24)
        if len(X) == 0:
            continue
        n_windows += len(X)

        cost_pred, util_pred = model.predict(X, verbose=0)
        cost_p50_scaled = cost_pred[:, :, 1]  # median quantile

        cost_p50 = target_scaler.inverse_transform_col(cost_p50_scaled, cost_idx)
        util_np = np.stack([
            target_scaler.inverse_transform_col(util_pred[:, :, j], util_idx[j])
            for j in range(3)
        ], axis=-1)

        for i in range(len(X)):
            true_cost, true_cpu, true_mem = daily_agg(
                Y[i, :, cost_idx], Y[i, :, util_idx[0]], Y[i, :, util_idx[1]]
            )
            pred_cost, pred_cpu, pred_mem = daily_agg(
                cost_p50[i], util_np[i, :, 0], util_np[i, :, 1]
            )

            true_cost_all.extend(true_cost); pred_cost_all.extend(pred_cost)
            true_cpu_all.extend(true_cpu); pred_cpu_all.extend(pred_cpu)
            true_mem_all.extend(true_mem); pred_mem_all.extend(pred_mem)

    print(f"Evaluated on {n_windows} independent 7-day test windows ({n_windows*7} day-ahead forecasts)\n")

    results = {
        "n_test_windows": n_windows,
        "cost_per_day_$": metrics(true_cost_all, pred_cost_all),
        "avg_cpu_%": metrics(true_cpu_all, pred_cpu_all),
        "avg_memory_%": metrics(true_mem_all, pred_mem_all),
    }

    for section, vals in results.items():
        if section == "n_test_windows":
            continue
        print(f"{section}: MAE={vals['MAE']} RMSE={vals['RMSE']} MAPE={vals['MAPE_%']}% R2={vals['R2']}")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> ml/metrics.json")


if __name__ == "__main__":
    main()