"""
Train the LSTM forecaster (TensorFlow/Keras) and save it + the scalers needed
for inference.

Writes:
  ml/lstm_forecaster.keras  - trained model
  ml/scaler.json            - feature + target scalers needed at inference time
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.generate_data import generate_dataset, chronological_split
from ml.lstm_forecaster import (
    build_model, Scaler, build_dataset, add_calendar_features,
    ALL_FEATURE_COLS, FEATURE_COLS, make_pinball_loss, OUTPUT_LEN,
)

MODEL_DIR = Path(__file__).resolve().parent


def split_profiles(days_per_profile=150):
    profile_dfs = generate_dataset(days_per_profile)
    train, val, test = {}, {}, {}
    for name, df in profile_dfs.items():
        tr, va, te = chronological_split(df)
        train[name], val[name], test[name] = tr, va, te
    return train, val, test


def train():
    print("Generating data...")
    train_dfs, val_dfs, test_dfs = split_profiles(days_per_profile=150)

    # fit scaler on TRAIN ONLY (no peeking at val/test stats)
    scaler = Scaler()
    all_train_feat = np.concatenate([
        add_calendar_features(df)[ALL_FEATURE_COLS].values.astype(np.float32)
        for df in train_dfs.values()
    ])
    scaler.fit(all_train_feat)

    print("Building windows...")
    X_train, Y_train = build_dataset(train_dfs, scaler, stride=6)
    X_val, Y_val = build_dataset(val_dfs, scaler, stride=24)
    X_test, Y_test = build_dataset(test_dfs, scaler, stride=24)
    print(f"train windows: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    # IMPORTANT: cost (~0.04-0.11 std) and cpu/mem/disk (~11-26 std) are on
    # wildly different scales. Training on raw units lets utilization error
    # drown out the cost signal almost completely. Scale every target to
    # comparable units before training, convert back to real units after.
    n_targets = len(FEATURE_COLS)
    target_scaler = Scaler()
    target_scaler.fit(Y_train.reshape(-1, n_targets))

    def scale_targets(Y):
        flat = target_scaler.transform(Y.reshape(-1, n_targets))
        return flat.reshape(Y.shape)

    Y_train_s = scale_targets(Y_train)
    Y_val_s = scale_targets(Y_val)

    cost_idx = FEATURE_COLS.index("cost_per_hour")
    util_idx = [FEATURE_COLS.index(c) for c in ["cpu_usage", "memory_usage", "disk_usage"]]

    cost_train = Y_train_s[:, :, cost_idx:cost_idx + 1]      # (N,168,1)
    util_train = Y_train_s[:, :, util_idx]                   # (N,168,3)
    cost_val = Y_val_s[:, :, cost_idx:cost_idx + 1]
    util_val = Y_val_s[:, :, util_idx]

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"cost_out": make_pinball_loss(), "util_out": "mse"},
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]

    print("Training...")
    model.fit(
        X_train,
        {"cost_out": cost_train, "util_out": util_train},
        validation_data=(X_val, {"cost_out": cost_val, "util_out": util_val}),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(MODEL_DIR / "lstm_forecaster.keras")

    with open(MODEL_DIR / "scaler.json", "w") as f:
        json.dump({
            "mean": scaler.mean.tolist(), "std": scaler.std.tolist(), "cols": ALL_FEATURE_COLS,
            "target_mean": target_scaler.mean.tolist(), "target_std": target_scaler.std.tolist(),
            "target_cols": FEATURE_COLS,
        }, f)

    print("Saved model -> ml/lstm_forecaster.keras")
    print("Saved scalers -> ml/scaler.json")


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"Done in {time.time() - t0:.1f}s")