"""
LSTM-based cloud cost/usage forecaster (TensorFlow / Keras).

Design (kept deliberately simple):
- Input: last 72 hours (3 days) of [cpu, memory, disk, cost, hour_sin, hour_cos,
  dow_sin, dow_cos, is_weekend]  -> 9 features per timestep.
- Body: 2-layer LSTM, take the final hidden state.
- Two output heads off that single hidden state:
    1. cost_out  -> 168 hours x 3 quantiles (P10, P50, P90)  [genuine confidence]
    2. util_out  -> 168 hours x 3 channels (cpu, memory, disk) point estimates
- This replaces the old approach of 4 separate LightGBM models predicted
  recursively hour-by-hour (which compounds error over a 7-day horizon).
  Here the whole week comes out of one forward pass.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_LEN = 72          # hours of history the model looks at
OUTPUT_LEN = 168         # hours forecast (7 days)
FEATURE_COLS = ["cpu_usage", "memory_usage", "disk_usage", "cost_per_hour"]
N_RAW_FEATURES = len(FEATURE_COLS)
N_FEATURES = N_RAW_FEATURES + 5  # + hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
QUANTILES = (0.1, 0.5, 0.9)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(float)
    return df


ALL_FEATURE_COLS = FEATURE_COLS + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


class Scaler:
    """Simple mean/std scaler fit on training data only, reused for val/test/inference."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, arr: np.ndarray):
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def inverse_transform_col(self, arr: np.ndarray, col_idx: int) -> np.ndarray:
        return arr * self.std[col_idx] + self.mean[col_idx]


def make_windows(df: pd.DataFrame, scaler: Scaler, stride: int = 6):
    """Slice one continuous time series into (input_window, output_window) pairs.
    Returns scaled X (N, INPUT_LEN, N_FEATURES) and raw (unscaled) y for cost/cpu/mem/disk
    of shape (N, OUTPUT_LEN, 4) — kept unscaled so loss/metrics are in real units.
    """
    df = add_calendar_features(df)
    feat = df[ALL_FEATURE_COLS].values.astype(np.float32)
    raw_targets = df[FEATURE_COLS].values.astype(np.float32)  # cpu, mem, disk, cost - unscaled

    scaled_feat = scaler.transform(feat)

    X, Y = [], []
    n = len(df)
    last_start = n - (INPUT_LEN + OUTPUT_LEN)
    for start in range(0, last_start + 1, stride):
        in_end = start + INPUT_LEN
        out_end = in_end + OUTPUT_LEN
        X.append(scaled_feat[start:in_end])
        Y.append(raw_targets[in_end:out_end])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def build_dataset(profile_dfs: dict, scaler: Scaler, stride: int):
    """profile_dfs: {profile_name: df} -> concatenated windows across profiles."""
    Xs, Ys = [], []
    for _name, df in profile_dfs.items():
        X, Y = make_windows(df, scaler, stride=stride)
        if len(X):
            Xs.append(X)
            Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys)


def build_model(n_features=N_FEATURES, hidden_size=64, output_len=OUTPUT_LEN):
    """Functional-API Keras model: 2-layer LSTM encoder -> two dense heads."""
    inputs = keras.Input(shape=(INPUT_LEN, n_features), name="history")

    x = layers.LSTM(hidden_size, return_sequences=True, dropout=0.1, name="lstm_1")(inputs)
    x = layers.LSTM(hidden_size, dropout=0.1, name="lstm_2")(x)  # final hidden state only

    # cost head -> 3 quantiles per hour
    cost_x = layers.Dense(128, activation="relu")(x)
    cost_x = layers.Dense(output_len * 3)(cost_x)
    cost_out = layers.Reshape((output_len, 3), name="cost_out")(cost_x)

    # utilization head -> cpu, memory, disk point estimate per hour
    util_x = layers.Dense(128, activation="relu")(x)
    util_x = layers.Dense(output_len * 3)(util_x)
    util_out = layers.Reshape((output_len, 3), name="util_out")(util_x)

    model = keras.Model(inputs=inputs, outputs=[cost_out, util_out], name="lstm_forecaster")
    return model


def make_pinball_loss(quantiles=QUANTILES):
    """Quantile (pinball) loss. y_true: (batch, output_len, 1) - broadcast against
    y_pred: (batch, output_len, 3) quantile predictions."""
    q_tensor = tf.constant(quantiles, dtype=tf.float32)  # shape (3,)

    def loss_fn(y_true, y_pred):
        y_true_b = tf.broadcast_to(y_true, tf.shape(y_pred))
        err = y_true_b - y_pred
        loss = tf.maximum((q_tensor - 1.0) * err, q_tensor * err)
        return tf.reduce_mean(loss)

    return loss_fn