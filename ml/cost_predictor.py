"""
LSTM-based 7-day cost / utilization forecaster.

Replaces the old recursive LightGBM approach entirely:
  - One forward pass over the last 72 hours of history produces the WHOLE
    7-day forecast directly (no recursive hour-by-hour loop, so no
    compounding error).
  - Cost predictions come with genuine P10/P50/P90 confidence intervals
    learned by the model - "confidence" below is derived from how tight
    that interval is, not a hardcoded number.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from ml.lstm_forecaster import (
    Scaler, add_calendar_features, ALL_FEATURE_COLS, FEATURE_COLS,
    INPUT_LEN, OUTPUT_LEN, make_pinball_loss,
)

MODEL_DIR = Path(__file__).resolve().parent


class CostPredictor:
    def __init__(self):
        self.model = keras.models.load_model(
            MODEL_DIR / "lstm_forecaster.keras",
            custom_objects={"loss_fn": make_pinball_loss()},
            compile=False,
        )

        with open(MODEL_DIR / "scaler.json") as f:
            d = json.load(f)

        self.feature_scaler = Scaler()
        self.feature_scaler.mean = np.array(d["mean"], dtype=np.float32)
        self.feature_scaler.std = np.array(d["std"], dtype=np.float32)

        self.target_scaler = Scaler()
        self.target_scaler.mean = np.array(d["target_mean"], dtype=np.float32)
        self.target_scaler.std = np.array(d["target_std"], dtype=np.float32)

        self.cost_idx = FEATURE_COLS.index("cost_per_hour")
        self.util_idx = [FEATURE_COLS.index(c) for c in ["cpu_usage", "memory_usage", "disk_usage"]]

    # -----------------------------------------------------------
    # Build the model's required 72-hour input window from raw user data
    # -----------------------------------------------------------
    def _build_input_window(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        if len(df) < INPUT_LEN:
            # Not enough history for a full 72h window - repeat what we have
            # so the API still returns a forecast instead of erroring out.
            reps = int(np.ceil(INPUT_LEN / len(df)))
            df = pd.concat([df] * reps, ignore_index=True).tail(INPUT_LEN).reset_index(drop=True)
        else:
            df = df.tail(INPUT_LEN).reset_index(drop=True)

        df_feat = add_calendar_features(df)
        feat = df_feat[ALL_FEATURE_COLS].values.astype(np.float32)
        scaled = self.feature_scaler.transform(feat)
        return scaled[np.newaxis, :, :]  # add batch dim -> (1, 72, n_features)

    # -----------------------------------------------------------
    # Predict next 7 days - single forward pass, no recursion
    # -----------------------------------------------------------
    def predict_next_7_days(self, df: pd.DataFrame):
        X = self._build_input_window(df)
        cost_pred, util_pred = self.model.predict(X, verbose=0)

        cost_p10 = self.target_scaler.inverse_transform_col(cost_pred[0, :, 0], self.cost_idx)
        cost_p50 = self.target_scaler.inverse_transform_col(cost_pred[0, :, 1], self.cost_idx)
        cost_p90 = self.target_scaler.inverse_transform_col(cost_pred[0, :, 2], self.cost_idx)

        cpu = self.target_scaler.inverse_transform_col(util_pred[0, :, 0], self.util_idx[0])
        mem = self.target_scaler.inverse_transform_col(util_pred[0, :, 1], self.util_idx[1])

        cpu = np.clip(cpu, 0, 100)
        mem = np.clip(mem, 0, 100)
        cost_p10 = np.clip(cost_p10, 0, None)
        cost_p50 = np.clip(cost_p50, 0, None)
        cost_p90 = np.clip(cost_p90, 0, None)

        last_ts = pd.to_datetime(df["timestamp"]).max()
        timestamps = [last_ts + pd.Timedelta(hours=h + 1) for h in range(OUTPUT_LEN)]

        hourly = pd.DataFrame({
            "timestamp": timestamps,
            "cost_p10": cost_p10, "cost_p50": cost_p50, "cost_p90": cost_p90,
            "cpu_usage": cpu, "memory_usage": mem,
        })
        hourly["date"] = hourly["timestamp"].dt.date

        daily = hourly.groupby("date", as_index=False).agg(
            cost=("cost_p50", "sum"),
            cost_low=("cost_p10", "sum"),
            cost_high=("cost_p90", "sum"),
            avg_cpu=("cpu_usage", "mean"),
            avg_memory=("memory_usage", "mean"),
        )

        # Genuine confidence: derived from how wide the P10-P90 band is
        # relative to the median forecast. A tight band = high confidence.
        # This replaces the old hardcoded `confidence: 0.9`.
        spread = (daily["cost_high"] - daily["cost_low"]) / daily["cost"].clip(lower=1e-6)
        daily["confidence"] = (1 - spread).clip(0.3, 0.99)

        trend_strength = round(float(daily["cost"].pct_change().mean() or 0), 4)
        if trend_strength > 0.01:
            trend = "increasing"
        elif trend_strength < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"

        result = []
        for _, r in daily.iterrows():
            result.append({
                "date": str(r["date"]),
                "weekday": pd.to_datetime(r["date"]).strftime("%A"),
                "cost": round(float(r["cost"]), 4),
                "cost_range": [round(float(r["cost_low"]), 4), round(float(r["cost_high"]), 4)],
                "avg_cpu": round(float(r["avg_cpu"]), 2),
                "avg_memory": round(float(r["avg_memory"]), 2),
                "confidence": round(float(r["confidence"]), 2),
            })

        return {
            "predicted_cost_next_7_days": result,
            "total_predicted_weekly_cost": round(float(daily["cost"].sum()), 4),
            "trend": trend,
            "trend_strength": trend_strength,
        }

    # -----------------------------------------------------------
    # Predict peak loads (based on recent actual history, not a forecast)
    # -----------------------------------------------------------
    def predict_performance(self, df):
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        cpu_roll = df["cpu_usage"].rolling(6).mean()
        mem_roll = df["memory_usage"].rolling(6).mean()

        cpu_peak = float(cpu_roll.max())
        mem_peak = float(mem_roll.max())
        peak_time = str(df.loc[cpu_roll.idxmax(), "timestamp"])

        combined = (cpu_peak + mem_peak) / 2
        if combined > 80:
            risk = "high"
        elif combined > 50:
            risk = "medium"
        else:
            risk = "low"

        return {
            "expected_peak_cpu": round(cpu_peak, 2),
            "expected_peak_memory": round(mem_peak, 2),
            "peak_time_prediction": peak_time,
            "bottleneck_risk": risk,
        }

    # -----------------------------------------------------------
    # Optimization suggestions
    # NOTE: these remain simple rule-based heuristics (not model outputs) -
    # e.g. "idle at night -> suggest scheduling". That's intentional: this
    # part isn't a forecasting problem, just business rules. The confidence
    # values here are fixed by design, separate from the forecast confidence
    # above (which IS model-derived).
    # -----------------------------------------------------------
    def optimization_opportunities(self, df, cost_pred):
        df = df.sort_values("timestamp").reset_index(drop=True)

        ops = []
        hist_est = float(df["cost_per_hour"].mean() * 24 * 7)
        pred_week = cost_pred["total_predicted_weekly_cost"]

        if pred_week > hist_est:
            ops.append({
                "type": "right_sizing",
                "savings_potential": round(pred_week - hist_est, 2),
                "confidence": 0.9,
            })

        df["hour"] = df["timestamp"].dt.hour
        night = df[df["hour"] <= 4]
        if len(night) and night["cpu_usage"].mean() < 15:
            ops.append({
                "type": "scheduling",
                "savings_potential": round(pred_week * 0.2, 2),
                "confidence": 0.87,
            })

        if df["cpu_usage"].max() > 80 and df["cpu_usage"].mean() < 35:
            ops.append({
                "type": "burstable_instance",
                "savings_potential": round(pred_week * 0.18, 2),
                "confidence": 0.85,
            })

        if df["cpu_usage"].max() - df["cpu_usage"].mean() > 40:
            ops.append({
                "type": "auto_scaling",
                "savings_potential": round(pred_week * 0.15, 2),
                "confidence": 0.8,
            })

        if not ops:
            ops.append({
                "type": "monitoring",
                "savings_potential": 0,
                "confidence": 0.7,
            })

        return ops

    # -----------------------------------------------------------
    # MAIN ENTRYPOINT
    # -----------------------------------------------------------
    def analyze(self, df):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        cost_pred = self.predict_next_7_days(df)
        perf = self.predict_performance(df)
        ops = self.optimization_opportunities(df, cost_pred)

        return {
            "cost_predictions": cost_pred,
            "performance_predictions": perf,
            "optimization_opportunities": ops,
        }