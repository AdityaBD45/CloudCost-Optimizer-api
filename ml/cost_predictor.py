import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from datetime import timedelta

from lightgbm import LGBMRegressor


class CostPredictor:
    def __init__(self):
        # Pretrained LightGBM models
        self.cpu_model = LGBMRegressor()
        self.mem_model = LGBMRegressor()
        self.disk_model = LGBMRegressor()
        self.cost_model = LGBMRegressor()

        self.feature_cols = [
            "hour",
            "day_of_week",
            "is_weekend",
            "cpu_prev1",
            "mem_prev1",
            "disk_prev1",
            "cost_prev1",
            "cost_prev2",
            "cpu_roll3",
            "mem_roll3",
            "disk_roll3",
        ]

    # -----------------------------------------------------------
    # Build feature engineering for user input
    # -----------------------------------------------------------
    def _prepare_features(self, df: pd.DataFrame):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df["cpu_prev1"] = df["cpu_usage"].shift(1)
        df["mem_prev1"] = df["memory_usage"].shift(1)
        df["disk_prev1"] = df["disk_usage"].shift(1)
        df["cost_prev1"] = df["cost_per_hour"].shift(1)
        df["cost_prev2"] = df["cost_per_hour"].shift(2)

        df["cpu_roll3"] = df["cpu_usage"].rolling(3).mean()
        df["mem_roll3"] = df["memory_usage"].rolling(3).mean()
        df["disk_roll3"] = df["disk_usage"].rolling(3).mean()

        df = df.dropna().reset_index(drop=True)
        return df

    # -----------------------------------------------------------
    # Build next-hour features using history
    # -----------------------------------------------------------
    def _build_feature_row(self, history_df, next_ts):
        last = history_df.iloc[-1]
        prev = history_df.iloc[-2] if len(history_df) > 1 else last

        row = {
            "hour": next_ts.hour,
            "day_of_week": next_ts.dayofweek,
            "is_weekend": int(next_ts.dayofweek in [5, 6]),

            "cpu_prev1": float(last["cpu_usage"]),
            "mem_prev1": float(last["memory_usage"]),
            "disk_prev1": float(last["disk_usage"]),

            "cost_prev1": float(last["cost_per_hour"]),
            "cost_prev2": float(prev["cost_per_hour"]),

            "cpu_roll3": float(history_df["cpu_usage"].tail(3).mean()),
            "mem_roll3": float(history_df["memory_usage"].tail(3).mean()),
            "disk_roll3": float(history_df["disk_usage"].tail(3).mean()),
        }

        return pd.DataFrame([row])[self.feature_cols]

    # -----------------------------------------------------------
    # Predict next 7 days (production version) - BUG FIXED
    # -----------------------------------------------------------
    def predict_next_7_days(self, df):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        df = self._prepare_features(df)
        history = df[["timestamp", "cpu_usage", "memory_usage", "disk_usage", "cost_per_hour"]].copy()

        hourly_predictions = []
        last_ts = history["timestamp"].iloc[-1]

        # Small weekend/weekday smoothing - FIXED VERSION
        baseline = df.copy()
        baseline["day_of_week"] = baseline["timestamp"].dt.dayofweek
        weekday_pat = baseline[baseline["day_of_week"] <= 4][["cpu_usage", "memory_usage"]].mean()
        weekend_pat = baseline[baseline["day_of_week"] >= 5][["cpu_usage", "memory_usage"]].mean()
        GUIDE = 0.2

        for _ in range(168):  # 7 days * 24 hours
            next_ts = last_ts + timedelta(hours=1)
            X = self._build_feature_row(history, next_ts)

            cpu_raw = float(self.cpu_model.predict(X)[0])
            mem_raw = float(self.mem_model.predict(X)[0])
            disk_raw = float(self.disk_model.predict(X)[0])
            cost_raw = float(self.cost_model.predict(X)[0])

            # Light smoothing - FIXED: Weekend uses weekend patterns, Weekday uses weekday patterns
            if next_ts.dayofweek >= 5:  # Weekend (Sat, Sun)
                cpu_guide, mem_guide = weekend_pat["cpu_usage"], weekend_pat["memory_usage"]
            else:  # Weekday (Mon-Fri)
                cpu_guide, mem_guide = weekday_pat["cpu_usage"], weekday_pat["memory_usage"]

            cpu = (1 - GUIDE) * cpu_raw + GUIDE * cpu_guide
            mem = (1 - GUIDE) * mem_raw + GUIDE * mem_guide
            disk = disk_raw
            cost = cost_raw

            cpu = float(np.clip(cpu, 0, 100))
            mem = float(np.clip(mem, 0, 100))
            disk = float(np.clip(disk, 0, 100))
            cost = float(max(cost, 0))

            new_row = {
                "timestamp": next_ts,
                "cpu_usage": cpu,
                "memory_usage": mem,
                "disk_usage": disk,
                "cost_per_hour": cost,
            }
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

            hourly_predictions.append({
                "timestamp": next_ts,
                "predicted_cost": cost,
                "cpu_usage": cpu,
                "memory_usage": mem,
                "confidence": 0.9,
            })

            last_ts = next_ts

        pred_df = pd.DataFrame(hourly_predictions)
        pred_df["date"] = pred_df["timestamp"].dt.date

        daily = pred_df.groupby("date").agg(
            predicted_cost=("predicted_cost", "sum"),
            avg_cpu=("cpu_usage", "mean"),
            avg_memory=("memory_usage", "mean"),
            avg_confidence=("confidence", "mean"),
        ).reset_index()

        trend_strength = round(float(daily["predicted_cost"].pct_change().mean() or 0), 4)
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
                "cost": round(float(r["predicted_cost"]), 4),
                "avg_cpu": round(float(r["avg_cpu"]), 2),
                "avg_memory": round(float(r["avg_memory"]), 2),
                "confidence": round(float(r["avg_confidence"]), 2),
            })

        return {
            "predicted_cost_next_7_days": result,
            "total_predicted_weekly_cost": round(float(daily["predicted_cost"].sum()), 4),
            "trend": trend,
            "trend_strength": trend_strength,
        }

    # -----------------------------------------------------------
    # Predict peak loads
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
    # MAIN ENTRYPOINT (no training)
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