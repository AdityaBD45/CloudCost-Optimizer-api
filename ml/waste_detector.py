import pandas as pd
import numpy as np


class WasteDetector:
    """
    Waste detection engine.

    Input:  DataFrame with columns:
            ['timestamp', 'cpu_usage', 'memory_usage', 'disk_usage', 'cost_per_hour']

    Output: Dict with:
        - waste_analysis
        - idle_periods
        - recommendations
    """

    def __init__(self,
                 cpu_threshold: float = 20.0,
                 memory_threshold: float = 30.0,
                 min_idle_hours: int = 1):
        """
        cpu_threshold: CPU % below this is considered underutilized
        memory_threshold: Memory % below this is considered underutilized
        min_idle_hours: minimum continuous hours to treat as an idle period
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.min_idle_hours = min_idle_hours

    # ========== MAIN ANALYSIS METHOD ==========

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Main entry point: takes a DataFrame and returns the full analysis dict
        matching the /detect-waste response structure.
        """

        # If dataset is empty → avoid crashes
        if df is None or df.empty:
            return {
                "waste_analysis": {
                    "underutilized_score": 0.0,
                    "total_waste_percentage": 0.0,
                    "estimated_monthly_savings": 0.0
                },
                "idle_periods": [],
                "recommendations": ["No data available for analysis."]
            }

        # Ensure sorting by time
        df = df.sort_values("timestamp").reset_index(drop=True)

        idle_periods = self.detect_idle_periods(df)
        waste_analysis = self.waste_summary(df, idle_periods)
        recommendations = self.generate_recommendations(df, idle_periods, waste_analysis)

        return {
            "waste_analysis": waste_analysis,
            "idle_periods": idle_periods,
            "recommendations": recommendations
        }

    # ========== IDLE PERIOD DETECTION ==========

    def detect_idle_periods(self, df: pd.DataFrame) -> list:
        """
        Find continuous periods where CPU and Memory are below thresholds.
        Returns a list of dicts with start, end, duration, avg cpu/mem, wasted_cost.
        """

        df = df.copy()
        df["idle"] = (df["cpu_usage"] < self.cpu_threshold) & \
                     (df["memory_usage"] < self.memory_threshold)

        idle_periods = []
        start_idx = None

        for i in range(len(df)):
            row_idle = bool(df.loc[i, "idle"])

            if row_idle and start_idx is None:
                start_idx = i

            elif not row_idle and start_idx is not None:
                end_idx = i - 1
                self._append_idle_block(df, start_idx, end_idx, idle_periods)
                start_idx = None

        # If last rows are idle
        if start_idx is not None:
            end_idx = len(df) - 1
            self._append_idle_block(df, start_idx, end_idx, idle_periods)

        return idle_periods

    def _append_idle_block(self, df: pd.DataFrame, start_idx: int, end_idx: int, idle_periods: list):
        """Helper: build an idle block dict and append if duration meets requirement."""
        duration = end_idx - start_idx + 1

        if duration < self.min_idle_hours:
            return

        block = df.iloc[start_idx:end_idx + 1]
        wasted_cost = float(block["cost_per_hour"].sum())

        idle_periods.append({
            "start": str(block["timestamp"].iloc[0]),
            "end": str(block["timestamp"].iloc[-1]),
            "duration_hours": int(duration),
            "avg_cpu": round(float(block["cpu_usage"].mean()), 2),
            "avg_memory": round(float(block["memory_usage"].mean()), 2),
            "wasted_cost": round(wasted_cost, 2),
        })

    # ========== WASTE SUMMARY ==========

    def waste_summary(self, df: pd.DataFrame, idle_periods: list) -> dict:
        """
        Calculate waste metrics.
        """

        total_hours = len(df)
        if total_hours == 0:
            return {
                "underutilized_score": 0.0,
                "total_waste_percentage": 0.0,
                "estimated_monthly_savings": 0.0
            }

        idle_hours = sum(p["duration_hours"] for p in idle_periods)
        underutilized_score = idle_hours / total_hours
        total_waste_percentage = round(underutilized_score * 100, 2)

        observed_wasted_cost = sum(p["wasted_cost"] for p in idle_periods)
        days_observed = max((df["timestamp"].max() - df["timestamp"].min()).days + 1, 1)
        factor = 30 / days_observed  # approximate monthly scaling
        estimated_monthly_savings = round(observed_wasted_cost * factor, 2)

        return {
            "underutilized_score": round(underutilized_score, 2),
            "total_waste_percentage": total_waste_percentage,
            "estimated_monthly_savings": estimated_monthly_savings
        }

    # ========== RECOMMENDATIONS ENGINE ==========

    def generate_recommendations(self, df: pd.DataFrame, idle_periods: list, waste_analysis: dict) -> list:
        """
        Rule-based recommendations based on usage and waste metrics.
        """

        recs = []

        underutilized_score = waste_analysis.get("underutilized_score", 0)
        est_savings = waste_analysis.get("estimated_monthly_savings", 0)

        # 1) Right-sizing
        if underutilized_score > 0.4:
            recs.append("Reduce VM size (e.g., from 4 CPU / 16GB to 2 CPU / 8GB).")

        # 2) Scheduling shutdowns
        if idle_periods:
            longest_idle = max(idle_periods, key=lambda x: x["duration_hours"])
            recs.append(
                f"Schedule automatic shutdown during long idle periods "
                f"(e.g., from {longest_idle['start']} to {longest_idle['end']})."
            )

        # 3) Burstable instance
        high_cpu_hours = (df["cpu_usage"] > 75).sum()
        if high_cpu_hours > 0 and underutilized_score > 0.3:
            recs.append("Switch to burstable instance types for occasional spikes.")

        # 4) Autoscaling
        if df["cpu_usage"].max() > 85 and df["cpu_usage"].min() < 15:
            recs.append("Enable auto-scaling to handle peaks and reduce waste during low usage.")

        # 5) Savings summary
        if est_savings > 0:
            recs.append(f"Potential monthly savings: ${est_savings}.")

        if not recs:
            recs.append("Resource utilization looks normal.")

        return recs


# ========== WRAPPER FOR FASTAPI OR OTHER MODULES ==========

def run_waste_detection(df):
    detector = WasteDetector()
    return detector.analyze(df)
