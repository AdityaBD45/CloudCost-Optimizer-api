import pandas as pd
import numpy as np

def generate_time_series_data(days=30, freq="h", anomalies=True):
    """
    Generate realistic time-series cloud infrastructure data:
      - Weekday peaks (9 AM–6 PM)
      - Weekend low usage
      - Morning ramp-up & evening cooldown
      - Night idle workload
      - Cost correlated with CPU/memory
      - Optional spike anomalies
    """
    timestamps = pd.date_range(start="2024-01-01", periods=days*24, freq=freq)
    rows = []

    for ts in timestamps:
        hour = ts.hour
        weekday = ts.weekday() < 5
        
        # -------------------------
        # Workload patterns
        # -------------------------
        if weekday:
            if 9 <= hour <= 17:   # Working hours peak
                cpu_base = 65
                mem_base = 75
                disk_base = 70
            elif 7 <= hour <= 8:  # Morning ramp-up
                cpu_base = 45
                mem_base = 55
                disk_base = 55
            elif 18 <= hour <= 20:  # Evening cooldown
                cpu_base = 40
                mem_base = 50
                disk_base = 50
            else:  # Night idle
                cpu_base = 15
                mem_base = 25
                disk_base = 35
        else:
            # Weekend — much lower usage
            if 10 <= hour <= 17:
                cpu_base = 25
                mem_base = 35
                disk_base = 40
            else:
                cpu_base = 10
                mem_base = 20
                disk_base = 30

        # -------------------------
        # Add natural fluctuation
        # -------------------------
        cpu = np.random.normal(cpu_base, 8)
        memory = np.random.normal(mem_base, 6)
        disk = np.random.normal(disk_base, 5)

        # -------------------------
        # COST FORMULA
        # -------------------------
        cost = (
            0.12 +
            0.002 * cpu +
            0.0015 * memory +
            np.random.normal(0, 0.01)
        )

        rows.append({
            "timestamp": ts,
            "cpu_usage": float(np.clip(cpu, 0, 100)),
            "memory_usage": float(np.clip(memory, 0, 100)),
            "disk_usage": float(np.clip(disk, 0, 100)),
            "cost_per_hour": round(max(cost, 0.10), 3)
        })

    df = pd.DataFrame(rows)

    # -------------------------
    # Add anomalies (optional)
    # -------------------------
    if anomalies:
        anomaly_idx = np.random.choice(len(df), size=3, replace=False)
        df.loc[anomaly_idx, "cpu_usage"] = np.random.randint(85, 100, size=3)
        df.loc[anomaly_idx, "memory_usage"] = np.random.randint(85, 100, size=3)
        df.loc[anomaly_idx, "cost_per_hour"] *= 1.4  # spike cost

    return df
