"""
Synthetic cloud usage data generator (v2).

Improvement over v1:
- 3 distinct workload profiles instead of 1 (webapp, database, batch)
- slow trend (usage drifting up over time) so the task isn't trivially flat
- noise scales with load (heteroscedastic), not fixed-size
- cost has a mild non-linear "burst pricing" kink above 80% cpu
- sparse anomaly spikes
- chronological train/val/test split helper (no shuffling — this is a time series)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def _profile_base(profile: str, hour: int, weekday: bool):
    """Return (cpu_base, mem_base, disk_base) for a given profile/hour/weekday."""
    if profile == "webapp":
        if weekday:
            if 9 <= hour <= 17:
                return 65, 75, 70
            elif 7 <= hour <= 8 or 18 <= hour <= 20:
                return 42, 52, 52
            else:
                return 15, 25, 35
        else:
            if 10 <= hour <= 17:
                return 25, 35, 40
            else:
                return 10, 20, 30

    if profile == "database":
        # flat-ish, mild daytime bump, always-on baseline
        if 8 <= hour <= 20:
            return 45, 60, 55
        return 35, 50, 50

    if profile == "batch":
        # short nightly batch window, idle rest of day
        if 1 <= hour <= 3:
            return 80, 60, 75
        return 8, 15, 20

    raise ValueError(profile)


def generate_profile(profile: str, days: int, start_date: str = "2024-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start=start_date, periods=days * 24, freq="h")
    rows = []
    total_hours = len(timestamps)

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        weekday = ts.weekday() < 5

        cpu_base, mem_base, disk_base = _profile_base(profile, hour, weekday)

        # slow upward trend over the whole period (simulates gradual real growth)
        trend_factor = 1.0 + 0.15 * (i / total_hours)
        cpu_base *= trend_factor
        mem_base *= trend_factor

        # noise that scales with the load itself (heteroscedastic, more realistic)
        cpu = RNG.normal(cpu_base, 4 + 0.12 * cpu_base)
        memory = RNG.normal(mem_base, 3 + 0.10 * mem_base)
        disk = RNG.normal(disk_base, 3 + 0.08 * disk_base)

        cpu = float(np.clip(cpu, 0, 100))
        memory = float(np.clip(memory, 0, 100))
        disk = float(np.clip(disk, 0, 100))

        # cost: linear in cpu/mem, but with a "burst pricing" kink above 80% cpu
        burst_penalty = 0.004 * max(cpu - 80, 0) ** 1.3
        cost = (
            0.10
            + 0.0021 * cpu
            + 0.0016 * memory
            + burst_penalty
            + RNG.normal(0, 0.012)
        )
        cost = round(max(cost, 0.08), 4)

        rows.append({
            "timestamp": ts,
            "cpu_usage": cpu,
            "memory_usage": memory,
            "disk_usage": disk,
            "cost_per_hour": cost,
        })

    df = pd.DataFrame(rows)

    # sparse anomaly spikes (~0.4% of rows) - e.g. traffic surge / runaway process
    n_anomalies = max(1, int(len(df) * 0.004))
    anomaly_idx = RNG.choice(len(df), size=n_anomalies, replace=False)
    df.loc[anomaly_idx, "cpu_usage"] = RNG.uniform(88, 100, size=n_anomalies)
    df.loc[anomaly_idx, "memory_usage"] = RNG.uniform(85, 100, size=n_anomalies)
    df.loc[anomaly_idx, "cost_per_hour"] *= RNG.uniform(1.3, 1.8, size=n_anomalies)

    df["profile"] = profile
    return df


def generate_dataset(days_per_profile: int = 150):
    """Generate all 3 profiles as separate dataframes (kept separate so we can
    split each one chronologically without leaking one profile's future into
    another profile's past)."""
    profiles = ["webapp", "database", "batch"]
    return {p: generate_profile(p, days_per_profile, start_date="2024-01-01") for p in profiles}


def chronological_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    """Split a single time-ordered dataframe into train/val/test by position,
    never by random shuffle (shuffling a time series leaks the future)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def build_train_val_test(days_per_profile: int = 150):
    """Generate all profiles, split each chronologically, then combine the
    splits across profiles into final train/val/test sets."""
    profile_dfs = generate_dataset(days_per_profile)

    train_parts, val_parts, test_parts = [], [], []
    for name, df in profile_dfs.items():
        tr, va, te = chronological_split(df)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, val, test


if __name__ == "__main__":
    train, val, test = build_train_val_test(days_per_profile=150)
    print(f"train: {len(train)} rows | val: {len(val)} rows | test: {len(test)} rows")
    print("\nPer-profile row counts (train):")
    print(train["profile"].value_counts())
    print("\nSample stats (train):")
    print(train[["cpu_usage", "memory_usage", "disk_usage", "cost_per_hour"]].describe().round(2))


    train.to_csv("data_preparation/train.csv", index=False)
    val.to_csv("data_preparation/val.csv", index=False)
    test.to_csv("data_preparation/test.csv", index=False)