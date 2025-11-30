import pandas as pd

def parse_csv(file_path):
    """Parse CSV and return standardized DataFrame."""
    df = pd.read_csv(file_path)

    required_cols = ["timestamp", "cpu_usage", "memory_usage", "disk_usage", "cost_per_hour"]

    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Missing column: {col}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def parse_json(json_data):
    """Convert JSON time-series input to DataFrame."""
    df = pd.DataFrame({
        "timestamp": json_data["timestamps"],
        "cpu_usage": json_data["cpu_usage"],
        "memory_usage": json_data["memory_usage"],
        "disk_usage": json_data.get("disk_usage", [None] * len(json_data["cpu_usage"])),
        "cost_per_hour": json_data["cost_per_hour"],
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def parse_input(input_data, is_csv=False, is_json=False):
    """Unified parser: routes to CSV or JSON parser."""
    if is_csv:
        return parse_csv(input_data)

    if is_json:
        return parse_json(input_data)

    raise Exception("Specify is_csv=True or is_json=True")
