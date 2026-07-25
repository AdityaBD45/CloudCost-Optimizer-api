import pandas as pd
import pytest

from data_preparation.generate_data import generate_profile
from data_preparation.parse_input import parse_csv
from ml.waste_detector import WasteDetector
from ml.cost_predictor import CostPredictor


# ---------- Data generation ----------

def test_generate_profile_produces_valid_schema():
    df = generate_profile("webapp", days=5)
    required_cols = {"timestamp", "cpu_usage", "memory_usage", "disk_usage", "cost_per_hour"}
    assert required_cols.issubset(df.columns)
    assert len(df) == 5 * 24
    assert df["cpu_usage"].between(0, 100).all()
    assert df["memory_usage"].between(0, 100).all()
    assert (df["cost_per_hour"] > 0).all()


def test_batch_profile_has_idle_hours():
    # batch profile is near-idle most of the day by design (see _profile_base)
    df = generate_profile("batch", days=3)
    assert (df["cpu_usage"] < 20).sum() > 0


# ---------- CSV parsing ----------

def test_parse_csv_roundtrip(tmp_path):
    df = generate_profile("webapp", days=2)
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    parsed = parse_csv(str(csv_path))
    assert pd.api.types.is_datetime64_any_dtype(parsed["timestamp"])
    assert set(["timestamp", "cpu_usage", "memory_usage", "disk_usage", "cost_per_hour"]).issubset(parsed.columns)
    assert len(parsed) == len(df)


def test_parse_csv_missing_column_raises(tmp_path):
    df = generate_profile("webapp", days=1).drop(columns=["cost_per_hour"])
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(Exception):
        parse_csv(str(csv_path))


# ---------- Waste detection ----------

def test_waste_detector_on_idle_heavy_data():
    df = generate_profile("batch", days=3)
    result = WasteDetector().analyze(df)

    assert set(["waste_analysis", "idle_periods", "recommendations"]).issubset(result.keys())
    assert 0 <= result["waste_analysis"]["total_waste_percentage"] <= 100
    assert isinstance(result["idle_periods"], list)
    assert len(result["idle_periods"]) > 0  # batch profile guarantees idle hours
    assert len(result["recommendations"]) > 0


def test_waste_detector_handles_empty_df():
    result = WasteDetector().analyze(pd.DataFrame())
    assert result["waste_analysis"]["total_waste_percentage"] == 0.0
    assert result["idle_periods"] == []


# ---------- Cost prediction (loads the real trained model) ----------

def test_cost_predictor_end_to_end():
    # needs >=72 hours of history (INPUT_LEN) - 5 days = 120 hours is enough
    df = generate_profile("webapp", days=5)
    model = CostPredictor()
    result = model.analyze(df)

    assert "cost_predictions" in result
    assert "performance_predictions" in result
    assert "optimization_opportunities" in result

    daily = result["cost_predictions"]["predicted_cost_next_7_days"]
    assert len(daily) == 7
    assert result["cost_predictions"]["trend"] in ("increasing", "decreasing", "stable")
    assert result["performance_predictions"]["bottleneck_risk"] in ("low", "medium", "high")