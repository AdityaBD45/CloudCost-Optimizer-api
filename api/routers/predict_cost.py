from fastapi import APIRouter, UploadFile, File, Form,Body
from typing import Optional
import pandas as pd
import pickle
import json
from pathlib import Path

from data_preparation.parse_input import parse_csv, parse_json

# Correct model path
MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "cost_predictor.pkl"

router = APIRouter(
    prefix="/predict-cost",
    tags=["cost_prediction"]
)

# Load model once globally at startup
with open(MODEL_PATH, "rb") as f:
    cost_model = pickle.load(f)


@router.post("")
async def predict_cost(
json_data: Optional[str] = Form(
        default=None,
        description="Paste JSON time-series input. Example:\n"
                    "{\n"
                    "  \"data\": [\n"
                    "    {\n"
                    "      \"timestamp\": \"2024-01-01 00:00:00\",\n"
                    "      \"cpu_usage\": 23.1,\n"
                    "      \"memory_usage\": 51.2,\n"
                    "      \"disk_usage\": 60.4,\n"
                    "      \"cost_per_hour\": 0.215\n"
                    "    }\n"
                    "  ]\n"
                    "}"
    ),
    file: UploadFile = File(
        default=None,
        description="Upload a CSV file with columns: timestamp, cpu_usage, memory_usage, disk_usage, cost_per_hour"
    )
):
    """
    Accept either JSON or CSV and run 7-day cost prediction.
    """

    # ---- Input Handling ----
    if file:
        df = parse_csv(file.file)

    elif json_data:
        # convert string → dict
        try:
            json_payload = json.loads(json_data)
        except Exception:
            return {"error": "Invalid JSON format. Expected JSON string."}

        df = parse_json(json_payload)

    else:
        return {"error": "Provide either JSON or CSV file."}

    # ---- Run Prediction ----
    try:
        result = cost_model.analyze(df)

        # unpack tuple (from analyze)
        if isinstance(result, tuple):
            result = result[0]

        return result
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}