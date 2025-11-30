from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
import pandas as pd
import json

from data_preparation.parse_input import parse_csv, parse_json
from ml.waste_detector import WasteDetector

router = APIRouter(
    prefix="/detect-waste",
    tags=["waste_detection"]
)

JSON_EXAMPLE = (
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
)

@router.post(
    "",
    summary="Detect Idle/Waste Periods",
    description="Upload a CSV file **or** paste JSON to analyze waste, idle hours, and optimization opportunities."
)
async def detect_waste(
    json_data: Optional[str] = Form(
        default=None,
        title="JSON Input",
        description="Paste JSON time-series data. " + JSON_EXAMPLE,
        example=JSON_EXAMPLE
    ),
    file: UploadFile = File(
        default=None,
        title="CSV File Upload",
        description="Upload CSV with: timestamp, cpu_usage, memory_usage, disk_usage, cost_per_hour"
    )
):
    """
    Accept either:
    • JSON time-series input (as Form string)
    • CSV upload
    """

    # -------------------------
    # Handle CSV Upload
    # -------------------------
    if file:
        df = parse_csv(file.file)

    # -------------------------
    # Handle JSON Input
    # -------------------------
    elif json_data:
        try:
            parsed_json = json.loads(json_data)
        except Exception:
            return {"error": "Invalid JSON format. Please paste valid JSON."}

        df = parse_json(parsed_json)

    else:
        return {"error": "Provide either JSON input or a CSV file."}

    # -------------------------
    # Run Waste Detection
    # -------------------------
    detector = WasteDetector()
    result = detector.analyze(df)

    return result
