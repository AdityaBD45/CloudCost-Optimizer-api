# Cloud Cost Optimizer — Backend API

A FastAPI service that accepts cloud usage CSV files and returns 7-day cost forecasts, utilization predictions, waste detection, and optimization recommendations — all powered by a trained LSTM neural network.

**Live API →** https://cloudcost-optimizer-api.onrender.com/docs

> The backend is on Render Free Tier. The first request after a period of inactivity may take 30–60 seconds (cold start). Subsequent requests are fast.

---

## Table of Contents

- [API Endpoints](#api-endpoints)
- [CSV Input Format](#csv-input-format)
- [ML Model Architecture](#ml-model-architecture)
- [Model Accuracy](#model-accuracy)
- [Project Structure](#project-structure)
- [Local Development](#local-development)
- [Training the Model](#training-the-model)
- [Tech Stack](#tech-stack)

---

## API Endpoints

### `POST /predict-cost`

Runs the LSTM model on uploaded usage data and returns a 7-day hourly forecast aggregated to daily totals.

**Request** — `multipart/form-data`

| Parameter | Type     | Required |
|-----------|----------|----------|
| `file`    | CSV file | ✅        |

**Response**

```json
{
  "cost_predictions": {
    "predicted_cost_next_7_days": [
      {
        "date": "2024-02-05",
        "weekday": "Monday",
        "cost": 6.55,
        "cost_range": [5.60, 7.60],
        "avg_cpu": 42.7,
        "avg_memory": 51.6,
        "confidence": 0.69
      }
    ],
    "total_predicted_weekly_cost": 41.12,
    "trend": "increasing",
    "trend_strength": 0.0288
  },
  "performance_predictions": {
    "expected_peak_cpu": 85.97,
    "expected_peak_memory": 92.7,
    "peak_time_prediction": "2024-01-25 15:00:00",
    "bottleneck_risk": "high"
  },
  "optimization_opportunities": [
    {
      "type": "scheduling",
      "savings_potential": 8.22,
      "confidence": 0.87
    }
  ]
}
```

**Key fields:**
- `cost_range` — real P10/P90 confidence interval from the model, not a hardcoded number
- `confidence` — derived from how tight the P10–P90 band is relative to the median forecast
- `trend` — `"increasing"` / `"decreasing"` / `"stable"` based on daily pct change

---

### `POST /detect-waste`

Rule-based analysis of the uploaded window — detects idle periods (CPU < 20% AND memory < 30%) and estimates monthly savings.

**Request** — `multipart/form-data`

| Parameter | Type     | Required |
|-----------|----------|----------|
| `file`    | CSV file | ✅        |

**Response**

```json
{
  "waste_analysis": {
    "underutilized_score": 0.28,
    "total_waste_percentage": 27.54,
    "estimated_monthly_savings": 39.34
  },
  "idle_periods": [
    {
      "start": "2024-01-06 00:00:00",
      "end": "2024-01-06 09:00:00",
      "duration_hours": 9.0,
      "avg_cpu": 9.34,
      "avg_memory": 23.31,
      "wasted_cost": 1.59
    }
  ],
  "recommendations": [
    "Schedule automatic shutdown during long idle periods."
  ]
}
```

---

### Example cURL Requests

```bash
# Predict cost
curl -X POST https://cloudcost-optimizer-api.onrender.com/predict-cost \
  -F "file=@sample_generated.csv"

# Detect waste
curl -X POST https://cloudcost-optimizer-api.onrender.com/detect-waste \
  -F "file=@sample_generated.csv"
```

---

## CSV Input Format

```
timestamp,cpu_usage,memory_usage,disk_usage,cost_per_hour
2024-01-01 00:00:00,15.3,42.1,38.7,0.18
2024-01-01 01:00:00,12.8,40.5,38.9,0.17
...
```

| Column          | Type     | Description                        |
|-----------------|----------|------------------------------------|
| `timestamp`     | datetime | Hourly timestamp                   |
| `cpu_usage`     | float    | CPU utilization (0–100%)           |
| `memory_usage`  | float    | Memory utilization (0–100%)        |
| `disk_usage`    | float    | Disk utilization (0–100%)          |
| `cost_per_hour` | float    | Actual cost in $ for that hour     |

A `sample_generated.csv` is included in the repo — upload it directly via the web app or API to test instantly.

> The model requires at least 72 rows (3 days of hourly data) to produce a forecast. If fewer rows are provided, they are repeated to fill the 72-hour input window — the API will still return a forecast rather than an error.

---

## ML Model Architecture

The forecasting model is a 2-layer LSTM implemented in TensorFlow/Keras.

### Why LSTM?

The previous version used 4 separate LightGBM regressors in a **recursive loop** — predicting one hour at a time and feeding each prediction back as the next input. Over 168 steps (7 days), this compounds errors significantly. The LSTM replaces that with a **single forward pass** that outputs the entire 7-day forecast at once.

### Input

- **Window size:** 72 timesteps (last 72 hours of history)
- **Features per timestep (9 total):**

| Feature       | Description                                   |
|---------------|-----------------------------------------------|
| `cpu_usage`   | CPU utilization %                             |
| `memory_usage`| Memory utilization %                          |
| `disk_usage`  | Disk utilization %                            |
| `cost_per_hour` | Actual hourly cost                          |
| `hour_sin`    | sin(2π × hour / 24) — cyclical hour encoding |
| `hour_cos`    | cos(2π × hour / 24) — cyclical hour encoding |
| `dow_sin`     | sin(2π × day_of_week / 7) — day encoding     |
| `dow_cos`     | cos(2π × day_of_week / 7) — day encoding     |
| `is_weekend`  | 1 if Saturday/Sunday, else 0                  |

Hour and day-of-week are encoded as sin/cos pairs rather than raw integers so the model sees that hour 23 and hour 0 are adjacent (cyclical), not 23 apart.

All features are normalized with a mean/std scaler fit on training data only. Scaler parameters are saved to `ml/scaler.json` and loaded at inference time.

### Architecture

```
Input: (batch, 72, 9)
         │
    ┌────▼────┐
    │  LSTM   │  hidden=64, dropout=0.1, return_sequences=True
    └────┬────┘
         │
    ┌────▼────┐
    │  LSTM   │  hidden=64, dropout=0.1, return_sequences=False
    └────┬────┘
         │
    final hidden state (batch, 64)
         │
   ┌─────┴──────┐
   │            │
┌──▼──┐      ┌──▼──┐
│Dense│      │Dense│    128 units, ReLU
│ 128 │      │ 128 │
└──┬──┘      └──┬──┘
   │            │
┌──▼──┐      ┌──▼──┐
│Dense│      │Dense│    output projections
│168×3│      │168×3│
└──┬──┘      └──┬──┘
   │            │
cost_out     util_out
(168, 3)     (168, 3)
P10/P50/P90  cpu/mem/disk
```

**Two output heads:**

| Head | Shape | Purpose |
|------|-------|---------|
| `cost_out` | `(168, 3)` | P10, P50, P90 quantile predictions for each of the next 168 hours |
| `util_out` | `(168, 3)` | Point estimates for cpu, memory, disk utilization per hour |

### Loss Functions

- **`cost_out`** is trained with **pinball (quantile) loss** at τ = 0.1, 0.5, 0.9. This forces the three outputs to be calibrated percentiles rather than three copies of the same estimate.
- **`util_out`** is trained with **MSE**.
- Both targets are normalized to comparable scale before computing loss (cost ≈ $0.08–0.85, CPU ≈ 0–100 — a ~250× scale gap would otherwise let utilization error dominate the gradient and prevent the model from learning cost accurately).

### Training Data

The model is trained on **synthetic data** generated by `data_preparation/generate_data.py`. Three distinct workload profiles are mixed together:

| Profile    | Pattern                                                      |
|------------|--------------------------------------------------------------|
| `webapp`   | Weekday 9–5 peaks, weekend troughs, occasional anomaly spikes |
| `database` | Flat/always-on, mild daytime bump, no strong daily cycle     |
| `batch`    | Near-idle all day with sharp nightly spikes (1–3am)          |

Each profile runs for 150 days with:
- A slow upward trend (15% usage growth over the period)
- Heteroscedastic noise (noise scales with load, not fixed size)
- Burst-pricing kink in cost above 80% CPU
- Random anomaly spikes (~0.4% of rows)

Data is split **chronologically** (no shuffling):

| Split | Fraction | Rows   | Purpose            |
|-------|----------|--------|--------------------|
| Train | 70%      | 7,560  | Model training     |
| Val   | 15%      | 1,620  | Early stopping     |
| Test  | 15%      | 1,620  | Final evaluation   |

Shuffling a time series would leak future data into training — the chronological split prevents this.

### Training Details

| Hyperparameter   | Value           |
|------------------|-----------------|
| Optimizer        | Adam (lr=1e-3)  |
| Batch size       | 32              |
| Max epochs       | 80              |
| Early stopping   | patience=8 on val_loss |
| Input window     | 72 hours        |
| Output window    | 168 hours       |
| Stride (train)   | 6 hours         |
| Stride (eval)    | 24 hours        |

---

## Model Accuracy

Evaluated on **39 independent held-out test windows** (273 day-ahead forecasts across 3 workload profiles). The model scores on data it never saw during training:

| Metric               | Value  |
|----------------------|--------|
| Cost MAE ($/day)     | $0.30  |
| Cost RMSE ($/day)    | $0.34  |
| Cost MAPE            | 5.1%   |
| **Cost R²**          | **0.92** |
| CPU MAE              | 2.3%   |
| CPU MAPE             | 6.9%   |
| **CPU R²**           | **0.95** |
| Memory MAE           | 2.7%   |
| Memory MAPE          | 6.1%   |
| **Memory R²**        | **0.97** |

Raw metrics are saved to `ml/metrics.json` and can be reproduced at any time by running `python ml/evaluate_models.py`.

---

## Project Structure

```
CloudCost-Optimizer-api/
│
├── api/
│   ├── main.py                   # FastAPI app, CORS, router registration
│   └── routers/
│       ├── predict_cost.py       # POST /predict-cost
│       └── detect_waste.py       # POST /detect-waste
│
├── data_preparation/
│   ├── generate_data.py          # Synthetic data generator (3 profiles, 150 days each)
│   └── parse_input.py            # CSV/JSON parsing and validation
│
├── ml/
│   ├── lstm_forecaster.py        # Model definition, Scaler, windowing utilities
│   ├── train_lstm.py             # Training script → produces lstm_forecaster.keras
│   ├── evaluate_models.py        # Evaluation → produces metrics.json
│   ├── cost_predictor.py         # Inference wrapper used by the API router
│   ├── waste_detector.py         # Rule-based idle period detection
│   ├── lstm_forecaster.keras     # Trained model weights
│   ├── scaler.json               # Feature + target scaler parameters
│   └── metrics.json              # Test-set evaluation results
│
├── sample_generated.csv          # Ready-to-use sample CSV for testing
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

## Local Development

```bash
# Clone the repo
git clone https://github.com/AdityaBD45/CloudCost-Optimizer-api.git
cd CloudCost-Optimizer-api

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn api.main:app --reload
```

API docs available at http://localhost:8000/docs

---

## Training the Model

If you want to retrain the model from scratch (e.g. with different hyperparameters or more data):

```bash
# Step 1 — train (produces ml/lstm_forecaster.keras and ml/scaler.json)
python ml/train_lstm.py

# Step 2 — evaluate (produces ml/metrics.json)
python ml/evaluate_models.py
```

Training takes roughly 2–5 minutes on a standard laptop CPU. The synthetic data generator runs automatically as part of the training script — no separate data download needed.

---

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| API         | FastAPI, Uvicorn                    |
| ML Model    | TensorFlow / Keras (LSTM)           |
| Data        | Pandas, NumPy                       |
| Evaluation  | scikit-learn                        |
| Deployment  | Docker, Render                      |

---

## Related

- **Frontend repo →** https://github.com/AdityaBD45/Cloud-Cost-Optimiser
- **Live app →** https://cloud-cost-optimiser-lemon.vercel.app/