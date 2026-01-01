# ⚙️ Cloud Cost Optimizer – Backend API

A FastAPI-based backend service that analyzes cloud infrastructure usage data and provides **AI-powered cost predictions** and **waste detection insights**.  
The API accepts **CSV uploads** and returns actionable optimization recommendations.

---

## 🚀 Live API

- **Base URL:**  
  👉 https://cloudcost-optimizer-api.onrender.com/docs




📌 Features

📈 7-Day Cost Prediction

🧠 Performance & Bottleneck Forecasting

♻️ Waste Detection & Idle Period Analysis

📂 CSV-Based Input (simple & user-friendly)

🔐 CORS enabled for frontend integration

☁️ Deployed on Render (Free Tier)





🧩 API Endpoints
1️⃣ Predict Cost

Endpoint

POST /predict-cost


Description

Predicts cloud cost for the next 7 days

Analyzes CPU & memory usage trends

Suggests optimization opportunities

Request Type

multipart/form-data

Parameters

Name	Type	Required
file	CSV file	✅




2️⃣ Detect Waste

Endpoint

POST /detect-waste


Description

Detects underutilized resources

Identifies idle periods

Estimates potential monthly savings

Request Type

multipart/form-data

Parameters

Name	Type	Required
file	CSV file	✅
📂 CSV Input Format

The API expects a CSV file with the following columns:

timestamp,cpu_usage,memory_usage,disk_usage,cost_per_hour

📌 Column Description
Column	Description
timestamp	Datetime of the record
cpu_usage	CPU usage (%)
memory_usage	Memory usage (%)
disk_usage	Disk usage (%)
cost_per_hour	Cost per hour ($)




🧪 Sample CSV File

A ready-to-use sample CSV file is included in this repository:

sample_generated.csv


👉 How to use:

Open the file in GitHub

Click Download

Upload it directly to the API endpoints




🧪 Example cURL Requests
Predict Cost
curl -X POST https://cloudcost-optimizer-api.onrender.com/predict-cost \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_generated.csv"

Detect Waste
curl -X POST https://cloudcost-optimizer-api.onrender.com/detect-waste \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_generated.csv"




🛠️ Tech Stack

FastAPI

Uvicorn

Pandas & NumPy

Scikit-learn

LightGBM

Docker

Render (Deployment)





⚠️ Note on Free Tier Hosting

This API is hosted on Render Free Tier:

The service may sleep after inactivity

First request may take 30–60 seconds to respond (cold start)

Subsequent requests are fast



🔗 Related Repositories

Frontend (Vercel):
👉 https://cloud-cost-optimiser-lemon.vercel.app/