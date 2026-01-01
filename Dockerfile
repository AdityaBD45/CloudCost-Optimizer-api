# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for pandas, numpy, lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose API port
EXPOSE 10000

# Launch FastAPI using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
