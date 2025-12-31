from fastapi import FastAPI
from api.routers.detect_waste import router as waste_router
from api.routers.predict_cost import router as cost_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Infrastructure Cost Optimizer API",
    version="1.0",
    description="API for waste detection and cost prediction"
)
# ✅ ADD THIS BLOCK (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow requests from ANY host
    allow_credentials=True,
    allow_methods=["*"],          # allow all HTTP methods
    allow_headers=["*"],          # allow all headers
)
# Register routers
app.include_router(waste_router)
app.include_router(cost_router)


@app.get("/")
def home():
    return {"message": "Infrastructure Cost Optimizer API is running"}
