from fastapi import FastAPI
from services.helm_service.api.helm_router import helm_router

app = FastAPI(
    title="FCNN Training and Evaluation Service",
    description="Microservice for FCNN operations via Celery and Redis",
    version="1.0.0"
)

app.include_router(helm_router)
@app.get("/", tags=["Health Check"])
async def root():
    return {"service": "FCNN Service", "status": "healthy"}