from fastapi import FastAPI
from services.mpdrnn_service.api.mpdrnn_router import mpdrnn_router

app = FastAPI(
    title="MPDRNN Service",
    description="Dedicated Microservice for Multi-Phase Deep Randomized Neural Networks",
    version="1.0.0"
)

app.include_router(mpdrnn_router)

@app.get("/", tags=["Health Check"])
async def root():
    return {"service": "MPDRNN Service", "status": "healthy"}