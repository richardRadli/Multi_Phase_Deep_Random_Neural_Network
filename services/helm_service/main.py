from fastapi import FastAPI
from services.helm_service.api.helm_router import helm_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="FCNN Training and Evaluation Service",
    description="Microservice for FCNN operations via Celery and Redis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(helm_router)
@app.get("/", tags=["Health Check"])
async def root():
    return {"service": "FCNN Service", "status": "healthy"}