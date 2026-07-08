from fastapi import FastAPI
from services.training_and_evaluation.api.fcnn_training_router import fcnn_router as fcnn_train_router
from services.training_and_evaluation.api.fcnn_testing_router import fcnn_test_router
from services.training_and_evaluation.api.helm_router import helm_router

app = FastAPI(
    title="FCNN Training and Evaluation Service",
    description="Microservice for FCNN operations via Celery and Redis",
    version="1.0.0"
)

app.include_router(fcnn_train_router)
app.include_router(fcnn_test_router)
app.include_router(helm_router)
@app.get("/", tags=["Health Check"])
async def root():
    return {"service": "FCNN Service", "status": "healthy"}