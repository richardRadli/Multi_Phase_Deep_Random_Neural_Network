import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from services.training_and_evaluation.tasks import test_fcnn_task

fcnn_test_router = APIRouter(prefix="/nn/fcnn", tags=["FCNN Model Evaluation"])

class FCNNTestConfig(BaseModel):
    dataset_name: str = Field(default="connect4", description="Name of the dataset to evaluate")
    batch_size: int = Field(default=128, description="Batch size for evaluation")
    seed: bool = Field(default=False)
    series_mode: bool = Field(default=False, description="True for multiple Excel series, False for single JSON report")


@fcnn_test_router.post("/test", status_code=status.HTTP_202_ACCEPTED)
async def test_fcnn_model(config: FCNNTestConfig):
    try:
        task = test_fcnn_task.delay(config=config.model_dump())
        logging.info(f"FCNN evaluation task queued with ID: {task.id}")
        return {
            "task_id": task.id,
            "status": "QUEUED"
        }
    except Exception as e:
        logging.error(f"Failed to push FCNN evaluation job to celery queue: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to push evaluation job to queue: {str(e)}"
        )