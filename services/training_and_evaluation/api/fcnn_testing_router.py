import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from enum import Enum
from config.dataset_config import VALID_DATASETS
from services.training_and_evaluation.tasks import test_fcnn_task

fcnn_test_router = APIRouter(prefix="/nn/fcnn", tags=["FCNN Model Evaluation"])

DatasetEnum = Enum("DatasetEnum", {ds.upper(): ds for ds in VALID_DATASETS})

class BatchSizeEnum(int, Enum):
    B16 = 16
    B32 = 32
    B64 = 64
    B128 = 128
    B256 = 256
    B512 = 512

class FCNNTestConfig(BaseModel):
    dataset_name: DatasetEnum = Field(description="A kiértékelendő dataset neve a gördülő listából")
    batch_size: BatchSizeEnum = Field(default=BatchSizeEnum.B128, description="Batch size a kiértékeléshez (csak a 2 hatványai engedélyezettek)")
    seed: bool = Field(default=False)
    series_mode: bool = Field(default=False, description="True több Excel sorozathoz, False egyetlen JSON jelentéshez")


@fcnn_test_router.post("/test", status_code=status.HTTP_202_ACCEPTED)
async def test_fcnn_model(config: FCNNTestConfig):
    try:
        task = test_fcnn_task.delay(config=config.model_dump(mode="json"))
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