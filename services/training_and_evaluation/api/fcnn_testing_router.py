import os
import sys
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from enum import Enum

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.dataset_config import VALID_DATASETS
from services.training_and_evaluation.tasks import celery_app, test_fcnn_task

fcnn_test_router = APIRouter(prefix="/nn/fcnn", tags=["FCNN Model Evaluation"])

DatasetEnum = Enum("DatasetEnum", {ds.upper(): ds for ds in VALID_DATASETS}, type=str)

class BatchSizeEnum(int, Enum):
    B16 = 16
    B32 = 32
    B64 = 64
    B128 = 128
    B256 = 256
    B512 = 512

class FCNNTestRemainingConfig(BaseModel):
    seed: bool = Field(default=False, description="True esetén fixálja a random seedet")
    series_mode: bool = Field(default=False, description="True több Excel sorozathoz, False egyetlen JSON jelentéshez")
    num_tests: int = Field(default=20, ge=1, description="A sorozatban futtatandó tesztek száma")
    epochs: int = Field(default=1000, ge=1, description="Hány epoch fusson ciklusonként?")

@fcnn_test_router.post("/test", status_code=status.HTTP_202_ACCEPTED)
async def test_fcnn_model(
    config: FCNNTestRemainingConfig,
    dataset_name: DatasetEnum = Query(..., description="A kiértékelendő dataset neve"),
    batch_size: BatchSizeEnum = Query(BatchSizeEnum.B128, description="Batch size (Kizárólag a 2 hatványai választhatóak)")
):
    try:
        config_payload = {
            "dataset_name": dataset_name.value,
            "batch_size": batch_size.value,
            **config.model_dump()
        }
        task = test_fcnn_task.delay(config=config_payload)
        return {"task_id": task.id, "status": "QUEUED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fcnn_test_router.post("/test/stop/{task_id}")
async def stop_fcnn_testing(task_id: str):
    try:
        celery_app.backend.client.set(f"fcnn:abort:{task_id}", "true")
        celery_app.control.revoke(task_id=task_id, terminate=True, signal="SIGKILL")
        return {
            "status": "ABORT_SIGNAL_SENT",
            "message": f"FCNN Testing task {task_id} successfully signaled to abort."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fcnn_test_router.get("/test/status/{task_id}")
async def get_fcnn_test_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    response = {"task_id": task_id, "status": task_result.state, "info": None}

    if task_result.state == "SUCCESS":
        response["info"] = task_result.get()
    elif task_result.state == "FAILURE":
        response["info"] = str(task_result.info)
    elif task_result.state == "PENDING":
        response["info"] = {"status": "Waiting in queue..."}
    elif task_result.state == "PROGRESS":
        response["info"] = task_result.info
    elif task_result.state == "REVOKED":
        response["status"] = "ABORTED"
        response["info"] = {"status": "FCNN Evaluation task was manually aborted."}

    return response