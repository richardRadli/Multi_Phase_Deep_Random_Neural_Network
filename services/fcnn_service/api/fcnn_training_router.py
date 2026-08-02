import logging
import os
import sys
from enum import Enum
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.dataset_config import VALID_DATASETS
from services.fcnn_service.tasks import celery_app, train_fcnn_task

fcnn_router = APIRouter(prefix="/nn/fcnn", tags=["FCNN Control & Evaluation"])

DatasetEnum = Enum("DatasetEnum", {ds.upper(): ds for ds in VALID_DATASETS}, type=str)


class BatchSizeEnum(int, Enum):
    B16 = 16
    B32 = 32
    B64 = 64
    B128 = 128
    B256 = 256
    B512 = 512


class OptimizerEnum(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


class FCNNTrainingRemainingConfig(BaseModel):
    seed: bool = Field(default=False, description="True esetén fixálja a random seedet")
    epochs: int = Field(default=1000, ge=1, description="A tanítási epoch-ok száma")
    patience: int = Field(default=10, ge=1, description="Early stopping türelmi idő korlátja")
    optimizer: OptimizerEnum = Field(default=OptimizerEnum.ADAM, description="Választható optimizer (adam / sgd)")
    batch_size: Optional[BatchSizeEnum] = Field(
        default=None,
        description="Batch size felülbírálása (ha nincs megadva, a dataset alapértelmezett értéke lép életbe)"
    )


@fcnn_router.post("/train", status_code=status.HTTP_202_ACCEPTED)
async def start_fcnn_training(
        config: FCNNTrainingRemainingConfig,
        dataset_name: DatasetEnum = Query(..., description="Válaszd ki az adathalmazt a listából")
):
    try:
        config_payload = {
            "dataset_name": dataset_name.value,
            **config.model_dump(mode="json", exclude_none=True)
        }
        task = train_fcnn_task.delay(config=config_payload)
        celery_app.backend.client.sadd("fcnn:active_tasks", task.id)

        return {"task_id": task.id, "status": "QUEUED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fcnn_router.get("/active-tasks")
async def list_active_tasks():
    try:
        active_ids = celery_app.backend.client.smembers("fcnn:active_tasks")
        task_ids = [tid.decode("utf-8") for tid in active_ids]
        return {"active_task_ids": task_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fcnn_router.post("/stop/{task_id}")
async def stop_fcnn_training(task_id: str):
    try:
        celery_app.backend.client.set(f"fcnn:abort:{task_id}", "true")
        celery_app.backend.client.srem("fcnn:active_tasks", task_id)

        logging.info(f"FCNN graceful abort signal set for task: {task_id}")

        return {
            "status": "ABORT_SIGNAL_SENT",
            "message": f"Task {task_id} successfully signaled to abort gracefully via Redis flag."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fcnn_router.get("/status/{task_id}")
async def get_fcnn_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)

    try:
        task_state = task_result.state
        task_info = task_result.info
    except Exception:
        return {
            "task_id": task_id,
            "status": "ABORTED",
            "info": {"status": "ABORTED", "message": "Task was manually aborted."}
        }

    response = {"task_id": task_id, "status": task_state, "info": None}

    if task_state == "PROGRESS":
        response["info"] = task_info
    elif task_state == "SUCCESS":
        response["info"] = task_result.get()
    elif task_state == "FAILURE":
        response["info"] = str(task_info)
    elif task_state == "PENDING":
        response["info"] = {"status": "Waiting in queue..."}
    elif task_state in ["ABORTED", "REVOKED"]:
        response["status"] = "ABORTED"
        if task_info and isinstance(task_info, dict):
            response["info"] = task_info
        else:
            response["info"] = {"status": "ABORTED", "message": "Task was manually aborted."}

    return response