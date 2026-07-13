import os
import sys
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.dataset_config import VALID_DATASETS
from services.helm_service.tasks import celery_app, helm_task

helm_router = APIRouter(prefix="/nn/helm", tags=["HELM Unified Control"])

DatasetEnum = Enum("DatasetEnum", {ds.upper(): ds for ds in VALID_DATASETS}, type=str)


class HELMConfig(BaseModel):
    seed: bool = Field(default=False, description="True esetén fixálja a random seedet")
    num_tests: int = Field(default=1, ge=1, description="1 = egyszeri futás, >1 = tesztsorozat Excel mentéssel")

    penalty: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Opcionális felülírás: Pozitív lebegőpontos szám (pl. 1e-5 vagy 0.15). Ha üres, a dataset saját gyári értéke marad."
    )
    scaling_factor: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Opcionális felülírás: Skálázási tényező 0.0 és 1.0 között. Ha üres, a dataset saját gyári értéke marad."
    )


@helm_router.post("/start", status_code=status.HTTP_202_ACCEPTED)
async def start_helm_process(
        config: HELMConfig,
        dataset_name: DatasetEnum = Query(..., description="Válaszd ki az adathalmazt a listából")
):
    try:
        config_payload = {
            "dataset_name": dataset_name.value,
            **config.model_dump()
        }
        task = helm_task.delay(config=config_payload)
        return {"task_id": task.id, "status": "QUEUED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@helm_router.post("/stop/{task_id}")
async def stop_helm_process(task_id: str):
    try:
        celery_app.backend.client.set(f"helm:abort:{task_id}", "true")
        celery_app.control.revoke(task_id=task_id, terminate=False)
        return {"status": "ABORT_SIGNAL_SENT", "message": f"HELM task {task_id} signaled to abort gracefully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@helm_router.get("/status/{task_id}")
async def get_helm_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    response = {"task_id": task_id, "status": task_result.state, "info": None}
    if task_result.state == "SUCCESS":
        response["info"] = task_result.get()
    elif task_result.state == "PROGRESS":
        response["info"] = task_result.info
    elif task_result.state == "FAILURE":
        response["info"] = str(task_result.info)
    elif task_result.state == "REVOKED":
        response["status"] = "ABORTED"
    return response