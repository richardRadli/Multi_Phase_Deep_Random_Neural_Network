import os
import sys
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.dataset_config import VALID_DATASETS
from services.mpdrnn_service.tasks import celery_app, mpdrnn_task

mpdrnn_router = APIRouter(prefix="/nn/mpdrnn", tags=["MPDRNN Unified Control"])

DatasetEnum = Enum("DatasetEnum", {ds.upper(): ds for ds in VALID_DATASETS}, type=str)

class ActivationEnum(str, Enum):
    ReLU = "ReLU"
    LeakyReLU = "LeakyReLU"
    Tanh = "Tanh"
    Sigmoid = "Sigmoid"
    Identity = "Identity"

class MethodEnum(str, Enum):
    BASE = "BASE"
    EXP_ORT = "EXP_ORT"
    EXP_ORT_C = "EXP_ORT_C"

class MPDRNNConfig(BaseModel):
    activation: ActivationEnum = Field(default=ActivationEnum.LeakyReLU, description="Aktivációs függvény")
    number_of_tests: int = Field(default=20, ge=1, description="Tesztek futtatási száma")
    seed: bool = Field(default=True, description="True esetén fixálja a random seedet")
    method: MethodEnum = Field(default=MethodEnum.BASE, description="Súlygenerálási módszer")
    sigma: float = Field(default=0.1, ge=0.01, le=1.0, description="Szórás (0.01 és 1.0 között)")

    penalty: Optional[float] = Field(default=None, description="Opcionális L2 penalty felülírás")
    rcond: Optional[float] = Field(default=None, description="Opcionális Moore-Penrose rcond felülírás")
    exp_neurons: Optional[List[int]] = Field(default=None, description="Opcionális exponenciális neurónszám lista")
    eq_neurons: Optional[List[int]] = Field(default=None, description="Opcionális egyenlő neurónszám lista")


@mpdrnn_router.post("/start", status_code=status.HTTP_202_ACCEPTED)
async def start_mpdrnn_process(
        config: MPDRNNConfig,
        dataset_name: DatasetEnum = Query(..., description="Válaszd ki az adathalmazt")
):
    try:
        config_payload = {
            "dataset_name": dataset_name.value,
            **config.model_dump()
        }
        task = mpdrnn_task.delay(config=config_payload)
        return {"task_id": task.id, "status": "QUEUED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mpdrnn_router.post("/stop/{task_id}")
async def stop_mpdrnn_process(task_id: str):
    try:
        celery_app.control.revoke(task_id=task_id, terminate=True, signal="SIGKILL")
        return {"status": "ABORT_SIGNAL_SENT", "message": f"MPDRNN task {task_id} revoked."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mpdrnn_router.get("/status/{task_id}")
async def get_mpdrnn_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    response = {"task_id": task_id, "status": task_result.state, "info": None}
    if task_result.state == "SUCCESS":
        response["info"] = task_result.get()
    elif task_result.state == "PROGRESS":
        response["info"] = task_result.info
    elif task_result.state == "FAILURE":
        response["info"] = str(task_result.info)
    return response