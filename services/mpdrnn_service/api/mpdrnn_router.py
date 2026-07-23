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
    number_of_tests: int = Field(default=20, ge=1, description="Tesztek futtatási száma")
    seed: bool = Field(default=True, description="True esetén fixálja a random seedet")

    sigma: Optional[float] = Field(default=0.1, ge=0.01, le=1.0,
                                   description="Szórás (BASE esetén figyelmen kívül hagyva)")
    penalty: Optional[float] = Field(default=None,
                                     description="L2 penalty felülírás (Kizárólag EXP_ORT_C-nél megengedett)")

    num_of_layers: Optional[int] = Field(default=3, ge=1, description="Rétegek száma")
    num_of_neurons: Optional[int] = Field(default=100, ge=1, description="Összes neuronszám")
    decay_rate: Optional[float] = Field(default=0.5, ge=0.0, description="Lecsengési ráta")

    hidden_neurons: Optional[List[int]] = Field(default=None,
                                                description="Manuálisan megadott rétegenkénti neuronszámok [L1, L2, L3]")

    rcond: Optional[float] = Field(default=None, description="Opcionális Moore-Penrose rcond felülírás")


@mpdrnn_router.post("/start", status_code=status.HTTP_202_ACCEPTED)
async def start_mpdrnn_process(
        config: MPDRNNConfig,
        dataset_name: DatasetEnum = Query(..., description="Válaszd ki az adathalmazt"),
        method: MethodEnum = Query(MethodEnum.BASE, description="Válaszd ki a súlygenerálási módszert"),
        activation: ActivationEnum = Query(ActivationEnum.LeakyReLU, description="Válaszd ki az aktivációs függvényt")
):
    try:
        if method != MethodEnum.EXP_ORT_C and config.penalty is not None:
            raise HTTPException(
                status_code=400,
                detail="A 'penalty' paraméter kizárólag a 'EXP_ORT_C' metódus esetén használható!"
            )

        config_payload = {
            "dataset_name": dataset_name.value,
            "method": method.value,
            "activation": activation.value,
            **config.model_dump()
        }
        task = mpdrnn_task.delay(config=config_payload)
        return {"task_id": task.id, "status": "QUEUED"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mpdrnn_router.post("/stop/{task_id}")
def stop_mpdrnn_task(task_id: str):
    celery_app.backend.client.set(f"mpdrnn:abort:{task_id}", 1, ex=3600)
    celery_app.control.revoke(task_id, terminate=False)
    return {
        "status": "ABORT_SIGNAL_SENT",
        "message": f"MPDRNN task {task_id} abort signal sent to Redis."
    }

@mpdrnn_router.get("/status/{task_id}")
def get_mpdrnn_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    response = {"task_id": task_id, "status": task_result.state, "info": None}
    if task_result.state == "SUCCESS":
        response["info"] = task_result.get()
    elif task_result.state == "PROGRESS":
        response["info"] = task_result.info
    elif task_result.state == "FAILURE":
        response["info"] = str(task_result.info)
    return response

@mpdrnn_router.get("/datasets", tags=["Config"])
def get_datasets():
    return {"datasets": VALID_DATASETS}