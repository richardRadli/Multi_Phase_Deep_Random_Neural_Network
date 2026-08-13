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
from services.dev_drnn_service.tasks import celery_app, mpdrnn_task, mpdrnn_tune_task

dev_drnn_router = APIRouter(prefix="/nn/dev_drnn", tags=["DevDRNN Unified Control"])

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


class SearchBackendEnum(str, Enum):
    OPTUNA = "optuna"
    RAY = "ray"


class DevDRNNConfig(BaseModel):
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


class DevDRNNTuneConfig(BaseModel):
    backend: SearchBackendEnum = Field(default=SearchBackendEnum.OPTUNA, description="Keresési backend (optuna vagy ray)")
    n_trials: int = Field(default=25, ge=1, le=100, description="Próbálkozások száma")
    seed: bool = Field(default=False, description="Fixálja a véletlenszám-generálást")

    method: MethodEnum = Field(default=MethodEnum.BASE, description="Súlygenerálási módszer a kísérlethez")
    activation: ActivationEnum = Field(default=ActivationEnum.LeakyReLU, description="Aktivációs függvény a kísérlethez")

    rcond_min: float = Field(default=1e-30, description="Minimum rcond érték")
    rcond_max: float = Field(default=1e-1, description="Maximum rcond érték")
    penalty_min: float = Field(default=0.1, description="Minimum penalty_term érték")
    penalty_max: float = Field(default=30.0, description="Maximum penalty_term érték")

    # Rétegenkénti neuronszám keresési határok
    l1_min: int = Field(default=600, ge=1, description="Layer 1 minimum neuronszám")
    l1_max: int = Field(default=1000, ge=1, description="Layer 1 maximum neuronszám")
    l2_min: int = Field(default=200, ge=1, description="Layer 2 minimum neuronszám")
    l2_max: int = Field(default=500, ge=1, description="Layer 2 maximum neuronszám")
    l3_min: int = Field(default=50, ge=1, description="Layer 3 minimum neuronszám")
    l3_max: int = Field(default=100, ge=1, description="Layer 3 maximum neuronszám")


@dev_drnn_router.post("/start", status_code=status.HTTP_202_ACCEPTED)
async def start_dev_drnn_process(
        config: DevDRNNConfig,
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


@dev_drnn_router.post("/tune", status_code=status.HTTP_202_ACCEPTED)
async def start_dev_drnn_tuning(
        config: DevDRNNTuneConfig,
        dataset_name: DatasetEnum = Query(..., description="Válaszd ki az adathalmazt")
):
    try:
        config_payload = {
            "dataset_name": dataset_name.value,
            **config.model_dump()
        }
        task = mpdrnn_tune_task.delay(config=config_payload)
        return {"task_id": task.id, "status": "QUEUED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@dev_drnn_router.post("/stop/{task_id}")
def stop_dev_drnn_task(task_id: str):
    celery_app.backend.client.set(f"mpdrnn:abort:{task_id}", 1, ex=3600)
    celery_app.control.revoke(task_id, terminate=False)
    return {
        "status": "ABORT_SIGNAL_SENT",
        "message": f"DevDRNN task {task_id} abort signal sent to Redis."
    }


@dev_drnn_router.post("/tune/stop/{task_id}")
def stop_dev_drnn_tune_task(task_id: str):
    celery_app.backend.client.set(f"mpdrnn:abort:{task_id}", 1, ex=3600)
    celery_app.control.revoke(task_id, terminate=False)
    return {
        "status": "ABORT_SIGNAL_SENT",
        "message": f"DevDRNN tune task {task_id} abort signal sent to Redis."
    }


@dev_drnn_router.get("/status/{task_id}")
def get_dev_drnn_status(task_id: str):
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


@dev_drnn_router.get("/tune/status/{task_id}")
def get_dev_drnn_tune_status(task_id: str):
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


@dev_drnn_router.get("/datasets", tags=["Config"])
def get_datasets():
    return {"datasets": VALID_DATASETS}