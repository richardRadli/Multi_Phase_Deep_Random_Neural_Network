import os
import sys
from enum import Enum
from typing import Any, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, Field

PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.dataset_config import VALID_DATASETS
from services.dev_drnn_aux_service.tasks import (
    celery_app,
    ipmpdrnn_task,
    ipmpdrnn_tune_task,
)

dev_drnn_aux_router = APIRouter(
    prefix="/nn/dev_drnn_aux",
    tags=["DevDRNN-Aux Unified Control"],
)


class ActivationEnum(str, Enum):
    RELU = "ReLU"
    LEAKY_RELU = "LeakyReLU"
    TANH = "Tanh"
    SIGMOID = "Sigmoid"
    IDENTITY = "Identity"


class MethodEnum(str, Enum):
    BASE = "BASE"
    EXP_ORT = "EXP_ORT"
    EXP_ORT_C = "EXP_ORT_C"


class SearchBackendEnum(str, Enum):
    OPTUNA = "optuna"
    RAY = "ray"


class DevDRNNAuxConfig(BaseModel):
    number_of_tests: int = Field(
        default=20, ge=1, description="Tesztek futtatási száma"
    )
    seed: bool = Field(
        default=True, description="True esetén rögzíti a random seedet"
    )
    subset_percentage: float = Field(
        default=0.1,
        ge=0.01,
        le=0.99,
        description="Metszési arány (Subset Percentage)",
    )
    num_aux_net: int = Field(
        default=1, ge=1, le=10, description="Segédhálózatok száma (Num Aux Nets)"
    )
    sigma: Optional[float] = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Szórás (BASE esetén figyelmen kívül hagyva)",
    )
    penalty: Optional[float] = Field(
        default=None,
        description="L2 penalty felülírás (Kizárólag EXP_ORT_C-nél engedélyezett)",
    )
    num_of_layers: Optional[int] = Field(
        default=3, ge=1, description="Rétegek száma"
    )
    num_of_neurons: Optional[int] = Field(
        default=100, ge=1, description="Összes neuronszám"
    )
    decay_rate: Optional[float] = Field(
        default=0.5, ge=0.0, description="Lecsengési ráta"
    )
    hidden_neurons: Optional[List[int]] = Field(
        default=None,
        description="Manuálisan megadott rétegenkénti neuronszámok [L1, L2, L3]",
    )
    rcond: Optional[float] = Field(
        default=None, description="Opcionális Moore-Penrose rcond feltétel"
    )


class DevDRNNAuxTuneConfig(BaseModel):
    backend: SearchBackendEnum = Field(
        default=SearchBackendEnum.OPTUNA,
        description="Keresési backend (optuna vagy ray)",
    )
    n_trials: int = Field(
        default=25, ge=1, le=100, description="Próbálkozások száma"
    )
    seed: bool = Field(
        default=False, description="Rögzíti a véletlenszám-generálást"
    )
    method: MethodEnum = Field(
        default=MethodEnum.BASE,
        description="Súlygenerálási módszer a kísérlethez",
    )
    activation: ActivationEnum = Field(
        default=ActivationEnum.LEAKY_RELU,
        description="Aktivációs függvény a kísérlethez",
    )
    rcond_min: float = Field(default=1e-30, description="Minimum rcond érték")
    rcond_max: float = Field(default=1e-1, description="Maximum rcond érték")
    penalty_min: float = Field(
        default=0.1, description="Minimum penalty_term érték"
    )
    penalty_max: float = Field(
        default=30.0, description="Maximum penalty_term érték"
    )
    sp_min: float = Field(
        default=0.1, ge=0.01, le=0.99, description="Subset Percentage minimum"
    )
    sp_max: float = Field(
        default=0.9, ge=0.01, le=0.99, description="Subset Percentage maximum"
    )
    num_aux_min: int = Field(default=1, ge=1, description="Num Aux Net minimum")
    num_aux_max: int = Field(default=5, ge=1, description="Num Aux Net maximum")
    l1_min: int = Field(
        default=600, ge=1, description="Layer 1 minimum neuronszám"
    )
    l1_max: int = Field(
        default=1000, ge=1, description="Layer 1 maximum neuronszám"
    )
    l2_min: int = Field(
        default=200, ge=1, description="Layer 2 minimum neuronszám"
    )
    l2_max: int = Field(
        default=500, ge=1, description="Layer 2 maximum neuronszám"
    )
    l3_min: int = Field(
        default=50, ge=1, description="Layer 3 minimum neuronszám"
    )
    l3_max: int = Field(
        default=100, ge=1, description="Layer 3 maximum neuronszám"
    )


def validate_and_clean_dataset_name(dataset_name: str) -> str:
    """Levágja a felesleges szóközöket és ellenőrzi az adathalmaz létezését."""
    cleaned = dataset_name.strip()
    if cleaned not in VALID_DATASETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Érvénytelen adathalmaz: '{cleaned}'. Választható adathalmazok: {VALID_DATASETS}",
        )
    return cleaned


@dev_drnn_aux_router.post("/start", status_code=status.HTTP_202_ACCEPTED)
async def start_dev_drnn_aux_process(
    config: DevDRNNAuxConfig = Body(default_factory=DevDRNNAuxConfig),
    dataset_name: str = Query(
        ...,
        description="Válaszd ki az adathalmazt",
        examples=["connect4", "mnist"],
    ),
    method: MethodEnum = Query(
        default=MethodEnum.BASE,
        description="Válaszd ki a súlygenerálási módszert",
    ),
    activation: ActivationEnum = Query(
        default=ActivationEnum.LEAKY_RELU,
        description="Válaszd ki az aktivációs függvényt",
    ),
):
    try:
        clean_dataset = validate_and_clean_dataset_name(dataset_name)

        if method != MethodEnum.EXP_ORT_C and config.penalty is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A 'penalty' paraméter kizárólag az 'EXP_ORT_C' metódus esetén használható!",
            )

        config_payload = {
            "dataset_name": clean_dataset,
            "method": method.value,
            "activation": activation.value,
            **config.model_dump(),
        }
        task = ipmpdrnn_task.delay(config=config_payload)
        return {"task_id": task.id, "status": "QUEUED"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@dev_drnn_aux_router.post("/tune", status_code=status.HTTP_202_ACCEPTED)
async def start_dev_drnn_aux_tuning(
    config: DevDRNNAuxTuneConfig = Body(default_factory=DevDRNNAuxTuneConfig),
    dataset_name: str = Query(
        ...,
        description="Válaszd ki az adathalmazt",
        examples=["connect4", "mnist"],
    ),
):
    try:
        clean_dataset = validate_and_clean_dataset_name(dataset_name)
        config_payload = {
            "dataset_name": clean_dataset,
            **config.model_dump(),
        }
        task = ipmpdrnn_tune_task.delay(config=config_payload)
        return {"task_id": task.id, "status": "QUEUED"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@dev_drnn_aux_router.post("/stop/{task_id}")
def stop_dev_drnn_aux_task(task_id: str):
    celery_app.backend.client.set(f"ipmpdrnn:abort:{task_id}", 1, ex=3600)
    celery_app.control.revoke(task_id, terminate=False)
    return {
        "status": "ABORT_SIGNAL_SENT",
        "message": f"DevDRNN-Aux task {task_id} abort signal sent to Redis.",
    }


@dev_drnn_aux_router.post("/tune/stop/{task_id}")
def stop_dev_drnn_aux_tune_task(task_id: str):
    celery_app.backend.client.set(f"ipmpdrnn:abort:{task_id}", 1, ex=3600)
    celery_app.control.revoke(task_id, terminate=False)
    return {
        "status": "ABORT_SIGNAL_SENT",
        "message": f"DevDRNN-Aux tune task {task_id} abort signal sent to Redis.",
    }


@dev_drnn_aux_router.get("/status/{task_id}")
def get_dev_drnn_aux_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    response: dict[str, Any] = {
        "task_id": task_id,
        "status": task_result.state,
        "info": None,
    }
    if task_result.state == "SUCCESS":
        response["info"] = task_result.get()
    elif task_result.state == "PROGRESS":
        response["info"] = task_result.info
    elif task_result.state == "FAILURE":
        response["info"] = str(task_result.info)
    elif task_result.state == "REVOKED":
        response["status"] = "ABORTED"
    return response


@dev_drnn_aux_router.get("/tune/status/{task_id}")
def get_dev_drnn_aux_tune_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    response: dict[str, Any] = {
        "task_id": task_id,
        "status": task_result.state,
        "info": None,
    }
    if task_result.state == "SUCCESS":
        response["info"] = task_result.get()
    elif task_result.state == "PROGRESS":
        response["info"] = task_result.info
    elif task_result.state == "FAILURE":
        response["info"] = str(task_result.info)
    elif task_result.state == "REVOKED":
        response["status"] = "ABORTED"
    return response


@dev_drnn_aux_router.get("/datasets", tags=["Config"])
def get_datasets():
    return {"datasets": VALID_DATASETS}