import os
import sys
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.dataset_config import VALID_DATASETS
from services.fcnn_service.tasks import celery_app, tune_fcnn_task

fcnn_tune_router = APIRouter(prefix="/nn/fcnn", tags=["FCNN Hyperparameter Tuning"])

DatasetEnum = Enum("DatasetEnum", {ds.upper(): ds for ds in VALID_DATASETS}, type=str)


class TuningBackendEnum(str, Enum):
    OPTUNA = "optuna"
    RAY = "ray"


class OptimizerEnum(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


class FCNNHyperparamTuningConfig(BaseModel):
    backend: TuningBackendEnum = Field(
        default=TuningBackendEnum.OPTUNA,
        description="A használt keresőmotor (optuna / ray)"
    )
    optimizer: OptimizerEnum = Field(
        default=OptimizerEnum.ADAM,
        description="A használt optimizer (adam / sgd)"
    )
    n_trials: int = Field(default=25, ge=1, le=200, description="Kísérletek / próbafutások száma")
    epochs: int = Field(default=100, ge=1, description="Max epoch-ok száma kísérletenként")
    patience: int = Field(default=10, ge=1, description="Early stopping türelmi idő kísérletenként")
    seed: bool = Field(default=False, description="Random seed rögzítése")

    # ---------------------------------------
    lr_min: float = Field(default=0.0004, ge=1e-6, description="Learning Rate alsó határa")
    lr_max: float = Field(default=0.1, le=1.0, description="Learning Rate felső határa")
    hidden_neurons_options: List[int] = Field(
        default=[216, 500, 866, 1000, 2000],
        description="Rejtett réteg választható neuronszámai"
    )
    batch_size_options: List[int] = Field(
        default=[32, 64, 128],
        description="Választható batch méretek"
    )
    momentum_min: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="SGD Momentum alsó határa")
    momentum_max: Optional[float] = Field(default=0.99, ge=0.0, le=1.0, description="SGD Momentum felső határa")


@fcnn_tune_router.post("/tune", status_code=status.HTTP_202_ACCEPTED)
async def start_fcnn_tuning(
    config: FCNNHyperparamTuningConfig,
    dataset_name: DatasetEnum = Query(..., description="A kiértékelendő adathalmaz neve")
):
    try:
        config_payload = {
            "dataset_name": dataset_name.value,
            **config.model_dump(mode="json", exclude_none=True)
        }
        task = tune_fcnn_task.delay(config=config_payload)
        return {"task_id": task.id, "status": "QUEUED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fcnn_tune_router.post("/tune/stop/{task_id}")
async def stop_fcnn_tuning(task_id: str):
    try:
        celery_app.backend.client.set(f"fcnn:abort:{task_id}", "true")
        return {
            "status": "ABORT_SIGNAL_SENT",
            "message": f"FCNN Tuning task {task_id} successfully signaled to abort gracefully via Redis."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fcnn_tune_router.get("/tune/status/{task_id}")
async def get_fcnn_tuning_status(task_id: str):
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
    elif task_result.state in ["ABORTED", "REVOKED"]:
        response["status"] = "ABORTED"
        response["info"] = {"status": "FCNN Hyperparameter Tuning task was manually aborted."}

    return response