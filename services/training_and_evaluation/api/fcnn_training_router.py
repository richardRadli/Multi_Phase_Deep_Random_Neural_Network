import logging
import os
import json
import time
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from services.training_and_evaluation.tasks import celery_app, train_fcnn_task, test_fcnn_task

fcnn_router = APIRouter(prefix="/nn/fcnn", tags=["FCNN Control & Evaluation"])


class FCNNTrainingConfig(BaseModel):
    dataset_name: str = Field(default="connect4")
    seed: bool = Field(default=False)
    epochs: int = Field(default=1000)
    num_tests: int = Field(default=1)


@fcnn_router.post("/train", status_code=status.HTTP_202_ACCEPTED)
async def start_fcnn_training(config: FCNNTrainingConfig):
    try:
        task = train_fcnn_task.delay(config=config.model_dump())
        return {"task_id": task.id, "status": "QUEUED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fcnn_router.post("/stop/{task_id}")
async def stop_fcnn_training(task_id: str):
    try:
        task_result = celery_app.AsyncResult(task_id)

        last_known_epoch = 0
        if task_result.state == "PROGRESS" and task_result.info:
            last_known_epoch = task_result.info.get("current_epoch", 0)

        celery_app.control.revoke(task_id=task_id, terminate=True, signal="SIGKILL")
        logging.info(f"FCNN training task {task_id} has been forcefully stopped.")

        PROJECT_ROOT = os.getenv("PROJECT_ROOT",
                                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        params_path = os.path.join(PROJECT_ROOT, "training_params.json")

        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                config_data = json.load(f)

            config_data["status"] = "aborted"
            config_data["end_time_str"] = time.strftime("%Y-%m-%d %H:%M:%S")
            config_data["final_epoch"] = last_known_epoch

            if "start_time" in config_data:
                config_data["execution_time_seconds"] = round(time.time() - config_data["start_time"], 4)

            with open(params_path, "w") as f:
                json.dump(config_data, f, indent=4)

        return {
            "status": "ABORTED",
            "message": f"Task {task_id} stopped. training_params.json updated with final epoch {last_known_epoch}."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fcnn_router.get("/status/{task_id}")
async def get_fcnn_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    response = {"task_id": task_id, "status": task_result.state, "info": None}

    if task_result.state == "PROGRESS":
        response["info"] = task_result.info
    elif task_result.state == "SUCCESS":
        response["info"] = task_result.get()
    elif task_result.state == "FAILURE":
        response["info"] = str(task_result.info)
    elif task_result.state == "PENDING":
        response["info"] = {"status": "Waiting in queue..."}
    elif task_result.state == "REVOKED":
        response["info"] = {"status": "Task was manually aborted by the user."}

    return response