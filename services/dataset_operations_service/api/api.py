import os
import sys
from contextlib import asynccontextmanager
from enum import Enum
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.dataset_config import VALID_DATASETS, general_dataset_configs
from services.dataset_operations_service.dataset_operations.convert_dataset import split_dataset
from services.dataset_operations_service.init_datasets import check_and_download_datasets


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A Dataset szerviz indulásakor lefut a letöltés/ellenőrzés
    check_and_download_datasets()
    yield


app = FastAPI(
    title="Dataset Operations Service",
    description="API for dataset conversion and dynamic splitting",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DatasetEnum = Enum("DatasetEnum", {ds.upper(): ds for ds in VALID_DATASETS})


@app.get("/")
async def root():
    """Health check endpoint a React státuszjelző számára."""
    return {"service": "Dataset Service", "status": "healthy"}


@app.post("/dataset/convert", tags=["Dataset Operations"])
def convert_and_split_dataset(
        dataset: DatasetEnum,
        train_ratio: float = Query(0.8, ge=0.1, le=0.9, description="Ratio of training data to total data")
):
    try:
        ds_cfg = general_dataset_configs(dataset.value)
        total_size = ds_cfg.get("dataset_size")
        cached_file = ds_cfg.get("cached_dataset_file")

        split_dataset(dataset_name=dataset.value, train_ratio=train_ratio)

        num_train_data = int(total_size * train_ratio)
        num_test_data = total_size - num_train_data

        return {
            "status": "success",
            "message": f"Dataset '{dataset.value}' successfully converted and split based on the new ratio.",
            "dynamic_split_results": {
                "train_percentage": f"{train_ratio * 100}%",
                "test_percentage": f"{round(1.0 - train_ratio, 2) * 100}%",
                "num_train_samples": num_train_data,
                "num_test_samples": num_test_data
            },
            "saved_output_path": cached_file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets", tags=["Datasets"])
def get_all_available_datasets():
    return {"datasets": VALID_DATASETS}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("services.dataset_operations_service.api.api:app", host="127.0.0.1", port=8000, reload=True)