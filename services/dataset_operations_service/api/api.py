import os
import sys
from enum import Enum
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.dataset_config import VALID_DATASETS, general_dataset_configs
from services.dataset_operations_service.dataset_operations.convert_dataset import split_dataset

app = FastAPI(
    title="Dataset Operations Service",
    description="API for dataset conversion and dynamic 3-way splitting (Train / Validation / Test)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DatasetEnum = Enum("DatasetEnum", {ds.upper(): ds for ds in VALID_DATASETS})


class ConvertDatasetRequest(BaseModel):
    dataset: DatasetEnum
    split_ratio: List[float] = Field(
        default=[0.7, 0.15, 0.15],
        description="List of ratios for [train, validation, test]"
    )

    @field_validator("split_ratio")
    def validate_split_ratio(cls, v):
        if len(v) != 3:
            raise ValueError("A split_ratio-nak pontosan 3 elemet kell tartalmaznia! (pl. [0.7, 0.15, 0.15])")
        if not abs(sum(v) - 1.0) < 1e-5:
            raise ValueError("A split_ratio elemeinek összege 1.0 (100%) kell, hogy legyen!")
        return v


@app.get("/")
async def root():
    """Health check endpoint a React státuszjelző számára."""
    return {"service": "Dataset Service", "status": "healthy"}


@app.post("/dataset/convert", tags=["Dataset Operations"])
def convert_and_split_dataset(request: ConvertDatasetRequest):
    try:
        dataset_name = request.dataset.value
        ds_cfg = general_dataset_configs(dataset_name)
        total_size = ds_cfg.get("dataset_size")
        cached_file = ds_cfg.get("cached_dataset_file")

        split_dataset(dataset_name=dataset_name, split_ratio=request.split_ratio)

        train_r, valid_r, test_r = request.split_ratio

        num_test = int(total_size * test_r)
        num_valid = int((total_size - num_test) * (valid_r / (train_r + valid_r)))
        num_train = total_size - num_test - num_valid

        return {
            "status": "success",
            "message": f"Dataset '{dataset_name}' successfully converted and split into 3 sets.",
            "dynamic_split_results": {
                "train_percentage": f"{round(train_r * 100, 2)}%",
                "valid_percentage": f"{round(valid_r * 100, 2)}%",
                "test_percentage": f"{round(test_r * 100, 2)}%",
                "num_train_samples": num_train,
                "num_valid_samples": num_valid,
                "num_test_samples": num_test,
                "total_samples": total_size
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