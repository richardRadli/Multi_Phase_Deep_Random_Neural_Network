import os
import sys
from enum import Enum
from fastapi import FastAPI, HTTPException, Query

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.dataset_config import VALID_DATASETS, general_dataset_configs
from services.dataset_operations_service.dataset_operations.convert_dataset import split_dataset

app = FastAPI(
    title="Dataset Operations Service",
    description="API for dataset conversion and dynamic splitting",
    version="1.0.0",
)

DatasetEnum = Enum("DatasetEnum", {ds.upper(): ds for ds in VALID_DATASETS})

@app.post("/dataset/convert", tags=["Dataset Operations"])
def conver_and_split_dataset(
        dataset: DatasetEnum,
        train_ratio: float = Query(0.8, ge=0.1, le=0.9, description="Ratio of training data to total data")
):
    try:
        ds_cfg = general_dataset_configs(dataset.value)
        total_size = ds_cfg.get("dataset_size")
        cached_file = ds_cfg.get("cached_dataset_file")

        split_dataset(dataset_name = dataset.value, train_ratio=train_ratio)

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


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("services.dataset_operations_service.api.api:app", host="127.0.0.1", port=8000, reload=True)