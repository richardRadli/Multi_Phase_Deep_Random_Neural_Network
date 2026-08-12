import os
from fastapi import FastAPI
from services.dev_drnn_aux_service.api.dev_drnn_aux_router import dev_drnn_aux_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="IPMPDRNN Service",
    description="Dedicated Microservice for Iterative Pruning Multi-Phase Deep Randomized Neural Networks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("/app/storage"):
    app.mount("/static", StaticFiles(directory="/app/storage"), name="static")

app.include_router(dev_drnn_aux_router)

@app.get("/", tags=["Health Check"])
async def root():
    return {"service": "IPMPDRNN Service", "status": "healthy"}