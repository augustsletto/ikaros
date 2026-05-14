from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time
import os

from .model_manager import ModelManager
from .batcher import BatchQueue
from .redis_batcher import RedisBatchQueue
from .config import ModelConfig
from .enums import Task, Optimization
from .metrics import (
    REQUEST_LATENCY, MODELS_LOADED,
    MODEL_LOAD_TIME, generate_latest, CONTENT_TYPE_LATEST,
)


class DeployRequest(BaseModel):
    model_id: str
    task: Task
    optimize: Optimization = Optimization.ONNX
    device: str = "cpu"

class PredictRequest(BaseModel):
    model_id: str
    input: str


USE_REDIS = os.getenv("IKAROS_REDIS", "false").lower() == "true"
REDIS_URL = os.getenv("IKAROS_REDIS_URL", "redis://localhost:6379")

manager = ModelManager()

if USE_REDIS:
    batcher = RedisBatchQueue(redis_url=REDIS_URL)
else:
    batcher = BatchQueue(manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if USE_REDIS:
        await batcher.connect()
    else:
        await batcher.start()
    yield
    if USE_REDIS:
        await batcher.disconnect()
    else:
        await batcher.stop()


app = FastAPI(title="Ikaros", version="0.1.0", lifespan=lifespan)


@app.post("/models")
async def deploy_model(request: DeployRequest):
    if USE_REDIS:
        return {"status": "deploy via worker", "note": "Deploy models on the worker process, not the gateway"}
    try:
        config = ModelConfig(
            model_id=request.model_id,
            task=request.task,
            optimize=request.optimize,
            device=request.device,
        )
        start = time.time()
        manager.deploy(config)
        MODEL_LOAD_TIME.labels(
            model_id=request.model_id,
            optimization=request.optimize.value,
        ).observe(time.time() - start)
        MODELS_LOADED.set(len(manager.models))
        return {
            "status": "deployed",
            "model": request.model_id,
            "task": request.task.value,
            "optimization": request.optimize.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictRequest):
    if not USE_REDIS and request.model_id not in manager.models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not deployed")
    try:
        config = manager.configs.get(request.model_id) if not USE_REDIS else None
        opt = config.optimize.value if config else "unknown"

        start = time.time()
        result = await batcher.submit(request.model_id, request.input)
        REQUEST_LATENCY.labels(
            model_id=request.model_id,
            optimization=opt,
        ).observe(time.time() - start)

        return {"model": request.model_id, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    return manager.list_models()


@app.delete("/models/{model_id:path}")
async def unload_model(model_id: str):
    if model_id not in manager.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not deployed")
    manager.unload(model_id)
    MODELS_LOADED.set(len(manager.models))
    return {"status": "unloaded", "model": model_id}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": len(manager.models),
        "redis_mode": USE_REDIS,
    }


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )