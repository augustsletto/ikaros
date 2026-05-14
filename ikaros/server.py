from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional

from .model_manager import ModelManager
from .batcher import BatchQueue
from .config import ModelConfig
from .enums import Task, Optimization


class DeployRequest(BaseModel):
    model_id: str
    task: Task
    optimize: Optimization = Optimization.ONNX
    device: str = "cpu"
    

class PredictRequest(BaseModel):
    model_id: str
    input: str
    
    
manager = ModelManager()
batcher = BatchQueue(manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await batcher.start()
    yield
    await batcher.stop()
    

app = FastAPI(title="Ikaros", version="0.1.0", lifespan=lifespan)


@app.post("/models")
async def deploy_model(request: DeployRequest):
    try:
        config = ModelConfig(
            model_id=request.model_id,
            task=request.task,
            optimize=request.optimize,
            device=request.device,
        )
        manager.deploy(config)
        return {
            "status": "deployed",
            "model": request.model_id,
            "task": request.task,
            "optimization": request.optimize.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/predict")
async def predict(request: PredictRequest):
    if request.model_id not in manager.models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not deployed")
    try:
        result = await batcher.submit(request.model_id, request.input)
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
    return {"status": "unloaded", "model": model_id}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": len(manager.models),
        "queue_depth": batcher.queue.qsize(),
    }