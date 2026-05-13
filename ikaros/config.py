from dataclasses import dataclass, field
from .enums import Task, Optimization
    
@dataclass
class ModelConfig:
    model_id: str # "openai/gpt-5.2"
    task: Task # ASR = automatic-speech-recognition"
    optimize: Optimization = Optimization.ONNX # ONNX, INT8, AUTO, NONE
    device: str = "cpu" # "cpu", "cuda", "cuda:0"
    max_batch_size: int = 32
    save_dir: str = "./cache" # optimized models dir
    
