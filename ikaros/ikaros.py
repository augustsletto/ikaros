from .model_manager import ModelManager
from .config import ModelConfig
from .enums import Task, Optimization


class ModelHandle:
    def __init__(self, model_id: str, manager: ModelManager):
        self.model_id = model_id
        self.manager = manager


    def predict(self, input_data):
        return self.manager.predict(self.model_id, input_data)
    
    def unload(self):
        self.manager.unload(self.model_id)



class Ikaros:
    def __init__(self):
        self.manager = ModelManager()
        
    
    def deploy(self, model_id: str, task: Task, optimize: Optimization = Optimization.ONNX, device: str = "cpu"):
        config = ModelConfig(
            model_id=model_id,
            task=task,
            optimize=optimize,
            device=device,
        )
        self.manager.deploy(config)
        return ModelHandle(model_id, self.manager)
    
    
        
    def predict(self, model_id: str, input_data):
        return self.manager.predict(model_id, input_data)
    
    def unload(self, model_id: str):
        self.manager.unload(model_id)
        
    def list_models(self):
        return self.manager.list_models()