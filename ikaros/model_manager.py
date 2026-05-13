from transformers import AutoTokenizer, AutoProcessor, pipeline
from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
    ORTModelForSeq2SeqLM,
    ORTModelForSpeechSeq2Seq,
    ORTModelForFeatureExtraction,
)
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from .config import ModelConfig
from .enums import Task, Optimization


TASK_TO_ORT = {
    Task.TEXT_CLASSIFICATION: ORTModelForSequenceClassification,
    Task.TOKEN_CLASSIFICATION: ORTModelForTokenClassification,
    Task.TRANSLATION: ORTModelForSeq2SeqLM,
    Task.ASR: ORTModelForSpeechSeq2Seq,
    Task.EMBEDDINGS: ORTModelForFeatureExtraction,
    Task.SUMMARIZATION: ORTModelForSeq2SeqLM
}


class ModelManager:
    def __init__(self):
        self.models = {} # model_id -> pipeline
        self.configs = {} # model_id -> ModelConfig
        
    def deploy(self, config: ModelConfig):
        if config.optimize == Optimization.NONE:
            pipe = self._load_raw(config)
        elif config.optimize == Optimization.ONNX:
            pipe = self._load_onnx(config)
        elif config.optimize == Optimization.INT8:
            pipe = self._load_int8(config)
        elif config.optimize == Optimization.AUTO:
            pipe = self._load_auto(config)
        
        self.models[config.model_id] = pipe
        self.configs[config.model_id] = config
        
        
    def predict(self, model_id: str, input_data):
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not deployed")
        return self.models[model_id](input_data)
    
    def unload(self, model_id: str):
        if model_id in self.models:
            del self.models[model_id]
            del self.configs[model_id]
            
    def list_models(self):
        return {
            model_id: {
                "task": config.task.value,
                "optimization": config.optimize.value,
                "device": config.device,
            }
            for model_id, config in self.configs.items()
        }
        
        
    def _load_raw(self, config: ModelConfig):
        return pipeline(
            config.task.value,
            model=config.model_id,
            device="cpu" if config.device == "cpu" else config.device,
        )    
        
        
    def _load_onnx(self, config: ModelConfig):
        ort_class = TASK_TO_ORT[config.task]
        provider = "CUDAExecutionProvider" if "cuda" in config.device else "CPUExecutionProvider"
        
        try:
            # Try 1: pre-existing ONNX on HF
            model = ort_class.from_pretrained(config.model_id, subfolder="onnx", provider=provider)
        except Exception:
            try:
                # Try 2: export to ONNX ourselves
                model = ort_class.from_pretrained(config.model_id, export=True, provider=provider)
            except Exception as e:
                # Try 3: model doesn't support ONNX, fall back to raw
                print(f"  [Ikaros] ONNX not available for {config.model_id}, using raw pipeline: {e}")
                return self._load_raw(config)
        
        tokenizer, processor = self._load_tokenizer(config)
        return pipeline(
            config.task.value,
            model=model,
            tokenizer=tokenizer,
            feature_extractor=processor.feature_extractor if processor else None,
            device="cpu" if config.device == "cpu" else config.device,
        )
            
    
    def _load_int8(self, config: ModelConfig):
        save_path = f"{config.save_dir}/{config.model_id.replace('/', '_')}_int8"
        ort_class = TASK_TO_ORT[config.task]
        provider = "CUDAExecutionProvider" if "cuda" in config.device else "CPUExecutionProvider"
        
        try:
            # Try 1: already quantized from a previous run
            model = ort_class.from_pretrained(save_path, provider=provider)
        except Exception:
            try:
                # Try 2: export to ONNX, then quantize
                model = ort_class.from_pretrained(config.model_id, export=True, provider=provider)
                quantizer = ORTQuantizer.from_pretrained(model)
                qconfig = AutoQuantizationConfig.avx2(is_static=False)
                quantizer.quantize(save_dir=save_path, quantization_config=qconfig)
                model = ort_class.from_pretrained(save_path, provider=provider)
            except Exception as e:
                # Try 3: quantization not supported, fall back to raw
                print(f"  [Ikaros] INT8 not available for {config.model_id}, using raw pipeline: {e}")
                return self._load_raw(config)
        
        tokenizer, processor = self._load_tokenizer(config)
        return pipeline(
            config.task.value,
            model=model,
            tokenizer=tokenizer,
            feature_extractor=processor.feature_extractor if processor else None,
            device="cpu" if config.device == "cpu" else config.device,
        )
        
        
    def _load_auto(self, config: ModelConfig):
        # TODO: benchmark all optimization levels, pick best, 
        # For now, default to ONNX
        return self._load_onnx(config)
    
    def _load_tokenizer(self, config: ModelConfig):
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        processor = None
        
        if config.task == Task.ASR:
            processor = AutoProcessor.from_pretrained(config.model_id)
            
        return tokenizer, processor