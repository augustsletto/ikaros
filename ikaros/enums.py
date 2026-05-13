# ikaros("asr", "kb-whisper")

from dataclasses import dataclass

from enum import Enum


class Task(str, Enum):
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    TRANSLATION = "translation"
    ASR = "automatic-speech-recognition"
    EMBEDDINGS = "feature-extraction"
    SUMMARIZATION = "summarization"
    

class Optimization(str, Enum):
    NONE = "none"
    ONNX = "onnx"
    INT8 = "int8"
    AUTO = "auto"
    