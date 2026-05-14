from prometheus_client import Histogram, Gauge, Counter, Info, generate_latest, CONTENT_TYPE_LATEST


# Request metrics
REQUEST_LATENCY = Histogram(
    "ikaros_request_latency_seconds",
    "End-to-end request latency",
    ["model_id", "optimization"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

INFERENCE_LATENCY = Histogram(
    "ikaros_inference_latency_seconds",
    "Model inference time only",
    ["model_id"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


REQUESTS_TOTAL = Counter(
    "ikaros_requests_total",
    "Total inference requests",
    ["model_id", "status"],
)

# Batching metrics
BATCH_SIZE = Histogram(
    "ikaros_batch_size",
    "Number of requests per batch",
    ["model_id"],
    buckets=[1, 2, 4, 8, 16, 32, 64],
)


QUEUE_DEPTH = Gauge(
    "ikaros_queue_depth",
    "Current number of requests waiting in queue",
)

# Model metrics
MODELS_LOADED = Gauge(
    "ikaros_models_loaded",
    "Number of models currently loaded",
)

MODEL_LOAD_TIME = Histogram(
    "ikaros_model_load_seconds",
    "Time to load and optimize a model",
    ["model_id", "optimization"],
    buckets=[1, 5, 10, 30, 60, 120],
)
