from ikaros.worker import Worker
from ikaros.enums import Task, Optimization

worker = Worker(redis_url="redis://localhost:6379")

# Deploy models on the worker
worker.deploy(
    "distilbert-base-uncased-finetuned-sst-2-english",
    task=Task.TEXT_CLASSIFICATION,
    optimize=Optimization.INT8,
)

# Start processing
worker.run()