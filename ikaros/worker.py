import json
import time
import redis
import threading
from prometheus_client import start_http_server
from .model_manager import ModelManager
from .config import ModelConfig
from .enums import Task, Optimization
from .metrics import BATCH_SIZE, INFERENCE_LATENCY, REQUESTS_TOTAL, MODELS_LOADED

# Colors
BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
DIM = '\033[2m'
NC = '\033[0m'


class Worker:
    def __init__(self, redis_url: str = "redis://localhost:6379", max_batch_size: int = 32, max_wait_ms: int = 50, metrics_port=8001):
        self.redis = redis.from_url(redis_url)
        self.manager = ModelManager()
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000
        self.request_queue = "ikaros:requests"
        self.result_prefix = "ikaros:result:"
        self._running = False
        self.metrics_port = metrics_port

    def deploy(self, model_id: str, task: Task, optimize: Optimization = Optimization.ONNX, device: str = "cpu"):
        config = ModelConfig(model_id=model_id, task=task, optimize=optimize, device=device)
        self.manager.deploy(config)
        MODELS_LOADED.set(len(self.manager.models))
        print(f"  {GREEN}■{NC} Deployed {BOLD}{model_id}{NC} {DIM}({optimize.value}){NC}")

    def run(self):
        """Main worker loop. Pulls batches from Redis, runs inference, pushes results."""
        self._running = True
        start_http_server(self.metrics_port)
        print(f"  {GREEN}■{NC} Metrics at {CYAN}http://localhost:{self.metrics_port}/metrics{NC}")
        print("")
        print(f"  {GREEN}{BOLD}✓ Worker ready — waiting for requests{NC}")
        print("")
        print(f"  {DIM}Try it:{NC}")
        print(f"  {DIM}  curl -X POST http://localhost:8000/predict \\{NC}")
        print(f"  {DIM}    -H \"Content-Type: application/json\" \\{NC}")
        print(f"  {DIM}    -d '{{\"model_id\":\"{list(self.manager.models.keys())[0]}\",\"input\":\"Hello world\"}}'{NC}")
        print("")

        while self._running:
            batch = []

            item = self.redis.brpop(self.request_queue, timeout=1)
            if not item:
                continue

            request = json.loads(item[1])
            batch.append(request)

            deadline = time.time() + self.max_wait_ms
            while len(batch) < self.max_batch_size and time.time() < deadline:
                item = self.redis.rpop(self.request_queue)
                if item:
                    batch.append(json.loads(item))
                else:
                    time.sleep(0.002)

            # Group by model
            model_groups = {}
            for req in batch:
                mid = req["model_id"]
                if mid not in model_groups:
                    model_groups[mid] = []
                model_groups[mid].append(req)

            # Process each group
            for model_id, requests in model_groups.items():
                inputs = [r["input"] for r in requests]

                BATCH_SIZE.labels(model_id=model_id).observe(len(inputs))
                print(f"  {YELLOW}→{NC} Batch {BOLD}{len(inputs)}{NC} reqs {DIM}({model_id}){NC}", end="")

                try:
                    start = time.time()
                    results = self.manager.predict(model_id, inputs)
                    elapsed = (time.time() - start) * 1000
                    INFERENCE_LATENCY.labels(model_id=model_id).observe(time.time() - start)
                    per_req = elapsed / len(inputs)
                    print(f"  {GREEN}✓{NC} {elapsed:.0f}ms {DIM}({per_req:.1f}ms/req){NC}")

                    for req, result in zip(requests, results):
                        result_key = f"{self.result_prefix}{req['id']}"
                        self.redis.set(result_key, json.dumps({"result": result}), ex=60)
                        REQUESTS_TOTAL.labels(model_id=model_id, status="success").inc()

                except Exception as e:
                    print(f"  {RED}✗ Error: {e}{NC}")
                    for req in requests:
                        result_key = f"{self.result_prefix}{req['id']}"
                        self.redis.set(result_key, json.dumps({"error": str(e)}), ex=60)
                        REQUESTS_TOTAL.labels(model_id=model_id, status="error").inc()

    def stop(self):
        self._running = False