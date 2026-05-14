import asyncio
import json
import time
import uuid
from typing import Optional
import redis.asyncio as aioredis
from .metrics import BATCH_SIZE, QUEUE_DEPTH, REQUESTS_TOTAL


class RedisBatchQueue:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.request_queue = "ikaros:requests"
        self.result_prefix = "ikaros:result:"
        self._pending = {}  # request_id -> asyncio.Future

    async def connect(self):
        self.redis = aioredis.from_url(self.redis_url)
        asyncio.create_task(self._poll_results())

    async def disconnect(self):
        if self.redis:
            await self.redis.close()

    async def submit(self, model_id: str, input_data: str) -> dict:
        """Called by the gateway. Puts request in Redis, waits for result."""
        request_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        request = json.dumps({
            "id": request_id,
            "model_id": model_id,
            "input": input_data,
            "timestamp": time.time(),
        })

        await self.redis.lpush(self.request_queue, request)
        queue_len = await self.redis.llen(self.request_queue)
        QUEUE_DEPTH.set(queue_len)

        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            REQUESTS_TOTAL.labels(model_id=model_id, status="timeout").inc()
            raise TimeoutError(f"Request {request_id} timed out after 30s")

    async def _poll_results(self):
        """Polls Redis for completed results and resolves futures."""
        while True:
            for request_id in list(self._pending.keys()):
                result_key = f"{self.result_prefix}{request_id}"
                result = await self.redis.get(result_key)

                if result:
                    await self.redis.delete(result_key)
                    future = self._pending.pop(request_id, None)
                    if future and not future.done():
                        parsed = json.loads(result)
                        if "error" in parsed:
                            future.set_exception(Exception(parsed["error"]))
                        else:
                            future.set_result(parsed["result"])

            await asyncio.sleep(0.005)  # 5ms poll interval