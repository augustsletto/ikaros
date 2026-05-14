import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from .metrics import BATCH_SIZE, QUEUE_DEPTH, INFERENCE_LATENCY, REQUESTS_TOTAL

class BatchQueue:
    def __init__(self, model_manager, max_batch_size=32, max_wait_ms=50):
        self.manager = model_manager
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000 # convert to seconds
        self.queue = asyncio.Queue()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        
    async def start(self):
        """Start the background batch processing loop"""
        self._running = True
        asyncio.create_task(self._process_loop())
        
    async def stop(self):
        self._running = False
        
    async def submit(self, model_id: str, input_data):
        """Submit a single request. Returns when the result is ready."""
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((model_id, input_data, future))
        QUEUE_DEPTH.set(self.queue.qsize())
        return await future
    
    async def _process_loop(self):
        """Runs forever. Collects batches and dispatches them."""
        while self._running:
            batch = []
            
            # Wait for first request (blocks until something arrives)
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                batch.append(item)
            except asyncio.TimeoutError:
                continue
            
            
            # Collect more requests for up to max_wait_ms
            await asyncio.sleep(0.005)  # 5ms — give time for more requests to arrive
            while not self.queue.empty() and len(batch) < self.max_batch_size:
                batch.append(self.queue.get_nowait())
                
            QUEUE_DEPTH.set(self.queue.qsize())

                
            
            # Group by model (different models can't be batched together)
            model_groups = {}
            for model_id, input_data, future in batch:
                if model_id not in model_groups:
                    model_groups[model_id] = []
                model_groups[model_id].append((input_data, future))
                
            
            # Process each model group
            for model_id, items in model_groups.items():
                inputs = [item[0] for item in items]
                futures = [item[1] for item in items]
                
                BATCH_SIZE.labels(model_id=model_id).observe(len(inputs))
                
                try:
                    loop = asyncio.get_event_loop()
                    
                    start = time.time()
                    
                    results = await loop.run_in_executor(
                        self._executor,
                        self.manager.predict,
                        model_id,
                        inputs
                    ) 
                    INFERENCE_LATENCY.labels(model_id=model_id).observe(time.time() - start)
                    
                    for future, result in zip(futures, results):
                        future.set_result(result)
                        REQUESTS_TOTAL.labels(model_id=model_id, status="success").inc()
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)
                        REQUESTS_TOTAL.labels(model_id=model_id, status="error").inc()
                        