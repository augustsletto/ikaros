import asyncio
import time
from ikaros import Ikaros, Task, Optimization
from ikaros.batcher import BatchQueue


async def main():
    server = Ikaros()

    print("Deploying model...")
    server.deploy(
        "distilbert-base-uncased-finetuned-sst-2-english",
        task=Task.TEXT_CLASSIFICATION,
        optimize=Optimization.INT8,
        device="cpu",
    )

    batcher = BatchQueue(server.manager, max_batch_size=32, max_wait_ms=50)
    await batcher.start()

    # --- Test 1: Single request through batcher ---
    print("\n=== Test 1: Single request ===")
    result = await batcher.submit(
        "distilbert-base-uncased-finetuned-sst-2-english",
        "This movie is amazing"
    )
    print(f"  Result: {result}")

    # --- Test 2: Many concurrent requests ---
    print("\n=== Test 2: 50 concurrent requests ===")
    texts = [
        f"This is test sentence number {i}" for i in range(50)
    ]

    start = time.time()
    tasks = [
        batcher.submit("distilbert-base-uncased-finetuned-sst-2-english", text)
        for text in texts
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f"  Processed {len(results)} requests in {elapsed:.3f}s")
    print(f"  Throughput: {len(results) / elapsed:.1f} requests/sec")
    print(f"  First result: {results[0]}")
    print(f"  Last result: {results[-1]}")

    # --- Test 3: Compare batched vs unbatched ---
    print("\n=== Test 3: Batched vs Unbatched (100 requests) ===")

    # Unbatched — one at a time
    start = time.time()
    for text in texts[:100]:
        server.predict("distilbert-base-uncased-finetuned-sst-2-english", text)
    unbatched_time = time.time() - start

    # Batched — all at once
    start = time.time()
    tasks = [
        batcher.submit("distilbert-base-uncased-finetuned-sst-2-english", text)
        for text in texts[:100]
    ]
    await asyncio.gather(*tasks)
    batched_time = time.time() - start

    print(f"  Unbatched: {unbatched_time:.3f}s ({100 / unbatched_time:.1f} rps)")
    print(f"  Batched:   {batched_time:.3f}s ({100 / batched_time:.1f} rps)")
    print(f"  Speedup:   {unbatched_time / batched_time:.1f}x")


# Add this as Test 4 in test_batcher.py

    # --- Test 4: Heavy load (500 requests) ---
    print("\n=== Test 4: Heavy load (500 requests) ===")

    heavy_texts = [f"Test sentence number {i} for heavy load benchmark" for i in range(500)]

    # Unbatched
    start = time.time()
    for text in heavy_texts:
        server.predict("distilbert-base-uncased-finetuned-sst-2-english", text)
    unbatched_time = time.time() - start

    # Batched
    start = time.time()
    tasks = [
        batcher.submit("distilbert-base-uncased-finetuned-sst-2-english", text)
        for text in heavy_texts
    ]
    await asyncio.gather(*tasks)
    batched_time = time.time() - start

    print(f"  Unbatched: {unbatched_time:.3f}s ({500 / unbatched_time:.1f} rps)")
    print(f"  Batched:   {batched_time:.3f}s ({500 / batched_time:.1f} rps)")
    print(f"  Speedup:   {unbatched_time / batched_time:.1f}x")



    await batcher.stop()


asyncio.run(main())