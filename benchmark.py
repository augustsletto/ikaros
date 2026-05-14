# benchmark.py
import time
import json
import os
from datetime import datetime
from ikaros import Ikaros, Task, Optimization

server = Ikaros()
test_text = "This movie was absolutely fantastic, I loved every minute of it"
RUNS = 200

results = []

print("=" * 60)
print("BENCHMARK: distilbert text-classification")
print(f"Input: '{test_text[:50]}...'")
print(f"Runs per optimization level: {RUNS}")
print("=" * 60)

for opt in [Optimization.NONE, Optimization.ONNX, Optimization.INT8]:
    print(f"\nLoading with {opt.value}...")
    
    load_start = time.time()
    handle = server.deploy(
        "distilbert-base-uncased-finetuned-sst-2-english",
        task=Task.TEXT_CLASSIFICATION,
        optimize=opt,
        device="cpu",
    )
    load_time = time.time() - load_start

    for _ in range(5):
        handle.predict(test_text)

    times = []
    for _ in range(RUNS):
        start = time.time()
        result = handle.predict(test_text)
        elapsed = time.time() - start
        times.append(elapsed)

    sorted_times = sorted(times)
    avg = sum(times) / len(times) * 1000
    p50 = sorted_times[len(times) // 2] * 1000
    p95 = sorted_times[int(len(times) * 0.95)] * 1000
    p99 = sorted_times[int(len(times) * 0.99)] * 1000
    total = sum(times)
    rps = RUNS / total

    entry = {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "text-classification",
        "optimization": opt.value,
        "device": "cpu",
        "runs": RUNS,
        "load_time_s": round(load_time, 3),
        "avg_ms": round(avg, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "throughput_rps": round(rps, 1),
        "accuracy_match": result[0]["label"] == "POSITIVE",
        "confidence": round(result[0]["score"], 4),
        "all_latencies_ms": [round(t * 1000, 2) for t in times],
    }
    results.append(entry)

    print(f"  Load time:  {load_time:.2f}s")
    print(f"  Avg:        {avg:.2f}ms")
    print(f"  p50:        {p50:.2f}ms")
    print(f"  p95:        {p95:.2f}ms")
    print(f"  p99:        {p99:.2f}ms")
    print(f"  Throughput: {rps:.1f} requests/sec")

    handle.unload()

# Save results
os.makedirs("logs/benchmarks", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filepath = f"logs/benchmarks/benchmark_{timestamp}.json"

with open(filepath, "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "input_text": test_text,
        "runs_per_level": RUNS,
        "results": results,
    }, f, indent=2)

print(f"\nResults saved to {filepath}")