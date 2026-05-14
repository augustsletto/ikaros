import asyncio
import aiohttp
import time
import json
import os
from datetime import datetime

URL = "http://localhost:8000/predict"
PAYLOAD = {
    "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
    "input": "This product is absolutely wonderful and I love it"
}

async def send_request(session):
    start = time.time()
    async with session.post(URL, json=PAYLOAD) as resp:
        result = await resp.json()
        latency = (time.time() - start) * 1000
        return {"latency_ms": latency, "status": resp.status}

async def run_load_test(num_users, duration_seconds):
    print(f"\nRunning: {num_users} concurrent users for {duration_seconds}s")
    
    all_requests = []
    batch_times = []
    end_time = time.time() + duration_seconds
    
    async with aiohttp.ClientSession() as session:
        while time.time() < end_time:
            tasks = [send_request(session) for _ in range(num_users)]
            batch_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_elapsed = (time.time() - batch_start) * 1000
            batch_times.append({"batch_size": len(results), "batch_ms": round(batch_elapsed, 2)})
            
            for r in results:
                if isinstance(r, Exception):
                    all_requests.append({"latency_ms": 0, "status": "error", "error": str(r)})
                else:
                    all_requests.append(r)

    # End-to-end latencies (what the user experiences)
    e2e_latencies = [r["latency_ms"] for r in all_requests if r.get("status") == 200]
    errors = sum(1 for r in all_requests if r.get("status") != 200)
    sorted_e2e = sorted(e2e_latencies)
    
    # Per-request within batch (model inference time)
    per_request_latencies = [bt["batch_ms"] / bt["batch_size"] for bt in batch_times]
    sorted_per_req = sorted(per_request_latencies)

    stats = {
        "concurrent_users": num_users,
        "duration_seconds": duration_seconds,
        "total_requests": len(e2e_latencies),
        "total_batches": len(batch_times),
        "errors": errors,
        "throughput_rps": round(len(e2e_latencies) / duration_seconds, 1),
        "end_to_end": {
            "avg_ms": round(sum(e2e_latencies) / len(e2e_latencies), 2) if e2e_latencies else 0,
            "p50_ms": round(sorted_e2e[len(sorted_e2e) // 2], 2) if e2e_latencies else 0,
            "p95_ms": round(sorted_e2e[int(len(sorted_e2e) * 0.95)], 2) if e2e_latencies else 0,
            "p99_ms": round(sorted_e2e[int(len(sorted_e2e) * 0.99)], 2) if e2e_latencies else 0,
            "min_ms": round(sorted_e2e[0], 2) if e2e_latencies else 0,
            "max_ms": round(sorted_e2e[-1], 2) if e2e_latencies else 0,
        },
        "per_request": {
            "avg_ms": round(sum(per_request_latencies) / len(per_request_latencies), 2) if per_request_latencies else 0,
            "p50_ms": round(sorted_per_req[len(sorted_per_req) // 2], 2) if per_request_latencies else 0,
            "p95_ms": round(sorted_per_req[int(len(sorted_per_req) * 0.95)], 2) if per_request_latencies else 0,
            "p99_ms": round(sorted_per_req[int(len(sorted_per_req) * 0.99)], 2) if per_request_latencies else 0,
        },
        "all_e2e_latencies_ms": [round(l, 2) for l in e2e_latencies],
        "all_per_request_latencies_ms": [round(l, 2) for l in per_request_latencies],
        "batch_details": batch_times,
    }
    
    print(f"  Requests: {stats['total_requests']}  Batches: {stats['total_batches']}  Errors: {stats['errors']}")
    print(f"  Throughput: {stats['throughput_rps']} rps")
    print(f"  End-to-end:   avg={stats['end_to_end']['avg_ms']}ms  p50={stats['end_to_end']['p50_ms']}ms  p95={stats['end_to_end']['p95_ms']}ms  p99={stats['end_to_end']['p99_ms']}ms")
    print(f"  Per-request:  avg={stats['per_request']['avg_ms']}ms  p50={stats['per_request']['p50_ms']}ms  p95={stats['per_request']['p95_ms']}ms  p99={stats['per_request']['p99_ms']}ms")
    
    return stats

async def main():
    print("=" * 60)
    print("IKAROS LOAD TEST")
    print("=" * 60)
    
    results = []
    for users in [1, 10, 25, 50, 100]:
        stats = await run_load_test(users, duration_seconds=10)
        results.append(stats)

    os.makedirs("logs/loadtests", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"logs/loadtests/loadtest_{timestamp}.json"
    
    with open(filepath, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "server": "http://localhost:8000",
            "model": PAYLOAD["model_id"],
            "test_input": PAYLOAD["input"],
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to {filepath}")

asyncio.run(main())