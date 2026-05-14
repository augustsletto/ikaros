#!/bin/bash
echo "Stopping Ikaros..."
pkill -f run_worker.py 2>/dev/null
pkill -f uvicorn 2>/dev/null
docker compose down
echo "Done."