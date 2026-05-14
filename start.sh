#!/bin/bash

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}${BOLD}  ╦╦╔═╔═╗╦═╗╔═╗╔═╗${NC}"
echo -e "${BLUE}${BOLD}  ║╠╩╗╠═╣╠╦╝║ ║╚═╗${NC}"
echo -e "${BLUE}${BOLD}  ╩╩ ╩╩ ╩╩╚═╚═╝╚═╝${NC}"
echo -e "${DIM}  inference server v0.1${NC}"
echo ""

# Kill any previous instances
pkill -f run_worker.py 2>/dev/null
pkill -f uvicorn 2>/dev/null
sleep 1

cleanup() {
    echo ""
    echo -e "${RED}■${NC} Shutting down..."
    kill $WORKER_PID 2>/dev/null
    kill $GATEWAY_PID 2>/dev/null
    docker-compose down -t 3 2>/dev/null
    echo -e "${GREEN}■${NC} Stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# 1. Docker
echo -e "${YELLOW}■${NC} Starting infrastructure ${DIM}(Redis, Prometheus, Grafana)${NC}"
docker-compose up -d 2>/dev/null
sleep 2
echo -e "${GREEN}■${NC} Infrastructure ready"

# 2. Worker
echo -e "${YELLOW}■${NC} Starting worker ${DIM}(loading model)${NC}"
uv run python run_worker.py &
WORKER_PID=$!
sleep 5
echo -e "${GREEN}■${NC} Worker ready"

# 3. Gateway
echo -e "${YELLOW}■${NC} Starting gateway"
IKAROS_REDIS=true uvicorn ikaros.server:app --host 0.0.0.0 --port 8000 --log-level warning &
GATEWAY_PID=$!
sleep 1
echo -e "${GREEN}■${NC} Gateway ready"

echo ""
echo -e "${BOLD}  Ikaros is running${NC}"
echo ""
echo -e "  ${CYAN}→${NC} Gateway     ${BOLD}${CYAN}http://localhost:8000${NC}"
echo -e "  ${CYAN}→${NC} Prometheus  ${BOLD}${CYAN}http://localhost:9090${NC}"
echo -e "  ${CYAN}→${NC} Grafana     ${BOLD}${CYAN}http://localhost:3001${NC}  ${DIM}admin/ikaros${NC}"
echo -e "  ${CYAN}→${NC} Metrics     ${BOLD}${CYAN}http://localhost:8000/metrics${NC}"
echo -e "  ${CYAN}→${NC} Worker      ${BOLD}${CYAN}http://localhost:8001/metrics${NC}"
echo ""
echo -e "  ${DIM}Press Ctrl+C to stop${NC}"
echo ""

wait
cleanup