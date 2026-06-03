#!/bin/bash
set -e

echo "Starting RunPod STT Server..."

# uvicorn with worker count from env or config
WORKERS=${WORKERS:-1}

exec uvicorn server:app --host 0.0.0.0 --port 8000 --workers $WORKERS
