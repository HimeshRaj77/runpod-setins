#!/bin/bash
set -e

echo "Starting RunPod STT Server..."

# Ask for Mac M4 16GB optimization
read -p "Do you want to use MacBook Air M4 16GB RAM optimized settings? (y/N): " USE_M4_SETTINGS
if [[ "$USE_M4_SETTINGS" =~ ^[Yy]$ ]]; then
    echo "Applying Mac M4 optimizations (MPS device, float16)..."
    export DEVICE="mps"
    export DTYPE="float16"
fi

# Ask for port
read -p "Enter the port to run the server on [8000]: " INPUT_PORT
PORT=${INPUT_PORT:-8000}
export PORT=$PORT

# uvicorn with worker count from env or config
WORKERS=${WORKERS:-1}

exec uvicorn server:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers $WORKERS \
    --ws-ping-interval 20 \
    --ws-ping-timeout 30
