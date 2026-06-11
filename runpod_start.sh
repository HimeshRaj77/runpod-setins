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

# Ask for VAD
read -p "Do you want to enable Voice Activity Detection (VAD)? (Y/n): " ENABLE_VAD
if [[ "$ENABLE_VAD" =~ ^[Nn]$ ]]; then
    echo "Disabling VAD..."
    export VAD_ENABLED="false"
else
    echo "Enabling VAD..."
    export VAD_ENABLED="true"
fi

# Ask for port
read -p "Enter the port to run the server on [8000]: " INPUT_PORT
PORT=${INPUT_PORT:-8000}
export PORT=$PORT

# Ask for LLM Provider
read -p "Which LLM Provider do you want to use? (ollama/groq) [ollama]: " INPUT_PROVIDER
PROVIDER=${INPUT_PROVIDER:-ollama}
export LLM_PROVIDER=$PROVIDER

if [ "$PROVIDER" = "groq" ]; then
    read -p "Enter your GROQ_API_KEY: " INPUT_GROQ_KEY
    export GROQ_API_KEY=$INPUT_GROQ_KEY
fi

# uvicorn with worker count from env or config
WORKERS=${WORKERS:-1}

exec uvicorn server:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers $WORKERS
