#!/bin/bash
# ============================================================================
# RunPod STT Server - Non-Interactive Production Startup Script
# 
# Usage: Set environment variables then run this script:
#   DEVICE=cuda DTYPE=float16 VAD_ENABLED=true PORT=8000 \
#   LLM_PROVIDER=groq GROQ_API_KEY=gsk_... \
#   bash runpod_start_noninteractive.sh
#
# Or run in background with nohup:
#   nohup bash runpod_start_noninteractive.sh > /tmp/stt_server.log 2>&1 &
# ============================================================================
set -e

echo "Starting RunPod STT Server (non-interactive mode)..."

# Defaults from environment (or sensible fallbacks)
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-float16}"
export VAD_ENABLED="${VAD_ENABLED:-true}"
export PORT="${PORT:-8000}"
export LLM_PROVIDER="${LLM_PROVIDER:-groq}"
export LLM_ENABLED="${LLM_ENABLED:-true}"
export LLM_MODE="${LLM_MODE:-conversational}"
export WORKERS="${WORKERS:-1}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export MIN_AUDIO_ENERGY="${MIN_AUDIO_ENERGY:-0.001}"

# GROQ_API_KEY must be set externally
if [ -z "$GROQ_API_KEY" ] && [ "$LLM_PROVIDER" = "groq" ]; then
    echo "ERROR: GROQ_API_KEY must be set when LLM_PROVIDER=groq"
    exit 1
fi

echo "Config: DEVICE=$DEVICE | VAD=$VAD_ENABLED | PORT=$PORT | LLM=$LLM_PROVIDER | LLM_ENABLED=$LLM_ENABLED | LLM_MODE=$LLM_MODE | WORKERS=$WORKERS"

# ============================================================================
# Launch uvicorn with WebSocket keep-alive pings
#
# CRITICAL: --ws-ping-interval 20 --ws-ping-timeout 30 is REQUIRED to prevent
# the RunPod proxy from treating idle WebSocket connections as timed-out and 
# closing them. Without these, worker pool connections drop after ~30-60s.
# ============================================================================
exec uvicorn server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS" \
    --ws-ping-interval 20 \
    --ws-ping-timeout 30 \
    --log-level "${LOG_LEVEL,,}"
