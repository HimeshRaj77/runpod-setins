#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# runpod_start_auto.sh — Non-interactive production launcher for RunPod STT
#
# This script bypasses all interactive prompts and starts the server with
# optimal low-latency settings. Use this instead of runpod_start.sh on RunPod.
#
# Key fixes vs the interactive script:
#   1. No read prompts — safe for `nohup`, `screen`, `pm2`, etc.
#   2. --ws-ping-interval 20 --ws-ping-timeout 30 keeps the proxy tunnel alive.
#   3. VAD_ENABLED=false — energy-based gate is faster; no Silero startup cost.
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "[runpod_start_auto] Starting RunPod STT Server (non-interactive mode)..."

# ── Device / precision ────────────────────────────────────────────────────────
# Auto-detect: use CUDA if available, otherwise CPU
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-float16}"

echo "[runpod_start_auto] Device: $DEVICE | Dtype: $DTYPE"

# ── VAD: use fast energy-based gate, skip Silero init overhead ────────────────
# The Node.js pipeline (RNNoise + local VAD) already pre-filters silence before
# audio arrives here.  Silero on the Python side adds latency without benefit.
export VAD_ENABLED="${VAD_ENABLED:-false}"
echo "[runpod_start_auto] VAD_ENABLED: $VAD_ENABLED"

# ── Port ──────────────────────────────────────────────────────────────────────
export PORT="${PORT:-8000}"
echo "[runpod_start_auto] Port: $PORT"

# ── LLM provider ─────────────────────────────────────────────────────────────
export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
echo "[runpod_start_auto] LLM_PROVIDER: $LLM_PROVIDER"

# ── Workers: single worker on GPU pod (multi-worker breaks GPU sharing) ───────
WORKERS="${WORKERS:-1}"
echo "[runpod_start_auto] Workers: $WORKERS"

echo "[runpod_start_auto] Launching uvicorn..."
exec uvicorn server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS" \
    --ws-ping-interval 20 \
    --ws-ping-timeout 30
