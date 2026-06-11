#!/bin/bash
echo "$(cat config.b64)" | base64 -d > /home/runpod-setins/config.py
echo "$(cat runpod_start.b64)" | base64 -d > /home/runpod-setins/runpod_start.sh
echo "$(cat llm_engine.b64)" | base64 -d > /home/runpod-setins/llm_engine.py
chmod +x /home/runpod-setins/runpod_start.sh
