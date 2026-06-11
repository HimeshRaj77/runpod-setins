#!/bin/bash
CMD1="echo '$(cat config.b64)' | base64 -d > /home/runpod-setins/config.py && exit"
ssh -tt 8birqqgcgxs9ng-6441142c@ssh.runpod.io "$CMD1"

CMD2="echo '$(cat runpod_start.b64)' | base64 -d > /home/runpod-setins/runpod_start.sh && exit"
ssh -tt 8birqqgcgxs9ng-6441142c@ssh.runpod.io "$CMD2"

CMD3="echo '$(cat llm_engine.b64)' | base64 -d > /home/runpod-setins/llm_engine.py && exit"
ssh -tt 8birqqgcgxs9ng-6441142c@ssh.runpod.io "$CMD3"

ssh -tt 8birqqgcgxs9ng-6441142c@ssh.runpod.io "chmod +x /home/runpod-setins/runpod_start.sh && exit"
