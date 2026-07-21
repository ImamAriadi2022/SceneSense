#!/bin/bash
export PYTHONUNBUFFERED=1
cd /workspaces/SceneSense
nohup python3 -u train.py > training_output.log 2>&1 &
echo "Training PID: $!"
