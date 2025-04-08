#!/bin/bash
MODEL_NAME="Dream-org/Dream-v0-Instruct-7B"
PORT=8000
HOST="0.0.0.0"

# Specify GPU IDs (comma-separated for multiple GPUs, single ID for single GPU, -1 or empty for CPU)
GPU_IDS="0" # Example: "0", "0,1,2,3", "-1", ""

echo "Starting Dream model server on $HOST:$PORT..."

if [ -z "$GPU_IDS" ] || [ "$GPU_IDS" == "-1" ]; then
  echo "Using CPU"
  python src/serve.py --model $MODEL_NAME --port $PORT --host $HOST --device -1
elif [[ "$GPU_IDS" == *,* ]]; then
  echo "Using multiple GPUs: $GPU_IDS"
  python src/serve.py --model $MODEL_NAME --port $PORT --host $HOST --gpus "$GPU_IDS"
else
  if ! [[ "$GPU_IDS" =~ ^[0-9]+$ ]]; then
      echo "Error: Invalid GPU_IDS format for single GPU: '$GPU_IDS'. Must be a non-negative integer."
      exit 1
  fi
  echo "Using single GPU: $GPU_IDS"
  python src/serve.py --model $MODEL_NAME --port $PORT --host $HOST --device "$GPU_IDS"
fi
