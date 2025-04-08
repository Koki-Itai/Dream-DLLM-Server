#!/bin/bash

python src/client.py \
  --host localhost \
  --port 8000 \
  --max_new_tokens 512 \
  --temperature 0.7 \
  --top_p 0.92 \
  --steps 256 \
  --alg "entropy" \
  --alg_temp 0.3
