#!/bin/bash
# Pinned vLLM launch for the cache-divergence study (pod lane).
# Usage: ./server_vllm.sh <hf-model-or-quant-path> <port> <alias> <cache:on|off> [extra args]
# vLLM version: pinned at pod setup, recorded in VERSIONS.md before any
# measured run. Cache arm is SERVER-LEVEL here (no per-request control):
#   on  -> automatic prefix caching enabled (V1 default)
#   off -> --no-enable-prefix-caching
# Determinism-relevant settings held constant across ALL runs:
#   --max-num-seqs 1   batch size 1, serial requests
#   --seed 42
#   max model len 16384 (matches llama.cpp lane context)
#   KV cache dtype: default fp16, never quantized
set -euo pipefail
MODEL="$1"
PORT="${2:-8000}"
ALIAS="${3:-model}"
CACHE="${4:-on}"
shift 4 || true
CACHE_FLAG=""
if [ "$CACHE" = "off" ]; then
  CACHE_FLAG="--no-enable-prefix-caching"
fi
exec vllm serve "$MODEL" \
  --served-model-name "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  --max-num-seqs 1 \
  --max-model-len 16384 \
  --seed 42 \
  $CACHE_FLAG \
  "$@"
