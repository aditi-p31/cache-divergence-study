#!/bin/bash
# Pinned SGLang launch for the cache-divergence study (pod lane).
# Usage: ./server_sglang.sh <hf-model-or-quant-path> <port> <alias> <cache:on|off> [extra args]
# SGLang version: pinned at pod setup, recorded in VERSIONS.md before any
# measured run. Cache arm is SERVER-LEVEL:
#   on  -> RadixAttention prefix cache enabled (default)
#   off -> --disable-radix-cache
# Determinism-relevant settings held constant across ALL runs:
#   --max-running-requests 1   batch size 1, serial requests
#   --random-seed 42
#   --context-length 16384 (matches other lanes)
set -euo pipefail
MODEL="$1"
PORT="${2:-8100}"
ALIAS="${3:-model}"
CACHE="${4:-on}"
shift 4 || true
CACHE_FLAG=""
if [ "$CACHE" = "off" ]; then
  CACHE_FLAG="--disable-radix-cache"
fi
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  --max-running-requests 1 \
  --context-length 16384 \
  --random-seed 42 \
  $CACHE_FLAG \
  "$@"
