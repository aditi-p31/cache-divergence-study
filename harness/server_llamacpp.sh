#!/bin/zsh
# Pinned llama-server launch for the cache-divergence study.
# Usage: ./server_llamacpp.sh <gguf-path> <port> <alias>
# Version: b10434 (commit 7e4c0a968), binary in bin/llama-b10434.
# Determinism-relevant settings held constant across ALL runs:
#   --parallel 1     single slot, serial requests, no continuous batching
#   -c 16384         fixed context
#   --seed 42        fixed rng seed (greedy anyway via temperature 0)
#   -ngl 99          full GPU offload
#   KV cache dtype   default f16 in both arms (never quantized)
# Cache arm is controlled PER REQUEST via "cache_prompt" true/false.
set -euo pipefail
MODEL="$1"
PORT="${2:-8080}"
ALIAS="${3:-model}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/bin/llama-b10434/llama-server" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  --parallel 1 \
  -c 16384 \
  -ngl 99 \
  --seed 42 \
  --slots
