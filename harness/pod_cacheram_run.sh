#!/bin/bash
# Isolate --cache-ram. Only cache-ON cells: the within-arm repeat comparison
# is the measurement, and off cells were only ever needed to satisfy the
# analyser. Server freshness is held constant (fresh server per arm), so the
# flag is the sole difference.
set -uo pipefail
export PATH=/usr/local/cuda-12.6/bin:$PATH
cd /workspace/study
R=/workspace/cacheram
SRV=/workspace/llama.cpp/build/bin/llama-server
G=/workspace/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
QW=Qwen/Qwen2.5-7B-Instruct
NEP=${NEP:-80}
mkdir -p "$R"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1; }
kill_srv(){ pkill -9 -f llama-server 2>/dev/null; for i in $(seq 1 40); do [ "$(gpu)" -lt 2000 ] && return 0; sleep 3; done; }
wait_up(){ for i in $(seq 1 200); do kill -0 "$1" 2>/dev/null || return 1
  curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null|grep -q 200 && return 0; sleep 3; done; return 1; }
arm(){ local label=$1; shift
  kill_srv; log "=== $label : flags $* ==="
  "$SRV" -m "$G" --alias cr --host 127.0.0.1 --port 8080 --parallel 1 -c 16384 \
    -ngl 99 --seed 42 --slots --slot-save-path /workspace/slots "$@" \
    > "/workspace/srv_$label.log" 2>&1 &
  wait_up $! || { log "SERVER FAIL $label"; return 1; }
  local t0=$(date +%s)
  PYTHONPATH=harness uv run python harness/run_episodes.py --model-hf-id "$QW" \
    --served-model cr --base-url http://127.0.0.1:8080/v1 --backend llamacpp \
    --family qwen --arm on --n-episodes "$NEP" --out-dir "$R/$label/main" >>/workspace/cr_cells.log 2>&1
  log "$label main done in $(( $(date +%s)-t0 ))s"
  PYTHONPATH=harness uv run python harness/run_episodes.py --model-hf-id "$QW" \
    --served-model cr --base-url http://127.0.0.1:8080/v1 --backend llamacpp \
    --family qwen --arm on --n-episodes "$NEP" --out-dir "$R/$label/repeat" >>/workspace/cr_cells.log 2>&1
  log "$label repeat done"; kill_srv; }
log "START"
arm cacheram-zero    --cache-ram 0
arm cacheram-default
log "COMPLETE"
