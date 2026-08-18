#!/bin/bash
# Does the quantization gradient survive under CONTROLLED cache configuration?
# Original RQ3 was measured with llama.cpp's LRU prompt cache at its default,
# which we have now shown drives run-to-run divergence. This re-measures the
# gradient with --cache-ram 0 on cross-arm divergence (cache-on vs cache-off),
# which does not depend on state carryover between passes.
# Q4_K_M is already measured (81.2%); this adds F16 and Q3_K_M.
set -uo pipefail
export PATH=/usr/local/cuda-12.6/bin:$PATH
cd /workspace/study
R=/workspace/quantgrad
SRV=/workspace/llama.cpp/build/bin/llama-server
QW=Qwen/Qwen2.5-7B-Instruct
NEP=${NEP:-80}
mkdir -p "$R" /workspace/models
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1; }
kill_srv(){ pkill -9 -f llama-server 2>/dev/null; for i in $(seq 1 40); do [ "$(gpu)" -lt 2000 ] && return 0; sleep 3; done; }
wait_up(){ for i in $(seq 1 300); do kill -0 "$1" 2>/dev/null || return 1
  curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null|grep -q 200 && return 0; sleep 3; done; return 1; }

cfg(){ # label file url
  local label=$1 f=/workspace/models/$2 url=$3
  if [ ! -f "$f" ]; then log "downloading $2"; wget -q "$url" -O "$f" || { log "DL FAIL $2"; return 1; }; fi
  kill_srv
  log "=== $label (--cache-ram 0) ==="
  "$SRV" -m "$f" --alias qg --host 127.0.0.1 --port 8080 --parallel 1 -c 16384 \
    -ngl 99 --seed 42 --slots --cache-ram 0 > "/workspace/srv_qg_$label.log" 2>&1 &
  wait_up $! || { log "SERVER FAIL $label"; return 1; }
  for a in on off; do
    local t0=$(date +%s)
    PYTHONPATH=harness uv run python harness/run_episodes.py --model-hf-id "$QW" \
      --served-model qg --base-url http://127.0.0.1:8080/v1 --backend llamacpp \
      --family qwen --arm $a --n-episodes "$NEP" --out-dir "$R/$label/main" >>/workspace/qg_cells.log 2>&1 \
      && log "$label arm_$a done in $(( $(date +%s)-t0 ))s" || log "$label arm_$a FAIL"
  done
  kill_srv; }

log "START quantization-gradient recheck"
cfg q3km Qwen2.5-7B-Instruct-Q3_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q3_K_M.gguf
cfg f16  Qwen2.5-7B-Instruct-f16.gguf     https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-f16.gguf
log "COMPLETE"
