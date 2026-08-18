#!/bin/bash
# Repair run. Two experiments the internal review said the paper needs.
# Rewritten after an adversarial code review found eight defects in the first
# draft of this file, each of which would have produced unusable data.
#
# E1  ordering effect, isolated. The original grid used two execution
#     orderings and never disclosed it. Here ONE llama.cpp config and ONE
#     vLLM config are each run under BOTH orderings, so ordering is the only
#     thing that varies:
#       A: on/main, off/main, on/repeat        (cache-off pass in between)
#       B: on/main, on/repeat, off/main, off/repeat  (cache-on adjacent)
#     Cell directories are main|repeat / arm_on|arm_off, which is the layout
#     analysis/analyze.py actually reads. The first draft wrote a pass named
#     "interleaved" that the analyser would have skipped in silence.
#
# E2  reset control: is the cached path reproducible when cache state is
#     restored? See harness/reset_control.py for why the llama.cpp erase
#     endpoint is not trusted.
#
# Required: POD_ID must be set, or the pod cannot stop itself.
set -uo pipefail
export PATH=/workspace/venv-vllm/bin:/usr/local/cuda-12.6/bin:$PATH
cd /workspace/study

: "${POD_ID:?set POD_ID=<runpod id> before launching, otherwise the pod cannot self-stop}"
command -v runpodctl >/dev/null || { echo "runpodctl missing; refusing to run"; exit 1; }
runpodctl get pod "$POD_ID" >/dev/null 2>&1 || echo "WARNING: runpodctl cannot see pod $POD_ID; check the API key"

START_TS=$(date +%s)
MAX_SECONDS=$((4*3600))
R=/workspace/repair
PERSIST=/mnt/persistent/cache-repair
VENV=/workspace/venv-vllm
SRV=/workspace/llama.cpp/build/bin/llama-server
M=/workspace/models
QW=Qwen/Qwen2.5-7B-Instruct
GGUF="$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
NEP=${NEP:-80}
RESET_N=${RESET_N:-40}

mkdir -p "$R" "$PERSIST" /workspace/slots
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
sync_out() { rsync -a "$R/" "$PERSIST/" 2>/dev/null; cp /workspace/repair.log "$PERSIST/" 2>/dev/null; }
finish() { log "FINISH: sync + stop"; sync_out; runpodctl stop pod "$POD_ID" || log "STOP FAILED - stop it manually"; }
trap finish EXIT
# hard watchdog: budget_ok is only checked between blocks, a hung block never reaches it
( sleep $((MAX_SECONDS + 1800)); runpodctl stop pod "$POD_ID" ) &
budget_ok() { [ $(( $(date +%s) - START_TS )) -lt "$MAX_SECONDS" ]; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
kill_servers() {
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f llama-server 2>/dev/null
  for i in $(seq 1 60); do u=$(gpu_used); [ "${u:-99999}" -lt 2000 ] && return 0; sleep 3; done
  log "WARNING gpu still ${u} MiB"; }
wait_health() {  # url [tries] [pid]
  local url=$1 tries=${2:-400} pid=${3:-}
  for i in $(seq 1 "$tries"); do
    [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { log "server process died during startup"; return 1; }
    curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null | grep -q 200 && return 0
    sleep 3
  done; return 1; }

cell() {  # backend baseurl arm outdir
  local backend=$1 baseurl=$2 arm=$3 outdir=$4
  if [ -f "$outdir/arm_$arm/.done" ]; then log "SKIP $outdir/arm_$arm"; return 0; fi
  rm -rf "$outdir/arm_$arm"   # no .done means any partial log is untrustworthy
  PYTHONPATH=harness uv run python harness/run_episodes.py \
    --model-hf-id "$QW" --served-model repair --base-url "$baseurl" \
    --backend "$backend" --family qwen --arm "$arm" \
    --n-episodes "$NEP" --out-dir "$outdir" >> /workspace/repair_cells.log 2>&1 \
    && { touch "$outdir/arm_$arm/.done"; log "cell OK $outdir/arm_$arm"; } \
    || log "cell FAIL $outdir/arm_$arm"
  sync_out; }

start_lcpp() {  # logname
  kill_servers
  "$SRV" -m "$GGUF" --alias repair --host 127.0.0.1 --port 8080 \
    --parallel 1 -c 16384 -ngl 99 --seed 42 --slots \
    --slot-save-path /workspace/slots --cache-ram 0 \
    > "/workspace/srv_$1.log" 2>&1 &
  LCPP_PID=$!
  wait_health http://127.0.0.1:8080/health 200 "$LCPP_PID"; }

start_vllm() {  # logname extra_flags...
  local name=$1; shift
  kill_servers
  VLLM_SERVER_DEV_MODE=1 VIRTUAL_ENV=$VENV $VENV/bin/vllm serve "$QW" \
    --served-model-name repair --host 127.0.0.1 --port 8000 \
    --max-num-seqs 1 --max-model-len 16384 --seed 42 \
    --gpu-memory-utilization 0.85 "$@" > "/workspace/srv_$name.log" 2>&1 &
  VLLM_PID=$!
  wait_health http://127.0.0.1:8000/health 400 "$VLLM_PID"; }

log "=== REPAIR RUN START (pod $POD_ID) ==="
log "gpu at start: $(gpu_used) MiB"
[ -f "$GGUF" ] || { log "MISSING MODEL $GGUF, downloading"; \
  wget -q "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" -O "$GGUF" \
  || { log "FATAL: model download failed"; exit 1; }; }

# ---------------- E1a llama.cpp, ordering B first, fresh server ----------------
if budget_ok; then
  if start_lcpp lcpp-orderB_on; then
    log "--- E1a llama.cpp ordering B (cache-on passes adjacent) ---"
    cell llamacpp http://127.0.0.1:8080/v1 on  "$R/lcpp-orderB/main"
    cell llamacpp http://127.0.0.1:8080/v1 on  "$R/lcpp-orderB/repeat"
    cell llamacpp http://127.0.0.1:8080/v1 off "$R/lcpp-orderB/main"
    cell llamacpp http://127.0.0.1:8080/v1 off "$R/lcpp-orderB/repeat"
  else log "SERVER FAIL llama.cpp orderB"; fi
fi

# fresh server so ordering A starts from the same virgin slot as ordering B
if budget_ok; then
  if start_lcpp lcpp-orderA_on; then
    log "--- E1a llama.cpp ordering A (cache-off pass in between) ---"
    cell llamacpp http://127.0.0.1:8080/v1 on  "$R/lcpp-orderA/main"
    cell llamacpp http://127.0.0.1:8080/v1 off "$R/lcpp-orderA/main"
    cell llamacpp http://127.0.0.1:8080/v1 on  "$R/lcpp-orderA/repeat"
    cell llamacpp http://127.0.0.1:8080/v1 off "$R/lcpp-orderA/repeat"
  else log "SERVER FAIL llama.cpp orderA"; fi
fi

# ---------------- E2 reset control, llama.cpp ----------------
if budget_ok; then
  if start_lcpp reset-lcpp_on; then
    log "--- E2 reset control, llama.cpp (toggle mode) ---"
    PYTHONPATH=harness uv run python harness/reset_control.py \
      --model-hf-id "$QW" --served-model repair \
      --base-url http://127.0.0.1:8080/v1 --backend llamacpp --mode toggle \
      --n "$RESET_N" --out-dir "$R/reset-lcpp" >> /workspace/repair_cells.log 2>&1 \
      && log "reset control lcpp OK" || log "reset control lcpp FAIL (check for ABORT)"
    sync_out
  fi
fi
kill_servers

# ---------------- E1b vLLM, one server per arm ----------------
if budget_ok; then
  log "--- E1b vLLM ordering B (cache-on passes adjacent) ---"
  if start_vllm vllm-orderB_on; then
    cell vllm http://127.0.0.1:8000/v1 on "$R/vllm-orderB/main"
    cell vllm http://127.0.0.1:8000/v1 on "$R/vllm-orderB/repeat"
  else log "SERVER FAIL vllm orderB/on"; tail -3 /workspace/srv_vllm-orderB_on.log | sed 's/^/    /'; fi
  if start_vllm vllm-orderB_off --no-enable-prefix-caching; then
    cell vllm http://127.0.0.1:8000/v1 off "$R/vllm-orderB/main"
    cell vllm http://127.0.0.1:8000/v1 off "$R/vllm-orderB/repeat"
  else log "SERVER FAIL vllm orderB/off"; fi
fi

# vLLM has no per-request cache control, so ordering A is approximated by
# running the two cache-on passes with a genuine cache-off pass between them
# on a SEPARATE server. Documented as not identical to llama.cpp's ordering A.
if budget_ok; then
  log "--- E1b vLLM ordering A (cache-off pass between cache-on passes) ---"
  if start_vllm vllm-orderA_on1; then
    cell vllm http://127.0.0.1:8000/v1 on "$R/vllm-orderA/main"
  fi
  if start_vllm vllm-orderA_off --no-enable-prefix-caching; then
    cell vllm http://127.0.0.1:8000/v1 off "$R/vllm-orderA/main"
  fi
  if start_vllm vllm-orderA_on2; then
    cell vllm http://127.0.0.1:8000/v1 on "$R/vllm-orderA/repeat"
  fi
  if start_vllm vllm-orderA_off2 --no-enable-prefix-caching; then
    cell vllm http://127.0.0.1:8000/v1 off "$R/vllm-orderA/repeat"
  fi
fi

# ---------------- E2 reset control, vLLM ----------------
if budget_ok; then
  if start_vllm reset-vllm_on; then
    log "--- E2 reset control, vLLM (dev-mode reset endpoint) ---"
    PYTHONPATH=harness uv run python harness/reset_control.py \
      --model-hf-id "$QW" --served-model repair \
      --base-url http://127.0.0.1:8000/v1 --backend vllm --mode toggle \
      --n "$RESET_N" --out-dir "$R/reset-vllm" >> /workspace/repair_cells.log 2>&1 \
      && log "reset control vllm OK" || log "reset control vllm FAIL (check for ABORT)"
    sync_out
  fi
fi
kill_servers

log "--- telemetry ---"
uv run python harness/extract_cache_telemetry.py /workspace -o "$R/cache_telemetry.json" 2>&1 | tail -4
sync_out
log "=== REPAIR RUN COMPLETE ==="
