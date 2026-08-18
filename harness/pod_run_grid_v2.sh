#!/bin/bash
# Grid v2: resumes v1 (.done markers), fixes vLLM startup (pre-download +
# 25-min health wait), drops llama-f16 GGUF cell (no such file upstream),
# reorders so llama.cpp lane runs while vLLM checkpoints download in
# background. Self-stops pod on exit.
set -uo pipefail
export PATH=/usr/local/cuda-12.6/bin:$PATH
cd /workspace/study

POD_ID=fxf8uebg8rqyde
START_TS=$(date +%s)
MAX_SECONDS=$((19*3600))
N_EP=80
RESULTS=/workspace/results
PERSIST=/mnt/persistent/cache-study
SRV_LLAMACPP=/workspace/llama.cpp/build/bin/llama-server
mkdir -p "$RESULTS" "$PERSIST" /workspace/models

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
sync_out() { rsync -a --exclude 'unused_raw.jsonl' "$RESULTS/" "$PERSIST/results/" 2>/dev/null; cp /workspace/grid2.log "$PERSIST/" 2>/dev/null; }
finish() { log "FINISH trap: syncing and stopping pod"; sync_out; runpodctl stop pod "$POD_ID" || log "runpodctl stop FAILED"; }
trap finish EXIT
budget_ok() { [ $(( $(date +%s) - START_TS )) -lt "$MAX_SECONDS" ]; }

wait_health() { for i in $(seq 1 "${2:-120}"); do curl -s "$1" | grep -qE '"ok"|"healthy"' && return 0; curl -s -o /dev/null -w '%{http_code}' "$1" | grep -q 200 && return 0; sleep 3; done; return 1; }
kill_servers() { pkill -f llama-server 2>/dev/null; pkill -f 'vllm serve' 2>/dev/null; sleep 8; }

run_cell() { local family=$1 hfid=$2 served=$3 backend=$4 baseurl=$5 arm=$6 outdir=$7
  [ -f "$outdir/.done" ] && { log "SKIP (done): $outdir"; return 0; }
  PYTHONPATH=harness uv run python harness/run_episodes.py \
    --model-hf-id "$hfid" --served-model "$served" --base-url "$baseurl" \
    --backend "$backend" --family "$family" --arm "$arm" \
    --n-episodes "$N_EP" --out-dir "$outdir" >> /workspace/cells.log 2>&1
  local rc=$?
  [ $rc -eq 0 ] && { touch "$outdir/.done"; log "cell OK: $outdir"; } || log "cell FAIL($rc): $outdir"
  return $rc; }

llamacpp_config() { local family=$1 hfid=$2 gguf=$3 name=$4
  budget_ok || return 1
  [ -f "$gguf" ] || { log "MISSING MODEL, skip: $name"; return 0; }
  [ -f "$RESULTS/$name/.config_done" ] && { log "SKIP config: $name"; return 0; }
  kill_servers
  "$SRV_LLAMACPP" -m "$gguf" --alias "$name" --host 127.0.0.1 --port 8080 \
    --parallel 1 -c 16384 -ngl 99 --seed 42 --slots > "/workspace/srv_$name.log" 2>&1 &
  wait_health http://127.0.0.1:8080/health || { log "SERVER FAIL: $name"; kill_servers; return 1; }
  for pass in main repeat; do for arm in on off; do
    run_cell "$family" "$hfid" "$name" llamacpp http://127.0.0.1:8080/v1 "$arm" "$RESULTS/$name/${pass}/arm_$arm"
    budget_ok || { kill_servers; return 1; }
  done; done
  kill_servers; touch "$RESULTS/$name/.config_done"; sync_out; }

predownload() { # sequential background chain of all vLLM checkpoints
  for repo in "$@"; do
    log "predownload start: $repo"
    for try in 1 2 3; do
      uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$repo')" && break
      log "predownload retry $try failed: $repo"; sleep 60
    done
  done
  log "predownload chain finished"
}

vllm_config() { local family=$1 hfid=$2 name=$3; shift 3
  budget_ok || return 1
  [ -f "$RESULTS/$name/.config_done" ] && { log "SKIP config: $name"; return 0; }
  # ensure checkpoint fully present (waits on background chain via lock-free retry)
  uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$hfid')" || { log "DOWNLOAD FAIL: $hfid"; return 1; }
  for arm in on off; do
    kill_servers
    local flag=""; [ "$arm" = off ] && flag="--no-enable-prefix-caching"
    VIRTUAL_ENV=/workspace/venv-vllm /workspace/venv-vllm/bin/vllm serve "$hfid" \
      --served-model-name "$name" --host 127.0.0.1 --port 8000 \
      --max-num-seqs 1 --max-model-len 16384 --seed 42 $flag "$@" \
      > "/workspace/srv_${name}_${arm}.log" 2>&1 &
    wait_health http://127.0.0.1:8000/health 500 || { log "VLLM SERVER FAIL: $name/$arm"; kill_servers; return 1; }
    for pass in main repeat; do
      run_cell "$family" "$hfid" "$name" vllm http://127.0.0.1:8000/v1 "$arm" "$RESULTS/$name/${pass}/arm_$arm"
      budget_ok || { kill_servers; return 1; }
    done
  done
  kill_servers; touch "$RESULTS/$name/.config_done"; sync_out; }

log "=== GRID V2 START ==="
QW=Qwen/Qwen2.5-7B-Instruct
LL=NousResearch/Meta-Llama-3.1-8B-Instruct
M=/workspace/models

predownload "$QW" "$LL" Qwen/Qwen2.5-7B-Instruct-AWQ \
  hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8 \
  RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 > /workspace/predl.log 2>&1 &

log "--- llama.cpp lane (runs while vLLM checkpoints download) ---"
llamacpp_config qwen  "$QW" "$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf"        lcpp-qwen7b-q4km
llamacpp_config llama "$LL" "$M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" lcpp-llama8b-q4km
llamacpp_config qwen  "$QW" "$M/Qwen2.5-7B-Instruct-Q8_0.gguf"          lcpp-qwen7b-q80
llamacpp_config llama "$LL" "$M/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"   lcpp-llama8b-q80
llamacpp_config qwen  "$QW" "$M/Qwen2.5-7B-Instruct-Q3_K_M.gguf"        lcpp-qwen7b-q3km
llamacpp_config llama "$LL" "$M/Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf" lcpp-llama8b-q3km
llamacpp_config qwen  "$QW" "$M/Qwen2.5-7B-Instruct-f16.gguf"           lcpp-qwen7b-f16
log "llama-f16 GGUF cell DROPPED: upstream repo publishes no f16 file (2026-08-15)"

log "--- GSM8K bridge: llama.cpp ---"
if budget_ok && [ ! -f "$RESULTS/bridge-lcpp-qwen7b-q4km/.done" ]; then
  kill_servers
  "$SRV_LLAMACPP" -m "$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf" --alias qwen7b-q4km \
    --host 127.0.0.1 --port 8080 --parallel 1 -c 16384 -ngl 99 --seed 42 --slots \
    > /workspace/srv_bridge_lcpp.log 2>&1 &
  if wait_health http://127.0.0.1:8080/health; then
    PYTHONPATH=harness uv run python harness/run_gsm8k.py \
      --model-hf-id "$QW" --served-model qwen7b-q4km \
      --base-url http://127.0.0.1:8080/v1 --backend llamacpp --cache on \
      --n 200 --out-dir "$RESULTS/bridge-lcpp-qwen7b-q4km" >> /workspace/cells.log 2>&1 \
      && { touch "$RESULTS/bridge-lcpp-qwen7b-q4km/.done"; log "bridge lcpp OK"; } || log "bridge lcpp FAIL"
  fi
  kill_servers; sync_out
fi

log "--- vLLM lane ---"
vllm_config qwen  "$QW" vllm-qwen7b-fp16
vllm_config llama "$LL" vllm-llama8b-fp16
vllm_config qwen  Qwen/Qwen2.5-7B-Instruct-AWQ                       vllm-qwen7b-awq
vllm_config llama hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 vllm-llama8b-awq
vllm_config qwen  RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8        vllm-qwen7b-int8
vllm_config llama RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 vllm-llama8b-int8

log "--- GSM8K bridge: vLLM ---"
if budget_ok && [ ! -f "$RESULTS/bridge-vllm-qwen7b-fp16/.done" ]; then
  for arm in on off; do
    kill_servers
    flag=""; [ "$arm" = off ] && flag="--no-enable-prefix-caching"
    VIRTUAL_ENV=/workspace/venv-vllm /workspace/venv-vllm/bin/vllm serve "$QW" \
      --served-model-name qwen7b-fp16 --host 127.0.0.1 --port 8000 \
      --max-num-seqs 1 --max-model-len 16384 --seed 42 $flag \
      > "/workspace/srv_bridge_vllm_$arm.log" 2>&1 &
    if wait_health http://127.0.0.1:8000/health 500; then
      PYTHONPATH=harness uv run python harness/run_gsm8k.py \
        --model-hf-id "$QW" --served-model qwen7b-fp16 \
        --base-url http://127.0.0.1:8000/v1 --backend vllm --cache "$arm" \
        --n 200 --out-dir "$RESULTS/bridge-vllm-qwen7b-fp16" >> /workspace/cells.log 2>&1 \
        && log "bridge vllm/$arm OK" || log "bridge vllm/$arm FAIL"
    fi
  done
  touch "$RESULTS/bridge-vllm-qwen7b-fp16/.done"
  kill_servers; sync_out
fi

log "=== GRID V2 COMPLETE ==="
sync_out
