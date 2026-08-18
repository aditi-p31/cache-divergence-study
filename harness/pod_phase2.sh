#!/bin/bash
# Comprehensive remaining lane, strict value order, resumable, budget-capped.
# Everything already carrying a .done / .config_done marker is skipped.
#
# Order reflects which reviewer objection each item removes:
#   1 vLLM agent config   -> "llama.cpp-specific bug?"
#   2 GSM8K bridges x3    -> outcome evidence with quantization dose-response
#   3 SGLang agent+bridge -> third independent backend, effect is systemic
#   4 Qwen 14B agent+bridge -> model-scale generality
#   5 vLLM Llama FP16     -> second backend model
#   6 extended 500-item bridges -> tighter CIs on the flip rate
set -uo pipefail
cd /workspace/study
POD_ID=fxf8uebg8rqyde
START_TS=$(date +%s)
MAX_SECONDS=$((12*3600))
RESULTS=/workspace/results
PERSIST=/mnt/persistent/cache-study
VENV=/workspace/venv-vllm
SGVENV=/workspace/venv-sglang
SRV=/workspace/llama.cpp/build/bin/llama-server
M=/workspace/models
QW=Qwen/Qwen2.5-7B-Instruct
QW14=Qwen/Qwen2.5-14B-Instruct

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
sync_out() { rsync -a "$RESULTS/" "$PERSIST/results/" 2>/dev/null; cp /workspace/phase2.log "$PERSIST/" 2>/dev/null; }
finish() { log "FINISH: sync + stop pod"; sync_out; runpodctl stop pod "$POD_ID" || log "STOP FAILED"; }
trap finish EXIT
budget_ok() { [ $(( $(date +%s) - START_TS )) -lt "$MAX_SECONDS" ]; }
wait_health() { for i in $(seq 1 ${2:-400}); do curl -s -o /dev/null -w '%{http_code}' "$1" 2>/dev/null | grep -q 200 && return 0; sleep 3; done; return 1; }
kill_servers() { pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f llama-server 2>/dev/null; pkill -9 -f sglang 2>/dev/null; sleep 8; }

agent_cells() {  # family hfid served baseurl backend arm name
  local family=$1 hfid=$2 served=$3 baseurl=$4 backend=$5 arm=$6 name=$7
  for pass in main repeat; do
    out="$RESULTS/$name/${pass}"
    [ -f "$out/arm_$arm/.done" ] && { log "SKIP cell $name/$pass/$arm"; continue; }
    PYTHONPATH=harness uv run python harness/run_episodes.py --model-hf-id "$hfid" \
      --served-model "$served" --base-url "$baseurl" --backend "$backend" \
      --family "$family" --arm "$arm" --n-episodes 80 --out-dir "$out" >> /workspace/cells.log 2>&1 \
      && { touch "$out/arm_$arm/.done"; log "cell OK $name/$pass/$arm"; } || log "cell FAIL $name/$pass/$arm"
    sync_out
    budget_ok || return 1
  done
}

vllm_agent() {  # family hfid name
  local family=$1 hfid=$2 name=$3
  budget_ok || return 1
  [ -f "$RESULTS/$name/.config_done" ] && { log "SKIP $name"; return 0; }
  for arm in on off; do
    kill_servers
    local flag=""; [ "$arm" = off ] && flag="--no-enable-prefix-caching"
    log "vllm server $name/$arm starting"
    VIRTUAL_ENV=$VENV $VENV/bin/vllm serve "$hfid" --served-model-name "$name" \
      --host 127.0.0.1 --port 8000 --max-num-seqs 1 --max-model-len 16384 --seed 42 $flag \
      > "/workspace/srv_${name}_${arm}.log" 2>&1 &
    wait_health http://127.0.0.1:8000/health || { log "SERVER FAIL $name/$arm"; tail -4 "/workspace/srv_${name}_${arm}.log" | sed 's/^/    /'; kill_servers; return 1; }
    log "vllm server $name/$arm UP"
    agent_cells "$family" "$hfid" "$name" http://127.0.0.1:8000/v1 vllm "$arm" "$name" || { kill_servers; return 1; }
  done
  kill_servers; touch "$RESULTS/$name/.config_done"; log "CONFIG COMPLETE $name"; sync_out
}

sglang_agent() {  # family hfid name
  local family=$1 hfid=$2 name=$3
  budget_ok || return 1
  [ -f "$RESULTS/$name/.config_done" ] && { log "SKIP $name"; return 0; }
  for arm in on off; do
    kill_servers
    local flag=""; [ "$arm" = off ] && flag="--disable-radix-cache"
    log "sglang server $name/$arm starting"
    $SGVENV/bin/python -m sglang.launch_server --model-path "$hfid" \
      --served-model-name "$name" --host 127.0.0.1 --port 8100 \
      --max-running-requests 1 --context-length 16384 --random-seed 42 $flag \
      > "/workspace/srv_${name}_${arm}.log" 2>&1 &
    wait_health http://127.0.0.1:8100/health || { log "SERVER FAIL $name/$arm"; tail -4 "/workspace/srv_${name}_${arm}.log" | sed 's/^/    /'; kill_servers; return 1; }
    log "sglang server $name/$arm UP"
    agent_cells "$family" "$hfid" "$name" http://127.0.0.1:8100/v1 sglang "$arm" "$name" || { kill_servers; return 1; }
  done
  kill_servers; touch "$RESULTS/$name/.config_done"; log "CONFIG COMPLETE $name"; sync_out
}

lcpp_agent() {  # family hfid gguf name
  local family=$1 hfid=$2 gguf=$3 name=$4
  budget_ok || return 1
  [ -f "$RESULTS/$name/.config_done" ] && { log "SKIP $name"; return 0; }
  [ -f "$gguf" ] || { log "MISSING $gguf"; return 0; }
  kill_servers
  "$SRV" -m "$gguf" --alias "$name" --host 127.0.0.1 --port 8080 --parallel 1 \
    -c 16384 -ngl 99 --seed 42 --slots > "/workspace/srv_$name.log" 2>&1 &
  wait_health http://127.0.0.1:8080/health 200 || { log "SERVER FAIL $name"; kill_servers; return 1; }
  for arm in on off; do
    agent_cells "$family" "$hfid" "$name" http://127.0.0.1:8080/v1 llamacpp "$arm" "$name" || { kill_servers; return 1; }
  done
  kill_servers; touch "$RESULTS/$name/.config_done"; log "CONFIG COMPLETE $name"; sync_out
}

bridge() {  # gguf name hfid n
  local gguf=$1 name=$2 hfid=${3:-$QW} n=${4:-200}
  budget_ok || { log "BUDGET STOP before bridge $name"; return 1; }
  [ -f "$RESULTS/bridge-$name/.done" ] && { log "SKIP bridge $name"; return 0; }
  [ -f "$gguf" ] || { log "MISSING $gguf"; return 0; }
  kill_servers
  "$SRV" -m "$gguf" --alias "$name" --host 127.0.0.1 --port 8080 --parallel 1 \
    -c 16384 -ngl 99 --seed 42 --slots > "/workspace/srv_bridge_$name.log" 2>&1 &
  wait_health http://127.0.0.1:8080/health 200 || { log "BRIDGE SERVER FAIL $name"; kill_servers; return 1; }
  PYTHONPATH=harness uv run python harness/run_gsm8k.py --model-hf-id "$hfid" \
    --served-model "$name" --base-url http://127.0.0.1:8080/v1 --backend llamacpp \
    --cache on --n "$n" --out-dir "$RESULTS/bridge-$name" >> /workspace/cells.log 2>&1 \
    && { touch "$RESULTS/bridge-$name/.done"; log "bridge OK $name"; } || log "bridge FAIL $name"
  kill_servers; sync_out
}

log "=== PHASE 2 START ==="
$VENV/bin/python -c "import transformers,vllm;print('vllm stack:',transformers.__version__,vllm.__version__)" 2>&1 | tail -1

log "--- 1. vLLM Qwen FP16 agent ---"
vllm_agent qwen "$QW" vllm-qwen7b-fp16 || log "vllm qwen fp16 incomplete"

log "--- 2. GSM8K bridges, quantization dose-response ---"
bridge "$M/Qwen2.5-7B-Instruct-f16.gguf"    qwen7b-f16
bridge "$M/Qwen2.5-7B-Instruct-Q8_0.gguf"   qwen7b-q80
bridge "$M/Qwen2.5-7B-Instruct-Q3_K_M.gguf" qwen7b-q3km

log "--- 3. SGLang third backend ---"
sglang_agent qwen "$QW" sglang-qwen7b-fp16 || log "sglang incomplete"

log "--- 4. Qwen 14B scale point ---"
if budget_ok && [ ! -f "$M/Qwen2.5-14B-Instruct-Q4_K_M.gguf" ]; then
  log "downloading 14B gguf"
  wget -q "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf" \
    -O "$M/Qwen2.5-14B-Instruct-Q4_K_M.gguf" || { log "14B DL FAIL"; rm -f "$M/Qwen2.5-14B-Instruct-Q4_K_M.gguf"; }
fi
lcpp_agent qwen "$QW14" "$M/Qwen2.5-14B-Instruct-Q4_K_M.gguf" lcpp-qwen14b-q4km
bridge "$M/Qwen2.5-14B-Instruct-Q4_K_M.gguf" qwen14b-q4km "$QW14"

log "--- 5. vLLM Llama FP16 ---"
vllm_agent llama NousResearch/Meta-Llama-3.1-8B-Instruct vllm-llama8b-fp16 || log "vllm llama incomplete"

log "--- 6. extended 500-item bridges ---"
bridge "$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf" qwen7b-q4km-n500 "$QW" 500

log "=== PHASE 2 COMPLETE ==="
