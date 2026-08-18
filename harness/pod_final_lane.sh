#!/bin/bash
# Final pod lane, budget-bounded, in value order:
#   0. fix transformers<5 for vLLM 0.11.0
#   1. vLLM Qwen FP16 agent config  -> kills "llama.cpp-specific" objection
#   2. GSM8K bridges across quant levels -> outcome evidence with a
#      dose-response curve (our weakest claim, cheapest to strengthen)
#   3. vLLM Llama FP16 if budget remains
# Self-stops the pod on exit.
set -uo pipefail
cd /workspace/study
POD_ID=fxf8uebg8rqyde
START_TS=$(date +%s)
MAX_SECONDS=$((6*3600))
RESULTS=/workspace/results
PERSIST=/mnt/persistent/cache-study
VENV=/workspace/venv-vllm
SRV=/workspace/llama.cpp/build/bin/llama-server
M=/workspace/models
QW=Qwen/Qwen2.5-7B-Instruct

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
sync_out() { rsync -a "$RESULTS/" "$PERSIST/results/" 2>/dev/null; cp /workspace/final_lane.log "$PERSIST/" 2>/dev/null; }
finish() { log "FINISH: sync + stop pod"; sync_out; runpodctl stop pod "$POD_ID" || log "STOP FAILED"; }
trap finish EXIT
budget_ok() { [ $(( $(date +%s) - START_TS )) -lt "$MAX_SECONDS" ]; }
wait_health() { for i in $(seq 1 ${2:-400}); do curl -s -o /dev/null -w '%{http_code}' "$1" 2>/dev/null | grep -q 200 && return 0; sleep 3; done; return 1; }
kill_servers() { pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f llama-server 2>/dev/null; sleep 8; }

log "=== STEP 0: transformers<5 for vLLM ==="
VIRTUAL_ENV=$VENV uv pip install "transformers<5" 2>&1 | tail -2
$VENV/bin/python -c "import transformers,vllm;print('stack:',transformers.__version__,vllm.__version__)" 2>&1 | tail -1

log "=== STEP 1: vLLM Qwen FP16 agent config ==="
vllm_agent() { local family=$1 hfid=$2 name=$3
  [ -f "$RESULTS/$name/.config_done" ] && { log "SKIP $name"; return 0; }
  for arm in on off; do
    budget_ok || { log "BUDGET STOP in $name"; return 1; }
    kill_servers
    local flag=""; [ "$arm" = off ] && flag="--no-enable-prefix-caching"
    log "server $name/$arm starting"
    VIRTUAL_ENV=$VENV $VENV/bin/vllm serve "$hfid" --served-model-name "$name" \
      --host 127.0.0.1 --port 8000 --max-num-seqs 1 --max-model-len 16384 --seed 42 $flag \
      > "/workspace/srv_${name}_${arm}.log" 2>&1 &
    if ! wait_health http://127.0.0.1:8000/health; then
      log "SERVER FAIL $name/$arm"; tail -4 "/workspace/srv_${name}_${arm}.log" | sed 's/^/    /'
      kill_servers; return 1; fi
    log "server $name/$arm UP"
    for pass in main repeat; do
      out="$RESULTS/$name/${pass}"
      [ -f "$out/arm_$arm/.done" ] && continue
      PYTHONPATH=harness uv run python harness/run_episodes.py --model-hf-id "$hfid" \
        --served-model "$name" --base-url http://127.0.0.1:8000/v1 --backend vllm \
        --family "$family" --arm "$arm" --n-episodes 80 --out-dir "$out" >> /workspace/cells.log 2>&1 \
        && { touch "$out/arm_$arm/.done"; log "cell OK $name/$pass/$arm"; } || log "cell FAIL $name/$pass/$arm"
      sync_out
      budget_ok || { kill_servers; return 1; }
    done
  done
  kill_servers; touch "$RESULTS/$name/.config_done"; log "CONFIG COMPLETE $name"; sync_out; }

vllm_agent qwen "$QW" vllm-qwen7b-fp16 || log "vLLM Qwen FP16 did not complete"

log "=== STEP 2: GSM8K bridges across quant levels (outcome dose-response) ==="
bridge() { local gguf=$1 name=$2
  budget_ok || { log "BUDGET STOP before bridge $name"; return 1; }
  [ -f "$RESULTS/bridge-$name/.done" ] && { log "SKIP bridge $name"; return 0; }
  [ -f "$gguf" ] || { log "MISSING $gguf"; return 0; }
  kill_servers
  "$SRV" -m "$gguf" --alias "$name" --host 127.0.0.1 --port 8080 --parallel 1 \
    -c 16384 -ngl 99 --seed 42 --slots > "/workspace/srv_bridge_$name.log" 2>&1 &
  wait_health http://127.0.0.1:8080/health 200 || { log "BRIDGE SERVER FAIL $name"; kill_servers; return 1; }
  PYTHONPATH=harness uv run python harness/run_gsm8k.py --model-hf-id "$QW" \
    --served-model "$name" --base-url http://127.0.0.1:8080/v1 --backend llamacpp \
    --cache on --n 200 --out-dir "$RESULTS/bridge-$name" >> /workspace/cells.log 2>&1 \
    && { touch "$RESULTS/bridge-$name/.done"; log "bridge OK $name"; } || log "bridge FAIL $name"
  kill_servers; sync_out; }

# f16 first (baseline), then descending precision: the dose-response curve
bridge "$M/Qwen2.5-7B-Instruct-f16.gguf"    qwen7b-f16
bridge "$M/Qwen2.5-7B-Instruct-Q8_0.gguf"   qwen7b-q80
bridge "$M/Qwen2.5-7B-Instruct-Q3_K_M.gguf" qwen7b-q3km
# Q4_K_M bridge already collected in the first grid run

log "=== STEP 3: vLLM Llama FP16 if budget remains ==="
vllm_agent llama NousResearch/Meta-Llama-3.1-8B-Instruct vllm-llama8b-fp16 || log "Llama vLLM skipped/failed"

log "=== FINAL LANE COMPLETE ==="
