#!/bin/bash
# vLLM lane, strict value order, budget-bounded. Each config is independently
# valuable: the first successful one already refutes "llama.cpp-specific bug".
# Self-stops the pod on exit so no idle billing after the last cell.
set -uo pipefail
cd /workspace/study
POD_ID=fxf8uebg8rqyde
START_TS=$(date +%s)
MAX_SECONDS=$((7*3600 + 1800))   # 7.5h of the ~8.8h balance, leaves margin
N_EP=80
RESULTS=/workspace/results
PERSIST=/mnt/persistent/cache-study
VENV=/workspace/venv-vllm

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
sync_out() { rsync -a "$RESULTS/" "$PERSIST/results/" 2>/dev/null; cp /workspace/vllm_lane.log "$PERSIST/" 2>/dev/null; }
finish() { log "FINISH: sync + stop pod"; sync_out; runpodctl stop pod "$POD_ID" || log "STOP FAILED"; }
trap finish EXIT
budget_ok() { [ $(( $(date +%s) - START_TS )) -lt "$MAX_SECONDS" ]; }

wait_health() { for i in $(seq 1 ${2:-400}); do
    curl -s -o /dev/null -w '%{http_code}' "$1" 2>/dev/null | grep -q 200 && return 0; sleep 3; done; return 1; }
kill_servers() { pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f VLLM 2>/dev/null; sleep 10; }

vllm_config() {  # family hfid name
  local family=$1 hfid=$2 name=$3
  budget_ok || { log "BUDGET STOP before $name"; return 1; }
  [ -f "$RESULTS/$name/.config_done" ] && { log "SKIP $name"; return 0; }
  for arm in on off; do
    kill_servers
    local flag=""; [ "$arm" = off ] && flag="--no-enable-prefix-caching"
    log "starting server $name/$arm"
    VIRTUAL_ENV=$VENV $VENV/bin/vllm serve "$hfid" \
      --served-model-name "$name" --host 127.0.0.1 --port 8000 \
      --max-num-seqs 1 --max-model-len 16384 --seed 42 $flag \
      > "/workspace/srv_${name}_${arm}.log" 2>&1 &
    if ! wait_health http://127.0.0.1:8000/health; then
      log "SERVER FAIL $name/$arm"; tail -5 "/workspace/srv_${name}_${arm}.log" | sed 's/^/    /'
      kill_servers; return 1
    fi
    log "server up $name/$arm"
    for pass in main repeat; do
      out="$RESULTS/$name/${pass}"        # runner appends arm_<arm>
      [ -f "$out/arm_$arm/.done" ] && { log "SKIP cell $name/$pass/$arm"; continue; }
      PYTHONPATH=harness uv run python harness/run_episodes.py \
        --model-hf-id "$hfid" --served-model "$name" \
        --base-url http://127.0.0.1:8000/v1 --backend vllm --family "$family" \
        --arm "$arm" --n-episodes "$N_EP" --out-dir "$out" >> /workspace/cells.log 2>&1 \
        && { touch "$out/arm_$arm/.done"; log "cell OK $name/$pass/$arm"; } \
        || log "cell FAIL $name/$pass/$arm"
      sync_out
      budget_ok || { log "BUDGET STOP mid-$name"; kill_servers; return 1; }
    done
  done
  kill_servers; touch "$RESULTS/$name/.config_done"; sync_out
  log "CONFIG COMPLETE: $name"
}

log "=== VLLM LANE START ==="
$VENV/bin/python -c "import vllm,torch;print('stack:',vllm.__version__,torch.__version__,torch.cuda.is_available())" 2>&1 | tail -1

vllm_config qwen  Qwen/Qwen2.5-7B-Instruct                             vllm-qwen7b-fp16
vllm_config llama NousResearch/Meta-Llama-3.1-8B-Instruct              vllm-llama8b-fp16
vllm_config qwen  Qwen/Qwen2.5-7B-Instruct-AWQ                         vllm-qwen7b-awq
vllm_config llama hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4   vllm-llama8b-awq

log "=== VLLM LANE COMPLETE ==="
