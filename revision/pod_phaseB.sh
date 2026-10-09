#!/bin/bash
# Phase B orchestrator v2, IEEE Access resubmission (Access-2026-42095).
# Usage: pod_phaseB.sh <block> [...]   blocks: preflight B0 B2 B1 B1R B3 B6 B4 B5
#   B0  hard gate: reproduce August token-for-token; logprobs=5 must not change the arithmetic
#   B2  cache-off references (top-5); gates: identical to August grid, identical across orders
#   B1  controlled sweep: canonical + 10 randomized histories x 4 formats, fresh server each
#   B1R fresh-process replicate at the canonical history for f16, q80, q3km
#   B3  sentinel-isolated episodes, 3 orders per format
#   B6  production-default prompt cache under 5 randomized histories (main + repeat on one server)
#   B4  reset control: 4 formats x 3 fresh sessions x 100 GSM8K items
#   B5  single-stream latency of serving modes, Q4_K_M
# Every pass: fresh server with the August controlled flags (unless stated), staged output,
# strict validation, atomic promotion, verified mirror to the persistent volume.
set -uo pipefail
W=/root/work; ST=$W/study; R=$W/results; REF=$W/references; ORD=$W/orders
SRV=$W/llama.cpp/build/bin/llama-server; M=$W/models
PY=$ST/.venv/bin/python; TOOLS="$PY $W/phaseb_tools.py"
QW=Qwen/Qwen2.5-7B-Instruct; QW_REV=a09a35458c702b33eeacc393d103063234e8bc28
LEDGER=/mnt/persistent/phaseB/T0
RATE_CENTS=${RATE_CENTS:-77}; BUDGET_CAP=${BUDGET_CAP:-23}; WALL_CAP_H=${WALL_CAP_H:-40}; GRACE_MIN=${GRACE_MIN:-45}
export PATH=/usr/local/cuda-12.6/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
CTRL="--cache-ram 0"
VALID_BLOCKS=" preflight B0 B2 B1 B1R B3 B6 B4 B5 B7 "
for B in "$@"; do [[ "$VALID_BLOCKS" == *" $B "* ]] || { echo "unknown block $B"; exit 2; }; done

exec 9>"$W/phaseB.lock"; flock -n 9 || { echo "another phaseB instance holds the lock"; exit 1; }
ap=$(cat "$W/autostop.pid" 2>/dev/null)
if [[ $ap =~ ^[0-9]+$ ]] && ps -o args= -p "$ap" 2>/dev/null | grep -q '[a]utostop\.sh'; then kill -KILL -- -"$ap" 2>/dev/null; fi
rm -f "$W/autostop.pid"
mkdir -p "$R" "$W/srvlogs" "$W/slots" "$R/gates" "$R/checks" "$R/alerts" "$R/_staging" "$R/_failed" /mnt/persistent/phaseB
[ -f "$LEDGER" ] || cp "$W/T0" "$LEDGER" 2>/dev/null || date -u +%s > "$LEDGER"
T0_EPOCH=$(cat "$LEDGER" 2>/dev/null); [[ $T0_EPOCH =~ ^[0-9]{10}$ ]] || T0_EPOCH=$(cat "$W/T0" 2>/dev/null)
[[ $T0_EPOCH =~ ^[0-9]{10}$ ]] || { echo "spend ledger unreadable"; exit 1; }

declare -A GGUF=([f16]=Qwen2.5-7B-Instruct-f16.gguf [q80]=Qwen2.5-7B-Instruct-Q8_0.gguf
                 [q4km]=Qwen2.5-7B-Instruct-Q4_K_M.gguf [q3km]=Qwen2.5-7B-Instruct-Q3_K_M.gguf
                 [q4deq]=Qwen2.5-7B-Instruct-Q4_K_M-deq-F16.gguf)
WILLIAMS=("f16 q80 q3km q4km" "q80 q4km f16 q3km" "q4km q3km q80 f16" "q3km f16 q4km q80")

log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$W/phaseB.log"; return 0; }
status(){ echo "$(date -u +%s) $*" > "$W/STATUS"; return 0; }
alert(){ log "ALERT $1: $2"; echo "{\"alert\": \"$1\", \"detail\": \"$2\"}" > "$R/alerts/$1.json"; return 0; }
cents(){ echo $(( ($(date +%s) - T0_EPOCH) * RATE_CENTS / 3600 )); }
hours_x100(){ echo $(( ($(date +%s) - T0_EPOCH) * 100 / 3600 )); }
gpu_mem(){ timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
kill_harness(){ pkill -9 -f '[h]arness/run_episodes.py'; pkill -9 -f '[h]arness/reset_control.py'; return 0; }
kill_srv(){ pkill -9 -f "[l]lama.cpp/build/bin/llama-server"
  for i in $(seq 1 60); do m=$(gpu_mem); [ -n "$m" ] && [ "$m" -lt 1500 ] && return 0; sleep 2; done
  log "WARN gpu memory not released: $(gpu_mem) MiB"; return 0; }
MIRROR_STALLED_PID=""
do_mirror(){ # background with a deadline: a stalled network mount must not hang the run.
  # MIRROR_FAILED alone blocks every automatic stop until a later mirror verifies.
  if [ -n "$MIRROR_STALLED_PID" ] && ps -o args= -p "$MIRROR_STALLED_PID" 2>/dev/null | grep -q '[m]irror\.sh'; then
    touch "$W/MIRROR_FAILED"; log "previous mirror still stalled; skipping this mirror"; return 1; fi
  MIRROR_STALLED_PID=""
  "$W/mirror.sh" 9>&- & local mp=$! i
  for i in $(seq 1 600); do kill -0 $mp 2>/dev/null || break; sleep 2; done
  if kill -0 $mp 2>/dev/null; then
    MIRROR_STALLED_PID=$mp; touch "$W/MIRROR_FAILED"; alert MIRROR_STALLED "mirror did not finish within 20 min"; return 1; fi
  if wait $mp; then rm -f "$W/MIRROR_FAILED"; return 0; fi
  touch "$W/MIRROR_FAILED"; alert MIRROR_FAILED "mirror to the persistent volume could not be verified"; return 1; }

FINISHED=0
CLEAN_END=0
finish(){ trap ':' HUP TERM INT; [ $FINISHED -eq 1 ] && exit 0; FINISHED=1
  kill_harness; kill_srv; do_mirror; status "$1 $2"; log "$1: $2"; touch "$W/$1"
  if [ -f "$W/MIRROR_FAILED" ]; then log "mirror failed: no auto-stop, pod left running"; exit 0; fi
  setsid nohup "$W/autostop.sh" "$GRACE_MIN" "$1: $2" 9>&- < /dev/null > /dev/null 2>&1 &
  echo $! > "$W/autostop.pid"; log "auto-stop armed: ${GRACE_MIN} min grace (HOLD suspends)"; exit 0; }
trap 'log "signal received"; finish ABORT "orchestrator interrupted by signal"' HUP TERM INT
trap '[ $FINISHED -eq 1 ] || [ $CLEAN_END -eq 1 ] || finish ABORT "unexpected exit"' EXIT

guard(){ local need_c=$(( $1 * 3 * RATE_CENTS / 120 ))
  [ $(( $(cents) + need_c )) -le $((BUDGET_CAP*100)) ] || finish ABORT "budget cap: ~\$$(( $(cents)/100 )) spent, next step may need $1 min"
  [ $(( $(hours_x100) + $1 * 100 / 60 )) -le $((WALL_CAP_H*100)) ] || finish ABORT "wall-clock cap"; }

start_srv(){ # fmt tag [server flags...]
  local fmt=$1 tag=$2; shift 2; kill_srv
  local t0; t0=$(date +%s%N)
  "$SRV" -m "$M/${GGUF[$fmt]}" --alias repair --host 127.0.0.1 --port 8080 --parallel 1 -c 16384 \
     -ngl 99 --seed 42 --slots --slot-save-path "$W/slots" "$@" > "$W/srvlogs/$tag.log" 2>&1 9>&- &
  local pid=$! deadline=$(( $(date +%s) + 300 ))
  while [ "$(date +%s)" -lt $deadline ]; do
    kill -0 $pid 2>/dev/null || { log "server died: $tag: $(tail -3 "$W/srvlogs/$tag.log" | tr '\n' ' ')"; return 1; }
    if curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null | grep -q 200; then
      "$PY" -c "print(round(($(date +%s%N) - $t0) / 1e9, 3))" > "$W/srvlogs/$tag.startup_s"
      curl -s -m 10 http://127.0.0.1:8080/props > "$W/srvlogs/$tag.props.json"
      curl -s -m 10 http://127.0.0.1:8080/slots > "$W/srvlogs/$tag.slots.json"
      echo "$*" > "$W/srvlogs/$tag.flags"
      timeout 20 nvidia-smi --query-gpu=name,driver_version,clocks.sm,temperature.gpu,power.draw --format=csv,noheader > "$W/srvlogs/$tag.gpu"
      return 0; fi
    sleep 0.5; done
  log "server never healthy: $tag"; return 1; }

data_fp(){ # fingerprint of everything that can change the generated data
  { echo "$*"; sha256sum "$SRV" "$W"/llama.cpp/build/bin/*.so 2>/dev/null; sha256sum "$ST"/harness/handler.py "$ST"/harness/run_episodes.py "$ST"/harness/reset_control.py
    cat "$M/SHA256SUMS"; } | sha256sum | cut -c1-32; }

run_harness(){ # outdir arm lp est_min [run_episodes args...]
  local out=$1 arm=$2 lp=$3 est=$4; shift 4
  ( cd "$ST" && CDS_LOGPROBS=$lp PYTHONPATH=harness timeout -k 60 $(( est * 60 * 3 )) "$PY" harness/run_episodes.py \
      --model-hf-id "$QW" --served-model repair --base-url http://127.0.0.1:8080/v1 --backend llamacpp \
      --family qwen --arm "$arm" --n-episodes 80 --out-dir "$out" "$@" ) >> "$W/passes.log" 2>&1 9>&-; }

promote(){ # staged_arm_dir final_arm_dir fingerprint ; old final is moved aside, never deleted first
  echo "$3" > "$1/.validated"; mkdir -p "$(dirname "$2")"
  if [ -e "$2" ]; then mv -T "$2" "$R/_failed/superseded_$(echo "$2" | tr '/' '_')_$(date +%s)" || { log "promote: cannot move aside $2"; return 1; }; fi
  mv -T "$1" "$2" || { log "promote failed: $2"; return 1; }
  do_mirror; return 0; }
fail_keep(){ local d=$R/_failed/$(echo "$1" | tr '/' '_')_$(date +%s); mkdir -p "$d"; [ -e "$2" ] && mv "$2" "$d/"; log "kept failed attempt in $d"; return 0; }

vflags(){ # arm lp args... -> validate flags
  local arm=$1 lp=$2; shift 2; local f="--arm $arm --lp $lp"
  [[ " $* " == *" --sentinel-reset "* ]] && f="$f --sentinel"
  local prev=""; for x in "$@"; do [ "$prev" = "--order-file" ] && f="$f --order-file $x"; prev=$x; done
  echo "$f"; }

pass(){ # label fmt arm lp est_min "server flags" [run_episodes args...]
  local label=$1 fmt=$2 arm=$3 lp=$4 est=$5 sflags=$6; shift 6
  local final="$R/$label/arm_$arm" stage="$R/_staging/$label" fp
  fp=$(data_fp "$label|$fmt|$arm|$lp|$sflags|$*|$(for x in "$@"; do [ -f "$x" ] && sha256sum "$x"; done)")
  if [ -f "$final/.validated" ]; then
    if grep -qx "$fp" "$final/.validated"; then log "skip (validated, same fingerprint) $label/arm_$arm"; return 0; fi
    if [ ! -s "$final/.validated" ] && $TOOLS validate "$final" $(vflags "$arm" "$lp" "$@") > "$final/validation.json"; then
      echo "$fp" > "$final/.validated"; do_mirror; log "adopted existing pass after revalidation: $label/arm_$arm"; return 0; fi
    log "existing $label/arm_$arm has a different fingerprint: rerunning"
  fi
  for attempt in 1 2; do
    guard "$est"; status "RUNNING $label arm_$arm attempt $attempt"
    rm -rf "${stage:?}"; mkdir -p "$stage"
    local t0; t0=$(date +%s); local tag; tag=$(echo "${label}_$arm" | tr '/' '_')
    # shellcheck disable=SC2086
    if ! start_srv "$fmt" "$tag" $sflags; then kill_srv; fail_keep "$label.a$attempt" "$stage"; continue; fi
    run_harness "$stage" "$arm" "$lp" "$est" "$@"; local rc=$?
    kill_harness; kill_srv
    cp "$W/srvlogs/$tag".* "$stage/arm_$arm/" 2>/dev/null
    local v; v=$($TOOLS validate "$stage/arm_$arm" $(vflags "$arm" "$lp" "$@")); local vrc=$?
    echo "$v" > "$stage/arm_$arm/validation.json" 2>/dev/null
    if [ $rc -eq 0 ] && [ $vrc -eq 0 ]; then
      if ! promote "$stage/arm_$arm" "$final" "$fp"; then fail_keep "$label.a$attempt" "$stage"; continue; fi
      log "OK $label/arm_$arm in $(( $(date +%s) - t0 ))s (~\$$(( $(cents)/100 )).$(( $(cents)%100 )) spent)"; return 0; fi
    log "INVALID $label/arm_$arm attempt $attempt (harness rc $rc): $(echo "$v" | head -c 600)"
    fail_keep "$label.a$attempt" "$stage"
  done
  return 1; }

gate(){ # name A B min_shared [lp_tol] -> 0 if identical (lp_tol only across top-k depths)
  local res rc; res=$($TOOLS compare "$2" "$3" --require-identical --min-shared "$4" --lp-tol "${5:-0}"); rc=$?
  echo "$res" > "$R/gates/$1.json"
  local s; s=$(echo "$res" | "$PY" -c 'import json,sys
try:
  r=json.load(sys.stdin); print(r.get("episodes"),"episodes;",r.get("n_diverged"),"differ;",r.get("shared_tokens"),"shared tokens;",r.get("shared_tokens_top1_logprob_differs"),"logprob diffs, max",r.get("max_abs_top1_logprob_diff"))
except Exception as e: print("unparseable:",e)')
  if [ $rc -eq 0 ]; then log "GATE PASS $1: $s"; return 0; fi
  log "GATE FAIL $1 (rc $rc): $s"; return 1; }

note(){ # name A B [lp_tol] : recorded comparison; ALERT when not identical
  local res rc; res=$($TOOLS compare "$2" "$3" --require-identical --min-shared 1 --lp-tol "${4:-0}"); rc=$?
  echo "$res" > "$R/checks/$1.json"
  local s; s=$(echo "$res" | "$PY" -c 'import json,sys
try:
  r=json.load(sys.stdin); print(r.get("n_diverged"),"of",r.get("episodes"),"differ;",r.get("shared_tokens_top1_logprob_differs"),"logprob diffs, max",r.get("max_abs_top1_logprob_diff"))
except Exception as e: print("unparseable:",e)')
  if [ $rc -eq 0 ]; then log "check $1: identical ($s)"; else alert "$1" "$s"; fi; return 0; }

block_fail(){ mkdir -p "$R/$1"; touch "$R/$1/.failed"; log "BLOCK $1 FAILED: $2 (continuing with the next block)"; status "BLOCK_FAILED $1"; }

run_block(){
  case $1 in B1|B1R|B2|B3|B4|B5|B6|B7) rm -f "$R/$1/.done" "$R/$1/.failed";; esac
  [ "$1" = B2 ] && rm -f "$R/B2"/.ok_* "$R/B2"/.off_per_history_*
  case $1 in
preflight)
  log "PREFLIGHT v2"; status "PREFLIGHT"
  [ -f "$W/SETUP_DONE" ] || finish ABORT "setup not done"
  local v; v=$("$SRV" --version 2>&1); [[ $v == *7e4c0a9* ]] || finish ABORT "llama-server is not commit 7e4c0a9"
  mountpoint -q /mnt/persistent || finish ABORT "persistent volume is not mounted"
  (cd "$M" && sha256sum -c SHA256SUMS > /dev/null) || finish ABORT "model checksum failure"
  "$W/stop_pod.sh" --check > /dev/null 2>&1 || log "WARN pod-scoped key cannot read the pod (self-stop unavailable)"
  for r in aug_q4km_orderB_on aug_q4km_orderB_off aug_grid_f16_off aug_grid_q80_off aug_grid_q4km_off aug_grid_q3km_off aug_ctrl_f16_on aug_ctrl_q4km_on aug_ctrl_q3km_on; do
    [ -f "$REF/$r/raw_requests.jsonl" ] || finish ABORT "missing reference $r"; done
  (cd "$REF" && sha256sum -c SHA256SUMS > /dev/null) || finish ABORT "reference checksum failure"
  "$PY" - "$ORD" <<'PYX' || finish ABORT "order files invalid"
import json, sys, hashlib
o = sys.argv[1]; d = json.load(open(f"{o}/orders.json"))
assert hashlib.sha256(json.dumps(d["orders"], sort_keys=True).encode()).hexdigest() == d["sha256"]
canon = d["orders"]["0"]
for k in range(11):
    f = json.load(open(f"{o}/order_{k}.json")); assert f == d["orders"][str(k)] and sorted(f) == sorted(canon) and len(set(f)) == 80
PYX
  flock -n "$W/watchdog.lock" true && finish ABORT "watchdog is not running"
  [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -gt 15 ] || finish ABORT "less than 15 GB free on the container disk"
  "$PY" -c "
from transformers import AutoTokenizer; from datasets import load_dataset
AutoTokenizer.from_pretrained('$QW', revision='$QW_REV'); AutoTokenizer.from_pretrained('$QW'); load_dataset('openai/gsm8k','main',split='test')" >> "$W/passes.log" 2>&1 || finish ABORT "tokenizer/GSM8K prefetch failed"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  { sha256sum "$ST"/harness/*.py "$W"/phaseb_tools.py "$ST"/uv.lock "$SRV"; uv pip freeze --python "$PY" 2>/dev/null; } > "$R/env_manifest.txt"
  start_srv q4km preflight_contract $CTRL || finish ABORT "contract server failed"
  "$PY" - <<'PYX' >> "$W/phaseB.log" 2>&1 || { kill_srv; finish ABORT "contract test failed: top-5 logprobs or telemetry missing"; }
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="EMPTY")
r = c.completions.create(model="repair", prompt="The capital of France is", max_tokens=8, temperature=0.0,
                         logprobs=5, extra_body={"seed": 42, "cache_prompt": True}).model_dump()
ct = r["choices"][0]["logprobs"]["content"]
assert len(ct) == r["usage"]["completion_tokens"] > 0
assert all(len(t["top_logprobs"]) == 5 and t["top_logprobs"][0]["id"] == t["id"] for t in ct)
assert "cache_n" in r["timings"]
print("contract test OK:", len(ct), "tokens with 5 alternatives each")
PYX
  kill_srv
  rm -rf "${R:?}/_staging/smoke"; start_srv q4km preflight_sentinel $CTRL || finish ABORT "sentinel smoke server failed"
  ( cd "$ST" && CDS_LOGPROBS=5 PYTHONPATH=harness timeout -k 30 900 "$PY" harness/run_episodes.py --model-hf-id "$QW" --served-model repair \
      --base-url http://127.0.0.1:8080/v1 --backend llamacpp --family qwen --arm on --n-episodes 2 \
      --out-dir "$R/_staging/smoke" --sentinel-reset ) >> "$W/passes.log" 2>&1 9>&-
  kill_harness; kill_srv
  $TOOLS validate "$R/_staging/smoke/arm_on" --arm on --lp 5 --n 2 --sentinel > "$R/checks/sentinel_smoke.json" || finish ABORT "sentinel smoke test failed: $(head -c 300 "$R/checks/sentinel_smoke.json")"
  log "sentinel smoke test OK"; rm -rf "${R:?}/_staging/smoke"
  do_mirror || finish ABORT "mirror to the persistent volume failed in preflight"
  touch "$W/PREFLIGHT_OK"; log "preflight PASS"; status "PREFLIGHT_OK" ;;

B0)
  log "=== B0 GATE: reproduce August exactly; logprobs=5 must not change the arithmetic ==="
  rm -f "$R/B0/.gate_passed"
  pass B0/q4km_off_lp1 q4km off 1 16 "$CTRL" || finish ABORT "B0 pass failed"
  gate G0a_off_vs_august "$R/B0/q4km_off_lp1/arm_off" "$REF/aug_q4km_orderB_off" 20000 || finish ABORT "G0a failed: cache-off does not reproduce August"
  pass B0/q4km_on_lp1 q4km on 1 8 "$CTRL" || finish ABORT "B0 pass failed"
  gate G0b_on_vs_august "$R/B0/q4km_on_lp1/arm_on" "$REF/aug_q4km_orderB_on" 15000 || finish ABORT "G0b failed: cache-on does not reproduce August"
  pass B0/q4km_on_lp5 q4km on 5 8 "$CTRL" || finish ABORT "B0 pass failed"
  gate G0c_lp5_on_invariance "$R/B0/q4km_on_lp5/arm_on" "$R/B0/q4km_on_lp1/arm_on" 15000 1e-3 || finish ABORT "G0c failed: logprobs=5 changes the cache-on computation"
  pass B0/q4km_off_lp5 q4km off 5 16 "$CTRL" || finish ABORT "B0 pass failed"
  gate G0d_lp5_off_invariance "$R/B0/q4km_off_lp5/arm_off" "$R/B0/q4km_off_lp1/arm_off" 20000 1e-3 || finish ABORT "G0d failed: logprobs=5 changes the cache-off computation"
  touch "$R/B0/.gate_passed"; do_mirror; log "B0 GATE PASSED"; status "B0_PASSED" ;;

B2)
  [ -f "$R/B0/.gate_passed" ] || finish ABORT "B2 requires the B0 gate"
  log "=== B2 cache-off references ==="
  local bad=0
  for f in f16 q80 q4km q3km; do
    if ! pass B2/${f}_order0 $f off 5 25 "$CTRL"; then bad=1; log "B2 $f order0 failed; continuing"; continue; fi
    gate B2_${f}_order0_vs_august_grid "$R/B2/${f}_order0/arm_off" "$REF/aug_grid_${f}_off" 15000 1e-3 || alert "B2_${f}_vs_august" "cache-off pass on this machine differs from the August grid"
    if ! pass B2/${f}_order1 $f off 5 25 "$CTRL" --order-file "$ORD/order_1.json"; then bad=1; log "B2 $f order1 failed; continuing"; continue; fi
    gate B2_${f}_order1_vs_order0 "$R/B2/${f}_order1/arm_off" "$R/B2/${f}_order0/arm_off" 15000 \
      || { alert "B2_${f}_history_dependence" "cache-off output depends on episode order; B1 will collect an off pass per history"; touch "$R/B2/.off_per_history_$f"; }
    touch "$R/B2/.ok_$f"
  done
  [ -f "$R/B2/q4km_order0/arm_off/.validated" ] && note B2_q4km_order0_vs_B0_off_lp5 "$R/B2/q4km_order0/arm_off" "$R/B0/q4km_off_lp5/arm_off"
  do_mirror
  if [ $bad -eq 0 ]; then touch "$R/B2/.done"; log "B2 DONE"; status "B2_DONE"; else block_fail B2 "one or more passes failed"; fi ;;

B1)
  [ -f "$R/B0/.gate_passed" ] || finish ABORT "B1 requires the B0 gate"
  log "=== B1 controlled sweep: canonical + 10 randomized histories x 4 formats ==="
  local bad=0
  for f in f16 q80 q4km q3km; do [ -f "$R/B2/.ok_$f" ] || { bad=1; log "B1: no B2 verdict for $f; cache-off per history undecided (rerun B2, then B1)"; }; done
  for k in 0 1 2 3 4 5 6 7 8 9 10; do
    for f in ${WILLIAMS[$((k % 4))]}; do
      if [ $k -eq 0 ]; then
        if ! pass B1/${f}_order0 $f on 5 12 "$CTRL"; then bad=1; log "B1 $f order0 failed; continuing"; continue; fi
        case $f in f16|q4km|q3km) note B1_${f}_order0_vs_august_controlled "$R/B1/${f}_order0/arm_on" "$REF/aug_ctrl_${f}_on" 1e-3;; esac
        [ $f = q4km ] && note B1_q4km_order0_vs_B0_on_lp5 "$R/B1/q4km_order0/arm_on" "$R/B0/q4km_on_lp5/arm_on"
      else
        if ! pass B1/${f}_order$k $f on 5 12 "$CTRL" --order-file "$ORD/order_$k.json"; then bad=1; log "B1 $f order$k failed; continuing"; continue; fi
      fi
      if [ -f "$R/B2/.off_per_history_$f" ] && [ $k -gt 1 ]; then
        pass B1off/${f}_order$k $f off 5 25 "$CTRL" --order-file "$ORD/order_$k.json" || { bad=1; log "B1off $f order$k failed; continuing"; }; fi
    done
    log "B1 history $k complete"; status "B1_HISTORY_$k"
  done
  do_mirror
  if [ $bad -eq 0 ]; then touch "$R/B1/.done"; log "B1 DONE"; status "B1_DONE"; else block_fail B1 "one or more passes failed"; fi ;;

B1R)
  log "=== B1R fresh-process replicate at the canonical history ==="
  local bad=0
  for f in f16 q80 q3km; do
    if pass B1R/${f}_order0_rep $f on 5 12 "$CTRL"; then
      [ -f "$R/B1/${f}_order0/arm_on/.validated" ] && note B1R_${f}_rep_vs_B1_order0 "$R/B1R/${f}_order0_rep/arm_on" "$R/B1/${f}_order0/arm_on"
    else bad=1; fi
  done
  do_mirror
  if [ $bad -eq 0 ]; then touch "$R/B1R/.done"; log "B1R DONE"; status "B1R_DONE"; else block_fail B1R "a replicate failed"; fi ;;

B3)
  log "=== B3 sentinel-isolated histories ==="
  local bad=0
  for f in f16 q80 q4km q3km; do
    if ! pass B3/${f}_order0 $f on 5 12 "$CTRL" --sentinel-reset; then bad=1; continue; fi
    for k in 1 2; do
      if pass B3/${f}_order$k $f on 5 12 "$CTRL" --sentinel-reset --order-file "$ORD/order_$k.json"; then
        note B3_${f}_order${k}_vs_order0 "$R/B3/${f}_order$k/arm_on" "$R/B3/${f}_order0/arm_on"
      else bad=1; fi
    done
  done
  do_mirror
  if [ $bad -eq 0 ]; then touch "$R/B3/.done"; log "B3 DONE"; status "B3_DONE"; else block_fail B3 "one or more passes failed"; fi ;;

B6)
  log "=== B6 production-default prompt cache under randomized histories ==="
  local bad=0
  for k in 1 2 3 4 5; do
    local label=B6/q4km_default_order$k ok=0 fp6
    fp6=$(data_fp "$label|default-promptcache|$k|$(sha256sum "$ORD/order_$k.json")")
    if [ -f "$R/$label/.validated" ] && grep -qx "$fp6" "$R/$label/.validated"; then log "skip (validated) $label"; continue; fi
    for attempt in 1 2; do
      guard 20; status "RUNNING $label attempt $attempt"
      local stage="$R/_staging/$label"; rm -rf "${stage:?}"; mkdir -p "$stage"
      local tag=B6_order${k}_a$attempt
      start_srv q4km "$tag" || { kill_srv; fail_keep "$label.a$attempt" "$stage"; continue; }
      run_harness "$stage/main" on 1 8 --order-file "$ORD/order_$k.json"; local r1=$?
      run_harness "$stage/repeat" on 1 8 --order-file "$ORD/order_$k.json"; local r2=$?
      kill_harness; kill_srv
      cp "$W/srvlogs/$tag".* "$stage/" 2>/dev/null
      if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] \
         && $TOOLS validate "$stage/main/arm_on" --arm on --lp 1 --order-file "$ORD/order_$k.json" > "$stage/validation_main.json" \
         && $TOOLS validate "$stage/repeat/arm_on" --arm on --lp 1 --order-file "$ORD/order_$k.json" > "$stage/validation_repeat.json"; then
        echo "$fp6" > "$stage/.validated"
        local final="$R/$label"; mkdir -p "$(dirname "$final")"
        if [ -e "$final" ]; then mv -T "$final" "$R/_failed/superseded_B6_order${k}_$(date +%s)" || { log "B6: cannot move aside $final"; fail_keep "$label.a$attempt" "$stage"; continue; }; fi
        mv -T "$stage" "$final" || { log "B6: promote failed"; fail_keep "$label.a$attempt" "$stage"; continue; }; do_mirror
        $TOOLS compare "$final/main/arm_on" "$final/repeat/arm_on" > "$R/checks/B6_order${k}_rerun.json"
        log "OK $label: rerun divergence $("$PY" -c "import json;print(json.load(open('$R/checks/B6_order${k}_rerun.json'))['n_diverged'])")/80"
        ok=1; break; fi
      log "INVALID $label attempt $attempt (rc $r1/$r2)"; fail_keep "$label.a$attempt" "$stage"
    done
    [ $ok -eq 1 ] || { bad=1; log "B6 order $k failed; continuing"; }
  done
  do_mirror
  if [ $bad -eq 0 ]; then touch "$R/B6/.done"; log "B6 DONE"; status "B6_DONE"; else block_fail B6 "one or more histories failed"; fi ;;

B4)
  log "=== B4 reset control: 4 formats x 3 fresh sessions x 100 items ==="
  local bad=0
  for s in 1 2 3; do for f in ${WILLIAMS[$((s % 4))]}; do
    local label=B4/${f}_session$s out="$R/B4/${f}_session$s" ok=0 fp4
    fp4=$(data_fp "$label|reset|$s|$f")
    if [ -f "$out/.validated" ] && grep -qx "$fp4" "$out/.validated"; then log "skip (validated) $label"; continue; fi
    for attempt in 1 2; do
      guard 40; status "RUNNING $label attempt $attempt"
      local stage="$R/_staging/$label"; rm -rf "${stage:?}"; mkdir -p "$stage"
      start_srv $f "B4_${f}_s${s}_a$attempt" $CTRL || { kill_srv; fail_keep "$label.a$attempt" "$stage"; continue; }
      ( cd "$ST" && PYTHONPATH=harness timeout -k 60 7200 "$PY" harness/reset_control.py --model-hf-id "$QW" \
          --served-model repair --base-url http://127.0.0.1:8080/v1 --backend llamacpp --mode toggle \
          --n 100 --item-seed $((100 + s)) --logprobs 5 --out-dir "$stage" ) >> "$W/passes.log" 2>&1 9>&-; local rc=$?
      kill_harness; kill_srv
      cp "$W/srvlogs/B4_${f}_s${s}_a$attempt".* "$stage/" 2>/dev/null
      local v; v=$($TOOLS reset "$stage" --n 100 --lp 5); local vrc=$?; echo "$v" > "$stage/validation.json"
      if [ $rc -eq 0 ] && [ $vrc -eq 0 ]; then
        echo "$fp4" > "$stage/.validated"; mkdir -p "$R/B4"
        if [ -e "$out" ]; then mv -T "$out" "$R/_failed/superseded_B4_${f}_s${s}_$(date +%s)" || { log "B4: cannot move aside $out"; fail_keep "$label.a$attempt" "$stage"; continue; }; fi
        mv -T "$stage" "$out" || { log "B4: promote failed"; fail_keep "$label.a$attempt" "$stage"; continue; }; do_mirror
        log "OK $label: $v"; ok=1; break; fi
      log "INVALID $label attempt $attempt (rc $rc): $(echo "$v" | head -c 400)"; fail_keep "$label.a$attempt" "$stage"
    done
    [ $ok -eq 1 ] || { bad=1; log "B4 $f session $s failed; continuing"; }
  done; done
  do_mirror
  if [ $bad -eq 0 ]; then touch "$R/B4/.done"; log "B4 DONE"; status "B4_DONE"; else block_fail B4 "one or more sessions failed"; fi ;;

B5)
  log "=== B5 single-stream latency, Q4_K_M ==="
  start_srv q4km B5_warmup_untimed $CTRL; kill_srv
  local MODES=("default ctrl off" "ctrl off default" "off default ctrl") bad=0
  for s in 1 2 3; do for m in ${MODES[$((s-1))]}; do
    case $m in
      default) pass B5/s${s}_promptcache_default q4km on 1 8 "" --order-file "$ORD/order_0.json" || bad=1 ;;
      ctrl)    pass B5/s${s}_promptcache_disabled q4km on 1 8 "$CTRL" --order-file "$ORD/order_0.json" || bad=1 ;;
      off)     pass B5/s${s}_recompute q4km off 1 16 "$CTRL" --order-file "$ORD/order_0.json" || bad=1 ;;
    esac
  done; done
  do_mirror
  if [ $bad -eq 0 ]; then touch "$R/B5/.done"; log "B5 DONE"; status "B5_DONE"; else block_fail B5 "one or more passes failed"; fi ;;

B7)  # kernels vs weight values: Q4_K_M weights dequantized to F16, served on the F16 kernels
  [ -f "$R/B0/.gate_passed" ] || finish ABORT "B7 requires the B0 gate"
  log "=== B7 dequantized-weights control ==="
  (cd "$M" && sha256sum -c SHA256SUMS.deq > /dev/null) || { block_fail B7 "dequantized model checksum"; return; }
  local bad=0
  pass B7/q4deq_order0 q4deq off 5 25 "$CTRL" || bad=1
  for k in 0 1 2 3; do
    if [ $k -eq 0 ]; then pass B7/q4deq_order0 q4deq on 5 12 "$CTRL" || bad=1
    else pass B7/q4deq_order$k q4deq on 5 12 "$CTRL" --order-file "$ORD/order_$k.json" || bad=1; fi
  done
  [ -f "$R/B7/q4deq_order0/arm_off/.validated" ] && note B7_deq_off_vs_q4km_off "$R/B7/q4deq_order0/arm_off" "$R/B2/q4km_order0/arm_off"
  [ -f "$R/B7/q4deq_order0/arm_off/.validated" ] && note B7_deq_off_vs_f16_off "$R/B7/q4deq_order0/arm_off" "$R/B2/f16_order0/arm_off"
  do_mirror
  if [ $bad -eq 0 ]; then touch "$R/B7/.done"; log "B7 DONE"; status "B7_DONE"; else block_fail B7 "one or more passes failed"; fi ;;
esac; }

for B in "$@"; do
  [ "$B" != preflight ] && [ -f "$W/PREFLIGHT_OK" ] && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  run_block "$B"
done
incomplete=""; for B in "$@"; do [ "$B" = preflight ] || [ "$B" = B0 ] || [ -f "$R/$B/.done" ] || incomplete="$incomplete $B"; done
if [ "${PHASEB_FINAL:-1}" = 1 ]; then
  if [ -z "$incomplete" ]; then finish DONE "all blocks complete: $*"; else finish INCOMPLETE "blocks with failures:$incomplete"; fi
fi
kill_srv; do_mirror; log "blocks complete (no auto-stop requested): $* ; incomplete:${incomplete:- none}"; status "IDLE_AFTER $*"; CLEAN_END=1
