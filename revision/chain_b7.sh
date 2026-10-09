#!/bin/bash
# Runs B7 on the pod as soon as the main orchestrator exits, so the follow-on block
# does not depend on the operator's laptop being awake. B7 starts only when the main
# run ended DONE or INCOMPLETE (never after ABORT). The new orchestrator invocation
# kills the auto-stop timer that the main run armed. Opt out: touch $W/NO_CHAIN.
W=${CHAIN_W:-/root/work}; MAIN_PID=$1
[[ $MAIN_PID =~ ^[0-9]+$ ]] || { echo "usage: chain_b7.sh <main orchestrator pid>"; exit 2; }
clog(){ echo "[$(date -u +%FT%TZ)] chain: $*" >> "$W/phaseB.log"; }
while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 20; done
sleep 3
st=$(cut -d' ' -f2 "$W/STATUS" 2>/dev/null)
[ -f "$W/NO_CHAIN" ] && { clog "NO_CHAIN present; B7 not started (main ended '$st')"; exit 0; }
case "$st" in
  DONE|INCOMPLETE) clog "main run ended '$st'; starting B7" ;;
  *) clog "main run ended '$st'; B7 not started"; exit 0 ;;
esac
[ -n "$CHAIN_DRY" ] && { clog "dry run: would start B7"; exit 0; }
cd "$W" && PHASEB_FINAL=1 ./pod_phaseB.sh B7 < /dev/null >> "$W/chain_b7.out" 2>&1
clog "B7 orchestrator exited rc=$?"
