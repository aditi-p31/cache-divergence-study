#!/bin/bash
# Independent watchdog. Stops the pod (after a VERIFIED mirror) when there has
# been no progress for IDLE_MIN minutes, whether or not processes exist (a hung
# orchestrator counts as idle), or when spend since the pod-start ledger exceeds
# HARD_CAP. A HOLD file younger than 3 hours suspends the idle rule. If the mirror
# cannot be verified the pod is left running (data before cost).
W=/root/work; LEDGER=/mnt/persistent/phaseB/T0
IDLE_MIN=${IDLE_MIN:-45}; HARD_CAP_CENTS=${HARD_CAP_CENTS:-2450}; RATE_CENTS=${RATE_CENTS:-77}
exec 8>"$W/watchdog.lock"; flock -n 8 || exit 0
log(){ echo "[$(date -u +%FT%TZ)] watchdog: $*" >> "$W/watchdog.log"; return 0; }
stop_now(){ log "STOP: $1"
  pkill -9 -f '[h]arness/run_episodes.py'; pkill -9 -f '[h]arness/reset_control.py'; pkill -9 -f '[l]lama-server'
  if timeout -s KILL 1500 "$W/mirror.sh" 8>&-; then
    if "$W/stop_pod.sh" "watchdog: $1" 8>&-; then exit 0; fi
    log "stop refused: pod left running; watchdog keeps watching at a long interval"; STOP_REFUSED=1
  else log "mirror NOT verified: leaving the pod running"; touch "$W/MIRROR_FAILED"; fi; }
STOP_REFUSED=0
start=$(date +%s); last=$start
t0=$(cat "$W/T0" 2>/dev/null); [[ $t0 =~ ^[0-9]{10}$ ]] || t0=$start
log "started (idle ${IDLE_MIN} min on progress, hard cap ${HARD_CAP_CENTS} cents)"
while true; do
  if [ "$STOP_REFUSED" = 1 ]; then sleep 1800 8>&-; else sleep 120 8>&-; fi
  now=$(date +%s)
  v=$(timeout -s KILL 10 cat "$LEDGER" 2>/dev/null); [[ $v =~ ^[0-9]{10}$ ]] && t0=$v
  cents=$(( (now - t0) * RATE_CENTS / 3600 ))
  [ $cents -ge $HARD_CAP_CENTS ] && stop_now "hard spend cap: ~\$$((cents/100))"
  newest=$(find "$W/results" "$W/srvlogs" "$W/STATUS" "$W/passes.log" "$W/phaseB.log" -newermt "@$last" -type f 2>/dev/null | head -1)
  [ -n "$newest" ] && last=$now
  hold=0; [ -f "$W/HOLD" ] && [ $(( now - $(stat -c %Y "$W/HOLD") )) -lt 10800 ] && hold=1
  if [ $hold -eq 0 ] && [ $(( now - last )) -gt $(( IDLE_MIN * 60 )) ]; then
    stop_now "no progress for $(( (now - last) / 60 )) min"; last=$now
  fi
done
