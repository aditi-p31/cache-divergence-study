#!/bin/bash
# Detached grace timer started by the orchestrator after DONE/ABORT. Waits the
# grace period (and while a young HOLD exists), then stops the pod only if no
# orchestrator holds the run lock and the mirror verifies.
W=/root/work; GRACE_MIN=${1:-45}; REASON=${2:-unspecified}
sleep $((GRACE_MIN*60))
while [ -f "$W/HOLD" ] && [ $(( $(date +%s) - $(stat -c %Y "$W/HOLD") )) -lt 10800 ]; do sleep 300; done
flock -n "$W/phaseB.lock" true || exit 0
[ -f "$W/MIRROR_FAILED" ] && exit 0
"$W/mirror.sh" || { touch "$W/MIRROR_FAILED"; exit 0; }
"$W/stop_pod.sh" "autostop after: $REASON"
