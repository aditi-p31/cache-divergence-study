#!/bin/bash
# Copies results and diagnostics to the persistent volume and VERIFIES the copy:
# every raw_requests.jsonl, .validated, summary.json and run_meta.json under
# results must exist in the mirror with an identical sha256. Files verified
# before (same local sha) are not re-hashed on the network volume.
# Exit 0 only when every such file is verified (or there is nothing to copy).
W=/root/work; R=$W/results; CACHE=$W/.mirror_verified
POD="$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=' | cut -d= -f2-)"
MIR=/mnt/persistent/phaseB/run_${POD:-unknown}
mountpoint -q /mnt/persistent || { echo "persistent volume not mounted" >&2; exit 1; }
timeout -s KILL 60 mkdir -p "$MIR/results" "$MIR/diag" || exit 1
touch "$CACHE"
[ -d "$R" ] || exit 0
timeout -k 10 900 rsync -a --exclude '_staging/' "$R/" "$MIR/results/" || exit 1
for f in phaseB.log passes.log STATUS watchdog.log stop_pod.log setup.log; do [ -f "$W/$f" ] && timeout -s KILL 60 cp "$W/$f" "$MIR/diag/" 2>/dev/null; done
[ -d "$W/srvlogs" ] && timeout -k 10 300 rsync -a "$W/srvlogs/" "$MIR/diag/srvlogs/" 2>/dev/null
timeout -s KILL 120 sync -f "$MIR" 2>/dev/null
cd "$R" || exit 1
find . -type f \( -name raw_requests.jsonl -o -name .validated -o -name summary.json -o -name run_meta.json \
     -o -name episode_meta.json -o -name sentinel.jsonl -o -name model_results.json -o -name validation.json -o -name '*.startup_s' \) \
     -not -path "./_staging/*" -print0 | while IFS= read -r -d '' f; do
  s=$(sha256sum "$f" | cut -d' ' -f1)
  grep -qxF "$s $f" "$CACHE" && continue
  [ -f "$MIR/results/$f" ] || { echo "missing in mirror: $f" >&2; exit 1; }
  m=$(timeout -k 5 300 sha256sum "$MIR/results/$f" | cut -d' ' -f1)
  if [ "$s" != "$m" ]; then   # rsync quick-check can skip a same-size rewrite: copy directly, verify again
    timeout -s KILL 120 cp -f "$f" "$MIR/results/$f" 2>/dev/null; timeout -s KILL 60 sync -f "$MIR" 2>/dev/null
    m=$(timeout -k 5 300 sha256sum "$MIR/results/$f" | cut -d' ' -f1)
    [ "$s" = "$m" ] || { echo "sha mismatch after direct copy: $f" >&2; exit 1; }
  fi
  echo "$s $f" >> "$CACHE"
done || exit 1
exit 0
