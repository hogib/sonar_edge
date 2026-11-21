#!/usr/bin/env bash
# Pull baseline runs from the server every 20 min; score once all six families have 3 complete seeds.
cd "$(dirname "$0")/../.."
while true; do
  uv run --extra baselines python -m sonar_edge.baselines.collect pull > bench/baselines/logs/collect_pull.log 2>&1 || echo "pull failed $(date +%H:%M)" >> bench/baselines/logs/collect_pull.log
  n=$(uv run --extra baselines python -c "
from sonar_edge.baselines.collect import FAMILIES, complete_seeds
print(sum(len(complete_seeds(f)) == 3 for f in FAMILIES))" 2>/dev/null || echo 0)
  echo "$(date +%H:%M) families complete: $n/6" >> bench/baselines/logs/collect_progress.log
  if [ "$n" = "6" ]; then
    uv run --extra baselines python -m sonar_edge.baselines.collect score > bench/baselines/logs/collect_score.log 2>&1
    echo "SCORED $(date +%H:%M) exit $?"; exit 0
  fi
  ping -c 1200 -i 1 -q 127.0.0.1 > /dev/null 2>&1   # ~20 min pause
done
