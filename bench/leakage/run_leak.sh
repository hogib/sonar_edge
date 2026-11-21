#!/usr/bin/env bash
# Leakage experiment (sonar_edge.leakage): train Det-S, YOLOv8n (pretrained) and Det-W on the random
# image-level folds, 3 seeds each, with the recipes of the scene-grouped CV. Runs on the GPU server
# after the baseline sessions have finished:
#   tmux new-session -d -s sb_leak 'bash bench/leakage/run_leak.sh > bench/leakage/logs/run_leak.log 2>&1'
set -euo pipefail
cd "$(dirname "$0")/../.."
export LC_ALL=C.UTF-8 OMP_NUM_THREADS=4 PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
mkdir -p bench/leakage/logs

# 1. wait for the baseline sessions (exact tmux names; no pgrep, so no self-match)
for s in sb_yolo_scr sb_ssd_scr sb_nano_pre sb_nano_scr; do
  while tmux has-session -t "=$s" 2>/dev/null; do sleep 300; done
  echo "$(date '+%F %H:%M') $s finished"
done
echo "$(date '+%F %H:%M') start"

# 2. random folds: regenerate and check against the synced file; YOLO lists under data/baselines/yolo_random
export SONAR_EDGE_SPLIT=splits/random_folds_v1.json
uv run --extra baselines python -m sonar_edge.leakage make
uv run --extra baselines python -m sonar_edge.leakage diag
uv run --extra baselines python -m sonar_edge.baselines.folds --suffix _random

# 3. Det-S, YOLOv8n pretrained, Det-W (recipes as in docs: small/small_wide, scale 4/3, 150 epochs, strong aug)
for seed in 0 1 2; do
  name=rand_small_360_cv; [ "$seed" = 0 ] || name=rand_small_360_s${seed}_cv
  ./run_cv.sh "$name" small 4/3 150 strong "$seed" --gpu-mem-frac 0.15 > "bench/leakage/logs/$name.log" 2>&1
  echo "$(date '+%F %H:%M') $name done"
done
.venv-yolo/bin/python -u src/sonar_edge/baselines/yolo_run.py --init pretrained --epochs 60 \
  --data-subdir yolo_random --prefix rand_yolo > bench/leakage/logs/rand_yolo_pretrained.log 2>&1
echo "$(date '+%F %H:%M') rand_yolo_pretrained done"
for seed in 0 1 2; do
  name=rand_small_wide_360_cv; [ "$seed" = 0 ] || name=rand_small_wide_360_s${seed}_cv
  ./run_cv.sh "$name" small_wide 4/3 150 strong "$seed" --gpu-mem-frac 0.15 > "bench/leakage/logs/$name.log" 2>&1
  echo "$(date '+%F %H:%M') $name done"
done
echo "$(date '+%F %H:%M') all done" | tee bench/leakage/logs/leak.done
