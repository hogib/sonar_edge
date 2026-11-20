#!/usr/bin/env bash
# 5-fold x 3-seed CV of the polar-input `small` detector (same recipe as small@360: 150 epochs, strong aug).
#   bench/polar/run_polar_cv.sh NAME BEAMS      e.g.  small_polar 128   |   small_polar192 192
# Fold k test, fold (k+1) mod 5 validation. Runs: runs/NAME_cv_f{k} (seed 0), runs/NAME_s{1,2}_cv_f{k}.
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 LC_ALL=C.UTF-8
name=$1; beams=$2
for seed in 0 1 2; do
  tag=$([ "$seed" = 0 ] && echo "${name}_cv" || echo "${name}_s${seed}_cv")
  for k in 0 1 2 3 4; do
    v=$(( (k + 1) % 5 )); out=runs/${tag}_f$k
    if [ -f "$out/results.json" ]; then echo "skip $out (done)"; continue; fi
    uv run sonar-edge train --config small --polar --polar-beams "$beams" --epochs 150 --aug strong \
      --test-fold "$k" --val-fold "$v" --seed "$seed" --gpu-mem-frac 0.04 --out "$out" 2>&1 \
      | grep --line-buffered -v Warning > "$out.log"
    grep -E "^(config|peak|test: mAP)" "$out.log"
  done
done
echo "ALL DONE $name"
