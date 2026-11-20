# E2E on-device verification (b06dc97) — handoff, 2026-11-20

## Status: DONE. Nothing is still running on either board.

All runs finished before the reboot. Only the write-up is left (docs/M8-pipeline.md §3/§4, not started; parent's job).

## Results (test fold 0, 439 native 320x480 fan frames + same frames back-projected to 128x439 polar)

**Identity vs the laptop x86 reference:** JSON (with "ms" removed, source basename) and mask PGMs matched for 439/439 frames on every board, model, mode and variant. This covers Pi 4 (AArch64 default flags) and Pi 2 (ARMv7, neon1 and neon4).

**Tests:** `make test` and `make test-frontend` pass on all builds.

**mAP is identical on every board:**

| Model | Fan | Polar |
|---|---|---|
| small360 | 63.81 | 63.17 |
| segdice | 57.09 | 52.32 |

For segdice, polar is -4.8; the laptop shows the same drop, so it is not a device effect.

**Mean ms per frame (front end / network / decode / seg / total):**

| Board | Model, mode | Front end | Network | Decode | Seg | Total |
|---|---|---|---|---|---|---|
| Pi 4 | small360 fan | 1.40 | 20.14 | 0.37 | – | 21.90 |
| Pi 4 | small360 polar | 0.53 | 20.07 | 0.36 | – | 20.96 |
| Pi 4 | segdice fan | 1.40 | 21.20 | 0.40 | 0.24 | 23.24 |
| Pi 4 | segdice polar | 0.54 | 21.56 | 0.40 | 0.24 | 22.74 |
| Pi 2 neon1 | small360 fan | 5.44 | 121.91 | 1.15 | – | 128.50 |
| Pi 2 neon1 | small360 polar | 2.85 | 121.66 | 1.15 | – | 125.65 |
| Pi 2 neon1 | segdice fan | 5.44 | 127.71 | 1.27 | 2.15 | 136.56 |
| Pi 2 neon1 | segdice polar | 2.82 | 127.32 | 1.27 | 2.15 | 133.56 |

Pi 2 neon4 totals: 132.36 / 131.09 / 140.13 / 139.01 (slower than neon1 for both models).

**Thermal:** Pi 4 ran at 53–58 C and Pi 2 at 41–43 C; `throttled` was 0x0 throughout.

## Files

- `run_e2e.py`: board runner. `compare_e2e.py`: identity check and mAP scoring (c_eval tags `e2e_<board>_<model>_<variant>_<mode>`).
- `e2e_summary.json`: everything combined. `run_pi4.log`, `run_pi2.log`.
- `outputs/<board>/`: summary_*.json and compare_*.json. The jsonl files were dropped (large); `outputs/order.json` holds the frame order.

**Raw outputs still on the boards:**
- Pi 4: `~/sonar_edge_e2e_full/{out,masks}`
- Pi 2: `~/sonar_edge_e2e_full/{out,masks}`

## To regenerate (only if needed)

Inputs come from the PNGs via `sonar_edge.polar.to_polar`. The script is in this session's transcript; it writes `in/fan/<idx>.pgm`, `in/polar.u8` and `in/order.json`. Then:

```
python3 run_e2e.py WORK LABEL [--cflags ...] [--variants neon1 neon4] [--cool 50]
uv run python bench/e2e/compare_e2e.py REF WORK LABEL --score
```
