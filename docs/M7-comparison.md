# M7: on-device 5-fold CV and MCNN under the same protocol

Done 2026-11-11, after review of `report/deployment.pdf`. Long-form version:
`report/comparison.pdf`.

## 1. On-device 5-fold CV

**Problem.** The deployment report quoted 63.8 mAP on the Pi next to 52.6 pooled
5-fold mAP. 63.8 is one model (small@360, seed 0) on its own test fold 0
(439 frames). Fold mAP for that model ranges 44.7-66.2, so the two figures are not
comparable.

**Method.** All 30 fold models (small@360 and small_wide@360, seeds 0-2, folds
0-4) were exported (`export_c`), built on the Pi 4 (NEON 4-pixel kernel, -O2),
golden-tested (30/30 pass), and run over their own test folds. Scoring:
`python -m sonar_edge.c_eval runs/<r> --test-fold k --val-fold (k+1)%5 --pred
dets.txt --tag pi4`, which now also writes `pred_test_pi4.json`. Pooling:
`sonar-edge cv` has a third variant `pi4`, and `sonar-edge seeds` aggregates it.

**Results** (pooled mAP@0.5 over 1,868 frames, mean +- sd of 3 seeds):

| Model | float | int8 Python | int8 Pi 4 | Pi per seed |
|---|---|---|---|---|
| small@360 | 52.6 +- 1.9 | 52.3 +- 1.7 | **52.3 +- 1.7** | 53.7 52.8 50.4 |
| small_wide@360 | 56.9 +- 2.7 | 56.7 +- 2.6 | **56.7 +- 2.7** | 58.8 53.7 57.7 |

Per-fold mean of seeds (Pi): small 59.5 / 64.5 / 52.0 / 47.5 / 66.0; small_wide
66.5 / 66.6 / 58.1 / 52.6 / 69.9. Pi and Python int8 differ by <= 0.07 per fold,
because the C program's detection-emission threshold (logit -4) is lower; the head
maps are bit-identical.

## 2. MCNN on the scene folds (accuracy)

`src/sonar_edge/mcnn_cv.py`.
- **Crops.** All 2,826 non-Wall boxes, cut at native resolution from the
  `watertank-segmentation` frames; max 299x253, zero-padded to 305x305. These are
  not the 2,364 crops of `watertank-cropped`.
- **Split and selection.** Fold of the parent frame; test k, val (k+1)%5, train
  the rest; best-val-accuracy epoch.
- **Training.** sonar_cnn recipe: AdamW 1e-3, wd 1e-4, cosine, 30 epochs, batch
  32; seeds 0-2; about 15 s per fold on the RTX 5090.

Detector on the same task (`src/sonar_edge/gt_cls.py`):
- **Prediction.** For each non-Wall GT box, the class of the highest-scoring
  detection with IoU >= 0.5.
- **Errors.** No such detection counts as "missed" (an error); a Wall label is
  an error.
- **Detections used.** The Pi detections of section 1.

| Model | boxes given | accuracy % | macro-F1 % | missed % | acc per fold (seed mean) |
|---|---|---|---|---|---|
| MCNN | yes | 77.7 +- 0.5 | 71.4 +- 1.7 | - | 84.5 89.6 47.0 83.6 78.5 |
| small@360 | no | 74.2 +- 1.3 | 70.4 +- 3.0 | 7.4 | 83.8 85.1 55.4 62.1 70.8 |
| small_wide@360 | no | 77.6 +- 2.4 | 74.4 +- 2.7 | 6.3 | 85.7 88.8 63.0 68.7 70.3 |

Per-class recall, in the order MCNN / small / small_wide:

| Class | MCNN | small | small_wide |
|---|---|---|---|
| Bottle | 86.5 | 92.0 | 90.7 |
| Can | 66.7 | 38.6 | 38.2 |
| Chain | 82.8 | 68.6 | 65.2 |
| Drink-carton | 86.9 | 75.5 | 89.0 |
| Hook | 94.0 | 94.3 | 96.7 |
| Propeller | 65.7 | 63.6 | 63.5 |
| Shampoo-bottle | 36.6 | 67.0 | 54.1 |
| Standing-bottle | 71.8 | 28.7 | 59.0 |
| Tire | 93.6 | 92.6 | 97.5 |
| Valve | 32.1 | 51.0 | 62.8 |

Int8-Python detections give 74.1 / 77.4 accuracy (missed 8.5-9.0%, higher
threshold).

**Discussion.**
- **Leak.** MCNN falls from 99.8% (random split, sonar_cnn) to 77.7% under
  scene folds.
- **Detector vs MCNN.** The detectors, not given the boxes, are on a par with
  it; small_wide matches its accuracy and exceeds it in macro-F1.
- **Fold 3.** MCNN's advantage is concentrated in fold 3 (61 of 65
  standing-bottles; their box shape is distinctive, and box size alone gave 98%
  in sonar_cnn).
- **Fold 2.** MCNN's weakness is fold 2: 158 of 240 valves, so it trains on 69
  valves, and fold-2 valve recall is 17.5%.

## 3. Speed under one engine (Pi 4)

ONNX Runtime 1.31.1, float32, CPU EP. 10 warm-up runs, then 3 x 50 timed runs
(repeats within 1.5%). Peak RSS from VmHWM; the interpreter + ORT baseline is
45.5 MB. MCNN cost per frame = 1.51 crops (2,826 objects / 1,868 frames; batch 2
costs exactly 2x batch 1).

| Model | params | MMAC/frame | ORT 1 thr ms | ORT 4 thr ms | C int8 ms | RSS ORT / C MB |
|---|---|---|---|---|---|---|
| MCNN per crop | 2,797,706 | 234.7 | 66.5 | 24.7 | - | 75.5 / - |
| MCNN per frame | | 355 | 100.6 | 37.4 | - | |
| small@360 | 42,463 | 78.4 | 22.4 | 8.8 | 20.3 | 58.1 / 2.3 |
| small_wide@360 | 135,359 | 203.5 | 48.7 | 17.2 | 46.5 | 58.7 / 2.6 |

- **Same engine.** The deployed detector is 4.5x faster per frame than MCNN
  (3.0x faster than one MCNN crop), and MCNN still needs a proposal stage.
- **Runtime vs ORT speed.** The int8 C runtime is only 10% faster than ORT
  float on one core.
- **What the runtime adds.** Its value is 25x less memory, no dependencies and
  bit-exactness, not speed.
- **Correction to M5 §3.5.** 490 M MACs used 2.08 boxes/frame including Wall;
  MCNN classifies 1.51 objects/frame, so 355 M.

## 4. External data (surveyed, not used)

The ARIS sets in `mvaldenegro/marine-debris-fls-datasets`:
- **quarry-fullsize:** about 7,200 real-water frames, no boxes; useful for a
  qualitative test, or after hand-labelling a subset.
- **turntable-cropped:** crops, 12 classes partly overlapping; useful as an
  external classifier test.

UATD (Xie et al., Sci. Data 2022): Gemini 1200ik, about 9,000 frames, only tyre
overlaps; useful only class-agnostically or for fine-tuning.

## 5. Open

- Pi 5 (`udot` kernel), when a board is available.
- Qualitative run on the quarry frames.
- Raising the scores: deploy small_wide (+4.4), distillation from wide@480, and
  rare-class sampling or copy-paste for can / standing-bottle / valve.
