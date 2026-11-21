# Leakage experiment: random image-level split vs scene-grouped folds

**Question.** How much does an image-level random split, the protocol of the published
work on this dataset, inflate detection scores compared with our scene-grouped 5-fold CV?
Measured with our own code and models, so that the split is the only thing that differs.

## Method

- **Random split** (`splits/random_folds_v1.json`, `sonar_edge.leakage make`).
  - All 1,868 frames are assigned to five folds by `numpy.random.default_rng(0).permutation`, using the fold sizes of the grouped split (439, 388, 383, 230, 428).
  - Every train, validation and test set therefore has the same size as in the grouped CV. The file reproduces identically on the laptop and the server.
  - The scene ids of the grouped split are copied into the file so that overlap can be measured.
- **Protocol unchanged.**
  - Fold k is the test set, fold (k+1) mod 5 the validation set.
  - The training recipes are the same:
    - Det-S: `small`, scale 4/3, 150 epochs, strong aug;
    - Det-W: `small_wide`, same settings;
    - YOLOv8n: COCO-pretrained, 60 epochs, the `yolo_run.py` recipe.
  - 3 seeds each, and the same evaluator: pooled VOC AP@0.5 over 11 classes, mAP without Wall, and the MCNN-task accuracy / F1.
  - Comparisons are paired by seed: same seed and recipe, only the split differs.
- **Plumbing.**
  - Training reads the fold assignment through `split.SPLIT_FILE`. The environment variable `SONAR_EDGE_SPLIT=<file>` overrides it for every consumer (dataset, fold export, `sonar-edge cv`). Unset, everything behaves as before.
  - The YOLO lists go to `data/baselines/yolo_random/` (`folds --suffix _random`).
  - YOLO runs are named `rand_yolo_pretrained_s<seed>_f<k>` (`yolo_run.py --data-subdir yolo_random --prefix rand_yolo`).
  - Detector runs are `runs/rand_small_360[_s<seed>]_cv_f<k>` and `runs/rand_small_wide_360[_s<seed>]_cv_f<k>`.
  - Scoring passes each split file explicitly (`adapters.score(..., split_file=)`), so `leakage score` needs no environment variable.

## Scene overlap (`overlap.json`)

This table shows, for each test fold, the share of test frames whose reconstructed scene also occurs in the training folds:

| Split | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| Scene-grouped | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Random | 90.2% | 94.1% | 93.0% | 92.2% | 96.0% |

## Running

On the server, the tmux session `sb_leak` runs `bench/leakage/run_leak.sh`.
- It waits until the baseline sessions (`sb_yolo_scr`, `sb_ssd_scr`, `sb_nano_pre`, `sb_nano_scr`) have exited.
- It then trains Det-S (3 seeds), YOLOv8n pretrained (3 seeds) and Det-W (3 seeds) in sequence.
- The log is `bench/leakage/logs/run_leak.log`; `logs/leak.done` marks completion.

Collect and score on the laptop:

```bash
rsync -a --include 'rand_*/' --include 'rand_*/pred_test.json' --include 'rand_*/best.pt' \
  --include 'rand_*/results.json' --include 'rand_*/info.json' --include 'rand_*/weights/' \
  --include 'rand_*/weights/best.pt' --include 'rand_*_cv.json' --exclude '*' \
  server:Projeler/oguz/sonar_edge/runs/ runs/
rsync -a server:Projeler/oguz/sonar_edge/bench/leakage/logs/ bench/leakage/logs/
uv run --extra baselines python -m sonar_edge.leakage score    # -> bench/leakage/summary.json
```

`score` also re-scores the grouped runs with the same code. They reproduce the paper's values: Det-S 52.6 ± 1.9, Det-W 56.9 ± 2.7, YOLOv8n pretrained 62.8 ± 1.8 (pooled mAP, 11 classes).

## Results

Pending.
