# M10: detection on polar (beam x range) input

Done 2026-11-20 on the GPU server. Results in `bench/polar/summary.json`;
runs under `runs/small_polar*_cv*` (git-ignored); code in
`src/sonar_edge/polar_data.py` and `train.py --polar --polar-beams`.

## 1. Question

Should the network run on the sonar's native geometry instead of the fan
image? In polar form an object's extent along the range axis does not
depend on its bearing, the black corners of the fan disappear, and the
front end (0.5-1.8 ms of about 22 ms, M8) would not be needed.

## 2. Method

- **Data.** Each 320 x 480 fan frame of the dataset is sampled back onto a
  beam x range grid. The geometry is the fitted fan (`se_fan_watertank`): a
  single bilinear resampling of the full-resolution image, with 440 range
  bins (one per fan pixel along the radius, divisible by the stride of 8)
  and 128 or 192 beams. These are back-projected frames, not raw ARIS data:
  the dataset has none (M8).
- **Boxes.** For each ground-truth box, the class's mask pixels inside the box
  are mapped to polar coordinates, and the polar box is their bounding box.
  All 3,882 boxes had mask pixels; 6 boxes smaller than one bin were dropped.
  - Check: a box mapped to polar and back overlaps the original with median
    IoU 0.84 (3.1% below 0.5).
  - Used as perfect detections, the round-tripped boxes score 98.9 fan-space
    mAP, so the conversion costs at most about 1 point.
- **Model and training.** `small`, same recipe as small@360: 150 epochs, strong
  augmentation, batch 16, lr 2e-3, seeds 0-2, the scene folds of M1.
  Horizontal flip becomes a beam flip, which is still physical. Zoom and shift
  are kept but are no longer true geometric transforms in polar.
- **Variants.**
  - 440 x 128: native beam count, 51.1 M MACs.
  - 440 x 192: compute-matched, 76.7 M MACs against 78.4 M for small@360;
    the extra beams are sampled from the fan image.
- **Scoring.** Polar detections are mapped back to the fan image and scored
  with the same evaluator and pooling as small@360, against the original fan
  ground truth. The paired comparison uses the 15 fold-seed pairs.

## 3. Results

| Model | MACs | Pooled fan-space mAP (3 seeds) | Per fold (seed mean) | Paired diff vs small@360 | Better / worse |
|---|---|---|---|---|---|
| small@360, fan image | 78.4 M | **52.6 +- 1.9** | 61.0 65.1 52.2 48.2 67.5 | - | - |
| polar 440 x 128 | 51.1 M | 45.8 +- 2.5 | 55.8 60.8 45.9 34.8 54.3 | -8.5 +- 5.1 | 1 / 14 |
| polar 440 x 192 | 76.7 M | 47.1 +- 1.0 | 55.2 67.8 45.8 35.5 57.3 | -6.5 +- 5.8 | 2 / 13 |
| polar 440 x 192, int8 | 76.7 M | 46.9 +- 1.3 | 57.5 66.5 46.1 35.9 57.2 | - | - |

Scored in polar space against the polar boxes, the models reach 50.1
(128 beams) and 50.7 (192 beams), so the box conversion is not the cause.

Per-class AP, fan space, seed mean (small@360 / polar 192):
- **Wall** drops most: 82.2 / 64.3.
- **Chain** 39.6 / 34.9, **Drink-carton** 63.1 / 58.8, **Shampoo-bottle**
  33.3 / 19.5.
- **Bottle** is level: 85.4 / 87.2.

## 4. Conclusion

At matched compute, polar input loses 6.5 points pooled and is worse in 13 of
15 fold-seed pairs. The detector keeps the fan-image input; the front end
costs little (M8).

Two caveats limit the conclusion to "not better on these data with this
recipe":
- The polar frames are interpolated twice (beams -> fan -> polar). Real raw
  polar data would be interpolated once.
- Zoom and shift augmentation are not geometrically faithful in polar.

Testing native polar input properly needs labelled raw ARIS data.
