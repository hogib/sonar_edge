# M8: end-to-end C pipeline (sonar frame in, detections + mask out)

Done 2026-11-20. Closes the "scope of end-to-end" limitation of the project
report: until now the C runtime started from a 360x240 frame that Python had
resized, and returned boxes in network pixels only.

## 1. Pipeline

```
fan image 320x480 ─┐
                   ├─ front end (one LUT pass) ─ network (int8, one pass) ─┬─ decode: boxes, classes, probabilities
raw polar B x S ───┘   runtime/frontend.c          runtime/sonar_edge.c    └─ segmentation labels (same head map)
                                                                              │
                         boxes -> fan-image pixels -> range / bearing ────────┴─ JSON line per frame (+ mask PGM, overlay PPM)
```

- **Front end** (`runtime/frontend.{c,h}`): built once, then one table-driven
  integer pass per frame. Two modes:
  - *fan*: the Cartesian fan image the sonar software and the dataset
    produce, resized to the network input;
  - *polar*: raw beam data, `n_samples` rows (range, near to far) x `n_beams`
    columns, rendered straight to the network input with the fan geometry
    (uniform beam angles over the field of view, or a per-beam table).
- **Single pass** (`se_process`, `se_decode`): one forward pass gives both the
  detections (same decoder, threshold and top-K as `se_detect`) and the label
  map. Before this, `sonar-detect -s` ran the network twice.
- **Sonar coordinates**: `se_fe_to_fan` maps network pixels back to fan-image
  pixels; `se_fan_to_polar` gives range and bearing (0 = straight ahead,
  positive right) of each box centre.
- **CLI** `sonar-edge-run`: one JSON line per frame, with per-stage times,
  detections (class name, logit, probability, box in network and fan pixels,
  range, bearing) and per-class mask cell counts. `-m DIR` writes label PGMs;
  `-o DIR` writes an overlay PPM (boxes above `--show P`, default 0.3, and the
  mask on the fan image). Input is PGM files at network size or fan size, or
  `--polar BEAMSxSAMPLES` raw frames from a file or stdin.
- **Models**: `export_c --out DIR` exports a model without touching the
  shipped `runtime/weights.h`, and now writes class names. `make W=DIR` builds
  another export. The end-to-end model with a mask is `small_360_segdice`
  (M6 §3.4, fold 0) in `runtime/models/small_360_segdice/`. The shipped
  header is still `small_360_cv_f0`, re-exported only to add the class names
  (`diff` shows nothing else changed).

## 2. Fan geometry of the dataset

Fitted to the fan edges and arcs of all 1,868 frames (`se_fan_watertank`):

| Quantity | Value |
|---|---|
| Apex (u, v), px | (160.0, 602.7), below the image |
| Field of view | 30.0 deg (fit 30.07; ARIS 3000 spec 30) |
| Near / far radius, px | 138.6 / 578.0 (residuals <= 1 px) |
| Metres per pixel | unknown (0): the dataset documents no range setting, so ranges are in pixels |

## 3. Verification

| Check | Result |
|---|---|
| Fan mode vs the Python resize (Pillow BOX, 22-bit fixed point, horizontal then vertical pass) | bit-identical, 1,868 / 1,868 frames, at 360x240 and 240x160 |
| 240x160 path (training uses an integer block mean in float32 with round-half-to-even, not Pillow) | separate exact path, bit-identical |
| `make test` (golden head map) | passes, shipped model and segdice |
| `make test-frontend` (stored frames, geometry round-trips, polar constant frame) | passes |
| `sonar-edge-run` vs `sonar-detect`, full test fold 0, network-size input | identical: 25,007 lines (small@360), 26,571 (segdice); masks identical |
| Native 320x480 fan input vs pre-resized PGM | identical detections and masks |
| segdice box mAP, C vs Python int8 (test fold 0) | 57.09 vs 57.08 (tied-score ordering, as in M4) |
| Pi 4 vs laptop, 10 fan frames | identical output |
| Pi 4 `sonar-detect` after the `se_decode` refactor | 20.07-20.28 ms (20.3 before) |

**Polar mode** cannot be checked against real data: the dataset holds only
fan images. Test polar frames were made by sampling the fan images back onto
a beam x range grid (`src/sonar_edge/polar.py`) and then rendered by the C
front end. On test fold 0, shipped model (single fold, single seed):

| Polar input | Mean abs pixel diff vs fan path (inside the fan) | mAP |
|---|---|---|
| 128 beams (ARIS 3000) | 4.2 | 63.17 |
| fan path (reference) | 0 | 63.81 |

Per-class AP moves by up to about 10 points on the rare classes. The loss
comes from resampling the image twice (fan -> polar -> network grid), so
real polar data, resampled only once, should do no worse. This is
not shown.

## 4. Cost (Pi 4, one core, ms per frame)

10 frames x 20 repeats x 3; ended at 58.4 C, `throttled=0x0`.

| Model / input | Front end | Network | Decode | Seg | Total |
|---|---|---|---|---|---|
| small@360, fan | 1.79 | 19.77 | 0.36 | - | 21.91 |
| small_360_segdice, fan | 1.77 | 20.99 | 0.39 | 0.24 | 23.38 |
| small_360_segdice, polar 128x512 | 1.77 | 20.82 | 0.38 | 0.24 | 23.21 |

The front end adds about 9% (scalar loops, not vectorised), and the mask
about 1%. Front-end tables: 873 KB static at 360x240 (one sample per output
pixel in polar mode; `-DSE_FE_PTS=4` for 2x2 sampling: mAP 63.27, 2.43 MB).

## 5. Limitations and open items

- No real raw ARIS data: polar mode has been tested only on back-projected
  frames. A `.aris` file reader needs recordings.
- Range is reported in pixels (no metric scale for the dataset).
- The front end is scalar; NEON would cut its 1.8 ms.
- The segdice model gives up box mAP for masks (M6 §3.4: pooled 49.8 vs 52.6
  box-only), so `small@360` stays the detection-only build.
- The full-fold run on the Pi was not repeated for the new CLI; its network
  and decoder are the code verified in M5/M7, and 10 frames matched.
