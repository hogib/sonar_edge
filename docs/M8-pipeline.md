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
  `--polar BEAMSxSAMPLES` raw frames from a file or stdin, or Sound Metrics
  recordings with `--aris [--fit metric|window] [--every N] [--first K]
  [--count N] rec.aris ...` (section 6).
- **Models**: `export_c --out DIR` exports a model without touching the
  shipped `runtime/weights.h`, and now writes class names. `make W=DIR` builds
  another export. The end-to-end model with a mask is `small_360_segdice`
  (M6 §3.4, fold 0) in `runtime/models/small_360_segdice/`. The shipped
  header is still `small_360_cv_f0`, re-exported only to add the class names
  (`diff` shows nothing else changed). `frontend.o` is now built with the
  same `$(WDEF)` as `sonar_edge.o`, so with `make W=DIR` the front-end tables
  take that model's input size (before, they took the shipped header's).

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

- The tank dataset has no raw data, so on it polar mode is tested only on
  back-projected frames. The raw path has since run on real ARIS recordings
  from another dataset (section 6), but those have no labels. Detection
  accuracy on raw sonar data is still unmeasured, and the tank-trained
  models do not transfer to that river data.
- Range is in pixels for the tank data (no metric scale); in metres for
  `.aris` input.
- The front end is scalar; NEON would cut its 1.8 ms.
- The segdice model gives up box mAP for masks (M6 §3.4: pooled 49.8 vs 52.6
  box-only), so `small@360` stays the detection-only build.
- The full-fold on-device run was done afterwards: on the Pi 4 and the Pi 2
  (both kernels), all 439 frames are identical to the laptop, for both models
  and both input modes (`bench/e2e/`).

## 6. Real raw ARIS data (DANS)

**Data.** *Detection of plastic litter using an ARIS sonar* (Radboud
University 2021; DANS, doi:10.17026/dans-xee-n9rq, CC BY 4.0): 71 public
`.aris` recordings of plastic objects and controls in the river Waal.
Downloaded: the 23 files of the 16 April session and 6 of 19 April
(`bench/dans/dans_list.txt`, `download.sh`); 28 distinct recordings used
(`PlasticContainer_Raw_1_195` duplicates `PlasticContainer`). All are ARIS
3000, 128 beams, 15 fps (13.7 for the 4.75 m window). Windows start at
0.70 m and end at 1.86-4.75 m. Up to 200 frames per recording (every N-th),
4,780 frames per model.

**Reader** (`runtime/aris.{c,h}`, from SoundMetrics `aris-file-sdk`
commit 5329f18, MIT, `runtime/beamwidths/`):
- 1024-byte file header, then per frame a 1024-byte header and
  `uint8[samples][beams]`;
- beam 0 is the right-most beam, so the reader flips the order to our
  left-most-first convention;
- per-beam angles come from the SDK's ARIS 3000 tables;
- the range window comes from SampleStartDelay, SamplePeriod and SoundSpeed.

**Validation** (frames 0, 100 and 190 of PlasticContainer):
- against a NumPy projection written from the SDK formulas: max |diff| 1
  grey level, 98.5% of 47,001 px identical;
- against pyARIS: correlation 0.93 raw and 0.984 after a 5x5 blur, best at
  zero shift. The mirrored image gives 0.49, which confirms the beam order.

**Geometry.** `--fit metric` (default) draws the true fan shape, with the
window's far edge on the training fan's far radius (578 px). This gives
m_per_px = window end / 578 (5.3 mm/px for 3.08 m), so ranges are reported
in metres. The apparent object size in pixels therefore depends on the
window and differs from the tank. `--fit window` instead stretches the
window onto the training near and far radii, with ranges in pixels.

**Equality and cost.**
- Pi 4 and laptop give identical output on all 195 PlasticContainer
  frames, both models (timings and the file path aside).
- After landing the reader, `sonar-edge-run` on the 439 test-fold frames
  (network-size and native fan input, both models) is identical to b06dc97
  except the timings.

| Model | Machine | Front end | Network | Decode | Mask | Total ms |
|---|---|---|---|---|---|---|
| small@360 | laptop | 0.10 | 20.30 | 0.11 | - | 20.50 |
| small@360 | Pi 4 | 0.55 | 20.14 | 0.37 | - | 21.06 |
| segdice | laptop | 0.10 | 20.31 | 0.11 | 0.04 | 20.56 |
| segdice | Pi 4 | 0.55 | 21.20 | 0.41 | 0.24 | 22.40 |

10 frames x 20 repeats. Pi 4 at 57.4 C before and 55.0 C after;
`throttled` 0x0.

**Detections** (no labels, so no accuracy). Frames with a detection at
p >= 0.3; almost all detections are Chain, a few Wall:

| Recordings | Window end (m) | Frames | small@360 | segdice |
|---|---|---|---|---|
| Controls, no plastic (2) | 1.87-1.89 | 334 | 133 (40%) | 192 (57%) |
| Containers (6) | 1.86-3.08 | 1,019 | 341 (33%) | 357 (35%) |
| Pieces and rim (8) | 1.89-2.50 | 1,412 | 336 (24%) | 894 (63%) |
| Cups (6) | 1.89 | 1,024 | 460 (45%) | 1,012 (99%) |
| 19 April session (6) | 3.51-4.75 | 991 | 0 | 0 |
| All (28) | | 4,780 | 1,270 (27%) | 2,455 (51%) |

**Interpretation.**
- The models do not transfer from the tank to the river. Detections fire on
  textured bed and vegetation, as often in the controls as with plastic
  (PinAlone_Control: 133/196 frames for small@360, 185/196 for segdice), and
  the rate follows the recording set and window rather than the plastic.
- The objects could not be located by eye, so it is unknown whether any box
  lands on one. The longer-range 19 April recordings give no detections.
- What this validates is the raw-data path on real ARIS data: reader,
  geometry, front end, metric ranges, on-device exactness and cost (front
  end 0.55 ms, about a third of a Pi 4 core at 15 fps in total). It is not a
  measure of detection accuracy on raw data, and not evidence of field
  performance; that needs labelled target-environment recordings and
  fine-tuning.

Figures: `report/fig/dans_{container,control,cup,river}.png`. Data and
scripts: `bench/dans/`. The `.aris` files stay in `data/raw/dans_aris/`
(git-ignored).
