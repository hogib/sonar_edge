# sonar-edge: plan

End-to-end marine-debris detection on forward-looking sonar frames, deployed as a
dependency-free C11 runtime on a Raspberry Pi. Successor to `../sonar_cnn`, which
reimplemented the MCNN paper and showed that its ~99.9% rests on two shortcuts
(crop size and near-duplicate frames). This project builds the system the paper
only gestures at, and measures what it costs.

## Decisions (agreed 2026-11-02)

| Decision | Choice | Why |
|---|---|---|
| Pipeline | One fully convolutional network on the whole 320x480 frame, anchor-free centre-heatmap head (CenterNet-style) producing per-class heatmap + box size at stride 4 | One graph to port; fixed input so memory is fully static; box size becomes a regressed output, so the crop-size leak cannot exist |
| Language | C11, no dependencies, no malloc in the inference path | Compiles everywhere; the runtime needs ~6 ops; integer-only so results are bit-identical across machines |
| Arithmetic | INT8 weights (per-channel) and activations, int32 accumulation, BatchNorm folded, TFLite-style integer requantization | No float on device; verifiable bit-exactly against a Python simulation |
| Target | Raspberry Pi (model TBD; aarch64 Linux). Laptop x86 for development | Optional NEON path behind a compile flag; plain C remains the reference |
| Training | Python (uv), in this repo under `src/sonar_edge/` | Training never runs on device; Python exports weights + golden vectors |
| Evaluation split | By scene: frames grouped by matching box layout (frame indices turned out to be shuffled), scenes assigned to 5 folds, final numbers by 5-fold cross-validation; fold 0/1 as dev test/val | Avoids the near-duplicate leak demonstrated in sonar_cnn; some classes live in a single scene, so only CV evaluates every class |

## Data

`mvaldenegro/marine-debris-fls-datasets`, folder `md_fls_dataset/data/watertank-segmentation/`:
1,868 frames (`Images/*.png`, 320x480, 1 channel) with `BoxAnnotations/*.xml`
(x y w h, 11 classes). Masks are not used. `quarry-fullsize/` (~7,000 unlabelled
real-world frames) serves only as an out-of-distribution sanity check.

## Verification chain

```
PyTorch float model
  -> Python integer simulation (numpy, exactly the C arithmetic)   [must match C bit for bit]
  -> weights.h + golden input/output vectors
  -> C runtime tests (bit-exact)
  -> C pipeline CLI mAP on test split == Python mAP
```

mAP reported in the final write-up is measured from the C binary, not PyTorch.

## Milestones

Status (2026-11-03): M0-M4 done on the development split (docs/M0-M4); M1
iteration done (strong augmentation +5 mAP; `small` at 360x240 shipped since 2026-11-04 (M1 §6.4 item 7: C 63.81 mAP dev test, 20.1 ms/frame laptop), `wide` the accuracy
reference); 5-fold CV done (M1 §6): `small` 49.4 float / 49.7 int8 pooled mAP at 240x160 (shipped until 2026-11-04), `wide` 55.2 / 56.2 at 13x the compute, `small_wide` 50.2 / 49.6 at 240x160 (width alone does not help, M1 §6.3) but 56.9 +- 2.7 / 56.7 at 360x240 over three seeds (M1 §6.6: width x resolution interact, +4.3 over small@360, every seed pairing and fold; 48 channels give no measurable gain, 50.9 one seed, M1 §6.7; accuracy reference at 204 M MACs, 45% of `wide`'s), `small` at 360x240 52.6 +- 1.9 over 3 seeds vs 48.1 +- 1.1 for `small` (M1 §6.5; resolution is the factor and saturates there, 480x320 gives 54.1 at 4x; 360x240 now shipped; the 240x160 model is the fallback if the Pi latency is too high); segmentation head done (M6: +0.3 M MACs, pooled mIoU 33.9 / objects 28.2 with background weight 0.1, boxes unchanged at 52.1; sub-pixel stride-4 variant 35.0 at +1.6 M MACs; + Dice term: weight 1 gives 42.0 mIoU at -2.3 box mAP over two seeds, weight 0.3 gives 38.9 with boxes inside the box-only range; wide backbone 46.2 mIoU but the same -2.7 box cost; C runtime emits the labels); M5 done on a Pi 4 (2026-11-10, docs/M5-benchmark.md, report/deployment.pdf): output byte-identical to the laptop, NEON kernel (exact, 4 pixels per weight pass) 6.2x, deployed model 20.3 ms/frame one core (49 fps), 187 fps on four, 2.3 MB RSS; small_wide@360 46.5 ms; energy not measured (no meter). M7 (2026-11-11, docs/M7-comparison.md, report/comparison.pdf): all 30 fold models run on the Pi, pooled on-device mAP 52.3 +- 1.7 (small@360) / 56.7 +- 2.7 (small_wide@360), equal to int8 Python; MCNN retrained on the scene folds 77.7% accuracy (99.8% on the random split); detectors on MCNN's naming task without the boxes 74.2 / 77.6% (macro-F1 70.4 / 74.4 vs 71.4); ONNX Runtime on the Pi: MCNN 100.6 ms per average frame vs detector 22.4 ms. Raspberry Pi 2 (2026-11-20, M5 §3.7): NEON kernel ported to 32-bit ARMv7, output byte-identical, deployed model 124.5 ms one core / 30.7 fps four cores. M8 (2026-11-20, docs/M8-pipeline.md): end-to-end C pipeline: front end (fan image bit-exact with the training resize; raw polar beams, tested only on back-projected frames), one pass for boxes + mask, boxes in fan pixels with range/bearing, `sonar-edge-run` JSON CLI; Pi 4 21.9 ms per fan frame with small@360, 23.4 ms with the segdice model. M9 (2026-11-20, docs/M9-streaming.md): SeedLink-style emulator (`sonar-stream`, TCP, sequence numbers, resume) replaying DANS .aris at 15 fps to `sonar-edge-run --stream` on the Pis; stream output identical to file mode; 30 min at 15 fps: Pi 4 1 worker 15.0 fps, 0.03% dropped, latency p50 29.9 ms (100 Mbit link after a cable fault; 22.6 ms at 1 Gbit), 56.9 C max, no throttling; Pi 2 3 workers 14.97 fps, p50 144.6 ms, 50.3 C max. M10 (2026-11-20, docs/M10-polar.md): polar (beam x range, back-projected) input loses to the fan image, 47.1 +- 1.0 vs 52.6 +- 1.9 pooled at matched compute (worse in 13/15 pairs); fan input stays.

| | Milestone | Done when |
|---|---|---|
| M0 | Data | Frames + boxes fetched and parsed; scene-fold split defined and justified; box-size and class statistics; class list confirmed |
| M1 | Float reference model | mAP@0.5 and per-class AP on the sequence split; sanity run on quarry frames; compute budget (MACs) stated |
| M2 | Integer model | Post-training quantization (QAT only if it loses > 1 mAP point); bit-exact integer simulation; weights header + golden vectors exported |
| M3 | C runtime | Ops + static arena + model graph; C tests bit-exact against goldens; builds with `-std=c11 -Wall -Wextra -Werror`, no malloc |
| M4 | Pipeline CLI (Linux) | PNG frame in -> detections out; batch mode over the test split reproduces Python mAP exactly |
| M5 | Benchmark | Latency, peak RAM, binary size, energy per frame on laptop and Pi; compared with the paper's MCNN crop-classifier cost |
| M6 | Report | Extends the sonar_cnn PDF with deployment results |

## Compute budget

Target <= ~100M MACs per frame so a Pi Zero 2 class device runs several frames per
second in plain C and a Pi 4/5 runs in real time. Backbone follows VarCNN (stride-2
early convs, <= 128 channels); the frame may be downsampled 2x before the network
if M1 shows no mAP cost.

## Repo layout

```
src/sonar_edge/   Python: data, model, train, quantize, export, eval
runtime/          C11: include/, src/, tests/, weights/ (generated)
tools/            CLI and benchmark programs
platforms/        build glue per target (linux-x86, rpi)
docs/             this plan, results
data/             raw dataset (git-ignored)
```

## Risks

- 1,868 frames of one tank scene is small for a detector; small-object recall
  (hooks, end-on standing bottles) is the likely weak spot.
- Heatmap heads can be sensitive to PTQ; QAT is the fallback.
- Standing-bottle (94% of boxes in one scene), valve (66%) and shampoo-bottle
  (42%) are concentrated in single scenes: when that scene is the test fold the
  model has barely seen the class. Their per-class AP must be read with this in mind.
