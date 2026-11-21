# Baselines for the paper

All baselines are trained and scored under the detector's protocol: the scene-grouped
5-fold split (`splits/scene_folds_v1.json`; test k, validation (k+1) mod 5, training
the rest), 3 seeds, and our own evaluator. Pooled VOC AP@0.5 comes from `cv.pool`, and
the MCNN task from `gt_cls.classify_boxes` plus `mcnn_cv.scores`.

Code is in `src/sonar_edge/baselines/`. Optional dependencies are installed with
`uv run --extra baselines ...`; torchvision 0.29.2 matches torch 2.15.0+cu132.

## Data and adapters

- `folds.py` exports the frames once at native 320x480 (grey PNG) to `data/baselines/`
  (git-ignored). It writes YOLO labels and per-fold lists and yaml files, and COCO
  JSON per fold and split (category id = class index + 1). All 11 classes are
  included, Wall too.
- `adapters.py` converts any tool's detections to `{index: [(cls, score, x0, y0, x1,
  y1)]}` in native pixels and scores them. Checks (`adapters check`, plus a label round trip):

| Check | Result |
|---|---|
| Ground truth fed in as predictions | pooled mAP 100.00, MCNN-task accuracy 100.00 |
| small@360 seed 0, rescaled to native pixels | pooled mAP 54.6401, identical to `sonar-edge cv` |
| YOLO label files converted back to boxes | pooled mAP 100.0, MCNN-task accuracy 100.0 |

## Input size

The detectors see the frame box-filtered by 1.25, i.e. 256x384 (W x H). That is the
stride-32 size closest to our detector's 240x360, about 14% more pixels. YOLO uses
`imgsz=384` with rectangular batches, which gives the same 256x384.

## Classifiers on MCNN's task (`classifiers.py`)

- **Networks:** torchvision MobileNetV3-Small and ShuffleNetV2 x0.5, trained from
  scratch like MCNN, with the first convolution taking one grey channel.
- **Data and recipe:** exactly MCNN's, from `mcnn_cv.py`: the same 2,826 crops
  zero-padded to 305x305, AdamW 1e-3 / wd 1e-4, cosine schedule, 30 epochs, batch 32,
  best-validation epoch.
- **MACs:** counted with the same hooks as MCNN's 234.7 M (`macs.py`).

## SSDLite-MobileNetV3-Large (`ssdlite.py`)

- **Setup:** torchvision `ssdlite320_mobilenet_v3_large` at a fixed 256x384 input, with
  an ImageNet-pretrained backbone (torchvision's default; the baseline's best realistic
  setting) and a detection head trained from scratch.
- **Training:** our "strong" augmentation, AdamW with a one-cycle schedule, batch 16.
- **Epochs:** 60 rather than our 150, because the pretrained backbone converges in a
  few epochs (a 2-epoch smoke test already reached 58.6 validation mAP).
- **Model selection:** on validation mAP, evaluated every 5 epochs.
- **Scratch variant:** `--scratch` (no backbone weights), as a side note.

## Export and on-device plan (later, on the Pis)

- **Export:** every baseline goes to ONNX (opset 17) via `torch.onnx.export`; YOLO
  via `yolo export format=onnx`.
- **Engines:**
  - ONNX Runtime float32 and int8 (static quantization with calibration frames from
    the training folds), as for MCNN in M7;
  - ncnn for YOLOv8n and NanoDet-Plus via `pnnx` (ONNX or TorchScript to ncnn,
    including ncnn's int8 tool), built natively on the Pi 4 and Pi 2.
- **What to measure:** latency on 1 and 4 cores, peak RSS, model size and accuracy
  after quantization, scored through `adapters.py`.

## Settings decided (2026-11-21)

**Initialisation.** Every detector is reported in two settings: pretrained (main row)
and from scratch, like our detector and MCNN.

**Epochs.**

| Setting | Epochs | Reason |
|---|---|---|
| Pretrained | 60 | pretrained backbones converge within a few epochs |
| Scratch | 150 | the same as our detector |

**SSDLite (torchvision).**
- Pretrained setting: ImageNet MobileNetV3-Large backbone. Trained on the laptop (RTX 3050).
- Scratch setting: the **same full architecture** with the backbone re-initialised.
  Without backbone weights, torchvision silently builds a reduced tail (480 instead of
  960 channels); this was caught when the ONNX export failed and was fixed before the
  scratch CV restarted. Trained on the GPU server.
- 3,876,080 parameters in both settings.

**YOLOv8n.**
- Ultralytics 8.4.216, AGPL-3.0, in `.venv-yolo` on the server (torch 2.15.0+cu132).
- Pretrained setting: COCO `yolov8n.pt` (sha256 f59b3d83…). Scratch setting: `yolov8n.yaml`.
- Ultralytics' own default augmentation and schedule: imgsz 384, square letterboxed
  training with mosaic, rectangular 256x384 inference, batch 16.
- Selection: `best.pt` on the validation fold. Test predictions use conf 0.001, NMS 0.7, max 100.
- Script: `src/sonar_edge/baselines/yolo_run.py`.

**NanoDet-Plus-m.**
- RangiLyu/nanodet commit be9b4a9, Apache-2.0, in `.venv-nanodet` on the server.
- NanoDet pins torch<2.0 and Lightning 1.x, which cannot run on the RTX 5090 (sm_120).
  So we use its model and loss code (`build_model`, `forward_train` with the auxiliary
  head, `get_bboxes`) on torch 2.15 with pytorch-lightning 1.9.5 (import only) and
  setuptools<81.
- Our own loop reproduces `nanodet-plus-m_320.yml`:
  - AdamW 1e-3 / weight decay 0.05;
  - 500-step warm-up, then cosine down to 5e-5;
  - gradient clipping 35, EMA 0.9998;
  - auxiliary head detached after epoch 10;
  - batch 16 (the config uses 96 over 8 GPUs).
- Our "strong" augmentation, input 256x384.
- Pretrained setting: ImageNet ShuffleNetV2-1.0x backbone (config default). Scratch setting: none.
- Script: `src/sonar_edge/baselines/nanodet_run.py`.

**Server jobs** (tmux `sb_*`, logs in `bench/baselines/logs/` on the server; at most
3 jobs at a time on a shared GPU, about 6 GB of ours):
- `sb_yolo_pre` → `sb_nano_pre` (queued behind it);
- `sb_yolo_scr` → `sb_nano_scr` (queued behind it);
- `sb_ssd_scr`.

Collect with `python -m sonar_edge.baselines.collect pull|score|onnx`.

**ONNX export.**
- SSDLite: `collect onnx` (16 MB, torchvision decoder included).
- YOLO: `.venv-yolo/bin/yolo export model=<run>/weights/best.pt format=onnx imgsz=384,256 opset=17`.
- NanoDet: export with its ONNX forward (`tools/export_onnx.py` logic).
- Timing does not depend on the weights, so one model per family is enough (seed 0, fold 0).

## MCNN with proposals (`mcnn_proposals.py`)

- **Pipeline:** Det-S (small@360, on-device int8 outputs from M7), class labels
  discarded, class-agnostic NMS at 0.5. Each proposal is cropped, padded to 305 and
  classified by the MCNN model of the same fold and seed.
- **Thresholds:** two proposal thresholds; Det-S alone at the same threshold is the
  like-for-like reference.
- **Check:** at logit -4, Det-S scores 74.19, which reproduces M7's 74.2.

| Proposals | MCNN + proposals: acc / F1 / missed | Det-S alone: acc / F1 / missed | Crops per frame | MACs per frame | Pi 4 ms per frame (estimate) |
|---|---|---|---|---|---|
| p >= 0.3 | 55.1±1.9 / 53.1±1.2 / 34.5% | 61.1±1.8 / 60.9±2.8 / 34.2% | 1.94 | 533 M | about 149 |
| p >= 0.018 (logit -4) | 66.7±1.7 / 60.3±1.4 / 11.4% | 74.2±1.3 / 70.4±3.0 / 7.4% | 22.5 | 5.36 G | about 1,516 |

The Pi 4 time is estimated as Det-S C int8 20.3 ms plus crops x 66.5 ms (MCNN on ONNX
Runtime, one thread, M7). It was not measured end to end.

## On-device latency (Pi 4, ONNX Runtime fp32)

`src/sonar_edge/baselines/pi_latency.py` (export + bench); results in `pi_latency.json`
(raw ORT: `pi_latency_raw_pi4.json`). The method matches M7 §3: ORT 1.31.1, CPU EP, fp32,
ORT_ENABLE_ALL, 10 warm-up runs and then 3 x 50 timed runs, with intra-op threads 1 and 4.
Each setting runs in its own process, and its peak RSS (VmHWM) is recorded. Only the
network is timed: there is no NMS, and the timings cover the fold-0 seed-0 pretrained
checkpoints. The board stayed unthrottled (0x0) and went from 48.2 to 56.4 °C.

| Model | Params | MACs | Input | 1 thread ms | 4 threads ms | Peak RSS MiB |
|---|---|---|---|---|---|---|
| Det-S (reference) | 42 k | 78.4 M | 1x360x240 | 22.4 | 9.0 | 64.6 |
| SSDLite-MobileNetV3 | 3.88 M | 488 M | 3x384x256 | 202.4 | 94.2 | 104.2 |
| YOLOv8n | 3.14 M | 1038 M | 3x384x256 | 285.8 | 107.8 | 89.6 |

- **Neither baseline reaches 15 frames/s** on all four cores before NMS: YOLOv8n gives
  9.3 fps and SSDLite 10.6 fps.
- **Det-S** under the same engine is 12.7x faster than YOLOv8n and 9.0x faster than
  SSDLite on one core.
- **Graph contents.** The YOLOv8n export includes DFL box decoding. The SSDLite graph
  ends at the raw head outputs, so its anchor decoding and NMS come on top.
- **Peak RSS** comes from this script. It is not comparable with M7's 58.1 MB, which
  was measured another way; for Det-S this script reads 64.6 MiB.
- **Pi 2 via ncnn.** No onnxruntime wheel exists for armv7l with Python 3.13 (PyPI
  and piwheels), so the Pi 2 was timed with ncnn. The models were converted from ONNX
  with pnnx, giving blobs in0 -> out0[, out1], and run with ncnn 1.0.20261020 (pip
  wheel), fp32, fp16 off. To get a same-engine comparison across boards, ncnn was run
  on the Pi 4 as well. On the Pi 4 the ncnn outputs matched ORT to within 0.1% of the
  output range. The method was the same (10 warm-up + 3 x 50 runs), and neither board
  throttled (0x0); the Pi 2 went from 42.8 to 47.6 °C.

ncnn fp32, network only (ms; peak RSS of the process in MiB):

| Model | Pi 4 1 thr | Pi 4 4 thr | Pi 2 1 thr | Pi 2 4 thr | RSS Pi 4 / Pi 2 |
|---|---|---|---|---|---|
| Det-S | 19.5 | 11.4 | 146.8 | 53.7 | 48.5 / 39.9 |
| SSDLite-MobileNetV3 | 140.4 | 93.2 | 1020.8 | 426.7 | 77.2 / 69.5 |
| YOLOv8n | 157.8 | 92.3 | 1260.8 | 439.3 | 106.3 / 97.9 |

- **On the Pi 2, the baselines run at 2.3 fps (YOLOv8n) and 2.3 fps (SSDLite) on four
  cores, and Det-S runs at 18.6 fps.** The deployed C int8 runtime does better still,
  at 124.5 ms on one core and 30.7 fps with four workers (M5 §3.7).
- **On the Pi 4, ncnn speeds up the baselines on one core:** YOLOv8n goes from 286 ms
  under ORT to 158 ms, and SSDLite from 202 to 140 ms. On four cores, though, neither
  baseline gets past about 11 fps under either engine. For Det-S, ncnn on four threads
  (11.4 ms) is slower than ORT (9.0 ms), because the model is too small to split well.
- **Raw results** are in `pi_latency_raw_pi4_ncnn.json` and
  `pi_latency_raw_pi2_ncnn.json`, and the summary is in `pi_latency.json["ncnn"]`.
  The ncnn files themselves (param/bin) are not kept in the repo; regenerate them with
  `pnnx bench/baselines/onnx/<model>.onnx`.

## int8 baselines (ncnn post-training quantization)

Question answered: Det-S is deployed as int8 while the baselines above were timed in fp32.
Does int8 change the comparison? Script: `src/sonar_edge/baselines/int8_ncnn.py`; results:
`int8_ncnn.json` (accuracy, latency summary, settings), raw timings
`int8_ncnn_raw_pi4.json` and `int8_ncnn_raw_pi2.json`.

**Method.**
- **Tools:** ncnn 1.0.20261020. `ncnn2table`/`ncnn2int8` come from the official release
  `ncnn-20261020-ubuntu-2604.zip` (sha256 c4d6646e…dfaa19e); the pip wheel of the same
  version was used on the laptop, the Pi 4 and the Pi 2.
- **Models:** the seed-0 checkpoint of every fold, converted ONNX → ncnn with
  `pnnx fp16=0`, since ncnn2int8 needs fp32 weights.
- **Calibration:** on 64 training frames of the same fold, drawn exactly like the Det-S
  calibration (`default_rng(0).choice`). Inputs are the training-time inputs: the
  box-filtered frames for SSDLite and Det-S, and Ultralytics' LetterBox output (cv2
  bilinear, from `.venv-yolo`) for YOLOv8n.
- **Calibration methods tried:** ncnn's `kl` (default), `aciq` and `percentile`.
  `eq` crashes `ncnn2table` (segfault in the weight-scale search).
- **Accuracy:** every test frame, with each baseline's own postprocessing
  (torchvision SSD postprocess; Ultralytics-style NMS with conf 0.001, IoU 0.7, max 100),
  scored by `adapters.score`.

**Accuracy, seed 0, pooled mAP over 5 folds (%):**

| Model | PyTorch | ncnn fp32 | int8 KL | int8 ACIQ | int8 percentile |
|---|---|---|---|---|---|
| SSDLite-MobileNetV3 | 62.64 | 62.64 | 38.38 | 60.80 | **63.26** |
| YOLOv8n | 64.00 | 64.01 | 48.37 | 64.03 | **63.75** |

- **ncnn fp32 reproduces** the PyTorch / Ultralytics predictions, so the export,
  inputs and postprocessing are right.
- **With percentile calibration, int8 costs neither baseline any accuracy.** This is
  the method closest to Det-S's own 99.99th-percentile calibration. ncnn's default KL
  calibration loses 16–24 points on these sonar images. The baselines are therefore
  credited with the best method.

**Latency, ncnn, network only, no NMS (ms, mean of 3 × 50 runs).** Fold-0 models;
baselines use the percentile tables, Det-S int8 uses a KL table (latency only):

| Model | Pi 4 1 thr fp32 → int8 | Pi 4 4 thr fp32 → int8 | Pi 2 1 thr fp32 → int8 | Pi 2 4 thr fp32 → int8 |
|---|---|---|---|---|
| Det-S | 19.5 → 11.9 | 11.3 → 5.8 | 146.8 → 91.4 | 53.7 → 34.1 |
| SSDLite-MobileNetV3 | 140.3 → 136.8 | 93.1 → 76.7 | 1020.8 → 1050.6 | 426.7 → 391.2 |
| YOLOv8n | 157.4 → 131.6 | 92.0 → 68.2 | 1260.8 → 879.7 | 439.3 → 321.9 |

- **Pi 4, int8, four cores:** YOLOv8n reaches 14.7 fps and SSDLite 13.0 fps for the
  network alone. YOLOv8n is now just below the 15 fps of the sonar, before NMS and
  preprocessing, and it needs all four cores to get there. On one core int8 gives
  7.6 fps (YOLOv8n) and 7.3 fps (SSDLite).
- **Det-S** under ncnn int8 runs at 11.9 ms on one Pi 4 core, 11× faster than int8
  YOLOv8n and 11.5× faster than int8 SSDLite. The project's own C int8 runtime does
  20.3 ms on the Pi 4 and 124.5 ms on the Pi 2; ncnn's int8 kernels are faster for
  Det-S on both boards.
- **Pi 2:** int8 baselines run at 3.1 fps (YOLOv8n) and 2.6 fps (SSDLite) on four
  cores, against 29 fps for Det-S int8 under ncnn.
- **Memory (peak RSS, int8, 1 thread):** Pi 4: Det-S 40.9 MiB, YOLOv8n 68.7 MiB,
  SSDLite 60.6 MiB.
- **Throttling:** none (0x0). The Pi 4 went from 46.7 to 56.0 °C and the Pi 2 from
  40.1 to 45.5 °C.

**Caveats.**
- The Pi 2 fp32 column comes from the earlier run, whose models had fp16-stored
  weights with fp32 arithmetic. The Pi 4 fp32 column was re-run alongside int8 and
  matches the earlier run within 0.5 ms.
- Accuracy was measured for seed 0 only.
- Latency is the network alone. NMS and the SSD anchor decoding are extra for the
  baselines, while Det-S's decoder is a small peak search.
