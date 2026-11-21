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
