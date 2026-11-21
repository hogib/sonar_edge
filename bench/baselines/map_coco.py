"""COCO-style mAP@[0.5:0.95] (mean over IoU 0.50, 0.55, ..., 0.95 of the pooled VOC AP over
11 classes) for Det-S, Det-W and the reference detectors, from the stored test predictions
of the scene-grouped CV; same evaluator and ground truth as the AP@0.5 results.

  uv run --extra baselines python bench/baselines/map_coco.py   # -> bench/baselines/map_coco.json
"""
import json
from pathlib import Path

import numpy as np
import torch

from sonar_edge.cv import fold_gt, load_pred
from sonar_edge.data import class_names, load_frames
from sonar_edge.metrics import evaluate

ROOT = Path(__file__).resolve().parents[2]
THR = [round(0.5 + 0.05 * i, 2) for i in range(10)]
frames = load_frames()
classes = class_names(frames)
gt = {}
for k in range(5):
    gt.update(fold_gt(frames, classes, k, 1.0))


def det(base, s):
    return f"{base}_cv" if s == 0 else f"{base}_s{s}_cv"


models = {"Det-S": [det("small_360", s) for s in range(3)], "Det-W": [det("small_wide_360", s) for s in range(3)]}
for fam in ["yolo_pretrained", "ssdlite_pretrained", "nanodet_pretrained", "yolo_scratch", "ssdlite_scratch", "nanodet_scratch"]:
    models[fam] = [f"bl_{fam}_s{s}" for s in range(3)]

out = {"iou_thresholds": THR, "models": {}}
for m, prefixes in models.items():
    per_seed = []
    for pre in prefixes:
        runs = [ROOT / "runs" / f"{pre}_f{k}" for k in range(5)]
        scale = torch.load(runs[0] / "best.pt", map_location="cpu", weights_only=False)["scale"]
        p = {}
        for r in runs:
            p.update({i: [(d[0], d[1], *(v * scale for v in d[2:])) for d in ds] for i, ds in load_pred(r / "pred_test.json").items()})
        maps = [evaluate(p, gt, len(classes), t)[1] for t in THR]
        per_seed.append({"map50": maps[0], "map75": maps[5], "map50_95": float(np.mean(maps)), "per_thr": maps})
    agg = {k: [float(np.mean([s[k] for s in per_seed])), float(np.std([s[k] for s in per_seed], ddof=1))] for k in ["map50", "map75", "map50_95"]}
    out["models"][m] = {"runs": prefixes, "per_seed": per_seed, **agg}
    print(f"{m:20s} mAP50 {100*agg['map50'][0]:.1f}  mAP75 {100*agg['map75'][0]:.1f}  mAP50:95 {100*agg['map50_95'][0]:.1f}±{100*agg['map50_95'][1]:.1f}")
(ROOT / "bench/baselines/map_coco.json").write_text(json.dumps(out, indent=1))
