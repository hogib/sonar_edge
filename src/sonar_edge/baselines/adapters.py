"""Score third-party detections with the project's own evaluator.

Every tool's output is converted to the format of runs/*/pred_test.json:
image index -> [(cls, score, x0, y0, x1, y1)], boxes in NATIVE frame pixels (320x480),
class index in class_names order (Wall included). Scoring then uses the same code and
the same ground truth as the detector CV: pooled VOC AP@0.5 over the five test folds
(cv.pool) and MCNN's task (gt_cls.classify_boxes + mcnn_cv.scores).

  uv run python -m sonar_edge.baselines.adapters check     # GT-as-prediction and small@360 checks
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..cv import fold_gt, load_pred, pool
from ..data import class_names, load_frames
from ..gt_cls import classify_boxes
from ..metrics import evaluate
from ..mcnn_cv import scores

ROOT = Path(__file__).resolve().parents[3]


def save_pred(pred, path):
    Path(path).write_text(json.dumps({str(k): [list(map(float, d)) for d in v] for k, v in pred.items()}))


def score(preds, scale=1.0, frames=None, classes=None):
    """preds: list of 5 per-fold dicts (fold k = test fold k), boxes at `scale` (native = 1).
    Returns dict with pooled mAP, per-class AP, per-fold mAP and the MCNN-task scores."""
    frames = frames or load_frames()
    classes = classes or class_names(frames)
    gts = [fold_gt(frames, classes, k, scale) for k in range(5)]
    aps, m = pool(preds, gts, len(classes))
    per_fold = [evaluate(p, g, len(classes))[1] for p, g in zip(preds, gts)]
    wall = classes.index("Wall")
    obj = [c for c in classes if c != "Wall"]
    remap = np.array([obj.index(c) if c != "Wall" else len(obj) for c in classes] + [-1])
    yt, yp = [], []
    for p, g in zip(preds, gts):
        t, q = classify_boxes(p, g, wall)
        yt.append(remap[t])
        yp.append(remap[q])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    acc, mf1, rec = scores(yt, yp, len(obj))
    return {"map_pooled": m, "ap_pooled": {c: (None if np.isnan(a) else float(a)) for c, a in zip(classes, aps)},
            "map_per_fold": per_fold, "mcnn_task": {"acc": acc, "macro_f1": mf1, "missed": float(np.mean(yp == -1)),
                                                    "recall": dict(zip(obj, map(float, rec)))}}


def cmd_check(args):
    frames = load_frames()
    classes = class_names(frames)
    # 1. ground truth as predictions (score 1): every AP must be 1, MCNN-task accuracy 1
    gts = [fold_gt(frames, classes, k, 1.0) for k in range(5)]
    perfect = [{i: [(c, 1.0, *b) for c, *b in g] for i, g in gt.items()} for gt in gts]
    s = score(perfect, 1.0, frames, classes)
    print(f"GT as prediction: pooled mAP {100 * s['map_pooled']:.2f}, MCNN-task acc {100 * s['mcnn_task']['acc']:.2f}")
    # 2. an existing detector CV (small@360 seed 0), mapped to native pixels through the adapter
    runs = [ROOT / "runs" / f"small_360_cv_f{k}" for k in range(5)]
    scale = 4 / 3
    native = [{i: [(d[0], d[1], *(v * scale for v in d[2:])) for d in ds] for i, ds in load_pred(r / "pred_test.json").items()}
              for r in runs]
    s2 = score(native, 1.0, frames, classes)
    ref = json.loads((ROOT / "runs" / "small_360_cv_cv.json").read_text())["float"]["map_pooled"]
    print(f"small@360 seed 0 via adapter: pooled mAP {100 * s2['map_pooled']:.4f} (sonar-edge cv: {100 * ref:.4f})")
    out = ROOT / "bench" / "baselines" / "adapter_check.json"
    out.write_text(json.dumps({"gt_as_pred_map": s["map_pooled"], "gt_as_pred_acc": s["mcnn_task"]["acc"],
                               "small360_s0_adapter_map": s2["map_pooled"], "small360_s0_cv_map": ref}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    c = sub.add_parser("check")
    c.set_defaults(func=cmd_check)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
