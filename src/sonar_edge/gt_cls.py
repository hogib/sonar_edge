"""Score the detector on MCNN's task: classify the annotated objects.

For every ground-truth box of the 10 object classes (Wall excluded, as for
MCNN), the detector's prediction is the class of its highest-scoring detection
that overlaps the box with IoU >= 0.5; a box with no such detection counts as
an error ("missed"). The detector is not told where the boxes are, so this is
a harder task than MCNN's, which is handed the crop. Detections come from the
5-fold test predictions (int8, or a file of on-device outputs per fold).

  uv run python -m sonar_edge.gt_cls runs/small_360_cv runs/small_360_s1_cv runs/small_360_s2_cv [--variant int8|pi4]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cv import fold_gt, load_pred
from .data import class_names, load_frames
from .mcnn_cv import scores

ROOT = Path(__file__).resolve().parents[2]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def classify_boxes(pred, gt, wall, thr=0.5):
    """-> (true labels, predicted labels with -1 for missed), over non-Wall GT boxes."""
    yt, yp = [], []
    for idx, boxes in gt.items():
        dets = pred.get(idx, [])
        for g in boxes:
            if g[0] == wall:
                continue
            best = max(((d[1], d[0]) for d in dets if iou(d[2:], g[1:]) >= thr), default=None)
            yt.append(g[0])
            yp.append(-1 if best is None else best[1])
    return np.array(yt), np.array(yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--variant", default="int8", help="int8 (pred_test_int8.json) or pi4 (pred_test_pi4.json)")
    args = ap.parse_args()
    frames = load_frames()
    classes = class_names(frames)
    wall = classes.index("Wall")
    obj = [c for c in classes if c != "Wall"]
    remap = np.array([obj.index(c) if c != "Wall" else len(obj) for c in classes] + [-1])  # -1 -> missed
    fn = {"int8": "pred_test_int8.json", "float": "pred_test.json"}.get(args.variant, f"pred_test_{args.variant}.json")
    res = []
    for p in args.prefixes:
        prefix = ROOT / p
        runs = [prefix.parent / f"{prefix.name}_f{k}" for k in range(5)]
        scale = torch.load(runs[0] / "best.pt", map_location="cpu", weights_only=False)["scale"]
        yt, yp, per_fold = [], [], []
        for k, r in enumerate(runs):
            t, q = classify_boxes(load_pred(r / fn), fold_gt(frames, classes, k, scale), wall)
            yt.append(remap[t])
            yp.append(remap[q])  # Wall predictions -> index 10 (always wrong), missed -> -1
            per_fold.append(float(np.mean(yt[-1] == yp[-1])))
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        acc, mf1, rec = scores(yt, yp, len(obj))
        res.append({"prefix": p, "acc": acc, "macro_f1": mf1, "recall": rec, "acc_per_fold": per_fold,
                    "missed": float(np.mean(yp == -1)), "n": int(len(yt))})
        print(f"{p}: acc {100 * acc:.2f}  macro-F1 {100 * mf1:.2f}  missed {100 * res[-1]['missed']:.1f}%  "
              f"per fold " + " ".join(f"{100 * a:.1f}" for a in per_fold) + f"  ({len(yt)} boxes)")
    a = np.array([r["acc"] for r in res])
    f = np.array([r["macro_f1"] for r in res])
    R = np.array([r["recall"] for r in res])
    sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0
    print(f"mean: acc {100 * a.mean():.2f} +- {100 * sd(a):.2f}  macro-F1 {100 * f.mean():.2f} +- {100 * sd(f):.2f}")
    for c, r in zip(obj, R.mean(0)):
        print(f"   {c:16s} recall {100 * r:5.1f}")
    out = {"variant": args.variant, "runs": res, "classes": obj, "acc_mean": float(a.mean()), "acc_sd": sd(a),
           "macro_f1_mean": float(f.mean()), "macro_f1_sd": sd(f), "recall_mean": R.mean(0).tolist()}
    dest = ROOT / f"{args.prefixes[0]}_gtcls_{args.variant}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
