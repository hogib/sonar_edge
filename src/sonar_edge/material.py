"""Derived task: microplastic sources vs other debris.

Materials from the dataset paper (Valdenegro-Toro et al. 2025, arXiv:2503.22880,
App. B, Table V(a)): shampoo bottle plastic, tire rubber, can / hook / propeller /
valve metal, standing bottle glass; drink-carton and chain materials are not
stated. Grouping (user's decision, 2026-11-21):

  microplastic sources (1): Shampoo-bottle, Tire, Drink-carton (assumed laminated carton)
  other debris (0):         Can, Chain, Hook, Propeller, Valve, Standing-bottle
  excluded:                 Bottle (glass and plastic bottles share one label), Wall

Merging classes removes confusions within a group, so the binary scores are not
comparable with the 10/11-class scores.

(a) MCNN task, binary: every non-excluded GT box gets the group of the class
    predicted for it (detectors: highest-scoring detection with IoU >= 0.5, as in
    gt_cls; MCNN: its crop prediction). Missed boxes and predictions of an
    excluded class (Bottle, Wall) count as errors.
(b) Detection, binary: GT and detections merged into the two groups, excluded
    classes dropped from both, pooled VOC AP@0.5 per group (detectors only).

  uv run python -m sonar_edge.material                          # defaults below
  uv run python -m sonar_edge.material --det runs/foo_cv runs/foo_s1_cv runs/foo_s2_cv --variant pi4
Baselines: pass their run prefixes with --det; they need pred_test_<variant>.json
per fold in our (cls, score, x0, y0, x1, y1) format at the run's working scale.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch

from .cv import fold_gt, load_pred
from .data import class_names, load_frames
from .gt_cls import classify_boxes
from .metrics import evaluate

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
MICRO = ("Shampoo-bottle", "Tire", "Drink-carton")
OTHER = ("Can", "Chain", "Hook", "Propeller", "Valve", "Standing-bottle")
EXCLUDED = ("Bottle", "Wall")


def group_of(name):
    return 1 if name in MICRO else 0 if name in OTHER else None


def binary_scores(tg, pg):
    """tg: true group (0/1); pg: predicted group (0/1, -1 = error). -> metrics dict."""
    tg, pg = np.asarray(tg), np.asarray(pg)
    tp = np.sum((pg == 1) & (tg == 1))
    rec = tp / max(np.sum(tg == 1), 1)
    prec = tp / max(np.sum(pg == 1), 1)
    return {"acc": float(np.mean(pg == tg)), "micro_recall": float(rec), "micro_precision": float(prec),
            "micro_f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "other_recall": float(np.mean(pg[tg == 0] == 0)), "n": int(len(tg)), "n_micro": int(np.sum(tg == 1))}


def mean_sd(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def detector_runs(prefix, variant, frames, classes):
    """-> (task-a dict, task-b dict, 10-class accuracy check) for one prefix (one seed, 5 folds)."""
    wall = classes.index("Wall")
    grp = np.array([-2 if group_of(c) is None else group_of(c) for c in classes])
    fn = {"int8": "pred_test_int8.json", "float": "pred_test.json"}.get(variant, f"pred_test_{variant}.json")
    runs = [ROOT / f"{prefix}_f{k}" for k in range(5)]
    scale = torch.load(runs[0] / "best.pt", map_location="cpu", weights_only=False)["scale"]
    tg, pg, yt_all, yp_all, on_bottle = [], [], [], [], 0
    pred_b, gt_b = {}, {}
    for k, r in enumerate(runs):
        pred, gt = load_pred(r / fn), fold_gt(frames, classes, k, scale)
        yt, yp = classify_boxes(pred, gt, wall)
        yt_all.append(yt)
        yp_all.append(yp)
        keep = grp[yt] >= 0
        t = grp[yt[keep]]
        p = np.where(yp[keep] < 0, -1, grp[np.maximum(yp[keep], 0)])
        on_bottle += int(np.sum(yp[keep] == classes.index("Bottle")))
        p = np.where(p < 0, -1, p)                      # missed, Bottle or Wall -> error
        tg.append(t)
        pg.append(p)
        for img, ds in pred.items():                    # task (b): merged groups
            pred_b[img] = [(int(grp[d[0]]), *d[1:]) for d in ds if grp[d[0]] >= 0]
        for img, gs in gt.items():
            gt_b[img] = [(int(grp[g[0]]), *g[1:]) for g in gs if grp[g[0]] >= 0]
    yt_all, yp_all = np.concatenate(yt_all), np.concatenate(yp_all)
    a = binary_scores(np.concatenate(tg), np.concatenate(pg))
    a["pred_on_bottle"] = on_bottle
    aps, m = evaluate(pred_b, gt_b, 2)
    b = {"ap_other": float(aps[0]), "ap_micro": float(aps[1]), "map": m}
    return a, b, float(np.mean(yt_all == yp_all))


def mcnn_runs(seed):
    rs = [json.loads((ROOT / "runs" / f"mcnn_cv_s{seed}_f{k}" / "results.json").read_text()) for k in range(5)]
    classes = rs[0]["classes"]
    grp = np.array([-2 if group_of(c) is None else group_of(c) for c in classes])
    yt = np.concatenate([r["labels"] for r in rs])
    yp = np.concatenate([r["test_pred"] for r in rs])
    keep = grp[yt] >= 0
    p = grp[yp[keep]]
    a = binary_scores(grp[yt[keep]], np.where(p < 0, -1, p))
    a["pred_on_bottle"] = int(np.sum(yp[keep] == classes.index("Bottle")))
    return a, float(np.mean(yt == yp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", nargs="+", action="append",
                    help="run prefixes of one model (one per seed); repeat for several models")
    ap.add_argument("--variant", default="pi4")
    ap.add_argument("--mcnn-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default="bench/material/summary.json")
    args = ap.parse_args()
    dets = args.det or [["runs/small_360_cv", "runs/small_360_s1_cv", "runs/small_360_s2_cv"],
                        ["runs/small_wide_360_cv", "runs/small_wide_360_s1_cv", "runs/small_wide_360_s2_cv"]]
    frames = load_frames()
    classes = class_names(frames)
    ka = ["acc", "micro_recall", "micro_precision", "micro_f1", "other_recall"]
    out = {"groups": {"micro": MICRO, "other": OTHER, "excluded": EXCLUDED}, "variant": args.variant, "models": {}}
    if args.mcnn_seeds:
        rows, chk = zip(*[mcnn_runs(s) for s in args.mcnn_seeds])
        out["models"]["MCNN"] = {"task_a": mean_sd(rows, ka), "task_a_per_seed": rows,
                                 "check_10class_acc": mean_sd([{"acc": c} for c in chk], ["acc"])["acc"]}
    for prefs in dets:
        res = [detector_runs(p, args.variant, frames, classes) for p in prefs]
        a, b, chk = zip(*res)
        out["models"][prefs[0]] = {"seeds": prefs, "task_a": mean_sd(a, ka), "task_a_per_seed": a,
                                   "task_b": mean_sd(b, ["ap_other", "ap_micro", "map"]), "task_b_per_seed": b,
                                   "check_10class_acc": mean_sd([{"acc": c} for c in chk], ["acc"])["acc"]}
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    for name, m in out["models"].items():
        s = "  ".join(f"{k} {100 * v['mean']:.1f}±{100 * v['sd']:.1f}" for k, v in m["task_a"].items())
        print(f"{name}: (a) {s}  | 10-class check acc {100 * m['check_10class_acc']['mean']:.2f}")
        if "task_b" in m:
            print("   (b) " + "  ".join(f"{k} {100 * v['mean']:.1f}±{100 * v['sd']:.1f}" for k, v in m["task_b"].items()))
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
