"""Leakage experiment: how much does an image-level random split inflate the scores?

The scene-grouped folds (splits/scene_folds_v1.json) keep every reconstructed scene in
one fold. The random split assigns the same 1,868 frames to five folds at random, with
the fold sizes of the grouped split (so every train/val/test set has the same size as
in the grouped CV), and keeps the scene ids so the overlap can be measured. Everything
else is unchanged: fold k test, (k+1) mod 5 validation, same recipes and evaluator.

  uv run python -m sonar_edge.leakage make      # -> splits/random_folds_v1.json (seed 0)
  uv run python -m sonar_edge.leakage diag      # scene overlap per test fold, both splits
  SONAR_EDGE_SPLIT=splits/random_folds_v1.json ./run_cv.sh rand_small_360_cv small 4/3 150 strong 0
  uv run python -m sonar_edge.leakage score     # -> bench/leakage/summary.json

Training reads the fold assignment through split.SPLIT_FILE (env SONAR_EDGE_SPLIT);
scoring passes each split file explicitly, so `score` needs no environment variable.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .split import DEFAULT_SPLIT_FILE

ROOT = Path(__file__).resolve().parents[2]
RANDOM_SPLIT = ROOT / "splits" / "random_folds_v1.json"
OUT = ROOT / "bench" / "leakage"
SEED = 0

# model -> (grouped run prefixes or names, random ones, scale of the predictions)
DET = {"Det-S": ("small_360", "rand_small_360"), "Det-W": ("small_wide_360", "rand_small_wide_360")}


def det_prefix(base, seed):
    return f"{base}_cv" if seed == 0 else f"{base}_s{seed}_cv"


def cmd_make(args):
    d = json.loads(DEFAULT_SPLIT_FILE.read_text())
    idx = sorted(int(i) for i in d["fold"])
    sizes = np.bincount([d["fold"][str(i)] for i in idx], minlength=d["k"])
    perm = np.random.default_rng(SEED).permutation(idx)
    fold, start = {}, 0
    for f, n in enumerate(sizes):
        for i in perm[start:start + n]:
            fold[int(i)] = f
        start += n
    if RANDOM_SPLIT.exists() and not args.force:
        old = {int(k): v for k, v in json.loads(RANDOM_SPLIT.read_text())["fold"].items()}
        if old != fold:
            raise SystemExit(f"{RANDOM_SPLIT} exists and differs; use --force")
    RANDOM_SPLIT.write_text(json.dumps({
        "method": f"random image-level assignment (numpy default_rng({SEED}).permutation), fold sizes of "
                  f"{DEFAULT_SPLIT_FILE.name}; leakage experiment only",
        "n_groups": d["n_groups"], "k": d["k"], "group": d["group"], "fold": {str(k): v for k, v in fold.items()},
    }, indent=0))
    print(f"wrote {RANDOM_SPLIT.relative_to(ROOT)}: fold sizes {sizes.tolist()}")


def overlap(split_file):
    """Per test fold k: share of test frames whose scene also occurs in the training folds
    (not k, not (k+1) mod 5), and in training or validation."""
    d = json.loads(Path(split_file).read_text())
    group = {int(i): g for i, g in d["group"].items()}
    fold = {int(i): f for i, f in d["fold"].items()}
    rows = []
    for k in range(d["k"]):
        v = (k + 1) % d["k"]
        test = [i for i, f in fold.items() if f == k]
        tr = {group[i] for i, f in fold.items() if f not in (k, v)}
        va = {group[i] for i, f in fold.items() if f == v}
        rows.append({"fold": k, "n_test": len(test),
                     "scene_in_train": float(np.mean([group[i] in tr for i in test])),
                     "scene_in_train_or_val": float(np.mean([group[i] in tr | va for i in test]))})
    return rows


def cmd_diag(args):
    out = {}
    for name, f in (("grouped", DEFAULT_SPLIT_FILE), ("random", RANDOM_SPLIT)):
        out[name] = overlap(f)
        print(f"{name}: test frames whose scene is in training, per fold: "
              + "  ".join(f"{100 * r['scene_in_train']:.1f}%" for r in out[name]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlap.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {(OUT / 'overlap.json').relative_to(ROOT)}")


def _score_runs(runs, scale, split_file, frames, classes):
    from .baselines.adapters import score
    from .cv import load_pred

    if not all((r / "pred_test.json").exists() for r in runs):
        return None
    preds = [{i: [(d[0], d[1], *(v * scale for v in d[2:])) for d in ds] for i, ds in load_pred(r / "pred_test.json").items()}
             for r in runs]
    s = score(preds, 1.0, frames, classes, split_file=split_file)
    debris = [c for c in classes if c != "Wall"]
    vals = [s["ap_pooled"][c] for c in debris if s["ap_pooled"][c] is not None]
    return {"map11": s["map_pooled"], "map10": float(np.mean(vals)), "acc": s["mcnn_task"]["acc"],
            "f1": s["mcnn_task"]["macro_f1"], "missed": s["mcnn_task"]["missed"], "map_per_fold": s["map_per_fold"]}


def cmd_score(args):
    import torch

    from .data import class_names, load_frames

    frames = load_frames()
    classes = class_names(frames)
    rr = lambda name: ROOT / "runs" / name
    models = {}
    for m, (gbase, rbase) in DET.items():
        for split, base, sf in (("grouped", gbase, DEFAULT_SPLIT_FILE), ("random", rbase, RANDOM_SPLIT)):
            for seed in range(3):
                runs = [rr(f"{det_prefix(base, seed)}_f{k}") for k in range(5)]
                if not (runs[0] / "best.pt").exists():
                    continue
                scale = torch.load(runs[0] / "best.pt", map_location="cpu", weights_only=False)["scale"]
                r = _score_runs(runs, scale, sf, frames, classes)
                if r:
                    models.setdefault(m, {}).setdefault(split, {})[seed] = r
    for split, prefix, sf in (("grouped", "yolo_pretrained", DEFAULT_SPLIT_FILE),
                              ("random", "rand_yolo_pretrained", RANDOM_SPLIT)):
        for seed in range(3):
            r = _score_runs([rr(f"{prefix}_s{seed}_f{k}") for k in range(5)], 1.0, sf, frames, classes)
            if r:
                models.setdefault("YOLOv8n (pretrained)", {}).setdefault(split, {})[seed] = r

    summary = {"overlap": {n: overlap(f) for n, f in (("grouped", DEFAULT_SPLIT_FILE), ("random", RANDOM_SPLIT))},
               "models": {}}
    keys = ("map11", "map10", "acc", "f1", "missed")
    for m, by in models.items():
        e = {}
        for split, seeds in by.items():
            e[split] = {"seeds": sorted(seeds), "per_seed": seeds}
            for key in keys:
                v = [seeds[s][key] for s in sorted(seeds)]
                e[split][key] = [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None]
        common = sorted(set(by.get("grouped", {})) & set(by.get("random", {})))
        if common:  # paired by seed: same seed, same recipe, only the split differs
            d = [by["random"][s]["map11"] - by["grouped"][s]["map11"] for s in common]
            e["delta_map11_random_minus_grouped"] = {"seeds": common, "mean": float(np.mean(d)),
                                                     "sd": float(np.std(d, ddof=1)) if len(d) > 1 else None}
        summary["models"][m] = e
        f = lambda t: f"{100 * t[0]:.1f}" + (f"±{100 * t[1]:.1f}" if t[1] is not None else "")
        line = "  ".join(f"{split}: mAP11 {f(e[split]['map11'])} (seeds {e[split]['seeds']})"
                         for split in ("grouped", "random") if split in e)
        dl = e.get("delta_map11_random_minus_grouped")
        print(f"{m:22s} {line}" + (f"  delta {100 * dl['mean']:+.1f}" if dl else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {(OUT / 'summary.json').relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--force", action="store_true")
    mk.set_defaults(func=cmd_make)
    sub.add_parser("diag").set_defaults(func=cmd_diag)
    sub.add_parser("score").set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
