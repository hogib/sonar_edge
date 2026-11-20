"""Pool the polar-input CV runs and compare with small@360 (fan input).

For each variant and seed, the five folds' test detections (polar boxes) are mapped back to the
fan image (polar_data.polar_box_to_fan) and pooled against the ORIGINAL fan GT of all 1,868 frames,
with the same evaluator and pooling as `sonar-edge cv`. Polar-space pooled mAP (against the derived
polar GT) is reported for reference. Paired comparison: per-fold fan-space mAP minus that of
small@360 (runs/small_360{,_s1,_s2}_cv_cv.json, float), over the 15 fold-seed pairs.

  uv run python bench/polar/polar_cv.py [--out bench/polar/summary.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sonar_edge import polar_data as pd
from sonar_edge.cv import fold_gt, load_pred, pool
from sonar_edge.data import class_names, load_frames
from sonar_edge.metrics import evaluate
from sonar_edge.split import SPLIT_FILE

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"polar_440x128": ("small_polar", 128), "polar_440x192": ("small_polar192", 192)}
FAN_REF = ["small_360_cv", "small_360_s1_cv", "small_360_s2_cv"]
PRED = {"float": "pred_test.json", "int8": "pred_test_int8.json"}


def tags(name):
    return [f"{name}_cv", f"{name}_s1_cv", f"{name}_s2_cv"]


def polar_gt(pf, classes, fold):
    fold_of = {int(i): f for i, f in json.loads(SPLIT_FILE.read_text())["fold"].items()}
    return {p.index: [(classes.index(b.cls), b.x, b.y, b.x + b.w, b.y + b.h) for b in p.boxes]
            for p in pf if fold_of[p.index] == fold}


def summarise(per_seed, classes):
    m = np.array([s["fan_pooled"] for s in per_seed])
    sd = float(m.std(ddof=1)) if len(m) > 1 else 0.0
    return {"seeds": len(m), "fan_pooled_mean": float(m.mean()), "fan_pooled_sd": sd,
            "polar_pooled_mean": float(np.mean([s["polar_pooled"] for s in per_seed])),
            "per_fold_mean": np.mean([s["fan_per_fold"] for s in per_seed], 0).tolist(),
            "ap_mean": {c: float(np.nanmean([s["fan_ap"][c] if s["fan_ap"][c] is not None else np.nan for s in per_seed]))
                        for c in classes}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/polar/summary.json")
    args = ap.parse_args()
    frames = load_frames()
    classes = class_names(frames)
    fan_gts = [fold_gt(frames, classes, k, 1) for k in range(5)]
    out = {"classes": classes}

    ref = [json.loads((ROOT / "runs" / f"{t}_cv.json").read_text())["float"] for t in FAN_REF]
    ref_folds = np.array([r["map_per_fold"] for r in ref])  # seeds x folds
    out["small_360"] = {"fan_pooled": [r["map_pooled"] for r in ref], "per_fold": ref_folds.tolist(),
                        "fan_pooled_mean": float(np.mean([r["map_pooled"] for r in ref])),
                        "fan_pooled_sd": float(np.std([r["map_pooled"] for r in ref], ddof=1)),
                        "ap_mean": {c: float(np.nanmean([r["ap_pooled"][c] if r["ap_pooled"][c] is not None else np.nan
                                                         for r in ref])) for c in classes}}

    for var, (name, beams) in VARIANTS.items():
        pd.set_beams(beams)
        pf, _ = pd.cached_polar()
        pol_gts = [polar_gt(pf, classes, k) for k in range(5)]
        res = {"beams": beams}
        for kind, fn in PRED.items():
            per_seed = []
            for t in tags(name):
                runs = [ROOT / "runs" / f"{t}_f{k}" for k in range(5)]
                if not all((r / fn).exists() for r in runs):
                    continue
                pol = [load_pred(r / fn) for r in runs]
                fan = [{i: [(d[0], d[1], *pd.polar_box_to_fan(*d[2:])) for d in ds] for i, ds in p.items()} for p in pol]
                aps, m = pool(fan, fan_gts, len(classes))
                _, mp = pool(pol, pol_gts, len(classes))
                per_seed.append({"run": t, "fan_pooled": m, "polar_pooled": mp,
                                 "fan_per_fold": [evaluate(p, g, len(classes))[1] for p, g in zip(fan, fan_gts)],
                                 "fan_ap": {c: (None if np.isnan(a) else float(a)) for c, a in zip(classes, aps)}})
            if not per_seed:
                continue
            r = {"per_seed": per_seed} | summarise(per_seed, classes)
            if kind == "float" and len(per_seed) == len(FAN_REF):
                d = (np.array([s["fan_per_fold"] for s in per_seed]) - ref_folds).ravel()
                r["paired_vs_small_360"] = {"pairs": int(d.size), "mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)),
                                            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
                                            "pooled_diff_per_seed": [s["fan_pooled"] - x for s, x in
                                                                     zip(per_seed, out["small_360"]["fan_pooled"])]}
            res[kind] = r
        out[var] = res

    (ROOT / args.out).write_text(json.dumps(out, indent=1))
    s = out["small_360"]
    print(f"small@360 fan        pooled {100 * s['fan_pooled_mean']:.1f} +- {100 * s['fan_pooled_sd']:.1f}  "
          f"per fold {' '.join(f'{100 * x:.1f}' for x in ref_folds.mean(0))}")
    for var in VARIANTS:
        for kind in PRED:
            r = out[var].get(kind)
            if not r:
                continue
            line = (f"{var} {kind:5s} pooled fan {100 * r['fan_pooled_mean']:.1f} +- {100 * r['fan_pooled_sd']:.1f} "
                    f"(polar-space {100 * r['polar_pooled_mean']:.1f}, {r['seeds']} seeds)  per fold "
                    + " ".join(f"{100 * x:.1f}" for x in r["per_fold_mean"]))
            if "paired_vs_small_360" in r:
                p = r["paired_vs_small_360"]
                line += (f"  | paired diff {100 * p['mean_diff']:+.1f} +- {100 * p['sd']:.1f} "
                         f"({p['n_better']} better / {p['n_worse']} worse of {p['pairs']})")
            print(line)
    print("per-class pooled AP (fan space, seed mean): class  small@360  " + "  ".join(VARIANTS))
    for c in classes:
        print(f"   {c:16s} {100 * s['ap_mean'][c]:5.1f}  " +
              "  ".join(f"{100 * out[v]['float']['ap_mean'][c]:5.1f}" if out[v].get('float') else "  -  " for v in VARIANTS))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
