"""Collect, score and export the detector baselines.

  uv run --extra baselines python -m sonar_edge.baselines.collect pull     # rsync predictions/checkpoints from the GPU server
  uv run --extra baselines python -m sonar_edge.baselines.collect score    # all families with complete seeds
  uv run --extra baselines python -m sonar_edge.baselines.collect onnx     # SSDLite ONNX (YOLO/NanoDet: see README)

Families: <tool>_<init>, runs/<family>_s<seed>_f<k>/pred_test.json in native pixels
(SSDLite pretrained = runs/ssdlite_s*, trained on the laptop). `score` writes
bench/baselines/detectors_summary.json (pooled mAP over 11 classes and over the 10
debris classes, per fold, per class, MCNN task) and, through shim run dirs
runs/bl_<family>_s<seed>_f<k> (pred_test.json + best.pt {"scale": 1.0}), runs
sonar_edge.material for the microplastic task -> bench/baselines/material_summary.json.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..cv import load_pred
from ..data import class_names, load_frames
from .adapters import score

ROOT = Path(__file__).resolve().parents[3]
SERVER = "server@100.123.239.75:Projeler/oguz/sonar_edge"
FAMILIES = {"ssdlite_pretrained": "ssdlite", "ssdlite_scratch": "ssdlite_scratch",
            "yolo_pretrained": "yolo_pretrained", "yolo_scratch": "yolo_scratch",
            "nanodet_pretrained": "nanodet_pretrained", "nanodet_scratch": "nanodet_scratch"}


def run(family, seed, k):
    return ROOT / "runs" / f"{FAMILIES[family]}_s{seed}_f{k}"


def complete_seeds(family):
    return [s for s in range(3) if all((run(family, s, k) / "pred_test.json").exists() for k in range(5))]


def cmd_pull(args):
    for pat in ("yolo_*", "nanodet_*", "ssdlite_scratch_*"):
        subprocess.run(["rsync", "-a", "--include", pat + "/", "--include", pat + "/pred_test.json",
                        "--include", pat + "/results.json", "--include", pat + "/info.json", "--include", pat + "/results.csv",
                        "--include", pat + "/best.pt", "--include", pat + "/weights/", "--include", pat + "/weights/best.pt",
                        "--exclude", "*", f"{SERVER}/runs/", str(ROOT / "runs") + "/"], check=True)
    subprocess.run(["rsync", "-a", f"{SERVER}/bench/baselines/logs/", str(ROOT / "bench/baselines/logs/server") + "/"],
                   check=True)
    for f in FAMILIES:
        print(f"{f:20s} complete seeds {complete_seeds(f)}")


def cmd_score(args):
    frames = load_frames()
    classes = class_names(frames)
    debris = [c for c in classes if c != "Wall"]
    out, mat_args = {}, []
    for fam in FAMILIES:
        seeds = complete_seeds(fam)
        if not seeds:
            continue
        rows = []
        for s in seeds:
            preds = [load_pred(run(fam, s, k) / "pred_test.json") for k in range(5)]
            r = score(preds, 1.0, frames, classes)
            r["map10_pooled"] = float(np.mean([r["ap_pooled"][c] for c in debris]))
            rows.append({"seed": s} | r)
            for k in range(5):  # shim dirs for material.py (it rescales by best.pt["scale"])
                d = ROOT / "runs" / f"bl_{fam}_s{s}_f{k}"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(run(fam, s, k) / "pred_test.json", d / "pred_test.json")
                torch.save({"scale": 1.0, "note": "shim: native-pixel predictions"}, d / "best.pt")
        mat_args.append([f"runs/bl_{fam}_s{s}" for s in seeds])
        agg = lambda key: (float(np.mean([r[key] for r in rows])), float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else None)
        aggt = lambda key: (float(np.mean([r["mcnn_task"][key] for r in rows])),
                            float(np.std([r["mcnn_task"][key] for r in rows], ddof=1)) if len(rows) > 1 else None)
        out[fam] = {"seeds": seeds, "per_seed": rows, "map11": agg("map_pooled"), "map10": agg("map10_pooled"),
                    "mcnn_acc": aggt("acc"), "mcnn_f1": aggt("macro_f1"), "missed": aggt("missed"),
                    "map_per_fold_mean": np.mean([r["map_per_fold"] for r in rows], 0).tolist(),
                    "ap_pooled_mean": {c: float(np.mean([r["ap_pooled"][c] for r in rows])) for c in classes}}
        f = lambda t: f"{100 * t[0]:.1f}" + (f"±{100 * t[1]:.1f}" if t[1] is not None else "")
        print(f"{fam:20s} seeds {seeds}  mAP11 {f(out[fam]['map11'])}  mAP10 {f(out[fam]['map10'])}  "
              f"MCNN-task acc {f(out[fam]['mcnn_acc'])} F1 {f(out[fam]['mcnn_f1'])} missed {f(out[fam]['missed'])}")
    dest = ROOT / "bench" / "baselines" / "detectors_summary.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")
    if mat_args and not args.no_material:
        cmd = ["uv", "run", "python", "-m", "sonar_edge.material", "--variant", "float", "--mcnn-seeds",
               "--out", "bench/baselines/material_summary.json"]
        for a in mat_args:
            cmd += ["--det", *a]
        subprocess.run(cmd, cwd=ROOT, check=True)


def cmd_onnx(args):
    """Network-only ONNX of each SSDLite family's s0 f0 checkpoint (the graph timed on the Pi;
    torchvision's full model would also contain resize, anchor decoding and NMS)."""
    from .pi_latency import export_ssdlite
    for fam in ("ssdlite_pretrained", "ssdlite_scratch"):
        if not (run(fam, 0, 0) / "best.pt").exists():
            print(f"{fam}: no checkpoint yet")
            continue
        export_ssdlite(run(fam, 0, 0), run(fam, 0, 0) / "model.onnx")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    sub.add_parser("pull").set_defaults(func=cmd_pull)
    s = sub.add_parser("score")
    s.add_argument("--no-material", action="store_true")
    s.set_defaults(func=cmd_score)
    sub.add_parser("onnx").set_defaults(func=cmd_onnx)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
