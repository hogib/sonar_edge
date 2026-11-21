"""MCNN with proposals: the realistic two-stage use of a crop classifier.

Stage 1 (proposals): the deployed detector small@360 ("Det-S"), on-device int8 outputs
(runs/small_360{,_s1,_s2}_cv_f<k>/pred_test_pi4.json, M7), boxes with probability >= 0.3
(logit >= -0.847, the operating threshold of the overlays), class labels DISCARDED,
class-agnostic NMS at IoU 0.5. Stage 2: every proposal is cut from the native 320x480
frame, zero-padded to 305x305 (proposals larger than 305 px are shrunk to fit, keeping
the aspect ratio; counted) and classified by the MCNN model of the same test fold and
seed (runs/mcnn_cv_s<seed>_f<k>, trained on ground-truth crops of the other folds).
A detection is (MCNN class, p_det * p_MCNN, box); MCNN has no Wall class.

Scored with the project evaluator (adapters.score): MCNN task (each GT debris box takes
the class of the highest-scoring proposal with IoU >= 0.5; none = missed) and pooled
AP (over the 10 debris classes, since MCNN cannot output Wall). Cost per frame:
Det-S + n_crops x MCNN. Predictions in native pixels -> runs/mcnn_prop<tag>_s<seed>_f<k>/.
For a like-for-like reference, Det-S alone is scored on the MCNN task with the same
threshold (its own class labels).

  uv run python -m sonar_edge.baselines.mcnn_proposals                 # p >= 0.3
  uv run python -m sonar_edge.baselines.mcnn_proposals --thr-prob 0.018  # logit -4, the evaluation threshold
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..cv import load_pred
from ..data import class_names, load_frames
from ..mcnn_cv import MCNN, PAD
from ..metrics import iou
from .adapters import save_pred, score

ROOT = Path(__file__).resolve().parents[3]
DET = {0: "small_360_cv", 1: "small_360_s1_cv", 2: "small_360_s2_cv"}
SCALE = 4 / 3
DET_MACS, MCNN_MACS = 78.4e6, 234.7e6            # M5 / M7, counted the same way
DET_MS, MCNN_MS_ORT1 = 20.3, 66.5                 # Pi 4, one core: C int8 Det-S; ORT float MCNN per crop (M7)


def nms_agnostic(dets, thr=0.5):
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if all(iou(d[2:], k[2:]) < thr for k in keep):
            keep.append(d)
    return keep


def crop_pad(img, box):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    c = img[y0:y1, x0:x1]
    shrunk = False
    if c.shape[0] > PAD or c.shape[1] > PAD:
        from PIL import Image
        f = PAD / max(c.shape)
        c = np.asarray(Image.fromarray(c).resize((max(1, int(c.shape[1] * f)), max(1, int(c.shape[0] * f))), Image.BOX))
        shrunk = True
    out = np.zeros((PAD, PAD), np.float32)
    h, w = c.shape
    t, l = (PAD - h) // 2, (PAD - w) // 2
    out[t:t + h, l:l + w] = c / 255.0
    return out, shrunk


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr-prob", type=float, default=0.3,
                    help="proposal threshold on the detector probability (0.018 = logit -4, the evaluation threshold)")
    args = ap.parse_args()
    thr_logit = math.log(args.thr_prob / (1 - args.thr_prob))
    tag = "p" + f"{args.thr_prob:g}".replace(".", "")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_frames()
    classes = class_names(frames)
    by_id = {f.index: f for f in frames}
    summary = {"threshold_prob": args.thr_prob, "runs": []}
    for seed, det in DET.items():
        preds, det_preds, n_crops, n_frames, n_shrunk = [], [], 0, 0, 0
        for k in range(5):
            ck = torch.load(ROOT / "runs" / f"mcnn_cv_s{seed}_f{k}" / "best.pt", map_location="cpu", weights_only=False)
            mcls = ck["classes"]
            to11 = [classes.index(c) for c in mcls]
            m = MCNN(len(mcls))
            m.load_state_dict(ck["model"])
            m.to(device).eval()
            dp = load_pred(ROOT / "runs" / f"{det}_f{k}" / "pred_test_pi4.json")
            out, dout = {}, {}
            for idx, ds in dp.items():
                kept = [(d[0], d[1], *(v * SCALE for v in d[2:])) for d in ds if d[1] >= thr_logit]
                dout[idx] = kept
                props = nms_agnostic(kept)
                n_frames += 1
                n_crops += len(props)
                if not props:
                    out[idx] = []
                    continue
                img = by_id[idx].load()
                xs = []
                for p in props:
                    c, sh = crop_pad(img, p[2:])
                    xs.append(c)
                    n_shrunk += sh
                with torch.no_grad():
                    pr = torch.softmax(m(torch.from_numpy(np.stack(xs))[:, None].to(device)), 1).cpu().numpy()
                out[idx] = [(to11[int(q.argmax())], float(1 / (1 + math.exp(-p[1])) * q.max()), *p[2:])
                            for p, q in zip(props, pr)]
            preds.append(out)
            det_preds.append(dout)
            r = ROOT / "runs" / f"mcnn_prop{tag}_s{seed}_f{k}"
            r.mkdir(parents=True, exist_ok=True)
            save_pred(out, r / "pred_test.json")
            torch.save({"scale": 1.0, "note": "shim for material.py; predictions in native pixels"}, r / "best.pt")
        s = score(preds, 1.0, frames, classes)
        sd = score(det_preds, 1.0, frames, classes)
        debris = [c for c in classes if c != "Wall"]
        s["map10_pooled"] = float(np.mean([s["ap_pooled"][c] for c in debris]))
        cpf = n_crops / n_frames
        s |= {"seed": seed, "detector": det, "crops_per_frame": cpf, "n_shrunk": n_shrunk,
              "det_alone_same_thr": sd["mcnn_task"],
              "macs_per_frame": DET_MACS + cpf * MCNN_MACS, "pi4_ms_per_frame_est": DET_MS + cpf * MCNN_MS_ORT1}
        summary["runs"].append(s)
        t, td = s["mcnn_task"], sd["mcnn_task"]
        print(f"seed {seed}: MCNN+prop acc {100 * t['acc']:.2f} F1 {100 * t['macro_f1']:.2f} missed {100 * t['missed']:.1f}%"
              f" | Det-S same thr acc {100 * td['acc']:.2f} F1 {100 * td['macro_f1']:.2f} missed {100 * td['missed']:.1f}%"
              f" | mAP10 {100 * s['map10_pooled']:.2f} crops/frame {cpf:.2f} (shrunk {n_shrunk})", flush=True)
    sdv = lambda v: float(np.std(v, ddof=1))
    for key in ("acc", "macro_f1", "missed"):
        v = np.array([r["mcnn_task"][key] for r in summary["runs"]])
        summary[f"{key}_mean"], summary[f"{key}_sd"] = float(v.mean()), sdv(v)
        v = np.array([r["det_alone_same_thr"][key] for r in summary["runs"]])
        summary[f"det_{key}_mean"], summary[f"det_{key}_sd"] = float(v.mean()), sdv(v)
    v = np.array([r["map10_pooled"] for r in summary["runs"]])
    summary["map10_mean"], summary["map10_sd"] = float(v.mean()), sdv(v)
    c = float(np.mean([r["crops_per_frame"] for r in summary["runs"]]))
    summary["crops_per_frame_mean"] = c
    summary["cost"] = {"det_macs": DET_MACS, "mcnn_macs_per_crop": MCNN_MACS, "macs_per_frame": DET_MACS + c * MCNN_MACS,
                       "pi4_ms_det_c_int8": DET_MS, "pi4_ms_mcnn_ort1_per_crop": MCNN_MS_ORT1,
                       "pi4_ms_per_frame_est": DET_MS + c * MCNN_MS_ORT1}
    print(f"mean: MCNN+prop acc {100 * summary['acc_mean']:.2f} +- {100 * summary['acc_sd']:.2f} F1 "
          f"{100 * summary['macro_f1_mean']:.2f} +- {100 * summary['macro_f1_sd']:.2f} missed {100 * summary['missed_mean']:.1f}%"
          f" | Det-S acc {100 * summary['det_acc_mean']:.2f} +- {100 * summary['det_acc_sd']:.2f} F1 "
          f"{100 * summary['det_macro_f1_mean']:.2f} +- {100 * summary['det_macro_f1_sd']:.2f} missed {100 * summary['det_missed_mean']:.1f}%"
          f" | crops/frame {c:.2f}, {summary['cost']['macs_per_frame'] / 1e6:.0f} M MACs, ~{summary['cost']['pi4_ms_per_frame_est']:.0f} ms/frame Pi 4")
    dest = ROOT / "bench" / "baselines" / f"mcnn_proposals_{tag}_summary.json"
    dest.write_text(json.dumps(summary, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
