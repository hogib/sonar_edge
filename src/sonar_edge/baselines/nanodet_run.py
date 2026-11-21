"""NanoDet-Plus-m (RangiLyu/nanodet, Apache-2.0) under the detector's CV protocol.

NanoDet pins torch<2.0 and pytorch-lightning 1.x, which do not run on current GPUs
(sm_120). We therefore use NanoDet's own model and loss code (build_model,
NanoDetPlus.forward_train with the auxiliary assigner head, head.get_bboxes) on torch
2.15 in .venv-nanodet, with a plain training loop that reproduces the config's
schedule (config/nanodet-plus-m_320.yml at ext/nanodet commit be9b4a9): AdamW lr 1e-3
wd 0.05, linear warm-up 500 steps from ratio 1e-4, cosine to 5e-5, gradient clipping
35, EMA of the weights (decay 0.9998), auxiliary head detached after epoch 10. Batch 16
(config: 96 on 8 GPUs). Input 256x384 (W x H) as for every baseline; augmentation is
the detector's "strong" set (BoxDataset in ssdlite.py), the grey frame repeated to 3
channels and normalised with the config's mean/std. Initialisation: ImageNet
ShuffleNetV2-1.0x backbone (config default, --init pretrained) or none (--init
scratch). Model selection on the validation fold's mAP (EMA weights, every 5 epochs).
Test predictions (score >= 0.05, NMS 0.6, max 100: NanoDet's decoder) in native pixels
-> runs/nanodet_<init>_s<seed>_f<k>/pred_test.json.

  .venv-nanodet/bin/python src/sonar_edge/baselines/nanodet_run.py --init pretrained --epochs 60
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sonar_edge.baselines.adapters import save_pred  # noqa: E402
from sonar_edge.baselines.ssdlite import SCALE, BoxDataset, collate, native_gt  # noqa: E402
from sonar_edge.data import class_names, load_frames  # noqa: E402
from sonar_edge.metrics import evaluate  # noqa: E402
from sonar_edge.split import load_split  # noqa: E402

CFG = ROOT / "ext" / "nanodet" / "config" / "nanodet-plus-m_320.yml"
MEAN = torch.tensor([103.53, 116.28, 123.675]).view(1, 3, 1, 1)
STD = torch.tensor([57.375, 57.12, 58.395]).view(1, 3, 1, 1)


def build(n_classes, pretrained):
    from nanodet.model.arch import build_model
    from nanodet.util import cfg, load_config

    load_config(cfg, str(CFG))
    cfg.defrost()
    cfg.model.arch.head.num_classes = n_classes
    cfg.model.arch.aux_head.num_classes = n_classes
    cfg.model.arch.backbone.pretrain = pretrained
    return build_model(cfg.model)


def to_input(imgs, device):
    x = torch.stack(imgs).to(device) * 255.0  # B,3,H,W grey repeated, 0..255
    return (x - MEAN.to(device)) / STD.to(device)


@torch.no_grad()
def predict(model, ds, device, n_classes, bs=32):
    model.eval()
    out = {}
    dl = torch.utils.data.DataLoader(ds, bs, shuffle=False, num_workers=0, collate_fn=collate)
    for imgs, _, idx in dl:
        x = to_input(imgs, device)
        preds = model(x)
        cls, reg = preds.split([n_classes, 4 * (model.head.reg_max + 1)], dim=-1)
        res = model.head.get_bboxes(cls, reg, {"img": x})
        for (det, lab), i in zip(res, idx):
            out[i] = [(int(l), float(d[4]), *(float(v) * SCALE for v in d[:4]))
                      for d, l in zip(det.cpu().numpy(), lab.cpu().numpy())]
    return out


def train_fold(k, seed, args, device, frames, classes):
    torch.manual_seed(seed)
    v = (k + 1) % 5
    split = load_split(k, v)
    parts = {s: [f for f in frames if split[f.index] == s] for s in ("train", "val", "test")}
    ds = {s: BoxDataset(p, classes, 8, SCALE, train=(s == "train"), seed=seed, aug="strong") for s, p in parts.items()}
    model = build(len(classes), args.init == "pretrained").to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    dl = torch.utils.data.DataLoader(ds["train"], args.bs, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    total = args.epochs * len(dl)
    warm, lr0, lr_min = 500, 1e-3, 5e-5

    def lr_at(step):
        if step < warm:
            return lr0 * (1e-4 + (1 - 1e-4) * step / warm)
        t = (step - warm) / max(1, total - warm)
        return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * t))

    gt_val = native_gt(ds["val"])
    best, log, t0, step = (-1.0, 0, None), [], time.time(), 0
    for ep in range(args.epochs):
        model.train()
        model.set_epoch(ep)
        te, tot = time.time(), 0.0
        for imgs, tg, _ in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            meta = {"img": to_input(imgs, device),
                    "gt_bboxes": [t["boxes"].numpy().astype(np.float32) for t in tg],
                    "gt_labels": [(t["labels"].numpy() - 1).astype(np.int64) for t in tg],
                    "gt_bboxes_ignore": [np.zeros((0, 4), np.float32) for _ in tg]}
            _, loss, _ = model.forward_train(meta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
            opt.step()
            step += 1
            d = 0.9998 * (1 - math.exp(-step / 2000))
            with torch.no_grad():
                for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                    if pe.dtype.is_floating_point:
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    else:
                        pe.copy_(pm)
            tot += loss.item()
        entry = {"epoch": ep + 1, "loss": tot / len(dl), "s": round(time.time() - te, 1)}
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            m = evaluate(predict(ema, ds["val"], device, len(classes)), gt_val, len(classes))[1]
            entry["val_map"] = m
            if m > best[0]:
                best = (m, ep + 1, {n: t.detach().cpu().clone() for n, t in ema.state_dict().items()})
        log.append(entry)
        if args.verbose:
            print(json.dumps(entry), flush=True)
    ema.load_state_dict(best[2])
    return ema, predict(ema, ds["test"], device, len(classes)), {
        "init": args.init, "seed": seed, "test_fold": k, "val_fold": v, "best_epoch": best[1], "val_map": best[0],
        "epochs": args.epochs, "train_s": round(time.time() - t0, 1), "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["pretrained", "scratch"], required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_frames()
    classes = class_names(frames)
    for seed in args.seeds:
        for k in args.folds:
            out = ROOT / "runs" / f"nanodet_{args.init}_s{seed}_f{k}"
            if (out / "pred_test.json").exists():
                print(f"skip {out.name} (done)", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            model, pred, r = train_fold(k, seed, args, device, frames, classes)
            torch.save({"model": model.state_dict(), "classes": classes, "scale": SCALE}, out / "best.pt")
            save_pred(pred, out / "pred_test.json")
            (out / "results.json").write_text(json.dumps(r))
            print(f"{out.name}: best epoch {r['best_epoch']}, val mAP {100 * r['val_map']:.1f}, {r['train_s']:.0f} s",
                  flush=True)


if __name__ == "__main__":
    main()
