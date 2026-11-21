"""SSDLite-MobileNetV3-Large (torchvision) under the detector's CV protocol.

Input 256x384 (W x H): the 320x480 frame box-filtered by 1.25, the stride-32-friendly
size closest to the detector's 240x360 (scale 4/3); the grey frame is repeated to 3
channels. Augmentation is the detector's "strong" set (flip, shift, zoom, noise,
gain/bias), copied from FrameDataset. Backbone ImageNet-pretrained (torchvision
default for SSDLite; the baseline's best realistic setting), detection head from
scratch. --scratch: the same full architecture with a re-initialised backbone
(torchvision would otherwise build a reduced backbone tail without weights). Optimiser as the detector: AdamW + one-cycle, 150 epochs, batch 16. Model
selection on the validation fold's mAP (evaluated every `--eval-every` epochs and at
the end). Predictions are written in native frame pixels and scored with
sonar_edge.baselines.adapters, i.e. the same evaluator and GT as the detector.

  uv run --extra baselines python -m sonar_edge.baselines.ssdlite train --seeds 0 1 2
  uv run --extra baselines python -m sonar_edge.baselines.ssdlite summary --seeds 0 1 2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..cv import load_pred
from ..data import class_names, load_frames
from ..dataset import FrameDataset
from ..metrics import evaluate
from ..split import load_split
from .adapters import save_pred, score

ROOT = Path(__file__).resolve().parents[3]
SCALE = 1.25  # 320x480 -> 256x384


class BoxDataset(FrameDataset):
    """FrameDataset's images and augmentation, returning boxes instead of CenterNet targets."""

    def __getitem__(self, i):
        f = self.frames[i]
        img = self._prep(self.images[i])
        boxes = [(self.cidx[b.cls], b.x / self.scale, b.y / self.scale,
                  (b.x + b.w) / self.scale, (b.y + b.h) / self.scale) for b in f.boxes]
        if self.train:  # same steps and parameters as FrameDataset.__getitem__
            if self.rng.random() < 0.5:
                img = img[:, ::-1]
                boxes = [(c, self.w - x1, y0, self.w - x0, y1) for c, x0, y0, x1, y1 in boxes]
            j = max(1, int(8 / self.scale))
            dx, dy = (int(v) for v in self.rng.integers(-j, j + 1, size=2))
            if dx or dy:
                img = np.roll(img, (dy, dx), axis=(0, 1))
                boxes = [(c, np.clip(x0 + dx, 0, self.w), np.clip(y0 + dy, 0, self.h),
                          np.clip(x1 + dx, 0, self.w), np.clip(y1 + dy, 0, self.h)) for c, x0, y0, x1, y1 in boxes]
                boxes = [b for b in boxes if b[3] - b[1] >= 2 and b[4] - b[2] >= 2]
            if self.aug == "strong" and self.rng.random() < 0.7:
                img, boxes, _ = self._zoom(img, boxes)
            if self.aug == "strong":
                img = img + self.rng.normal(0, self.rng.uniform(0, 0.04), img.shape).astype(np.float32)
            img = np.clip(img * self.rng.uniform(0.7, 1.3) + self.rng.uniform(-0.1, 0.1), 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(img, np.float32))[None].expand(3, -1, -1)
        b = torch.tensor([bb[1:] for bb in boxes], dtype=torch.float32).reshape(-1, 4)
        lab = torch.tensor([bb[0] + 1 for bb in boxes], dtype=torch.int64)  # 0 = background
        return img, {"boxes": b, "labels": lab}, f.index


def collate(batch):
    return tuple(zip(*batch))


def make_model(n_classes, pretrained=True):
    import torchvision
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    # torchvision builds a reduced MobileNetV3 tail when no backbone weights are given;
    # build the full architecture in both settings and, for scratch, re-initialise the
    # backbone (torchvision's MobileNetV3 init), so the two settings differ only in init.
    wb = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    m = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=wb, num_classes=n_classes + 1,
                                      score_thresh=0.01, detections_per_img=100)
    if not pretrained:
        import torch.nn as nn
        for mod in m.backbone.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)
                if isinstance(mod, nn.BatchNorm2d):
                    mod.reset_running_stats()
            elif isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0, 0.01)
                nn.init.zeros_(mod.bias)
    h, w = int(round(480 / SCALE)), int(round(320 / SCALE))
    # images arrive at 256x384 already; keep the model's normalisation, no resize
    m.transform = GeneralizedRCNNTransform(min(h, w), max(h, w), m.transform.image_mean, m.transform.image_std,
                                           size_divisible=1, fixed_size=(w, h))
    return m


@torch.no_grad()
def predict(model, ds, device, bs=32):
    """-> image index -> [(cls, score, x0, y0, x1, y1)] in NATIVE frame pixels."""
    model.eval()
    out = {}
    dl = torch.utils.data.DataLoader(ds, bs, shuffle=False, num_workers=0, collate_fn=collate)
    for imgs, _, idx in dl:
        res = model([im.to(device) for im in imgs])
        for r, i in zip(res, idx):
            out[i] = [(int(l) - 1, float(s), *(float(v) * SCALE for v in bx))
                      for bx, l, s in zip(r["boxes"].cpu(), r["labels"].cpu(), r["scores"].cpu())]
    return out


def native_gt(ds):
    return {i: [(c, *(v * SCALE for v in b)) for c, *b in g] for i, g in ds.gt_boxes().items()}


def train_fold(k, seed, args, device, frames, classes):
    torch.manual_seed(seed)
    v = (k + 1) % 5
    split = load_split(k, v)
    parts = {s: [f for f in frames if split[f.index] == s] for s in ("train", "val", "test")}
    ds = {s: BoxDataset(p, classes, 8, SCALE, train=(s == "train"), seed=seed, aug="strong") for s, p in parts.items()}
    model = make_model(len(classes), not args.scratch).to(device)
    dl = torch.utils.data.DataLoader(ds["train"], args.bs, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    gt_val = native_gt(ds["val"])
    best, log, t0 = (-1.0, 0, None), [], time.time()
    for ep in range(args.epochs):
        model.train()
        te, tot = time.time(), 0.0
        for imgs, tg, _ in dl:
            ld = model([im.to(device) for im in imgs], [{kk: vv.to(device) for kk, vv in t.items()} for t in tg])
            loss = sum(ld.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        entry = {"epoch": ep + 1, "loss": tot / len(dl), "s": round(time.time() - te, 1)}
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            m = evaluate(predict(model, ds["val"], device), gt_val, len(classes))[1]
            entry["val_map"] = m
            if m > best[0]:
                best = (m, ep + 1, {n: t.detach().cpu().clone() for n, t in model.state_dict().items()})
        log.append(entry)
        if args.verbose:
            print(json.dumps(entry), flush=True)
    model.load_state_dict(best[2])
    return model, predict(model, ds["test"], device), {"seed": seed, "test_fold": k, "val_fold": v, "best_epoch": best[1],
                                                       "val_map": best[0], "train_s": round(time.time() - t0, 1), "log": log}


def run_dir(seed, k, scratch):
    return ROOT / "runs" / f"ssdlite{'_scratch' if scratch else ''}_s{seed}_f{k}"


def cmd_train(args):
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_frames()
    classes = class_names(frames)
    for seed in args.seeds:
        for k in args.folds:
            out = run_dir(seed, k, args.scratch)
            if (out / "results.json").exists() and not args.force:
                print(f"skip {out.name} (done)", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            model, pred, r = train_fold(k, seed, args, device, frames, classes)
            torch.save({"model": model.state_dict(), "classes": classes, "scale": SCALE}, out / "best.pt")
            save_pred(pred, out / "pred_test.json")
            (out / "results.json").write_text(json.dumps(r))
            print(f"ssdlite seed {seed} fold {k}: best epoch {r['best_epoch']}, val mAP {100 * r['val_map']:.1f}, "
                  f"{r['train_s']:.0f} s", flush=True)


def cmd_summary(args):
    frames = load_frames()
    classes = class_names(frames)
    res = []
    for seed in args.seeds:
        preds = [load_pred(run_dir(seed, k, args.scratch) / "pred_test.json") for k in range(5)]
        s = score(preds, 1.0, frames, classes)
        res.append({"seed": seed} | s)
        print(f"seed {seed}: pooled mAP {100 * s['map_pooled']:.2f}  per fold "
              + " ".join(f"{100 * x:.1f}" for x in s["map_per_fold"])
              + f"  MCNN task acc {100 * s['mcnn_task']['acc']:.2f} F1 {100 * s['mcnn_task']['macro_f1']:.2f}")
    m = np.array([r["map_pooled"] for r in res])
    print(f"mean pooled mAP {100 * m.mean():.2f} +- {100 * (m.std(ddof=1) if len(m) > 1 else 0):.2f}")
    dest = ROOT / "bench" / "baselines" / f"ssdlite{'_scratch' if args.scratch else ''}_summary.json"
    dest.write_text(json.dumps({"runs": res, "map_mean": float(m.mean()),
                                "map_sd": float(m.std(ddof=1)) if len(m) > 1 else None}, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser("train")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--eval-every", type=int, default=5)
    t.add_argument("--scratch", action="store_true", help="no ImageNet backbone weights")
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--verbose", action="store_true")
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_train)
    s = sub.add_parser("summary")
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--scratch", action="store_true")
    s.set_defaults(func=cmd_summary)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
