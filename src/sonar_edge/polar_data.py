"""Polar-input detector data (M8 experiment): the dataset's fan images and boxes in beam x range space.

The dataset has no raw ARIS data, so the "polar" frames here are back-projected
from the 320x480 fan images with the measured fan geometry (sonar_edge.polar.FAN):
rows = range bins near -> far, columns = beams left -> right. This is one
bilinear resampling of the full-resolution fan image, the same count as the fan
model's BOX resize to 360x240.

Boxes. A fan-image box is not a rectangle in polar space. Each GT box is mapped
through its object pixels: the label-mask pixels of the box's class inside the
box (both from the dataset), each pixel's four corners transformed to
(range bin, beam), and the polar bounding box taken. A box with no such mask
pixels falls back to its outline, sampled densely. Predictions are mapped back
to the fan image the same way (polar-box outline -> fan bounding box), so a
polar model can be scored against the original fan GT with the existing
evaluator.

  uv run python -m sonar_edge.polar_data check [--n 6] [--out DIR]   # round-trip IoU stats + overlays
"""

import argparse
from pathlib import Path

import numpy as np

from .data import Box, Frame, class_names, load_frames
from .polar import FAN, bilinear

N_BEAMS = 128      # ARIS Explorer 3000; set_beams() changes it (e.g. 192 for a compute-matched input)
N_SAMPLES = 440    # range bins: (578.0 - 138.6) / 440 = 0.999 fan px per bin, divisible by the stride 8


def set_beams(n):
    """Number of beam columns of the polar grid (sampled directly from the fan image at n angles)."""
    global N_BEAMS
    N_BEAMS = int(n)


def fan_to_polar(u, v, n_samples=N_SAMPLES, n_beams=None, g=FAN):
    """Continuous fan-image coordinates -> continuous (row, col) of the polar grid (bin edges at integers)."""
    n_beams = n_beams or N_BEAMS
    du, dv = np.asarray(u, float) - g["apex_u"], g["apex_v"] - np.asarray(v, float)
    r = np.hypot(du, dv)
    th = np.degrees(np.arctan2(du, dv))
    row = (r - g["r_min_px"]) / (g["r_max_px"] - g["r_min_px"]) * n_samples
    col = (th + g["fov_deg"] / 2) / g["fov_deg"] * n_beams
    return row, col


def polar_to_fan(row, col, n_samples=N_SAMPLES, n_beams=None, g=FAN):
    n_beams = n_beams or N_BEAMS
    r = g["r_min_px"] + np.asarray(row, float) * (g["r_max_px"] - g["r_min_px"]) / n_samples
    th = np.radians(-g["fov_deg"] / 2 + np.asarray(col, float) * g["fov_deg"] / n_beams)
    return g["apex_u"] + r * np.sin(th), g["apex_v"] - r * np.cos(th)


def grid(n_samples=N_SAMPLES, n_beams=None):
    """Fan coordinates of every polar cell centre."""
    n_beams = n_beams or N_BEAMS
    rr, cc = np.meshgrid(np.arange(n_samples) + 0.5, np.arange(n_beams) + 0.5, indexing="ij")
    return polar_to_fan(rr, cc, n_samples, n_beams)


def image_to_polar(img, gu, gv):
    return np.clip(np.round(bilinear(img.astype(np.float64), gu, gv)), 0, 255).astype(np.uint8)


def mask_to_polar(mask, gu, gv):
    """Nearest-neighbour label lookup; outside the image -> 0."""
    x, y = np.floor(gu).astype(int), np.floor(gv).astype(int)
    ok = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    out = np.zeros(gu.shape, np.uint8)
    out[ok] = mask[y[ok], x[ok]]
    return out


def _outline(x0, y0, x1, y1, n=64):
    t = np.linspace(0, 1, n)
    xs = np.concatenate([x0 + (x1 - x0) * t, x0 + (x1 - x0) * t, np.full(n, x0), np.full(n, x1)])
    ys = np.concatenate([np.full(n, y0), np.full(n, y1), y0 + (y1 - y0) * t, y0 + (y1 - y0) * t])
    return xs, ys


def _clipbox(r0, c0, r1, c1, n_samples=N_SAMPLES, n_beams=None):
    n_beams = n_beams or N_BEAMS
    return (float(np.clip(c0, 0, n_beams)), float(np.clip(r0, 0, n_samples)),
            float(np.clip(c1, 0, n_beams)), float(np.clip(r1, 0, n_samples)))


def box_to_polar(b: Box, mask, cls_label):
    """Fan GT box -> polar (x0, y0, x1, y1) = (beam0, bin0, beam1, bin1), and whether the mask was used."""
    used = False
    if mask is not None:
        sub = mask[b.y : b.y + b.h, b.x : b.x + b.w] == cls_label
        ys, xs = np.nonzero(sub)
        if len(xs):
            xs, ys = xs + b.x, ys + b.y
            cu = np.concatenate([xs, xs + 1, xs, xs + 1]).astype(float)
            cv = np.concatenate([ys, ys, ys + 1, ys + 1]).astype(float)
            used = True
    if not used:
        cu, cv = _outline(b.x, b.y, b.x + b.w, b.y + b.h)
    row, col = fan_to_polar(cu, cv)
    return _clipbox(row.min(), col.min(), row.max(), col.max()), used


def polar_box_to_fan(x0, y0, x1, y1):
    """Polar box (beam0, bin0, beam1, bin1) -> fan-image bounding box of its outline."""
    cols, rows = _outline(x0, y0, x1, y1)
    u, v = polar_to_fan(rows, cols)
    return float(u.min()), float(v.min()), float(u.max()), float(v.max())


def polar_frames(frames=None, classes=None):
    """(frames with polar boxes, polar images N x N_SAMPLES x N_BEAMS uint8, stats)."""
    frames = frames if frames is not None else load_frames()
    classes = classes or class_names(frames)
    gu, gv = grid()
    imgs, out, n_mask, n_box = [], [], 0, 0
    for f in frames:
        imgs.append(image_to_polar(f.load(), gu, gv))
        m = f.load_mask() if f.mask_path is not None else None
        boxes = []
        for b in f.boxes:
            (x0, y0, x1, y1), used = box_to_polar(b, m, classes.index(b.cls) + 1)
            n_box += 1
            n_mask += used
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                boxes.append(Box(b.cls, x0, y0, x1 - x0, y1 - y0))
        out.append(Frame(f.index, f.image_path, boxes, f.mask_path))
    return out, np.stack(imgs), {"boxes": n_box, "from_mask": n_mask}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def cmd_check(args):
    """Round trip GT fan box -> polar box -> fan box, IoU vs the original; overlays for a few frames."""
    frames = load_frames()
    classes = class_names(frames)
    pf, imgs, st = polar_frames(frames, classes)
    ious = {c: [] for c in classes}
    for f in frames:
        m = f.load_mask() if f.mask_path is not None else None
        for b in f.boxes:
            q, _ = box_to_polar(b, m, classes.index(b.cls) + 1)
            ious[b.cls].append(iou(polar_box_to_fan(*q), b.xyxy) if q[2] > q[0] and q[3] > q[1] else 0.0)
    n_kept = sum(len(p.boxes) for p in pf)
    print(f"{st['boxes']} GT boxes, {st['from_mask']} from mask pixels, {n_kept} kept in polar")
    allv = np.concatenate([np.array(v) for v in ious.values()])
    print(f"round-trip IoU (fan GT -> polar -> fan) median {np.median(allv):.3f}, p10 {np.percentile(allv, 10):.3f}, "
          f"< 0.5: {np.mean(allv < 0.5) * 100:.1f}%")
    for c in classes:
        v = np.array(ious[c])
        print(f"   {c:16s} n {len(v):4d}  median {np.median(v):.3f}  p10 {np.percentile(v, 10):.3f}  <0.5 {np.mean(v < 0.5) * 100:5.1f}%")
    if args.out:
        from PIL import Image, ImageDraw
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for i in rng.choice(len(frames), args.n, replace=False):
            f, p = frames[i], pf[i]
            fan = Image.fromarray(f.load()).convert("RGB")
            pol = Image.fromarray(imgs[i][::-1]).convert("RGB").resize((N_BEAMS * 2, N_SAMPLES))  # far at top
            df, dp = ImageDraw.Draw(fan), ImageDraw.Draw(pol)
            for b in f.boxes:
                df.rectangle(b.xyxy, outline=(255, 0, 0))
            for q in p.boxes:
                back = polar_box_to_fan(q.x, q.y, q.x + q.w, q.y + q.h)
                df.rectangle(back, outline=(0, 255, 0))
                dp.rectangle((2 * q.x, N_SAMPLES - (q.y + q.h), 2 * (q.x + q.w), N_SAMPLES - q.y), outline=(0, 255, 0))
            canvas = Image.new("RGB", (fan.width + pol.width + 10, max(fan.height, pol.height)))
            canvas.paste(fan, (0, 0))
            canvas.paste(pol, (fan.width + 10, 0))
            canvas.save(out / f"polar_check_{f.index}.png")
        print(f"wrote {args.n} overlays to {out}")


DATA = Path(__file__).resolve().parents[2] / "data"


def cache_path():
    return DATA / ("polar_cache_v1.npz" if N_BEAMS == 128 else f"polar_cache_v1_b{N_BEAMS}.npz")


def cached_polar():
    """(polar frames, images), computed once and cached under data/ (git-ignored)."""
    import json
    frames = load_frames()
    CACHE = cache_path()
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        boxes = json.loads(str(z["boxes"]))
        pf = [Frame(f.index, f.image_path, [Box(*b) for b in boxes[str(f.index)]], f.mask_path) for f in frames]
        return pf, z["images"]
    pf, imgs, _ = polar_frames(frames)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, images=imgs, boxes=json.dumps({str(f.index): [[b.cls, b.x, b.y, b.w, b.h] for b in f.boxes] for f in pf}))
    return pf, imgs


def dev_polar_datasets(stride, test_fold=0, val_fold=1, seed=0, aug="basic", beams=128):
    """Same split and recipe as dataset.dev_datasets, on polar frames (N_SAMPLES x N_BEAMS, rows near -> far)."""
    from .dataset import FrameDataset
    from .split import load_split

    set_beams(beams)
    pf, imgs = cached_polar()
    classes = class_names(load_frames())
    split = load_split(test_fold, val_fold)
    ds = {}
    for k in ("train", "val", "test"):
        idx = [i for i, f in enumerate(pf) if split[f.index] == k]
        d = FrameDataset.__new__(FrameDataset)  # bypass the fan-image loading of __init__
        d.frames, d.images, d.masks = [pf[i] for i in idx], imgs[idx], None
        d.aug, d.seg, d.seg_up = aug, False, 1
        d.cidx, d.n_classes = {c: i for i, c in enumerate(classes)}, len(classes)
        d.stride, d.scale, d.train, d.max_objs = stride, 1, k == "train", 16
        d.rng = np.random.default_rng(seed)
        d.h, d.w = N_SAMPLES, N_BEAMS
        d.gh, d.gw = d.h // stride, d.w // stride
        ds[k] = d
    return ds, classes


def cmd_fanscore(args):
    """Map a polar run's test detections back to the fan image and score them against the original fan GT."""
    import json
    from .metrics import evaluate
    from .split import load_split

    run = Path(args.run)
    frames = load_frames()
    classes = class_names(frames)
    split = load_split(args.test_fold, args.val_fold)
    res = {}
    for name in args.pred:
        pred = {int(k): [(d[0], d[1], *polar_box_to_fan(*d[2:])) for d in v]
                for k, v in json.loads((run / name).read_text()).items()}
        gt = {f.index: [(classes.index(b.cls), *b.xyxy) for b in f.boxes] for f in frames if split[f.index] == "test"}
        aps, m = evaluate(pred, gt, len(classes))
        res[name] = {"map": m, "ap": {c: (None if np.isnan(a) else float(a)) for c, a in zip(classes, aps)}}
        print(f"{name}: fan-space mAP@0.5 {100 * m:.2f}  " + " ".join(f"{c}:{100 * a:.1f}" for c, a in zip(classes, aps)))
    (run / "fanscore.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    fs = sub.add_parser("fanscore")
    fs.add_argument("run")
    fs.add_argument("--pred", nargs="+", default=["pred_test.json"])
    fs.add_argument("--test-fold", type=int, default=0)
    fs.add_argument("--val-fold", type=int, default=1)
    fs.set_defaults(func=cmd_fanscore)
    c = sub.add_parser("check")
    c.add_argument("--n", type=int, default=6)
    c.add_argument("--out", default="")
    c.set_defaults(func=cmd_check)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
