"""int8 post-training quantization of the baseline detectors with ncnn, to answer whether the
fp32 baselines are compared fairly with the int8 Det-S runtime.

Pipeline per model and fold (seed 0 checkpoints):
  ONNX (pi_latency.export_ssdlite / Ultralytics export at 384x256 / pi_latency.export_nanodet)
  -> pnnx <model>.onnx fp16=0             (ncnn fp32 weights; ncnn2int8 needs fp32 weights)
  -> ncnn2table (KL calibration)           on 64 training frames of that fold, drawn as in
                                           train.quantize: default_rng(0).choice(len(train), 64)
  -> ncnn2int8                             (int8 weights + activation scales)
Accuracy: ncnn fp32 and int8 models run on every test frame of their fold on the laptop CPU,
with each baseline's own postprocessing (SSDLite: torchvision's SSD postprocess_detections,
score 0.01, NMS 0.55, 100 per image; YOLOv8n: Ultralytics-style NMS, conf 0.001, IoU 0.7,
100 per image; NanoDet-Plus-m: NanoDet's get_bboxes + multiclass_nms, score 0.05, NMS 0.6,
100 per image, re-implemented in nanodet_post) and are scored by adapters.score (pooled VOC
AP@0.5, the CV evaluator). NanoDet's graph (pi_latency.export_nanodet) takes the normalised
input, so its calibration passes mean/norm = NanoDet's mean and 1/std to ncnn2table.
Inputs are exactly the training-time inputs: SSDLite and Det-S the dataset's box-filtered
frames (uint8 at 256x384 / 240x360), YOLOv8n Ultralytics' LetterBox output (cv2 INTER_LINEAR,
computed in .venv-yolo and stored as yolo_inputs.npz).

  W=<work dir with fp32/*.ncnn.{param,bin} and yolo_inputs.npz>; T=<ncnn tools bin dir>
  uv run --extra baselines --with ncnn==1.0.20261020 python -m sonar_edge.baselines.int8_ncnn calib --work $W
  ... ncnn2table / ncnn2int8 (see cmd_calib's printed commands, or `quantize --tools $T`)
  uv run --extra baselines --with ncnn==1.0.20261020 python -m sonar_edge.baselines.int8_ncnn quantize --work $W --tools $T
  uv run --extra baselines --with ncnn==1.0.20261020 python -m sonar_edge.baselines.int8_ncnn eval --work $W
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from ..data import class_names, load_frames
from ..split import load_split

ROOT = Path(__file__).resolve().parents[3]
SHAPES = {"ssdlite": (3, 384, 256), "yolov8n": (3, 384, 256), "nanodet": (3, 384, 256), "dets": (1, 360, 240)}
SCALES = {"ssdlite": 1.25, "yolov8n": 1.25, "nanodet": 1.25, "dets": 4 / 3}
# NanoDet normalises outside its graph (nanodet_run.to_input): (x * 255 - mean) / std per channel
NANO_MEAN = (103.53, 116.28, 123.675)
NANO_STD = (57.375, 57.12, 58.395)


def fold_parts(frames, k):
    split = load_split(k, (k + 1) % 5)
    return {s: [f for f in frames if split[f.index] == s] for s in ("train", "val", "test")}


class Inputs:
    """uint8 HxW network inputs (single grey channel) for any frame, per model."""

    def __init__(self, model, frames, classes, work):
        self.model, self.frames, self.classes = model, frames, classes
        if model == "yolov8n":
            z = np.load(Path(work) / "yolo_inputs.npz")
            self.yolo = dict(zip(z["idx"].tolist(), z["img"]))

    def get(self, fs):
        if self.model == "yolov8n":
            return [self.yolo[f.index] for f in fs]
        from ..dataset import FrameDataset
        ds = FrameDataset(fs, self.classes, 8, SCALES[self.model], train=False)
        out = []
        for i in range(len(fs)):
            x = ds._prep(ds.images[i])
            assert x.shape == SHAPES[self.model][1:], x.shape
            out.append(np.round(x * 255).astype(np.uint8))
        return out


def tensor(model, u8):
    c = SHAPES[model][0]
    x = u8.astype(np.float32) / 255.0
    if model == "nanodet":
        m, s = (np.array(v, np.float32).reshape(3, 1, 1) for v in (NANO_MEAN, NANO_STD))
        return np.ascontiguousarray((x[None] * 255.0 - m) / s, dtype=np.float32)
    return np.ascontiguousarray(np.broadcast_to(x, (c, *x.shape)), dtype=np.float32)


def cmd_calib(args):
    from PIL import Image

    frames = load_frames()
    classes = class_names(frames)
    work = Path(args.work)
    for model in args.models:
        inp = Inputs(model, frames, classes, work)
        for k in (args.folds if model != "dets" else [0]):
            train = fold_parts(frames, k)["train"]
            pick = np.random.default_rng(0).choice(len(train), 64, replace=False)
            d = work / "calib" / f"{model}_f{k}"
            d.mkdir(parents=True, exist_ok=True)
            names = []
            for j, img in zip(pick, inp.get([train[j] for j in pick])):
                p = d / f"{train[j].index}.png"
                if SHAPES[model][0] == 3:
                    Image.fromarray(np.stack([img] * 3, -1)).save(p)
                else:
                    Image.fromarray(img).save(p)
                names.append(str(p))
            (d / "list.txt").write_text("\n".join(names) + "\n")
            print(f"{model} f{k}: 64 calibration frames -> {d}")


def cmd_quantize(args):
    work, tools = Path(args.work), Path(args.tools)
    qdir = work / args.tag
    qdir.mkdir(exist_ok=True)
    log = {}
    for model in args.models:
        for k in (args.folds if model != "dets" else [0]):
            name = f"{model}_f{k}" if model != "dets" else "dets"
            c, h, w = SHAPES[model]
            fp = work / "fp32" / f"{name}.ncnn"
            table = qdir / f"{name}.table"
            calib = work / "calib" / f"{model}_f{k}" / "list.txt"
            px = "RGB" if c == 3 else "GRAY"
            mean = ",".join(["0"] * c)
            norm = ",".join(["0.00392156862745098"] * c)
            if model == "nanodet":
                mean = ",".join(map(str, NANO_MEAN))
                norm = ",".join(repr(1 / v) for v in NANO_STD)
            cmd = [str(tools / "ncnn2table"), f"{fp}.param", f"{fp}.bin", str(calib), str(table),
                   f"mean=[{mean}]", f"norm=[{norm}]", f"shape=[{w},{h},{c}]", f"pixel={px}",
                   f"thread={args.threads}", f"method={args.method}"]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"ncnn2table {name} failed: {r.stderr[-800:]}")
            q = qdir / f"{name}.ncnn"
            r2 = subprocess.run([str(tools / "ncnn2int8"), f"{fp}.param", f"{fp}.bin", f"{q}.param", f"{q}.bin",
                                 str(table)], capture_output=True, text=True)
            if r2.returncode:
                raise SystemExit(f"ncnn2int8 {name} failed: {r2.stderr[-800:]}")
            log[name] = {"table_cmd": " ".join(cmd[1:]), "s": round(time.time() - t0, 1),
                         "int8_bin_bytes": (q.parent / f"{q.name}.bin").stat().st_size}
            print(name, json.dumps(log[name]), flush=True)
    (qdir / "quantize_log.json").write_text(json.dumps(log, indent=1))


class Net:
    def __init__(self, param, threads):
        import re

        import ncnn
        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_fp16_storage = self.net.opt.use_fp16_arithmetic = self.net.opt.use_fp16_packed = False
        self.net.load_param(str(param))
        self.net.load_model(str(param)[:-6] + ".bin")
        self.outs = sorted({t for l in open(param) for t in l.split() if re.fullmatch(r"out\d+", t)})

    def __call__(self, x):
        m = self.ncnn.Mat(x)
        ex = self.net.create_extractor()
        ex.input("in0", m)
        res = []
        for o in self.outs:
            _, y = ex.extract(o)
            res.append(np.array(y).copy())
        return res


class SSDPost:
    def __init__(self, n_classes):
        import torch
        from torchvision.models.detection.image_list import ImageList

        from .ssdlite import make_model
        self.torch = torch
        self.m = make_model(n_classes, pretrained=False).eval()
        x = torch.zeros(1, 3, 384, 256)
        with torch.no_grad():
            feats = list(self.m.backbone(x).values())
            self.anchors = self.m.anchor_generator(ImageList(x, [(384, 256)]), feats)

    def __call__(self, outs):
        torch = self.torch
        boxes, logits = outs  # (N, 4), (N, C+1)
        with torch.no_grad():
            det = self.m.postprocess_detections({"bbox_regression": torch.from_numpy(boxes)[None],
                                                 "cls_logits": torch.from_numpy(logits)[None]},
                                                self.anchors, [(384, 256)])[0]
        return [(int(l) - 1, float(s), *(float(v) * 1.25 for v in b))
                for b, l, s in zip(det["boxes"], det["labels"], det["scores"])]


def yolo_post(outs, conf=0.001, iou=0.7, max_det=100, max_wh=7680):
    import torch
    import torchvision

    p = torch.from_numpy(outs[0])  # (4 + nc, N): cx, cy, w, h in input pixels, class probabilities
    xywh, cls = p[:4].T, p[4:].T
    s, j = cls.max(1)
    keep = s > conf
    xywh, s, j = xywh[keep], s[keep], j[keep]
    xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    order = s.argsort(descending=True)[:30000]
    xyxy, s, j = xyxy[order], s[order], j[order]
    k = torchvision.ops.nms(xyxy + j[:, None].float() * max_wh, s, iou)[:max_det]
    b = xyxy[k] / 0.8  # letterbox gain 384/480, no padding
    b[:, 0::2] = b[:, 0::2].clamp(0, 320)
    b[:, 1::2] = b[:, 1::2].clamp(0, 480)
    return [(int(c), float(sc), *map(float, bb)) for c, sc, bb in zip(j[k], s[k], b)]


def nanodet_post(outs, n_classes=11, reg_max=7, strides=(8, 16, 32, 64), h=384, w=256,
                 score_thr=0.05, iou=0.6, max_num=100):
    """NanoDet-Plus head.get_bboxes + multiclass_nms (ext/nanodet be9b4a9) on the exported graph's
    output (N, C + 4 (reg_max + 1)) whose class part is already sigmoid (head._forward_onnx)."""
    import torch
    import torchvision

    p = torch.from_numpy(np.asarray(outs[0], np.float32).reshape(-1, n_classes + 4 * (reg_max + 1)))
    scores, reg = p[:, :n_classes], p[:, n_classes:]
    pri = []
    for s in strides:
        fh, fw = math.ceil(h / s), math.ceil(w / s)
        y, x = torch.meshgrid(torch.arange(fh, dtype=torch.float32) * s, torch.arange(fw, dtype=torch.float32) * s,
                              indexing="ij")
        pri.append(torch.stack([x.flatten(), y.flatten(), torch.full((fh * fw,), float(s))], -1))
    pri = torch.cat(pri)
    assert len(pri) == len(p), (len(pri), p.shape)
    d = torch.softmax(reg.reshape(-1, 4, reg_max + 1), -1) @ torch.linspace(0, reg_max, reg_max + 1)
    d = d * pri[:, 2:3]
    b = torch.stack([(pri[:, 0] - d[:, 0]).clamp(0, w), (pri[:, 1] - d[:, 1]).clamp(0, h),
                     (pri[:, 0] + d[:, 2]).clamp(0, w), (pri[:, 1] + d[:, 3]).clamp(0, h)], -1)
    valid = scores > score_thr
    pi, ci = valid.nonzero(as_tuple=True)
    if len(pi) == 0:
        return []
    bb, sc = b[pi], scores[pi, ci]
    keep = torchvision.ops.nms(bb + (ci.to(bb) * (bb.max() + 1))[:, None], sc, iou)[:max_num]
    return [(int(ci[k]), float(sc[k]), *(float(v) * SCALES["nanodet"] for v in bb[k])) for k in keep]


def cmd_eval(args):
    from ..cv import load_pred
    from .adapters import score
    from .collect import run as run_dir

    frames = load_frames()
    classes = class_names(frames)
    work = Path(args.work)
    out = {}
    fam = {"ssdlite": "ssdlite_pretrained", "yolov8n": "yolo_pretrained", "nanodet": "nanodet_pretrained"}
    for model in args.models:
        inp = Inputs(model, frames, classes, work)
        post = {"ssdlite": lambda: SSDPost(len(classes)), "yolov8n": lambda: yolo_post,
                "nanodet": lambda: (lambda o: nanodet_post(o, len(classes)))}[model]()
        res = {}
        ref = [load_pred(run_dir(fam[model], 0, k) / "pred_test.json") for k in range(5)]
        res["pytorch"] = score(ref, 1.0, frames, classes)
        for prec in args.precisions:
            preds, t0 = [], time.time()
            for k in range(5):
                test = fold_parts(frames, k)["test"]
                net = Net(work / prec / f"{model}_f{k}.ncnn.param", args.threads)
                pk = {}
                for f, img in zip(test, inp.get(test)):
                    pk[f.index] = post(net(tensor(model, img)))
                preds.append(pk)
            res[prec] = score(preds, 1.0, frames, classes)
            res[prec]["eval_s"] = round(time.time() - t0, 1)
            (work / f"pred_{model}_{prec}.json").write_text(json.dumps([{str(i): v for i, v in p.items()} for p in preds]))
        for key in ("pytorch", *args.precisions):
            r = res[key]
            print(f"{model:8s} {key:7s} pooled mAP {100 * r['map_pooled']:.2f}  per fold "
                  + " ".join(f"{100 * x:.1f}" for x in r["map_per_fold"])
                  + f"  acc {100 * r['mcnn_task']['acc']:.2f}", flush=True)
        out[model] = {key: {"map_pooled": res[key]["map_pooled"], "map_per_fold": res[key]["map_per_fold"],
                            "mcnn_acc": res[key]["mcnn_task"]["acc"], "mcnn_f1": res[key]["mcnn_task"]["macro_f1"]}
                      for key in ("pytorch", *args.precisions)}
    dest = Path(args.out)
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    for model, r in out.items():
        prev.setdefault("accuracy_seed0", {}).setdefault(model, {}).update(r)
    dest.write_text(json.dumps(prev, indent=1))
    print(f"wrote {dest}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    for name, fn in (("calib", cmd_calib), ("quantize", cmd_quantize), ("eval", cmd_eval)):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--models", nargs="+", default=["ssdlite", "yolov8n", "dets"] if name != "eval"
                       else ["ssdlite", "yolov8n"])
        p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        p.add_argument("--threads", type=int, default=8)
        p.set_defaults(func=fn)
        if name == "quantize":
            p.add_argument("--tools", required=True)
            p.add_argument("--method", default="kl")
            p.add_argument("--tag", default="int8", help="output subdir of --work")
        if name == "eval":
            p.add_argument("--precisions", nargs="+", default=["fp32", "int8"],
                           help="subdirs of --work holding <model>_f<k>.ncnn.param")
            p.add_argument("--out", default=str(ROOT / "bench" / "baselines" / "int8_ncnn.json"))
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
