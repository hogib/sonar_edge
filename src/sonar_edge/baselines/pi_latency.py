"""On-device latency of the baseline detectors under ONNX Runtime (fp32, CPU EP), with the
method of M7 §3 (bench_ort.py on the Pi 4): 10 warm-up runs, then 3 x 50 timed runs,
intra-op threads 1 and 4, inter-op 1, ORT_ENABLE_ALL; one model per process so that peak
RSS (VmHWM) belongs to that model alone. Network only: no NMS / box decoding to image
coordinates outside what the exported graph contains (see `export`).

  # SSDLite (local): backbone + head on the 384x256 input (normalisation included)
  uv run --extra baselines python -m sonar_edge.baselines.pi_latency export-ssdlite
  # YOLOv8n (server, .venv-yolo): Ultralytics ONNX export at imgsz (384, 256); the graph
  # includes the Detect head's DFL box decoding but no NMS
  .venv-yolo/bin/python src/sonar_edge/baselines/pi_latency.py export-yolo
  # NanoDet-Plus-m (server, .venv-nanodet, CPU): NanoDet's ONNX forward, aux head excluded
  CUDA_VISIBLE_DEVICES= .venv-nanodet/bin/python src/sonar_edge/baselines/pi_latency.py export-nanodet --folds 0 1 2 3 4
  # on the board (needs only numpy + onnxruntime)
  python pi_latency.py bench yolov8n.onnx ssdlite.onnx --out pi_latency_raw.json
  # ncnn (Pi 2 and Pi 4): pnnx-converted models, shapes CxHxW in model order
  python pi_latency.py bench dets.ncnn.param yolov8n.ncnn.param ssdlite.ncnn.param \
      --shapes 1x360x240 3x384x256 3x384x256 --out pi_latency_ncnn.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "bench" / "baselines" / "onnx"


def export_ssdlite(run_dir, dest):
    """Network-only SSDLite graph (normalisation + backbone + head) of the checkpoint in run_dir."""
    import torch
    import torch.nn as nn

    from .ssdlite import make_model

    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    m = make_model(len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["model"])
    m.eval()

    class Net(nn.Module):  # what the model computes per image, minus resize (none at 256x384) and NMS
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("mean", torch.tensor(m.transform.image_mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(m.transform.image_std).view(1, 3, 1, 1))

        def forward(self, x):
            f = list(self.m.backbone((x - self.mean) / self.std).values())
            h = self.m.head(f)
            return h["bbox_regression"], h["cls_logits"]

    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Net(m), (torch.zeros(1, 3, 384, 256),), dest, opset_version=17, dynamo=False,
                      input_names=["x"], output_names=["boxes", "logits"])
    print(f"wrote {Path(dest).resolve().relative_to(ROOT)}")


def cmd_export_ssdlite(args):
    export_ssdlite(ROOT / "runs" / "ssdlite_s0_f0", OUT / "ssdlite.onnx")


def cmd_export_yolo(args):
    import shutil

    from ultralytics import YOLO

    p = YOLO(str(ROOT / "runs" / "yolo_pretrained_s0_f0" / "weights" / "best.pt")).export(
        format="onnx", imgsz=(384, 256), opset=17, simplify=False, dynamic=False, nms=False)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, OUT / "yolov8n.onnx")
    print(f"wrote {(OUT / 'yolov8n.onnx').relative_to(ROOT)}")


def export_nanodet(ckpt, dest):
    """Network-only NanoDet-Plus-m graph with NanoDet's own ONNX forward (tools/export_onnx.py:
    plain torch.onnx.export of the model, whose head switches to _forward_onnx during export):
    backbone + PAN + head, no auxiliary head, output (1, 2040, C + 32) = sigmoid class scores
    and raw DFL logits per prior. Input is the normalised 3x384x256 tensor ((x*255 - mean) / std
    is done outside the graph, as in NanoDet's own ncnn/ONNX demos); no box decoding, no NMS.
    Run in .venv-nanodet (needs ext/nanodet). Prints params / MACs (sonar_edge.baselines.macs)."""
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from sonar_edge.baselines.macs import count_macs, count_params
    from sonar_edge.baselines.nanodet_run import build

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build(len(ck["classes"]), False)
    m.load_state_dict(ck["model"])
    m.eval()
    x = torch.zeros(1, 3, 384, 256)
    inf = [m.backbone, m.fpn, m.head]
    params = sum(count_params(p) for p in inf)
    macs = count_macs(m, x)  # m.forward = backbone + fpn + head only
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, (x,), dest, opset_version=17, dynamo=False, input_names=["data"], output_names=["output"])
    print(f"wrote {dest}: inference params {params}, MACs {macs}")


CHILD = r"""
import json, sys, time, numpy as np, onnxruntime as ort
path, threads = sys.argv[1], int(sys.argv[2])
so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
i = s.get_inputs()[0]
x = np.random.default_rng(0).random(i.shape, dtype=np.float32)
for _ in range(10): s.run(None, {i.name: x})
reps, single = [], []
for _ in range(3):
    t0 = time.perf_counter()
    for _ in range(50):
        t1 = time.perf_counter(); s.run(None, {i.name: x}); single.append(1e3 * (time.perf_counter() - t1))
    reps.append(1e3 * (time.perf_counter() - t0) / 50)
st = dict(l.split(":", 1) for l in open("/proc/self/status") if ":" in l)
print(json.dumps({"input": i.shape, "ms": reps, "mean_ms": float(np.mean(reps)),
                  "median_ms": float(np.median(single)), "vmhwm": st["VmHWM"].strip(), "ort": ort.__version__}))
"""


# ncnn (pip wheel; armv7 wheel exists, onnxruntime has none for the Pi 2). Models converted
# ONNX -> ncnn with pnnx (`pnnx model.onnx`); blobs: input in0, outputs out0[, out1].
# Same protocol: 10 warm-up + 3 x 50 timed runs, fp32 (fp16 storage/arith off, so the
# arithmetic matches ORT fp32), one Extractor per run.
CHILD_NCNN = r"""
import json, sys, time, numpy as np, ncnn
param, threads = sys.argv[1], int(sys.argv[2]); shape = tuple(map(int, sys.argv[3].split("x")))
import re; outs = sorted({t for l in open(param) for t in l.split() if re.fullmatch(r"out\d+", t)})
assert outs, "no output blobs found"
net = ncnn.Net(); net.opt.num_threads = threads
net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = net.opt.use_fp16_packed = False
net.load_param(param); net.load_model(param[:-6] + ".bin")
a = np.random.default_rng(0).random(shape, dtype=np.float32)  # keep alive: ncnn.Mat shares its memory
x = ncnn.Mat(a)
def run():
    ex = net.create_extractor(); ex.input("in0", x)
    for o in outs: ex.extract(o)
for _ in range(10): run()
reps, single = [], []
for _ in range(3):
    t0 = time.perf_counter()
    for _ in range(50):
        t1 = time.perf_counter(); run(); single.append(1e3 * (time.perf_counter() - t1))
    reps.append(1e3 * (time.perf_counter() - t0) / 50)
st = dict(l.split(":", 1) for l in open("/proc/self/status") if ":" in l)
print(json.dumps({"input": list(shape), "outputs": outs, "ms": reps, "mean_ms": float(np.mean(reps)),
                  "median_ms": float(np.median(single)), "vmhwm": st["VmHWM"].strip(), "ncnn": ncnn.__version__}))
"""


def cmd_bench(args):
    out = {}
    for path in args.models:
        for t in args.threads:
            if path.endswith(".param"):  # ncnn: path=model.ncnn.param, shape from --shapes in order
                shape = args.shapes[args.models.index(path)]
                cmd = [sys.executable, "-c", CHILD_NCNN, path, str(t), shape]
            else:
                cmd = [sys.executable, "-c", CHILD, path, str(t)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"{cmd[-3:]} failed ({r.returncode}): {r.stderr[-500:]}")
            res = json.loads(r.stdout.strip().splitlines()[-1])
            name = Path(path).name.split(".")[0] + ("_ncnn" if path.endswith(".param") else "")
            out[f"{name}_t{t}"] = res
            print(name, t, json.dumps(res), flush=True)
            time.sleep(10)
    Path(args.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    sub.add_parser("export-ssdlite").set_defaults(func=cmd_export_ssdlite)
    sub.add_parser("export-yolo").set_defaults(func=cmd_export_yolo)
    e = sub.add_parser("export-nanodet")
    e.add_argument("--folds", type=int, nargs="+", default=[0])
    e.set_defaults(func=lambda a: [export_nanodet(ROOT / "runs" / f"nanodet_pretrained_s0_f{k}" / "best.pt",
                                                  OUT / ("nanodet.onnx" if k == 0 else f"nanodet_f{k}.onnx"))
                                   for k in a.folds])
    b = sub.add_parser("bench")
    b.add_argument("models", nargs="+")
    b.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    b.add_argument("--shapes", nargs="*", default=[], help="CxHxW per model, for .param (ncnn) models")
    b.add_argument("--out", default="pi_latency_raw.json")
    b.set_defaults(func=cmd_bench)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
