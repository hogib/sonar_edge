"""Det-S under ncnn (fp32, int8 KL, int8 percentile) on fold 0 / seed 0, against PyTorch and the C int8 runtime.
Usage: uv run --extra baselines --with ncnn==1.0.20261020 python bench/baselines/dets_ncnn_eval.py <int8 work dir>
Work dir layout as in src/sonar_edge/baselines/int8_ncnn.py (fp32/, int8/, int8_dets_percentile/)."""
import json, sys
from pathlib import Path
import numpy as np, torch
from sonar_edge.baselines.int8_ncnn import Inputs, Net, tensor, fold_parts
from sonar_edge.cv import load_pred, fold_gt
from sonar_edge.data import class_names, load_frames
from sonar_edge.metrics import evaluate
from sonar_edge.targets import decode
W = Path(sys.argv[1]); R = Path("runs/small_360_cv_f0")
frames = load_frames(); classes = class_names(frames); C = len(classes)
test = fold_parts(frames, 0)["test"]
imgs = Inputs("dets", frames, classes, W).get(test)
gt = fold_gt(frames, classes, 0, 1.0)
def sc(pred):  # pred boxes at working scale -> native (x4/3)
    p = {i: [(d[0], d[1], *(v * 4 / 3 for v in d[2:])) for d in ds] for i, ds in pred.items()}
    return evaluate(p, gt, C)[1]
res = {"pytorch_fp32": sc(load_pred(R / "pred_test.json")), "c_int8_runtime": sc(load_pred(R / "pred_test_int8.json"))}
for tag, d in [("ncnn_fp32", "fp32"), ("ncnn_int8_kl", "int8"), ("ncnn_int8_percentile", "int8_dets_percentile")]:
    net = Net(W / d / "dets.ncnn.param", 8); pred = {}
    for f, img in zip(test, imgs):
        out = torch.from_numpy(np.array(net(tensor("dets", img))[0]))
        if out.dim() == 3: out = out[None]
        stride = 240 // out.shape[-1]
        dets = decode(out[:, :C], out[:, C:C + 2], out[:, C + 2:C + 4], stride, 32, -4.0)[0]
        pred[f.index] = dets
    res[tag] = sc(pred)
for k, v in res.items(): print(f"{k:22s} fold-0 test mAP {100 * v:.2f}")
json.dump({"fold": 0, "seed": 0, "n_test": len(test), **res}, open(W / "dets_fold0_acc.json", "w"), indent=1)
