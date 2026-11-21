"""MCNN under ncnn (fp32 and int8) for the Pi 2 / Pi 4 per-crop timing, with an accuracy check on fold 0.

  W=<work dir>; T=<ncnn tools bin dir (ncnn2table, ncnn2int8)>; P=<pnnx>
  cp runs/mcnn_cv_s0_f0/mcnn.onnx $W/mcnn.onnx && (cd $W && $P mcnn.onnx inputshape=[1,1,305,305] fp16=0)
  uv run --extra baselines --with ncnn==1.0.20261020 python bench/baselines/mcnn_ncnn.py calib $W
  $T/ncnn2table $W/mcnn.ncnn.param $W/mcnn.ncnn.bin $W/calib/list.txt $W/mcnn_<method>.table \
      mean=[0] norm=[0.00392156862745098] shape=[305,305,1] pixel=GRAY thread=8 method=<method>
  $T/ncnn2int8 $W/mcnn.ncnn.param $W/mcnn.ncnn.bin $W/mcnn_int8.ncnn.param $W/mcnn_int8.ncnn.bin $W/mcnn_<method>.table
  uv run --extra baselines --with ncnn==1.0.20261020 python bench/baselines/mcnn_ncnn.py eval $W mcnn mcnn_int8 ...

Calibration: 64 training crops of fold 0 (folds 2-4; fold 1 is validation), drawn with
default_rng(0).choice as everywhere else, zero-padded to 305x305 exactly as in training
(mcnn_cv.pad). Eval: every fold-0 test crop through the seed-0 fold-0 model; accuracy and
agreement with the PyTorch predictions in runs/mcnn_cv_s0_f0/results.json."""
import json
import sys
from pathlib import Path

import numpy as np

from sonar_edge.baselines.int8_ncnn import Net
from sonar_edge.mcnn_cv import object_crops, pad

W = Path(sys.argv[2])
crops, y, folds, ids, classes = object_crops()
if sys.argv[1] == "calib":
    from PIL import Image
    tr = np.where((folds != 0) & (folds != 1))[0]
    pick = tr[np.random.default_rng(0).choice(len(tr), 64, replace=False)]
    d = W / "calib"
    d.mkdir(parents=True, exist_ok=True)
    X = pad([crops[i] for i in pick]).numpy()
    names = []
    for i, x in zip(pick, X):
        p = d / f"crop{i}.png"
        Image.fromarray(np.round(x[0] * 255).astype(np.uint8)).save(p)
        names.append(str(p))
    (d / "list.txt").write_text("\n".join(names) + "\n")
    print(f"64 calibration crops -> {d}")
else:
    ref = json.loads(Path("runs/mcnn_cv_s0_f0/results.json").read_text())
    te = np.array(ref["test_idx"])
    X = pad([crops[i] for i in te]).numpy()
    out = {"fold": 0, "seed": 0, "n_test": len(te), "pytorch_acc": ref["test_acc"]}
    for name in sys.argv[3:]:
        net = Net(W / f"{name}.ncnn.param", 8)
        yp = np.array([int(np.argmax(net(x)[0])) for x in X])
        out[name] = {"acc": float((yp == y[te]).mean()), "agree_pytorch": float((yp == np.array(ref["test_pred"])).mean())}
        print(name, out[name])
    (W / "mcnn_fold0_acc.json").write_text(json.dumps(out, indent=1))
