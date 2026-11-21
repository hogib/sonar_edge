"""Microplastic-source task (material.py groups) for the crop classifiers MCNN, MobileNetV3-Small and
ShuffleNetV2 x0.5, from runs/<prefix>_s<seed>_f<k>/results.json -> bench/baselines/cls_material_summary.json.
Reproduces MCNN's bench/material/summary.json row exactly."""
import json
import numpy as np
from sonar_edge import material as M
ROOT = M.ROOT
def runs(prefix, seed):
    rs = [json.loads((ROOT / "runs" / f"{prefix}_s{seed}_f{k}" / "results.json").read_text()) for k in range(5)]
    classes = rs[0]["classes"]
    grp = np.array([-2 if M.group_of(c) is None else M.group_of(c) for c in classes])
    yt = np.concatenate([r["labels"] for r in rs]); yp = np.concatenate([r["test_pred"] for r in rs])
    keep = grp[yt] >= 0; p = grp[yp[keep]]
    return M.binary_scores(grp[yt[keep]], np.where(p < 0, -1, p))
out = {}
for pre in ("mcnn_cv", "cls_mnv3s", "cls_shufv2x05"):
    rows = [runs(pre, s) for s in range(3)]
    out[pre] = {k: (100 * np.mean([r[k] for r in rows]), 100 * np.std([r[k] for r in rows], ddof=1))
                for k in ("acc", "micro_recall", "micro_precision", "micro_f1", "other_recall")}
    print(pre, {k: f"{m:.1f}±{s:.1f}" for k, (m, s) in out[pre].items()}, "n", rows[0].get("n"))
json.dump(out, open(ROOT / "bench" / "baselines" / "cls_material_summary.json", "w"), indent=1)
