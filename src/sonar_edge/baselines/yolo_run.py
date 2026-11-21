"""YOLOv8n (Ultralytics, AGPL-3.0) under the detector's CV protocol. Runs in its own
venv (.venv-yolo: ultralytics 8.4.216, torch 2.15.0+cu132); standalone, no sonar_edge
import. Data from sonar_edge.baselines.folds (data/baselines/yolo/fold<k>.yaml: train =
folds other than k and k+1, val = k+1, test = k).

Settings: imgsz 384 (training letterboxed square with Ultralytics' default augmentation
incl. mosaic; inference letterboxed rectangular, i.e. 256x384 for the 320x480 frame,
the size used for every baseline), batch 16, Ultralytics' default optimiser
schedule. Initialisation: COCO-pretrained yolov8n.pt (--init pretrained) or the model
yaml (--init scratch). Model selection: Ultralytics' best.pt (fitness on the
validation fold). Test predictions: conf 0.001, NMS IoU 0.7, max 100 per image, boxes
in native pixels -> runs/yolo_<init>_s<seed>_f<k>/pred_test.json, the format of
sonar_edge.baselines.adapters.

  .venv-yolo/bin/python src/sonar_edge/baselines/yolo_run.py --init pretrained --epochs 60 --seeds 0 1 2
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "baselines"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["pretrained", "scratch"], required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--data-subdir", default="yolo", help="fold lists under data/baselines/ (yolo_random: leakage run)")
    ap.add_argument("--prefix", default="yolo", help="run name prefix (rand_yolo: leakage run)")
    args = ap.parse_args()
    from ultralytics import YOLO

    wdir = ROOT / "weights"
    wdir.mkdir(exist_ok=True)
    for seed in args.seeds:
        for k in args.folds:
            name = f"{args.prefix}_{args.init}_s{seed}_f{k}"
            out = ROOT / "runs" / name
            if (out / "pred_test.json").exists():
                print(f"skip {name} (done)", flush=True)
                continue
            t0 = time.time()
            init = str(wdir / "yolov8n.pt") if args.init == "pretrained" else "yolov8n.yaml"
            model = YOLO(init)
            model.train(data=str(DATA / args.data_subdir / f"fold{k}.yaml"), imgsz=384, epochs=args.epochs, batch=args.batch,
                        seed=seed, deterministic=True, workers=args.workers, device=0, project=str(ROOT / "runs"),
                        name=name, exist_ok=True, pretrained=(args.init == "pretrained"), plots=False, verbose=False)
            best = YOLO(str(out / "weights" / "best.pt"))
            test = [l for l in (DATA / args.data_subdir / f"fold{k}_test.txt").read_text().split("\n") if l]
            pred = {}
            for r in best.predict(source=test, imgsz=384, conf=0.001, iou=0.7, max_det=100, stream=True, verbose=False,
                                  device=0):
                idx = int(Path(r.path).stem)
                b = r.boxes
                pred[str(idx)] = [[int(c), float(s), *map(float, xy)]
                                  for c, s, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
            (out / "pred_test.json").write_text(json.dumps(pred))
            (out / "info.json").write_text(json.dumps({"init": args.init, "seed": seed, "test_fold": k,
                                                        "val_fold": (k + 1) % 5, "epochs": args.epochs,
                                                        "data": args.data_subdir,
                                                        "train_s": round(time.time() - t0, 1)}))
            print(f"{name}: done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
