"""Export the scene-grouped folds for third-party detectors.

The frames are exported once at native resolution (320x480, 8-bit grey PNG); each tool
resizes them itself, so no image is resampled twice. For fold k the training set is
every fold except k and (k+1) mod 5, validation is (k+1) mod 5, test is k
(splits/scene_folds_v1.json), as for the detector CV. All 11 classes, Wall included.

  data/baselines/images/<idx>.png            frames (shared by all folds and tools)
  data/baselines/labels/<idx>.txt            YOLO labels: cls cx cy w h, normalized
  data/baselines/yolo/fold<k>_{train,val,test}.txt and fold<k>.yaml
  data/baselines/coco/fold<k>_{train,val,test}.json   COCO, category id = class index + 1

  uv run python -m sonar_edge.baselines.folds
"""

import json
from pathlib import Path

from PIL import Image

from ..data import FRAME_H, FRAME_W, class_names, load_frames
from ..split import SPLIT_FILE

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "baselines"


def fold_of():
    return {int(i): f for i, f in json.loads(SPLIT_FILE.read_text())["fold"].items()}


def split_ids(k, folds):
    v = (k + 1) % 5
    return {"train": [i for i, f in folds.items() if f not in (k, v)],
            "val": [i for i, f in folds.items() if f == v],
            "test": [i for i, f in folds.items() if f == k]}


def main():
    frames = load_frames()
    classes = class_names(frames)
    cidx = {c: i for i, c in enumerate(classes)}
    folds = fold_of()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    (OUT / "yolo").mkdir(parents=True, exist_ok=True)
    (OUT / "coco").mkdir(parents=True, exist_ok=True)
    by_id = {f.index: f for f in frames}
    for f in frames:
        img = OUT / "images" / f"{f.index}.png"
        if not img.exists():
            Image.fromarray(f.load()).save(img)
        lines = []
        for b in f.boxes:
            cx, cy = (b.x + b.w / 2) / FRAME_W, (b.y + b.h / 2) / FRAME_H
            lines.append(f"{cidx[b.cls]} {cx:.6f} {cy:.6f} {b.w / FRAME_W:.6f} {b.h / FRAME_H:.6f}")
        (OUT / "labels" / f"{f.index}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
    for k in range(5):
        ids = split_ids(k, folds)
        for split, lst in ids.items():
            (OUT / "yolo" / f"fold{k}_{split}.txt").write_text(
                "\n".join(str(OUT / "images" / f"{i}.png") for i in sorted(lst)) + "\n")
            imgs, anns = [], []
            for i in sorted(lst):
                imgs.append({"id": i, "file_name": f"{i}.png", "width": FRAME_W, "height": FRAME_H})
                for b in by_id[i].boxes:
                    anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cidx[b.cls] + 1,
                                 "bbox": [b.x, b.y, b.w, b.h], "area": b.w * b.h, "iscrowd": 0})
            (OUT / "coco" / f"fold{k}_{split}.json").write_text(
                json.dumps({"images": imgs, "annotations": anns, "categories": cats}))
        (OUT / "yolo" / f"fold{k}.yaml").write_text(
            f"path: {OUT}\ntrain: yolo/fold{k}_train.txt\nval: yolo/fold{k}_val.txt\ntest: yolo/fold{k}_test.txt\n"
            f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
        print(f"fold {k}: " + ", ".join(f"{s} {len(v)}" for s, v in ids.items()))
    (OUT / "classes.json").write_text(json.dumps(classes))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(frames)} frames, {len(classes)} classes)")


if __name__ == "__main__":
    main()
