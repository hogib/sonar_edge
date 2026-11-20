"""Run both models over DANS ARIS recordings with sonar-edge-run --aris; summarise detections.

usage: SE_RUN_SMALL=<dir>/sonar-edge-run SE_RUN_SEG=<dir>/sonar-edge-run python3 -I run_dans.py OUTDIR FILE.aris [...]
(the two binaries: runtime/ built as is, and with `make W=models/small_360_segdice`)
Per recording and model: frames processed, frames with >= 1 detection at p >= 0.3, detections
per frame by class at p >= 0.3, and the mean stage times (one pass, -n 1).
"""
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"small360": os.environ["SE_RUN_SMALL"], "segdice": os.environ["SE_RUN_SEG"]}
P = 0.3
MAX_FRAMES = 200   # per recording: every Nth frame so that at most this many are run


def n_frames(path):
    import struct
    beams = {1: 48, 2: 48, 3: 96, 4: 96, 5: 96, 6: 64, 7: 64, 8: 64, 9: 128, 10: 128, 11: 128, 12: 128}
    with open(path, "rb") as f:
        f.seek(1024)
        h = f.read(1024)
    fs = beams[struct.unpack_from("<I", h, 436)[0]] * struct.unpack_from("<I", h, 468)[0]
    return (os.path.getsize(path) - 1024) // (1024 + fs)


def main():
    out, files = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    summary = {}
    for path in files:
        name = os.path.basename(path)[:-5]
        nf = n_frames(path)
        every = max(1, -(-nf // MAX_FRAMES))
        for m, binp in MODELS.items():
            p = subprocess.run([binp, "--aris", "--every", str(every), path], capture_output=True, text=True, check=True)
            lines = [json.loads(l) for l in p.stdout.splitlines()]
            json_path = os.path.join(out, f"{name}.{m}.jsonl")
            with open(json_path, "w") as f:
                f.write(p.stdout)
            cls = collections.Counter()
            frames_hit = 0
            mask_cells = collections.Counter()
            ms = collections.defaultdict(float)
            for d in lines:
                hits = [x for x in d["detections"] if x["prob"] >= P]
                frames_hit += bool(hits)
                cls.update(x["name"] for x in hits)
                for k, v in d["ms"].items():
                    ms[k] += v / len(lines)
                for k, v in d.get("mask", {}).get("cells", {}).items():
                    if k != "background":
                        mask_cells[k] += v / len(lines)
            summary.setdefault(name, {"frames_total": nf, "every": every, "header": p.stderr.strip().splitlines()[0]})[m] = {
                "frames": len(lines), "frames_with_det": frames_hit,
                "dets_per_frame": {k: round(v / len(lines), 3) for k, v in cls.most_common()},
                "mask_cells_per_frame": {k: round(v, 1) for k, v in mask_cells.most_common()},
                "ms": {k: round(v, 2) for k, v in ms.items()}}
            s = summary[name][m]
            print(f"{name:42s} {m:8s} {s['frames']:4d}/{nf} fr, with det {s['frames_with_det']:4d}, per frame "
                  + ", ".join(f"{k} {v}" for k, v in list(s["dets_per_frame"].items())[:5]), flush=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
