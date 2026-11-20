"""Compare a board's sonar-edge-run output with the laptop reference, score mAP, summarise timings.

  uv run python bench/e2e/compare_e2e.py REF_DIR BOARD_DIR LABEL [--score]

REF_DIR / BOARD_DIR are run_e2e.py work dirs (out/*.jsonl, masks/). JSON lines
are compared with the per-stage times ("ms") removed and "source" reduced to
its basename (paths differ between machines); masks are compared byte for
byte. With --score, detections are written in sonar-detect's line format and
scored with sonar_edge.c_eval (tag e2e_<LABEL>_<mode>).
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

RUNS = {"small360": "runs/small_360_cv_f0", "segdice": "runs/small_360_segdice_cv_f0"}
ROOT = Path(__file__).resolve().parents[2]


def norm(line):
    d = json.loads(line)
    d.pop("ms")
    d["source"] = os.path.basename(d["source"])
    return json.dumps(d, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("board")
    ap.add_argument("label")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    ref, brd = Path(a.ref), Path(a.board)
    order = json.loads((ref / "in/order.json").read_text())["index"]
    res = {}
    for f in sorted((brd / "out").glob("*.jsonl")):
        tag = f.stem                                   # model_variant_mode
        model, var, mode = tag.split("_")
        rf = next((ref / "out").glob(f"{model}_*_{mode}.jsonl"))
        L, R = f.read_text().splitlines(), rf.read_text().splitlines()
        same = sum(norm(x) == norm(y) for x, y in zip(L, R)) if len(L) == len(R) else 0
        md, rmd = brd / "masks" / tag, ref / "masks" / rf.stem
        masks = sorted(p.name for p in md.glob("*.pgm")) if md.exists() else []
        msame = sum((md / n).read_bytes() == (rmd / n).read_bytes() for n in masks if (rmd / n).exists())
        ms = [json.loads(x)["ms"] for x in L]
        mean = {k: round(sum(m[k] for m in ms) / len(ms), 3) for k in ms[0]} if ms else {}
        r = {"frames": len(L), "json_identical": same, "masks": len(masks), "masks_identical": msame,
             "ref_masks": len(list(rmd.glob("*.pgm"))) if rmd.exists() else 0, "ms_mean": mean}
        if a.score:
            pf = brd / "out" / f"{tag}.det"
            with open(pf, "w") as fh:
                for k, x in enumerate(L):
                    d = json.loads(x)
                    idx = order[k] if mode == "polar" else int(Path(d["source"]).stem)
                    for t in d["detections"]:
                        fh.write(f"{idx}.pgm {t['cls']} {t['score']:.6f} " + " ".join(f"{v:.3f}" for v in t["box"]) + "\n")
            out = subprocess.run([sys.executable, "-m", "sonar_edge.c_eval", RUNS[model], "--pred", str(pf),
                                  "--tag", f"e2e_{a.label}_{model}_{var}_{mode}"],
                                 cwd=ROOT, capture_output=True, text=True, check=True).stdout
            r["map"] = float(re.search(r"C runtime mAP@0.5 ([\d.]+)", out).group(1))
        res[tag] = r
        print(tag, json.dumps(r), flush=True)
    (brd / "out" / f"compare_{a.label}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
