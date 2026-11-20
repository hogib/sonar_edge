"""Bytes per frame of a compact detection message, from sonar-edge-run --stream JSON lines.

Keeps detections with probability >= 0.3, each as {name, prob, box_fan, range, bearing_deg}
(rounded, compact JSON, no whitespace); an empty frame costs 2 bytes ("[]").
  python3 bench/stream/bandwidth.py RUN_DIR > bench/stream/bandwidth.json
RUN_DIR holds the 1-worker 15 fps sweep outputs (pi4/sw_*_w1_f15_latest.jsonl).
"""
import glob, json, statistics as st, sys

out = {}
for f in sorted(glob.glob(sys.argv[1] + "/pi4/sw_*_w1_f15_latest.jsonl")):
    sizes = []
    for line in open(f):
        d = json.loads(line)
        if "detections" not in d:
            continue
        slim = [{"name": x["name"], "prob": round(x["prob"], 3), "box_fan": [round(v, 1) for v in x["box_fan"]],
                 "range": round(x["range"], 1), "bearing_deg": round(x["bearing_deg"], 1)}
                for x in d["detections"] if x.get("prob", 0) >= 0.3]
        sizes.append(len(json.dumps(slim, separators=(",", ":"))))
    out[f.split("/")[-1]] = {"frames": len(sizes), "mean_bytes": round(st.mean(sizes), 1), "median_bytes": st.median(sizes),
                             "max_bytes": max(sizes)}
json.dump(out, sys.stdout, indent=1)
