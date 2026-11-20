"""Series and plots for the stream runs.

  uv run --with matplotlib python make_plots.py SCRATCH_RUNS_DIR

Writes, next to this script:
  series_<board>_long.csv   10-s windows of the 30-min runs: processed fps, latency p50/p95/max,
                            dropped in window, temperature, ARM clock, throttled, CPU %
  latency_cdf.png           capture->result latency CDF, operating points of both boards
  long_runs.png             30-min runs: temperature, clock, processed fps, latency p95 vs time
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def acks_ms(path):
    rows = list(csv.DictReader(open(path)))
    return [(int(r["t_capture_ns"]) / 1e9, (int(r["t_ack_ns"]) - int(r["t_capture_ns"])) / 1e6) for r in rows]


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def series(runs, board, tag, win=10.0):
    a = acks_ms(runs / board / f"{tag}.acks.csv")
    s = list(csv.DictReader(open(runs / board / f"{tag}.samples.csv")))
    st = [json.loads(l)["stream"] for l in open(runs / board / f"{tag}.jsonl")]
    t0 = a[0][0]
    n_win = int((a[-1][0] - t0) // win) + 1
    out = []
    for k in range(n_win):
        lo, hi = t0 + k * win, t0 + (k + 1) * win
        lat = [l for t, l in a if lo <= t < hi]
        ss = [x for x in s if lo <= float(x["t_unix"]) < hi]
        drop_lo = [x["dropped_total"] for x in st if lo <= x["t_capture_ns"] / 1e9 < hi]
        out.append({
            "t_s": round(k * win, 1), "processed_fps": round(len(lat) / win, 2),
            "lat_p50_ms": round(pct(lat, .5), 2), "lat_p95_ms": round(pct(lat, .95), 2), "lat_max_ms": round(max(lat), 2) if lat else "",
            "dropped_in_window": (max(drop_lo) - min(drop_lo)) if drop_lo else "",
            "temp_c": round(sum(float(x["temp_c"]) for x in ss) / len(ss), 1) if ss else float("nan"),
            "arm_mhz": round(min(int(x["arm_hz"]) for x in ss) / 1e6) if ss else float("nan"),
            "throttled": "|".join(sorted({x["throttled"] for x in ss})) if ss else "",
            "cpu_pct": round(sum(float(x["cpu_pct"]) for x in ss) / len(ss), 1) if ss else float("nan"),
        })
    with open(HERE / f"series_{board}_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


def main():
    runs = Path(sys.argv[1])
    longs = {}
    for board in ("pi4", "pi2"):
        p = HERE / f"summary_{board}_long.json"
        if p.exists():
            tag = next(iter(json.loads(p.read_text())))
            longs[board] = (tag, series(runs, board, tag))

    # latency CDF at the operating points (and the 30-min runs)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    picks = []
    for board, label in (("pi4", "Pi 4"), ("pi2", "Pi 2")):
        sw = HERE / f"summary_{board}_sweep.json"
        if not sw.exists():
            continue
        S = json.loads(sw.read_text())
        for tag, s in S.items():
            if s["fps"] == 15 and s["policy"] == "latest" and s["model"] == "small":
                picks.append((f"{label}, {s['workers']} worker{'s' if s['workers'] > 1 else ''}", runs / board / f"{tag}.acks.csv"))
    for name, path in picks:
        lat = sorted(l for _, l in acks_ms(path))
        if not lat:
            continue
        ax.plot(lat, [i / len(lat) for i in range(1, len(lat) + 1)], label=name, lw=1.4)
    ax.axvline(1000 / 15, color="0.5", ls="--", lw=1)
    ax.text(1000 / 15 * 1.03, 0.05, "frame period\n(15 fps)", fontsize=8, color="0.35")
    ax.set_xscale("log")
    ticks = [20, 30, 50, 100, 150, 200, 300, 500, 1000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("capture to result latency (ms, server clock, log scale)")
    ax.set_ylabel("fraction of processed frames")
    ax.set_title("Live stream, 15 fps, latest-frame policy, small@360")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "latency_cdf.png", dpi=150)

    if longs:
        fig, axs = plt.subplots(4, 1, figsize=(6.4, 7.2), sharex=True)
        for board, (tag, ser) in longs.items():
            t = [r["t_s"] / 60 for r in ser]
            lab = {"pi4": "Pi 4", "pi2": "Pi 2"}[board]
            axs[0].plot(t, [r["temp_c"] for r in ser], label=lab)
            axs[1].plot(t, [r["arm_mhz"] for r in ser], label=lab)
            axs[2].plot(t, [r["processed_fps"] for r in ser], label=lab)
            axs[3].plot(t, [r["lat_p95_ms"] for r in ser], label=lab)
        axs[0].set_ylabel("temp (C)")
        axs[1].set_ylabel("min ARM clock\nin window (MHz)")
        axs[2].set_ylabel("processed fps")
        axs[3].set_ylabel("latency p95\n(ms)")
        axs[3].set_xlabel("time (min)")
        for a in axs:
            a.grid(alpha=.3)
        axs[0].legend(fontsize=8)
        axs[0].set_title("30-min live stream at 15 fps (latest-frame policy)")
        fig.tight_layout()
        fig.savefig(HERE / "long_runs.png", dpi=150)


if __name__ == "__main__":
    main()
