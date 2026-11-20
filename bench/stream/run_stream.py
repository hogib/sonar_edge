"""Live-stream runs: sonar-stream on the laptop (wired, 192.168.1.113) -> sonar-edge-run
--stream on a Pi. One run = cool the board, start the server, start the Pi sampler,
run the client until the server closes, fetch outputs, summarise.

  python3 run_stream.py BOARD PLAN [--out DIR]     BOARD = pi4 | pi2, PLAN = correct | sweep | long

Summaries go to DIR/summary_<board>_<plan>.json; per-run raw files to DIR/raw/ (big
JSONL dumps stay in the scratch dir given by $STREAM_SCRATCH).
"""
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAPTOP = "192.168.1.113"
SERVER = os.environ.get("SONAR_STREAM_BIN", str(ROOT / "runtime/sonar-stream"))
SCRATCH = Path(os.environ.get("STREAM_SCRATCH", "/tmp/stream_runs"))
DATA = ROOT / "data/raw/dans_aris"
MIX = ["2021-04-16_030900_SideSmaller.aris", "2021-04-16_035403_Cup.aris", "2021-04-16_025439_HalfPlasticContainer.aris",
       "2021-04-16_041904_PinAlone_Control.aris", "2021-04-19_122217_SideContainer.aris",
       "2021-04-16_024146_PlasticContainer2.aris"]
BOARDS = {
    "pi4": {"ssh": "testpi4@192.168.1.111", "cmd": "", "port": 5801},
    "pi2": {"ssh": "vegsjumper", "cmd": "command ", "port": 5802},
}


def ssh(board, cmd, check=True, timeout=None):
    b = BOARDS[board]
    r = subprocess.run(["ssh", "-o", "ServerAliveInterval=15", b["ssh"], cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{board}: {cmd}: {r.stderr.strip()}")
    return r.stdout.strip()


def temp(board):
    c = BOARDS[board]["cmd"]
    return float(ssh(board, f"{c}vcgencmd measure_temp").split("=")[-1].rstrip("'C"))


def cool(board, limit=50.0, max_s=420):
    t0 = time.time()
    t = temp(board)
    while t >= limit and time.time() - t0 < max_s:
        time.sleep(15)
        t = temp(board)
    return t, round(time.time() - t0)


def pct(v, q):
    if not v:
        return None
    v = sorted(v)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def run(board, tag, model="small", workers=1, policy="latest", fps=None, duration=60.0, recs=None, no_loop=False):
    b = BOARDS[board]
    c = b["cmd"]
    raw = SCRATCH / board
    raw.mkdir(parents=True, exist_ok=True)
    keep = HERE / "raw" / board
    keep.mkdir(parents=True, exist_ok=True)
    t_cool, cool_s = cool(board)
    acks = raw / f"{tag}.acks.csv"
    srv_log = raw / f"{tag}.server.log"
    recs = recs or [str(DATA / r) for r in MIX]
    cmd = [SERVER, "--bind", LAPTOP, "-p", str(b["port"]), "--log", str(acks)]
    if fps:
        cmd += ["--fps", str(fps)]
    if duration:
        cmd += ["--duration", str(duration)]
    if no_loop:
        cmd += ["--no-loop"]
    srv = subprocess.Popen(cmd + recs, stderr=open(srv_log, "w"))
    time.sleep(0.5)
    samp_dur = (duration or 600) + 20
    ssh(board, f"cd ~/sonar_edge_stream && mkdir -p out && ({c}nohup python3 sampler.py out/{tag}.samples.csv {samp_dur} 2 "
               f"> /dev/null 2>&1 < /dev/null &)")
    t_start = time.time()
    # the client runs detached on the Pi, so a dropped ssh session cannot end the run; poll until it and the server exit
    ssh(board, f"cd ~/sonar_edge_stream && ({c}nohup ./{model}/sonar-edge-run --stream {LAPTOP}:{b['port']} --policy {policy} "
               f"--workers {workers} --name {tag[:23]} > out/{tag}.jsonl 2> out/{tag}.client.log < /dev/null & "
               f"echo $! > out/{tag}.pid)")
    while True:
        time.sleep(10)
        try:
            alive = ssh(board, f"{c}kill -0 $({c}cat ~/sonar_edge_stream/out/{tag}.pid) 2>/dev/null && echo y || echo n",
                        check=False, timeout=30)
        except subprocess.TimeoutExpired:
            continue
        if alive == "n" and srv.poll() is not None:
            break
        if alive == "n" and time.time() - t_start > (duration or 0) + 120:
            srv.terminate()
    wall = time.time() - t_start
    srv.wait(timeout=60)
    ssh(board, f"{c}pkill -f 'sampler.py out/{tag}.samples.csv'", check=False)
    for ext in ("jsonl", "client.log", "samples.csv"):
        subprocess.run(["scp", "-q", f"{b['ssh']}:sonar_edge_stream/out/{tag}.{ext}", str(raw / f"{tag}.{ext}")], check=True)
    s = summarise(raw, tag, fps or 15.0, duration)
    s.update({"board": board, "tag": tag, "model": model, "workers": workers, "policy": policy, "fps": fps or 15.0,
              "duration_s": duration, "wall_s": round(wall, 1), "temp_before_c": t_cool, "cool_wait_s": cool_s})
    # keep the small files
    for ext in ("samples.csv", "client.log", "server.log"):
        src = raw / f"{tag}.{ext}"
        if src.exists():
            (keep / src.name).write_text(src.read_text())
    return s


def summarise(raw, tag, fps, duration):
    srv = (raw / f"{tag}.server.log").read_text()
    produced = int(srv.split("produced ")[1].split(" frames")[0]) if "produced " in srv else None
    lat, rtd = [], []
    rows = list(csv.DictReader(open(raw / f"{tag}.acks.csv")))
    for r in rows:
        lat.append((int(r["t_ack_ns"]) - int(r["t_capture_ns"])) / 1e6)
    recs = [json.loads(l) for l in open(raw / f"{tag}.jsonl")]
    st = [r["stream"] for r in recs]
    rtd = [x["recv_to_done_ms"] for x in st]
    wait = [x["wait_ms"] for x in st]
    proc = [x["proc_ms"] for x in st]
    qd = [x["queue_depth"] for x in st]
    stage = {k: statistics.fmean(r["ms"][k] for r in recs) for k in ("frontend", "network", "decode", "seg", "total")} if recs else {}
    period_ms = 1000.0 / fps
    met = sum(1 for x in lat if x <= period_ms)
    samples = list(csv.DictReader(open(raw / f"{tag}.samples.csv"))) if (raw / f"{tag}.samples.csv").exists() else []
    temps = [float(x["temp_c"]) for x in samples if x["temp_c"]]
    clocks = [int(x["arm_hz"]) for x in samples if x["arm_hz"].isdigit()]
    thr = sorted({x["throttled"] for x in samples})
    cpu = [float(x["cpu_pct"]) for x in samples[1:]]
    out = {
        "produced": produced, "processed": len(recs), "acked": len(rows),
        "processed_fps": round(len(recs) / duration, 2) if duration else None,
        "dropped_latest": max((x["dropped_total"] for x in st), default=0),
        "gaps": max((x["gaps_total"] for x in st), default=0),
        "dropped_pct": round(100.0 * (1 - len(recs) / produced), 2) if produced else None,
        "latency_ms": {"p50": pct(lat, .5), "p95": pct(lat, .95), "p99": pct(lat, .99), "max": max(lat) if lat else None,
                       "mean": statistics.fmean(lat) if lat else None},
        "recv_to_done_ms": {"p50": pct(rtd, .5), "p95": pct(rtd, .95), "p99": pct(rtd, .99), "max": max(rtd) if rtd else None},
        "wait_ms": {"p50": pct(wait, .5), "p95": pct(wait, .95)},
        "proc_ms_mean": statistics.fmean(proc) if proc else None,
        "stage_ms_mean": stage,
        "deadline_ms": period_ms,
        "deadline_met_pct_of_produced": round(100.0 * met / produced, 2) if produced else None,
        "deadline_met_pct_of_processed": round(100.0 * met / len(lat), 2) if lat else None,
        "queue_depth": {"mean": statistics.fmean(qd) if qd else None, "max": max(qd, default=0)},
        "temp_c": {"start": temps[0] if temps else None, "end": temps[-1] if temps else None, "max": max(temps, default=None)},
        "arm_mhz": {"min": min(clocks) / 1e6 if clocks else None, "median": statistics.median(clocks) / 1e6 if clocks else None},
        "throttled": thr, "cpu_pct_mean": statistics.fmean(cpu) if cpu else None,
        "emission_lag": srv.strip().splitlines()[-1].split("emission lag ")[-1] if "emission lag" in srv else None,
    }
    for k in ("latency_ms", "recv_to_done_ms", "wait_ms"):
        out[k] = {a: (round(b, 2) if isinstance(b, float) else b) for a, b in out[k].items()}
    return out


def plan_runs(board, plan):
    if plan == "correct":
        pc = str(DATA / "2021-04-16_023917_PlasticContainer.aris")
        return [dict(tag=f"correct_{m}", model=m, workers=1, policy="queue", duration=0, recs=[pc], no_loop=True)
                for m in ("small", "segdice")]
    if plan == "sweep" and board == "pi4":
        rs = []
        for fps in (15, 30, 60, 120):
            for w in (1, 2, 4):
                for pol in (("latest", "queue") if fps <= 60 else ("latest",)):
                    rs.append(dict(tag=f"sw_small_w{w}_f{fps}_{pol}", workers=w, policy=pol, fps=fps))
        rs += [dict(tag=f"sw_segdice_w1_f15_{pol}", model="segdice", workers=1, policy=pol, fps=15) for pol in ("latest", "queue")]
        return rs
    if plan == "sweep" and board == "pi2":
        rs = []
        for fps in (5, 10, 15):
            for w in (1, 2, 3, 4):
                for pol in ("latest", "queue"):
                    rs.append(dict(tag=f"sw_small_w{w}_f{fps}_{pol}", workers=w, policy=pol, fps=fps))
        rs += [dict(tag=f"sw_segdice_w{w}_f15_latest", model="segdice", workers=w, policy="latest", fps=15) for w in (3, 4)]
        return rs
    if plan == "long":
        w = int(os.environ.get("LONG_WORKERS", "1"))
        return [dict(tag=f"long_small_w{w}_f15_latest", workers=w, policy="latest", fps=15, duration=1800.0)]
    raise SystemExit(f"unknown plan {plan}")


def main():
    board, plan = sys.argv[1], sys.argv[2]
    out = HERE / f"summary_{board}_{plan}.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for cfg in plan_runs(board, plan):
        if cfg["tag"] in res and "--redo" not in sys.argv:
            continue
        s = run(board, **cfg)
        res[cfg["tag"]] = s
        out.write_text(json.dumps(res, indent=1))
        L = s["latency_ms"]
        print(f"{board} {cfg['tag']}: processed {s['processed']}/{s['produced']} ({s['processed_fps']} fps), "
              f"lat p50 {L['p50']} p95 {L['p95']} p99 {L['p99']} ms, met {s['deadline_met_pct_of_produced']}%, "
              f"temp {s['temp_c']['start']}->{s['temp_c']['end']} (max {s['temp_c']['max']}), thr {s['throttled']}", flush=True)


if __name__ == "__main__":
    main()
