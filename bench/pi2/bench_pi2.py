"""M5 on the Raspberry Pi 2 (Cortex-A7, 32-bit ARMv7): same models, frames and method as the Pi 4.

Builds per model:
  scalar_v6  Raspbian default target (ARMv6 + VFP, no NEON), as `make` gives on this OS
  scalar_a7  -mcpu=cortex-a7 -mfpu=neon-vfpv4 -DSE_NO_NEON (gcc may auto-vectorise)
  neon1      a7 flags, NEON 1-pixel kernel (-DSE_NEON_1PX)
  neon4      a7 flags, NEON 4-pixel kernel
Each: golden test, size, one-core latency (3 repeats over 10 frames).
neon4 also: all test-fold-0 frames -> detections, compared byte-for-byte with the Pi 4 scalar output.
Fastest NEON build and scalar_a7: peak RSS and 4-copy throughput. Temperatures and throttle flags throughout.
"""
import filecmp, glob, json, os, subprocess, sys, time

ROOT = os.path.expanduser("~/sonar_edge_pi2")
PGM = {"small_cv_f0": "small_cv_f0", "small_360_cv_f0": "small_360_cv_f0",
       "small_wide_360_cv_f0": "small_360_cv_f0", "wide_cv_f0": "wide_cv_f0"}
BASE = "-std=c11 -O2 -Wall -Wextra -Werror -pedantic"
A7 = BASE + " -mcpu=cortex-a7 -mfpu=neon-vfpv4"
VARIANTS = {"scalar_v6": BASE + " -DSE_NO_NEON", "scalar_a7": A7 + " -DSE_NO_NEON",
            "neon1": A7 + " -DSE_NEON_1PX", "neon4": A7}
REPEAT_S = 25.0  # target seconds per latency repeat


def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
def temp(): return sh("vcgencmd measure_temp").split("=")[-1]
def thr(): return sh("vcgencmd get_throttled").split("=")[-1]
def frames(m): return sorted(glob.glob(f"{ROOT}/pgm/{PGM[m]}/*.pgm"), key=lambda p: int(os.path.basename(p)[:-4]))
def sel(m): f = frames(m); return f[:: len(f) // 10][:10]
def ms(e): return float(e.split("mean ")[1].split(" ms")[0])
def run_ms(b, m, n): return ms(subprocess.run([b, "-n", str(n)] + sel(m), capture_output=True, text=True, check=True).stderr)


def cool(limit=50.0, max_s=180):
    t0 = time.time()
    while float(temp().rstrip("'C")) > limit and time.time() - t0 < max_s:
        time.sleep(10)


def latency(b, m):
    est = run_ms(b, m, 2)                       # two passes over the 10 frames (the CLI reports a mean only for -n > 1), also warms up
    n = max(2, round(REPEAT_S * 1000 / (10 * est)))  # the CLI prints a mean only for -n > 1
    return n, [run_ms(b, m, n) for _ in range(3)]


def peak_mem(b, m):
    p = subprocess.Popen([b, "-n", "100000", sel(m)[0]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    st = dict(l.split(":", 1) for l in open(f"/proc/{p.pid}/status") if ":" in l)
    p.kill(); p.wait()
    return {k: st[k].strip() for k in ("VmHWM", "VmRSS", "VmSize")}


def throughput(b, m, n):
    t_start = temp()
    t0 = time.time()
    ps = [subprocess.Popen([b, "-n", str(n)] + sel(m), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    per = [ms(p.communicate()[1]) for p in ps]
    wall = time.time() - t0
    return {"n": n, "per_proc_ms": per, "frames_per_s": 4 * 10 * n / wall, "wall_s": round(wall, 1),
            "temp_start": t_start, "temp_end": temp(), "throttled": thr()}


def main():
    models = sys.argv[1:] or list(PGM)
    out_f = f"{ROOT}/bench_pi2.json"
    out = json.load(open(out_f)) if os.path.exists(out_f) else {}
    out["_board"] = {"model": sh("cat /proc/device-tree/model | tr -d '\\0'"), "kernel": sh("uname -r"),
                     "gcc": sh("gcc --version | head -1"), "governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
                     "max_khz": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"), "throttled_start": thr()}
    for m in models:
        r = out.setdefault(m, {})
        for var, fl in VARIANTS.items():
            dst = f"{ROOT}/build/{var}/{m}"
            sh(f"rm -rf {dst} && mkdir -p {dst} && cp -r {ROOT}/models/{m}/. {dst}/ && make -s -C {dst} clean")
            err = sh(f'make -s -C {dst} CFLAGS="{fl}" all 2>&1')
            v = {"cflags": fl, "build": err or "ok",
                 "golden": sh(f'make -s -C {dst} CFLAGS="{fl}" test 2>&1 | tail -1'),
                 "size": sh(f"size {dst}/sonar-detect | tail -1"),
                 "neon_insns": int(sh(f"objdump -d {dst}/sonar_edge.o | grep -c -E 'vmlal|vmla\\.'") or 0)}
            b = f"{dst}/sonar-detect"
            cool()
            v["temp_start"] = temp()
            if var == "neon4":
                t0 = time.time()
                p = subprocess.run([b] + frames(m), capture_output=True, text=True, check=True)
                fn = f"{ROOT}/dets_{var}_{m}.txt"
                with open(fn, "w") as f:
                    for line in p.stdout.splitlines():
                        a = line.split(); a[0] = os.path.basename(a[0]); f.write(" ".join(a) + "\n")
                v["fold_frames"] = len(frames(m))
                v["fold_s"] = round(time.time() - t0, 1)
                v["identical_to_pi4_scalar"] = filecmp.cmp(fn, f"{ROOT}/ref/dets_{m}.txt", shallow=False)
                cool()
            v["n"], v["latency_ms"] = latency(b, m)
            v["temp_end"], v["throttled"] = temp(), thr()
            r[var] = v
            print(m, var, json.dumps(v), flush=True)
            json.dump(out, open(out_f, "w"), indent=1)
        best = min(("neon1", "neon4"), key=lambda k: sum(r[k]["latency_ms"]))
        r["best_neon"] = best
        for var in ("scalar_a7", best):
            b = f"{ROOT}/build/{var}/{m}/sonar-detect"
            r[var]["mem"] = peak_mem(b, m)
            cool()
            n = max(2, round(60_000 / (10 * sum(r[var]["latency_ms"]) / 3)))  # ~60 s per copy, -n > 1
            r[var]["throughput4"] = throughput(b, m, n)
            print(m, var, "mem+tp", json.dumps({k: r[var][k] for k in ("mem", "throughput4")}), flush=True)
            json.dump(out, open(out_f, "w"), indent=1)
    out["_board"]["throttled_end"] = thr()
    json.dump(out, open(out_f, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
