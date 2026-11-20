"""Board sampler for stream runs (runs on the Pi): every PERIOD s append
  t_unix, temp_c, arm_hz, throttled, load1, cpu_pct
to OUT.csv, for DURATION s or until killed.

  python3 sampler.py OUT.csv DURATION [PERIOD]
"""
import subprocess
import sys
import time


def vc(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def cpu_times():
    f = open("/proc/stat").readline().split()[1:]
    v = list(map(int, f))
    idle = v[3] + v[4]
    return sum(v), idle


def main():
    out, duration = sys.argv[1], float(sys.argv[2])
    period = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    t_end = time.time() + duration
    tot0, idle0 = cpu_times()
    with open(out, "w") as f:
        f.write("t_unix,temp_c,arm_hz,throttled,load1,cpu_pct\n")
        while time.time() < t_end:
            temp = vc("measure_temp").split("=")[-1].rstrip("'C")
            clk = vc("measure_clock", "arm").split("=")[-1]
            thr = vc("get_throttled").split("=")[-1]
            load1 = open("/proc/loadavg").read().split()[0]
            tot, idle = cpu_times()
            cpu = 100.0 * (1 - (idle - idle0) / max(1, tot - tot0))
            tot0, idle0 = tot, idle
            f.write(f"{time.time():.3f},{temp},{clk},{thr},{load1},{cpu:.1f}\n")
            f.flush()
            time.sleep(period)


if __name__ == "__main__":
    main()
