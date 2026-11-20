"""End-to-end on-device run of sonar-edge-run (M8) over test fold 0.

Run on each board (and on the laptop for the reference) from a work dir that
holds `runtime/` (clean sources), `in/fan/<idx>.pgm` (439 native 320x480 fan
images), `in/polar.u8` (the same frames back-projected to 439 x 128 polar,
sonar_edge.polar) and `in/order.json`:

  python3 run_e2e.py WORKDIR LABEL [--cflags "..."] [--variants neon4 neon1] [--cool 50]

Per model (shipped small_360_cv_f0, models/small_360_segdice) and variant:
build `all sonar-edge-run`, run `make test` and `make test-frontend`, then
sonar-edge-run -m over all fan frames and over all polar frames. Writes
WORKDIR/out/<model>_<variant>_<mode>.jsonl, masks under WORKDIR/masks/, and
WORKDIR/out/summary_<LABEL>.json (build/test results, temperatures, throttle).
"""
import argparse, json, os, shutil, subprocess, time

MODELS = {"small360": "", "segdice": "models/small_360_segdice"}
BASE = "-std=c11 -O2 -Wall -Wextra -Werror -pedantic"


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def vc(arg):
    if not shutil.which("vcgencmd"):
        return None
    return run(["vcgencmd", arg])[1].split("=")[-1]


def temp_c():
    t = vc("measure_temp")
    return float(t.rstrip("'C")) if t else None


def cool(limit):
    t0 = time.time()
    while limit and (temp_c() or 0) > limit and time.time() - t0 < 600:
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("label")
    ap.add_argument("--cflags", default=BASE)
    ap.add_argument("--variants", nargs="+", default=["default"])
    ap.add_argument("--cool", type=float, default=0)
    a = ap.parse_args()
    W = os.path.abspath(a.work)
    order = json.load(open(f"{W}/in/order.json"))
    fans = [f"{W}/in/fan/{i}.pgm" for i in order["index"]]
    os.makedirs(f"{W}/out", exist_ok=True)
    summ = {"label": a.label, "cflags": a.cflags, "uname": run(["uname", "-a"])[1],
            "throttled_start": vc("get_throttled"), "runs": {}}
    for model, wdir in MODELS.items():
        for var in a.variants:
            fl = a.cflags + (" -DSE_NEON_1PX" if var == "neon1" else "")
            b = f"{W}/build/{model}_{var}"
            shutil.rmtree(b, ignore_errors=True)
            shutil.copytree(f"{W}/runtime", b)
            mk = ["make", "-s", "-C", b, f"CFLAGS={fl}"] + ([f"W={wdir}"] if wdir else [])
            r = {"cflags": fl}
            r["build_rc"], r["build"] = run(mk + ["clean", "all", "sonar-edge-run"])
            r["test_rc"], r["test"] = run(mk + ["test"])
            r["test_frontend_rc"], r["test_frontend"] = run(mk + ["test-frontend"])
            r["test"] = r["test"].splitlines()[-1] if r["test"] else ""
            r["test_frontend"] = r["test_frontend"].splitlines()[-1] if r["test_frontend"] else ""
            exe = f"{b}/sonar-edge-run"
            for mode in ("fan", "polar"):
                tag = f"{model}_{var}_{mode}"
                md = f"{W}/masks/{tag}"
                shutil.rmtree(md, ignore_errors=True)
                os.makedirs(md)
                cool(a.cool)
                args = [exe, "-m", md] + (fans if mode == "fan" else
                                          ["--polar", f"{order['beams']}x{order['samples']}", f"{W}/in/polar.u8"])
                t_start, thr_start = temp_c(), vc("get_throttled")
                t0 = time.time()
                p = subprocess.run(args, capture_output=True, text=True)
                wall = time.time() - t0
                open(f"{W}/out/{tag}.jsonl", "w").write(p.stdout)
                r[mode] = {"rc": p.returncode, "stderr": p.stderr.strip()[-500:], "lines": len(p.stdout.splitlines()),
                           "wall_s": round(wall, 2), "temp_start": t_start, "temp_end": temp_c(),
                           "throttled_start": thr_start, "throttled_end": vc("get_throttled")}
                print(tag, json.dumps({k: r[k] for k in ("build_rc", "test", "test_frontend")}), json.dumps(r[mode]), flush=True)
            summ["runs"][f"{model}_{var}"] = r
            json.dump(summ, open(f"{W}/out/summary_{a.label}.json", "w"), indent=1)
    summ["throttled_end"] = vc("get_throttled")
    json.dump(summ, open(f"{W}/out/summary_{a.label}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
