"""Check the C ARIS reader + front end against two independent projections of the same raw frame.

  ref A: NumPy, written from the SDK formulas (docs/understanding-aris-data.md) and the SDK
         beam-centre table, bilinear in (sample, beam): same interpolation as the C front end,
         separate code for parsing, geometry and beam lookup.
  ref B: pyARIS (EminentCodfish), its own header parser, beam table and remap (nearest
         neighbour, metric grid of one sample length per pixel), resampled onto our pixels.

usage: venv/bin/python -I validate.py FILE.aris K [K ...]   (needs aris_dump built next to this file)
"""
import os
import re
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BEAMS = {1: 48, 2: 48, 3: 96, 4: 96, 5: 96, 6: 64, 7: 64, 8: 64, 9: 128, 10: 128, 11: 128, 12: 128}


def sdk_table(n):
    txt = open(os.path.join(HERE, "beamwidths", f"BeamWidths_ARIS3000_{n}.h")).read()
    rows = re.findall(r"DEFINE_BEAMWIDTH3\((\d+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)", txt)
    return np.array([[float(c), float(lo), float(hi)] for _, c, lo, hi in rows])


def raw_frame(path, k):
    with open(path, "rb") as f:
        f.seek(1024)
        h = f.read(1024)
        pm, sp = struct.unpack_from("<I", h, 436)[0], struct.unpack_from("<I", h, 452)[0]
        c = struct.unpack_from("<f", h, 464)[0]
        spb, ssd = struct.unpack_from("<I", h, 468)[0], struct.unpack_from("<I", h, 476)[0]
        nb = BEAMS[pm]
        f.seek(1024 + k * (1024 + nb * spb))
        h = f.read(1024)
        c = struct.unpack_from("<f", h, 464)[0]
        data = np.frombuffer(f.read(nb * spb), np.uint8).reshape(spb, nb).astype(np.float64)
    return data, nb, spb, ssd * 1e-6 * c / 2, sp * 1e-6 * c / 2


def pgm(path):
    b = open(path, "rb").read()
    m = re.match(rb"P5\s+(\d+)\s+(\d+)\s+255\s", b)
    w, h = int(m.group(1)), int(m.group(2))
    return np.frombuffer(b[m.end():], np.uint8).reshape(h, w).astype(np.float64)


def stats(name, a, b, mask):
    d = np.abs(a - b)[mask]
    print(f"  {name}: {mask.sum()} px, mean |diff| {d.mean():.2f}, median {np.median(d):.0f}, p99 {np.percentile(d, 99):.0f}, "
          f"max {d.max():.0f}, identical {100 * np.mean(d == 0):.1f}%, corr {np.corrcoef(a[mask], b[mask])[0, 1]:.4f}")


def main():
    path, ks = sys.argv[1], [int(k) for k in sys.argv[2:]]
    sys.path.insert(0, os.path.join(HERE, "ext", "pyARIS"))
    import pyARIS
    for k in ks:
        out = os.path.join(HERE, f"val_{k}.pgm")
        geo = subprocess.run([os.path.join(HERE, "aris_dump"), path, str(k), "0", out], capture_output=True, text=True, check=True).stdout.split()
        au, av, fov, rmin, rmax, mpp, fw, fh = map(float, geo[:8])
        ours = pgm(out)
        H, W = ours.shape
        data, nb, ns, ws, sl = raw_frame(path, k)
        tab = sdk_table(nb)
        # pixel centres of the network input -> fan pixels -> range (m), bearing (deg, + right)
        y, x = np.mgrid[0:H, 0:W]
        u, v = (x + 0.5) * fw / W, (y + 0.5) * fh / H
        dx, dy = u - au, -(v - av)
        r = np.hypot(dx, dy) * mpp
        bear = np.degrees(np.arctan2(dx, dy))
        # ref A: SDK sample range and beam-centre table (SDK angle = -bearing, beam 0 right-most)
        s = (r - ws) / sl
        th = -bear
        cen = tab[:, 0]
        b = np.interp(th, cen, np.arange(nb), left=np.nan, right=np.nan)
        inside = (s >= 0) & (s <= ns - 1) & (th >= tab[0, 1]) & (th <= tab[-1, 2])
        b = np.clip(np.nan_to_num(b, nan=0.0), 0, nb - 1)
        b = np.where(th < cen[0], 0, np.where(th > cen[-1], nb - 1, b))
        sc = np.clip(s, 0, ns - 1)
        s0, b0 = np.minimum(sc.astype(int), ns - 2), np.minimum(b.astype(int), nb - 2)
        fs, fb = sc - s0, b - b0
        refA = ((1 - fs) * (1 - fb) * data[s0, b0] + (1 - fs) * fb * data[s0, b0 + 1]
                + fs * (1 - fb) * data[s0 + 1, b0] + fs * fb * data[s0 + 1, b0 + 1])
        refA = np.where(inside, np.round(refA), 0)
        # ref B: pyARIS remap, then sample its metric image at our pixels (nearest)
        ad = pyARIS.DataImport(path)
        fr = pyARIS.FrameRead(ad, k)
        img = fr.remap.astype(np.float64)        # after flipud: row 0 = far
        pix = fr.sampleperiod * 1e-6 * fr.soundspeed / 2
        xm, ym = r * np.sin(np.radians(bear)), r * np.cos(np.radians(bear))
        col = np.round(xm / pix + ad.xdim / 2).astype(int)
        row = ad.ydim - 1 - np.round((ym - fr.WinStart) / pix).astype(int)
        okB = (col >= 0) & (col < ad.xdim) & (row >= 0) & (row < ad.ydim)
        refB = np.zeros_like(ours)
        refB[okB] = img[row[okB], col[okB]]
        core = inside & (ours > 0)
        print(f"frame {k}: {nb} beams x {ns} samples, window {ws:.3f}-{ws + ns * sl:.3f} m, our fan: r {rmin:.1f}-{rmax:.1f} px, "
              f"{1000 * mpp:.2f} mm/px, fov {fov:.2f}; pyARIS grid {ad.xdim}x{ad.ydim}")
        stats("C vs ref A (SDK formulas, bilinear)", ours, refA, inside)
        # pyARIS: compare away from the fan edges, where its narrower beam table (+-14.83 deg) and nearest
        # sampling differ by construction
        coreB = core & okB & (np.abs(bear) < 13.5) & (refB > 0)
        stats("C vs ref B (pyARIS, nearest)", ours, refB, coreB)
        stats("ref A vs ref B", refA, refB, coreB)
        np.save(os.path.join(HERE, f"val_{k}_refs.npy"), np.stack([ours, refA, refB]))


if __name__ == "__main__":
    main()
