"""Overlay strips for DANS recordings: segdice model, boxes at p >= SHOW and the mask.

usage: SE_RUN_SEG=<dir>/sonar-edge-run uv run python bench/dans/overlays.py OUTDIR [N_PER_REC] FILE.aris [...]
For each recording, N_PER_REC frames evenly spaced through it are rendered by
`sonar-edge-run -o` (overlay at network resolution); PNGs per frame plus one strip
per recording are written to OUTDIR.
"""
import glob
import os
import struct
import subprocess
import sys

from PIL import Image

BEAMS = {1: 48, 2: 48, 3: 96, 4: 96, 5: 96, 6: 64, 7: 64, 8: 64, 9: 128, 10: 128, 11: 128, 12: 128}


def n_frames(path):
    with open(path, "rb") as f:
        f.seek(1024)
        h = f.read(1024)
    fs = BEAMS[struct.unpack_from("<I", h, 436)[0]] * struct.unpack_from("<I", h, 468)[0]
    return (os.path.getsize(path) - 1024) // (1024 + fs)


def main():
    out = sys.argv[1]
    k = int(sys.argv[2]) if sys.argv[2].isdigit() else 6
    files = sys.argv[3:] if sys.argv[2].isdigit() else sys.argv[2:]
    binp = os.environ["SE_RUN_SEG"]
    for path in files:
        name = os.path.basename(path)[:-5]
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        nf = n_frames(path)
        every = max(1, nf // k)
        subprocess.run([binp, "-o", d, "--aris", "--every", str(every), "--count", str(k), path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        ims = []
        for p in sorted(glob.glob(os.path.join(d, "*.ppm"))):
            im = Image.open(p)
            im.save(p[:-4] + ".png")
            os.remove(p)
            ims.append(im)
        if ims:
            w, h = ims[0].size
            strip = Image.new("RGB", (w * len(ims), h))
            for i, im in enumerate(ims):
                strip.paste(im, (i * w, 0))
            strip.save(os.path.join(out, f"{name}.strip.png"))
        print(name, nf, "frames, every", every, "->", len(ims), "overlays", flush=True)


if __name__ == "__main__":
    main()
