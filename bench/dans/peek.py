"""Print ARIS file/frame header fields (offsets from SoundMetrics aris-file-sdk type-definitions)."""
import os
import struct
import sys

BEAMS = {1: 48, 2: 48, 3: 96, 4: 96, 5: 96, 6: 64, 7: 64, 8: 64, 9: 128, 10: 128, 11: 128, 12: 128}


def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def f32(b, o): return struct.unpack_from("<f", b, o)[0]


def frame_hdr(b):
    return {"FrameIndex": u32(b, 0), "Version": hex(u32(b, 12)), "PingMode": u32(b, 436), "SamplePeriod": u32(b, 452),
            "FrameRate": f32(b, 460), "SoundSpeed": f32(b, 464), "SamplesPerBeam": u32(b, 468),
            "SampleStartDelay": u32(b, 476), "LargeLens": u32(b, 480), "SystemType": u32(b, 484),
            "Reordered": u32(b, 516), "SampleRate": f32(b, 420)}


for p in sys.argv[1:]:
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        fh = f.read(1024)
        h0 = frame_hdr(f.read(1024))
    nb = BEAMS.get(h0["PingMode"], 0)
    fs = nb * h0["SamplesPerBeam"]
    n = (size - 1024) // (1024 + fs) if fs else 0
    c = h0["SoundSpeed"]
    ws = h0["SampleStartDelay"] * 1e-6 * c / 2
    sl = h0["SamplePeriod"] * 1e-6 * c / 2
    print(os.path.basename(p), f"file v={hex(u32(fh, 0))} hdrFrameCount={u32(fh, 4)} NumRawBeams={u32(fh, 16)} SPC={u32(fh, 24)}",
          f"| beams={nb} frames={n} rem={(size - 1024) % (1024 + fs) if fs else -1}",
          f"| window {ws:.2f}-{ws + sl * h0['SamplesPerBeam']:.2f} m, sample {sl * 100:.2f} cm", h0)
    if len(sys.argv) == 2 and n > 1:   # check that per-frame geometry stays constant
        keys = ("PingMode", "SamplePeriod", "SamplesPerBeam", "SampleStartDelay", "SoundSpeed", "SystemType", "LargeLens")
        changes = 0
        with open(p, "rb") as f:
            for k in range(n):
                f.seek(1024 + k * (1024 + fs))
                h = frame_hdr(f.read(1024))
                if any(h[x] != h0[x] for x in keys):
                    changes += 1
                    if changes <= 3:
                        print("  frame", k, {x: h[x] for x in keys})
        print("  frames with changed geometry:", changes)
