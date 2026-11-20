"""Compare a stream-mode JSONL (sonar-edge-run --stream) with file-mode output (--aris)
for the same recording: every field except timings ("ms", "stream", "frame", "source")
must match per recording frame index.

  python3 compare_stream.py file.jsonl stream.jsonl [--rec-id 0]
"""
import json
import sys


def key_fields(d):
    return {k: v for k, v in d.items() if k not in ("ms", "stream", "frame", "source")}


def main():
    a, b = sys.argv[1], sys.argv[2]
    rec_id = int(sys.argv[sys.argv.index("--rec-id") + 1]) if "--rec-id" in sys.argv else 0
    ref = {}
    for line in open(a):
        d = json.loads(line)
        ref[d["index"]] = key_fields(d)
    same = diff = 0
    seen = set()
    for line in open(b):
        d = json.loads(line)
        if d["stream"]["rec_id"] != rec_id:
            continue
        i = d["index"]
        if i in seen:   # looped recording: compare every pass
            pass
        seen.add(i)
        if key_fields(d) == ref.get(i):
            same += 1
        else:
            diff += 1
    missing = len(set(ref) - seen)
    print(json.dumps({"reference_frames": len(ref), "stream_frames_compared": same + diff, "identical": same,
                      "different": diff, "reference_frames_not_streamed": missing}))
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
