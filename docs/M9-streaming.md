# M9: real-time operation on a live sonar stream (emulated)

Measured 2026-11-20. Raw results, scripts and plots are in `bench/stream/`.

## 1. Motivation: the edge task

The deployment target is onboard debris detection on a small ROV or AUV.
The sonar produces frames at its own rate (15 fps for the ARIS 3000
recordings of M8 §6), and the vehicle's link cannot carry sonar video: one
raw frame is 53-124 KB (128 beams x 411-967 samples), 0.8-1.9 MB/s at
15 fps. The detector must therefore keep up with the sensor on the vehicle's
computer and send only results. Single-frame benchmarks (M5, M8) give the
cost per frame. They do not show whether a board keeps up with a stream
over time, how late each result is, what happens when it falls behind, or
whether it heats up and throttles during a long run. This milestone measures
those, with recorded sonar data replayed in real time over the network.

Scope: the system is an image-to-image pipeline (the sonar software's
fan image in, detections and mask out; M8). The emulator here sends raw
beam data from the DANS `.aris` recordings, so the front end's polar path
runs too. Driving a live ARIS over its own network protocol is future work.

## 2. Emulator and protocol

`sonar-stream` (`runtime/stream/sonar_stream.c`, protocol in
`runtime/stream/protocol.h`) replays `.aris` recordings at their recorded
frame rate (or `--fps F`), or fan PGMs at a chosen rate. The design
borrows from SeedLink, the seismology streaming protocol.

- **Transport:** TCP, one connection per client; several clients at once.
- **Messages from the client:** fixed 32-byte messages, all little-endian.
  - `HELO`: a name, for the server log.
  - `STRT [from_seq]`: start streaming, at the newest frame or at a sequence number.
  - `ACK`: one per processed frame.
  - `BYE`: close.
- **Frames from the server:** a 64-byte header, then `n_beams x n_samples` bytes of
  beam data. The header holds:
  - magic and version;
  - sequence number;
  - capture and send times;
  - recording and frame index;
  - the ARIS geometry the front end needs: system, ping mode, lens, sample period, start
    delay, sound speed and frame rate.
- **Resume:** the server keeps the last 256 frames in a ring buffer. A client that
  reconnects with `STRT from_seq` continues where it stopped, if that frame is
  still in the ring. Tested: a client resumed at sequence 45.
- **Fault injection:** `--jitter MS` and `--drop P` (dropped frames keep their
  sequence numbers, so clients see the gaps). Neither was used in the runs below.
- **Latency:** each frame's capture time is its nominal time on the server
  clock. The client's ACK carries that timestamp back, and the server logs
  capture-to-result latency as ACK arrival minus capture time. Everything is
  measured on one clock, because the laptop's clock was not NTP-synchronised
  (the Pis were, within +1.0 / -1.7 ms of each other). The measured latency
  therefore covers network transfer, waiting, processing and the ACK's return.

Client: `sonar-edge-run --stream HOST:PORT [--policy latest|queue] [--workers N]`
(`runtime/stream/stream_client.c`).
- **Threads and processes:** a receive thread reads frames. N worker processes
  each run the front end, network, decoder and (for the segdice model) the
  mask, then write a JSON line and an ACK. They are processes because the
  runtime uses static memory and is not thread-safe.
- **`latest` policy:** a free worker takes the newest frame. Older frames still
  waiting are skipped and counted as dropped. This suits an operator display or
  a control loop, which want the freshest result.
- **`queue` policy:** a bounded FIFO. Every frame is processed, and latency
  grows when the workers cannot keep up.

Measurement (`bench/stream/run_stream.py`, `sampler.py`):
- **Per run:** frames produced, processed and dropped; latency percentiles;
  queue depth.
- **Board state:** a sampler on the Pi records temperature, ARM clock,
  `get_throttled` and CPU load every 2 s.
- **Cooling:** boards were cooled below 50 C before each run.
- **Data:** six DANS recordings, looped. All are ARIS 3000, 128 beams, 15 fps,
  411-967 samples per beam.

## 3. Network path

| Path | TCP throughput | RTT (1400 B ping) |
|---|---|---|
| Laptop (wired, `eno1`) -> switch -> Pi 4 | 941 Mbit/s | 0.33 ms |
| Laptop (wired) -> switch -> Pi 2 | 94.2 Mbit/s (its 100 Mbit port) | 0.85 ms |

The stream needs at most about 15 Mbit/s, so neither link limits throughput.
Transfer time is part of the latency, though. A 128 x 823 frame (105 KB)
takes about 0.9 ms at 1 Gbit/s and 8.4 ms at 100 Mbit/s.

## 4. Correctness

The PlasticContainer recording (195 frames) was streamed with the queue
policy and one worker, and compared with file mode (`sonar-edge-run --aris`
on the laptop, same build). Compared fields: everything except timings and
stream metadata.

| Board | small@360 | small_360_segdice |
|---|---|---|
| Pi 4 | 195 / 195 identical | 195 / 195 identical |
| Pi 2 | 195 / 195 identical | 195 / 195 identical |

The `sonar-edge-run` changes leave file-mode output unchanged: 439 / 439
test-fold frames are identical to the committed build apart from timings.

## 5. Sweeps (60 s per run)

Latency is capture to result on the server clock, in ms.

**Pi 4, small@360**

| Rate | Workers | Policy | Processed fps (dropped) | Latency p50 / p95 / p99 | Max temp |
|---|---|---|---|---|---|
| 15 | 1 | latest | 15.0 (0%) | 22.6 / 23.0 / 39.6 | 51.6 C |
| 15 | 2 | latest | 15.0 (0%) | 22.7 / 39.8 / 44.4 | 53.5 C |
| 15 | 4 | latest | 15.0 (0%) | 22.7 / 39.7 / 48.8 | 54.0 C |
| 30 | 1 | latest | 30.0 (0%) | 22.3 / 22.7 / 23.0 | 56.0 C |
| 60 | 1 | latest | 46.2 (23.0%) | 30.8 / 38.3 / 39.0 | 59.4 C |
| 60 | 1 | queue | 52.0 (13.4%) | 5825 / 6308 / 7025 | 58.9 C |
| 60 | 2 | latest | 59.5 (0.8%) | 22.7 / 23.1 / 23.7 | 62.3 C |
| 60 | 4 | latest | 59.9 (0.2%) | 22.9 / 23.3 / 36.9 | 61.8 C |
| 120 | 2 | latest | 90.8 (24.3%) | 26.5 / 30.5 / 31.2 | 66.2 C |
| 120 | 4 | latest | 119.9 (0.1%) | 23.1 / 24.4 / 27.0 | 71.5 C |
| 15, segdice | 1 | latest | 15.0 (0%) | 23.9 / 24.4 / 41.9 | 54.5 C |

**Pi 2, small@360**

| Rate | Workers | Policy | Processed fps (dropped) | Latency p50 / p95 / p99 | Max temp |
|---|---|---|---|---|---|
| 5 | 1 | latest | 5.0 (0%) | 138.6 / 140.3 / 161.0 | 42.2 C |
| 10 | 1 | latest | 7.3 (26.8%) | 193.0 / 238.7 / 242.1 | 43.9 C |
| 10 | 1 | queue | 9.4 (6.2%) | 8915 / 16428 / 17096 | 43.9 C |
| 10 | 2 | latest | 10.0 (0.2%) | 141.0 / 142.9 / 196.2 | 44.4 C |
| 15 | 1 | latest | 7.4 (50.7%) | 178.2 / 207.5 / 210.0 | 45.5 C |
| 15 | 1 | queue | 9.2 (38.4%) | 17864 / 23081 / 24484 | 44.4 C |
| 15 | 2 | latest | 14.8 (1.7%) | 170.1 / 201.2 / 209.2 | 47.1 C |
| 15 | 2 | queue | 15.0 (0%) | 714 / 1201 / 1247 | 47.6 C |
| 15 | 3 | latest | 15.0 (0.1%) | 143.0 / 147.1 / 171.5 | 48.2 C |
| 15 | 4 | latest | 14.9 (0.7%) | 143.3 / 153.0 / 259.5 | 48.7 C |
| 15, segdice | 3 | latest | 15.0 (0.2%) | 151.2 / 156.8 / 211.6 | 49.2 C |

The tables are a selection; every run is in `summary_{pi4,pi2}_sweep.json`.
None of the sweep runs set a throttle flag.

## 6. Sustained runs (30 min, 15 fps, latest policy, small@360)

| | Pi 4, 1 worker | Pi 2, 3 workers |
|---|---|---|
| Frames processed | 26,993 / 27,000 (15.0 fps) | 26,939 / 27,000 (14.97 fps) |
| Dropped | 7 (0.03%) | 61 (0.23%) |
| Latency p50 / p95 / p99 / max | 29.9 / 31.9 / 47.3 / 352 ms | 144.6 / 149.0 / 160.1 / 1688 ms |
| Receive to result, p50 | 21.3 ms | 134.6 ms |
| Mean stage times (front end / network / decode) | 0.55 / 20.35 / 0.38 ms | 3.01 / 127.16 / 1.65 ms |
| CPU load, mean | 9% | 54% |
| Temperature start -> end (max) | 49.6 -> 53.5 C (56.9) | 39.5 -> 46.5 C (50.3) |
| ARM clock, median (min) | 1800 MHz (600) | 900 MHz (600) |
| `get_throttled` | 0x0 throughout | 0x0 throughout |

Plots: `long_runs.png` (temperature, clock, processed fps and latency over
time) and `latency_cdf.png` (sweep latency distributions at 15 fps).

## 7. Discussion

- **Both boards meet the edge task.**
  - The fanless Pi 4 keeps up with a 15 fps stream on one core for 30 minutes.
    Each result arrives about 30 ms after capture, at 9% CPU. The board levels
    off near 55 C, well below its 80 C soft limit, and never throttles. Its
    clock drops below 1.8 GHz now and then; with the throttle flags at 0, that
    is the `ondemand` governor reacting to the low load.
  - The Pi 2 also sustains 15 fps, but needs 3 of its 4 cores and returns each
    result about 145 ms after capture.
- **Throughput is not latency.** On the Pi 2 one frame takes about 130 ms, two
  frame periods. With 3 workers it works on several frames at once: the
  throughput matches the sensor, but each result is two frames old. The
  "deadline met" share (result within one frame period, 66.7 ms) is therefore
  0% on the Pi 2 at 15 fps, and 99% at 5 fps (200 ms period). A Pi 2
  vehicle would see the scene about 0.15 s late; a Pi 4 about 0.03 s late.
- **Headroom.** The Pi 4 runs at 30 fps on one worker and 120 fps on four, at
  about 23 ms latency, so the operating point uses about an eighth of its
  capacity. The 4-worker 120 fps run reached 71.5 C in 60 s, so sustained use
  near full load would need a heat sink (as M5 §3.3 found).
- **Policy.** When the workers cannot keep up, the queue policy lets latency
  grow without bound: 6-24 s backlogs within a minute. The latest policy keeps
  latency at about one processing time and drops frames instead. For a vehicle,
  latest is the right default.
- **Multi-worker tail on the Pi 4.** At 15 fps, 2-4 workers give a p95 of
  about 40 ms against 23 ms with one worker. About 15-20% of frames take
  around 37-40 ms (the step in `latency_cdf.png`). The cause was not
  investigated. One possibility is that a frame handed to an idle core waits
  for that core to clock up from 600 MHz under `ondemand`. One worker is the
  better configuration on the Pi 4.
- **Pi 4 link incident.** During the first Pi 4 30-minute run, its Ethernet
  link went down and up 131 times between 23:05 and 23:17 (`journalctl`),
  then came back at 100 Mbit/s instead of 1 Gbit/s.
  - In that window 360 of 11,025 frames were acknowledged, and the server
    skipped 11,036 frames in total.
  - The Pi 2 on the same switch was unaffected, so the Pi 4's cable or port is
    the likely cause.
  - That run is kept as `summary_pi4_long_interrupted.json`. Before the
    failure it ran at 22.7 ms p50 for 739 s.
  - The run was repeated over the 100 Mbit/s link, which still worked. That is
    why its latency (29.9 ms p50) is 7 ms above the sweep (22.6 ms): receive to
    result is 21.3 ms in both, and transferring a 105 KB frame takes about 8 ms
    at 100 Mbit/s against about 1 ms at 1 Gbit/s.
- **Bandwidth.**
  - The full JSON line `sonar-edge-run` writes per frame is about 13 KB
    (12,972 B small@360, 13,090 B segdice; mean over the 15 fps sweep). It
    lists all 32 top-scoring candidates with every field.
  - Keeping only detections with probability >= 0.3, and only class,
    probability, fan box, range and bearing, as compact JSON, takes 13 B
    (small@360) or 52 B (segdice) per frame on average on these recordings
    (median 2 B: most frames have none). `bench/stream/bandwidth.py`,
    `bandwidth.json`; one-worker 15 fps Pi 4 sweep outputs.
  - A raw frame is 53-124 KB. A vehicle would send the compact form, about
    three orders of magnitude less than the sonar video.
  - These recordings are river data with few true detections (M8 §6), so the
    compact size is a lower bound for scenes with many objects.

## 8. Limitations

- Emulated sensor: frames are replayed from recordings by a laptop. A real
  ARIS sends its frames over UDP in parts, with its own protocol. That
  protocol was researched but not implemented (future work).
- The laptop's clock was not NTP-synchronised. Latency is measured on the
  server clock alone, which avoids clock offsets but includes the ACK's
  return trip (under 1 ms on this network).
- The Pi 4's 30-minute run used a 100 Mbit/s link after the link failure.
  Its cable or port should be checked and the run repeated at 1 Gbit/s.
- One sustained run per board, small@360 only. Energy was not measured.
