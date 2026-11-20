# M5: on-device benchmark (Raspberry Pi 4) and NEON kernel

Measured 2026-11-10. Long-form version with explanations for readers outside
the project: `report/deployment.pdf`.

## 1. Setup

Raspberry Pi 4 Model B rev 1.5, 4 GB; BCM2711, 4x Cortex-A72 at 1.8 GHz
(ARMv8.0: NEON, no `sdot`/`udot`), 32 KB L1D per core, 1 MB shared L2; Debian 13,
kernel 6.18, gcc 14.2, governor `ondemand`, no fan. Runtime built natively with the
project makefile (`-std=c11 -O2 -Wall -Wextra -Werror -pedantic`); `-std=c11`
disables FP contraction, so the float box decoder rounds identically to x86.
Reference: laptop, AMD Ryzen 5 5600H, same flags.

Models: the four fold-0 CV networks `runs/small_cv_f0` (240x160, 34.8 M MACs),
`runs/small_360_cv_f0` (deployed, 78.4 M), `runs/small_wide_360_cv_f0` (203.5 M),
`runs/wide_cv_f0` (480x320, 450.2 M).

## 2. Method

- **Equivalence.** (a) golden test (`make test`, all 20,250 int32 head values of
  3 frames); (b) all 439 test-fold-0 frames through `sonar-detect` on the Pi, output
  scored on the laptop with `python -m sonar_edge.c_eval runs/<r> --pred dets.txt
  --tag pi4` (new option); (c) byte comparison of the Pi output with the laptop
  build's output on the same frames.
- **Latency.** `sonar-detect -n 50` over 10 frames spread over the fold (500
  calls of `se_detect`: network + decoder, no file I/O), 3 repeats; repeats
  agree within 1% (wide NEON: 3%), mean reported. One core.
- **Throughput.** 4 copies in parallel, frames / elapsed time.
- **Thermal.** `vcgencmd measure_temp` / `get_throttled` before and after each run.
- **Memory.** `VmHWM` from `/proc/<pid>/status` while running (the first attempt
  used `getrusage` of children from Python and reported the forked interpreter's
  11 MB; discarded). Program size from `size`.
- **Energy.** Not measured: the Pi 4 has no power sensor and no meter was
  available.

Scripts: `bench.py` (scalar, detections) and `bench2.py` (NEON, -O3, memory,
throughput) on the Pi under `~/sonar_edge_m5`; not part of the repo.

## 3. Results

### 3.1 Equivalence

Golden test passes on the Pi for all four models, scalar and NEON. Detection
output over the 439 frames is byte-identical to the laptop's (21,333 / 25,007 /
23,218 / 28,096 lines), hence identical mAP (58.94 / 63.81 / 68.12 / 68.60).
Pi-scored results in `runs/<r>/c_eval_test_pi4.json`. NEON output is byte-identical to scalar output on the full fold.

### 3.2 Latency, one core

| Model | MACs | Laptop scalar ms | Pi scalar ms (GMAC/s) | Pi NEON ms (GMAC/s) | NEON speed-up |
|---|---|---|---|---|---|
| small 240x160 | 34.8 M | 9.0 | 56.0 (0.62) | 10.2 (3.41) | 5.5x |
| **small 360x240 (deployed)** | 78.4 M | 20.1 | 125.9 (0.62) | **22.5 (3.48)** | 5.6x |
| small_wide 360x240 | 203.5 M | 27.6 | 300.1 (0.68) | 77.6 (2.62) | 3.9x |
| wide 480x320 | 450.2 M | 54.8 | 656.0 (0.69) | 140.9 (3.20) | 4.7x |

Deployed model, other builds: scalar `-O3 -mcpu=cortex-a72` 75.7 ms (gcc's
auto-vectoriser, 1.7x); NEON `-O3 -mcpu=cortex-a72` 21.5 ms (within spread of
`-O2`, so `-O2` stays). The M3 projection of 70-100 ms for the scalar Pi runtime
was too optimistic (Pi/laptop ratio 6.2 for 32-channel models, 11-12 for
64-channel ones). Both gcc 16.2 (laptop) and gcc 14.2 (Pi) auto-vectorise the
scalar dot product at -O2 (checked in the assembly), but the laptop gets 7-8
GMAC/s on 64-channel layers and the Pi 0.7; not analysed further.

### 3.3 Four cores

| Model | Kernel | Per copy ms | Frames/s | End temp |
|---|---|---|---|---|
| small 240x160 | scalar / NEON | 56.5 / 10.7-11.8 | 70.6 / 339 | 72.0 / 60.3 C |
| small 360x240 | scalar / NEON | 126.8 / 25.6-26.8 | 31.4 / 149 | 68.6 / 65.7 C |
| small_wide 360x240 | scalar / NEON | 303 / 162 | 13.2 / 24.4 | 75.9 / 71.5 C |
| wide 480x320 | scalar / NEON | 664 / 253-261 | 6.0 / 15.3 | 81.8* / 78.4** C |

No throttling in any run of the three smaller models (no fan; runs 1-3 min).
*The 4-core scalar wide run reached 81.8 C and set the sticky soft-temperature-limit
bit (`throttled=0x80000`); per-copy latency rose only 1% (664 vs 656 ms), but
sustained 4-core load needs a heat sink. **Run directly after; the flag is
cumulative, so throttling in this run cannot be ruled out. Scalar scales perfectly; NEON
32-channel models lose 5-15% per copy; 64-channel NEON loses ~50% per copy.

### 3.4 Memory and size (NEON build)

Deployed model: bss 854 KB (2 x 355 KB activations, 81 KB head map, 83 KB int16
weight copy), text 53 KB (42 KB weights), peak RSS 2.3 MB. Fallback 1.8 MB,
small_wide 2.6 MB, wide 3.4 MB.

### 3.5 Cost against MCNN

MCNN (sonar_cnn reimplementation, 305x305 crop): 2,797,706 parameters, 234.7 M
MACs per crop (counted with forward hooks); x 2.08 boxes per frame = ~490 M MACs
(corrected in M7: 2.08 includes Wall boxes, which MCNN does not classify; 1.51
objects per frame gives 355 M)
to classify the annotated objects of a frame, with no localisation. The deployed
detector: 78.4 M MACs for detection and classification of the whole frame, 42,463
parameters.

### 3.6 Four-pixel NEON kernel (added the same day)

`neon_block4`: 4 horizontally adjacent output pixels per pass, so each pair of
weight vectors loaded serves 16 `vmlal` instead of 4 (16 accumulators + 2 weight
+ 4 input registers, no spills in the generated code). Used when cin % 8 == 0
(all layers but the first); row tails and layer 0 use the 1-pixel path.
`-DSE_NEON_1PX` builds the 1-pixel kernel only. Golden test passes and the full
test-fold output is byte-identical to scalar for all four models.

| Model | NEON 1 px ms | NEON 4 px ms (GMAC/s) | vs scalar | 4 cores, per copy ms | 4 cores fps | End temp |
|---|---|---|---|---|---|---|
| small 240x160 | 10.2 (rebuild 10.4) | 9.0 (3.88) | 6.2x | 9.3-9.4 | 425 | 58.9 C |
| **small 360x240** | 22.5 (rebuild 22.1) | **20.3 (3.85)** | 6.2x | 21.0-21.3 | **187** | 64.2 C |
| small_wide 360x240 | 77.6 (rebuild 77.1) | 46.5 (4.37) | 6.4x | 59.1-59.5 | 67.1 | 72.0 C |
| wide 480x320 | 140.9 (rebuild 141.9) | 91.8 (4.91) | 7.1x | 100.5-101.5 | 39.4 | 78.8 C |

The 64-channel models gain most (1.7x / 1.5x) and now scale to 4 cores at
3.1-3.6x (was 1.9-2.2x); 32-channel models gain 11-13%. This is the prediction
of the weight-traffic explanation in section 4, so it is now supported (still no
cache-miss counts). The thermal flag was already set from the wide scalar run, so
whether 4-pixel 4-core runs throttled cannot be told from it; all stayed < 80 C.

### 3.7 Raspberry Pi 2 (32-bit ARMv7, added 2026-11-20)

**Board.** Raspberry Pi 2 Model B rev 1.1: BCM2836, 4x Cortex-A7 at 900 MHz
(32-bit ARMv7, NEON with VFPv4, 16 q registers), 1 GB; Raspbian 13 (armhf),
kernel 6.18 rpi-v7, gcc 14.2, governor `ondemand`, no heat sink.

**Port.** The NEON kernel used AArch64-only intrinsics (`vmlal_laneq_s16`,
`vmlal_high_*`). It now compiles on AArch32 as well through three macros
(`SE_MLAL_LO/HI`, `SE_MLAL_N_HI` in `runtime/sonar_edge.c`). On AArch32 these
take the lane from the matching 64-bit half of the activation vector. The
products and sums are the same, so the result is still exact by
construction. AArch64 code generation is unchanged: after the port the Pi 4
build passes the golden test, gives byte-identical full-fold output for all
four models, and runs at the same speed (8.9 / 20.2 / 46.2 / 91.6 ms).
Raspbian's gcc targets ARMv6 without NEON by default, so NEON builds need
`-mcpu=cortex-a7 -mfpu=neon-vfpv4` (`runtime/Makefile` comment).

**Method.** Same models, frames and measurements as §3.2–3.6. Four builds per
model:
- `scalar_v6`: Raspbian default target;
- `scalar_a7`: Cortex-A7 flags with `-DSE_NO_NEON`;
- `neon1`: 1-pixel kernel;
- `neon4`: 4-pixel kernel.

Latency repeats were sized to about 25 s each, three repeats; NEON repeats
agree within 1%. Four-copy throughput and peak RSS were measured on the
faster NEON build and on `scalar_a7`. Script, raw JSON and logs are in
`bench/pi2/`.

An earlier attempt failed: large packets to the board were lost (90% at
1400 B; ssh and copies stalled), and `get_throttled` read 0x50000
(under-voltage). With a stronger power supply the loss went to 0% and the
flags stayed 0x0 for the whole run.

**Equivalence.** The golden test passes for all 16 builds. `neon4` output on
all 439 test-fold-0 frames is byte-identical to the Pi 4 scalar output for
all four models.

| Model | scalar v6 ms | scalar A7 ms | NEON 1 px ms | NEON 4 px ms | NEON vs scalar A7 | 4 copies fps (per copy ms) | Peak RSS | Pi 4 NEON 4 px ms | Pi 4 / Pi 2 |
|---|---|---|---|---|---|---|---|---|---|
| small 240x160 | 295.1 | 288.8 | **54.8** | 56.3 | 5.3x | 69.8 (56-57) | 1.8 MB | 9.0 | 6.1x |
| **small 360x240 (deployed)** | 663.8 | 650.0 | **124.5** | 126.8 | 5.2x | **30.7** (126-130) | 2.3 MB | 20.3 | 6.1x |
| small_wide 360x240 | 1565.8 | 1555.0 | 392.2 | **315.6** | 4.9x | 12.0 (330-334) | 2.5 MB | 46.5 | 6.8x |
| wide 480x320 | 3418.8 | 3384.0 | 798.7 | **649.6** | 5.2x | 5.8 (680-689) | 3.3 MB | 91.8 | 7.1x |

Scalar on four cores: 13.6 / 6.0 / 2.5 / 1.15 fps. Temperatures stayed at
39–54 C, and `throttled` was 0x0 from start to end.

**Discussion.**
- **NEON speed-up.** About 5x over the Cortex-A7 scalar build, against 6.2–7.1x
  on the Pi 4. Telling gcc about the A7 (`scalar_a7`) gains only 1–2% over the
  ARMv6 default.
- **4-pixel kernel.** The block needs 16 accumulators + 2 weight + 4 input
  registers, and AArch32 has 16 q registers, so it spills. On the 32-channel
  models the spills cost slightly more than the saved weight loads (-2–3%).
  On the 64-channel models the saved weight traffic dominates (+24% / +23%),
  the same pattern as on the Pi 4 (§4). Best build: `neon1` for small,
  `neon4` for small_wide and wide.
- **Four cores.** Scaling is 94–96% of four times the one-core rate for every model. The Pi 4 lost 5–15% on
  its 32-channel NEON runs (§3.6). The Pi 2's slower cores ask less of the
  shared L2 and memory per second.
- **Deployability.** The deployed model runs at 8 fps on one core and 30.7 fps
  on four (single-frame latency stays 125 ms; the four-core figure assumes
  four frames in flight). small_wide@360 reaches 12 fps on four cores. Whether
  that keeps up with the sonar depends on its frame rate at the range used,
  which the dataset does not give.
- **Not measured.** Energy (no meter); thermal behaviour beyond a few minutes
  per run.

## 4. NEON kernel (runtime/sonar_edge.c, `conv3x3_neon`)

Vectorised over output channels: per output pixel, block of 16 output channels
(four int32x4 accumulators), tap t and input channel i, the activation x[t,i]
(u8 widened to s16, 8 at a time) multiplies 16 int16 weights with
`vmlal_laneq_s16` / `vmlal_high_laneq_s16` (4 MACs per instruction). Weights are
transposed once at first call from [co][t][ci] int8 to [t][ci][co] int16 into a
static buffer of `SE_W_ELEMS` (new define emitted by `export_c`). |w x| <=
127 * 255 < 2^15, so products are exact; int32 addition is associative and
|acc| < 2^28, so the result is bit-identical to the scalar kernel by
construction. Used on AArch64 for layers with cout % 16 == 0 (all layers here);
`-DSE_NO_NEON` forces scalar; other targets and headers without `SE_W_ELEMS`
compile the scalar path unchanged. No dot-product instructions, so it runs on
the Pi 4.

With the 1-pixel kernel, 64-channel models gained less (small_wide 3.9x, wide
4.7x vs 5.6x) and scaled poorly to 4 cores. Cause: the int16 weight table of a 64x64 layer
is 74 KB > 32 KB L1D, so it streams from L2 per output pixel, and four copies'
working sets exceed the 1 MB shared L2. This does not explain small_wide vs wide
(identical 64x64 layers), and cache misses were not measured. The lever,
processing 4 output pixels per weight pass, was then implemented (section 3.6)
and gave exactly the predicted pattern.

## 5. Conclusions

- Deployed model on a Pi 4: 20.3 ms / frame, 49 fps on one core, 187 fps on four
  (4-pixel kernel), 2.3 MB RSS; output bit-exact, so the CV accuracy holds on device. The
  240x160 fallback is not needed.
- small_wide@360 (best accuracy, 56.9 +- 2.7 pooled over 3 seeds) at 46.5 ms / 21.5 fps single core
  is now a deployment candidate.
- Raspberry Pi 2 (§3.7): same output byte for byte; deployed model 124.5 ms
  (8 fps) on one core, 30.7 fps on four, 6.1x slower than the Pi 4.
- Open: energy (external USB-C meter or INA219), sustained thermal test,
  Pi 5 `udot` kernel.
