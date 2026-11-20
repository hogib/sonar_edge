/* ARIS (.aris) reader. See aris.h. Header offsets from SoundMetrics
 * aris-file-sdk type-definitions/C (ArisFileHeaderOffsets, ArisFrameHeaderOffsets). */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64   /* recordings exceed 2 GB; needed on 32-bit ARM */
#include "aris.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define ARIS_HDR 1024
#define ARIS_MAGIC 0x05464444u

/* frame-header offsets */
enum {
    OFF_FRAME_INDEX = 0, OFF_VERSION = 12, OFF_PING_MODE = 436, OFF_SAMPLE_PERIOD = 452,
    OFF_FRAME_RATE = 460, OFF_SOUND_SPEED = 464, OFF_SAMPLES_PER_BEAM = 468,
    OFF_SAMPLE_START_DELAY = 476, OFF_LARGE_LENS = 480, OFF_SYSTEM_TYPE = 484, OFF_REORDERED = 516
};

/* Beam-centre angles from the SDK's beam-width-metrics (MIT, Copyright 2010-2018
 * Sound Metrics; see beamwidths/LICENSE-SoundMetrics). In the SDK, beam 0 is the
 * right-most beam and has the most negative angle. */
#define DEFINE_BEAMWIDTH3(beam, centre, left, right) (float)(centre),
static const float bw_3000_128[] = {
#include "beamwidths/BeamWidths_ARIS3000_128.h"
};
static const float bw_3000_64[] = {
#include "beamwidths/BeamWidths_ARIS3000_64.h"
};
static const float bw_1800_96[] = {
#include "beamwidths/BeamWidths_ARIS1800_96.h"
};
static const float bw_1800_1200_48[] = {
#include "beamwidths/BeamWidths_ARIS1800_1200_48.h"
};
static const float bw_tele_96[] = {
#include "beamwidths/BeamWidths_ARIS_Telephoto_96.h"
};
static const float bw_tele_48[] = {
#include "beamwidths/BeamWidths_ARIS_Telephoto_48.h"
};
#undef DEFINE_BEAMWIDTH3

static uint32_t rd_u32(const uint8_t *b, int o)
{
    return (uint32_t)b[o] | (uint32_t)b[o + 1] << 8 | (uint32_t)b[o + 2] << 16 | (uint32_t)b[o + 3] << 24;
}

static float rd_f32(const uint8_t *b, int o)
{
    const uint32_t u = rd_u32(b, o);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static int beams_from_ping_mode(uint32_t pm)   /* SDK common-code/FrameFuncs.c */
{
    if (pm == 1 || pm == 2) return 48;
    if (pm >= 3 && pm <= 5) return 96;
    if (pm >= 6 && pm <= 8) return 64;
    if (pm >= 9 && pm <= 12) return 128;
    return 0;
}

static int parse_frame(const uint8_t *b, se_aris_frame_t *h)
{
    if (rd_u32(b, OFF_VERSION) != ARIS_MAGIC) return -1;
    h->frame_index = rd_u32(b, OFF_FRAME_INDEX);
    h->ping_mode = rd_u32(b, OFF_PING_MODE);
    h->n_beams = beams_from_ping_mode(h->ping_mode);
    h->sample_period_us = rd_u32(b, OFF_SAMPLE_PERIOD);
    h->frame_rate = rd_f32(b, OFF_FRAME_RATE);
    h->sound_speed = rd_f32(b, OFF_SOUND_SPEED);
    const uint32_t spb = rd_u32(b, OFF_SAMPLES_PER_BEAM);
    h->sample_start_delay_us = rd_u32(b, OFF_SAMPLE_START_DELAY);
    h->large_lens = rd_u32(b, OFF_LARGE_LENS);
    h->system_type = rd_u32(b, OFF_SYSTEM_TYPE);
    if (!h->n_beams || spb < 2 || spb > 8192 || h->sample_period_us == 0 ||
        !(h->sound_speed > 1000.0f && h->sound_speed < 2000.0f) || rd_u32(b, OFF_REORDERED) == 0)
        return -1;   /* unknown ping mode, implausible header, or samples still in device order */
    h->n_samples = (int)spb;
    h->window_start_m = h->sample_start_delay_us * 1e-6 * h->sound_speed / 2;
    h->sample_len_m = h->sample_period_us * 1e-6 * h->sound_speed / 2;
    return 0;
}

int se_aris_open(se_aris_file_t *a, const char *path)
{
    uint8_t b[ARIS_HDR];
    memset(a, 0, sizeof *a);
    a->f = fopen(path, "rb");
    if (!a->f) { perror(path); return -1; }
    if (fread(b, 1, ARIS_HDR, a->f) != ARIS_HDR || rd_u32(b, 0) != ARIS_MAGIC) {
        fprintf(stderr, "%s: not an ARIS file (version magic)\n", path);
        goto fail;
    }
    if (fread(b, 1, ARIS_HDR, a->f) != ARIS_HDR || parse_frame(b, &a->first)) {
        fprintf(stderr, "%s: missing or unsupported first frame header\n", path);
        goto fail;
    }
    a->frame_bytes = (size_t)a->first.n_beams * (size_t)a->first.n_samples;
    if (fseeko(a->f, 0, SEEK_END)) goto fail;
    const off_t size = ftello(a->f);
    a->n_frames = (long long)((size - ARIS_HDR) / (off_t)(ARIS_HDR + a->frame_bytes));
    if (a->n_frames < 1) { fprintf(stderr, "%s: no complete frame\n", path); goto fail; }
    a->raw = malloc(a->frame_bytes);
    if (!a->raw) goto fail;
    return 0;
fail:
    fclose(a->f);
    a->f = NULL;
    return -1;
}

int se_aris_read(se_aris_file_t *a, long long k, se_aris_frame_t *hdr, uint8_t *polar)
{
    uint8_t b[ARIS_HDR];
    se_aris_frame_t h;
    if (k < 0 || k >= a->n_frames) return -1;
    const off_t off = (off_t)ARIS_HDR + (off_t)k * (off_t)(ARIS_HDR + a->frame_bytes);
    if (fseeko(a->f, off, SEEK_SET) || fread(b, 1, ARIS_HDR, a->f) != ARIS_HDR || parse_frame(b, &h))
        return -1;
    if (h.n_beams != a->first.n_beams || h.n_samples != a->first.n_samples) return -1;  /* frame size is fixed per file */
    if (fread(a->raw, 1, a->frame_bytes, a->f) != a->frame_bytes) return -1;
    const int nb = h.n_beams;
    for (int s = 0; s < h.n_samples; s++) {    /* flip beams: stored beam 0 is right-most */
        const uint8_t *in = a->raw + (size_t)s * nb;
        uint8_t *out = polar + (size_t)s * nb;
        for (int j = 0; j < nb; j++) out[j] = in[nb - 1 - j];
    }
    if (hdr) *hdr = h;
    return 0;
}

void se_aris_close(se_aris_file_t *a)
{
    if (a->f) fclose(a->f);
    free(a->raw);
    memset(a, 0, sizeof *a);
}

int se_aris_beam_deg(const se_aris_frame_t *h, float *deg)
{
    const float *t = NULL;
    const int n = h->n_beams;
    if (h->large_lens) t = n == 96 ? bw_tele_96 : n == 48 ? bw_tele_48 : NULL;
    else if (h->system_type == 1) t = n == 128 ? bw_3000_128 : n == 64 ? bw_3000_64 : NULL;
    else if (h->system_type == 0) t = n == 96 ? bw_1800_96 : n == 48 ? bw_1800_1200_48 : NULL;
    else if (h->system_type == 2) t = n == 48 ? bw_1800_1200_48 : NULL;
    if (!t) return -1;
    /* left-most first: our beam j is SDK beam n-1-j; SDK angles are negative to the right */
    for (int j = 0; j < n; j++) deg[j] = -t[n - 1 - j];
    return 0;
}

void se_aris_geom(const se_aris_frame_t *h, const float *beam_deg, int fit, se_fan_geom_t *g)
{
    *g = se_fan_watertank;
    if (beam_deg) {   /* field of view spanned by the beam centres plus half a spacing at each side */
        const int n = h->n_beams;
        g->fov_deg = (beam_deg[n - 1] - beam_deg[0]) * (float)n / (float)(n - 1);
    }
    /* the front end puts sample i's centre at r_min + (i + 0.5) * (r_max - r_min) / n_samples;
     * ARIS sample i lies at window_start + i * sample_len */
    const double r0 = h->window_start_m - 0.5 * h->sample_len_m;
    const double r1 = h->window_start_m + (h->n_samples - 0.5) * h->sample_len_m;
    if (fit == SE_ARIS_FIT_WINDOW) {
        g->m_per_px = 0.0f;   /* radial scale is affine, not linear: report ranges in pixels */
    } else {
        const double mpp = r1 / se_fan_watertank.r_max_px;
        g->m_per_px = (float)mpp;
        g->r_max_px = se_fan_watertank.r_max_px;
        g->r_min_px = (float)(r0 / mpp);
    }
}

int se_aris_same_geom(const se_aris_frame_t *a, const se_aris_frame_t *b)
{
    return a->n_beams == b->n_beams && a->n_samples == b->n_samples && a->system_type == b->system_type &&
           a->large_lens == b->large_lens && a->sample_period_us == b->sample_period_us &&
           a->sample_start_delay_us == b->sample_start_delay_us &&
           fabsf(a->sound_speed - b->sound_speed) <= 1e-3f * a->sound_speed;
}
