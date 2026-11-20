/* Reader for Sound Metrics ARIS recordings (.aris), C11, no dependencies.
 *
 * Format: SoundMetrics aris-file-sdk (MIT), docs/understanding-aris-data.md and
 * type-definitions/C: a 1024-byte file header, then per frame a 1024-byte frame
 * header and uint8 samples [SamplesPerBeam][BeamCount], beam 0 right-most.
 * Frames come out in the front end's polar layout (frontend.h): rows = range
 * samples near to far, columns = beams, beam 0 LEFT-most.
 */
#pragma once
#include <stdint.h>
#include <stdio.h>

#include "frontend.h"

typedef struct {
    uint32_t frame_index;
    uint32_t system_type;       /* 0 = ARIS 1800, 1 = ARIS 3000, 2 = ARIS 1200 */
    uint32_t ping_mode, large_lens;
    int n_beams, n_samples;
    uint32_t sample_period_us, sample_start_delay_us;
    float sound_speed, frame_rate;
    double window_start_m;      /* range of sample 0 */
    double sample_len_m;        /* range step per sample */
} se_aris_frame_t;

typedef struct {
    FILE *f;
    long long n_frames;
    size_t frame_bytes;         /* n_beams * n_samples */
    se_aris_frame_t first;      /* header of frame 0 */
    uint8_t *raw;               /* one frame as stored */
} se_aris_file_t;

/* Open and validate (version magic, ping mode, reordered samples, size).
 * Returns 0, or -1 with a message on stderr. */
int se_aris_open(se_aris_file_t *a, const char *path);
/* Read frame k (0-based): header into *hdr (may be NULL), samples into polar
 * (n_samples x n_beams, beam 0 left-most). Returns 0 or -1. */
int se_aris_read(se_aris_file_t *a, long long k, se_aris_frame_t *hdr, uint8_t *polar);
void se_aris_close(se_aris_file_t *a);

/* Beam-centre bearings in degrees (positive right), left-most beam first, from
 * the SDK's beam-width tables for this system, beam count and lens. Returns 0,
 * or -1 if no table matches. */
int se_aris_beam_deg(const se_aris_frame_t *h, float *deg);

/* Fan geometry to render this frame into, based on se_fan_watertank (same
 * image size, apex and field of view as the training images):
 *   SE_ARIS_FIT_METRIC  true shape: far edge of the window on the training
 *                       fan's far radius, near edge where the true near/far
 *                       ratio puts it; m_per_px set, so ranges are metres.
 *   SE_ARIS_FIT_WINDOW  the window stretched onto the training fan's near and
 *                       far radius (radial scale changes); ranges in pixels. */
enum { SE_ARIS_FIT_METRIC = 0, SE_ARIS_FIT_WINDOW = 1 };
void se_aris_geom(const se_aris_frame_t *h, const float *beam_deg, int fit, se_fan_geom_t *g);
/* 1 if two frame headers give the same rendering geometry (sound speed within 0.1%). */
int se_aris_same_geom(const se_aris_frame_t *a, const se_aris_frame_t *b);
