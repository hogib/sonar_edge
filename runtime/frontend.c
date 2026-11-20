/* sonar_edge front end. See frontend.h.
 *
 * Fan mode reproduces the training pipeline (sonar_edge.dataset._prep followed
 * by c_eval.export_pgm). For an integer factor s (480x320 -> 240x160) that is
 * the mean of each s x s block, divided by 255 and multiplied by 255 in
 * float32, rounded half to even; all of it depends only on the block sum, so
 * it is a table built at init. Same size: a copy. Otherwise it is Pillow's
 * Image.resize(..., Image.BOX) for 8-bit images
 * (Pillow 13, src/libImaging/Resample.c): per output column/row the box
 * filter's weights are computed in double exactly as Pillow does, normalised,
 * and converted to int32 with 22 fraction bits; a horizontal pass (rounded and
 * clipped to uint8) is followed by a vertical pass. The per-frame work is
 * integer only, and the result is bit-identical to the Python pipeline.
 *
 * Polar mode renders raw beam data directly at the network resolution: every
 * network pixel is sampled at its centre (or at 2 x 2 points of its footprint
 * with -DSE_FE_PTS=4, four times the table), each point maps
 * through the fan geometry to (range, bearing), hence to a fractional (sample,
 * beam) position, and is interpolated bilinearly with 8-bit weights. Points
 * outside the fan contribute 0, as the black corners of a fan image do. The
 * table (one source index and two 8-bit fractions per point) is built at init.
 */
#include "frontend.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "sonar_edge.h"

/* weights.h for SE_IN_H / SE_IN_W only, so the tables below are sized for the
 * compiled-in network; its arrays are unused here (and dropped by the compiler). */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "weights.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* Measured from the 1,868 dataset frames (pixels non-zero in > 50 % of the
 * frames): straight left/right edges meet at (159.9, 602.7), 15.04 and 15.03
 * degrees off the vertical (ARIS Explorer 3000: 30 degree field of view);
 * near and far arcs at 138.6 and 578.0 pixels from the apex (sd 0.4 and
 * 0.3 px along the arcs). The metric scale is not documented for the dataset. */
const se_fan_geom_t se_fan_watertank = {320, 480, 160.0f, 602.7f, 30.0f, 138.6f, 578.0f, 0.0f};

#define SE_FE_PREC 22                 /* Pillow PRECISION_BITS for 8-bit images */
#define SE_FE_MAX_SRC 1024            /* largest fan-image height (rows kept after the horizontal pass) */
#define SE_FE_MAX_K 16                /* box taps per output pixel: up to 7x downscale */
#ifndef SE_FE_PTS
#define SE_FE_PTS 1                   /* polar mode: sample points per network pixel, 1 (centre) or 4 (2 x 2); same mAP, 1/4 the table */
#endif

enum { FE_NONE, FE_FAN, FE_POLAR };

static int fe_mode = FE_NONE;
static int fe_src_w, fe_src_h, fe_out_w, fe_out_h, fe_src_bytes;
static float fe_sx, fe_sy;            /* fan pixels per network pixel */

/* fan mode: separable box filter */
static int hb[SE_IN_W][2], vb[SE_IN_H][2];               /* first tap, tap count */
static int32_t hk[SE_IN_W][SE_FE_MAX_K], vk[SE_IN_H][SE_FE_MAX_K];
static uint8_t fe_tmp[SE_FE_MAX_SRC * SE_IN_W];          /* after the horizontal pass */
static int fe_block;                                     /* integer factor (block mean), 0 = box filter */
static uint8_t fe_mean_lut[SE_FE_MAX_K * SE_FE_MAX_K * 255 + 1];  /* block sum -> output byte */

/* polar mode: SE_FE_PTS points per pixel; index of the (sample, beam) corner, -1 if outside */
static int32_t pidx[SE_IN_H * SE_IN_W][SE_FE_PTS];
static uint8_t pfs[SE_IN_H * SE_IN_W][SE_FE_PTS], pfb[SE_IN_H * SE_IN_W][SE_FE_PTS];
static int fe_beams;

static double box_filter(double x) { return (x > -0.5 && x <= 0.5) ? 1.0 : 0.0; }

/* Pillow precompute_coeffs + normalize_coeffs_8bpc for the box filter (support 0.5). */
static int box_coeffs(int in_size, int out_size, int (*bounds)[2], int32_t (*k)[SE_FE_MAX_K])
{
    const double scale = (double)in_size / out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 0.5 * filterscale;
    const int ksize = (int)ceil(support) * 2 + 1;
    if (ksize > SE_FE_MAX_K) return -1;
    for (int xx = 0; xx < out_size; xx++) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        double w[SE_FE_MAX_K], ww = 0.0;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        if (xmax > ksize) return -1;
        for (int x = 0; x < xmax; x++) {
            w[x] = box_filter((x + xmin - center + 0.5) * ss);
            ww += w[x];
        }
        for (int x = 0; x < SE_FE_MAX_K; x++) {
            double v = 0.0;
            if (x < xmax) v = ww != 0.0 ? w[x] / ww : w[x];
            k[xx][x] = v < 0 ? (int32_t)(-0.5 + v * (1 << SE_FE_PREC)) : (int32_t)(0.5 + v * (1 << SE_FE_PREC));
        }
        bounds[xx][0] = xmin;
        bounds[xx][1] = xmax;
    }
    return 0;
}

static uint8_t clip8(int32_t in)
{
    if (in >= (1 << SE_FE_PREC << 8)) return 255;
    if (in <= 0) return 0;
    return (uint8_t)(in >> SE_FE_PREC);
}

static int out_dims(const se_fan_geom_t *g)
{
    fe_out_h = se_input_height();
    fe_out_w = se_input_width();
    if (!g || g->width <= 0 || g->height <= 0 || fe_out_w != SE_IN_W || fe_out_h != SE_IN_H) return -1;
    fe_src_w = g->width;
    fe_src_h = g->height;
    fe_sx = (float)g->width / (float)fe_out_w;
    fe_sy = (float)g->height / (float)fe_out_h;
    return 0;
}

int se_fe_init_fan(const se_fan_geom_t *g)
{
    fe_mode = FE_NONE;
    if (out_dims(g) || g->width > SE_FE_MAX_SRC || g->height > SE_FE_MAX_SRC) return -1;
    fe_block = 0;
    if (g->width % fe_out_w == 0 && g->height % fe_out_h == 0 && g->width / fe_out_w == g->height / fe_out_h &&
        g->width / fe_out_w > 1) {
        /* numpy: block.mean() (float64, exact) -> float32 / 255 -> * 255 -> np.round (half to even) */
        const int s = g->width / fe_out_w;
        if (s > SE_FE_MAX_K) return -1;
        for (int sum = 0; sum <= s * s * 255; sum++) {
            const float m = (float)((double)sum / (s * s));
            const float t = m / 255.0f;
            const float o = t * 255.0f;
            fe_mean_lut[sum] = (uint8_t)rintf(o);   /* default rounding mode: nearest, ties to even */
        }
        fe_block = s;
    } else if (box_coeffs(g->width, fe_out_w, hb, hk) || box_coeffs(g->height, fe_out_h, vb, vk)) {
        return -1;
    }
    fe_src_bytes = g->width * g->height;
    fe_mode = FE_FAN;
    return 0;
}

int se_fe_init_polar(const se_fan_geom_t *g, int n_beams, int n_samples, const float *beam_deg)
{
    fe_mode = FE_NONE;
    if (out_dims(g) || n_beams < 2 || n_samples < 2 || !(g->r_max_px > g->r_min_px) || !(g->fov_deg > 0))
        return -1;
    const double dr = ((double)g->r_max_px - g->r_min_px) / n_samples;
    const double half = 0.5 * g->fov_deg, db = (double)g->fov_deg / n_beams;
    for (int y = 0; y < fe_out_h; y++) {
        for (int x = 0; x < fe_out_w; x++) {
            const int p = y * fe_out_w + x;
            for (int q = 0; q < SE_FE_PTS; q++) {
                const double ou = SE_FE_PTS == 1 ? 0.5 : 0.25 + 0.5 * (q & 1), ov = SE_FE_PTS == 1 ? 0.5 : 0.25 + 0.5 * (q >> 1);
                const double u = (x + ou) * fe_sx, v = (y + ov) * fe_sy;
                float rf, bf;
                se_fan_to_polar(g, (float)u, (float)v, &rf, &bf);
                double r = g->m_per_px > 0 ? rf / g->m_per_px : rf, bdeg = bf;
                double s = (r - g->r_min_px) / dr - 0.5, b;
                int inside = r >= g->r_min_px && r <= g->r_max_px;
                if (beam_deg) {
                    const double lo = beam_deg[0] - 0.5 * (beam_deg[1] - beam_deg[0]);
                    const double hi = beam_deg[n_beams - 1] + 0.5 * (beam_deg[n_beams - 1] - beam_deg[n_beams - 2]);
                    inside = inside && bdeg >= lo && bdeg <= hi;
                    int j = 0;
                    while (j < n_beams - 2 && bdeg >= beam_deg[j + 1]) j++;
                    b = j + (bdeg - beam_deg[j]) / (beam_deg[j + 1] - beam_deg[j]);
                } else {
                    inside = inside && fabs(bdeg) <= half;
                    b = (bdeg + half) / db - 0.5;
                }
                if (!inside) { pidx[p][q] = -1; pfs[p][q] = pfb[p][q] = 0; continue; }
                /* clamp to the sample grid (edge hold), keep (s0 + 1, b0 + 1) valid */
                if (s < 0) s = 0;
                if (s > n_samples - 1) s = n_samples - 1;
                if (b < 0) b = 0;
                if (b > n_beams - 1) b = n_beams - 1;
                int s0 = (int)s, b0 = (int)b;
                if (s0 > n_samples - 2) s0 = n_samples - 2;
                if (b0 > n_beams - 2) b0 = n_beams - 2;
                int fs = (int)lround((s - s0) * 256.0), fb = (int)lround((b - b0) * 256.0);
                if (fs > 255) { /* only at the far edge: all weight on s0 + 1 */
                    if (s0 + 1 < n_samples - 1) { s0++; fs = 0; } else fs = 255;
                }
                if (fb > 255) {
                    if (b0 + 1 < n_beams - 1) { b0++; fb = 0; } else fb = 255;
                }
                pidx[p][q] = s0 * n_beams + b0;
                pfs[p][q] = (uint8_t)fs;
                pfb[p][q] = (uint8_t)fb;
            }
        }
    }
    fe_beams = n_beams;
    fe_src_bytes = n_beams * n_samples;
    fe_mode = FE_POLAR;
    return 0;
}

int se_fe_src_bytes(void) { return fe_mode == FE_NONE ? 0 : fe_src_bytes; }

static void run_block(const uint8_t *src, uint8_t *dst)
{
    const int s = fe_block, sw = fe_src_w;
    for (int yy = 0; yy < fe_out_h; yy++)
        for (int xx = 0; xx < fe_out_w; xx++) {
            int sum = 0;
            for (int y = 0; y < s; y++)
                for (int x = 0; x < s; x++) sum += src[(size_t)(yy * s + y) * sw + xx * s + x];
            dst[(size_t)yy * fe_out_w + xx] = fe_mean_lut[sum];
        }
}

static void run_fan(const uint8_t *src, uint8_t *dst)
{
    const int ow = fe_out_w, oh = fe_out_h, sw = fe_src_w;
    /* horizontal pass over the rows the vertical pass reads */
    const int y_first = vb[0][0], y_last = vb[oh - 1][0] + vb[oh - 1][1];
    for (int y = y_first; y < y_last; y++) {
        const uint8_t *row = src + (size_t)y * sw;
        uint8_t *t = fe_tmp + (size_t)y * ow;
        for (int xx = 0; xx < ow; xx++) {
            const int x0 = hb[xx][0], n = hb[xx][1];
            const int32_t *k = hk[xx];
            int32_t ss = 1 << (SE_FE_PREC - 1);
            for (int x = 0; x < n; x++) ss += row[x0 + x] * k[x];
            t[xx] = clip8(ss);
        }
    }
    for (int yy = 0; yy < oh; yy++) {
        const int y0 = vb[yy][0], n = vb[yy][1];
        const int32_t *k = vk[yy];
        uint8_t *o = dst + (size_t)yy * ow;
        for (int xx = 0; xx < ow; xx++) {
            int32_t ss = 1 << (SE_FE_PREC - 1);
            for (int y = 0; y < n; y++) ss += fe_tmp[(size_t)(y0 + y) * ow + xx] * k[y];
            o[xx] = clip8(ss);
        }
    }
}

static void run_polar(const uint8_t *src, uint8_t *dst)
{
    const int n = fe_out_w * fe_out_h, nb = fe_beams;
    for (int p = 0; p < n; p++) {
        uint32_t acc = 0;
        for (int q = 0; q < SE_FE_PTS; q++) {
            const int32_t i = pidx[p][q];
            if (i < 0) continue;
            const uint32_t fs = pfs[p][q], fb = pfb[p][q];
            const uint8_t *a = src + i, *c = a + nb;
            const uint32_t top = (uint32_t)a[0] * (256 - fb) + (uint32_t)a[1] * fb;
            const uint32_t bot = (uint32_t)c[0] * (256 - fb) + (uint32_t)c[1] * fb;
            acc += top * (256 - fs) + bot * fs;   /* <= 255 * 2^16 */
        }
        /* mean of the points, rounded: 2^16 per point from the bilinear weights */
        dst[p] = SE_FE_PTS == 4 ? (uint8_t)((acc + (1u << 17)) >> 18) : (uint8_t)((acc + (1u << 15)) >> 16);
    }
}

void se_fe_run(const uint8_t *src, uint8_t *dst)
{
    if (fe_mode == FE_FAN && fe_block) run_block(src, dst);
    else if (fe_mode == FE_FAN) run_fan(src, dst);
    else if (fe_mode == FE_POLAR) run_polar(src, dst);
}

void se_fe_to_fan(float x, float y, float *u, float *v)
{
    *u = x * fe_sx;
    *v = y * fe_sy;
}

void se_fan_to_polar(const se_fan_geom_t *g, float u, float v, float *range, float *bearing_deg)
{
    /* "ahead" points from the apex towards the image: up if the apex is below the middle */
    const double ahead = g->apex_v > 0.5 * g->height ? -1.0 : 1.0;
    const double dx = (double)u - g->apex_u, dy = ahead * ((double)v - g->apex_v);
    const double r = sqrt(dx * dx + dy * dy);
    const double b = atan2(dx, dy) * (180.0 / 3.14159265358979323846);  /* positive = image right */
    *range = (float)(g->m_per_px > 0 ? r * g->m_per_px : r);
    *bearing_deg = (float)b;
}
