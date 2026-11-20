/* Front-end test (make test-frontend).
 *  1. Fan mode: the network input made from synthetic 320x480 frames hashes to the
 *     value the Python pipeline gives (tests/frontend_golden.h, from
 *     `python -m sonar_edge.polar golden`), for the compiled-in input size.
 *  2. Geometry: network -> fan -> (range, bearing) for points of known position.
 *  3. Polar mode: a constant polar frame renders to that constant inside the fan
 *     and 0 well outside it. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../frontend.h"
#include "../sonar_edge.h"
#include "frontend_golden.h"

#define FW 320
#define FH 480

static uint32_t fnv1a(const uint8_t *p, size_t n)
{
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x01000193u;
    return h;
}

static void lcg(uint8_t *f, uint32_t x)
{
    for (int i = 0; i < FW * FH; i++) {
        x = x * 1664525u + 1013904223u;
        f[i] = (uint8_t)(x >> 24);
    }
}

static void make_frame(int kind, uint8_t *f)
{
    switch (kind) {
    case 0: lcg(f, 12345u); break;
    case 1: for (int i = 0; i < FW * FH; i++) f[i] = (uint8_t)((i % FW) % 256); break;
    case 2: for (int i = 0; i < FW * FH; i++) f[i] = 255; break;
    case 3: for (int i = 0; i < FW * FH; i++) f[i] = (i % FW) % 2 ? 255 : 0; break;
    default: lcg(f, 777u); for (int i = 0; i < FW * FH; i++) f[i] &= 0xFE; break;
    }
}

static int near(float a, float b, float tol) { return fabsf(a - b) <= tol; }

int main(void)
{
    static uint8_t src[FW * FH], dst[512 * 512];
    const int w = se_input_width(), h = se_input_height();
    int fails = 0, checked = 0;

    if (se_fe_init_fan(&se_fan_watertank)) { printf("se_fe_init_fan failed\n"); return 1; }
    if (se_fe_src_bytes() != FW * FH) { printf("src bytes %d\n", se_fe_src_bytes()); fails++; }
    for (size_t i = 0; i < sizeof fe_golden / sizeof fe_golden[0]; i++) {
        if (fe_golden[i].w != w || fe_golden[i].h != h) continue;
        make_frame(fe_golden[i].kind, src);
        se_fe_run(src, dst);
        const uint32_t got = fnv1a(dst, (size_t)w * h);
        checked++;
        if (got != fe_golden[i].hash) {
            printf("fan kind %d %dx%d: hash %08x, expected %08x\n", fe_golden[i].kind, w, h, got, fe_golden[i].hash);
            fails++;
        }
    }
    if (!checked) { printf("no golden entry for input %dx%d\n", w, h); fails++; }

    /* geometry: the far-right corner of the network input maps to the fan image's */
    float u, v, r, b;
    se_fe_to_fan((float)w, (float)h, &u, &v);
    if (!near(u, FW, 1e-3f) || !near(v, FH, 1e-3f)) { printf("to_fan (%g, %g)\n", u, v); fails++; }
    const se_fan_geom_t *g = &se_fan_watertank;
    const float deg[] = {0.0f, 10.0f, -14.0f};
    for (int k = 0; k < 3; k++) {
        const double t = deg[k] * 3.14159265358979323846 / 180.0;
        se_fan_to_polar(g, (float)(g->apex_u + 300.0 * sin(t)), (float)(g->apex_v - 300.0 * cos(t)), &r, &b);
        if (!near(r, 300.0f, 1e-2f) || !near(b, deg[k], 1e-3f)) { printf("to_polar %g deg: r %g b %g\n", deg[k], r, b); fails++; }
    }

    /* polar mode: constant frame */
    const int nb = 128, ns = 439;
    if (se_fe_init_polar(g, nb, ns, NULL)) { printf("se_fe_init_polar failed\n"); return 1; }
    if (se_fe_src_bytes() != nb * ns) { printf("polar src bytes %d\n", se_fe_src_bytes()); fails++; }
    uint8_t *pol = malloc((size_t)nb * ns);
    for (int i = 0; i < nb * ns; i++) pol[i] = 200;
    se_fe_run(pol, dst);
    /* network pixel at the fan's centre line, mid range; and a corner outside the fan */
    const float sx = (float)FW / w, sy = (float)FH / h;
    const int cx = (int)(g->apex_u / sx), cy = (int)((g->apex_v - 0.5f * (g->r_min_px + g->r_max_px)) / sy);
    if (dst[cy * w + cx] != 200) { printf("polar inside: %d\n", dst[cy * w + cx]); fails++; }
    if (dst[0] != 0 || dst[(h - 1) * w] != 0) { printf("polar outside: %d %d\n", dst[0], dst[(h - 1) * w]); fails++; }
    free(pol);

    if (fails) { printf("front-end test FAILED (%d)\n", fails); return 1; }
    printf("front-end test passed: fan mode bit-identical to the Python pipeline (%d frames, %dx%d), geometry and polar checks ok\n",
           checked, w, h);
    return 0;
}
