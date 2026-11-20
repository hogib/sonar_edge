/* aris_dump FILE K FIT OUT.pgm: front-end output (network input) for frame K, plus geometry on stdout. */
#include <stdio.h>
#include <stdlib.h>
#include "../rt/aris.h"
#include "../rt/frontend.h"
#include "../rt/sonar_edge.h"

int main(int argc, char **argv)
{
    if (argc != 5) { fprintf(stderr, "usage: %s FILE K FIT OUT.pgm\n", argv[0]); return 2; }
    se_aris_file_t a;
    if (se_aris_open(&a, argv[1])) return 1;
    se_aris_frame_t h;
    uint8_t *buf = malloc(a.frame_bytes);
    if (se_aris_read(&a, atoll(argv[2]), &h, buf)) return 1;
    float ang[128];
    const int have = se_aris_beam_deg(&h, ang) == 0;
    se_fan_geom_t g;
    se_aris_geom(&h, have ? ang : NULL, atoi(argv[3]), &g);
    if (se_fe_init_polar(&g, h.n_beams, h.n_samples, have ? ang : NULL)) return 1;
    const int H = se_input_height(), W = se_input_width();
    uint8_t *out = malloc((size_t)H * W);
    se_fe_run(buf, out);
    FILE *f = fopen(argv[4], "wb");
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    fwrite(out, 1, (size_t)H * W, f);
    fclose(f);
    printf("%g %g %g %g %g %g %d %d %g %g %d %g\n", g.apex_u, g.apex_v, g.fov_deg, g.r_min_px, g.r_max_px, g.m_per_px,
           g.width, g.height, h.window_start_m, h.sample_len_m, h.n_samples, h.sound_speed);
    se_aris_close(&a);
    return 0;
}
