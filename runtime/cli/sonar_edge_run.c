/* sonar-edge-run: the end-to-end pipeline, sonar frame in, JSON out.
 *
 *   sonar-edge-run [-t THR] [-n REPEAT] [-m DIR] [-o DIR] frame.pgm ...
 *   sonar-edge-run [options] --polar BEAMSxSAMPLES [--beams FILE] raw.bin|-
 *   sonar-edge-run [options] --aris [--fit metric|window] [--every N] [--first K] [--count N] rec.aris ...
 *   sonar-edge-run [-t THR] --stream HOST:PORT [--policy latest|queue] [--workers N] [--qcap N]
 *                  [--from SEQ] [--name NAME] [--fit metric|window]
 *
 * Inputs
 *   PGM (P5, 8-bit) at the network's size: used as is (no front end).
 *   PGM at the fan-image size of se_fan_watertank (320x480): front end in fan
 *       mode (lookup-table resize, bit-exact with the training pipeline).
 *   --polar BxS: raw uint8 frames of S range samples (near to far) x B beams
 *       (beam 0 leftmost), back to back in a file or on stdin ("-"); each is
 *       rendered to the network input with the se_fan_watertank geometry.
 *       --beams FILE: B beam angles in degrees (text), else uniform over the fov.
 *   --aris: Sound Metrics ARIS recordings; each frame's beams are rendered with
 *       the SDK's per-beam angles and the frame's range window (aris.h).
 *       --fit metric (default): true fan shape, far edge on the training fan's
 *       far radius, ranges in metres; --fit window: the range window stretched
 *       onto the training fan's near and far radius, ranges in pixels.
 *       --every N / --first K / --count N select frames (every Nth from K).
 *   --stream: live frames from sonar-stream (stream/protocol.h); see
 *       stream/stream_client.c for the worker processes and the two policies.
 *
 * Output: one JSON object per frame and line on stdout:
 *   {"frame": k, "source": ..., "mode": "net|fan|polar",
 *    "ms": {"frontend", "network", "decode", "seg", "total"},
 *    "detections": [{"cls", "name", "score" (logit), "prob", "box" (network px
 *        x0 y0 x1 y1), "box_fan" (fan-image px), "range", "bearing_deg"}],
 *    "range_unit": "m" | "px",
 *    "mask": {"h", "w", "cells": {label: count, ...}}}   (models with a seg head)
 * Range and bearing are those of the box centre. With -n REPEAT > 1 every
 * stage runs REPEAT times and the times are means (benchmark mode).
 *   -m DIR  label map per frame, DIR/<name>.labels.pgm (0 = background, c + 1 = class c)
 *   -o DIR  overlay per frame, DIR/<name>.overlay.ppm: the input image (fan
 *           image in fan mode, else the network input) with the mask tinted
 *           and the boxes with probability >= --show P (default 0.3) drawn in
 *           class colours. The JSON lists every detection with logit >= THR
 *           (default -4, the evaluation setting, as sonar-detect).
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../aris.h"
#include "../frontend.h"
#include "../sonar_edge.h"

#define MAX_DETS 64

enum { MODE_NET, MODE_FAN, MODE_POLAR };
static const char *mode_name[] = {"net", "fan", "polar"};
static const se_fan_geom_t *geom = &se_fan_watertank;   /* geometry of the current input (ARIS: per recording) */

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void usage(const char *p)
{
    fprintf(stderr,
            "usage: %s [-t THR] [-n REPEAT] [-m DIR] [-o DIR [--show P]] frame.pgm ...\n"
            "       %s [-t THR] [-n REPEAT] [-m DIR] [-o DIR [--show P]] --polar BEAMSxSAMPLES [--beams FILE] raw.bin|-\n"
            "       %s [-t THR] [-n REPEAT] [-m DIR] [-o DIR [--show P]] --aris [--fit metric|window] [--every N] [--first K] [--count N] rec.aris ...\n"
            "       %s [-t THR] --stream HOST:PORT [--policy latest|queue] [--workers N] [--qcap N] [--from SEQ] [--name NAME] [--fit metric|window]\n",
            p, p, p, p);
}


/* Binary 8-bit PGM of any size; *buf is (re)allocated. */
static int read_pgm(const char *path, uint8_t **buf, int *h, int *w)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    int maxv;
    if (fscanf(f, "P5 %d %d %d", w, h, &maxv) != 3 || maxv != 255 || *w <= 0 || *h <= 0 || fgetc(f) == EOF) {
        fprintf(stderr, "%s: not a binary 8-bit PGM\n", path);
        fclose(f);
        return -1;
    }
    uint8_t *b = realloc(*buf, (size_t)*h * *w);
    if (!b) { fclose(f); return -1; }
    *buf = b;
    const size_t n = fread(b, 1, (size_t)*h * *w, f);
    fclose(f);
    if (n != (size_t)*h * *w) { fprintf(stderr, "%s: truncated\n", path); return -1; }
    return 0;
}

static void json_str(FILE *o, const char *s)
{
    fputc('"', o);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', o);
        if ((unsigned char)*s < 0x20) fprintf(o, "\\u%04x", (unsigned char)*s);
        else fputc(*s, o);
    }
    fputc('"', o);
}

static const char *cls_name(int c, char *tmp, size_t n)
{
    const char *s = se_class_name(c);
    if (s) return s;
    snprintf(tmp, n, "class%d", c);
    return tmp;
}

static const char *base_name(const char *p)
{
    const char *b = strrchr(p, '/');
    return b ? b + 1 : p;
}

/* Class colours for the overlay (label c + 1 uses palette[c]). */
static const uint8_t palette[][3] = {
    {230, 25, 75}, {60, 180, 75}, {255, 225, 25}, {0, 130, 200}, {245, 130, 48}, {145, 30, 180},
    {70, 240, 240}, {240, 50, 230}, {210, 245, 60}, {250, 190, 212}, {128, 128, 128}, {170, 110, 40},
};
#define N_PAL ((int)(sizeof palette / sizeof palette[0]))

static void put_px(uint8_t *rgb, int w, int h, int x, int y, const uint8_t *c)
{
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    memcpy(rgb + ((size_t)y * w + x) * 3, c, 3);
}

/* Overlay on an image of iw x ih pixels whose pixel (u, v) corresponds to the
 * network-input pixel (u * sx, v * sy). Boxes are given in image pixels. */
static int write_overlay(const char *path, const uint8_t *img, int iw, int ih, double sx, double sy,
                         const uint8_t *labels, int lh, int lw, const se_detection_t *d, const float (*box)[4], int n,
                         float show_logit)
{
    uint8_t *rgb = malloc((size_t)iw * ih * 3);
    if (!rgb) return -1;
    const double cell_x = (double)se_input_width() / lw, cell_y = (double)se_input_height() / lh;
    for (int v = 0; v < ih; v++) {
        for (int u = 0; u < iw; u++) {
            const uint8_t g = img[(size_t)v * iw + u];
            uint8_t *p = rgb + ((size_t)v * iw + u) * 3;
            p[0] = p[1] = p[2] = g;
            if (!labels) continue;
            const int lx = (int)((u + 0.5) * sx / cell_x), ly = (int)((v + 0.5) * sy / cell_y);
            if (lx < 0 || ly < 0 || lx >= lw || ly >= lh) continue;
            const int lab = labels[(size_t)ly * lw + lx];
            if (lab == 0) continue;
            const uint8_t *c = palette[(lab - 1) % N_PAL];
            for (int k = 0; k < 3; k++) p[k] = (uint8_t)((g + c[k]) / 2);
        }
    }
    for (int i = 0; i < n; i++) {
        if (d[i].score < show_logit) continue;
        const uint8_t *c = palette[d[i].cls % N_PAL];
        const int x0 = (int)lroundf(box[i][0]), y0 = (int)lroundf(box[i][1]);
        const int x1 = (int)lroundf(box[i][2]), y1 = (int)lroundf(box[i][3]);
        for (int x = x0; x <= x1; x++) { put_px(rgb, iw, ih, x, y0, c); put_px(rgb, iw, ih, x, y1, c); }
        for (int y = y0; y <= y1; y++) { put_px(rgb, iw, ih, x0, y, c); put_px(rgb, iw, ih, x1, y, c); }
    }
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); free(rgb); return -1; }
    fprintf(f, "P6\n%d %d\n255\n", iw, ih);
    fwrite(rgb, 1, (size_t)iw * ih * 3, f);
    fclose(f);
    free(rgb);
    return 0;
}

typedef struct {
    float thr, show_logit;
    int repeat;
    const char *maskdir, *ovldir;
    int frame_no;
} opts_t;

/* Front end (unless mode is MODE_NET), network, decode, segmentation; print one JSON line. */
static int process(const opts_t *o, int mode, const uint8_t *src, int src_w, int src_h, const char *source, int index, FILE *out)
{
    static uint8_t *net = NULL, *labels = NULL;
    const int H = se_input_height(), W = se_input_width();
    const int nl = se_seg_classes(), lh = se_seg_height(), lw = se_seg_width();
    if (!net) net = malloc((size_t)H * W);
    if (nl > 0 && !labels) labels = malloc((size_t)lh * lw);
    if (!net || (nl > 0 && !labels)) return -1;

    se_detection_t dets[MAX_DETS];
    int n = 0;
    double t_fe = 0, t_net = 0, t_dec = 0, t_seg = 0;
    for (int r = 0; r < o->repeat; r++) {
        double t0 = now_s();
        if (mode == MODE_NET) memcpy(net, src, (size_t)H * W);
        else se_fe_run(src, net);
        double t1 = now_s();
        const int32_t *hm = se_forward(net);
        double t2 = now_s();
        n = se_decode(hm, o->thr, dets, MAX_DETS);
        double t3 = now_s();
        if (nl > 0) se_segment_head(hm, labels);
        double t4 = now_s();
        if (mode != MODE_NET) t_fe += t1 - t0;
        t_net += t2 - t1; t_dec += t3 - t2; t_seg += t4 - t3;
    }
    const double k = 1e3 / o->repeat;
    t_fe *= k; t_net *= k; t_dec *= k; t_seg *= k;

    const se_fan_geom_t *g = geom;
    char tmp[32];
    float box_fan[MAX_DETS][4], box_net[MAX_DETS][4];
    fprintf(out, "{\"frame\": %d, \"source\": ", o->frame_no);
    json_str(out, source);
    if (index >= 0) fprintf(out, ", \"index\": %d", index);
    fprintf(out, ", \"mode\": \"%s\", \"ms\": {\"frontend\": %.3f, \"network\": %.3f, \"decode\": %.3f, \"seg\": %.3f, \"total\": %.3f}",
           mode_name[mode], t_fe, t_net, t_dec, t_seg, t_fe + t_net + t_dec + t_seg);
    fprintf(out, ", \"range_unit\": \"%s\", \"detections\": [", g->m_per_px > 0 ? "m" : "px");
    for (int i = 0; i < n; i++) {
        const se_detection_t *d = &dets[i];
        float u0, v0, u1, v1, uc, vc, range, bearing;
        se_fe_to_fan(d->x0, d->y0, &u0, &v0);
        se_fe_to_fan(d->x1, d->y1, &u1, &v1);
        se_fe_to_fan(0.5f * (d->x0 + d->x1), 0.5f * (d->y0 + d->y1), &uc, &vc);
        se_fan_to_polar(g, uc, vc, &range, &bearing);
        box_net[i][0] = d->x0; box_net[i][1] = d->y0; box_net[i][2] = d->x1; box_net[i][3] = d->y1;
        box_fan[i][0] = fminf(u0, u1); box_fan[i][1] = fminf(v0, v1);
        box_fan[i][2] = fmaxf(u0, u1); box_fan[i][3] = fmaxf(v0, v1);
        fprintf(out, "%s{\"cls\": %d, \"name\": ", i ? ", " : "", d->cls);
        json_str(out, cls_name(d->cls, tmp, sizeof tmp));
        fprintf(out, ", \"score\": %.6f, \"prob\": %.6f, \"box\": [%.3f, %.3f, %.3f, %.3f], \"box_fan\": [%.2f, %.2f, %.2f, %.2f]"
               ", \"range\": %.4f, \"bearing_deg\": %.3f}",
               d->score, 1.0 / (1.0 + exp(-(double)d->score)), d->x0, d->y0, d->x1, d->y1,
               box_fan[i][0], box_fan[i][1], box_fan[i][2], box_fan[i][3], range, bearing);
    }
    fprintf(out, "]");
    if (nl > 0) {
        long cnt[256] = {0};
        for (long i = 0; i < (long)lh * lw; i++) cnt[labels[i]]++;
        fprintf(out, ", \"mask\": {\"h\": %d, \"w\": %d, \"cells\": {", lh, lw);
        int first = 1;
        for (int c = 0; c < nl; c++) {
            if (!cnt[c]) continue;
            fprintf(out, "%s", first ? "" : ", ");
            json_str(out, c == 0 ? "background" : cls_name(c - 1, tmp, sizeof tmp));
            fprintf(out, ": %ld", cnt[c]);
            first = 0;
        }
        fprintf(out, "}}");
    }
    fprintf(out, "}\n");
    fflush(out);

    char name[512], path[4096];
    if (index >= 0) snprintf(name, sizeof name, "%s.%06d", base_name(source), index);
    else snprintf(name, sizeof name, "%s", base_name(source));
    if (o->maskdir && nl > 0) {
        snprintf(path, sizeof path, "%s/%s.labels.pgm", o->maskdir, name);
        FILE *f = fopen(path, "wb");
        if (!f) { perror(path); return -1; }
        fprintf(f, "P5\n%d %d\n255\n", lw, lh);
        fwrite(labels, 1, (size_t)lh * lw, f);
        fclose(f);
    }
    if (o->ovldir) {
        snprintf(path, sizeof path, "%s/%s.overlay.ppm", o->ovldir, name);
        int rc;
        if (mode == MODE_FAN)   /* draw on the fan image: fan pixel (u, v) <-> network pixel (u * W / src_w, v * H / src_h) */
            rc = write_overlay(path, src, src_w, src_h, (double)W / src_w, (double)H / src_h, nl > 0 ? labels : NULL, lh, lw,
                               dets, (const float (*)[4])box_fan, n, o->show_logit);
        else
            rc = write_overlay(path, net, W, H, 1.0, 1.0, nl > 0 ? labels : NULL, lh, lw, dets, (const float (*)[4])box_net, n, o->show_logit);
        if (rc) return -1;
    }
    return 0;
}

/* Every selected frame of each ARIS recording through the pipeline; the front
 * end is (re)built per recording and whenever the range window changes. */
static int run_aris(opts_t *o, int fit, long long every, long long first, long long count, int argc, char **argv, int i)
{
    static se_fan_geom_t g;
    for (; i < argc; i++) {
        se_aris_file_t a;
        if (se_aris_open(&a, argv[i])) return 1;
        const se_aris_frame_t *f0 = &a.first;
        fprintf(stderr, "%s: ARIS %s, %d beams x %d samples, window %.2f-%.2f m, %.1f fps, %lld frames\n", argv[i],
                f0->system_type == 1 ? "3000" : f0->system_type == 0 ? "1800" : f0->system_type == 2 ? "1200" : "?",
                f0->n_beams, f0->n_samples, f0->window_start_m, f0->window_start_m + f0->n_samples * f0->sample_len_m,
                f0->frame_rate, a.n_frames);
        uint8_t *buf = malloc(a.frame_bytes);
        float *ang = malloc(sizeof(float) * (size_t)f0->n_beams);
        if (!buf || !ang) return 1;
        se_aris_frame_t h, cur = {0};
        int ready = 0;
        long long done = 0;
        for (long long k = first; k < a.n_frames && (count <= 0 || done < count); k += every) {
            if (se_aris_read(&a, k, &h, buf)) { fprintf(stderr, "%s: frame %lld unreadable, stopping\n", argv[i], k); break; }
            if (!ready || !se_aris_same_geom(&h, &cur)) {
                const int have = se_aris_beam_deg(&h, ang) == 0;
                if (!have) fprintf(stderr, "%s: no beam table for this system/lens; uniform beams\n", argv[i]);
                se_aris_geom(&h, have ? ang : NULL, fit, &g);
                if (se_fe_init_polar(&g, h.n_beams, h.n_samples, have ? ang : NULL) != 0) {
                    fprintf(stderr, "%s: front end: bad polar geometry\n", argv[i]);
                    return 1;
                }
                geom = &g;
                cur = h;
                ready = 1;
            }
            if (process(o, MODE_POLAR, buf, h.n_beams, h.n_samples, argv[i], (int)k, stdout)) return 1;
            o->frame_no++;
            done++;
        }
        free(buf);
        free(ang);
        se_aris_close(&a);
    }
    return 0;
}

#include "../stream/stream_client.c"

int main(int argc, char **argv)
{
    opts_t o = {-4.0f, 0.0f, 1, NULL, NULL, 0};
    float show = 0.3f;
    int beams = 0, samples = 0, i = 1, aris = 0, fit = SE_ARIS_FIT_METRIC;
    long long every = 1, first = 0, count = 0;
    const char *beam_file = NULL, *stream = NULL, *sname = NULL;
    int policy_latest = 1, workers = 1, qcap = 32;
    uint64_t from = SS_LIVE;
    for (; i < argc && argv[i][0] == '-' && argv[i][1]; i++) {
        if (!strcmp(argv[i], "--stream") && i + 1 < argc) { stream = argv[++i]; continue; }
        if (!strcmp(argv[i], "--policy") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "latest")) policy_latest = 1;
            else if (!strcmp(v, "queue")) policy_latest = 0;
            else { usage(argv[0]); return 2; }
            continue;
        }
        if (!strcmp(argv[i], "--workers") && i + 1 < argc) { workers = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--qcap") && i + 1 < argc) { qcap = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--from") && i + 1 < argc) { from = (uint64_t)strtoull(argv[++i], NULL, 10); continue; }
        if (!strcmp(argv[i], "--name") && i + 1 < argc) { sname = argv[++i]; continue; }
        if (!strcmp(argv[i], "-t") && i + 1 < argc) o.thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) o.repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) o.maskdir = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) o.ovldir = argv[++i];
        else if (!strcmp(argv[i], "--polar") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &beams, &samples) != 2 || beams <= 0 || samples <= 0) { usage(argv[0]); return 2; }
        } else if (!strcmp(argv[i], "--beams") && i + 1 < argc) beam_file = argv[++i];
        else if (!strcmp(argv[i], "--aris")) aris = 1;
        else if (!strcmp(argv[i], "--fit") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "metric")) fit = SE_ARIS_FIT_METRIC;
            else if (!strcmp(v, "window")) fit = SE_ARIS_FIT_WINDOW;
            else { usage(argv[0]); return 2; }
        } else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--first") && i + 1 < argc) first = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--show") && i + 1 < argc) show = (float)atof(argv[++i]);
        else { usage(argv[0]); return 2; }
    }
    if (stream) {
        if (i < argc || o.repeat != 1 || o.maskdir || o.ovldir || beams || aris) { usage(argv[0]); return 2; }
        return run_stream(&o, stream, policy_latest, workers, qcap, from, sname, fit);
    }
    if (i >= argc || o.repeat < 1 || show <= 0.0f || show >= 1.0f || every < 1 || first < 0 || (aris && beams)) { usage(argv[0]); return 2; }
    o.show_logit = logf(show / (1.0f - show));
    if (o.maskdir && se_seg_classes() <= 0) fprintf(stderr, "-m: the compiled-in model has no segmentation head; no masks written\n");
    const int H = se_input_height(), W = se_input_width();
    const se_fan_geom_t *g = &se_fan_watertank;

    if (aris) return run_aris(&o, fit, every, first, count, argc, argv, i);
    if (beams) {   /* raw polar stream(s) */
        float *ang = NULL;
        if (beam_file) {
            FILE *f = fopen(beam_file, "r");
            if (!f) { perror(beam_file); return 1; }
            ang = malloc(sizeof(float) * (size_t)beams);
            for (int b = 0; b < beams; b++)
                if (fscanf(f, "%f", &ang[b]) != 1) { fprintf(stderr, "%s: need %d angles\n", beam_file, beams); return 1; }
            fclose(f);
        }
        if (se_fe_init_polar(g, beams, samples, ang) != 0) { fprintf(stderr, "front end: bad polar geometry\n"); return 1; }
        free(ang);
        const size_t fb = (size_t)beams * samples;
        uint8_t *buf = malloc(fb);
        for (; i < argc; i++) {
            FILE *f = strcmp(argv[i], "-") ? fopen(argv[i], "rb") : stdin;
            if (!f) { perror(argv[i]); return 1; }
            for (int k = 0;; k++) {
                const size_t got = fread(buf, 1, fb, f);
                if (got == 0) break;
                if (got != fb) { fprintf(stderr, "%s: partial frame %d (%zu of %zu bytes) ignored\n", argv[i], k, got, fb); break; }
                if (process(&o, MODE_POLAR, buf, beams, samples, strcmp(argv[i], "-") ? argv[i] : "stdin", k, stdout)) return 1;
                o.frame_no++;
            }
            if (f != stdin) fclose(f);
        }
        free(buf);
        return 0;
    }

    uint8_t *img = NULL;
    int fe_ready = 0;   /* fan-mode front end initialised (also gives the coordinate mapping in net mode) */
    for (; i < argc; i++) {
        int h, w, mode;
        if (read_pgm(argv[i], &img, &h, &w)) return 1;
        if (h == H && w == W) mode = MODE_NET;
        else if (h == g->height && w == g->width) mode = MODE_FAN;
        else {
            fprintf(stderr, "%s: size %dx%d is neither the network input %dx%d nor the fan image %dx%d\n",
                    argv[i], w, h, W, H, g->width, g->height);
            return 1;
        }
        if (!fe_ready) {
            if (se_fe_init_fan(g) != 0) { fprintf(stderr, "front end: bad fan geometry\n"); return 1; }
            fe_ready = 1;
        }
        if (process(&o, mode, img, w, h, argv[i], -1, stdout)) return 1;
        o.frame_no++;
    }
    free(img);
    return 0;
}
