/* sonar-stream: replay sonar recordings as a live TCP stream (protocol.h).
 *
 *   sonar-stream [-p PORT] [--bind ADDR] [--fps F] [--duration S] [--no-loop] [--jitter MS] [--drop P]
 *                [--ring N] [--seed K] [--log ACKS.csv] rec.aris ...
 *   sonar-stream [...] --fan --fps F frame.pgm ...      (fan images instead of beam data)
 *
 * Frames are emitted on a fixed schedule: frame k at t0 + k / rate, where rate
 * is --fps or each recording's own frame rate (ARIS header). Its capture
 * timestamp is that nominal time (CLOCK_REALTIME). --jitter delays each
 * emission by a uniform random 0..MS ms (the capture time stays nominal);
 * --drop P skips a frame with probability P (its sequence number is used, so
 * clients see the gap). Recordings loop unless --no-loop. The last N frames
 * (--ring, default 256) are kept so a client can resume (STRT from_seq) after a
 * reconnect. A client that cannot keep up is sent the frames still in the
 * ring; older ones are skipped and counted.
 *
 * Each ACK becomes one line of --log (CSV, server clock):
 *   client,seq,rec_id,rec_index,t_capture_ns,t_send_ns,t_sent_ns,t_ack_ns,recv_to_done_us,proc_us,worker,queue_depth
 * capture->result latency = t_ack_ns - t_capture_ns, measured on one clock.
 */
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "../aris.h"
#include "protocol.h"

#define MAX_CLIENTS 16
#define MAX_RECS 256

typedef struct {
    int is_fan;
    const char *path;
    se_aris_file_t a;     /* ARIS */
    int w, h;             /* fan PGM */
} rec_t;

typedef struct {
    uint64_t seq;
    int valid;
    size_t len;
    uint8_t *buf;         /* header + payload */
} slot_t;

typedef struct {
    int fd, started;
    char name[25];
    uint64_t next_seq;    /* next sequence number to send */
    uint8_t *out;         /* packet being sent (private copy) */
    size_t out_len, out_off;
    uint64_t out_seq;
    uint8_t in[SS_MSG_LEN];
    size_t in_len;
    uint64_t sent, skipped, acks;
    uint64_t *t_sent;     /* per ring slot: send-complete time of that seq to this client */
} client_t;

static volatile sig_atomic_t stop_flag = 0;
static void on_sig(int s) { (void)s; stop_flag = 1; }

static uint64_t real_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static double mono_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint64_t rng_state = 88172645463325252ull;
static double rnd(void)   /* xorshift64*, uniform [0, 1) */
{
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return (double)((rng_state * 2685821657736338717ull) >> 11) / 9007199254740992.0;
}

static int read_pgm(const char *path, uint8_t **buf, int *w, int *h)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    int maxv;
    if (fscanf(f, "P5 %d %d %d", w, h, &maxv) != 3 || maxv != 255 || *w <= 0 || *h <= 0 || *w > 65535 || *h > 65535 || fgetc(f) == EOF) {
        fprintf(stderr, "%s: not a binary 8-bit PGM\n", path);
        fclose(f);
        return -1;
    }
    if (buf) {
        uint8_t *b = realloc(*buf, (size_t)*w * *h);
        if (!b) { fclose(f); return -1; }
        *buf = b;
        if (fread(b, 1, (size_t)*w * *h, f) != (size_t)*w * *h) { fprintf(stderr, "%s: truncated\n", path); fclose(f); return -1; }
    }
    fclose(f);
    return 0;
}

static void drop_client(client_t *c, const char *why)
{
    fprintf(stderr, "sonar-stream: client %s closed (%s): sent %llu, skipped %llu, acks %llu\n", c->name,
            why, (unsigned long long)c->sent, (unsigned long long)c->skipped, (unsigned long long)c->acks);
    close(c->fd);
    free(c->out);
    free(c->t_sent);
    memset(c, 0, sizeof *c);
    c->fd = -1;
}

static void usage(const char *p)
{
    fprintf(stderr, "usage: %s [-p PORT] [--bind ADDR] [--fps F] [--duration S] [--no-loop] [--jitter MS] [--drop P] [--ring N] [--seed K] [--log ACKS.csv] rec.aris ...\n"
                    "       %s [...] --fan --fps F frame.pgm ...\n", p, p);
}

int main(int argc, char **argv)
{
    int port = 5800, loop = 1, fan = 0, ring_n = 256;
    double fps = 0, duration = 0, jitter_ms = 0, drop_p = 0;
    const char *log_path = NULL, *bind_addr = NULL;
    int i = 1;
    for (; i < argc && argv[i][0] == '-'; i++) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bind") && i + 1 < argc) bind_addr = argv[++i];
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) fps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc) duration = atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-loop")) loop = 0;
        else if (!strcmp(argv[i], "--jitter") && i + 1 < argc) jitter_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--drop") && i + 1 < argc) drop_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ring") && i + 1 < argc) ring_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) rng_state = (uint64_t)atoll(argv[++i]) * 2654435761ull + 1;
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) log_path = argv[++i];
        else if (!strcmp(argv[i], "--fan")) fan = 1;
        else { usage(argv[0]); return 2; }
    }
    if (i >= argc || ring_n < 2 || fps < 0 || (fan && fps <= 0) || drop_p < 0 || drop_p >= 1 || jitter_ms < 0) { usage(argv[0]); return 2; }
    const int n_recs = argc - i;
    if (n_recs > MAX_RECS && !fan) { fprintf(stderr, "too many recordings\n"); return 2; }

    /* sources */
    rec_t *recs = calloc((size_t)n_recs, sizeof *recs);
    size_t max_payload = 0;
    for (int r = 0; r < n_recs; r++) {
        recs[r].path = argv[i + r];
        recs[r].is_fan = fan;
        if (fan) {
            if (read_pgm(recs[r].path, NULL, &recs[r].w, &recs[r].h)) return 1;
            if ((size_t)recs[r].w * recs[r].h > max_payload) max_payload = (size_t)recs[r].w * recs[r].h;
        } else {
            if (se_aris_open(&recs[r].a, recs[r].path)) return 1;
            if (recs[r].a.frame_bytes > max_payload) max_payload = recs[r].a.frame_bytes;
            fprintf(stderr, "sonar-stream: %s: %lld frames, %d x %d, %.1f fps\n", recs[r].path, recs[r].a.n_frames,
                    recs[r].a.first.n_beams, recs[r].a.first.n_samples, recs[r].a.first.frame_rate);
        }
    }
    const size_t max_pkt = SS_HDR_LEN + max_payload;

    slot_t *ring = calloc((size_t)ring_n, sizeof *ring);
    for (int k = 0; k < ring_n; k++) ring[k].buf = malloc(max_pkt);
    client_t cl[MAX_CLIENTS];
    for (int k = 0; k < MAX_CLIENTS; k++) { memset(&cl[k], 0, sizeof cl[k]); cl[k].fd = -1; }
    FILE *logf = NULL;
    if (log_path) {
        logf = fopen(log_path, "w");
        if (!logf) { perror(log_path); return 1; }
        fprintf(logf, "client,seq,rec_id,rec_index,t_capture_ns,t_send_ns,t_sent_ns,t_ack_ns,recv_to_done_us,proc_us,worker,queue_depth\n");
    }

    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    const int lfd = socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa = {0};
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    sa.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind_addr && inet_pton(AF_INET, bind_addr, &sa.sin_addr) != 1) { fprintf(stderr, "--bind: bad IPv4 address\n"); return 2; }
    if (bind(lfd, (struct sockaddr *)&sa, sizeof sa) || listen(lfd, 8)) { perror("bind/listen"); return 1; }
    fcntl(lfd, F_SETFL, O_NONBLOCK);
    fprintf(stderr, "sonar-stream: listening on %s:%d, %d source(s), ring %d\n", bind_addr ? bind_addr : "*", port, n_recs, ring_n);

    /* production state */
    int cur = 0;
    long long cur_frame = 0;
    uint64_t seq = 0, produced = 0, dropped_inj = 0;
    double t0 = -1, t_nominal = 0, t_emit = 0;   /* monotonic seconds */
    uint64_t t0_real = 0;
    uint8_t *fanbuf = NULL;
    int done_sources = 0;
    double t_done_sources = -1;
    uint64_t emit_lag_max_us = 0, emit_lag_sum_us = 0;

    for (;;) {
        if (stop_flag) break;
        const double now = mono_s();
        if (t0 >= 0 && duration > 0 && now - t0 >= duration) done_sources = 1;   /* stop producing, collect the last ACKs */

        /* emit due frames (production starts with the first client START) */
        while (t0 >= 0 && !done_sources && now >= t_emit) {
            rec_t *R = &recs[cur];
            ss_hdr_t h = {0};
            uint8_t *payload;
            slot_t *s = &ring[seq % (uint64_t)ring_n];
            payload = s->buf + SS_HDR_LEN;
            double rate = fps;
            if (R->is_fan) {
                int w, h2;
                if (read_pgm(R->path, &fanbuf, &w, &h2)) return 1;
                memcpy(payload, fanbuf, (size_t)w * h2);
                h.n_beams = (uint16_t)w; h.n_samples = (uint16_t)h2;
                h.flags = SS_FAN;
                h.rec_index = 0;
            } else {
                se_aris_frame_t fh;
                if (se_aris_read(&R->a, cur_frame, &fh, payload)) { fprintf(stderr, "%s: frame %lld unreadable\n", R->path, cur_frame); return 1; }
                h.n_beams = (uint16_t)fh.n_beams; h.n_samples = (uint16_t)fh.n_samples;
                h.system_type = (uint8_t)fh.system_type; h.ping_mode = (uint8_t)fh.ping_mode; h.large_lens = (uint8_t)fh.large_lens;
                h.sample_period_us = fh.sample_period_us; h.sample_start_delay_us = fh.sample_start_delay_us;
                h.sound_speed = fh.sound_speed; h.frame_rate = fh.frame_rate;
                h.rec_index = (uint32_t)cur_frame;
                if (rate <= 0) rate = fh.frame_rate > 0 ? fh.frame_rate : 15.0;
            }
            h.seq = seq;
            h.rec_id = (uint16_t)cur;
            h.t_capture_ns = t0_real + (uint64_t)((t_nominal - t0) * 1e9);
            h.t_send_ns = real_ns();
            const uint64_t lag = (uint64_t)((now - t_nominal) * 1e6);
            emit_lag_sum_us += lag;
            if (lag > emit_lag_max_us) emit_lag_max_us = lag;
            if (drop_p > 0 && rnd() < drop_p) {
                s->valid = 0;   /* injected loss: the sequence number is spent, nothing is sent */
                dropped_inj++;
            } else {
                ss_pack_hdr(s->buf, &h);
                s->seq = seq;
                s->len = SS_HDR_LEN + (size_t)h.n_beams * h.n_samples;
                s->valid = 1;
            }
            seq++;
            produced++;
            /* advance the source */
            const long long nfr = R->is_fan ? 1 : R->a.n_frames;
            if (++cur_frame >= nfr) {
                cur_frame = 0;
                if (++cur >= n_recs) {
                    cur = 0;
                    if (!loop) done_sources = 1;
                }
            }
            t_nominal += 1.0 / rate;
            t_emit = t_nominal + (jitter_ms > 0 ? rnd() * jitter_ms * 1e-3 : 0);
        }
        if (done_sources) {   /* all sent: wait for the clients' last ACKs (at most 10 s) */
            int busy = 0;
            for (int k = 0; k < MAX_CLIENTS; k++)
                if (cl[k].fd >= 0 && (cl[k].out_len || cl[k].next_seq < seq || cl[k].acks < cl[k].sent)) busy = 1;
            if (t_done_sources < 0) t_done_sources = now;
            if (!busy || now - t_done_sources > 10.0) break;
        }

        /* hand the next frame to idle clients */
        const uint64_t oldest = seq > (uint64_t)ring_n ? seq - (uint64_t)ring_n : 0;
        for (int k = 0; k < MAX_CLIENTS; k++) {
            client_t *c = &cl[k];
            if (c->fd < 0 || !c->started || c->out_len) continue;
            while (c->next_seq < seq) {
                if (c->next_seq < oldest) { c->skipped += oldest - c->next_seq; c->next_seq = oldest; }
                slot_t *s = &ring[c->next_seq % (uint64_t)ring_n];
                if (!s->valid || s->seq != c->next_seq) { c->next_seq++; continue; }   /* injected loss */
                if (!c->out) c->out = malloc(max_pkt);
                memcpy(c->out, s->buf, s->len);
                c->out_len = s->len;
                c->out_off = 0;
                c->out_seq = c->next_seq;
                c->next_seq++;
                break;
            }
        }

        /* poll */
        struct pollfd pf[MAX_CLIENTS + 1];
        int idx[MAX_CLIENTS + 1], np = 0;
        pf[np].fd = lfd; pf[np].events = POLLIN; idx[np++] = -1;
        for (int k = 0; k < MAX_CLIENTS; k++) {
            if (cl[k].fd < 0) continue;
            pf[np].fd = cl[k].fd;
            pf[np].events = POLLIN | (cl[k].out_len ? POLLOUT : 0);
            idx[np++] = k;
        }
        int timeout_ms = 1000;
        if (t0 >= 0 && !done_sources) {
            const double dt = t_emit - mono_s();
            timeout_ms = dt <= 0 ? 0 : (int)(dt * 1000.0);   /* round down, then poll(0) the last < 1 ms */
            if (timeout_ms > 1000) timeout_ms = 1000;
        }
        if (poll(pf, (nfds_t)np, timeout_ms) < 0) { if (errno == EINTR) continue; perror("poll"); break; }

        for (int p = 0; p < np; p++) {
            if (!pf[p].revents) continue;
            if (idx[p] < 0) {   /* new connection */
                const int fd = accept(lfd, NULL, NULL);
                if (fd < 0) continue;
                int k = 0;
                while (k < MAX_CLIENTS && cl[k].fd >= 0) k++;
                if (k == MAX_CLIENTS) { close(fd); continue; }
                fcntl(fd, F_SETFL, O_NONBLOCK);
                setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                memset(&cl[k], 0, sizeof cl[k]);
                cl[k].fd = fd;
                cl[k].t_sent = calloc((size_t)ring_n, sizeof(uint64_t));
                snprintf(cl[k].name, sizeof cl[k].name, "#%d", k);
                continue;
            }
            client_t *c = &cl[idx[p]];
            if (pf[p].revents & POLLOUT && c->out_len) {
                const ssize_t n = send(c->fd, c->out + c->out_off, c->out_len - c->out_off, 0);
                if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) { drop_client(c, "send error"); continue; }
                if (n > 0) c->out_off += (size_t)n;
                if (c->out_off == c->out_len) {
                    c->t_sent[c->out_seq % (uint64_t)ring_n] = real_ns();
                    c->out_len = 0;
                    c->sent++;
                }
            }
            if (pf[p].revents & (POLLIN | POLLHUP | POLLERR)) {
                const ssize_t n = recv(c->fd, c->in + c->in_len, SS_MSG_LEN - c->in_len, 0);
                if (n <= 0) { if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue; drop_client(c, "disconnected"); continue; }
                c->in_len += (size_t)n;
                if (c->in_len < SS_MSG_LEN) continue;
                c->in_len = 0;
                const uint8_t *m = c->in;
                if (!memcmp(m, "HELO", 4)) {
                    char nm[25] = {0};
                    memcpy(nm, m + 8, 24);
                    snprintf(c->name, sizeof c->name, "%s", nm);
                    fprintf(stderr, "sonar-stream: hello from %s\n", c->name);
                } else if (!memcmp(m, "STRT", 4)) {
                    const uint64_t from = ss_get64(m + 4);
                    if (t0 < 0) {   /* start the clock with the first client */
                        t0 = mono_s();
                        t0_real = real_ns();
                        t_nominal = t_emit = t0;
                    }
                    const uint64_t old = seq > (uint64_t)ring_n ? seq - (uint64_t)ring_n : 0;
                    c->next_seq = from == SS_LIVE ? seq : (from < old ? old : (from > seq ? seq : from));
                    c->started = 1;
                    fprintf(stderr, "sonar-stream: %s starts at seq %llu\n", c->name, (unsigned long long)c->next_seq);
                } else if (!memcmp(m, "ACK ", 4)) {
                    const uint64_t s = ss_get64(m + 4), tc = ss_get64(m + 12);
                    const uint64_t t_ack = real_ns();
                    c->acks++;
                    if (logf) {
                        const slot_t *sl = &ring[s % (uint64_t)ring_n];
                        unsigned rid = 0, rix = 0;
                        uint64_t tsend = 0;
                        if (sl->valid && sl->seq == s) {
                            ss_hdr_t hh = {0};
                            ss_unpack_hdr(sl->buf, &hh);
                            rid = hh.rec_id; rix = hh.rec_index; tsend = hh.t_send_ns;
                        }
                        fprintf(logf, "%s,%llu,%u,%u,%llu,%llu,%llu,%llu,%u,%u,%u,%u\n", c->name, (unsigned long long)s, rid, rix,
                                (unsigned long long)tc, (unsigned long long)tsend,
                                (unsigned long long)c->t_sent[s % (uint64_t)ring_n], (unsigned long long)t_ack,
                                ss_get32(m + 20), ss_get32(m + 24), ss_get16(m + 28), ss_get16(m + 30));
                    }
                } else if (!memcmp(m, "BYE ", 4)) {
                    drop_client(c, "bye");
                }
            }
        }
    }
    for (int k = 0; k < MAX_CLIENTS; k++) if (cl[k].fd >= 0) drop_client(&cl[k], "server stopping");
    fprintf(stderr, "sonar-stream: produced %llu frames (%llu dropped by injection) in %.1f s; emission lag mean %.0f us, max %llu us\n",
            (unsigned long long)produced, (unsigned long long)dropped_inj, t0 >= 0 ? mono_s() - t0 : 0.0,
            produced ? (double)emit_lag_sum_us / (double)produced : 0.0, (unsigned long long)emit_lag_max_us);
    if (logf) fclose(logf);
    close(lfd);
    for (int r = 0; r < n_recs; r++) if (!recs[r].is_fan) se_aris_close(&recs[r].a);
    return 0;
}
