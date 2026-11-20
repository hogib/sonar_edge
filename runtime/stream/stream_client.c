/* sonar-edge-run --stream: real-time client for sonar-stream (protocol.h).
 * Included by cli/sonar_edge_run.c (it reuses that file's static process()).
 *
 *   sonar-edge-run [-t THR] --stream HOST:PORT [--policy latest|queue] [--workers N]
 *                  [--qcap N] [--from SEQ] [--name NAME] [--fit metric|window]
 *
 * The main process receives frames into shared-memory slots; N worker
 * processes (fork: the runtime keeps static state, one copy per process) each
 * run front end -> network -> decode (-> segmentation) on one frame at a time.
 *   latest  a worker always takes the newest received frame; older frames still
 *           waiting are discarded and counted as dropped (live display).
 *   queue   bounded FIFO of --qcap frames (default 32); when it is full the
 *           client stops reading the socket, so TCP backpressure reaches the
 *           server, which skips frames that leave its ring (seen as sequence gaps).
 * Per processed frame: one JSON line on stdout (the file-mode object plus
 * "stream": {...}), and an ACK to the server, which logs capture->result
 * latency on its own clock.
 */
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include "../stream/protocol.h"

#define SC_MAX_WORKERS 8
#define SC_DONE_RING 1024

enum { SC_FREE = 0, SC_FILLING, SC_READY, SC_BUSY };

typedef struct {
    int state;
    ss_hdr_t h;
    uint64_t t_recv_ns;      /* client CLOCK_REALTIME when the frame was complete */
    double recv_mono;        /* client CLOCK_MONOTONIC, same moment */
} sc_slot_t;

typedef struct {
    uint64_t seq, t_capture_ns;
    uint32_t recv_to_done_us, proc_us;
    uint16_t worker, qdepth;
} sc_done_t;

typedef struct {
    pthread_mutex_t mu, out_mu;
    pthread_cond_t cv;
    int quit;
    uint64_t dropped;        /* discarded by the latest policy */
    uint64_t gaps;           /* sequence numbers never received */
    uint64_t received, processed;
    sc_done_t done[SC_DONE_RING];
    unsigned done_head, done_tail;
} sc_shared_t;

static sc_shared_t *sh;
static sc_slot_t *slots;
static uint8_t *slot_data;
static size_t slot_bytes;
static int n_slots;

static uint64_t sc_real_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static int sc_read_full(int fd, void *buf, size_t n)
{
    uint8_t *p = buf;
    while (n) {
        const ssize_t r = recv(fd, p, n, 0);
        if (r == 0) return 1;
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        p += r; n -= (size_t)r;
    }
    return 0;
}

static int sc_send_full(int fd, const void *buf, size_t n)
{
    const uint8_t *p = buf;
    while (n) {
        const ssize_t r = send(fd, p, n, 0);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        p += r; n -= (size_t)r;
    }
    return 0;
}

static void *sc_shared_alloc(size_t n)
{
    const int fd = open("/dev/zero", O_RDWR);
    if (fd < 0) return NULL;
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return p == MAP_FAILED ? NULL : p;
}

/* Worker: take frames, run the pipeline, print JSON, report completion. */
static void sc_worker(int id, const opts_t *o0, int policy_latest, int fit, int notify_fd)
{
    opts_t o = *o0;
    se_fan_geom_t g;
    se_aris_frame_t cur;
    int ready = 0, fan_ready = 0;
    float ang[512];
    char *line = NULL;
    size_t line_cap = 0;
    for (;;) {
        pthread_mutex_lock(&sh->mu);
        int pick = -1, qdepth = 0;
        for (;;) {
            qdepth = 0;
            pick = -1;
            for (int s = 0; s < n_slots; s++) {
                if (slots[s].state != SC_READY) continue;
                qdepth++;
                if (pick < 0 || (policy_latest ? slots[s].h.seq > slots[pick].h.seq : slots[s].h.seq < slots[pick].h.seq)) pick = s;
            }
            if (pick >= 0 || sh->quit) break;
            pthread_cond_wait(&sh->cv, &sh->mu);
        }
        if (pick < 0) { pthread_mutex_unlock(&sh->mu); break; }
        if (policy_latest)
            for (int s = 0; s < n_slots; s++)
                if (s != pick && slots[s].state == SC_READY && slots[s].h.seq < slots[pick].h.seq) { slots[s].state = SC_FREE; sh->dropped++; }
        slots[pick].state = SC_BUSY;
        const uint64_t dropped_now = sh->dropped, gaps_now = sh->gaps;
        pthread_cond_broadcast(&sh->cv);
        pthread_mutex_unlock(&sh->mu);

        sc_slot_t *S = &slots[pick];
        const uint8_t *data = slot_data + (size_t)pick * slot_bytes;
        const double t_start = now_s();
        int mode;
        if (S->h.flags & SS_FAN) {
            mode = (S->h.n_beams == se_input_width() && S->h.n_samples == se_input_height()) ? MODE_NET : MODE_FAN;
            if (!fan_ready) { se_fe_init_fan(&se_fan_watertank); fan_ready = 1; ready = 0; }
            geom = &se_fan_watertank;
        } else {
            se_aris_frame_t fh = {0};
            fh.system_type = S->h.system_type; fh.ping_mode = S->h.ping_mode; fh.large_lens = S->h.large_lens;
            fh.n_beams = S->h.n_beams; fh.n_samples = S->h.n_samples;
            fh.sample_period_us = S->h.sample_period_us; fh.sample_start_delay_us = S->h.sample_start_delay_us;
            fh.sound_speed = S->h.sound_speed; fh.frame_rate = S->h.frame_rate;
            fh.sample_len_m = fh.sample_period_us * 1e-6 * fh.sound_speed / 2.0;
            fh.window_start_m = fh.sample_start_delay_us * 1e-6 * fh.sound_speed / 2.0;
            if (!ready || !se_aris_same_geom(&fh, &cur)) {
                const int have = fh.n_beams <= 512 && se_aris_beam_deg(&fh, ang) == 0;
                se_aris_geom(&fh, have ? ang : NULL, fit, &g);
                if (se_fe_init_polar(&g, fh.n_beams, fh.n_samples, have ? ang : NULL) != 0) {
                    fprintf(stderr, "worker %d: bad polar geometry\n", id);
                    _exit(1);
                }
                geom = &g;
                cur = fh;
                ready = 1;
                fan_ready = 0;
            }
            mode = MODE_POLAR;
        }
        FILE *ms = open_memstream(&line, &line_cap);
        char src[32];
        snprintf(src, sizeof src, "stream:%u", (unsigned)S->h.rec_id);
        o.frame_no = (int)S->h.seq;
        process(&o, mode, data, S->h.n_beams, S->h.n_samples, src, (int)S->h.rec_index, ms);
        fclose(ms);
        const double t_done = now_s();
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '}')) { if (line[--len] == '}') break; }
        line[len] = 0;   /* line without its closing brace */
        const double recv_to_done = t_done - S->recv_mono;
        pthread_mutex_lock(&sh->out_mu);
        printf("%s, \"stream\": {\"seq\": %llu, \"rec_id\": %u, \"rec_index\": %u, \"t_capture_ns\": %llu, \"t_recv_ns\": %llu"
               ", \"wait_ms\": %.3f, \"proc_ms\": %.3f, \"recv_to_done_ms\": %.3f, \"worker\": %d, \"queue_depth\": %d"
               ", \"dropped_total\": %llu, \"gaps_total\": %llu}}\n",
               line, (unsigned long long)S->h.seq, (unsigned)S->h.rec_id, (unsigned)S->h.rec_index,
               (unsigned long long)S->h.t_capture_ns, (unsigned long long)S->t_recv_ns,
               1e3 * (t_start - S->recv_mono), 1e3 * (t_done - t_start), 1e3 * recv_to_done, id, qdepth,
               (unsigned long long)dropped_now, (unsigned long long)gaps_now);
        fflush(stdout);
        pthread_mutex_unlock(&sh->out_mu);

        pthread_mutex_lock(&sh->mu);
        sc_done_t *d = &sh->done[sh->done_head % SC_DONE_RING];
        d->seq = S->h.seq;
        d->t_capture_ns = S->h.t_capture_ns;
        d->recv_to_done_us = (uint32_t)(recv_to_done * 1e6);
        d->proc_us = (uint32_t)((t_done - t_start) * 1e6);
        d->worker = (uint16_t)id;
        d->qdepth = (uint16_t)qdepth;
        sh->done_head++;
        sh->processed++;
        S->state = SC_FREE;
        pthread_cond_broadcast(&sh->cv);
        pthread_mutex_unlock(&sh->mu);
        const char x = 'x';
        if (write(notify_fd, &x, 1) < 0) { /* receiver gone */ }
    }
    free(line);
    _exit(0);
}

/* Send pending ACKs (receiver side). Returns -1 on socket error. */
static int sc_flush_acks(int fd)
{
    sc_done_t batch[64];
    for (;;) {
        int n = 0;
        pthread_mutex_lock(&sh->mu);
        while (n < 64 && sh->done_tail != sh->done_head) batch[n++] = sh->done[sh->done_tail++ % SC_DONE_RING];
        pthread_mutex_unlock(&sh->mu);
        if (!n) return 0;
        for (int k = 0; k < n; k++) {
            uint8_t m[SS_MSG_LEN];
            ss_pack_msg(m, "ACK ");
            ss_put64(m + 4, batch[k].seq);
            ss_put64(m + 12, batch[k].t_capture_ns);
            ss_put32(m + 20, batch[k].recv_to_done_us);
            ss_put32(m + 24, batch[k].proc_us);
            ss_put16(m + 28, batch[k].worker);
            ss_put16(m + 30, batch[k].qdepth);
            if (sc_send_full(fd, m, SS_MSG_LEN)) return -1;
        }
    }
}

static int run_stream(const opts_t *o, const char *hostport, int policy_latest, int workers, int qcap, uint64_t from,
                      const char *name, int fit)
{
    char host[256];
    const char *colon = strrchr(hostport, ':');
    if (!colon || colon == hostport || (size_t)(colon - hostport) >= sizeof host) { fprintf(stderr, "--stream HOST:PORT\n"); return 2; }
    memcpy(host, hostport, (size_t)(colon - hostport));
    host[colon - hostport] = 0;
    if (workers < 1 || workers > SC_MAX_WORKERS || qcap < 1) { fprintf(stderr, "--workers 1..%d, --qcap >= 1\n", SC_MAX_WORKERS); return 2; }

    struct addrinfo hints = {0}, *ai;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, colon + 1, &hints, &ai)) { fprintf(stderr, "%s: cannot resolve\n", hostport); return 1; }
    const int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0 || connect(fd, ai->ai_addr, ai->ai_addrlen)) { perror("connect"); return 1; }
    freeaddrinfo(ai);
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    signal(SIGPIPE, SIG_IGN);

    /* the first header tells the frame size; slots are sized for up to 4096 x 512 */
    slot_bytes = (size_t)4096 * 512;
    n_slots = policy_latest ? workers + 2 : workers + qcap;
    sh = sc_shared_alloc(sizeof *sh);
    slots = sc_shared_alloc(sizeof(sc_slot_t) * (size_t)n_slots);
    slot_data = sc_shared_alloc(slot_bytes * (size_t)n_slots);
    if (!sh || !slots || !slot_data) { fprintf(stderr, "shared memory allocation failed\n"); return 1; }
    pthread_mutexattr_t ma;
    pthread_mutexattr_init(&ma);
    pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&sh->mu, &ma);
    pthread_mutex_init(&sh->out_mu, &ma);
    pthread_condattr_t ca;
    pthread_condattr_init(&ca);
    pthread_condattr_setpshared(&ca, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&sh->cv, &ca);

    int pfd[2];
    if (pipe(pfd)) { perror("pipe"); return 1; }
    fflush(stdout);
    pid_t pid[SC_MAX_WORKERS];
    for (int w = 0; w < workers; w++) {
        pid[w] = fork();
        if (pid[w] < 0) { perror("fork"); return 1; }
        if (pid[w] == 0) { close(fd); close(pfd[0]); sc_worker(w, o, policy_latest, fit, pfd[1]); }
    }
    close(pfd[1]);
    fcntl(pfd[0], F_SETFL, O_NONBLOCK);

    uint8_t m[SS_MSG_LEN];
    ss_pack_msg(m, "HELO");
    snprintf((char *)m + 8, 24, "%s", name ? name : "sonar-edge-run");
    sc_send_full(fd, m, SS_MSG_LEN);
    ss_pack_msg(m, "STRT");
    ss_put64(m + 4, from);
    sc_send_full(fd, m, SS_MSG_LEN);
    fprintf(stderr, "stream: connected to %s, %d worker(s), policy %s, %d slots\n", hostport, workers,
            policy_latest ? "latest" : "queue", n_slots);

    uint64_t expect = from == SS_LIVE ? UINT64_MAX : from;
    const double t_begin = now_s();
    int rc = 0;
    for (;;) {
        struct pollfd pf[2] = {{fd, POLLIN, 0}, {pfd[0], POLLIN, 0}};
        if (poll(pf, 2, 1000) < 0) { if (errno == EINTR) continue; rc = 1; break; }
        if (pf[1].revents) {
            char junk[256];
            while (read(pfd[0], junk, sizeof junk) > 0) { }
            if (sc_flush_acks(fd)) { rc = 1; break; }
        }
        if (!(pf[0].revents & (POLLIN | POLLHUP | POLLERR))) continue;
        uint8_t hb[SS_HDR_LEN];
        ss_hdr_t h;
        const int r = sc_read_full(fd, hb, SS_HDR_LEN);
        if (r) { if (r < 0) rc = 1; break; }
        if (ss_unpack_hdr(hb, &h) || h.payload_len > slot_bytes) { fprintf(stderr, "stream: bad header\n"); rc = 1; break; }
        /* a slot to receive into */
        int s = -1;
        for (;;) {
            pthread_mutex_lock(&sh->mu);
            for (int k = 0; k < n_slots && s < 0; k++) if (slots[k].state == SC_FREE) s = k;
            if (s < 0 && policy_latest) {   /* replace the oldest waiting frame */
                for (int k = 0; k < n_slots; k++)
                    if (slots[k].state == SC_READY && (s < 0 || slots[k].h.seq < slots[s].h.seq)) s = k;
                if (s >= 0) sh->dropped++;
            }
            if (s >= 0) slots[s].state = SC_FILLING;
            pthread_mutex_unlock(&sh->mu);
            if (s >= 0) break;
            /* queue full: stop reading (TCP backpressure) until a worker frees a slot */
            struct pollfd pw = {pfd[0], POLLIN, 0};
            poll(&pw, 1, 5);
            char junk[256];
            while (read(pfd[0], junk, sizeof junk) > 0) { }
            if (sc_flush_acks(fd)) { rc = 1; break; }
        }
        if (rc) break;
        if (sc_read_full(fd, slot_data + (size_t)s * slot_bytes, h.payload_len)) { rc = 1; break; }
        const uint64_t t_recv = sc_real_ns();
        const double recv_mono = now_s();
        pthread_mutex_lock(&sh->mu);
        if (expect != UINT64_MAX && h.seq > expect) sh->gaps += h.seq - expect;
        expect = h.seq + 1;
        slots[s].h = h;
        slots[s].t_recv_ns = t_recv;
        slots[s].recv_mono = recv_mono;
        slots[s].state = SC_READY;
        sh->received++;
        pthread_cond_broadcast(&sh->cv);
        pthread_mutex_unlock(&sh->mu);
    }
    /* server closed: let the workers finish what they have, then stop them */
    pthread_mutex_lock(&sh->mu);
    sh->quit = 1;
    pthread_cond_broadcast(&sh->cv);
    pthread_mutex_unlock(&sh->mu);
    for (int w = 0; w < workers; w++) waitpid(pid[w], NULL, 0);
    sc_flush_acks(fd);
    ss_pack_msg(m, "BYE ");
    sc_send_full(fd, m, SS_MSG_LEN);
    close(fd);
    const double el = now_s() - t_begin;
    fprintf(stderr, "stream: %.1f s, received %llu, processed %llu (%.2f fps), dropped %llu, gaps %llu\n", el,
            (unsigned long long)sh->received, (unsigned long long)sh->processed, sh->processed / (el > 0 ? el : 1),
            (unsigned long long)sh->dropped, (unsigned long long)sh->gaps);
    return rc;
}
