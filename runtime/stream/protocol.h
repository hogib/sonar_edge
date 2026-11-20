/* sonar-stream wire protocol (SeedLink-like: TCP, sequence-numbered frame
 * packets, resume from a sequence number). All integers little-endian.
 *
 * Client -> server, fixed 32-byte messages:
 *   "HELO"  u32 0, name[24]                  identify (free text, for the server log)
 *   "STRT"  u64 from_seq (SS_LIVE = newest)  start streaming; resume from from_seq if it
 *                                            is still in the server's ring, else from the oldest
 *   "ACK "  u64 seq, u64 t_capture_ns (echo), u32 recv_to_done_us, u32 proc_us,
 *           u16 worker, u16 queue_depth      one per processed frame
 *   "BYE "                                   close
 * Server -> client, per frame: a 64-byte header, then n_beams * n_samples bytes.
 *   header: magic, version, header length, seq, capture and send times, recording id
 *   and frame index, geometry (beams, samples, ARIS system/ping mode/lens, sample
 *   period and start delay, sound speed, frame rate), flags.
 *   payload: ARIS beam data, n_samples rows (near to far) x n_beams (beam 0 left-most),
 *            or (flags & SS_FAN) a fan image n_samples rows x n_beams columns.
 */
#pragma once
#include <stdint.h>
#include <string.h>

#define SS_MAGIC 0x534e5253u      /* "SRNS" as little-endian bytes 'S','R','N','S' */
#define SS_VERSION 1
#define SS_HDR_LEN 64
#define SS_MSG_LEN 32
#define SS_LIVE UINT64_MAX
#define SS_FAN 1u                 /* payload is a fan image, not beam data */

typedef struct {
    uint64_t seq;                 /* stream sequence number, 0, 1, 2, ... */
    uint64_t t_capture_ns;        /* server CLOCK_REALTIME at the frame's nominal capture time */
    uint64_t t_send_ns;           /* server CLOCK_REALTIME when sending started */
    uint32_t rec_index;           /* frame index within the recording */
    uint16_t rec_id;              /* which recording (server argument order) */
    uint16_t n_beams, n_samples;
    uint8_t system_type, ping_mode, large_lens, flags;
    uint32_t sample_period_us, sample_start_delay_us;
    float sound_speed, frame_rate;
    uint32_t payload_len;         /* derived: n_beams * n_samples */
} ss_hdr_t;

static inline void ss_put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline void ss_put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static inline void ss_put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static inline uint16_t ss_get16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t ss_get32(const uint8_t *p) { uint32_t v = 0; for (int i = 3; i >= 0; i--) v = (v << 8) | p[i]; return v; }
static inline uint64_t ss_get64(const uint8_t *p) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = (v << 8) | p[i]; return v; }
static inline void ss_putf(uint8_t *p, float f) { uint32_t v; memcpy(&v, &f, 4); ss_put32(p, v); }
static inline float ss_getf(const uint8_t *p) { uint32_t v = ss_get32(p); float f; memcpy(&f, &v, 4); return f; }

static inline void ss_pack_hdr(uint8_t *b, const ss_hdr_t *h)
{
    memset(b, 0, SS_HDR_LEN);
    ss_put32(b, SS_MAGIC);
    ss_put16(b + 4, SS_VERSION);
    ss_put16(b + 6, SS_HDR_LEN);
    ss_put64(b + 8, h->seq);
    ss_put64(b + 16, h->t_capture_ns);
    ss_put64(b + 24, h->t_send_ns);
    ss_put32(b + 32, h->rec_index);
    ss_put16(b + 36, h->rec_id);
    ss_put16(b + 38, h->n_beams);
    ss_put16(b + 40, h->n_samples);
    b[42] = h->system_type; b[43] = h->ping_mode; b[44] = h->large_lens; b[45] = h->flags;
    ss_put32(b + 46, h->sample_period_us);
    ss_put32(b + 50, h->sample_start_delay_us);
    ss_putf(b + 54, h->sound_speed);
    ss_putf(b + 58, h->frame_rate);
    /* bytes 62..63 reserved (0); the payload length is n_beams * n_samples and is not sent */
}

static inline int ss_unpack_hdr(const uint8_t *b, ss_hdr_t *h)
{
    if (ss_get32(b) != SS_MAGIC || ss_get16(b + 4) != SS_VERSION || ss_get16(b + 6) != SS_HDR_LEN) return -1;
    h->seq = ss_get64(b + 8);
    h->t_capture_ns = ss_get64(b + 16);
    h->t_send_ns = ss_get64(b + 24);
    h->rec_index = ss_get32(b + 32);
    h->rec_id = ss_get16(b + 36);
    h->n_beams = ss_get16(b + 38);
    h->n_samples = ss_get16(b + 40);
    h->system_type = b[42]; h->ping_mode = b[43]; h->large_lens = b[44]; h->flags = b[45];
    h->sample_period_us = ss_get32(b + 46);
    h->sample_start_delay_us = ss_get32(b + 50);
    h->sound_speed = ss_getf(b + 54);
    h->frame_rate = ss_getf(b + 58);
    h->payload_len = (uint32_t)h->n_beams * h->n_samples;
    return 0;
}

/* Client messages. */
static inline void ss_pack_msg(uint8_t *b, const char tag[4])
{
    memset(b, 0, SS_MSG_LEN);
    memcpy(b, tag, 4);
}
