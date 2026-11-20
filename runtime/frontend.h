/* sonar_edge front end: sonar frame -> network input, and network coordinates
 * -> sonar coordinates. C11, no dependencies, integer remap at run time.
 *
 * Two input kinds, both turned into the network's se_input_height() x
 * se_input_width() uint8 frame by one precomputed lookup table:
 *   fan    the Cartesian, fan-shaped image the sonar software (and the
 *          dataset) produces, g->width x g->height. The resize to the
 *          network resolution is bit-exact with the training pipeline
 *          (PIL Image.BOX in sonar_edge.dataset), so accuracy is unchanged.
 *   polar  raw beam data, n_samples rows (range bins, near to far) x n_beams
 *          columns (beam 0 = leftmost, -fov/2), rendered straight to the
 *          network resolution with the fan geometry g.
 *
 * Fan geometry: the sonar sits at the apex (apex_u, apex_v) in fan-image
 * pixels (it may lie outside the image); bearing 0 points from the apex
 * towards the far edge, positive to the right of the image; range grows with
 * distance from the apex. m_per_px converts pixels to metres (0 = unknown,
 * then ranges are reported in pixels).
 */
#pragma once
#include <stdint.h>

typedef struct {
    int width, height;          /* fan image size, pixels */
    float apex_u, apex_v;       /* sonar position, fan-image pixels */
    float fov_deg;              /* total field of view, degrees */
    float r_min_px, r_max_px;   /* near and far edge of the fan, pixels from the apex */
    float m_per_px;             /* metres per fan-image pixel, 0 if unknown */
} se_fan_geom_t;

/* Geometry of the watertank dataset's 320x480 fan images (measured from the
 * images; m_per_px from the ARIS range setting if known, else 0). */
extern const se_fan_geom_t se_fan_watertank;

/* Front end state: built once, then run per frame. Static storage, one front
 * end at a time, not thread-safe. Return 0 on success, -1 on bad arguments. */
int se_fe_init_fan(const se_fan_geom_t *g);
int se_fe_init_polar(const se_fan_geom_t *g, int n_beams, int n_samples,
                     const float *beam_deg /* n_beams angles, or NULL for uniform over fov */);
/* src: fan image (g->height x g->width) or polar frame (n_samples x n_beams);
 * dst: se_input_height() x se_input_width(). */
void se_fe_run(const uint8_t *src, uint8_t *dst);
/* Input size the current front end expects, in bytes. */
int se_fe_src_bytes(void);

/* Network-input pixel (x, y) -> fan-image pixel (u, v), for the geometry the
 * front end was initialised with. */
void se_fe_to_fan(float x, float y, float *u, float *v);
/* Fan-image pixel -> range (metres, or pixels if m_per_px == 0) and bearing
 * (degrees, 0 = straight ahead, positive right). */
void se_fan_to_polar(const se_fan_geom_t *g, float u, float v, float *range, float *bearing_deg);
