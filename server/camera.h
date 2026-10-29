#ifndef CAMERA_H
#define CAMERA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct camera_config {
    const char *device;
    unsigned    width;
    unsigned    height;
    unsigned    fps;
};

int camera_start(const struct camera_config *cfg);

size_t camera_wait_frame(unsigned long *seq, unsigned char **buf, size_t *cap);

#ifdef __cplusplus
}
#endif

#endif