#ifndef VISION_H
#define VISION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vision_config {
    const char *detect_model;
    const char *recog_model;
    const char *faces_dir;
    int         detect_width;   /* detector input size, multiples of 32 */
    int         detect_height;
    float       score_threshold;
    float       match_threshold;
    float       recog_interval;
};

/* Load the models and known faces, then start the detection thread. Returns 0 or -1. */
int vision_start(const struct vision_config *cfg);

/* Wait up to 2 s for a result newer than *seq and copy its JSON into buf. Returns its length, 0 on timeout. */
size_t vision_wait_result(unsigned long *seq, char *buf, size_t size);

/* Copy a JSON array of the known face photos ({"file","name"}) into buf. Returns its length, 0 if it does not fit. */
size_t vision_list_faces(char *buf, size_t size);

/* Rename known photo `file` in faces_dir to person `name` (letters, digits, _ and -). Returns 0 or -1. */
int vision_rename_face(const char *file, const char *name);

/* Delete known photo `file` from faces_dir and stop matching it. Returns 0 or -1. */
int vision_delete_face(const char *file);

#ifdef __cplusplus
}
#endif

#endif
