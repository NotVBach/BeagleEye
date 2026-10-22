#include "camera.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#define NBUF 4

struct mbuf { void *start; size_t len; };

static int          cam_fd = -1;
static struct mbuf *bufs;
static unsigned     nbufs;

static pthread_mutex_t lock     = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  newframe = PTHREAD_COND_INITIALIZER;
static unsigned char  *frame;
static size_t          frame_len, frame_cap;
static unsigned long   frame_seq;

/* ioctl that retries when interrupted by a signal */
static int xioctl(int fd, unsigned long req, void *arg) {
    int r;
    do r = ioctl(fd, req, arg); while (r < 0 && errno == EINTR);
    return r;
}

/* Capture thread: dequeue each frame, store it as the latest frame, requeue the buffer */
static void *capture_loop(void *arg) {
    (void)arg;
    for (;;) {
        struct v4l2_buffer b = {0};
        b.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        if (xioctl(cam_fd, VIDIOC_DQBUF, &b) < 0) { perror("VIDIOC_DQBUF"); exit(1); }

        if (!(b.flags & V4L2_BUF_FLAG_ERROR) && b.bytesused > 0) {
            pthread_mutex_lock(&lock);
            if (b.bytesused > frame_cap) {
                unsigned char *p = realloc(frame, b.bytesused);
                if (p) { frame = p; frame_cap = b.bytesused; }
            }
            if (b.bytesused <= frame_cap) {
                memcpy(frame, bufs[b.index].start, b.bytesused);
                frame_len = b.bytesused;
                frame_seq++;
                pthread_cond_broadcast(&newframe);
            }
            pthread_mutex_unlock(&lock);
        }

        if (xioctl(cam_fd, VIDIOC_QBUF, &b) < 0) { perror("VIDIOC_QBUF"); exit(1); }
    }
    return NULL;
}

/* Open the camera, set MJPEG size/fps, map buffers and start the capture thread */
int camera_start(const struct camera_config *cfg) {
    cam_fd = open(cfg->device, O_RDWR);
    if (cam_fd < 0) { perror(cfg->device); return -1; }

    struct v4l2_format fmt = {0};
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = cfg->width;
    fmt.fmt.pix.height      = cfg->height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;
    if (xioctl(cam_fd, VIDIOC_S_FMT, &fmt) < 0) { perror("VIDIOC_S_FMT"); goto fail; }
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
        fprintf(stderr, "%s does not support MJPEG\n", cfg->device);
        goto fail;
    }
    if (fmt.fmt.pix.width != cfg->width || fmt.fmt.pix.height != cfg->height)
        fprintf(stderr, "Note: %ux%u not supported, driver chose %ux%u\n",
                cfg->width, cfg->height, fmt.fmt.pix.width, fmt.fmt.pix.height);

    struct v4l2_streamparm parm = {0};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1;
    parm.parm.capture.timeperframe.denominator = cfg->fps;
    if (xioctl(cam_fd, VIDIOC_S_PARM, &parm) < 0)
        perror("VIDIOC_S_PARM (keeping default fps)");

    const struct v4l2_fract *tpf = &parm.parm.capture.timeperframe;
    double fps = tpf->numerator ? (double)tpf->denominator / tpf->numerator : 0;
    printf("Camera %s: %ux%u MJPEG @ %.1f fps\n", cfg->device,
           fmt.fmt.pix.width, fmt.fmt.pix.height, fps);

    struct v4l2_requestbuffers req = {0};
    req.count  = NBUF;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(cam_fd, VIDIOC_REQBUFS, &req) < 0) { perror("VIDIOC_REQBUFS"); goto fail; }

    nbufs = req.count;
    bufs  = calloc(nbufs, sizeof *bufs);
    if (!bufs) { perror("calloc"); goto fail; }

    for (unsigned i = 0; i < nbufs; i++) {
        struct v4l2_buffer b = {0};
        b.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index  = i;
        if (xioctl(cam_fd, VIDIOC_QUERYBUF, &b) < 0) { perror("VIDIOC_QUERYBUF"); goto fail; }
        bufs[i].len   = b.length;
        bufs[i].start = mmap(NULL, b.length, PROT_READ | PROT_WRITE,
                             MAP_SHARED, cam_fd, b.m.offset);
        if (bufs[i].start == MAP_FAILED) { perror("mmap"); goto fail; }
        if (xioctl(cam_fd, VIDIOC_QBUF, &b) < 0) { perror("VIDIOC_QBUF"); goto fail; }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cam_fd, VIDIOC_STREAMON, &type) < 0) { perror("VIDIOC_STREAMON"); goto fail; }

    pthread_t t;
    if (pthread_create(&t, NULL, capture_loop, NULL) != 0) {
        fprintf(stderr, "pthread_create failed\n");
        goto fail;
    }
    pthread_detach(t);
    return 0;

fail:
    close(cam_fd);
    cam_fd = -1;
    return -1;
}

/* Wait for a frame newer than *seq and copy it into the caller's buffer */
size_t camera_wait_frame(unsigned long *seq, unsigned char **buf, size_t *cap) {
    pthread_mutex_lock(&lock);
    while (frame_seq == *seq)
        pthread_cond_wait(&newframe, &lock);
    *seq = frame_seq;

    if (frame_len > *cap) {
        unsigned char *p = realloc(*buf, frame_len);
        if (!p) { pthread_mutex_unlock(&lock); return 0; }
        *buf = p;
        *cap = frame_len;
    }
    memcpy(*buf, frame, frame_len);
    size_t len = frame_len;
    pthread_mutex_unlock(&lock);
    return len;
}