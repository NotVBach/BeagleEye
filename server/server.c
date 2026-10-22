// server/server.c
// Runs ON the BeagleBone Black (Debian, armhf).
// Serves ../www/index.html at "/" and a live MJPEG stream from a
// UVC webcam (Logitech C270) at "/stream".
//
// Build (on the BBB):  gcc -O2 -Wall -o server server.c -lpthread
// Run   (on the BBB):  cd server && ./server            (uses /dev/video0)
//                      ./server /dev/video1             (other device)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <linux/videodev2.h>

#define PORT     8080
#define WIDTH    1280
#define HEIGHT   720
#define FPS      30          // must be a listed rate; check: v4l2-ctl -d /dev/video0 --list-formats-ext
                             // (this C270: MJPEG 640x480 offers 30/24/20/15/10/7.5/5 fps)
#define NBUF     4
#define BOUNDARY "frame"

static char *html;
static long  html_size;

/* Latest JPEG frame, shared between the camera thread and client threads */
static pthread_mutex_t lock     = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  newframe = PTHREAD_COND_INITIALIZER;
static unsigned char  *frame;
static size_t          frame_len, frame_cap;
static unsigned long   frame_seq;

static int xioctl(int fd, unsigned long req, void *arg) {
    int r;
    do r = ioctl(fd, req, arg); while (r < 0 && errno == EINTR);
    return r;
}

static void die(const char *msg) { perror(msg); exit(1); }

/* ---------------- Camera capture (V4L2, MJPEG, mmap) ---------------- */

struct mbuf { void *start; size_t len; };

static void *camera_thread(void *arg) {
    const char *dev = arg;
    int fd = open(dev, O_RDWR);
    if (fd < 0) die(dev);

    // Ask the camera for MJPEG directly: no encoding work on the BBB
    struct v4l2_format fmt = {0};
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = WIDTH;
    fmt.fmt.pix.height      = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) die("VIDIOC_S_FMT");
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
        fprintf(stderr, "%s does not support MJPEG\n", dev);
        exit(1);
    }

    struct v4l2_streamparm parm = {0};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1;
    parm.parm.capture.timeperframe.denominator = FPS;
    xioctl(fd, VIDIOC_S_PARM, &parm);   // best effort

    printf("Camera %s: %ux%u MJPEG @ %u fps\n", dev,
           fmt.fmt.pix.width, fmt.fmt.pix.height,
           parm.parm.capture.timeperframe.denominator);

    struct v4l2_requestbuffers req = {0};
    req.count  = NBUF;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) die("VIDIOC_REQBUFS");

    struct mbuf *bufs = calloc(req.count, sizeof *bufs);
    for (unsigned i = 0; i < req.count; i++) {
        struct v4l2_buffer b = {0};
        b.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index  = i;
        if (xioctl(fd, VIDIOC_QUERYBUF, &b) < 0) die("VIDIOC_QUERYBUF");
        bufs[i].len   = b.length;
        bufs[i].start = mmap(NULL, b.length, PROT_READ | PROT_WRITE,
                             MAP_SHARED, fd, b.m.offset);
        if (bufs[i].start == MAP_FAILED) die("mmap");
        if (xioctl(fd, VIDIOC_QBUF, &b) < 0) die("VIDIOC_QBUF");
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) die("VIDIOC_STREAMON");

    for (;;) {
        struct v4l2_buffer b = {0};
        b.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd, VIDIOC_DQBUF, &b) < 0) die("VIDIOC_DQBUF");

        if (!(b.flags & V4L2_BUF_FLAG_ERROR) && b.bytesused > 0) {
            pthread_mutex_lock(&lock);
            if (b.bytesused > frame_cap) {
                frame     = realloc(frame, b.bytesused);
                frame_cap = b.bytesused;
            }
            memcpy(frame, bufs[b.index].start, b.bytesused);
            frame_len = b.bytesused;
            frame_seq++;
            pthread_cond_broadcast(&newframe);
            pthread_mutex_unlock(&lock);
        }

        if (xioctl(fd, VIDIOC_QBUF, &b) < 0) die("VIDIOC_QBUF");
    }
    return NULL;
}

/* ---------------- HTTP ---------------- */

static int send_all(int fd, const void *buf, size_t len) {
    const char *p = buf;
    while (len) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p   += n;
        len -= n;
    }
    return 0;
}

static void serve_index(int client) {
    char header[160];
    int hlen = snprintf(header, sizeof header,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %ld\r\n"
        "Connection: close\r\n\r\n", html_size);
    if (send_all(client, header, hlen) == 0)
        send_all(client, html, html_size);
}

static void serve_404(int client) {
    const char *r = "HTTP/1.1 404 Not Found\r\n"
                    "Content-Length: 0\r\nConnection: close\r\n\r\n";
    send_all(client, r, strlen(r));
}

static void serve_stream(int client) {
    const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=" BOUNDARY "\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n\r\n";
    if (send_all(client, hdr, strlen(hdr)) < 0) return;

    unsigned char *local = NULL;
    size_t cap = 0, len;
    unsigned long seen = 0;

    for (;;) {
        // Wait for a frame newer than the last one we sent
        pthread_mutex_lock(&lock);
        while (frame_seq == seen)
            pthread_cond_wait(&newframe, &lock);
        seen = frame_seq;
        if (frame_len > cap) { local = realloc(local, frame_len); cap = frame_len; }
        memcpy(local, frame, frame_len);
        len = frame_len;
        pthread_mutex_unlock(&lock);

        char part[128];
        int n = snprintf(part, sizeof part,
            "--" BOUNDARY "\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %zu\r\n\r\n", len);
        if (send_all(client, part, n) < 0 ||
            send_all(client, local, len) < 0 ||
            send_all(client, "\r\n", 2) < 0)
            break;   // browser closed the tab / disconnected
    }
    free(local);
}

static void *client_thread(void *arg) {
    int client = (int)(intptr_t)arg;

    // Drop clients that stop reading instead of blocking forever
    struct timeval tv = { .tv_sec = 5, .tv_usec = 0 };
    setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    char req[2048];
    ssize_t n = recv(client, req, sizeof req - 1, 0);
    if (n > 0) {
        req[n] = '\0';
        if (strncmp(req, "GET /stream", 11) == 0)
            serve_stream(client);
        else if (strncmp(req, "GET / ", 6) == 0 ||
                 strncmp(req, "GET /index.html", 15) == 0)
            serve_index(client);
        else
            serve_404(client);
    }
    close(client);
    return NULL;
}

int main(int argc, char **argv) {
    const char *dev = argc > 1 ? argv[1] : "/dev/video0";

    // Load HTML file
    FILE *f = fopen("www/index.html", "rb");
    if (!f) { perror("www/index.html"); return 1; }
    fseek(f, 0, SEEK_END);
    html_size = ftell(f);
    rewind(f);
    html = malloc(html_size);
    if (fread(html, 1, html_size, f) != (size_t)html_size) { perror("fread"); return 1; }
    fclose(f);

    signal(SIGPIPE, SIG_IGN);

    // Start camera capture
    pthread_t cam;
    if (pthread_create(&cam, NULL, camera_thread, (void *)dev) != 0) die("pthread_create");
    pthread_detach(cam);

    // Open TCP socket
    int server = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof opt);

    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(PORT);

    if (bind(server, (struct sockaddr *)&addr, sizeof addr) < 0) die("bind");
    listen(server, 20);
    printf("Open on http://<beaglebone-ip>:%d/\n", PORT);

    while (1) {
        int client = accept(server, NULL, NULL);
        if (client < 0) continue;
        pthread_t t;
        if (pthread_create(&t, NULL, client_thread, (void *)(intptr_t)client) != 0) {
            close(client);
            continue;
        }
        pthread_detach(t);
    }
}