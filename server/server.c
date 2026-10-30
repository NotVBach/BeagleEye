#include "camera.h"
#include "vision.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>

#define PORT       8080
#define HTML_PATH  "www/index.html"
#define BOUNDARY   "frame"

#define CAM_WIDTH  1280
#define CAM_HEIGHT 720
#define CAM_FPS    30

#define DETECT_MODEL    "models/face_detection_yunet_2023mar.tflite"
#define RECOG_MODEL     "models/face_recognition_sface_2021dec.tflite"
#define FACES_DIR       "faces"
#define DETECT_WIDTH    320    /* a 16:9 frame scaled to 320 wide is 180 high, padded to 192 */
#define DETECT_HEIGHT   192
#define SCORE_THRESHOLD 0.7f
#define MATCH_THRESHOLD 0.363f
#define RECOG_INTERVAL  2.0f

static char *html;
static long  html_size;
static int   vision_on;

/* Send the whole buffer, retrying on partial sends. Returns 0 or -1 */
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

/* Send index.html */
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

/* Send an empty 404 response */
static void serve_404(int client) {
    const char *r = "HTTP/1.1 404 Not Found\r\n"
                    "Content-Length: 0\r\nConnection: close\r\n\r\n";
    send_all(client, r, strlen(r));
}

/* Send a small response with the given status line, type and body */
static void serve_body(int client, const char *status, const char *type, const void *body, size_t len) {
    char header[200];
    int hlen = snprintf(header, sizeof header,
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n", status, type, len);
    if (send_all(client, header, hlen) == 0 && len)
        send_all(client, body, len);
}

static int query_value(const char *req, const char *key, char *out, size_t size) {
    const char *line = strchr(req, '\n');
    const char *q    = strchr(req, '?');
    const char *end  = q ? strchr(q, ' ') : NULL;   /* the space before "HTTP/1.1" */
    size_t klen = strlen(key);
    if (!q || !end || (line && end > line)) return -1;
    for (const char *p = q + 1; p < end; p = strchr(p, '&') ? strchr(p, '&') + 1 : end) {
        if (strncmp(p, key, klen) || p[klen] != '=') continue;
        size_t n = strcspn(p + klen + 1, "& ");
        if (n == 0 || n >= size || strspn(p + klen + 1, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") < n)
            return -1;
        memcpy(out, p + klen + 1, n);
        out[n] = '\0';
        return 0;
    }
    return -1;
}

/* Send the known faces as JSON */
static void serve_face_list(int client) {
    static char json[65536];
    static pthread_mutex_t json_lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&json_lock);
    size_t len = vision_list_faces(json, sizeof json);
    serve_body(client, "200 OK", "application/json", len ? json : "[]", len ? len : 2);
    pthread_mutex_unlock(&json_lock);
}

/* Rename a known face: POST /api/rename?file=new-123.jpg&name=anna */
static void serve_rename(int client, const char *req) {
    char file[80], name[80];
    if (query_value(req, "file", file, sizeof file) < 0 || query_value(req, "name", name, sizeof name) < 0 ||
        vision_rename_face(file, name) < 0)
        serve_body(client, "400 Bad Request", "text/plain", "rename failed", 13);
    else
        serve_body(client, "200 OK", "text/plain", "ok", 2);
}

/* Send one photo from FACES_DIR: GET /faces/<file> */
static void serve_face_photo(int client, const char *req) {
    const char *name = req + strlen("GET /faces/");
    size_t n = strcspn(name, " ?");
    char path[128];
    /* Plain file names only, so a request cannot reach outside the folder */
    if (n == 0 || n > 80 || name[0] == '.' ||
        strspn(name, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") < n) {
        serve_404(client);
        return;
    }
    snprintf(path, sizeof path, FACES_DIR "/%.*s", (int)n, name);
    FILE *f = fopen(path, "rb");
    if (!f) { serve_404(client); return; }
    char buf[256 * 1024];   /* face crops are a few tens of KB */
    size_t len = fread(buf, 1, sizeof buf, f);
    int whole = feof(f);
    fclose(f);
    if (!whole) { serve_404(client); return; }
    serve_body(client, "200 OK", "image/jpeg", buf, len);
}

/* Send camera frames as an MJPEG stream until the client disconnects */
static void serve_stream(int client) {
    const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=" BOUNDARY "\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n\r\n";
    if (send_all(client, hdr, strlen(hdr)) < 0) return;

    unsigned char *jpeg = NULL;
    size_t cap = 0;
    unsigned long seq = 0;

    for (;;) {
        size_t len = camera_wait_frame(&seq, &jpeg, &cap);
        if (len == 0) break;

        char part[128];
        int n = snprintf(part, sizeof part,
            "--" BOUNDARY "\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %zu\r\n\r\n", len);
        if (send_all(client, part, n) < 0 ||
            send_all(client, jpeg, len) < 0 ||
            send_all(client, "\r\n", 2) < 0)
            break;
    }
    free(jpeg);
}

/* Send face results as Server-Sent Events until the client disconnects */
static void serve_events(int client) {
    if (!vision_on) { serve_404(client); return; }

    const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n";
    if (send_all(client, hdr, strlen(hdr)) < 0) return;

    char json[8192];
    unsigned long seq = 0;

    for (;;) {
        size_t len = vision_wait_result(&seq, json, sizeof json);
        int fail;
        if (len == 0)
            fail = send_all(client, ": ping\n\n", 8);   /* keep-alive when no new result */
        else
            fail = send_all(client, "data: ", 6) < 0 ||
                   send_all(client, json, len) < 0 ||
                   send_all(client, "\n\n", 2) < 0;
        if (fail) break;
    }
}

/* Per-client thread: read the request and route it */
static void *client_thread(void *arg) {
    int client = (int)(intptr_t)arg;

    struct timeval tv = { .tv_sec = 5, .tv_usec = 0 };
    setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    char req[2048];
    ssize_t n = recv(client, req, sizeof req - 1, 0);
    if (n > 0) {
        req[n] = '\0';
        if (strncmp(req, "GET /stream", 11) == 0)
            serve_stream(client);
        else if (strncmp(req, "GET /events", 11) == 0)
            serve_events(client);
        else if (strncmp(req, "GET /api/faces", 14) == 0)
            serve_face_list(client);
        else if (strncmp(req, "POST /api/rename?", 17) == 0)
            serve_rename(client, req);
        else if (strncmp(req, "GET /faces/", 11) == 0)
            serve_face_photo(client, req);
        else if (strncmp(req, "GET / ", 6) == 0 ||
                 strncmp(req, "GET /index.html", 15) == 0)
            serve_index(client);
        else
            serve_404(client);
    }
    close(client);
    return NULL;
}

/* Read the HTML file into memory */
static int load_html(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END);
    html_size = ftell(f);
    rewind(f);
    html = malloc(html_size);
    if (!html || fread(html, 1, html_size, f) != (size_t)html_size) {
        perror("read html");
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

/* Load the page, start the camera and face detection, then accept clients on PORT */
int main(int argc, char **argv) {
    if (load_html(HTML_PATH) < 0) return 1;

    signal(SIGPIPE, SIG_IGN);

    struct camera_config cam = {
        .device = argc > 1 ? argv[1] : "/dev/video0",
        .width  = CAM_WIDTH,
        .height = CAM_HEIGHT,
        .fps    = CAM_FPS,
    };
    if (camera_start(&cam) < 0) return 1;

    /* If the models are missing, face detection is off but the video still works */
    struct vision_config vis = {
        .detect_model    = DETECT_MODEL,
        .recog_model     = RECOG_MODEL,
        .faces_dir       = FACES_DIR,
        .detect_width    = DETECT_WIDTH,
        .detect_height   = DETECT_HEIGHT,
        .score_threshold = SCORE_THRESHOLD,
        .match_threshold = MATCH_THRESHOLD,
        .recog_interval  = RECOG_INTERVAL,
    };
    vision_on = vision_start(&vis) == 0;

    int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) { perror("socket"); return 1; }
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof opt);

    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(PORT);

    if (bind(server, (struct sockaddr *)&addr, sizeof addr) < 0) { perror("bind"); return 1; }
    listen(server, 20);
    printf("Server running on port %d\n", PORT);
    fflush(stdout);

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
