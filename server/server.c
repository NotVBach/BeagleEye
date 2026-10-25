#include "camera.h"

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

static char *html;
static long  html_size;

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

/* Load the page, start the camera, then accept clients on PORT */
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