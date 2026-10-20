#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>

int main(void) {
    // Load HTML file
    FILE *f = fopen("../www/index.html", "rb");
    if (!f) { perror("www/index.html"); return 1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char *html = malloc(size);
    fread(html, 1, size, f);
    fclose(f);

    signal(SIGPIPE, SIG_IGN);

    // Open tcp socket
    int server = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof opt);

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8080);

    if (bind(server, (struct sockaddr *)&addr, sizeof addr) < 0) { perror("bind"); return 1; }
    listen(server, 20);
    printf("Open on http://localhost:8080/\n");

    char header[128];
    int hlen = snprintf(header, sizeof header,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %ld\r\n"
        "Connection: close\r\n\r\n", size);

    char request[4096];
    while(1) {
        int client = accept(server, NULL, NULL);
        if (client < 0) continue;
        recv(client, request, sizeof request, 0);
        send(client, header, hlen, 0);
        send(client, html, size, 0);
        close(client);
    }
}