/* tcp_probe.c - TCP latency ping-pong + throughput probe
 *
 * Build: gcc -O2 -pthread -o tcp_probe tcp_probe.c
 *
 * Usage:
 *   tcp_probe server <port> [max_conns]   (exit after max_conns conns closed; 0=forever)
 *   tcp_probe client <ip> <port> lat <nconn> <seconds>
 *   tcp_probe client <ip> <port> bw  <nconn> <seconds>
 *
 * Protocol: client sends 1 mode byte after connect.
 *   'L' -> server echoes every message back (ping-pong)
 *   'B' -> server sinks/discards everything (throughput)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <time.h>

#define LAT_MSG_SIZE 64
#define BW_BUF_SIZE  (1 << 20)      /* 1 MiB */
#define SOCK_BUF_SZ  (4 << 20)      /* 4 MiB */
#define LAT_ITERS    5000

static volatile int g_stop = 0;
/* Explicit SO_SNDBUF/SO_RCVBUF size for bw mode; 0 = leave kernel autotuning
 * (default). Override via TCP_PROBE_BUFSZ env (bytes). NOTE: on some
 * container netns (e.g. the GPU serverless fabric) an explicit SO_RCVBUF
 * produces a ~40-110KB advertised window and collapses throughput; the
 * default autotune path negotiates multi-MB windows. */
static int g_bufsz = 0;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static ssize_t read_full(int fd, void *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, (char *)buf + got, n - got);
        if (r <= 0) return r;
        got += (size_t)r;
    }
    return (ssize_t)got;
}

static ssize_t write_full(int fd, const void *buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, (const char *)buf + sent, n - sent);
        if (w <= 0) return w;
        sent += (size_t)w;
    }
    return (ssize_t)sent;
}

/* ---------------- server ---------------- */

static volatile int g_accepted = 0, g_closed = 0;

static void *server_conn(void *arg) {
    int fd = (int)(intptr_t)arg;
    char mode;
    if (read_full(fd, &mode, 1) != 1) { close(fd); __sync_fetch_and_add(&g_closed, 1); return NULL; }
    if (mode == 'L') {
        char buf[65536];
        for (;;) {
            ssize_t r = read(fd, buf, sizeof(buf));
            if (r <= 0) break;
            if (write_full(fd, buf, (size_t)r) <= 0) break;
        }
    } else { /* 'B' sink */
        if (g_bufsz > 0) {
            int sz = g_bufsz;
            setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
        }
        char *buf = malloc(BW_BUF_SIZE);
        if (buf) {
            while (read(fd, buf, BW_BUF_SIZE) > 0) {}
            free(buf);
        }
    }
    close(fd);
    __sync_fetch_and_add(&g_closed, 1);
    return NULL;
}

static int run_server(int port, int max_conns) {
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(lfd, 128) < 0) { perror("listen"); return 1; }
    fprintf(stderr, "server listening on port %d\n", port);
    for (;;) {
        if (max_conns > 0 && g_accepted >= max_conns) {
            while (g_closed < g_accepted) {
                struct timespec ts = { 0, 50 * 1000 * 1000 };
                nanosleep(&ts, NULL);
            }
            break;
        }
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        __sync_fetch_and_add(&g_accepted, 1);
        pthread_t th;
        if (pthread_create(&th, NULL, server_conn, (void *)(intptr_t)cfd) == 0)
            pthread_detach(th);
        else {
            close(cfd);
            __sync_fetch_and_add(&g_closed, 1);
        }
    }
    close(lfd);
    return 0;
}

/* ---------------- client common ---------------- */

static int connect_to(const char *ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) { close(fd); return -1; }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { close(fd); return -1; }
    return fd;
}

/* ---------------- client lat ---------------- */

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int run_lat(const char *ip, int port, int iters) {
    int fd = connect_to(ip, port);
    if (fd < 0) { perror("connect"); return 1; }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    if (write_full(fd, "L", 1) != 1) { perror("write mode"); close(fd); return 1; }

    char sbuf[LAT_MSG_SIZE], rbuf[LAT_MSG_SIZE];
    memset(sbuf, 0x5a, sizeof(sbuf));
    double *rtt = malloc(sizeof(double) * (size_t)iters);
    if (!rtt) { close(fd); return 1; }

    /* warmup */
    for (int i = 0; i < 100; i++) {
        if (write_full(fd, sbuf, LAT_MSG_SIZE) <= 0) break;
        if (read_full(fd, rbuf, LAT_MSG_SIZE) <= 0) break;
    }
    int done = 0;
    for (int i = 0; i < iters; i++) {
        double t0 = now_us();
        if (write_full(fd, sbuf, LAT_MSG_SIZE) <= 0) break;
        if (read_full(fd, rbuf, LAT_MSG_SIZE) <= 0) break;
        rtt[i] = now_us() - t0;
        done++;
    }
    close(fd);
    if (done == 0) { fprintf(stderr, "lat: no iterations completed\n"); free(rtt); return 1; }
    qsort(rtt, (size_t)done, sizeof(double), cmp_double);
    double sum = 0;
    for (int i = 0; i < done; i++) sum += rtt[i];
    printf("LAT iters=%d avg_us=%.2f p50_us=%.2f p99_us=%.2f min_us=%.2f max_us=%.2f\n",
           done, sum / done, rtt[done / 2], rtt[(int)((double)done * 0.99)],
           rtt[0], rtt[done - 1]);
    free(rtt);
    return 0;
}

/* ---------------- client bw ---------------- */

struct bw_ctx {
    const char *ip;
    int port;
    int idx;
    unsigned long long bytes;
    double elapsed_s;
};

static void *bw_worker(void *arg) {
    struct bw_ctx *ctx = arg;
    int fd = connect_to(ctx->ip, ctx->port);
    if (fd < 0) { perror("connect"); ctx->bytes = 0; return NULL; }
    if (g_bufsz > 0) {
        int sz = g_bufsz;
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
    }
    if (write_full(fd, "B", 1) != 1) { close(fd); return NULL; }
    char *buf = malloc(BW_BUF_SIZE);
    if (!buf) { close(fd); return NULL; }
    memset(buf, 0xa5, BW_BUF_SIZE);
    double t0 = now_us();
    unsigned long long n = 0;
    while (!g_stop) {
        if (write_full(fd, buf, BW_BUF_SIZE) <= 0) break;
        n += BW_BUF_SIZE;
    }
    ctx->elapsed_s = (now_us() - t0) / 1e6;
    ctx->bytes = n;
    free(buf);
    close(fd);
    return NULL;
}

static int run_bw(const char *ip, int port, int nconn, int seconds) {
    pthread_t *ths = calloc((size_t)nconn, sizeof(pthread_t));
    struct bw_ctx *ctxs = calloc((size_t)nconn, sizeof(struct bw_ctx));
    if (!ths || !ctxs) return 1;
    g_stop = 0;
    for (int i = 0; i < nconn; i++) {
        ctxs[i].ip = ip; ctxs[i].port = port; ctxs[i].idx = i;
        if (pthread_create(&ths[i], NULL, bw_worker, &ctxs[i]) != 0) {
            fprintf(stderr, "pthread_create failed at %d\n", i);
            nconn = i; break;
        }
    }
    struct timespec ts = { seconds, 0 };
    nanosleep(&ts, NULL);
    g_stop = 1;
    for (int i = 0; i < nconn; i++) pthread_join(ths[i], NULL);

    double total_gbps = 0, min_t = 1e30;
    unsigned long long total_bytes = 0;
    printf("BW conns=%d seconds=%d", nconn, seconds);
    for (int i = 0; i < nconn; i++) {
        double g = ctxs[i].elapsed_s > 0 ? (double)ctxs[i].bytes * 8.0 / ctxs[i].elapsed_s / 1e9 : 0;
        printf(" c%d=%.3f", i, g);
        total_gbps += g;
        total_bytes += ctxs[i].bytes;
        if (ctxs[i].elapsed_s > 0 && ctxs[i].elapsed_s < min_t) min_t = ctxs[i].elapsed_s;
    }
    printf(" total_gbps=%.3f", total_gbps);
    if (min_t < 1e29 && min_t > 0)
        printf(" agg_gbps=%.3f", (double)total_bytes * 8.0 / min_t / 1e9);
    printf("\n");
    free(ths); free(ctxs);
    return 0;
}

int main(int argc, char **argv) {
    const char *bs = getenv("TCP_PROBE_BUFSZ");
    if (bs) g_bufsz = atoi(bs);
    if (argc >= 3 && strcmp(argv[1], "server") == 0)
        return run_server(atoi(argv[2]), argc >= 4 ? atoi(argv[3]) : 0);
    if (argc == 7 && strcmp(argv[1], "client") == 0) {
        const char *ip = argv[2];
        int port = atoi(argv[3]);
        int nconn = atoi(argv[5]);
        int seconds = atoi(argv[6]);
        if (strcmp(argv[4], "lat") == 0) return run_lat(ip, port, LAT_ITERS);
        if (strcmp(argv[4], "bw") == 0)  return run_bw(ip, port, nconn, seconds);
    }
    fprintf(stderr,
        "usage:\n"
        "  %s server <port> [max_conns]\n"
        "  %s client <ip> <port> lat <nconn> <seconds>\n"
        "  %s client <ip> <port> bw  <nconn> <seconds>\n",
        argv[0], argv[0], argv[0]);
    return 2;
}
