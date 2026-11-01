/* kvshape_xport.c - KV-cache handoff block-shape transfer benchmark (TCP)
 *
 * Build: gcc -O2 -pthread -o kvshape_xport kvshape_xport.c
 *
 *   kvshape_xport server <port>
 *   kvshape_xport client <ip> <port> -m A|B -n <conns> -w <0|1> -p FP8|FP4
 *
 * Block plan (FP8: 31934 blocks, 531,448,576 B; FP4: 487,408,384 B),
 * order: state/tail blocks first (hca_state, csa_state, indexer_state,
 * swa_kv), then csa_kv, indexer_kv, hca_kv. Blocks striped id%N per conn.
 * Mode A: one message per block (16B hdr + payload). Mode B: aggregate
 * consecutive blocks into ~1MiB batches. Warm: 24MiB dummy per conn first.
 * Server writes into preallocated region, verifies pattern on 1% sampled
 * blocks + last block; per-conn DONE -> 1-byte ack. Client reports total
 * time (first payload byte -> last ack), effective Gbps, 200ms rate CSV.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define M_BLK   0x4B564C31u
#define M_BATCH 0x4B564232u
#define M_DUMMY 0x44554D59u
#define M_DONE  0x444F4E45u
#define MAX_BLOCKS 40000
#define BATCH_SZ (1 << 20)
#define WARM_PER_CONN (24 << 20)
#define REGION_SZ (600ULL << 20)

struct blk { uint32_t tag, id, len; uint64_t off; };
static struct blk g_plan[MAX_BLOCKS];
static int g_nblocks;
static uint64_t g_total;

enum { T_CSA_KV, T_HCA_KV, T_IDX_KV, T_IDX_ST, T_CSA_ST, T_HCA_ST, T_SWA_KV };

static uint32_t plan_size(int fp4, int tag) {
    switch (tag) {
    case T_CSA_KV: return 37440;
    case T_HCA_KV: return 1728;
    case T_IDX_KV: return fp4 ? 4352 : 8448;
    case T_IDX_ST: return 16384;
    case T_CSA_ST: return 65536;
    case T_HCA_ST: return 524288;
    case T_SWA_KV: return 74880;
    }
    return 0;
}

static void gen_plan(int fp4) {
    static const int order[][2] = {
        {T_HCA_ST, 20}, {T_CSA_ST, 42}, {T_IDX_ST, 42}, {T_SWA_KV, 86},
        {T_CSA_KV, 10752}, {T_IDX_KV, 10752}, {T_HCA_KV, 10240},
    };
    g_nblocks = 0; g_total = 0;
    for (unsigned g = 0; g < sizeof(order) / sizeof(order[0]); g++) {
        uint32_t sz = plan_size(fp4, order[g][0]);
        for (int i = 0; i < order[g][1]; i++) {
            struct blk *b = &g_plan[g_nblocks];
            b->tag = order[g][0]; b->id = g_nblocks; b->len = sz; b->off = g_total;
            g_total += sz; g_nblocks++;
        }
    }
}

static inline uint8_t pat(uint32_t id, uint32_t j) {
    return (uint8_t)(((id * 0x9E3779B1u) ^ (j * 0x85EBCA6Bu)) >> 13);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
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

static uint8_t *g_region;

struct srv_conn { int fd; };

static int verify_block(uint32_t id, const uint8_t *data, uint32_t len) {
    for (uint32_t j = 0; j < len; j += 97) /* sample stride inside block */
        if (data[j] != pat(id, j)) return -1;
    return 0;
}

static void *srv_worker(void *arg) {
    int fd = (int)(intptr_t)arg;
    uint32_t magic;
    uint64_t nblk = 0, nbytes = 0, nver = 0, nfail = 0;
    static __thread uint8_t scratch[BATCH_SZ + 64];
    int done = 0;
    while (!done) {
        if (read_full(fd, &magic, 4) != 4) break;
        if (magic == M_DUMMY) {
            uint32_t len;
            if (read_full(fd, &len, 4) != 4) break;
            uint32_t left = len;
            while (left) {
                uint32_t c = left > sizeof(scratch) ? sizeof(scratch) : left;
                if (read_full(fd, scratch, c) != (ssize_t)c) { done = 1; break; }
                left -= c;
            }
        } else if (magic == M_BLK) {
            uint32_t hdr[3];
            if (read_full(fd, hdr, 12) != 12) break;
            uint32_t id = hdr[1], len = hdr[2];
            struct blk *b = &g_plan[id];
            if (read_full(fd, g_region + b->off, len) != (ssize_t)len) break;
            nblk++; nbytes += len;
            if (id % 100 == 0 || id == (uint32_t)g_nblocks - 1) {
                nver++;
                if (verify_block(id, g_region + b->off, len)) nfail++;
            }
        } else if (magic == M_BATCH) {
            uint32_t blen;
            if (read_full(fd, &blen, 4) != 4) break;
            if (read_full(fd, scratch, blen) != (ssize_t)blen) break;
            uint32_t p = 0;
            while (p < blen) {
                uint32_t *h = (uint32_t *)(scratch + p);
                uint32_t id = h[1], len = h[2];
                struct blk *b = &g_plan[id];
                memcpy(g_region + b->off, scratch + p + 12, len);
                nblk++; nbytes += len;
                if (id % 100 == 0 || id == (uint32_t)g_nblocks - 1) {
                    nver++;
                    if (verify_block(id, g_region + b->off, len)) nfail++;
                }
                p += 12 + len;
            }
        } else if (magic == M_DONE) {
            uint32_t hdr[2];
            read_full(fd, hdr, 8);
            uint8_t ack = (nfail == 0) ? 1 : 0;
            write_full(fd, &ack, 1);
            printf("SRVCONN blocks=%llu bytes=%llu verified=%llu fail=%llu\n",
                   (unsigned long long)nblk, (unsigned long long)nbytes,
                   (unsigned long long)nver, (unsigned long long)nfail);
            done = 1;
        } else break;
    }
    close(fd);
    return NULL;
}

static int run_server(int port) {
    gen_plan(0);
    g_region = malloc(REGION_SZ);
    if (!g_region) { fprintf(stderr, "region alloc failed\n"); return 1; }
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct timeval tv = {120, 0};
    setsockopt(lfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port),
                               .sin_addr.s_addr = htonl(INADDR_ANY)};
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) || listen(lfd, 128)) { perror("bind"); return 1; }
    fprintf(stderr, "kvserver ready port=%d\n", port);
    for (;;) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) { if (errno == EINTR) continue; break; } /* timeout -> exit */
        pthread_t th;
        if (pthread_create(&th, NULL, srv_worker, (void *)(intptr_t)cfd) == 0)
            pthread_detach(th);
        else close(cfd);
    }
    return 0;
}

/* ---------------- client ---------------- */

static uint8_t *g_content;
static atomic_ulong g_sent;
static volatile int g_run_done = 0;
static double g_t0;
static pthread_barrier_t g_barrier;

struct cli_arg {
    struct sockaddr_in dst;
    int tid, nconn, warm;
    char mode;
    double t_ack;
    uint64_t bytes;
};

static void *stats_thread(void *arg) {
    (void)arg;
    double lt = g_t0;
    unsigned long lb = 0;
    while (!g_run_done) {
        struct timespec ts = {0, 200 * 1000 * 1000};
        nanosleep(&ts, NULL);
        double t = now();
        unsigned long b = atomic_load(&g_sent);
        printf("KVCSV %llu,%.3f\n", (unsigned long long)((t - g_t0) * 1000),
               (double)(b - lb) * 8 / (t - lt) / 1e9);
        lt = t; lb = b;
    }
    return NULL;
}

static void *cli_worker(void *arg) {
    struct cli_arg *a = arg;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    if (connect(fd, (struct sockaddr *)&a->dst, sizeof(a->dst))) { perror("connect"); return NULL; }

    if (a->warm) {
        static __thread uint8_t dbuf[1 << 20];
        memset(dbuf, 0xd0, sizeof(dbuf));
        *(uint32_t *)dbuf = M_DUMMY;
        *(uint32_t *)(dbuf + 4) = (1 << 20) - 8;
        for (int i = 0; i < WARM_PER_CONN >> 20; i++)
            if (write_full(fd, dbuf, sizeof(dbuf)) <= 0) goto out;
    }
    pthread_barrier_wait(&g_barrier); /* sync: t0 = release of all conns */
    /* timed phase */
    uint64_t myb = 0;
    if (a->mode == 'A') {
        for (int i = a->tid; i < g_nblocks; i += a->nconn) {
            struct blk *b = &g_plan[i];
            uint32_t hdr[4] = {M_BLK, b->tag, b->id, b->len};
            struct iovec iov[2] = {{hdr, 16}, {g_content + b->off, b->len}};
            ssize_t w = writev(fd, iov, 2);
            if (w <= 0) goto out;
            myb += b->len;
            atomic_fetch_add(&g_sent, b->len);
        }
    } else {
        static __thread uint8_t batch[BATCH_SZ + 64];
        uint32_t used = 8;
        *(uint32_t *)batch = M_BATCH;
        for (int i = a->tid; i < g_nblocks; i += a->nconn) {
            struct blk *b = &g_plan[i];
            if (used + 12 + b->len > sizeof(batch)) {
                *(uint32_t *)(batch + 4) = used - 8;
                if (write_full(fd, batch, used) <= 0) goto out;
                used = 8;
                *(uint32_t *)batch = M_BATCH;
            }
            uint32_t *h = (uint32_t *)(batch + used);
            h[0] = b->tag; h[1] = b->id; h[2] = b->len;
            memcpy(batch + used + 12, g_content + b->off, b->len);
            used += 12 + b->len;
            myb += b->len;
            atomic_fetch_add(&g_sent, b->len);
        }
        if (used > 8) {
            *(uint32_t *)(batch + 4) = used - 8;
            if (write_full(fd, batch, used) <= 0) goto out;
        }
    }
    {
        uint32_t fin[3] = {M_DONE, (uint32_t)(myb > 0), 0};
        if (write_full(fd, fin, 12) <= 0) goto out;
        uint8_t ack;
        if (read_full(fd, &ack, 1) != 1) goto out;
        if (!ack) fprintf(stderr, "tid=%d server verify FAILED\n", a->tid);
    }
out:
    a->t_ack = now();
    a->bytes = myb;
    close(fd);
    return NULL;
}

static int run_client(const char *ip, int port, char mode, int nconn, int warm, int fp4) {
    gen_plan(fp4);
    if (!g_content) {
        g_content = malloc(REGION_SZ);
        if (!g_content) return 1;
        double t = now();
        for (int i = 0; i < g_nblocks; i++) {
            struct blk *b = &g_plan[i];
            for (uint32_t j = 0; j < b->len; j++) g_content[b->off + j] = pat(b->id, j);
        }
        fprintf(stderr, "content gen %.1f MB in %.2fs\n", g_total / 1e6, now() - t);
    }
    struct sockaddr_in dst = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    if (inet_pton(AF_INET, ip, &dst.sin_addr) != 1) return 1;
    pthread_t th[72], st;
    static struct cli_arg args[72];
    atomic_store(&g_sent, 0);
    g_run_done = 0;
    pthread_barrier_init(&g_barrier, NULL, nconn + 1);
    for (int i = 0; i < nconn; i++) {
        args[i].dst = dst; args[i].tid = i; args[i].nconn = nconn;
        args[i].warm = warm; args[i].mode = mode; args[i].t_ack = 0; args[i].bytes = 0;
        pthread_create(&th[i], NULL, cli_worker, &args[i]);
    }
    pthread_barrier_wait(&g_barrier);
    g_t0 = now();
    pthread_create(&st, NULL, stats_thread, NULL);
    double tack = 0;
    uint64_t tb = 0;
    for (int i = 0; i < nconn; i++) {
        pthread_join(th[i], NULL);
        if (args[i].t_ack > tack) tack = args[i].t_ack;
        tb += args[i].bytes;
    }
    g_run_done = 1;
    pthread_join(st, NULL);
    double el = tack - g_t0;
    printf("KVRESULT mode=%c nconn=%d warm=%d plan=%s bytes=%llu blocks=%d total_ms=%.1f eff_gbps=%.3f\n",
           mode, nconn, warm, fp4 ? "FP4" : "FP8", (unsigned long long)tb, g_nblocks,
           el * 1000, (double)tb * 8 / el / 1e9);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc == 3 && !strcmp(argv[1], "server"))
        return run_server(atoi(argv[2]));
    if (argc >= 6 && !strcmp(argv[1], "client")) {
        char mode = 'B'; int nconn = 16, warm = 0, fp4 = 0;
        for (int i = 4; i < argc; i++) {
            if (!strcmp(argv[i], "-m") && i + 1 < argc) mode = argv[++i][0];
            else if (!strcmp(argv[i], "-n") && i + 1 < argc) nconn = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-w") && i + 1 < argc) warm = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-p") && i + 1 < argc) fp4 = !strcmp(argv[++i], "FP4");
        }
        if (nconn < 1 || nconn > 72) { fprintf(stderr, "nconn out of range\n"); return 2; }
        return run_client(argv[2], atoi(argv[3]), mode, nconn, warm, fp4);
    }
    fprintf(stderr, "usage:\n  %s server <port>\n  %s client <ip> <port> -m A|B -n <conns> -w <0|1> -p FP8|FP4\n", argv[0], argv[0]);
    return 2;
}
