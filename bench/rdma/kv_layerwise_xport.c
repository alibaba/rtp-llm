/* kv_layerwise_xport.c - replay the measured 32K PD-sep KV publish rhythm over TCP.
 *
 * Models one CEP4PP2(8 rank) -> DEP4(rank0) request exactly as observed on
 * 2026-10-23 (TCP cache-store, PPU decode host, 32760-token prompt):
 *   - 8 P ranks, one TCP connection each (= framework TcpClient: 1 channel/peer)
 *   - per rank the D asks 82 (stage0) / 85 (stage1) per-layer-per-tag buffers
 *   - every rank serves an identical-size 1/4 CP share: 65.9 MB (stage0 rank),
 *     71.2 MB (stage1 rank); stage0 ranks get layers 0..21, stage1 22..42
 *   - layer L of a stage becomes ready at t_ready(L) (measured write windows:
 *     stage0 layer l at 45.5*l ms, stage1 layer l at 1060+45.8*l ms)
 * Wire format per message: 16B header {layer, tag, nblocks, bytes} + payload.
 *
 * Sender ("P", on a GPU host): for each ready (layer,tag) message it either
 *   frame: copies each block D2H individually into a pinned staging buffer
 *          with a stream sync after every block (copyBlock -> execNoBlockCopy
 *          per block, as in RequestBlockBufferStore::makeValidBlock), then
 *          write()s the whole message on its rank conn;
 *   eng:   one D2H for the message's contiguous source range, then write().
 * Receiver ("D", on a GPU host): reads each message into pinned memory then
 *   frame: H2D per block with a sync after every block (TcpCacheStoreLoad-
 *          ServiceClosure calls execNoBlockCopy per block);
 *   eng:   one H2D for the whole message.
 *   h2d_par caps how many connections may run their H2D section at once
 *   (framework closures run on the ~2 arpc io threads; 0 = unlimited).
 *   It stamps the time the LAST message of the LAST layer lands in device
 *   memory and replies 8 bytes so the sender can print the tail.
 *
 * Timing printed by the receiver (receiver clock, aligned to the first byte
 * of the run via a start message): last_layer_ready -> all_landed tail.
 *
 * Build: gcc -O2 -pthread -o kv_layerwise_xport kv_layerwise_xport.c -ldl
 *   kv_layerwise_xport recv <port> <gpu> <frame|eng> <nconn> [h2d_par]
 *   kv_layerwise_xport send <ip> <port> <gpu> <frame|eng> <rounds> [scale] [conn_per_rank] [defer]
 *   scale: multiply per-rank bytes (1.0 = measured 548.5 MB total; 0.25 = CP-deduplicated)
 *   defer: 0 release each layer at its production time; 1 release a stage's
 *          layers only at the stage's last-layer time (observed framework stall)
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

typedef int CUresult_t;
typedef unsigned long long CUdeviceptr_t;
typedef void *CUstream_t, *CUcontext_t;
static struct {
    CUresult_t (*init)(unsigned);
    CUresult_t (*retain)(CUcontext_t *, int);
    CUresult_t (*set_current)(CUcontext_t);
    CUresult_t (*alloc)(CUdeviceptr_t *, size_t);
    CUresult_t (*host_alloc)(void **, size_t, unsigned);
    CUresult_t (*htod_a)(CUdeviceptr_t, const void *, size_t, CUstream_t);
    CUresult_t (*dtoh_a)(void *, CUdeviceptr_t, size_t, CUstream_t);
    CUresult_t (*stream_create)(CUstream_t *, unsigned);
    CUresult_t (*stream_sync)(CUstream_t);
} cu;
static CUcontext_t g_ctx;

static int cuda_up(int gpu) {
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen libcuda: %s\n", dlerror()); return -1; }
#define L(f, s) do { *(void **)(&cu.f) = dlsym(h, s); if (!cu.f) { fprintf(stderr, "missing %s\n", s); return -1; } } while (0)
    L(init, "cuInit"); L(retain, "cuDevicePrimaryCtxRetain"); L(set_current, "cuCtxSetCurrent");
    L(alloc, "cuMemAlloc_v2"); L(host_alloc, "cuMemHostAlloc");
    L(htod_a, "cuMemcpyHtoDAsync_v2"); L(dtoh_a, "cuMemcpyDtoHAsync_v2");
    L(stream_create, "cuStreamCreate"); L(stream_sync, "cuStreamSynchronize");
#undef L
    if (cu.init(0) || cu.retain(&g_ctx, gpu) || cu.set_current(g_ctx)) { fprintf(stderr, "cuda init gpu%d failed\n", gpu); return -1; }
    return 0;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static void sleep_until(double t) { double d; while ((d = t - now()) > 0) { struct timespec s = {(time_t)d, (long)((d - (time_t)d) * 1e9)}; nanosleep(&s, NULL); } }
static int read_full(int fd, void *b, size_t n) { size_t o = 0; while (o < n) { ssize_t r = read(fd, (char *)b + o, n - o); if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; } o += r; } return 0; }
static int write_full(int fd, const void *b, size_t n) { size_t o = 0; while (o < n) { ssize_t r = write(fd, (const char *)b + o, n - o); if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; } o += r; } return 0; }

/* ---- measured per-rank plan (one rank = 1/4 CP share of its stage) ---- */
/* tags: 0 swa_kv 1 csa_kv 2 indexer_kv 3 indexer_state 4 csa_state 5 hca_kv 6 hca_state */
struct msg { int layer, tag, nblocks; uint32_t blk; };
#define MAXMSG 128
struct rank_plan { int nmsg; struct msg m[MAXMSG]; double t_ready[MAXMSG]; uint64_t bytes; };
static struct rank_plan g_rank[8];
static double g_scale = 1.0;
static int g_defer = 0; /* 1: release a stage's messages only at its last-layer time */

static void add(struct rank_plan *p, int layer, int tag, int nb, uint32_t blk, double t) {
    struct msg *m = &p->m[p->nmsg]; m->layer = layer; m->tag = tag; m->nblocks = nb; m->blk = (uint32_t)(blk * g_scale);
    p->t_ready[p->nmsg] = t; p->nmsg++; p->bytes += (uint64_t)nb * m->blk;
}

/* Per-rank shares from D's PD_CACHE_KEY_READ_BLOCK (request 3605772662934315044):
 *   csa_kv 37440B x (1280 s0 / 1408 s1) blocks per rank, indexer_kv 4352B (same counts),
 *   hca_kv 1728B x 1280, swa_kv 74880B x (44 s0 / 42 s1), csa_state 65536B x (20/22),
 *   indexer_state 16384B x (20/22), hca_state 524288B x 10 (= 65.9 / 71.2 MB per rank).
 * csa/indexer layers = even, hca layers = odd (DSv4 alternating), swa every layer. */
static void build_plan(void) {
    memset(g_rank, 0, sizeof(g_rank));
    for (int r = 0; r < 8; r++) {
        struct rank_plan *p = &g_rank[r];
        int s1 = r >= 4;
        int base = s1 ? 22 : 0, nl = s1 ? 21 : 22;
        double t0 = s1 ? 1060.0 : 0.0, dt = s1 ? 45.8 : 45.5;
        int csa_layers = 0, hca_layers = 0;
        for (int l = 0; l < nl; l++) { int g = base + l; if (g >= 2) { if (g % 2 == 0) csa_layers++; else hca_layers++; } }
        int csa_per_layer = (s1 ? 1408 : 1280) / (csa_layers ? csa_layers : 1);
        int hca_per_layer = 1280 / (hca_layers ? hca_layers : 1);
        int st_per_layer_csa = (s1 ? 22 : 20) / (csa_layers ? csa_layers : 1);
        int hst_per_layer = 10 / (hca_layers ? hca_layers : 1);
        for (int l = 0; l < nl; l++) {
            int g = base + l;
            double t = (t0 + dt * l) / 1000.0;
            add(p, g, 0, 2, 74880, t);                       /* swa_kv: 2 blocks/layer */
            if (g < 2) continue;
            if (g % 2 == 0) {
                add(p, g, 1, csa_per_layer, 37440, t);
                add(p, g, 2, csa_per_layer, 4352, t);
                add(p, g, 3, st_per_layer_csa ? st_per_layer_csa : 1, 16384, t);
                add(p, g, 4, st_per_layer_csa ? st_per_layer_csa : 1, 65536, t);
            } else {
                add(p, g, 5, hca_per_layer, 1728, t);
                if (hst_per_layer) add(p, g, 6, hst_per_layer, 524288, t);
            }
        }
    }
}

enum { M_START = 1, M_DATA = 2, M_END = 3 };
struct hdr { uint32_t type, layer, tag, nblocks; uint32_t blk, pad; double t_ready; };

/* ---------------- receiver ---------------- */
static int g_mode_eng;
static int g_h2d_par;
static sem_t g_h2d_sem;
static CUdeviceptr_t g_ddev;
static uint64_t g_dcap = (uint64_t)1 << 30;
static atomic_int g_conn_done;
static double g_t_last_land[8];
static double g_t_start;

static void *recv_conn(void *arg) {
    int fd = (int)(intptr_t)arg;
    cu.set_current(g_ctx);
    CUstream_t st; cu.stream_create(&st, 0);
    void *pin = NULL; cu.host_alloc(&pin, 64 << 20, 0);
    struct hdr h; int rank = -1; uint64_t dev_off = 0; double last = 0;
    for (;;) {
        if (read_full(fd, &h, sizeof(h))) break;
        if (h.type == M_START) { rank = (int)h.layer; dev_off = (uint64_t)rank * (128 << 20); if (rank == 0) g_t_start = now(); continue; }
        if (h.type == M_END) break;
        uint64_t len = (uint64_t)h.nblocks * h.blk;
        if (read_full(fd, pin, len)) break;
        if (g_h2d_par > 0) sem_wait(&g_h2d_sem);
        if (g_mode_eng) {
            cu.htod_a(g_ddev + dev_off, pin, len, st);
            cu.stream_sync(st);
        } else {
            for (uint32_t i = 0; i < h.nblocks; i++) {
                cu.htod_a(g_ddev + dev_off + (uint64_t)i * h.blk, (char *)pin + (uint64_t)i * h.blk, h.blk, st);
                cu.stream_sync(st);
            }
        }
        if (g_h2d_par > 0) sem_post(&g_h2d_sem);
        last = now();
    }
    if (rank >= 0) g_t_last_land[rank] = last;
    double ack = last;
    write_full(fd, &ack, sizeof(ack));
    close(fd);
    atomic_fetch_add(&g_conn_done, 1);
    return NULL;
}

static int run_recv(int port, int gpu, int nconn) {
    if (cuda_up(gpu) || cu.alloc(&g_ddev, g_dcap)) return 1;
    int lfd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = htonl(INADDR_ANY)};
    if (bind(lfd, (void *)&a, sizeof(a)) || listen(lfd, 64)) { perror("bind"); return 1; }
    fprintf(stderr, "recv ready port=%d mode=%s nconn=%d h2d_par=%d\n", port, g_mode_eng ? "eng" : "frame", nconn, g_h2d_par);
    for (;;) {
        int c = accept(lfd, NULL, NULL);
        if (c < 0) continue;
        int bufsz = 0; (void)bufsz;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        pthread_t th; pthread_create(&th, NULL, recv_conn, (void *)(intptr_t)c); pthread_detach(th);
    }
}

/* ---------------- sender ---------------- */
struct snd { int fd, rank, conn_in_rank, nconn_rank; double t0; CUdeviceptr_t src; double t_last_send; double ack; uint64_t sent; };
static void *send_rank(void *arg) {
    struct snd *s = arg;
    cu.set_current(g_ctx);
    CUstream_t st; cu.stream_create(&st, 0);
    void *pin = NULL; cu.host_alloc(&pin, 64 << 20, 0);
    struct hdr h = {M_START, (uint32_t)s->rank, 0, 0, 0, 0, 0};
    write_full(s->fd, &h, sizeof(h));
    struct rank_plan *p = &g_rank[s->rank];
    for (int i = 0; i < p->nmsg; i++) {
        if (i % s->nconn_rank != s->conn_in_rank) continue;
        struct msg *m = &p->m[i];
        double tr = p->t_ready[i];
        if (g_defer) { tr = 0; for (int q = 0; q < p->nmsg; q++) if (p->t_ready[q] > tr) tr = p->t_ready[q]; }
        sleep_until(s->t0 + tr);
        uint64_t len = (uint64_t)m->nblocks * m->blk;
        if (g_mode_eng) {
            cu.dtoh_a(pin, s->src, len, st);
            cu.stream_sync(st);
        } else {
            for (int b = 0; b < m->nblocks; b++) {
                cu.dtoh_a((char *)pin + (uint64_t)b * m->blk, s->src + (uint64_t)b * m->blk, m->blk, st);
                cu.stream_sync(st);
            }
        }
        struct hdr d = {M_DATA, (uint32_t)m->layer, (uint32_t)m->tag, (uint32_t)m->nblocks, m->blk, 0, p->t_ready[i]};
        write_full(s->fd, &d, sizeof(d));
        write_full(s->fd, pin, len);
        s->sent += len;
    }
    s->t_last_send = now();
    struct hdr e = {M_END, 0, 0, 0, 0, 0, 0};
    write_full(s->fd, &e, sizeof(e));
    read_full(s->fd, &s->ack, sizeof(s->ack));
    close(s->fd);
    return NULL;
}

static int run_send(const char *ip, int port, int gpu, int rounds, int conn_per_rank) {
    if (cuda_up(gpu)) return 1;
    CUdeviceptr_t src; if (cu.alloc(&src, 64 << 20)) return 1;
    double last_ready_s0 = 0, last_ready_s1 = 0;
    for (int i = 0; i < g_rank[0].nmsg; i++) if (g_rank[0].t_ready[i] > last_ready_s0) last_ready_s0 = g_rank[0].t_ready[i];
    for (int i = 0; i < g_rank[4].nmsg; i++) if (g_rank[4].t_ready[i] > last_ready_s1) last_ready_s1 = g_rank[4].t_ready[i];
    uint64_t tot = 0; for (int r = 0; r < 8; r++) tot += g_rank[r].bytes;
    printf("plan: total=%.1fMB rank0=%.1fMB rank4=%.1fMB msgs/rank=%d/%d last_ready s0=%.0fms s1=%.0fms mode=%s conn/rank=%d defer=%d\n",
           tot / 1e6, g_rank[0].bytes / 1e6, g_rank[4].bytes / 1e6, g_rank[0].nmsg, g_rank[4].nmsg,
           last_ready_s0 * 1000, last_ready_s1 * 1000, g_mode_eng ? "eng" : "frame", conn_per_rank, g_defer);
    for (int rd = 0; rd < rounds; rd++) {
        int n = 8 * conn_per_rank;
        struct snd s[64]; pthread_t th[64];
        for (int k = 0; k < n; k++) {
            s[k].rank = k / conn_per_rank; s[k].conn_in_rank = k % conn_per_rank; s[k].nconn_rank = conn_per_rank;
            s[k].src = src; s[k].sent = 0;
            s[k].fd = socket(AF_INET, SOCK_STREAM, 0);
            int one = 1; setsockopt(s[k].fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)}; inet_pton(AF_INET, ip, &a.sin_addr);
            if (connect(s[k].fd, (void *)&a, sizeof(a))) { perror("connect"); return 1; }
        }
        usleep(200000);
        double t0 = now() + 0.05;
        for (int k = 0; k < n; k++) { s[k].t0 = t0; pthread_create(&th[k], NULL, send_rank, &s[k]); }
        double last_send = 0, last_send_s1 = 0; uint64_t sent = 0;
        for (int k = 0; k < n; k++) {
            pthread_join(th[k], NULL); sent += s[k].sent;
            if (s[k].t_last_send > last_send) last_send = s[k].t_last_send;
            if (s[k].rank >= 4 && s[k].t_last_send > last_send_s1) last_send_s1 = s[k].t_last_send;
        }
        double t_ack = now();
        printf("round %d: sent=%.1fMB  sender_last_write@%.1fms (s1 last ready %.0fms) -> tail(last_ready->last_write)=%.1fms  "
               "ack_rtt_close=%.1fms\n", rd, sent / 1e6, (last_send - t0) * 1000, last_ready_s1 * 1000,
               (last_send_s1 - t0 - last_ready_s1) * 1000, (t_ack - t0 - last_ready_s1) * 1000);
        fflush(stdout);
        usleep(500000);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 6 && !strcmp(argv[1], "recv")) {
        g_mode_eng = !strcmp(argv[4], "eng");
        g_h2d_par = argc >= 7 ? atoi(argv[6]) : 0;
        if (g_h2d_par > 0) sem_init(&g_h2d_sem, 0, g_h2d_par);
        build_plan();
        return run_recv(atoi(argv[2]), atoi(argv[3]), atoi(argv[5]));
    }
    if (argc >= 7 && !strcmp(argv[1], "send")) {
        g_mode_eng = !strcmp(argv[5], "eng");
        if (argc >= 8) g_scale = atof(argv[7]);
        int cpr = argc >= 9 ? atoi(argv[8]) : 1;
        if (argc >= 10) g_defer = atoi(argv[9]);
        build_plan();
        return run_send(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[6]), cpr);
    }
    fprintf(stderr, "usage:\n  %s recv <port> <gpu> <frame|eng> <nconn> [h2d_par]\n  %s send <ip> <port> <gpu> <frame|eng> <rounds> [scale] [conn_per_rank] [defer]\n", argv[0], argv[0]);
    return 2;
}
