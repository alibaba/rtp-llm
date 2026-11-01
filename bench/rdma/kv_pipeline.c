/* kv_pipeline.c - KV handoff full-chain pipeline benchmark (CUDA driver API via dlopen)
 *
 * Chain: VRAM -> (D2H async, pinned ring slots) -> N TCP conns -> peer host
 *        -> (optional H2D async) -> peer VRAM, with sampled verify.
 *
 * Build: gcc -O2 -pthread -o kv_pipeline kv_pipeline.c -ldl
 *
 *   kv_pipeline bench <gpu_idx>
 *   kv_pipeline server <port> <h2d 0|1> <gpu_idx>
 *   kv_pipeline client <ip> <port> -n <conns> -S <slots> -Z <slot_MiB> -g <gpu> [-b <total_MiB>]
 *
 * Block plan = real 128K FP8 plan (531,448,576 B, 31934 blocks) unless -b
 * gives a bulk size in MiB (then uniform 1MiB blocks).
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
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
#include <time.h>
#include <unistd.h>

typedef int CUresult_t;
typedef unsigned long long CUdeviceptr_t;
typedef void *CUstream_t, *CUevent_t, *CUcontext_t;
typedef int CUdevice_t;

static struct {
    CUresult_t (*init)(unsigned);
    CUresult_t (*retain)(CUcontext_t *, CUdevice_t);
    CUresult_t (*set_current)(CUcontext_t);
    CUresult_t (*alloc)(CUdeviceptr_t *, size_t);
    CUresult_t (*free_mem)(CUdeviceptr_t);
    CUresult_t (*host_alloc)(void **, size_t, unsigned);
    CUresult_t (*free_host)(void *);
    CUresult_t (*htod)(CUdeviceptr_t, const void *, size_t);
    CUresult_t (*dtoh)(void *, CUdeviceptr_t, size_t);
    CUresult_t (*htod_a)(CUdeviceptr_t, const void *, size_t, CUstream_t);
    CUresult_t (*dtoh_a)(void *, CUdeviceptr_t, size_t, CUstream_t);
    CUresult_t (*stream_create)(CUstream_t *, unsigned);
    CUresult_t (*stream_sync)(CUstream_t);
    CUresult_t (*stream_destroy)(CUstream_t);
    CUresult_t (*event_create)(CUevent_t *, unsigned);
    CUresult_t (*event_record)(CUevent_t, CUstream_t);
    CUresult_t (*event_sync)(CUevent_t);
    CUresult_t (*event_query)(CUevent_t);
    CUresult_t (*event_destroy)(CUevent_t);
    CUresult_t (*get_error)(CUresult_t, const char **);
} cu;

static void *g_cuda_lib;
static int g_cuda_ok;

static int cuda_load(void) {
    g_cuda_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!g_cuda_lib) { fprintf(stderr, "dlopen libcuda: %s\n", dlerror()); return -1; }
#define L(f, s) do { *(void **)(&cu.f) = dlsym(g_cuda_lib, s); if (!cu.f) { fprintf(stderr, "missing %s\n", s); return -1; } } while (0)
    L(init, "cuInit"); L(retain, "cuDevicePrimaryCtxRetain"); L(set_current, "cuCtxSetCurrent");
    L(alloc, "cuMemAlloc_v2"); L(free_mem, "cuMemFree_v2");
    L(host_alloc, "cuMemHostAlloc"); L(free_host, "cuMemFreeHost");
    L(htod, "cuMemcpyHtoD_v2"); L(dtoh, "cuMemcpyDtoH_v2");
    L(htod_a, "cuMemcpyHtoDAsync_v2"); L(dtoh_a, "cuMemcpyDtoHAsync_v2");
    L(stream_create, "cuStreamCreate"); L(stream_sync, "cuStreamSynchronize");
    L(stream_destroy, "cuStreamDestroy_v2");
    L(event_create, "cuEventCreate");
    *(void **)(&cu.event_record) = dlsym(g_cuda_lib, "cuEventRecord_v2");
    if (!cu.event_record) *(void **)(&cu.event_record) = dlsym(g_cuda_lib, "cuEventRecord");
    if (!cu.event_record) { fprintf(stderr, "missing cuEventRecord\n"); return -1; }
    L(event_sync, "cuEventSynchronize"); L(event_query, "cuEventQuery");
    L(event_destroy, "cuEventDestroy_v2");
    *(void **)(&cu.get_error) = dlsym(g_cuda_lib, "cuGetErrorName");
#undef L
    if (cu.init(0)) { fprintf(stderr, "cuInit failed\n"); return -1; }
    g_cuda_ok = 1;
    return 0;
}

static CUcontext_t g_cuctx;
static int cuda_dev(int idx) {
    CUresult_t rc = cu.retain(&g_cuctx, idx);
    if (rc) { const char *e = "?"; if (cu.get_error) cu.get_error(rc, &e); fprintf(stderr, "retain gpu%d rc=%d %s\n", idx, rc, e); return -1; }
    cu.set_current(g_cuctx);
    return 0;
}

/* ---------- block plan (same as kvshape_xport.c) ---------- */
#define MAX_BLOCKS 40000
struct blk { uint32_t tag, id, len; uint64_t off; };
static struct blk g_plan[MAX_BLOCKS];
static int g_nblocks;
static uint64_t g_total;
enum { T_CSA_KV, T_HCA_KV, T_IDX_KV, T_IDX_ST, T_CSA_ST, T_HCA_ST, T_SWA_KV };
static void gen_plan(void) {
    static const int order[][2] = {
        {T_HCA_ST, 20}, {T_CSA_ST, 42}, {T_IDX_ST, 42}, {T_SWA_KV, 86},
        {T_CSA_KV, 10752}, {T_IDX_KV, 10752}, {T_HCA_KV, 10240},
    };
    static const uint32_t sz[] = {37440, 1728, 8448, 16384, 65536, 524288, 74880};
    g_nblocks = 0; g_total = 0;
    for (unsigned g = 0; g < 7; g++)
        for (int i = 0; i < order[g][1]; i++) {
            struct blk *b = &g_plan[g_nblocks];
            b->tag = order[g][0]; b->id = g_nblocks; b->len = sz[order[g][0]];
            b->off = g_total; g_total += sz[order[g][0]]; g_nblocks++;
        }
}
static void gen_bulk(uint64_t mib) {
    g_nblocks = 0; g_total = 0;
    uint64_t n = mib;
    for (uint64_t i = 0; i < n; i++) {
        struct blk *b = &g_plan[g_nblocks];
        b->tag = 99; b->id = g_nblocks; b->len = 1 << 20; b->off = g_total;
        g_total += 1 << 20; g_nblocks++;
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

#define M_SLOT 0x534C4F54u
#define M_DONE 0x444F4E45u

/* ---------------- bench mode ---------------- */
static int run_bench(int gpu) {
    if (cuda_load() || cuda_dev(gpu)) return 1;
    gen_plan();
    size_t bytes = g_total;
    void *host = NULL;
    CUdeviceptr_t dev = 0;
    if (cu.host_alloc(&host, bytes, 0) || cu.alloc(&dev, bytes)) { fprintf(stderr, "alloc fail\n"); return 1; }
    for (int i = 0; i < g_nblocks; i++) {
        struct blk *b = &g_plan[i];
        for (uint32_t j = 0; j < b->len; j++) ((uint8_t *)host)[b->off + j] = pat(b->id, j);
    }
    double t = now();
    if (cu.htod(dev, host, bytes)) { fprintf(stderr, "htod fail\n"); return 1; }
    printf("BENCH htod_bulk bytes=%llu ms=%.1f gbps=%.2f\n", (unsigned long long)bytes,
           (now() - t) * 1000, (double)bytes * 8 / (now() - t) / 1e9);

    CUstream_t st[4];
    CUevent_t ev[4];
    for (int i = 0; i < 4; i++) { cu.stream_create(&st[i], 0); cu.event_create(&ev[i], 0); }

    t = now();
    cu.dtoh(host, dev, bytes);
    double d_bulk = now() - t;
    printf("BENCH dtoh_bulk bytes=%llu ms=%.1f gbps=%.2f\n", (unsigned long long)bytes,
           d_bulk * 1000, (double)bytes * 8 / d_bulk / 1e9);

    /* block-shaped D2H on 4 async streams */
    t = now();
    for (int i = 0; i < g_nblocks; i++) {
        struct blk *b = &g_plan[i];
        int s = i % 4;
        cu.dtoh_a((uint8_t *)host + b->off, dev + b->off, b->len, st[s]);
        if (i % 1024 == 1023) cu.event_record(ev[s], st[s]);
    }
    for (int i = 0; i < 4; i++) cu.stream_sync(st[i]);
    double d_shaped = now() - t;
    printf("BENCH dtoh_blockshaped bytes=%llu blocks=%d ms=%.1f gbps=%.2f\n",
           (unsigned long long)bytes, g_nblocks, d_shaped * 1000, (double)bytes * 8 / d_shaped / 1e9);

    t = now();
    for (int i = 0; i < g_nblocks; i++) {
        struct blk *b = &g_plan[i];
        cu.htod_a(dev + b->off, (uint8_t *)host + b->off, b->len, st[i % 4]);
    }
    for (int i = 0; i < 4; i++) cu.stream_sync(st[i]);
    double h_shaped = now() - t;
    printf("BENCH htod_blockshaped bytes=%llu blocks=%d ms=%.1f gbps=%.2f\n",
           (unsigned long long)bytes, g_nblocks, h_shaped * 1000, (double)bytes * 8 / h_shaped / 1e9);

    t = now();
    cu.htod(dev, host, bytes);
    printf("BENCH htod_bulk2 bytes=%llu ms=%.1f gbps=%.2f\n", (unsigned long long)bytes,
           (now() - t) * 1000, (double)bytes * 8 / (now() - t) / 1e9);
    return 0;
}

/* ---------------- server ---------------- */
static int pat_at_is(uint64_t off, uint8_t v) {
    int lo = 0, hi = g_nblocks - 1, b = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (g_plan[mid].off <= off) { b = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return pat(g_plan[b].id, (uint32_t)(off - g_plan[b].off)) == v;
}

static uint8_t *g_region;
static int g_h2d, g_gpu = 0;
static CUdeviceptr_t g_dev;
static int g_dev_ok;
static atomic_ulong g_h2d_bytes;

struct slot_sync {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int h2d_done;
};

static void *srv_conn(void *arg) {
    if (g_h2d && g_dev_ok) cu.set_current(g_cuctx);
    int fd = (int)(intptr_t)arg;
    static __thread uint8_t scratch[1 << 20];
    (void)scratch;
    for (;;) {
        uint32_t hdr[4];
        if (read_full(fd, hdr, 16) != 16) break;
        if (hdr[0] == M_DONE) break;
        if (hdr[0] != M_SLOT) break;
        uint64_t off = ((uint64_t)hdr[2] << 32) | hdr[1];
        uint32_t len = hdr[3];
        if (read_full(fd, g_region + off, len) != (ssize_t)len) break;
        if ((off >> 20) % 16 == 0) { /* verify 1/16 of slots, 32 strided bytes */
            for (int q = 0; q < 32; q++) {
                uint64_t p = off + (uint64_t)q * (len / 32);
                if (!pat_at_is(p, g_region[p])) {
                    int lo2 = 0, hi2 = g_nblocks - 1, b2 = 0;
                    while (lo2 <= hi2) { int m = (lo2 + hi2) / 2; if (g_plan[m].off <= p) { b2 = m; lo2 = m + 1; } else hi2 = m - 1; }
                    fprintf(stderr, "VERIFY_FAIL off=%llu got=%02x want=%02x blk=%u blkid=%u j=%llu\n",
                            (unsigned long long)p, g_region[p],
                            pat(g_plan[b2].id, (uint32_t)(p - g_plan[b2].off)), b2, g_plan[b2].id,
                            (unsigned long long)(p - g_plan[b2].off));
                    break;
                }
            }
        }
        if (g_h2d && g_dev_ok) {
            CUresult_t rh = cu.htod_a(g_dev + off, g_region + off, len, 0);
            if (rh) {
                const char *e = "?";
                if (cu.get_error) cu.get_error(rh, &e);
                fprintf(stderr, "H2DERR rc=%d %s\n", rh, e);
                cu.htod(g_dev + off, g_region + off, len);
            } else {
                cu.stream_sync(0);
            }
            atomic_fetch_add(&g_h2d_bytes, len);
        }
    }
    uint8_t ack = 1;
    write_full(fd, &ack, 1);
    close(fd);
    return NULL;
}

static int run_server(int port, int h2d, int gpu, int bulk_mib) {
    if (bulk_mib) gen_bulk(bulk_mib); else gen_plan();
    size_t cap = (size_t)4096 << 20;
    g_region = malloc(cap);
    if (!g_region) { fprintf(stderr, "region alloc fail\n"); return 1; }
    memset(g_region, 0, cap);
    g_h2d = h2d;
    if (h2d) {
        if (cuda_load() || cuda_dev(gpu)) return 1;
        if (cu.alloc(&g_dev, cap)) { fprintf(stderr, "dev alloc fail\n"); return 1; }
        g_dev_ok = 1;
    }
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct timeval tv = {120, 0};
    setsockopt(lfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port),
                               .sin_addr.s_addr = htonl(INADDR_ANY)};
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) || listen(lfd, 128)) { perror("bind"); return 1; }
    fprintf(stderr, "pipeserver ready port=%d h2d=%d\n", port, h2d);
    for (;;) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) { if (errno == EINTR) continue; break; }
        pthread_t th;
        if (pthread_create(&th, NULL, srv_conn, (void *)(intptr_t)cfd) == 0) pthread_detach(th);
        else close(cfd);
    }
    return 0;
}

/* server-side verify helper: called out-of-band after run via client request? kept simple:
 * verification done by client comparing pattern of received? No - server verifies on demand. */

/* ---------------- client ---------------- */
struct ring_slot {
    CUevent_t ev;
    CUstream_t st;
    int has_ev;
};

static CUdeviceptr_t g_cdev;
static uint8_t *g_pinned;
static uint64_t g_slot_bytes;
static int g_nslots_ring, g_nconn;
static uint64_t g_total_slots;
static pthread_barrier_t g_bar;
static atomic_long g_next_slot;
static atomic_ulong g_wait_ev_ns, g_wait_free_ns;
static double g_t0;
static int g_hostmode = 0;
static uint8_t *g_hostbuf;

struct cli_arg { struct sockaddr_in dst; int tid; double wait_ev, wait_free; uint64_t sent; };

static void *cli_worker(void *arg) {
    struct cli_arg *a = arg;
    if (!g_hostmode) cu.set_current(g_cuctx);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    if (connect(fd, (struct sockaddr *)&a->dst, sizeof(a->dst))) { perror("connect"); return NULL; }

    struct ring_slot *ring = calloc(g_nslots_ring, sizeof(struct ring_slot));
    if (!g_hostmode)
        for (int i = 0; i < g_nslots_ring; i++) {
        CUresult_t rs = cu.stream_create(&ring[i].st, 0);
        CUresult_t re = cu.event_create(&ring[i].ev, 0);
        if (rs || re) fprintf(stderr, "CREATEERR tid=%d i=%d rs=%d re=%d\n", a->tid, i, rs, re);
    }
    pthread_barrier_wait(&g_bar);

    double w_ev = 0, w_free = 0;
    uint64_t sent = 0;
    for (long k = a->tid, c = 0; k < (long)g_total_slots; k += g_nconn, c++) {
        struct ring_slot *sl = &ring[c % g_nslots_ring];
        if (sl->has_ev) {
            double t = now();
            cu.event_sync(sl->ev);
            w_free += now() - t; /* waited for slot's previous D2H to drain before reuse... */
        }
        uint64_t off = (uint64_t)k * g_slot_bytes;
        uint64_t len = g_slot_bytes;
        if (off + len > g_total) len = g_total - off;
        const uint8_t *srcp;
        if (g_hostmode) {
            srcp = g_hostbuf + off;
        } else {
            /* issue D2H async for this slot, wait, then send */
            double t = now();
            CUresult_t r1 = cu.dtoh_a(g_pinned + ((uint64_t)a->tid * g_nslots_ring + (c % g_nslots_ring)) * g_slot_bytes, g_cdev + off, len, sl->st);
            CUresult_t r2 = cu.event_record(sl->ev, sl->st);
            sl->has_ev = 1;
            CUresult_t r3 = cu.event_sync(sl->ev);
            if (r1 || r2 || r3) {
                const char *e = "?";
                if (cu.get_error) cu.get_error(r1 ? r1 : (r2 ? r2 : r3), &e);
                fprintf(stderr, "CUDAERR tid=%d k=%ld r1=%d r2=%d r3=%d err=%s\n", a->tid, k, r1, r2, r3, e);
            }
            w_ev += now() - t;
            srcp = g_pinned + ((uint64_t)a->tid * g_nslots_ring + (c % g_nslots_ring)) * g_slot_bytes;
        }
        uint32_t hdr[4] = {M_SLOT, (uint32_t)(off & 0xffffffffu), (uint32_t)(off >> 32), (uint32_t)len};
        if (write_full(fd, hdr, 16) <= 0) break;
        if (write_full(fd, srcp, len) <= 0) break;
        sent += len;
        /* slot cannot be reused until D2H of next reuse overwrites; D2H into pinned
         * while a previous send may still read: wait send done before next D2H issue —
         * ensured because we reuse this slot only in a later iteration of this same thread */
    }
    {
        uint32_t fin[4] = {M_DONE, 0, 0, 0};
        write_full(fd, fin, 16);
        uint8_t ack;
        read_full(fd, &ack, 1);
    }
    a->wait_ev = w_ev; a->wait_free = w_free; a->sent = sent;
    close(fd);
    return NULL;
}

static int run_client(const char *ip, int port, int nconn, int nslots, int z_mib, int gpu, int bulk_mib) {
    if (bulk_mib) gen_bulk(bulk_mib); else gen_plan();
    g_slot_bytes = (uint64_t)z_mib << 20;
    g_total_slots = (g_total + g_slot_bytes - 1) / g_slot_bytes;
    g_nslots_ring = nslots; g_nconn = nconn;
    uint8_t *stage;
    if (g_hostmode) {
        stage = malloc(g_total);
        for (int i = 0; i < g_nblocks; i++) {
            struct blk *b = &g_plan[i];
            for (uint32_t j = 0; j < b->len; j++) stage[b->off + j] = pat(b->id, j);
        }
        g_hostbuf = stage;
        g_pinned = stage;
        fprintf(stderr, "hostmode %.1fMB ready\n", g_total / 1e6);
    } else {
        if (cuda_load() || cuda_dev(gpu)) return 1;
        if (cu.host_alloc((void **)&g_pinned, (size_t)nconn * nslots * g_slot_bytes, 0)) { fprintf(stderr, "pinned fail\n"); return 1; }
        if (cu.alloc(&g_cdev, g_total)) { fprintf(stderr, "dev alloc fail\n"); return 1; }
        stage = malloc(g_total);
        if (!stage) return 1;
        for (int i = 0; i < g_nblocks; i++) {
            struct blk *b = &g_plan[i];
            for (uint32_t j = 0; j < b->len; j++) stage[b->off + j] = pat(b->id, j);
        }
        double tu = now();
        if (cu.htod(g_cdev, stage, g_total)) { fprintf(stderr, "upload fail\n"); return 1; }
        free(stage);
        fprintf(stderr, "upload %.1fMB in %.2fs\n", g_total / 1e6, now() - tu);
    }

    pthread_t th[72];
    static struct cli_arg args[72];
    atomic_store(&g_next_slot, 0);
    pthread_barrier_init(&g_bar, NULL, nconn + 1);
    struct sockaddr_in dst = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    inet_pton(AF_INET, ip, &dst.sin_addr);
    for (int i = 0; i < nconn; i++) {
        args[i].dst = dst; args[i].tid = i;
        pthread_create(&th[i], NULL, cli_worker, &args[i]);
    }
    pthread_barrier_wait(&g_bar);
    g_t0 = now();
    uint64_t sent = 0;
    double wev = 0, wfr = 0;
    for (int i = 0; i < nconn; i++) { pthread_join(th[i], NULL); sent += args[i].sent; wev += args[i].wait_ev; wfr += args[i].wait_free; }
    double el = now() - g_t0;
    printf("PIPERESULT n=%d S=%d Z=%dMiB bytes=%llu total_ms=%.1f eff_gbps=%.3f wait_ev_s=%.2f wait_free_s=%.2f scope=%s\n",
           nconn, nslots, z_mib, (unsigned long long)sent, el * 1000,
           (double)sent * 8 / el / 1e9, wev, wfr, "d2h+tcp(+h2d if server)");
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc == 3 && !strcmp(argv[1], "bench"))
        return run_bench(atoi(argv[2]));
    if (argc >= 5 && !strcmp(argv[1], "server"))
        return run_server(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argc >= 6 ? atoi(argv[5]) : 0);
    if (argc >= 6 && !strcmp(argv[1], "client")) {
        int nconn = 16, nslots = 4, z = 16, gpu = 4, bulk = 0;
        const char *ip = argv[2]; int port = atoi(argv[3]);
        for (int i = 4; i < argc; i++) {
            if (!strcmp(argv[i], "-n") && i + 1 < argc) nconn = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-S") && i + 1 < argc) nslots = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-Z") && i + 1 < argc) z = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-g") && i + 1 < argc) gpu = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-b") && i + 1 < argc) bulk = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-H")) g_hostmode = 1;
        }
        return run_client(ip, port, nconn, nslots, z, gpu, bulk);
    }
    fprintf(stderr, "usage:\n  %s bench <gpu>\n  %s server <port> <h2d 0|1> <gpu> [bulk_mib]\n  %s client <ip> <port> -n N -S S -Z MiB -g gpu [-b MiB]\n",
            argv[0], argv[0], argv[0]);
    return 2;
}
