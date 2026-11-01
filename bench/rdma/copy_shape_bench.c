/* copy_shape_bench.c - D2H/H2D cost for the measured per-rank 32K KV message shapes.
 *
 * Per-rank messages (see kv_layerwise_xport.c): csa_kv 37440B blocks, indexer_kv
 * 4352B, hca_kv 1728B, swa_kv 74880B, states 16-512KiB. For each block size it
 * times one "message" (N blocks) copied (a) per block on one stream + sync
 * (framework execNoBlockCopy shape) and (b) as a single contiguous copy.
 * Build: gcc -O2 -o copy_shape_bench copy_shape_bench.c -ldl
 *   copy_shape_bench <gpu> <d2h|h2d>
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int R; typedef unsigned long long DP; typedef void *S, *C;
static R (*cuInit_)(unsigned); static R (*retain)(C *, int); static R (*setcur)(C);
static R (*dalloc)(DP *, size_t); static R (*halloc)(void **, size_t, unsigned);
static R (*htod)(DP, const void *, size_t, S); static R (*dtoh)(void *, DP, size_t, S);
static R (*screate)(S *, unsigned); static R (*ssync)(S);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gpu> <d2h|h2d>\n", argv[0]); return 2; }
    int gpu = atoi(argv[1]), d2h = !strcmp(argv[2], "d2h");
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    *(void **)&cuInit_ = dlsym(h, "cuInit"); *(void **)&retain = dlsym(h, "cuDevicePrimaryCtxRetain");
    *(void **)&setcur = dlsym(h, "cuCtxSetCurrent"); *(void **)&dalloc = dlsym(h, "cuMemAlloc_v2");
    *(void **)&halloc = dlsym(h, "cuMemHostAlloc"); *(void **)&htod = dlsym(h, "cuMemcpyHtoDAsync_v2");
    *(void **)&dtoh = dlsym(h, "cuMemcpyDtoHAsync_v2"); *(void **)&screate = dlsym(h, "cuStreamCreate");
    *(void **)&ssync = dlsym(h, "cuStreamSynchronize");
    C ctx; DP dev; void *pin; S st;
    if (cuInit_(0) || retain(&ctx, gpu) || setcur(ctx)) { fprintf(stderr, "cuda init fail\n"); return 1; }
    size_t cap = 64 << 20;
    if (dalloc(&dev, cap) || halloc(&pin, cap, 0) || screate(&st, 0)) { fprintf(stderr, "alloc fail\n"); return 1; }
    memset(pin, 1, cap);
    struct { const char *name; uint32_t blk; int n; } cases[] = {
        {"csa_kv(64 blk/msg)", 37440, 64}, {"csa_kv(128)", 37440, 128}, {"indexer_kv(64)", 4352, 64},
        {"hca_kv(64)", 1728, 64}, {"swa_kv(2)", 74880, 2}, {"csa_state(1)", 65536, 1}, {"hca_state(1)", 524288, 1},
    };
    printf("dir=%s gpu=%d  per-message cost, median of 50\n", d2h ? "D2H" : "H2D", gpu);
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        double best[2] = {0, 0};
        for (int mode = 0; mode < 2; mode++) {
            double ts[50];
            for (int it = 0; it < 50; it++) {
                double t = now();
                if (mode == 0) {
                    for (int b = 0; b < cases[c].n; b++) {
                        size_t off = (size_t)b * cases[c].blk;
                        if (d2h) dtoh((char *)pin + off, dev + off, cases[c].blk, st);
                        else htod(dev + off, (char *)pin + off, cases[c].blk, st);
                    }
                } else {
                    size_t len = (size_t)cases[c].n * cases[c].blk;
                    if (d2h) dtoh(pin, dev, len, st); else htod(dev, pin, len, st);
                }
                ssync(st);
                ts[it] = (now() - t) * 1e6;
            }
            for (int i = 0; i < 50; i++) for (int j = i + 1; j < 50; j++) if (ts[j] < ts[i]) { double x = ts[i]; ts[i] = ts[j]; ts[j] = x; }
            best[mode] = ts[25];
        }
        double mb = (double)cases[c].n * cases[c].blk / 1e6;
        printf("  %-20s %7.3fMB  per-block %8.1fus (%6.1f Gb/s)   one-copy %8.1fus (%6.1f Gb/s)\n", cases[c].name, mb,
               best[0], mb * 8e3 / best[0], best[1], mb * 8e3 / best[1]);
    }
    return 0;
}
