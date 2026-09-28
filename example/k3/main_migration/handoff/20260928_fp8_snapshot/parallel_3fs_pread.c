#define _GNU_SOURCE
#define _LARGEFILE64_SOURCE
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Task-local diagnostic shim for one Kimi shard at a time in each process. */
#define BLOCK (4UL * 1024 * 1024)
#define MIN_INTERCEPT (1024UL * 1024)
#define MAX_FILE (32ULL * 1024 * 1024 * 1024)

typedef struct {
    int fd;
    dev_t dev;
    ino_t ino;
    unsigned char *data;
    size_t file_size;
    size_t map_size;
    int ready;
} cache_t;

typedef struct {
    int fd;
    unsigned char *data;
    size_t file_size;
    size_t blocks;
    int worker;
    int workers;
    int failed;
} worker_arg_t;

static pthread_mutex_t cache_mu = PTHREAD_MUTEX_INITIALIZER;
static cache_t cache = {.fd = -1};

static void clear_cache(void) {
    if (cache.data) munmap(cache.data, cache.map_size);
    memset(&cache, 0, sizeof(cache));
    cache.fd = -1;
}

static int eligible(int fd, size_t count, struct stat *st) {
    if (count < MIN_INTERCEPT || fstat(fd, st) != 0 ||
        !S_ISREG(st->st_mode) || st->st_size <= 0 ||
        (uint64_t)st->st_size > MAX_FILE) return 0;
    char proc[64], target[PATH_MAX];
    snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
    ssize_t n = readlink(proc, target, sizeof(target) - 1);
    if (n <= 0) return 0;
    target[n] = '\0';
    const char *prefix = "/mnt/hf3fs/3fs/models/kimi/";
    size_t len = strlen(target);
    return strncmp(target, prefix, strlen(prefix)) == 0 &&
           len >= 12 && strcmp(target + len - 12, ".safetensors") == 0;
}

static void *read_blocks(void *opaque) {
    worker_arg_t *a = opaque;
    for (size_t block = (size_t)a->worker; block < a->blocks;
         block += (size_t)a->workers) {
        size_t offset = block * BLOCK;
        size_t expected = a->file_size - offset;
        if (expected > BLOCK) expected = BLOCK;
        ssize_t got = syscall(SYS_pread64, a->fd, a->data + offset,
                              BLOCK, (off64_t)offset);
        if (got != (ssize_t)expected) {
            a->failed = 1;
            break;
        }
    }
    return NULL;
}

static int populate(int fd, const struct stat *st) {
    size_t file_size = (size_t)st->st_size;
    int workers = 64;
    const char *setting = getenv("K3_3FS_PREAD_THREADS");
    if (setting && *setting) workers = atoi(setting);
    if (workers < 1) workers = 1;
    /* FastSafetensors SHM showed no read-time gain at 128/256 workers. */
    if (workers > 64) workers = 64;
    size_t blocks = (file_size + BLOCK - 1) / BLOCK;
    size_t map_size = blocks * BLOCK;
    unsigned char *data = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) return -1;
    pthread_t threads[64];
    worker_arg_t args[64];
    int started = 0, failed = 0;
    for (int i = 0; i < workers; ++i) {
        args[i] = (worker_arg_t){.fd = fd, .data = data,
                                 .file_size = file_size, .blocks = blocks,
                                 .worker = i, .workers = workers};
        if (pthread_create(&threads[i], NULL, read_blocks, &args[i]) != 0) {
            failed = 1;
            break;
        }
        ++started;
    }
    for (int i = 0; i < started; ++i) {
        pthread_join(threads[i], NULL);
        failed |= args[i].failed;
    }
    if (failed) {
        munmap(data, map_size);
        return -1;
    }
    cache = (cache_t){.fd = fd, .dev = st->st_dev, .ino = st->st_ino,
                      .data = data, .file_size = file_size,
                      .map_size = map_size, .ready = 1};
    fprintf(stderr, "task 3FS parallel pread: fd=%d bytes=%zu threads=%d ready\n",
            fd, file_size, workers);
    return 0;
}

ssize_t pread64(int fd, void *buf, size_t count, off64_t offset) {
    struct stat st;
    if (!eligible(fd, count, &st) || offset < 0)
        return syscall(SYS_pread64, fd, buf, count, offset);
    pthread_mutex_lock(&cache_mu);
    if (cache.ready && (cache.fd != fd || cache.dev != st.st_dev ||
                        cache.ino != st.st_ino ||
                        cache.file_size != (size_t)st.st_size))
        clear_cache();
    if (!cache.ready && populate(fd, &st) != 0) {
        pthread_mutex_unlock(&cache_mu);
        return syscall(SYS_pread64, fd, buf, count, offset);
    }
    size_t begin = (size_t)offset;
    size_t actual = begin >= cache.file_size ? 0 : cache.file_size - begin;
    if (actual > count) actual = count;
    if (actual) memcpy(buf, cache.data + begin, actual);
    pthread_mutex_unlock(&cache_mu);
    return (ssize_t)actual;
}

ssize_t pread(int fd, void *buf, size_t count, off_t offset) {
    return pread64(fd, buf, count, (off64_t)offset);
}

int close(int fd) {
    pthread_mutex_lock(&cache_mu);
    if (cache.fd == fd) clear_cache();
    pthread_mutex_unlock(&cache_mu);
    return syscall(SYS_close, fd);
}
