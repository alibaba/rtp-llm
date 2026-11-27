#ifndef BLOCK_TREE_CACHE_WORKER_H
#define BLOCK_TREE_CACHE_WORKER_H

#include "block_tree_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BTC_RANK_WORKER_ABI_VERSION 1u

typedef struct btc_rank_worker btc_rank_worker;

typedef struct {
    uint32_t abi_version;
    uint32_t world_rank;
    uint32_t group_count;
    int32_t device_slots;
    int32_t host_slots;
} btc_rank_worker_config;

typedef struct {
    uint32_t group;
    int32_t device_slot;
    int32_t host_slot;
} btc_rank_page;

typedef struct {
    btc_result result;
    uint32_t safe;
} btc_rank_completion;

typedef struct {
    uint64_t submitted_jobs;
    uint64_t executed_batches;
    uint64_t failed_batches;
} btc_rank_worker_stats;

// A batch is a bounded contiguous range within the supplied job pages. Submit
// runs on a native WorkLoop owner. It must call done exactly once, inline or
// asynchronously, after its I/O can no longer access the ranges. A success
// value of zero denotes a safe failure. Neither callback may throw.
typedef void (*btc_rank_copy_done_fn)(void* context, uint32_t success);
typedef void (*btc_rank_submit_fn)(void* payload, size_t first_page, size_t page_count,
                                  btc_rank_copy_done_fn done, void* context);
typedef void (*btc_rank_release_fn)(void* payload);

// The caller serializes external calls per worker. Close drains the common
// WorkLoops before releasing remaining payloads.
BTC_API btc_status btc_rank_worker_open(const btc_rank_worker_config* config, btc_rank_worker** worker);
BTC_API btc_status btc_rank_worker_close(btc_rank_worker* worker);

// Pages are grouped contiguously by group. The worker takes ownership of
// payload with release on every return, including rejection. Submission never
// changes the caller's device/Host layout. Job identifiers are unique until
// their safe completion is consumed.
BTC_API btc_status btc_rank_worker_submit(btc_rank_worker* worker,
                                         uint64_t job,
                                         uint32_t load,
                                         const btc_rank_page* pages,
                                         size_t count,
                                         void* payload,
                                         btc_rank_submit_fn submit,
                                         btc_rank_release_fn release);

// Poll returns BTC_PENDING while the business result is outstanding. It will
// keep payload ownership until a completion proves it safe. Wait also waits
// for that proof, before the caller resumes device work on these ranges.
BTC_API btc_status btc_rank_worker_poll(btc_rank_worker* worker, uint64_t job, btc_rank_completion* completion);
BTC_API btc_status btc_rank_worker_wait(btc_rank_worker* worker, uint64_t job, btc_rank_completion* completion);
BTC_API btc_status btc_rank_worker_get_stats(btc_rank_worker* worker, btc_rank_worker_stats* stats);

#ifdef __cplusplus
}
#endif

#endif
