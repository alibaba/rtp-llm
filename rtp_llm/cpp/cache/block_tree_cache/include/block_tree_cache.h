#ifndef BLOCK_TREE_CACHE_H
#define BLOCK_TREE_CACHE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(BTC_SHARED)
#if defined(BTC_BUILDING_LIBRARY)
#define BTC_API __declspec(dllexport)
#else
#define BTC_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define BTC_API __attribute__((visibility("default")))
#else
#define BTC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BTC_ABI_VERSION 2u
#define BTC_HOST_POOL UINT64_MAX
#define BTC_DISK_POOL (UINT64_MAX - 1u)

typedef struct btc_cache btc_cache;
typedef struct btc_plan btc_plan;
typedef struct btc_batch btc_batch;
typedef uint32_t btc_status;
typedef uint32_t btc_rule;
typedef uint32_t btc_work_kind;
typedef uint32_t btc_result;
typedef uint32_t btc_event_kind;

enum {
    BTC_OK = 0, BTC_INVALID = 1, BTC_PENDING = 2, BTC_RETRY = 3,
    BTC_WAITING = 4, BTC_CLOSED = 5, BTC_IO = 6, BTC_INTERNAL = 7
};
enum { BTC_FULL = 1, BTC_LINEAR = 2, BTC_SWA = 3 };
enum { BTC_EVICT_LRU = 0, BTC_EVICT_LFU = 1, BTC_EVICT_FIFO = 2 };
enum { BTC_PAGE_READY = 1, BTC_PAGE_LOAD = 2, BTC_PAGE_JOIN = 3 };
enum {
    BTC_WORK_HOST_TO_DEVICE = 1, BTC_WORK_DEVICE_TO_HOST = 2,
    BTC_WORK_DISK_TO_HOST = 3, BTC_WORK_HOST_TO_DISK = 4,
    BTC_WORK_REMOTE_QUERY = 5, BTC_WORK_REMOTE_TO_HOST = 6,
    BTC_WORK_HOST_TO_REMOTE = 7, BTC_WORK_RELEASE = 8,
    BTC_WORK_REMOTE_TO_DEVICE = 9, BTC_WORK_DEVICE_TO_REMOTE = 10,
    BTC_WORK_DISK_TO_DEVICE = 11
};
enum { BTC_SUCCESS = 1, BTC_FAILED = 2, BTC_TIMEOUT = 3, BTC_REJECTED = 4 };
enum { BTC_EVENT_READY = 1, BTC_EVENT_RECOMPUTE = 2, BTC_EVENT_CANCELLED = 3, BTC_EVENT_PROGRESS = 4 };

typedef struct {
    const void* data;
    size_t size;
} btc_bytes;

typedef struct {
    void* data;
    size_t size;
} btc_buffer;

typedef struct {
    btc_buffer memory;
    size_t stride_bytes;
    size_t payload_bytes;
} btc_host_region;

typedef struct {
    const btc_bytes* keys;
    size_t count;
} btc_keys;

typedef struct {
    uint32_t group;
    uint64_t end_token;
} btc_position;

typedef struct {
    uint64_t pool;
    uint64_t slot;
    uint64_t generation;
    uint64_t hold;
} btc_page;

typedef struct {
    uint32_t id;
    btc_rule rule;
    uint64_t page_tokens;
    uint64_t window_tokens;
    size_t members;
    /* Optional device pool per member, or NULL to let the engine select. */
    const uint64_t* member_pools;
    /* If any group declares a local quota, that tier uses per-group quotas.
     * Slot ranges are assigned in ascending group ID order. */
    uint64_t host_slots;
    uint64_t disk_slots;
} btc_group;

typedef struct {
    btc_work_kind kind;
    const uint32_t* participants;
    size_t count;
} btc_route;

typedef struct {
    uint32_t abi_version;
    btc_bytes name_space;
    uint64_t key_tokens;
    const btc_group* groups;
    size_t group_count;
    uint64_t host_slots;
    uint64_t disk_slots;
    btc_bytes disk_directory;
    const btc_route* routes;
    size_t route_count;
    /* Return local pages with hold=0 through RELEASE when the last lease ends.
     * Reuse of a local slot follows successful, safe release acknowledgment. */
    uint32_t release_local_pages;
    uint32_t device_eviction_policy;
    /* Executors can operate existing per-rank Disk files using virtual slots. */
    uint32_t external_disk_io;
    /* Device pressure is supplied by the engine. Host pressure is measured
     * by the shared native pool. Applies to a Device/Host cache with optional KVCM. */
    uint32_t tiered_eviction;
} btc_config;

typedef struct {
    uint64_t max_tokens;
    /* NULL selects all groups. */
    const uint32_t* groups;
    size_t group_count;
    /* NULL selects all boundaries divisible by the required group sizes. */
    const uint64_t* boundaries;
    size_t boundary_count;
    /* Upper bound on logical positions pending device load or join. */
    size_t max_load_pages;
    uint32_t allow_remote;
} btc_limits;

typedef struct {
    btc_position position;
    uint32_t state;
    const btc_page* pages;
    size_t page_count;
    uint64_t operation;
    btc_work_kind route;
} btc_plan_page;

typedef struct {
    uint64_t request;
    uint64_t ready_tokens;
    uint64_t recoverable_tokens;
    const btc_plan_page* pages;
    size_t page_count;
    const btc_plan_page* ready_pages;
    size_t ready_page_count;
    uint64_t host_tokens;
} btc_plan_view;

typedef struct {
    btc_position position;
    const btc_page* pages;
    size_t page_count;
} btc_binding;

typedef struct {
    uint64_t instance;
    uint64_t operation;
} btc_operation;

typedef struct {
    btc_position position;
    btc_bytes key;
    const btc_page* source;
    size_t source_count;
    const btc_page* target;
    size_t target_count;
    btc_bytes remote;
} btc_work_page;

typedef struct {
    btc_operation id;
    btc_work_kind kind;
    uint64_t request;
    btc_bytes name_space;
    const uint32_t* participants;
    size_t participant_count;
    const btc_work_page* pages;
    size_t page_count;
} btc_work;

typedef struct {
    uint64_t cursor;
    uint64_t request;
    btc_event_kind kind;
    uint64_t tokens;
} btc_event;

typedef struct {
    const btc_work* work;
    size_t work_count;
    const btc_event* events;
    size_t event_count;
} btc_batch_view;

typedef struct {
    btc_position position;
    btc_bytes descriptor;
} btc_remote_object;

typedef struct {
    btc_operation id;
    uint32_t participant;
    btc_result result;
    /* A failure may be reported before the executor stops touching pages. */
    uint32_t safe;
    /* Remote writes must also be committed and queryable. */
    uint32_t committed;
    const btc_remote_object* objects;
    size_t object_count;
} btc_receipt;

/* Poll results own their views. Inputs are borrowed for the duration of a call.
 * Status text for the last error is thread-local. */
BTC_API uint32_t btc_abi_version(void);
BTC_API const char* btc_last_error(void);
/* Optional bounded diagnostics. A cycle scans [cursor, end) in batches.
 * Concurrent removal can move nodes between batches; start each new cycle at zero.
 * The callback receives only FULL-group edges that violate prefix residency.
 * Tier masks express Device=1, Host=2 and Disk=4. */
enum { BTC_ON_DEVICE = 1, BTC_ON_HOST = 2, BTC_ON_DISK = 4 };
enum { BTC_IDLE = 0, BTC_LOADING = 1, BTC_EVICTING = 2 };
typedef struct {
    btc_bytes key;
    uint32_t tiers;
    uint32_t activity;
} btc_node_state;
typedef void (*btc_prefix_issue_fn)(void* context, uint32_t group,
                                    btc_node_state parent, btc_node_state child);
BTC_API btc_status btc_enable_prefix_scan(btc_cache* cache);
BTC_API btc_status btc_scan_prefix(btc_cache* cache, size_t cursor, size_t end, size_t budget,
                                   size_t* tree_size, size_t* visited,
                                   btc_prefix_issue_fn report, void* context);
/* Number of pages covered by a legal boundary, or zero for invalid geometry. */
BTC_API size_t btc_required_pages(btc_rule rule, uint64_t page_tokens, uint64_t window_tokens,
                                  uint64_t end_token);
BTC_API btc_status btc_open(const btc_config* config, btc_cache** cache);
/* Borrow existing writable storage for all Host slots. Slot zero begins at
 * memory.data. Keep the region mapped at that address until btc_close returns
 * BTC_OK, including while drain or close returns BTC_PENDING. */
BTC_API btc_status btc_open_with_host_region(const btc_config* config, const btc_host_region* region,
                                             btc_cache** cache);
/* Resolve the payload at a public Host slot in this process. A work item owns
 * its slot until its safe receipt. Looking up an address does not acquire it. */
BTC_API btc_status btc_host_slot_buffer(btc_cache* cache, uint32_t group, uint64_t slot, btc_buffer* out);
BTC_API btc_status btc_match(btc_cache* cache, uint64_t request, btc_keys keys,
                             const btc_limits* limits, btc_plan** plan);
/* Device/Host admission can bound the recoverable suffix relative to its
 * READY prefix. UINT64_MAX leaves the suffix unlimited. */
BTC_API btc_status btc_match_capped(btc_cache* cache, uint64_t request, btc_keys keys,
                                    const btc_limits* limits, uint64_t max_load_tokens, btc_plan** plan);
/* After PENDING, poll by request ID. OK transfers one plan to the caller.
 * RETRY means that admission ended. Cancel on request timeout, changed input,
 * or allocator shortage to release its reservations. */
BTC_API btc_status btc_poll_match(btc_cache* cache, uint64_t request, btc_plan** plan);
BTC_API btc_plan_view btc_plan_get(const btc_plan* plan);
BTC_API void btc_plan_free(btc_plan* plan);
/* Only BTC_PAGE_LOAD entries take new ownership. READY and JOIN holds already
 * belong to the cache. Request references remain the engine's responsibility. */
BTC_API btc_status btc_bind(btc_cache* cache, btc_plan* plan, const btc_binding* bindings, size_t count);
/* Passing the same nonzero hold for an alias of (pool, slot, generation)
 * reuses that hold. A distinct hold is returned once via RELEASE. */
BTC_API btc_status btc_publish(btc_cache* cache, btc_keys keys, uint64_t committed_tokens,
                               const btc_binding* pages, size_t count);
/* Resident publications keep the committed prefix outside capacity reclaim. */
BTC_API btc_status btc_publish_resident(btc_cache* cache, btc_keys keys, uint64_t committed_tokens,
                                        const btc_binding* pages, size_t count, size_t* resident_keys);

#define BTC_NO_DEVICE_GROUP UINT32_MAX
typedef struct {
    btc_bytes key;
    uint32_t group;
} btc_catalog_item;
typedef void (*btc_catalog_reader)(void* opaque, uint64_t version, const btc_catalog_item* items, size_t count);
/* The visitor receives a copied catalog outside the cache lock. Each logical
 * key is present, with BTC_NO_DEVICE_GROUP when it has no device replica. */
BTC_API btc_status btc_read_device_catalog(btc_cache* cache, btc_catalog_reader reader, void* opaque);
typedef struct {
    btc_bytes key;
    uint32_t group;
    uint32_t added;
} btc_device_change;
typedef void (*btc_device_change_reader)(void* opaque, const btc_device_change* items, size_t count);
BTC_API btc_status btc_track_device_changes(btc_cache* cache, int enabled);
BTC_API btc_status btc_read_device_changes(btc_cache* cache, btc_device_change_reader reader, void* opaque);
BTC_API btc_status btc_device_stats(btc_cache* cache, size_t* keys, size_t* groups);
BTC_API btc_status btc_poll(btc_cache* cache, btc_batch** batch);
/* Counts outstanding work without exposing it for dispatch. */
BTC_API btc_status btc_pending_work(btc_cache* cache, size_t* count);
BTC_API btc_batch_view btc_batch_get(const btc_batch* batch);
BTC_API void btc_batch_free(btc_batch* batch);
/* Mark a participant accepted, or complete it with a receipt. Unsettled
 * participants remain in poll results for replay after a lost response. */
BTC_API btc_status btc_accept(btc_cache* cache, btc_operation id, uint32_t participant);
BTC_API btc_status btc_complete(btc_cache* cache, const btc_receipt* receipts, size_t count);
BTC_API btc_status btc_ack_events(btc_cache* cache, uint64_t cursor);
BTC_API btc_status btc_reclaim(btc_cache* cache, uint64_t pool, size_t pressure,
                               const btc_page* eligible, size_t count);
/* Physical capacity excludes reserved slots. Supply a new complete eligible
 * set before dispatch and before settling worker receipts. */
/* Cheap preflight so a scheduler below high water need not enumerate its holds.
 * Once a balancing round starts, continue sampling through safe release. */
BTC_API btc_status btc_device_balance_needed(btc_cache* cache, uint64_t pool, size_t total, size_t free_slots,
                                              uint8_t* needed);
BTC_API btc_status btc_host_capacity(btc_cache* cache, uint32_t group, size_t* total, size_t* free_slots);
BTC_API btc_status btc_balance_device(btc_cache* cache, uint64_t pool, size_t total, size_t free_slots,
                                      const btc_page* eligible, size_t count);
/* For each binding, return whether its exact device allocation is currently
 * installed at that group and position on this prefix path. */
BTC_API btc_status btc_installed_device_pages(btc_cache* cache, btc_keys keys, const btc_binding* bindings,
                                              size_t count, uint8_t* installed);
/* Copies eligible device positions to Host. If remove_device is set, the
 * device replica is removed only once the entire copy is successful and safe. */
BTC_API btc_status btc_demote(btc_cache* cache, uint64_t pool, size_t pressure,
                              const btc_page* eligible, size_t count, uint32_t remove_device);
BTC_API btc_status btc_reclaim_local(btc_cache* cache, uint64_t pool, uint32_t group, size_t pressure);
BTC_API btc_status btc_set_local_eviction_policy(btc_cache* cache, uint64_t pool, uint32_t policy);
BTC_API btc_status btc_reclaim_related(btc_cache* cache, uint64_t pool, size_t pressure,
                                      const btc_page* eligible, size_t count);
BTC_API btc_status btc_invalidate(btc_cache* cache, const btc_page* pages, size_t count);
BTC_API btc_status btc_cancel(btc_cache* cache, uint64_t request);
BTC_API btc_status btc_drain(btc_cache* cache);
/* A successful close destroys the handle. BTC_PENDING leaves it live. */
BTC_API btc_status btc_close(btc_cache* cache);

/* Disk routes perform local file I/O through this call. Supply one writable
 * host buffer per work page, in work order. The executor sends the receipt. */
BTC_API btc_status btc_execute_disk(btc_cache* cache, btc_operation id,
                                    const btc_buffer* host_buffers, size_t count);

#ifdef __cplusplus
}
#endif

#endif
