#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

namespace rtp_llm {

class CompleteTokenIds;
using CompleteTokenIdsPtr = std::shared_ptr<CompleteTokenIds>;

typedef int32_t          GroupIdType;
typedef std::vector<int> LayerIdsType;

enum class KVCacheChangeType : uint8_t {
    STORED  = 0,
    REMOVED = 1,
};

enum class KVCacheSnapshotReason : uint8_t {
    NONE                = 0,
    FORCED              = 1,
    HISTORY_GAP         = 2,
    FUTURE_CURSOR       = 3,
    GENERATION_MISMATCH = 4,
    INVALID_CURSOR      = 5,
};

struct KVCacheChange {
    int64_t                  version = -1;
    KVCacheChangeType        type    = KVCacheChangeType::STORED;
    CacheKeyType             cache_key{0};
    std::vector<GroupIdType> group_ids;
};

struct KVCacheSnapshotEntry {
    CacheKeyType             cache_key{0};
    std::vector<GroupIdType> group_ids;
};

struct BlockAddrInfo {
    void* kv_addr       = nullptr;
    void* kv_scale_addr = nullptr;
};

struct KVCacheInfo {
    size_t                            available_kv_cache = 0;
    size_t                            total_kv_cache     = 0;
    size_t                            block_size         = 0;
    std::vector<CacheKeyType>         cached_keys;
    int64_t                           version                      = -1;
    int64_t                           cache_event_version          = -1;
    uint32_t                          cache_event_protocol_version = 0;
    int64_t                           next_cache_event_version     = -1;
    int64_t                           oldest_cache_event_version   = -1;
    uint64_t                          cache_event_generation       = 0;
    bool                              cache_event_reset_required   = false;
    bool                              cache_event_has_more         = false;
    KVCacheSnapshotReason             cache_event_snapshot_reason  = KVCacheSnapshotReason::NONE;
    std::vector<KVCacheChange>        cache_events;
    std::vector<KVCacheSnapshotEntry> cache_event_snapshot;
};

struct BlockIdPair {
    BlockIdxType src;
    BlockIdxType dst;
};

struct TaggedBlockIdPair {
    std::string  tag;
    BlockIdxType src;
    BlockIdxType dst;
};

// Process-local tensor representation. group_id is resolved from a stable tag
// immediately before execution and is never used as an external identity.
struct GroupBlockIdPair {
    GroupIdType  group_id;
    BlockIdxType src;
    BlockIdxType dst;
};

static_assert(sizeof(GroupBlockIdPair) == 3 * sizeof(int32_t),
              "GroupBlockIdPair must match the three-column int32 tensor layout");

struct MatchResult {
    size_t           reuse_length = 0;
    size_t           reuse_blocks = 0;
    BlockIndicesType block_indices;
};

struct MallocInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    int64_t                 request_id          = 0;
    bool                    verbose             = true;  // for failed log
    bool                    reuse_cache         = true;
    bool                    enable_device_cache = true;
    // Sparse tail-group cleanup is only valid for incremental allocation.
    // Prefill init keeps reused prefix slots intact because model-path kernels
    // still read them by prefix_length.
    bool enable_remove_skipped_blocks = true;
    // Override for incrMalloc's seqLength read; -1 = fall back to complete_token_ids->seqLength().
    // Lets the state machine feed the publish-time value instead of racing with the async worker.
    int incr_seq_len_override = -1;

    int incrSeqLen() const;
};

struct MallocResult {
    bool success;
    int  reuse_len;

    int64_t match_cost_time_us = 0;
};

struct FreeInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;

    int64_t request_id = 0;
};

struct InsertInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    bool                    is_resident;
};

}  // namespace rtp_llm
