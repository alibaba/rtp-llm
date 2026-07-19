#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

namespace rtp_llm {

class CompleteTokenIds;
using CompleteTokenIdsPtr = std::shared_ptr<CompleteTokenIds>;

class AsyncContext;

typedef int32_t          GroupIdType;
typedef std::vector<int> LayerIdsType;

struct BlockAddrInfo {
    void* kv_addr       = nullptr;
    void* kv_scale_addr = nullptr;
};

struct KVCacheInfo {
    size_t                    available_kv_cache = 0;
    size_t                    total_kv_cache     = 0;
    size_t                    block_size         = 0;
    std::vector<CacheKeyType> cached_keys;
    int64_t                   version = -1;
};

struct BlockIdPair {
    BlockIdxType src;
    BlockIdxType dst;
};

// for p2p connector when TP settings of prefill & decode are different.
struct KVPartitionBytes {
    size_t k_off = 0;
    size_t k_sz  = 0;
    size_t v_off = 0;
    size_t v_sz  = 0;
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

    // Async load_back context produced when the allocator commits the deferred
    // load_back (see LoadBackTicket); nullptr when no load_back was triggered
    // (device cache disabled, no host/disk hit, or the ticket was aborted).
    std::shared_ptr<AsyncContext> async_context = nullptr;
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
