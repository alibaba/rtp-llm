#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

namespace rtp_llm {

class AsyncContext;

class CompleteTokenIds;
using CompleteTokenIdsPtr = std::shared_ptr<CompleteTokenIds>;

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

struct MallocInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    int64_t                 request_id          = 0;
    bool                    verbose             = true;  // for failed log
    bool                    reuse_cache         = true;
    bool                    enable_cache_lookup = true;
    // Sparse tail-group cleanup is only valid for incremental allocation.
    // Prefill init keeps reused prefix slots intact because model-path kernels
    // still read them by prefix_length.
    bool enable_remove_skipped_blocks = true;
    // Override for incrMalloc's seqLength read; -1 = fall back to complete_token_ids->seqLength().
    // Lets the state machine feed the publish-time value instead of racing with the async worker.
    int incr_seq_len_override = -1;

    int incrSeqLen() const;
};

// Separates "the pools are momentarily full, retry later" from "this request can never fit".
// A RETRYABLE failure keeps the stream WAITING instead of erroring it out under cache pressure.
enum class MallocStatus : uint8_t {
    NONE = 0,
    RETRYABLE_RESOURCE_EXHAUSTED,
    PERMANENT_RESOURCE_EXHAUSTED,
    INTERNAL_ERROR,
};

struct MallocResult {
    MallocResult() = default;

    constexpr MallocResult(bool         success,
                           int          reuse_len,
                           int64_t      match_cost_time_us = 0,
                           MallocStatus status             = MallocStatus::NONE):
        success(success),
        reuse_len(reuse_len),
        match_cost_time_us(match_cost_time_us),
        // A success never carries a failure status, and a failure that forgot to
        // classify itself is surfaced as an internal error rather than NONE.
        status(success                      ? MallocStatus::NONE :
               status == MallocStatus::NONE ? MallocStatus::INTERNAL_ERROR :
                                              status) {}

    MallocResult(bool                          success,
                 int                           reuse_len,
                 int64_t                       match_cost_time_us,
                 std::shared_ptr<AsyncContext> async_context,
                 int                           host_reuse_len = 0,
                 int                           disk_reuse_len = 0):
        success(success),
        reuse_len(reuse_len),
        match_cost_time_us(match_cost_time_us),
        status(success ? MallocStatus::NONE : MallocStatus::INTERNAL_ERROR),
        async_context(std::move(async_context)),
        host_reuse_len(host_reuse_len),
        disk_reuse_len(disk_reuse_len) {}

    bool         success            = false;
    int          reuse_len          = 0;
    int64_t      match_cost_time_us = 0;
    MallocStatus status             = MallocStatus::INTERNAL_ERROR;

    std::shared_ptr<AsyncContext> async_context = nullptr;

    int host_reuse_len = 0;
    int disk_reuse_len = 0;

    int64_t match_end_time_us          = 0;
    int64_t malloc_begin_time_us       = 0;
    int64_t load_prepare_latency_us    = 0;
    int64_t block_aligned_input_length = 0;
    bool    load_attempted             = false;
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
    Tier                    target_tier{Tier::DEVICE};
    bool                    write_remote{true};
};

}  // namespace rtp_llm
