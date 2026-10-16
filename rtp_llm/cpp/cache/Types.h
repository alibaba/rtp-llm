#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
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

enum class PrefixCacheMediumStatus : uint8_t {
    NOT_CHECKED = 0,
    DISABLED,
    CHECKED,
};

struct PrefixCacheGroupObservation {
    size_t                                 group_set_id{0};
    std::vector<std::string>              member_tags;
    std::array<std::optional<size_t>, 3>  max_pos;
    std::array<std::optional<CacheKeyType>, 3> max_key;
};

struct PrefixCachePlanPoint {
    size_t       index{0};
    CacheKeyType key{0};
    size_t       pos{0};
};

enum class PrefixCachePlanState : uint8_t {
    NOT_FORMED = 0,
    FORMED,
    TERMINAL_UNFORMED,
};

inline const char* prefixCachePlanStateName(PrefixCachePlanState state) {
    switch (state) {
        case PrefixCachePlanState::FORMED:
            return "formed";
        case PrefixCachePlanState::TERMINAL_UNFORMED:
            return "terminal_unformed";
        case PrefixCachePlanState::NOT_FORMED:
            break;
    }
    return "not_formed";
}

enum class PrefixCacheRemoteQueryStatus : uint8_t {
    UNSET = 0,
    NOT_QUERIED,
    HIT,
    MISS,
    ERROR,
};

struct PrefixCacheRemoteObservation {
    PrefixCacheRemoteQueryStatus query_status{PrefixCacheRemoteQueryStatus::NOT_QUERIED};
    std::optional<size_t>        candidate_pos;
    std::optional<bool>          effective;
};

struct PrefixCacheMatchObservation {
    int64_t                                input_tokens{0};
    std::optional<size_t>                  content_pos;
    std::optional<CacheKeyType>            content_key;
    size_t                                 content_node_offset{0};
    bool                                   content_partial{false};
    std::optional<size_t>                  local_valid_pos;
    std::optional<CacheKeyType>            local_valid_key;
    std::array<PrefixCacheMediumStatus, 3> medium_status;
    std::vector<PrefixCacheGroupObservation> groups;
    int64_t                                match_time_us{0};
};

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
    // Upper-layer request preparation has run the existing hash utility.
    // The flag also preserves the first-attempt-only metric event on retries.
    bool cache_keys_prepared = false;
    bool cache_keys_initialized_now = false;
    // Per-malloc attempt start in currentTimeUs() microseconds, captured before
    // upper-layer key hashing. Zero lets direct/legacy manager callers keep
    // their original manager-entry timestamp. Copied by value across async work.
    int64_t malloc_begin_time_us = 0;

    // Immutable global request snapshot. Deferred loads retain it through
    // MallocInfo's by-value callback capture; cache derives local mappings.
    struct RequestLayout {
        CacheKeysType match_keys;  // full global cumulative hashes
        // Immutable global token lengths for this malloc attempt. Deferred
        // materialization must not reread a stream's mutable token cursor.
        int request_seq_len = 0;
        int request_incr_seq_len = 0;
        int request_common_seq_len = 0;
        int request_total_seq_len = 0;
        int request_reserve_step   = 0;
        // First prefill-key initialization only: the old recent/theory metric
        // drops an unaligned final key in every batch before CP sampling.
        // An empty vector is still a valid snapshot, hence optional.
        std::optional<std::vector<CacheKeysType>> theory_window_keys;
        // Owned full-input token snapshot, populated only when complete prefix
        // observability is enabled. Empty means content matching is unformed.
        std::vector<int> token_ids;
    };
    std::shared_ptr<const RequestLayout> request_layout;

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

    MallocResult(bool success, int reuse_len, int64_t match_cost_time_us, std::shared_ptr<AsyncContext> async_context):
        success(success),
        reuse_len(reuse_len),
        match_cost_time_us(match_cost_time_us),
        status(success ? MallocStatus::NONE : MallocStatus::INTERNAL_ERROR),
        async_context(std::move(async_context)) {}

    bool         success            = false;
    int          reuse_len          = 0;
    int64_t      match_cost_time_us = 0;
    MallocStatus status             = MallocStatus::INTERNAL_ERROR;

    std::shared_ptr<AsyncContext> async_context = nullptr;

    // Lower-tier reuse is published only after async_context completes.

    int64_t match_end_time_us          = 0;
    int64_t malloc_begin_time_us       = 0;
    int64_t load_prepare_latency_us    = 0;
    int64_t block_aligned_input_length = 0;
    bool    load_attempted             = false;

    // Diagnostic-only value snapshot; absence means the match observation is
    // not formed rather than a zero endpoint.
    std::optional<PrefixCacheMatchObservation> match_observation;
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
    // Optional request-owned plan-state sink. The cache writes only the state;
    // plan points remain owned by the store task.
    PrefixCachePlanState*   prefix_plan_state{nullptr};
};

}  // namespace rtp_llm
