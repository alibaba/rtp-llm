#pragma once

#include <cstddef>
#include <functional>
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

struct MatchResult {
    size_t           reuse_length = 0;
    size_t           reuse_blocks = 0;
    BlockIndicesType block_indices;
};

// Pins a scheduler-visible device-cache match until the allocator consumes it.
// The release callback makes rejected/cancelled admission automatically return
// references; consume transfers those references into the destination stream
// without another cache lookup or block copy.
class KVCacheAdmissionReservation {
public:
    using ConsumeFn = std::function<int(BatchKVCacheResource&)>;
    using ReleaseFn = std::function<void()>;

    KVCacheAdmissionReservation(BatchKVCacheResourcePtr preview_resource, ConsumeFn consume, ReleaseFn release):
        preview_resource_(std::move(preview_resource)), consume_(std::move(consume)), release_(std::move(release)) {}

    ~KVCacheAdmissionReservation() {
        if (release_) {
            release_();
        }
    }

    const BatchKVCacheResourcePtr& previewResource() const {
        return preview_resource_;
    }

    int consume(BatchKVCacheResource& destination) {
        if (!consume_) {
            return 0;
        }
        const int reuse_blocks = consume_(destination);
        consume_               = {};
        release_               = {};
        return reuse_blocks;
    }

private:
    BatchKVCacheResourcePtr preview_resource_;
    ConsumeFn               consume_;
    ReleaseFn               release_;
};

using KVCacheAdmissionReservationPtr = std::shared_ptr<KVCacheAdmissionReservation>;

struct MallocInfo {
    BatchKVCacheResourcePtr        batch_kv_cache_resource;
    CompleteTokenIdsPtr            complete_token_ids;
    int64_t                        request_id          = 0;
    bool                           verbose             = true;  // for failed log
    bool                           reuse_cache         = true;
    bool                           enable_device_cache = true;
    KVCacheAdmissionReservationPtr admission_reservation;
    // Sparse tail-group cleanup is only valid for incremental allocation.
    // Prefill init keeps reused prefix slots intact because model-path kernels
    // still read them by prefix_length.
    bool enable_remove_skipped_blocks = true;
    // Override for incrMalloc's seqLength read; -1 = fall back to complete_token_ids->seqLength().
    // Lets the state machine feed the publish-time value instead of racing with the async worker.
    int incr_seq_len_override = -1;

    int incrSeqLen() const;
};

// Keep transient cache pressure distinct from requests that can never fit.
// The scheduler may leave a transiently blocked stream in WAITING, while a
// permanent or internal allocation failure must still terminate the request.
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
        status(success                      ? MallocStatus::NONE :
               status == MallocStatus::NONE ? MallocStatus::INTERNAL_ERROR :
                                              status) {}

    bool success   = false;
    int  reuse_len = 0;

    int64_t      match_cost_time_us = 0;
    MallocStatus status             = MallocStatus::INTERNAL_ERROR;
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
