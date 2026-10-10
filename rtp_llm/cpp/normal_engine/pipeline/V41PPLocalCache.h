#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

class V41PPLocalCache;
class V41PPAdmission;

// Allocation exhaustion is a recoverable admission failure, distinct from an
// unexpected implementation/CUDA error that must poison a coordinated session.
class V41PPAllocationFailure: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A prepared dispatch holds physical pages independently of request retirement.
// The caller must drain GPU work and PP tickets before dropping its last copy.
class V41PPCacheBatch final {
public:
    struct Binding {
        std::array<int64_t, 6>       span;
        int64_t                     local_generation;
        KVCacheAllocationLeasePtr   lease;
        bool                        newly_admitted;
    };

    const std::vector<Binding>& bindings() const {
        return bindings_;
    }
    std::vector<KVCacheAllocationLeasePtr> leases() const;
    torch::Tensor kernelBlockTable() const;
    bool initialized() const {
        return initialized_;
    }

private:
    friend class V41PPLocalCache;
    friend class V41PPAdmission;
    struct Allocation {
        ~Allocation();
        std::shared_ptr<KVCacheManager> manager;
        BatchKVCacheResourcePtr         resource;
        CompleteTokenIdsPtr             tokens;
        KVCacheAllocationLeasePtr       lease;
        int64_t                         local_generation;
        int64_t                         request_id;
    };
    struct Request {
        std::array<int64_t, 3>       source;
        int64_t                     next_chunk;
        int64_t                     next_position;
        std::shared_ptr<Allocation> allocation;
    };

    std::shared_ptr<const char>                domain_;
    std::shared_ptr<const CacheTopology>       topology_;
    std::map<int64_t, Request>                 next_requests_;
    std::vector<Binding>                      bindings_;
    int64_t                                   source_high_water_ = 0;
    bool                                      committed_ = false;
    bool                                      worker_ready_ = false;
    bool                                      initialized_ = false;
};

using V41PPCacheBatchPtr = std::shared_ptr<V41PPCacheBatch>;

// Serialized, short eager CEP4PP2 allocation admission. No CUDA writes or
// collectives occur here. The executor must coordinate rejection across ranks
// and commit only after a successful import/forward. This is not a scheduler.
class V41PPLocalCache final {
public:
    V41PPLocalCache(std::shared_ptr<KVCacheManager> manager,
                    const ModelConfig&            model,
                    const ParallelismConfig&      parallelism);
    V41PPLocalCache(const V41PPLocalCache&)            = delete;
    V41PPLocalCache& operator=(const V41PPLocalCache&) = delete;

    V41PPCacheBatchPtr prepare(const torch::Tensor& spans,
                               const std::vector<KVCacheAllocationLeasePtr>& root_leases = {});
    void commit(const V41PPCacheBatchPtr& batch);
    void abort(const V41PPCacheBatchPtr& batch);
    void retire(const std::array<int64_t, 3>& source);
    bool ownsBatch(const V41PPCacheBatchPtr& batch) const;
    size_t activeRequests() const {
        return requests_.size();
    }

private:
    friend class V41PPAdmission;
    using Allocation = V41PPCacheBatch::Allocation;
    using Request = V41PPCacheBatch::Request;
    using InitializationOps = std::vector<std::pair<torch::Tensor, torch::Tensor>>;

    // Construct every tagged operation before the coordinator agrees to write.
    InitializationOps prepareInitialization(const V41PPCacheBatchPtr& batch) const;

    std::shared_ptr<Allocation> allocate(int64_t request_id, const KVCacheAllocationLeasePtr& root_lease);
    void requireLease(const KVCacheAllocationLeasePtr& lease) const;
    void requirePending(const V41PPCacheBatchPtr& batch) const;

    std::shared_ptr<KVCacheManager>      manager_;
    std::shared_ptr<const CacheTopology> topology_;
    std::shared_ptr<const char>          domain_ = std::make_shared<const char>(0);
    std::map<int64_t, Request>           requests_;
    std::weak_ptr<V41PPCacheBatch>        pending_;
    bool                                borrow_root_;
    int64_t                             local_generation_ = 0;
    int64_t                             source_high_water_ = 0;
};

}  // namespace rtp_llm
