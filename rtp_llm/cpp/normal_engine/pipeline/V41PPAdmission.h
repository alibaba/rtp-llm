#pragma once

#include <functional>

#include "rtp_llm/cpp/normal_engine/pipeline/V41PPLocalCache.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

// One serialized eager CEP4PP2 admission session. This coordinator uses the
// registered WORLD group before either stage writes caches or waits for an
// activation. Fresh tagged initialization is an explicit later phase; this
// coordinator does not execute a model or send PP frames.
class V41PPAdmission final {
public:
    // Must perform its stage-agreed read-only preflight on every member. This
    // callback must not initialize caches, run layers or exchange activations.
    using WorkerPreflight = std::function<void(const GptModelInputs&, const V41PPCacheBatchPtr&)>;

    V41PPAdmission(std::shared_ptr<KVCacheManager> manager,
                   const ModelConfig&            model,
                   const ParallelismConfig&      parallelism);

    // Replace source page tables/leases only after all eight ranks agree.
    // The caller must retain this batch through transport/consumer completion.
    V41PPCacheBatchPtr prepare(GptModelInputs& inputs);
    V41PPCacheBatchPtr prepareWithWorker(GptModelInputs& inputs, const WorkerPreflight& preflight);
    // Clear fresh requests only, after worker readiness and WORLD agreement.
    // A fatal failure retains pending leases; drain GPU work before aborting.
    void initializeFresh(const V41PPCacheBatchPtr& batch);
    void commit(const V41PPCacheBatchPtr& batch);
    void abort(const V41PPCacheBatchPtr& batch);
    void retire(const std::array<int64_t, 3>& source);
    bool ownsBatch(const V41PPCacheBatchPtr& batch) const;
    size_t activeRequests() const;
    bool failed() const {
        return failed_;
    }

private:
    V41PPCacheBatchPtr prepareImpl(GptModelInputs& inputs, const WorkerPreflight* preflight);

    V41PPLocalCache local_;
    int64_t        vocab_size_;
    bool           root_;
    bool           failed_ = false;
    int64_t        round_ = 0;
    V41PPCacheBatchPtr writing_batch_;
};

}  // namespace rtp_llm
