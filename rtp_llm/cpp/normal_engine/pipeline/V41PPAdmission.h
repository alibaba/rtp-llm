#pragma once

#include <functional>

#include "rtp_llm/cpp/normal_engine/pipeline/V41PPLocalCache.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

// Wire layout of the per-round admission packet: one int64 vector per rank, gathered across the
// agreement scope. Header-level because this layout is protocol, not implementation detail -- the
// test binding exposes kV41AdmissionPacketSize so the mocked-gather tests can assert the real
// shape instead of hardcoding it. (The pre-existing red targets D2/D3 in guide-deviations.md are
// exactly that hardcoding, broken when 12cfb351c3 moved kV41MaxInitialPPPrefillTokens 256 -> 32768.)
inline constexpr int64_t kV41AdmissionTokenLimit   = 2 * kV41MaxInitialPPPrefillTokens;
inline constexpr int64_t kV41AdmissionSpanOffset   = 4;
inline constexpr int64_t kV41AdmissionTokenOffset  = kV41AdmissionSpanOffset + 12;
inline constexpr int64_t kV41AdmissionWindowOffset = kV41AdmissionTokenOffset + kV41AdmissionTokenLimit;
inline constexpr int64_t kV41AdmissionPacketSize   = kV41AdmissionWindowOffset + 4 * kV41AdmissionTokenLimit;

// One serialized eager CEP4PP2 admission session. This coordinator uses the
// registered WORLD group before either stage writes caches or waits for an
// activation. Fresh tagged initialization is an explicit later phase; this
// coordinator does not execute a model or send PP frames.
class V41PPAdmission final {
public:
    // Must perform its stage-agreed read-only preflight on every member. This
    // callback must not initialize caches, run layers or exchange activations.
    using WorkerPreflight = std::function<void(const GptModelInputs&, const V41PPCacheBatchPtr&)>;

    // `async_overlap` selects the asynchronous pipeline contract for the retained
    // local dispatch chain (see V41PPLocalCache). The synchronous default keeps the
    // serialized WORLD admission protocol byte-for-byte for A/B and regressions.
    V41PPAdmission(std::shared_ptr<KVCacheManager> manager,
                   const ModelConfig&              model,
                   const ParallelismConfig&        parallelism,
                   bool                            async_overlap = false);

    // Replace source page tables/leases only after all eight ranks agree.
    // The caller must retain this batch through transport/consumer completion.
    V41PPCacheBatchPtr prepare(GptModelInputs& inputs);
    V41PPCacheBatchPtr prepareWithWorker(GptModelInputs& inputs, const WorkerPreflight& preflight);
    // Clear fresh requests only, after worker readiness and WORLD agreement.
    // A fatal failure retains pending leases; drain GPU work before aborting.
    void initializeFresh(const V41PPCacheBatchPtr& batch);
    // Begin before layer execution/activation receives. The returned transfer
    // round counts successful executions independently of rejected admissions.
    int64_t beginExecution(const V41PPCacheBatchPtr& batch);
    // The caller must drain GPU work and transport/consumers before completion.
    // Any failed completion poisons WORLD and retains the batch until abort.
    void   completeExecution(const V41PPCacheBatchPtr& batch, bool successful);
    void   commit(const V41PPCacheBatchPtr& batch);
    void   abort(const V41PPCacheBatchPtr& batch);
    void   retire(const std::array<int64_t, 3>& source);
    bool   ownsBatch(const V41PPCacheBatchPtr& batch) const;
    size_t activeRequests() const;
    bool   failed() const {
        return failed_;
    }

private:
    V41PPCacheBatchPtr prepareImpl(GptModelInputs& inputs, const WorkerPreflight* preflight);

    // Agreement scope for every per-chunk packet. Synchronous: WORLD (all eight ranks),
    // which is what pins the two stages to the same chunk. Asynchronous: STAGE (the four
    // ranks of this PP stage), so cross-stage progress is ordered only by the plan /
    // activation / result channels.
    ParallelMode agreementMode() const {
        return async_overlap_ ? ParallelMode::STAGE : ParallelMode::WORLD;
    }
    int64_t agreementPeers() const {
        return async_overlap_ ? stage_size_ : 8;
    }

    V41PPLocalCache local_;
    int64_t         vocab_size_;
    bool            root_;
    bool            async_overlap_ = false;
    int64_t         stage_size_    = 1;
    // This coordinator's PP stage. Only the test-only injection point in initializeFresh
    // needs it; the agreement scope is derived from async_overlap_ and stage_size_.
    int64_t            pp_rank_         = 0;
    bool               failed_          = false;
    int64_t            round_           = 0;
    int64_t            execution_round_ = 0;
    V41PPCacheBatchPtr writing_batch_;
};

}  // namespace rtp_llm
