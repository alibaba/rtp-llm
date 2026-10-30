#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"

namespace rtp_llm {

struct CacheStageContext {
    const char* operation{"load"};
    const char* source{"DISK"};
    const char* target{"DEVICE"};
    const char* path{"disk_to_device"};

    static CacheStageContext local(Tier source, Tier target) {
        const char* path = "unknown";
        if (source == Tier::DISK && target == Tier::DEVICE)
            path = "disk_to_device";
        if (source == Tier::DEVICE && target == Tier::DISK)
            path = "device_to_disk";
        if (source == Tier::HOST && target == Tier::DEVICE)
            path = "host_to_device";
        if (source == Tier::DEVICE && target == Tier::HOST)
            path = "device_to_host";
        if (source == Tier::HOST && target == Tier::DISK)
            path = "host_to_disk";
        if (source == Tier::DISK && target == Tier::HOST)
            path = "disk_to_host";
        return {target == Tier::DEVICE ? "load" : "store", tierName(source), tierName(target), path};
    }
    static CacheStageContext storage(const char* operation) {
        return {operation,
                operation == std::string_view("write") ? "DEVICE" : "REMOTE",
                operation == std::string_view("write") ? "REMOTE" : "DEVICE",
                "storage_backend"};
    }
    // A logical load may settle HOST and DISK sources together. Do not charge
    // that completion to just the last physical transfer direction.
    static CacheStageContext workflow(const char*                            operation,
                                      const std::vector<TransferDescriptor>& descriptors) noexcept {
        if (descriptors.empty())
            return {operation, "NONE", "NONE", "unknown"};
        auto identity      = local(descriptors.front().source_tier, descriptors.front().target_tier);
        identity.operation = operation;
        for (const auto& descriptor : descriptors) {
            if (descriptor.source_tier != descriptors.front().source_tier)
                identity.source = "MIXED";
            if (descriptor.target_tier != descriptors.front().target_tier)
                identity.target = "MIXED";
        }
        if (identity.source == std::string_view("MIXED") || identity.target == std::string_view("MIXED"))
            identity.path = identity.target == std::string_view("DEVICE") ? "mixed_to_device" : "mixed";
        return identity;
    }
};

class TransferStageTimer;
class SharedTransferStageTimer {
public:
    SharedTransferStageTimer() = default;
    explicit operator bool() const noexcept {
        return state_ != nullptr;
    }
    void finish(bool success) const noexcept;

private:
    explicit SharedTransferStageTimer(std::shared_ptr<TransferStageTimer> state) noexcept: state_(std::move(state)) {}
    std::shared_ptr<TransferStageTimer> state_;
    friend class TransferStageTimer;
};

// Value timers never allocate. Shared timers are best-effort: creation failure
// yields an empty handle, and every caller continues its original business work.
class TransferStageTimer {
    struct ReporterRef {
        const BlockTreeCacheMetricsReporter* reporter;
    };

public:
    TransferStageTimer(std::shared_ptr<BlockTreeCacheMetricsReporter> reporter,
                       const char*                                    stage,
                       int64_t                                        request_id,
                       size_t                                         begin,
                       size_t                                         end,
                       int64_t                                        rank        = -1,
                       const char*                                    scope       = "worker_batch",
                       uint64_t                                       transfer_id = 0,
                       CacheStageContext                              identity    = {}) noexcept:
        TransferStageTimer(
            ReporterRef{reporter.get()}, stage, request_id, begin, end, rank, scope, transfer_id, identity) {
        reporter_owner_ = std::move(reporter);
    }
    // Borrowed observers are used only by workflows whose owner drains all
    // callbacks before destroying the reporter (BlockTreeCache destruction).
    TransferStageTimer(const BlockTreeCacheMetricsReporter& reporter,
                       const char*                          stage,
                       int64_t                              request_id,
                       size_t                               begin,
                       size_t                               end,
                       int64_t                              rank        = -1,
                       const char*                          scope       = "worker_batch",
                       uint64_t                             transfer_id = 0,
                       CacheStageContext                    identity    = {}) noexcept:
        TransferStageTimer(ReporterRef{&reporter}, stage, request_id, begin, end, rank, scope, transfer_id, identity) {}

private:
    TransferStageTimer(ReporterRef       reporter,
                       const char*       stage,
                       int64_t           request_id,
                       size_t            begin,
                       size_t            end,
                       int64_t           rank,
                       const char*       scope,
                       uint64_t          transfer_id,
                       CacheStageContext identity) noexcept:
        reporter_(reporter.reporter),
        stage_(stage),
        rank_(rank),
        scope_(scope),
        identity_(identity),
        enabled_((reporter_ && reporter_->stageMetricsEnabled()) || AsyncProfileSession::enabled()),
        start_us_(enabled_ ? AsyncProfileSession::steadyNowUs() : 0),
        span_(enabled_ ? AsyncProfileSpan::fromSteady(stage,
                                                      request_id,
                                                      begin,
                                                      end,
                                                      start_us_,
                                                      rank,
                                                      transfer_id,
                                                      identity.path,
                                                      scope,
                                                      identity.operation) :
                         AsyncProfileSpan{}) {}

public:
    ~TransferStageTimer() {
        finish(false);
    }
    static SharedTransferStageTimer create(std::shared_ptr<BlockTreeCacheMetricsReporter> reporter,
                                           const char*                                    stage,
                                           int64_t                                        request_id,
                                           size_t                                         begin,
                                           size_t                                         end,
                                           int64_t                                        rank        = -1,
                                           const char*                                    scope       = "worker_batch",
                                           uint64_t                                       transfer_id = 0,
                                           CacheStageContext                              identity    = {}) noexcept {
        if ((!reporter || !reporter->stageMetricsEnabled()) && !AsyncProfileSession::enabled())
            return {};
        try {
            if (allocation_probe_)
                allocation_probe_(stage);
            return SharedTransferStageTimer(std::make_shared<TransferStageTimer>(
                std::move(reporter), stage, request_id, begin, end, rank, scope, transfer_id, identity));
        } catch (...) {
            return {};
        }
    }
    static SharedTransferStageTimer create(const BlockTreeCacheMetricsReporter& reporter,
                                           const char*                          stage,
                                           int64_t                              request_id,
                                           size_t                               begin,
                                           size_t                               end,
                                           int64_t                              rank        = -1,
                                           const char*                          scope       = "worker_batch",
                                           uint64_t                             transfer_id = 0,
                                           CacheStageContext                    identity    = {}) noexcept {
        if (!reporter.stageMetricsEnabled() && !AsyncProfileSession::enabled())
            return {};
        try {
            if (allocation_probe_)
                allocation_probe_(stage);
            return SharedTransferStageTimer(std::make_shared<TransferStageTimer>(
                reporter, stage, request_id, begin, end, rank, scope, transfer_id, identity));
        } catch (...) {
            return {};
        }
    }
    void finish(bool success) noexcept {
        if (!enabled_ || finished_.load(std::memory_order_relaxed)
            || finished_.exchange(true, std::memory_order_acq_rel))
            return;
        const auto end_us = AsyncProfileSession::steadyNowUs();
        span_.finishAtSteady(end_us);
        if (reporter_)
            reporter_->reportTransferStage(stage_,
                                           end_us - start_us_,
                                           success,
                                           rank_,
                                           scope_,
                                           identity_.operation,
                                           identity_.source,
                                           identity_.target);
    }

private:
    // Per-thread fault seam, so tests fail observer allocations without failing
    // unrelated business allocations or other requests.
    inline static thread_local void (*allocation_probe_)(const char*) = nullptr;
    friend class TransferStageTimerTestPeer;
    std::shared_ptr<BlockTreeCacheMetricsReporter> reporter_owner_;
    const BlockTreeCacheMetricsReporter*           reporter_;
    const char*                                    stage_;
    int64_t                                        rank_;
    const char*                                    scope_;
    CacheStageContext                              identity_;
    bool                                           enabled_{false};
    int64_t                                        start_us_{0};
    AsyncProfileSpan                               span_;
    std::atomic<bool>                              finished_{false};
};
inline void SharedTransferStageTimer::finish(bool success) const noexcept {
    if (state_)
        state_->finish(success);
}

}  // namespace rtp_llm
