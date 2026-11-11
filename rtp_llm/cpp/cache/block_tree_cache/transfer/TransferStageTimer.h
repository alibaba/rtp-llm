#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string_view>
#include <utility>
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
        if (source == Tier::HOST && target == Tier::DEVICE)
            path = "host_to_device";
        if (source == Tier::DEVICE && target == Tier::HOST)
            path = "device_to_host";
        if (source == Tier::HOST && target == Tier::DISK)
            path = "host_to_disk";
        return {target == Tier::DEVICE ? "load" : "evict", tierName(source), tierName(target), path};
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
// Value scopes default to thread execution; shared timers default to lifecycle.
// Queue/resource waits explicitly select Wait at their observation sites.
class TransferStageTimer {
    struct ReporterRef {
        const BlockTreeCacheMetricsReporter* reporter;
    };

public:
    TransferStageTimer() noexcept                            = default;
    TransferStageTimer(const TransferStageTimer&)            = delete;
    TransferStageTimer& operator=(const TransferStageTimer&) = delete;
    // Ownership may move between the submitter and its Entry, never concurrently
    // with finish(). Disarm the source so its destructor cannot report failure.
    TransferStageTimer(TransferStageTimer&& other) noexcept {
        moveFrom(other);
    }
    TransferStageTimer& operator=(TransferStageTimer&& other) noexcept {
        if (this != &other) {
            finish(false);
            moveFrom(other);
        }
        return *this;
    }
    explicit operator bool() const noexcept {
        return enabled_;
    }

    TransferStageTimer(std::shared_ptr<BlockTreeCacheMetricsReporter> reporter,
                       const char*                                    stage,
                       int64_t                                        request_id,
                       size_t                                         begin,
                       size_t                                         end,
                       int64_t                                        rank        = -1,
                       const char*                                    scope       = "worker_batch",
                       uint64_t                                       transfer_id = 0,
                       CacheStageContext                              identity    = {},
                       AsyncProfileKind                               kind = AsyncProfileKind::Execution) noexcept:
        TransferStageTimer(
            ReporterRef{reporter.get()}, stage, request_id, begin, end, rank, scope, transfer_id, identity, kind) {
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
                       CacheStageContext                    identity    = {},
                       AsyncProfileKind                     kind        = AsyncProfileKind::Execution) noexcept:
        TransferStageTimer(
            ReporterRef{&reporter}, stage, request_id, begin, end, rank, scope, transfer_id, identity, kind) {}

    // A WorkLoop keeps its reporter alive until every Entry and callback drains.
    // A nullable borrowed observer also supports profiling without a reporter.
    TransferStageTimer(const BlockTreeCacheMetricsReporter* reporter,
                       const char*                          stage,
                       int64_t                              request_id,
                       size_t                               begin,
                       size_t                               end,
                       int64_t                              rank        = -1,
                       const char*                          scope       = "worker_batch",
                       uint64_t                             transfer_id = 0,
                       CacheStageContext                    identity    = {},
                       AsyncProfileKind                     kind        = AsyncProfileKind::Execution) noexcept:
        TransferStageTimer(
            ReporterRef{reporter}, stage, request_id, begin, end, rank, scope, transfer_id, identity, kind) {}

private:
    void moveFrom(TransferStageTimer& other) noexcept {
        reporter_owner_ = std::move(other.reporter_owner_);
        reporter_       = other.reporter_;
        stage_          = other.stage_;
        rank_           = other.rank_;
        scope_          = other.scope_;
        identity_       = other.identity_;
        start_us_       = other.start_us_;
        span_           = std::move(other.span_);
        finished_.store(other.finished_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        enabled_ = std::exchange(other.enabled_, false);
    }
    TransferStageTimer(ReporterRef       reporter,
                       const char*       stage,
                       int64_t           request_id,
                       size_t            begin,
                       size_t            end,
                       int64_t           rank,
                       const char*       scope,
                       uint64_t          transfer_id,
                       CacheStageContext identity,
                       AsyncProfileKind  kind) noexcept:
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
                                                      identity.operation,
                                                      kind) :
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
                                           CacheStageContext                              identity    = {},
                                           AsyncProfileKind kind = AsyncProfileKind::Lifecycle) noexcept {
        if ((!reporter || !reporter->stageMetricsEnabled()) && !AsyncProfileSession::enabled())
            return {};
        try {
            if (allocation_probe_)
                allocation_probe_(stage);
            return SharedTransferStageTimer(std::make_shared<TransferStageTimer>(
                std::move(reporter), stage, request_id, begin, end, rank, scope, transfer_id, identity, kind));
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
                                           CacheStageContext                    identity    = {},
                                           AsyncProfileKind kind = AsyncProfileKind::Lifecycle) noexcept {
        if (!reporter.stageMetricsEnabled() && !AsyncProfileSession::enabled())
            return {};
        try {
            if (allocation_probe_)
                allocation_probe_(stage);
            return SharedTransferStageTimer(std::make_shared<TransferStageTimer>(
                reporter, stage, request_id, begin, end, rank, scope, transfer_id, identity, kind));
        } catch (...) {
            return {};
        }
    }
    // Return the first observation's elapsed time for consumers sharing this
    // boundary. Disabled/already-finished timers return -1 without a clock read.
    int64_t finish(bool success) noexcept {
        if (!enabled_ || finished_.load(std::memory_order_relaxed)
            || finished_.exchange(true, std::memory_order_acq_rel))
            return -1;
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
        return end_us - start_us_;
    }

private:
    // Per-thread fault seam, so tests fail observer allocations without failing
    // unrelated business allocations or other requests.
    inline static thread_local void (*allocation_probe_)(const char*) = nullptr;
    friend class TransferStageTimerTestPeer;
    std::shared_ptr<BlockTreeCacheMetricsReporter> reporter_owner_;
    const BlockTreeCacheMetricsReporter*           reporter_{nullptr};
    const char*                                    stage_{nullptr};
    int64_t                                        rank_{-1};
    const char*                                    scope_{"worker_batch"};
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
