#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"

namespace rtp_llm {

struct CacheStageContext {
    const char* operation{"load"};
    const char* source{"DISK"};
    const char* target{"DEVICE"};
    const char* path{"disk_to_device"};
    const BlockTreeCacheMetricsReporter* reporter{nullptr};
    int64_t                              request_id{-1};
    uint64_t                             transfer_id{0};
    size_t                               block_count{0};
    bool                                 enabled{false};

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
    static CacheStageContext workflow(const char* operation, DescriptorView descriptors) noexcept {
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

    // Freeze eligibility before observation-only work. Later profiling sessions
    // must not turn a disabled workflow into a partially observed one.
    static CacheStageContext forWorkflow(const BlockTreeCacheMetricsReporter& reporter,
                                         DescriptorView                       descriptors,
                                         const char*                          operation,
                                         int64_t                              request_id,
                                         uint64_t                             transfer_id) noexcept {
        const bool enabled = reporter.stageMetricsEnabled() || AsyncProfileSession::enabled();
        auto       context =
            enabled ? workflow(operation, descriptors) : CacheStageContext{operation, "NONE", "NONE", "unknown"};
        context.reporter    = &reporter;
        context.request_id  = request_id;
        context.transfer_id = transfer_id;
        context.block_count = descriptors.size();
        context.enabled     = enabled;
        return context;
    }
    static CacheStageContext
    forTask(const BlockTreeCacheMetricsReporter& reporter, const TransferTask& task, const char* operation) noexcept {
        return forWorkflow(reporter, task.descriptors(), operation, task.requestId().value_or(-1), task.traceId());
    }
    // Physical transfers have already been partitioned by direction.
    static CacheStageContext forTransfer(const BlockTreeCacheMetricsReporter& reporter,
                                         const TransferTask&                  task,
                                         const char*                          operation) noexcept {
        const bool enabled = reporter.stageMetricsEnabled() || AsyncProfileSession::enabled();
        auto       context = CacheStageContext{operation, "NONE", "NONE", "unknown"};
        if (enabled && !task.descriptors().empty()) {
            const auto& first = task.descriptors().front();
            context           = local(first.source_tier, first.target_tier);
            context.operation = operation;
        }
        context.reporter    = &reporter;
        context.request_id  = task.requestId().value_or(-1);
        context.transfer_id = task.traceId();
        context.block_count = task.descriptors().size();
        context.enabled     = enabled;
        return context;
    }
    static CacheStageContext forLock(const BlockTreeCacheMetricsReporter* reporter, const char* operation) noexcept {
        auto context     = CacheStageContext{operation, "NONE", "NONE", "cache_lock"};
        context.reporter = reporter;
        context.enabled  = (reporter && reporter->stageMetricsEnabled()) || AsyncProfileSession::enabled();
        return context;
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
    void discard() const noexcept;

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
    // Disabled timers have no boundary; preserve the default-clock sentinel.
    int64_t startSteadyUs() const noexcept {
        return enabled_ ? start_us_ : -1;
    }

    // Synchronous scopes opt in to normal/exceptional exit handling. Async
    // owners retain the default: an unfinished timer is a failed observation.
    void finishOnScopeExit(bool success = true) noexcept {
        success_    = success;
        exceptions_ = std::uncaught_exceptions();
    }
    void setSuccess(bool success) noexcept {
        success_ = success;
    }

    TransferStageTimer(const CacheStageContext& context,
                       const char*              stage,
                       const char*              scope,
                       AsyncProfileKind         kind = AsyncProfileKind::Execution) noexcept:
        TransferStageTimer(ReporterRef{context.reporter},
                           stage,
                           context.request_id,
                           0,
                           context.block_count,
                           -1,
                           scope,
                           context.transfer_id,
                           context,
                           kind,
                           -1,
                           context.enabled) {}

    static TransferStageTimer scope(const CacheStageContext& context,
                                    const char*              stage,
                                    const char*              scope,
                                    bool                     success = true,
                                    AsyncProfileKind         kind    = AsyncProfileKind::Execution) noexcept {
        TransferStageTimer timer(context, stage, scope, kind);
        timer.finishOnScopeExit(success);
        return timer;
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
                       AsyncProfileKind                     kind        = AsyncProfileKind::Execution,
                       const AsyncProfileSpan*              parent      = nullptr,
                       int64_t                              start_us    = -1) noexcept:
        TransferStageTimer(ReporterRef{reporter},
                           stage,
                           request_id,
                           begin,
                           end,
                           rank,
                           scope,
                           transfer_id,
                           identity,
                           kind,
                           start_us,
                           true,
                           parent) {}

private:
    void moveFrom(TransferStageTimer& other) noexcept {
        reporter_owner_ = std::move(other.reporter_owner_);
        reporter_       = other.reporter_;
        stage_          = other.stage_;
        rank_           = other.rank_;
        scope_          = other.scope_;
        operation_      = other.operation_;
        source_         = other.source_;
        target_         = other.target_;
        start_us_       = other.start_us_;
        success_        = other.success_;
        exceptions_     = other.exceptions_;
        span_           = std::move(other.span_);
        finished_.store(other.finished_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        enabled_ = std::exchange(other.enabled_, false);
    }
    TransferStageTimer(ReporterRef             reporter,
                       const char*             stage,
                       int64_t                 request_id,
                       size_t                  begin,
                       size_t                  end,
                       int64_t                 rank,
                       const char*             scope,
                       uint64_t                transfer_id,
                       CacheStageContext       identity,
                       AsyncProfileKind        kind,
                       int64_t                 start_us = -1,
                       bool                    eligible = true,
                       const AsyncProfileSpan* parent   = nullptr) noexcept:
        reporter_(reporter.reporter),
        stage_(stage),
        rank_(rank),
        scope_(scope),
        operation_(identity.operation),
        source_(identity.source),
        target_(identity.target),
        enabled_(eligible
                 && ((reporter_ && reporter_->stageMetricsEnabled())
                     || (parent ? parent->active() : AsyncProfileSession::enabled()))),
        start_us_(enabled_ ? (start_us >= 0 ? start_us : AsyncProfileSession::steadyNowUs()) : 0),
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
                                                      kind,
                                                      parent) :
                         AsyncProfileSpan{}) {}

public:
    const AsyncProfileSpan& trace() const noexcept {
        return span_;
    }
    ~TransferStageTimer() {
        finish(exceptions_ >= 0 && success_ && exceptions_ == std::uncaught_exceptions());
    }
    static TransferStageTimer atSteady(const CacheStageContext& context,
                                       const char*              stage,
                                       const char*              scope,
                                       int64_t                  start_us,
                                       AsyncProfileKind         kind = AsyncProfileKind::Execution) noexcept {
        return TransferStageTimer(ReporterRef{context.reporter},
                                  stage,
                                  context.request_id,
                                  0,
                                  context.block_count,
                                  -1,
                                  scope,
                                  context.transfer_id,
                                  context,
                                  kind,
                                  start_us,
                                  context.enabled);
    }
    static SharedTransferStageTimer create(const CacheStageContext& context,
                                           const char*              stage,
                                           const char*              scope,
                                           AsyncProfileKind         kind = AsyncProfileKind::Lifecycle) noexcept {
        if (!context.enabled)
            return {};
        if (context.reporter)
            return create(*context.reporter,
                          stage,
                          context.request_id,
                          0,
                          context.block_count,
                          -1,
                          scope,
                          context.transfer_id,
                          context,
                          kind);
        return create(std::shared_ptr<BlockTreeCacheMetricsReporter>{},
                      stage,
                      context.request_id,
                      0,
                      context.block_count,
                      -1,
                      scope,
                      context.transfer_id,
                      context,
                      kind);
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
    // Disabled/already-finished timers return -1 without a clock read.
    int64_t finish(bool success) noexcept {
        return finishAtSteady(-1, success);
    }
    // Resource boundaries can end one stage and start the next from one clock
    // read. The span and KMonitor sample always use the same supplied boundary.
    int64_t finishAtSteady(int64_t end_us, bool success) noexcept {
        if (!enabled_ || finished_.load(std::memory_order_relaxed)
            || finished_.exchange(true, std::memory_order_acq_rel))
            return -1;
        if (end_us < 0)
            end_us = AsyncProfileSession::steadyNowUs();
        span_.finishAtSteady(end_us);
        if (reporter_)
            reporter_->reportTransferStage(
                stage_, end_us - start_us_, success, rank_, scope_, operation_, source_, target_);
        return end_us - start_us_;
    }

    // Rejected admission never entered the queue and contributes no sample.
    void discard() noexcept {
        if (!enabled_ || finished_.exchange(true, std::memory_order_acq_rel))
            return;
        span_.discard();
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
    const char*                                    operation_{nullptr};
    const char*                                    source_{nullptr};
    const char*                                    target_{nullptr};
    bool                                           enabled_{false};
    bool                                           success_{false};
    int                                            exceptions_{-1};
    int64_t                                        start_us_{0};
    AsyncProfileSpan                               span_;
    std::atomic<bool>                              finished_{false};
};
inline void SharedTransferStageTimer::finish(bool success) const noexcept {
    if (state_)
        state_->finish(success);
}

inline void SharedTransferStageTimer::discard() const noexcept {
    if (state_)
        state_->discard();
}

}  // namespace rtp_llm
