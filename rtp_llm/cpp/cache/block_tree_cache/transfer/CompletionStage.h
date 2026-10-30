#pragma once

#include <exception>
#include <memory>
#include <optional>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"

namespace rtp_llm {

// All names/identities are literals. IDs are trace-only; no per-request metric
// series or clocks/allocations are introduced when both observers are disabled.
struct CompletionStageContext {
    const BlockTreeCacheMetricsReporter* reporter;
    CacheStageContext                    identity;
    int64_t                              request_id{-1};
    uint64_t                             transfer_id{0};
    size_t                               block_count{0};
    bool                                 success{true};
    bool                                 enabled{false};

    // Capture eligibility once, before doing observation-only work. A capture
    // started later observes subsequent completions, not an incomplete context.
    static CompletionStageContext forWorkflow(const BlockTreeCacheMetricsReporter&   reporter,
                                              const std::vector<TransferDescriptor>& descriptors,
                                              const char*                            operation,
                                              int64_t                                request_id,
                                              uint64_t                               transfer_id,
                                              bool                                   success = true) noexcept {
        const bool enabled = reporter.stageMetricsEnabled() || AsyncProfileSession::enabled();
        return {&reporter,
                enabled ? CacheStageContext::workflow(operation, descriptors) :
                          CacheStageContext{operation, "NONE", "NONE", "unknown"},
                request_id,
                transfer_id,
                descriptors.size(),
                success,
                enabled};
    }

    static CompletionStageContext forTask(const BlockTreeCacheMetricsReporter& reporter,
                                          const TransferTask&                  task,
                                          const char*                          operation,
                                          bool                                 success = true) noexcept {
        return forWorkflow(
            reporter, task.descriptors(), operation, task.requestId().value_or(-1), task.traceId(), success);
    }
    // Physical tasks have already been partitioned by direction by their owner.
    static CompletionStageContext forTransfer(const BlockTreeCacheMetricsReporter& reporter,
                                              const TransferTask&                  task,
                                              const char*                          operation,
                                              bool                                 success = true) noexcept {
        const bool enabled  = reporter.stageMetricsEnabled() || AsyncProfileSession::enabled();
        auto       identity = CacheStageContext{operation, "NONE", "NONE", "unknown"};
        if (enabled && !task.descriptors().empty()) {
            const auto& first  = task.descriptors().front();
            identity           = CacheStageContext::local(first.source_tier, first.target_tier);
            identity.operation = operation;
        }
        return {&reporter,
                identity,
                task.requestId().value_or(-1),
                task.traceId(),
                task.descriptors().size(),
                success,
                enabled};
    }
    SharedTransferStageTimer start(const char* stage, const char* scope) const noexcept {
        if (!enabled)
            return {};
        return TransferStageTimer::create(
            *reporter, stage, request_id, 0, block_count, -1, scope, transfer_id, identity);
    }
};

class CompletionStageScope {
public:
    CompletionStageScope(const CompletionStageContext& context, const char* stage, const char* scope) noexcept:
        success_(context.success), exceptions_(std::uncaught_exceptions()) {
        if (context.enabled)
            timer_.emplace(*context.reporter,
                           stage,
                           context.request_id,
                           0,
                           context.block_count,
                           -1,
                           scope,
                           context.transfer_id,
                           context.identity);
    }
    ~CompletionStageScope() {
        if (timer_)
            timer_->finish(success_ && exceptions_ == std::uncaught_exceptions());
    }
    void setSuccess(bool success) noexcept {
        success_ = success;
    }

private:
    std::optional<TransferStageTimer> timer_;
    bool                              success_;
    int                               exceptions_;
};

// Queue and total cross the same execution boundary, so share one allocation.
class CompletionStageTimers {
    struct State {
        explicit State(const CompletionStageContext& context) noexcept:
            total(*context.reporter,
                  "completion_total",
                  context.request_id,
                  0,
                  context.block_count,
                  -1,
                  "business_completion",
                  context.transfer_id,
                  context.identity),
            queue(*context.reporter,
                  "completion_queue_wait",
                  context.request_id,
                  0,
                  context.block_count,
                  -1,
                  "business_completion",
                  context.transfer_id,
                  context.identity) {}
        TransferStageTimer total;
        TransferStageTimer queue;
    };

public:
    static CompletionStageTimers create(const CompletionStageContext& context) noexcept {
        CompletionStageTimers timers;
        if (context.enabled) {
            try {
                timers.state_ = std::make_shared<State>(context);
            } catch (...) {
                // Optional observation must never prevent business completion.
            }
        }
        return timers;
    }
    explicit operator bool() const noexcept {
        return state_ != nullptr;
    }
    void finishQueue(bool success) const noexcept {
        if (state_)
            state_->queue.finish(success);
    }
    void finish(bool success) const noexcept {
        if (state_)
            state_->total.finish(success);
    }

private:
    std::shared_ptr<State> state_;
};

}  // namespace rtp_llm
