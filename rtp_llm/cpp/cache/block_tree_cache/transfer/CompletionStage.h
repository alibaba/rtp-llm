#pragma once

#include <exception>

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

    static CompletionStageContext forTask(const BlockTreeCacheMetricsReporter& reporter,
                                          const TransferTask&                  task,
                                          const char*                          operation,
                                          bool                                 success = true) noexcept {
        return {&reporter,
                CacheStageContext::workflow(operation, task.descriptors()),
                task.requestId().value_or(-1),
                task.traceId(),
                task.descriptors().size(),
                success};
    }
    SharedTransferStageTimer start(const char* stage, const char* scope) const noexcept {
        return TransferStageTimer::create(
            *reporter, stage, request_id, 0, block_count, -1, scope, transfer_id, identity);
    }
};

class CompletionStageScope {
public:
    CompletionStageScope(const CompletionStageContext& context, const char* stage, const char* scope) noexcept:
        timer_(*context.reporter,
               stage,
               context.request_id,
               0,
               context.block_count,
               -1,
               scope,
               context.transfer_id,
               context.identity),
        success_(context.success),
        exceptions_(std::uncaught_exceptions()) {}
    ~CompletionStageScope() {
        timer_.finish(success_ && exceptions_ == std::uncaught_exceptions());
    }
    void setSuccess(bool success) noexcept {
        success_ = success;
    }

private:
    TransferStageTimer timer_;
    bool               success_;
    int                exceptions_;
};

}  // namespace rtp_llm
