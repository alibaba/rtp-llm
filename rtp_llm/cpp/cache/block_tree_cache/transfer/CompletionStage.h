#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"

namespace rtp_llm {

// Queue and total cross the same execution boundary, so share one allocation.
class CompletionStageTimers {
    struct State {
        explicit State(const CacheStageContext& context) noexcept:
            total(context, "completion_total", "business_completion", AsyncProfileKind::Lifecycle),
            queue(context, "completion_queue_wait", "business_completion", AsyncProfileKind::Wait) {}
        TransferStageTimer total;
        TransferStageTimer queue;
    };

public:
    static CompletionStageTimers create(const CacheStageContext& context) noexcept {
        CompletionStageTimers timers;
        if (context.enabled) {
            try {
                timers.state_ = std::make_shared<State>(context);
            } catch (...) {
                // Optional stage_context must never prevent business completion.
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
