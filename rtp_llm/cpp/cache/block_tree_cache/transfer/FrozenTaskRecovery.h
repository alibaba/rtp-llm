#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionQuery.h"

namespace rtp_llm {

// A single recovery pass. The caller owns scheduling and shutdown; this class
// never starts a thread, removes records, or changes the original failure.
class FrozenTaskRecovery {
public:
    using TaskSource = std::function<std::vector<std::shared_ptr<FrozenTask>>()>;
    using Query = std::function<TransferCompletionQueryResult(const FrozenTask::ExecutionInfo&, int)>;

    struct Summary {
        size_t inspected{0};
        size_t queried{0};
        size_t released{0};
        size_t pending{0};
        size_t query_failed{0};
        bool   skipped_running{false};
        bool   stopped{false};
        bool   invalid_timeout{false};
        bool   source_failed{false};
    };

    FrozenTaskRecovery(TaskSource source, Query query);

    // Only FROZEN tasks are queried. Stop is checked before/after each query and
    // immediately before release; a throwing stop predicate stops this pass.
    // Positive query timeout is required. Diagnostics are best effort and do
    // not revoke an already validated buffer-safety proof.
    Summary runOnce(int query_timeout_ms, const std::function<bool()>& should_stop = {}) noexcept;

private:
    void logPending(const Summary& summary) noexcept;

    TaskSource source_;
    Query      query_;
    std::atomic<bool> running_{false};
    // Accessed only by the successful running_ claimant.
    std::chrono::steady_clock::time_point next_pending_log_{};
};

}  // namespace rtp_llm
