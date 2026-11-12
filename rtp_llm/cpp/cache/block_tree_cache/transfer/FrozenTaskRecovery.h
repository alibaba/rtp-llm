#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionQuery.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionCleanup.h"

namespace rtp_llm {

// Explicit recovery passes, with an optional background worker and completion
// cleanup phase. Neither phase changes the original business failure.
class FrozenTaskRecovery {
public:
    using TaskSource = std::function<std::vector<std::shared_ptr<FrozenTask>>()>;
    using Query = std::function<TransferCompletionQueryResult(const FrozenTask::ExecutionInfo&, int)>;
    using Cleanup = std::function<TransferCompletionCleanup::Summary(int, const std::function<bool()>&)>;

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
        bool   cleanup_failed{false};
    };

    FrozenTaskRecovery(TaskSource source, Query query, Cleanup cleanup = {});
    ~FrozenTaskRecovery();

    // The owner serializes start/stop/destruction outside the recovery thread.
    // Nonpositive parameters are rejected without consuming the start attempt.
    // A valid attempt is single-use, including thread-creation failure; stop
    // also prevents a later start. Failure returns false without throwing.
    bool start(int interval_ms, int query_timeout_ms) noexcept;
    // Wake interval waits and join after any current query returns. No further
    // background callbacks or releases run after stop returns. Does not force unfreeze.
    // The owner must finish any concurrent explicit runOnce calls before stop
    // or destruction; stop only joins the worker owned by this instance.
    void stop() noexcept;

    // Only FROZEN tasks are queried. Stop is checked before/after each query and
    // immediately before release; a throwing stop predicate stops this pass.
    // Positive query timeout is required. Diagnostics are best effort and do
    // not revoke an already validated buffer-safety proof.
    Summary runOnce(int query_timeout_ms, const std::function<bool()>& should_stop = {}) noexcept;

private:
    void backgroundLoop(int interval_ms, int query_timeout_ms) noexcept;
    void logPending(const Summary& summary) noexcept;

    TaskSource source_;
    Query      query_;
    Cleanup    cleanup_;
    std::atomic<bool> running_{false};
    // Accessed only by the successful running_ claimant.
    std::chrono::steady_clock::time_point next_pending_log_{};
    bool                      start_attempted_{false};
    std::atomic<bool>         stop_requested_{false};
    std::mutex                wait_mutex_;
    std::condition_variable  wait_cv_;
    std::thread               worker_;
};

}  // namespace rtp_llm
