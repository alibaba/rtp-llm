#pragma once

#include <cstddef>
#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <mutex>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionTypes.h"

namespace rtp_llm {

// Pending worker-record deletion only. Nodes never own pools, FrozenTasks or
// business callbacks; the owner supplies the transport for each drain call.
class TransferCompletionCleanup: public std::enable_shared_from_this<TransferCompletionCleanup> {
private:
    struct Entry {
        TransferCompletionTarget target;
        TransferExecutionTrace   trace_id;
    };

public:
    using Forget = std::function<TransferCompletionForgetResult(
        const TransferCompletionTarget&, const std::vector<TransferExecutionTrace>&, int)>;

    class Prepared {
    public:
        Prepared() = default;
        Prepared(Prepared&&) noexcept            = default;
        Prepared& operator=(Prepared&&) noexcept = default;
        Prepared(const Prepared&)                = delete;
        Prepared& operator=(const Prepared&)     = delete;

        // Transfers the preallocated node at most once. Destruction alone never
        // submits cleanup, including when the operation has not become safe.
        void submit() noexcept;
        bool empty() const noexcept {
            return node_.empty();
        }

    private:
        friend class TransferCompletionCleanup;
        std::shared_ptr<TransferCompletionCleanup> owner_;
        std::list<Entry>                          node_;
    };

    struct Summary {
        size_t attempted{0};
        size_t forgotten{0};
        size_t pending{0};
        bool   stopped{false};
        bool   failed{false};
    };

    // Call before submission, while allocation and validation can still fail.
    // This queue must be shared-owned. Routing is normalized by original rank.
    Prepared prepare(TransferCompletionTarget target, TransferExecutionTrace trace_id);

    // Includes queued and currently draining entries until deletion is confirmed.
    size_t size() const;

    // Processes only the initial queue snapshot, at most 1024 keys per routing
    // batch. timeout_ms applies to each callback. A failed entry waits for the
    // next drain; concurrent submissions also wait for a later drain.
    Summary drain(const Forget& forget, int timeout_ms,
                  const std::function<bool()>& should_stop = {}) noexcept;

private:
    void logFailure(const char* reason, const Entry* entry = nullptr,
                    const TransferCompletionForgetEntryResult* response = nullptr) noexcept;
    void submit(std::list<Entry>& node) noexcept;
    void requeue(std::list<Entry>& entries) noexcept;

    mutable std::mutex mutex_;
    std::list<Entry>   pending_;
    std::chrono::steady_clock::time_point next_failure_log_{};
    size_t             outstanding_{0};
    size_t             next_warning_{4096};
};

}  // namespace rtp_llm
