#pragma once

#include <atomic>

namespace rtp_llm {

// Invariant: once seal() is called, no new ops will be registered, and isStopped() will
// eventually become true when started_ops == finished_ops.
struct P2PTransferLease {
    std::atomic<bool> no_more_new_ops{false};
    std::atomic<int>  started_ops{0};
    std::atomic<int>  finished_ops{0};

    void onTransferStarted() {
        started_ops.fetch_add(1, std::memory_order_relaxed);
    }

    void onTransferFinished() {
        finished_ops.fetch_add(1, std::memory_order_relaxed);
    }

    void updateFinishedOps(int completed_ops) {
        int counted = finished_ops.load(std::memory_order_relaxed);
        while (counted < completed_ops
               && !finished_ops.compare_exchange_weak(counted, completed_ops, std::memory_order_relaxed)) {}
    }

    void seal() {
        no_more_new_ops.store(true, std::memory_order_release);
    }

    bool isStopped() const {
        return no_more_new_ops.load(std::memory_order_acquire)
               && (started_ops.load(std::memory_order_relaxed) == finished_ops.load(std::memory_order_relaxed));
    }

    bool isSealed() const {
        return no_more_new_ops.load(std::memory_order_acquire);
    }

    int startedOps() const {
        return started_ops.load(std::memory_order_relaxed);
    }

    int finishedOps() const {
        return finished_ops.load(std::memory_order_relaxed);
    }
};

}  // namespace rtp_llm
