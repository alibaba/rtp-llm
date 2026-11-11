#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskRecovery.h"

#include <algorithm>
#include <exception>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

using State = TransferTaskRegistry::State;

struct Proof {
    bool valid{false};
    bool safe{false};
    const char* error{nullptr};
};

// Validate the semantic query result independently of its injectable producer.
// No allocation is needed to validate a complete proof. Both trace alternatives
// and every original rank/address must match, with no duplicates or omissions.
Proof validate(const FrozenTask::ExecutionInfo& execution, const TransferCompletionQueryResult& query) noexcept {
    if (!validTransferExecutionTrace(execution.trace_id) || query.trace_id != execution.trace_id)
        return {false, false, "recovery query trace does not match frozen execution"};
    const auto count = execution.expected_rank_count;
    if (count == 0 || execution.expected_peers.size() != count || query.ranks.size() != count
        || (!execution.remote && count != 1))
        return {false, false, "recovery query does not cover the original participants"};
    if (!query.error.empty())
        return {false, false, nullptr};
    bool all_safe = true;
    for (size_t rank = 0; rank < count; ++rank) {
        const FrozenTask::ExpectedPeer* peer = nullptr;
        const TransferCompletionRankResult* row = nullptr;
        for (const auto& candidate : execution.expected_peers) {
            if (candidate.rank == rank) {
                if (peer)
                    return {false, false, "frozen execution has duplicate participant ranks"};
                peer = &candidate;
            }
        }
        for (const auto& candidate : query.ranks) {
            if (candidate.rank == rank) {
                if (row)
                    return {false, false, "recovery query has duplicate participant ranks"};
                row = &candidate;
            }
        }
        if (!peer || !row || peer->address.empty() || peer->address != row->server_address)
            return {false, false, "recovery query rank or address does not match frozen execution"};
        if (execution.remote && (!row->cq_received || !row->cq_ok || !row->grpc_code))
            return {false, false, "remote recovery query has no successful CQ and gRPC evidence"};
        if (!row->error.empty() || (row->grpc_code && *row->grpc_code != 0)
            || (row->cq_received && !row->cq_ok))
            return {false, false, "recovery query participant has a transport error"};
        if (row->state != State::UNKNOWN && row->state != State::RUNNING && row->state != State::BUFFER_SAFE)
            return {false, false, "recovery query participant has an invalid completion state"};
        if (row->admission_closed && row->state != State::UNKNOWN)
            return {false, false, "admission ban requires UNKNOWN completion state"};
        all_safe = all_safe && row->bufferSafe();
    }
    if (query.all_safe != all_safe)
        return {false, false, "recovery query summary disagrees with participant completion states"};
    return {true, all_safe, nullptr};
}

void recordQuery(const std::shared_ptr<FrozenTask>& task, TransferCompletionQueryResult result) noexcept {
    try {
        task->recordRecoveryQuery(std::make_shared<const TransferCompletionQueryResult>(std::move(result)));
    } catch (...) {
        // A diagnostic allocation failure cannot invalidate a proof that was
        // already checked from the returned result, or alter frozen ownership.
    }
}

void recordException(const std::shared_ptr<FrozenTask>& task, const char* error) noexcept {
    try {
        TransferCompletionQueryResult failure;
        failure.trace_id = task->execution().trace_id;
        failure.error = error;
        for (const auto& peer : task->execution().expected_peers) {
            TransferCompletionRankResult row;
            row.rank = peer.rank;
            row.server_address = peer.address;
            row.error = error;
            failure.ranks.push_back(std::move(row));
        }
        recordQuery(task, std::move(failure));
    } catch (...) {
    }
}

void logReleased(const FrozenTask& task) noexcept {
    try {
        const auto snapshot = task.snapshot();
        const auto duration_ms = std::max<int64_t>(0, snapshot.released_at_us - snapshot.frozen_at_us) / 1000;
        try {
            const auto description = task.debugString();
            RTP_LLM_LOG_INFO("FrozenTask recovery released frozen_duration_ms=%lld %s",
                             static_cast<long long>(duration_ms), description.c_str());
        } catch (...) {
            RTP_LLM_LOG_INFO("FrozenTask recovery released frozen_duration_ms=%lld (diagnostics unavailable)",
                             static_cast<long long>(duration_ms));
        }
    } catch (...) {
        // Release and its count are already complete; logging cannot repeat it.
    }
}

}  // namespace

FrozenTaskRecovery::FrozenTaskRecovery(TaskSource source, Query query, Cleanup cleanup):
    source_(std::move(source)), query_(std::move(query)), cleanup_(std::move(cleanup)) {}

FrozenTaskRecovery::~FrozenTaskRecovery() {
    stop();
}

bool FrozenTaskRecovery::start(int interval_ms, int query_timeout_ms) noexcept {
    if (interval_ms <= 0 || query_timeout_ms <= 0 || start_attempted_
        || stop_requested_.load(std::memory_order_acquire))
        return false;
    start_attempted_ = true;
    try {
        worker_ = std::thread([this, interval_ms, query_timeout_ms] { backgroundLoop(interval_ms, query_timeout_ms); });
        return true;
    } catch (...) {
        stop_requested_.store(true, std::memory_order_release);
        return false;
    }
}

void FrozenTaskRecovery::stop() noexcept {
    {
        // Pair the predicate change with the wait mutex so a notification
        // cannot be lost between checking the flag and entering the wait.
        std::lock_guard<std::mutex> lock(wait_mutex_);
        stop_requested_.store(true, std::memory_order_release);
    }
    wait_cv_.notify_all();
    if (worker_.joinable())
        worker_.join();
}

void FrozenTaskRecovery::backgroundLoop(int interval_ms, int query_timeout_ms) noexcept {
    try {
        const auto stopped = [this] { return stop_requested_.load(std::memory_order_acquire); };
        while (!stopped()) {
            (void)runOnce(query_timeout_ms, stopped);
            std::unique_lock<std::mutex> lock(wait_mutex_);
            if (wait_cv_.wait_for(lock, std::chrono::milliseconds(interval_ms), stopped))
                break;
        }
    } catch (...) {
        stop_requested_.store(true, std::memory_order_release);
        try {
            RTP_LLM_LOG_WARNING("FrozenTask recovery background worker stopped after an unexpected exception");
        } catch (...) {
        }
    }
}

FrozenTaskRecovery::Summary
FrozenTaskRecovery::runOnce(int query_timeout_ms, const std::function<bool()>& should_stop) noexcept {
    Summary summary;
    if (query_timeout_ms <= 0) {
        summary.invalid_timeout = true;
        return summary;
    }
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
        summary.skipped_running = true;
        return summary;
    }
    struct Retire {
        std::atomic<bool>& running;
        ~Retire() { running.store(false, std::memory_order_release); }
    } retire{running_};
    const auto stop = [&]() noexcept {
        if (!should_stop)
            return false;
        try {
            if (!should_stop())
                return false;
        } catch (...) {
            // A failed shutdown predicate is never permission to release.
        }
        summary.stopped = true;
        return true;
    };
    if (stop())
        return summary;
    // Cleanup owns only previously settled identities. A task released in this
    // recovery pass becomes eligible for cleanup on a later pass.
    if (cleanup_) {
        try {
            summary.cleanup_failed = cleanup_(query_timeout_ms, should_stop).failed;
        } catch (...) {
            summary.cleanup_failed = true;
        }
        if (stop())
            return summary;
    }
    std::vector<std::shared_ptr<FrozenTask>> tasks;
    try {
        tasks = source_();
    } catch (...) {
        summary.source_failed = true;
        logPending(summary);
        return summary;
    }
    for (const auto& task : tasks) {
        if (!task)
            continue;
        ++summary.inspected;
        if (task->snapshot().state != FrozenTask::State::FROZEN)
            continue;
        if (stop())
            break;
        ++summary.queried;
        bool safe = false;
        try {
            auto result = query_(task->execution(), query_timeout_ms);
            const auto proof = validate(task->execution(), result);
            safe = proof.valid && proof.safe;
            if (!proof.valid) {
                ++summary.query_failed;
                // Preserve the returned rows and identity, but annotate why an
                // inconsistent injected result cannot authorize this release.
                result.all_safe = false;
                if (result.error.empty() && proof.error) {
                    try { result.error = proof.error; } catch (...) {}
                }
            }
            recordQuery(task, std::move(result));
        } catch (const std::exception& error) {
            ++summary.query_failed;
            recordException(task, error.what());
        } catch (...) {
            ++summary.query_failed;
            recordException(task, "recovery query threw an unknown exception");
        }
        if (stop())
            break;
        if (safe && task->snapshot().state == FrozenTask::State::FROZEN) {
            if (stop())
                break;
            if (task->unfreeze()) {
                ++summary.released;
                logReleased(*task);
            }
        }
        if (task->snapshot().state == FrozenTask::State::FROZEN)
            ++summary.pending;
    }
    logPending(summary);
    return summary;
}

void FrozenTaskRecovery::logPending(const Summary& summary) noexcept {
    if (summary.pending == 0 && summary.query_failed == 0 && !summary.source_failed && !summary.cleanup_failed)
        return;
    const auto now = std::chrono::steady_clock::now();
    if (now < next_pending_log_)
        return;
    next_pending_log_ = now + std::chrono::seconds(60);
    try {
        RTP_LLM_LOG_INFO("FrozenTask recovery pending inspected=%zu queried=%zu released=%zu pending=%zu "
                         "query_failed=%zu source_failed=%d cleanup_failed=%d",
                         summary.inspected, summary.queried, summary.released, summary.pending,
                         summary.query_failed, static_cast<int>(summary.source_failed),
                         static_cast<int>(summary.cleanup_failed));
    } catch (...) {
    }
}

}  // namespace rtp_llm
