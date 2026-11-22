#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskRecovery.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

using State = TransferTaskRegistry::State;

bool validRank(const TransferCompletionRankResult& row, bool uses_rpc) {
    return row.error.empty() && (!uses_rpc || row.grpc_code.has_value())
           && (!row.grpc_code || *row.grpc_code == 0)
           && (row.state == State::UNKNOWN || row.state == State::RUNNING || row.state == State::BUFFER_SAFE)
           && (!row.admission_closed || row.state == State::UNKNOWN);
}

bool matchesQuery(const FrozenTask::ExecutionInfo& execution, const std::vector<size_t>& pending,
                  const TransferCompletionQueryResult& query) {
    if (execution.key.trace_id.empty() || query.trace_id != execution.key.trace_id
        || !query.error.empty() || pending.empty() || query.ranks.size() != pending.size()
        || (!execution.uses_rpc && execution.rank_count != 1))
        return false;
    bool all_safe = true;
    for (size_t i = 0; i < pending.size(); ++i) {
        const auto& row = query.ranks[i];
        if (row.rank != pending[i] || row.rank >= execution.rank_count)
            return false;
        all_safe = all_safe && validRank(row, execution.uses_rpc) && row.bufferSafe();
    }
    return query.all_safe == all_safe;
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
        failure.trace_id = task->execution().key.trace_id;
        failure.error = error;
        for (size_t rank = 0; rank < task->execution().rank_count; ++rank) {
            TransferCompletionRankResult row;
            row.rank = rank;
            row.error = error;
            failure.ranks.push_back(std::move(row));
        }
        recordQuery(task, std::move(failure));
    } catch (...) {
    }
}

void logReleased(const FrozenTask& task, bool forced) noexcept {
    const auto snapshot = task.snapshot();
    const auto duration_ms = std::max<int64_t>(0, snapshot.released_at_us - snapshot.frozen_at_us) / 1000;
    try {
        if (forced) {
            RTP_LLM_LOG_WARNING("FrozenTask recovery forced release after timeout frozen_duration_ms=%lld",
                                static_cast<long long>(duration_ms));
        } else {
            RTP_LLM_LOG_INFO("FrozenTask recovery released frozen_duration_ms=%lld",
                             static_cast<long long>(duration_ms));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FrozenTask release logging failed: forced=%d frozen_duration_ms=%lld error=%s\n",
                     static_cast<int>(forced), static_cast<long long>(duration_ms), error.what());
    } catch (...) {
        std::fprintf(stderr, "FrozenTask release logging failed: forced=%d frozen_duration_ms=%lld unknown exception\n",
                     static_cast<int>(forced), static_cast<long long>(duration_ms));
    }
}

}  // namespace

FrozenTaskRecovery::FrozenTaskRecovery(FrozenTaskReclaimer& coordinator, Query query, Cleanup cleanup, OnReleased on_released,
                                       int force_release_timeout_ms):
    coordinator_(coordinator), query_(std::move(query)), cleanup_(std::move(cleanup)),
    on_released_(std::move(on_released)),
    force_release_timeout_us_(static_cast<int64_t>(force_release_timeout_ms) * 1000) {
    if (force_release_timeout_ms < 0) {
        throw std::invalid_argument("frozen force release timeout must be nonnegative");
    }
}

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
        tasks = coordinator_.frozenTasks();
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
        std::vector<size_t> confirmed;
        try {
            const auto pending = task->pendingWorkers();
            auto result = query_(task->execution(), pending, query_timeout_ms);
            if (matchesQuery(task->execution(), pending, result)) {
                bool failed = false;
                for (const auto& row : result.ranks) {
                    if (!validRank(row, task->execution().uses_rpc))
                        failed = true;
                    else if (row.bufferSafe())
                        confirmed.push_back(row.rank);
                }
                summary.query_failed += failed;
            } else {
                ++summary.query_failed;
                result.all_safe = false;
                if (result.error.empty())
                    result.error = "recovery response does not match the requested workers or trace";
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
        const auto snapshot = task->snapshot();
        const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count();
        const bool expired = force_release_timeout_us_ > 0
                             && now_us - snapshot.frozen_at_us >= force_release_timeout_us_;
        bool released = false;
        bool forced = false;
        try {
            for (const auto rank : confirmed) {
                if (stop())
                    break;
                released = coordinator_.unfreezeTargetBlocksInOneWorkerWithTraceID(
                    task->execution().key, rank);
            }
        } catch (...) {
            ++summary.query_failed;
        }
        if (expired && !released && !stop() && snapshot.state == FrozenTask::State::FROZEN)
            forced = task->completeRecovery(*coordinator_.cleanup());
        if (released || forced) {
            ++summary.released;
            try {
                if (on_released_)
                    on_released_(task->execution(), forced);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_WARNING("FrozenTask release observer failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("FrozenTask release observer failed: unknown exception");
            }
            logReleased(*task, forced);
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
