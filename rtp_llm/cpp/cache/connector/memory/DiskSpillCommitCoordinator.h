#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/connector/memory/DiskSpillBlockCache.h"
#include "rtp_llm/cpp/cache/connector/memory/DiskSpillTypes.h"

namespace rtp_llm {

// DiskSpillCommitCoordinator manages async spill lifecycle per README §"异步 commit 状态机".
//
//   RESERVED  -> STAGING -> PWRITE_INFLIGHT -> COMMITTED
//                       \-> ABORTING -> LEAKED | FREE
//
// Caller (KVCacheMemoryConnector::ensureEnoughFreeBlocks via flushPendingSpills)
// transfers ownership of the staging payload to the coordinator and gets a
// SpillJobId immediately. The coordinator:
//
//   1. submits local pwrite to the DiskSpillBlockCache (which fans out to IoWorker)
//   2. invokes a user-supplied broadcastSpillFn to send SPILL_MEMORY_TO_DISK to workers
//   3. polls SPILL_WRITE_STATUS via a user-supplied pollWorkerStatusFn
//   4. on all ranks complete -> commit + on_complete(COMMITTED)
//   5. on any rank fail / timeout -> abort + broadcastDeleteFn + on_complete(LEAKED or FREE)
//
// The coordinator owns a single worker thread that drives the state machine on
// a tick of `poll_interval_ms`. Timeouts are checked against config_.stage_ack_timeout_ms
// and config_.commit_timeout_ms.
//
// Master pull semantics for SPILL_WRITE_STATUS:
//   - The coordinator's worker thread loops over inflight jobs and queries each
//     rank via pollWorkerStatusFn(rank, job_id) until SUCCESS/FAILED/timeout.
//   - Lost responses are retried implicitly by the next poll tick.
//
// `local_pwrite_callback` is invoked from the IoWorker thread when the local
// pwrite completes; it must be thread-safe.
class DiskSpillCommitCoordinator {
public:
    struct Config {
        int64_t stage_ack_timeout_ms{1000};
        int64_t commit_timeout_ms{5000};
        int64_t poll_interval_ms{50};
        size_t  max_inflight_jobs{4096};
    };

    using OnCompleteFn = std::function<void(SpillJobId, SpillStageState)>;

    // Called once per spill to ask the caller to broadcast SPILL_MEMORY_TO_DISK
    // to all workers. Returns true if broadcast was successfully dispatched (not
    // necessarily acked). False indicates a permanent failure and the spill is
    // aborted.
    using BroadcastSpillFn = std::function<bool(SpillJobId, const DiskSpillBlockCache::DiskItem&)>;

    // Called once per abort to ask the caller to broadcast DELETE_DISK_SLOT.
    using BroadcastDeleteFn = std::function<bool(const DiskSpillBlockCache::DiskItem&)>;

    // Per-rank status poll. Caller queries the worker for the given job_id and
    // returns one of PENDING/SUCCESS/FAILED/UNKNOWN_JOB.
    using PollWorkerStatusFn = std::function<SpillWriteStatus(int rank, SpillJobId)>;

public:
    DiskSpillCommitCoordinator(std::shared_ptr<DiskSpillBlockCache> cache,
                               Config                               config,
                               int                                  worker_count,
                               BroadcastSpillFn                     spill_fn,
                               BroadcastDeleteFn                    delete_fn,
                               PollWorkerStatusFn                   poll_fn);
    ~DiskSpillCommitCoordinator();

    DiskSpillCommitCoordinator(const DiskSpillCommitCoordinator&)            = delete;
    DiskSpillCommitCoordinator& operator=(const DiskSpillCommitCoordinator&) = delete;

    bool start();
    void stop();

    // Submit a fully-staged spill. Coordinator owns staging_data and runs the
    // async commit. on_complete fires (from coordinator thread or local pwrite
    // thread) when the job reaches a terminal state.
    SpillJobId submitSpill(const DiskSpillBlockCache::DiskItem& slot,
                           std::vector<char>                    staging_data,
                           OnCompleteFn                         on_complete);

    // Notify the coordinator that worker rank N has acked pwrite (SUCCESS or
    // FAILED). Used by SPILL_WRITE_STATUS handler when a worker pushes a status
    // ahead of the master's pull cycle. Optional.
    void notifyWorkerStatus(SpillJobId job_id, int rank, SpillWriteStatus status);

    // Get current state of a job (for status RPC).
    SpillStageState getJobState(SpillJobId job_id) const;
    SpillWriteStatus getJobWriteStatus(SpillJobId job_id) const;

    size_t inflightJobs() const;

    // Test-only: synchronously drain all inflight jobs (with timeout).
    bool drainForTest(int64_t timeout_ms);

private:
    struct SpillJob {
        SpillJobId                       id{0};
        DiskSpillBlockCache::DiskItem    slot{};
        // shared_ptr so the IoWorker callback can keep the staging buffer alive
        // past job erase, in the abort-before-pwrite-completes case.
        std::shared_ptr<std::vector<char>> staging_data;
        OnCompleteFn                     on_complete;
        SpillStageState                  state{SpillStageState::RESERVED};
        std::chrono::steady_clock::time_point created_at;
        std::chrono::steady_clock::time_point staging_done_at;
        // Worker rank ack: rank -> status. For TP=1 (no workers), this is empty.
        std::unordered_map<int, SpillWriteStatus> worker_status;
        bool                                      local_pwrite_done{false};
        bool                                      local_pwrite_ok{false};
        bool                                      spill_broadcast_sent{false};
    };

    void mainLoop();
    void tickLocked();
    void onLocalPwriteComplete(SpillJobId id, bool ok);
    void terminate(SpillJob& job, SpillStageState final_state, std::unique_lock<std::mutex>& lock);
    bool allWorkersDone(const SpillJob& job) const;
    bool anyWorkerFailed(const SpillJob& job) const;
    bool tryDispatchLocalPwrite(SpillJob& job);
    bool tryBroadcastSpill(SpillJob& job);

    std::shared_ptr<DiskSpillBlockCache>  cache_;
    Config                                config_;
    int                                   worker_count_{0};
    BroadcastSpillFn                      spill_fn_;
    BroadcastDeleteFn                     delete_fn_;
    PollWorkerStatusFn                    poll_fn_;
    std::atomic<bool>                     running_{false};
    std::atomic<SpillJobId>               next_job_id_{1};
    mutable std::mutex                    mutex_;
    std::condition_variable               cv_;
    std::unordered_map<SpillJobId, SpillJob> jobs_;
    std::thread                           thread_;
    // Pending IoWorker callback counter. submitSpill -> tryDispatchLocalPwrite
    // increments; onLocalPwriteComplete decrements. stop() waits until this
    // reaches 0 so the coordinator object outlives every dangling pwrite
    // callback that captured `this` — otherwise the IoWorker (owned by
    // DiskSpillBlockCache which has a longer lifetime than the coordinator in
    // some usages, including tests where coordinator is on the stack) will
    // invoke a use-after-free callback.
    mutable std::mutex                    pending_callbacks_mutex_;
    std::condition_variable               pending_callbacks_cv_;
    int                                   pending_callbacks_{0};
};

using DiskSpillCommitCoordinatorPtr = std::shared_ptr<DiskSpillCommitCoordinator>;

}  // namespace rtp_llm
