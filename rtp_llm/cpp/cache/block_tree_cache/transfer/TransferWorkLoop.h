#pragma once

#include <array>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcCopyExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

class DiskIOExecutor;

// One operation owner. Disk workers fill their Entry's result and publish its
// completion queue under mutex_; CRC stage callbacks run in this owner's stack.
// Business references stay upstream. The borrowed disk executor, when needed,
// outlives this owner's shutdown/join; pass nullptr only for routes without Disk.
class TransferWorkLoop {
public:
    using BeforeCompleteCallback = std::function<void(const TransferResult&)>;

    TransferWorkLoop(std::vector<GroupSetPtr>                       groups,
                     std::unique_ptr<CrcCopyExecutor>               crc,
                     DiskIOExecutor*                                disk_io_executor,
                     TransferOperation                              operation,
                     size_t                                         batch_limit,
                     size_t                                         queue_capacity,
                     std::shared_ptr<HostStagingBlockPool>          staging,
                     std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter = nullptr,
                     int64_t                                        world_rank       = 0,
                     size_t                                         worker_index     = 0,
                     std::shared_ptr<HostStagingBlockPool>          swa_staging      = nullptr);
    ~TransferWorkLoop();
    TransferWorkLoop(const TransferWorkLoop&)            = delete;
    TransferWorkLoop& operator=(const TransferWorkLoop&) = delete;

    void start();
    // PerRank supplies a validated, nonempty homogeneous batch within batch_limit_.
    // Each stage expires during resource waits and before submission.
    // Submitted stages retain their buffers until completion, even after deadline.
    // With retryable supplied, a full owner returns null and leaves both
    // arguments intact. Other failures are terminal. Without it, full admission
    // returns a completed failure and records one rejected observation.
    std::shared_ptr<TransferAsyncContext>
    execute(TransferTask&& task, BeforeCompleteCallback&& before_complete = {}, bool* retryable = nullptr);
    // Fixed limit supplied by PerRank for both operation owners.
    size_t batchLimit() const noexcept {
        return batch_limit_;
    }
    void stopAdmission();
    void cancelPendingStagingTransfers();
    // External lifecycle thread only; never wait from a WorkLoop callback.
    // Closes admission, drains callbacks and joins.
    // The caller serializes shutdown() and destruction; sequential repeats are safe.
    void shutdown();
    void waitForIdle();
    // Accepted Entries, including incoming, resource waits, I/O and callbacks.
    size_t outstandingBatches() const;

private:
    std::shared_ptr<TransferAsyncContext> reject(const TransferTask& task, const char* message, ErrorCode code);
    struct Entry;
    using Entries       = std::list<std::unique_ptr<Entry>>;
    using WaitDeadlines = std::multimap<TransferTask::Clock::time_point, Entry*>;
    // Intrusive, allocation-free indexes. The disk completion queue uses mutex_;
    // other queues belong to the owner. An Entry belongs to at most one queue.
    struct Queue {
        Entry* head{nullptr};
        Entry* tail{nullptr};
        bool   empty() const {
            return head == nullptr;
        }
        void   push(Entry& entry);
        Entry* pop();
        void   remove(Entry& entry);
        Queue  take();
    };
    enum class Step {
        DISK_READ,
        DISK_WRITE,
        CRC_LOAD,
        CRC_EVICT,
        PPU_HOST_COPY
    };
    // Constructed by the submitter; otherwise owned by the WorkLoop. While a Disk
    // operation is pending, its callback exclusively writes result; the owner
    // consumes its mutex-published completion before accessing it. Each Entry is one batch.
    struct Entry {
        Entry(TransferTask task, BeforeCompleteCallback before_complete);
        TransferTask                               task;
        std::shared_ptr<TransferBatchAsyncContext> context;
        // Latest step status and one result per descriptor in this batch.
        TransferResult                              result;
        BeforeCompleteCallback                      before_complete;
        Step                                        step{Step::DISK_READ};  // Set by the initial plan.
        std::vector<HostBufferView>                 hosts;
        HostStagingBlockPool::HostStagingBlockBatch staging;
        Entries::iterator                           position;
        Entry*                                      next{nullptr};
        Entry*                                      previous{nullptr};
        Queue*                                      waiting_on{nullptr};
        std::optional<WaitDeadlines::iterator>      wait_deadline;
        bool                                        initial_plan{true}, step_started{false};
        // Queue observation ends at initial plan. Its storage is then reused
        // for the terminal CRC lifecycle, which ends after Entry destruction.
        TransferStageTimer      task_timer, queue_or_finalize_timer, batch_timer, staging_timer, step_timer;
        CrcCopyExecutor::Timing crc_timing;
    };
    struct StagingResource {
        std::array<std::shared_ptr<HostStagingBlockPool>, 2> pools;  // FULL/default, SWA.
        Queue                                                waiting;
    };
    void run();
    // false means the Entry is waiting for staging or has finished.
    bool               plan(Entry& entry);
    bool               acquireStaging(Entry& entry, bool waiting = false);
    void               releaseStaging(Entry& entry);
    void               retryStaging();
    size_t             stagingIndex(const Entry& entry) const;
    void               tryStep(Entry& entry);
    void               submitCrc(Entry& entry);
#if defined(USE_PPU) && USE_PPU
    void               submitPpuHostCopy(Entry& entry);
#endif
    void               submitDisk(Entry& entry);
    void               publishDisk(Entry& entry, BlockIOStatus status) noexcept;
    void               pollOnce();
    static void        recordDiskResult(Entry& entry, BlockIOStatus status) noexcept;
    static void        recordResult(Entry& entry, TransferResult result) noexcept;
    void               consumeStage(Entry& entry);
    void               finish(Entry& entry);
    void               fail(Entry& entry, TransferResult result);
    void               cancelStaging();
    void               waitForResource(Entry& entry, Queue& waiting);
    void               removeWaiter(Entry& entry);
    void               expireResourceWaiters();
    const GroupSetPtr& group(const Entry& entry) const;
    static bool        isDeviceCopy(Step step);
    // An explicit start_us backfills the interval for both observers.
    TransferStageTimer timer(const TransferTask&     task,
                             const char*             stage,
                             size_t                  begin,
                             size_t                  end,
                             AsyncProfileKind        kind,
                             const char*             scope    = "worker_batch",
                             const AsyncProfileSpan* parent   = nullptr,
                             int64_t                 start_us = -1) const noexcept;
    void               startStepTimer(Entry& entry, const char* stage);

    // Private policy: OOM is fatal and never becomes a task failure or retry.
    template<typename Action>
    static decltype(auto) abortOnOom(Action&& action) {
        try {
            return std::forward<Action>(action)();
        } catch (const std::bad_alloc&) {
            Logger::getEngineLogger().flush();
            std::abort();
        }
    }

    // A submission can throw only before I/O or after a confirmed drain.
    // Accepted asynchronous work settles through its completion callback.
    template<typename Action>
    void withEntry(Entry& entry, Action&& action) noexcept {
        try {
            abortOnOom(std::forward<Action>(action));
        } catch (const std::exception& error) {
            abortOnOom([&] { fail(entry, TransferResult::executionException(true, error.what())); });
        }
    }

    // All resources below outlive the owner thread's join.
    // ==================== Fixed configuration/handles: no WorkLoop lock ====================
    // Initialized before start(); CRC is exclusive to this owner until join.
    std::vector<GroupSetPtr>                       groups_;
    std::unique_ptr<CrcCopyExecutor>               crc_;
    DiskIOExecutor* const                          disk_io_executor_;  // Owned by the engine through shutdown/join.
    TransferOperation                              operation_;
    size_t                                         batch_limit_, capacity_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter_;
    int64_t                                        world_rank_;
    std::string                                    thread_name_;

    // ==================== Owner-thread-only state: no lock ====================
    // Only load owners have staging. Both class pools and their single wait
    // queue are private to this owner and outlive every staging lease.
    StagingResource      staging_;
    std::map<int, Queue> crc_waiting_;
    Entries              entries_;
    Queue                ready_, completed_;
    WaitDeadlines        wait_deadlines_;
    bool                 has_crc_pending_{false};

    // ==================== Shared state: protected by mutex_ ====================
    // API callers, owner and Disk callbacks use this mutex; cv_ waits on it.
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    accepting_{true}, cancel_staging_{false};
    size_t                  entry_count_{0};
    Entries                 incoming_;
    Queue                   disk_completed_;

    // ==================== Lifecycle: caller-serialized ====================
    // start() runs once before external use; the lifecycle caller serializes shutdown/join.
    std::thread thread_;
};

}  // namespace rtp_llm
