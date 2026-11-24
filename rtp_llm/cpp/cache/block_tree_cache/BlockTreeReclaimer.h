#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"

class FunctionRequestPB;
class FunctionResponsePB;

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BroadcastManager;

class BlockTreeReclaimer final: public FrozenTaskReclaimer, public TransferTaskRegistry {
public:
    explicit BlockTreeReclaimer(std::shared_ptr<BlockTreeCacheMetricsReporter> metrics = {});
    ~BlockTreeReclaimer() override;

    bool registerOnWorkerTaskWithTraceID(const std::string& id, OfferedCallback offeredQuery = {}, OfferedCallback offeredForget = {}) override;
    bool ioFinishedOnWorkerWithTraceID(const std::string& id) override;
    std::vector<State> queryIfIoFinishedInOneWorkerWithTraceIDs(const std::vector<std::string>& ids) override;
    std::vector<ForgetResult> forgetTaskInfoOnWorker(const std::vector<std::string>& ids) override;

    void configure(int rank, int64_t interval_ms, int64_t timeout_ms, int64_t force_release_timeout_ms,
                   std::shared_ptr<BroadcastManager> broadcast);
    bool start();
    void stop() noexcept;

    struct Participants {
        bool remote;
        size_t ranks;
    };

    bool registerTask(TransferCompletionKey trace, Participants participants,
                      std::vector<TransferTargetBlock> targets, std::string backend = "local");
    void unregisterTask(const TransferCompletionKey& trace);
    void freezeTargetBlocksWithTraceIDAndStillIOWorkerID(const TransferCompletionKey& trace,
                                                         std::vector<TransferTargetBlock> target_blocks,
                                                         const std::vector<size_t>& pending_workers) noexcept override;
    bool unfreezeTargetBlocksInOneWorkerWithTraceID(const TransferCompletionKey& trace, size_t worker) override;

    static bool executeFunction(BlockTreeReclaimer* owner, int rank, const FunctionRequestPB& request,
                                FunctionResponsePB& response);

private:
    using Clock = std::chrono::steady_clock;
    using Expirations = std::multimap<Clock::time_point, std::string>;
    struct Record {
        State state{State::RUNNING};
        OfferedCallback query, forget;
        size_t queries{0};
        bool forgetting{false};
        Expirations::iterator expiry;
    };

    State queryOne(const std::string& id);
    ForgetResult forgetOne(const std::string& id);
    State localState(const std::string& id) const;
    void extendMissing(Record& record);
    void cleanupExpiredMissing();
    std::function<Clock::time_point()> now_{Clock::now};
    mutable std::mutex worker_mutex_;
    std::unordered_map<std::string, Record> records_;
    Expirations expirations_;

    struct Route {
        TransferCompletionKey trace;
        Participants participants;
    };
    struct Task {
        std::list<Route> forget;
        std::vector<TransferTargetBlock> targets;
        std::string backend;
        std::vector<size_t> pending;
        Clock::time_point frozen_at;
        std::vector<std::function<void()>> release;
    };
    using TaskMap = std::map<std::string, Task>;

    void unregisterTask(TaskMap::node_type task, bool forced);
    bool release(const TransferCompletionKey& trace, std::optional<size_t> worker);
    void queryFrozen();
    void releaseExpired();
    void drainForget();
    void periodic(void (BlockTreeReclaimer::*work)(), int64_t interval_ms);
    std::vector<size_t> query(const Route& route, const std::vector<size_t>& workers);
    std::vector<bool> forget(const std::vector<TransferCompletionKey>& traces, Participants participants);

    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_;
    std::shared_ptr<BroadcastManager> broadcast_;
    int rank_{0};
    int64_t interval_ms_{0}, timeout_ms_{1000}, force_release_timeout_ms_{60000};
    std::mutex mutex_;
    TaskMap registered_, frozen_;
    std::list<Route> forget_queue_;
    std::atomic<bool> stopped_{false};
    std::condition_variable cv_;
    std::vector<std::thread> threads_;
};

}
