#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

class FunctionRequestPB;
class FunctionResponsePB;

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BroadcastManager;

class BlockTreeReclaimer final {
public:
    enum class State { UNKNOWN, RUNNING, BUFFER_SAFE, REJECTED };
    enum class ForgetResult { FAILED, DONE, RUNNING, INVALID };
    using QueryCallback = std::function<bool(std::chrono::steady_clock::time_point)>;
    using ForgetCallback = std::function<void(bool)>;

    explicit BlockTreeReclaimer(std::shared_ptr<BlockTreeCacheMetricsReporter> metrics = {});
    ~BlockTreeReclaimer();

    bool registerOnWorkerTaskWithTraceID(const std::string& id, QueryCallback offeredQuery = {}, ForgetCallback offeredForget = {});
    bool ioFinishedOnWorkerWithTraceID(const std::string& id);
    std::vector<State> queryIfIoFinishedInOneWorkerWithTraceIDs(const std::vector<std::string>& ids);
    std::vector<ForgetResult> forgetTaskInfoOnWorker(const std::vector<std::string>& ids, bool forced = false);

    void configure(int rank, int64_t interval_ms, int64_t timeout_ms, int64_t force_release_timeout_ms,
                   std::shared_ptr<BroadcastManager> broadcast);
    void setRecoveryMaintenance(std::function<void()> maintenance);
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
                                                         const std::vector<size_t>& pending_workers) noexcept;
    bool markWorkerSafe(const TransferCompletionKey& trace, size_t worker);

    static bool executeFunction(BlockTreeReclaimer* owner, int rank, const FunctionRequestPB& request,
                                FunctionResponsePB& response, std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max());

private:
    using Clock = std::chrono::steady_clock;
    using Expirations = std::multimap<Clock::time_point, std::string>;
    struct Record {
        State state{State::RUNNING};
        QueryCallback query;
        ForgetCallback forget;
        size_t queries{0};
        bool forgetting{false};
        Expirations::iterator expiry;
    };

    Record& findOrRejectMissing(const std::string& id);
    State queryOne(const std::string& id, Clock::time_point deadline = Clock::time_point::max());
    ForgetResult forgetOne(const std::string& id, bool forced = false);
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
        bool forced{false};
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
    bool forceRelease(const TransferCompletionKey& trace);
    void queryFrozen();
    void releaseExpired();
    void drainForget();
    void periodic(void (BlockTreeReclaimer::*work)(), int64_t interval_ms);
    void queryWorkers(const std::vector<size_t>& workers, const TransferCompletionKey* traces, size_t count);
    std::vector<bool> forget(const std::vector<TransferCompletionKey>& traces, Participants participants, bool forced = false);

    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_;
    std::shared_ptr<BroadcastManager> broadcast_;
    int rank_{0};
    int64_t interval_ms_{0}, timeout_ms_{1000}, force_release_timeout_ms_{60000};
    std::function<void()> recovery_maintenance_;
    std::mutex mutex_;
    TaskMap registered_, frozen_;
    std::list<Route> forget_queue_;
    std::atomic<bool> stopped_{false};
    std::condition_variable cv_;
    std::vector<std::thread> threads_;
};

}
