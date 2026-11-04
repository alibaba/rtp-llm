#pragma once

#include "rtp_llm/flexlb/navi_sched/data/AgentUpdateToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentCacheR.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentPendingCacheR.h"
#include "rtp_llm/flexlb/navi_sched/resource/RequestLatencyTracker.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentSchedulerConfigR.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentWorkloadOwnership.h"
#include "autil/LockFreeQueue.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include <autil/Lock.h>
#include <atomic>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace navi_sched {

struct NaviSchedLatencyModelSnapshot;
class AgentReportR;

class AgentReportBridgeR : public navi::ExternalResource {
public:
    AgentReportBridgeR();

public:
    void setAgentReportR(const std::shared_ptr<AgentReportR> &agentReportR);
    std::shared_ptr<AgentReportR> getAgentReportR() const;

public:
    static const std::string RESOURCE_ID;

private:
    mutable autil::ThreadMutex _lock;
    std::weak_ptr<AgentReportR> _agentReportR;
};

NAVI_TYPEDEF_PTR(AgentReportBridgeR);

class AgentReportR : public navi::Resource {
public:
    struct Stats {
        uint64_t publishedEventCount = 0;
        uint64_t publishedRequestReceivedCount = 0;
        uint64_t publishedPromptTokenCount = 0;
        uint64_t publishedFinishedBatchCount = 0;
        uint64_t publishedAbnormalFinishedBatchCount = 0;
        uint64_t publishedNormalFinishedRequestCount = 0;
        uint64_t publishedAbnormalFinishedRequestCount = 0;
        uint64_t deliverySuccessCount = 0;
        uint64_t deliveryFailureCount = 0;
        uint64_t noSubscriberEventCount = 0;
        uint64_t publishedWorkloadUpdateCount = 0;
        uint64_t suppressedLatencyValidationBatchCount = 0;
    };
    struct WorkloadPublishWindowStats {
        uint64_t sampleCount = 0;
        int64_t totalTimeUs = 0;
        int64_t maxTimeUs = 0;
    };
    struct WorkloadOwnershipStats {
        size_t batchCount = 0;
        size_t requestCount = 0;
        size_t unbatchedRequestCount = 0;
        int64_t rawObserverEventTimeUs = 0;
    };
    struct LatencyValidationDiagnostics {
        NaviSchedLatencyValidationStats stats;
        int64_t lastUpdateTimeUs = 0;
        std::vector<int64_t> bucketLastUpdateTimeUs;
    };

    ~AgentReportR();

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    NaviSchedServerInfo getServerInfo() const;
    void setLatencyModelSnapshot(
        const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot);
    bool setInflightTracker(
        const std::shared_ptr<ConcurrentRequestLatencyTracker> &tracker);
    std::shared_ptr<const NaviSchedLatencyModelSnapshot>
    getLatencyModelSnapshot() const;
    AgentSchedulerConfigR::Snapshot getSchedulerConfigSnapshot() const;
    bool setInflightStats(const RequestLatencyTracker::Snapshot &snapshot);
    RequestLatencyTracker::Snapshot getInflightStats() const;
    bool getLiveInflightStats(RequestLatencyTracker::Snapshot &snapshot) const;
    size_t reportLatencyModelUpdated(
        const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
        bool recomputeWaitingInflight = false);
    uint64_t reportWorkloadUpdated(const NaviSchedWorkloadSnapshot &snapshot,
                                   const std::vector<int64_t> &coveredNativeBatchIds = {});
    bool reportNativeBatchReady(const NaviSchedBatchPtr &batch);
    bool reportNativeRequestReceived(const NaviSchedRequestStats &request,
                                    int32_t batchWaitingRequests, int32_t queuedBatchRequests,
                                    int32_t runningRequests, int64_t eventTimeUs);
    bool reportNativeUnbatchedRequestsRemoved(const std::vector<uint64_t> &requestIds);
    bool reportNativeRequestsRemoved(int64_t localBatchId,
                                    const std::vector<uint64_t> &requestIds);
    bool reportNativeBatchRemoved(int64_t localBatchId);
    size_t reportSchedulerConfigUpdated(uint64_t requestGeneration,
                                        uint64_t appliedGeneration,
                                        const std::string &configHash,
                                        const std::string &configJson,
                                        const std::string &error) const;
    size_t reportRequestReceived(uint64_t requestId, int64_t promptTokenCount,
                                 int32_t inflightRequests,
                                 int32_t batchWaitingRequests,
                                 int32_t queuedBatchRequests, int32_t runningRequests,
                                 int64_t eventTimeUs) const;
    size_t reportExecutionLoad(int32_t batchWaitingRequests, int32_t queuedBatchRequests,
                               int32_t runningRequests,
                               double queueTotalPredictedLatencyMs,
                               double queueDecayHeadRemainingMs,
                               uint64_t batchTimeoutCount,
                               uint64_t abnormalBatchFinishCount,
                               int64_t eventTimeUs,
                               const NaviSchedLocalBatchInfo &localBatch = {}) const;
    size_t reportBatchFinished(int32_t inflightRequests, int32_t batchWaitingRequests,
                               int32_t queuedBatchRequests, int32_t runningRequests,
                               double latencyMs, double queueTotalPredictedLatencyMs,
                               double queueDecayHeadRemainingMs,
                               const NaviSchedCacheUpdate &cacheUpdate,
                               int32_t completedRequestCount, int64_t cacheHitTokens,
                               int64_t cacheMissTokens, uint64_t batchTimeoutCount,
                               uint64_t abnormalBatchFinishCount,
                               bool abnormalBatchFinished, int64_t eventTimeUs,
                               const NaviSchedLocalBatchInfo &localBatch = {},
                               double agentPredictedExecutionLatencyMs = 0.0,
                               double backendQueueLatencyMs = 0.0,
                               const NaviSchedLatencyValidationStats &validationStats =
                                   NaviSchedLatencyValidationStats()) const;
    bool latencyValidationEnabled() const;
    void setLatencyValidationEnabled(bool enabled);
    bool pdSeparated() const { return _pdSeparated; }
    bool usesUnifiedLatencyPrediction() const {
        return _unifiedLatencyPrediction;
    }
    size_t reportCacheUpdated(const NaviSchedCacheUpdate &update) const;
    bool reportPendingCacheReady(
        const NaviSchedPendingCacheReservation &reservation) const;
    bool reportPendingCacheCompleted(int64_t localBatchId) const;
    bool reportPendingCacheAborted(int64_t localBatchId) const;
    bool replacePendingCache(
        int64_t localBatchId,
        const NaviSchedPendingCacheReservation *remainingReservation) const;
    size_t reportBatchRequestsAborted(
        int32_t inflightRequests, int32_t batchWaitingRequests,
        int32_t queuedBatchRequests, int32_t runningRequests,
        double queueTotalPredictedLatencyMs,
        double queueDecayHeadRemainingMs, uint64_t batchTimeoutCount,
        uint64_t abnormalBatchFinishCount, int32_t abortedRequestCount,
        int64_t eventTimeUs,
        const NaviSchedLocalBatchInfo &localBatch) const;
    bool reportPendingCacheSnapshot(const std::string &collectorId) const;
    NaviSchedRequestStats makeRequestStats(const NaviSchedBatchRequest &request) const;
    AgentCacheR::RequestAudit makeRequestAudit(
        const NaviSchedBatchRequest &request) const;
    size_t beginStop(int64_t eventTimeUs);
    bool waitForStopAcks(int64_t timeoutUs);
    bool confirmStop(const std::string &collectorId);
    bool subscribe(const std::string &collectorId, const navi::AsyncPipePtr &asyncPipe);
    bool unsubscribe(const std::string &collectorId, const navi::AsyncPipePtr &asyncPipe);
    size_t publish(const AgentUpdateToMasterDataPtr &update) const;
    bool finish(const std::string &collectorId);
    void finishAll();
    size_t getSubscriberCount() const;
    Stats getStats() const;
    WorkloadOwnershipStats getWorkloadOwnershipStats() const;
    NaviSchedLatencyValidationStats getLatencyValidationStats() const;
    LatencyValidationDiagnostics getLatencyValidationDiagnostics() const;
    NaviSchedWorkloadSnapshotConstPtr getWorkloadSnapshot() const;
    WorkloadPublishWindowStats takeWorkloadPublishWindowStats() const;
    size_t getCachedBlockCount() const;
    AgentCacheR::State getCacheState() const;
    AgentPendingCacheR::State getPendingCacheState() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    struct Subscriber {
        std::string collectorId;
        std::weak_ptr<navi::AsyncPipe> asyncPipe;
    };
    NAVI_TYPEDEF_PTR(Subscriber);

    struct QueuedPublish {
        std::vector<AgentUpdateToMasterDataPtr> updates;
        std::vector<SubscriberPtr> subscribers;
        bool trackStats = false;
        mutable autil::ThreadCond completionCond;
        bool completed = false;
        size_t publishCount = 0;
        int64_t maxSetDataTimeUs = 0;
    };
    using QueuedPublishPtr = std::shared_ptr<QueuedPublish>;

    bool setHandshake(const AgentUpdateToMasterDataPtr &handshake);
    static AgentUpdateToMasterDataPtr makeUpdate(NaviSchedAgentEvent::Type event,
                                                 int64_t eventTimeUs);
    void attachInflightStats(const AgentUpdateToMasterDataPtr &update) const;
    QueuedPublishPtr enqueueWorkloadLocked(NaviSchedWorkloadSnapshot &snapshot,
                                          AgentUpdateToMasterDataPtr update = {});
    void finishWorkloadPublish(const QueuedPublishPtr &task) const;
    NaviSchedWorkloadSnapshot observedWorkloadLocked() const;
    bool retirePendingCacheLocked(int64_t localBatchId,
                                  QueuedPublishPtr *publishTask = nullptr) const;
    AgentUpdateToMasterDataPtr makeHandshakeUpdate() const;
    QueuedPublishPtr enqueuePublish(
        const AgentUpdateToMasterDataPtr &update, bool trackStats = true) const;
    QueuedPublishPtr enqueuePublishLocked(
        std::vector<AgentUpdateToMasterDataPtr> updates,
        std::vector<SubscriberPtr> subscribers, bool trackStats,
        bool allowStopping = false) const;
    void drainPublishQueue() const;
    size_t waitPublish(const QueuedPublishPtr &publishTask) const;
    void deliverPublish(const QueuedPublishPtr &publishTask) const;
    void recordPublishedEvent(const AgentUpdateToMasterDataPtr &update,
                              bool noSubscribers) const;
    bool removeSubscriberLocked(const std::string &collectorId,
                                const SubscriberPtr *expected = nullptr) const;
    bool removeSubscriber(const SubscriberPtr &subscriber) const;

private:
    RESOURCE_DEPEND_DECLARE();

private:
    RESOURCE_DEPEND_ON(AgentCacheR, _agentCacheR);
    RESOURCE_DEPEND_ON(AgentPendingCacheR, _agentPendingCacheR);
    RESOURCE_DEPEND_ON(AgentReportBridgeR, _agentReportBridgeR);
    RESOURCE_DEPEND_ON(AgentSchedulerConfigR, _agentSchedulerConfigR);
    mutable autil::ReadWriteLock _latencySnapshotLock;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> _latencySnapshot;
    std::shared_ptr<const RequestLatencyTracker::Snapshot> _inflightStats =
        std::make_shared<const RequestLatencyTracker::Snapshot>();
    std::shared_ptr<ConcurrentRequestLatencyTracker> _inflightTracker;
    std::atomic<uint64_t> _inflightRecomputeCount{0};
    std::atomic<uint64_t> _inflightRecomputedRequestCount{0};
    std::atomic<uint64_t> _inflightRecomputeFailureCount{0};
    NaviSchedWorkloadSnapshotConstPtr _workloadSnapshot;
    NaviSchedWorkloadSnapshot _observedWorkload;
    AgentWorkloadOwnership _nativeWorkload;
    mutable autil::ReadWriteLock _latencyValidationStatsLock;
    mutable NaviSchedLatencyValidationStats _latencyValidationStats;
    mutable int64_t _latencyValidationLastUpdateTimeUs = 0;
    mutable std::vector<int64_t> _latencyValidationBucketLastUpdateTimeUs;
    mutable autil::ThreadCond _mutex;
    mutable autil::ThreadMutex _cacheReportLock;
    mutable autil::LockFreeQueue<QueuedPublishPtr> _publishQueue;
    mutable std::atomic<bool> _publishDraining{false};
    AgentUpdateToMasterDataPtr _handshake;
    NaviSchedServerInfo _serverInfo;
    int64_t _agentStartTimeUs = 0;
    bool _pdSeparated = true;
    bool _unifiedLatencyPrediction = false;
    std::atomic<bool> _latencyValidationEnabled{false};
    bool _workloadPublishTimingEnabled = false;
    std::string _latencyModelFingerprint;
    mutable std::map<std::string, SubscriberPtr> _subscribers;
    mutable std::set<std::string> _pendingStopAcks;
    bool _stopping = false;
    mutable std::atomic<uint64_t> _publishedEventCount{0};
    mutable std::atomic<uint64_t> _publishedRequestReceivedCount{0};
    mutable std::atomic<uint64_t> _publishedPromptTokenCount{0};
    mutable std::atomic<uint64_t> _publishedFinishedBatchCount{0};
    mutable std::atomic<uint64_t> _publishedAbnormalFinishedBatchCount{0};
    mutable std::atomic<uint64_t> _publishedNormalFinishedRequestCount{0};
    mutable std::atomic<uint64_t> _publishedAbnormalFinishedRequestCount{0};
    mutable std::atomic<uint64_t> _deliverySuccessCount{0};
    mutable std::atomic<uint64_t> _deliveryFailureCount{0};
    mutable std::atomic<uint64_t> _noSubscriberEventCount{0};
    mutable std::atomic<uint64_t> _publishedWorkloadUpdateCount{0};
    mutable std::atomic<uint64_t> _suppressedLatencyValidationBatchCount{0};
    mutable autil::LockFreeQueue<int64_t> _workloadPublishTimeQueue;
    DECLARE_LOGGER();
};

NAVI_TYPEDEF_PTR(AgentReportR);

} // namespace navi_sched
