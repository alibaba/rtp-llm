#pragma once

#include <autil/Lock.h>
#include <atomic>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "aios/navi_sched/data/AgentUpdateToMasterData.h"
#include "aios/navi_sched/data/ClientRequestToMasterData.h"
#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/resource/MasterBatchReservation.h"
#include "aios/navi_sched/resource/MasterLatencyModelR.h"
#include "aios/navi_sched/resource/MasterPendingCacheR.h"
#include "aios/navi_sched/scheduler/ScheduleContext.h"
#include "aios/navi_sched/scheduler/ScheduleContextInternal.h"
#include "kmonitor/client/MetricsReporterR.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"

class NaviSchedResourceTest;
class NaviSchedClusterTest;

namespace navi_sched {

class MasterScheduleContextKernel;

class MasterStateR : public navi::Resource {
    friend class ::NaviSchedResourceTest;
    friend class ::NaviSchedClusterTest;

public:
    using LatencyInfo = NaviSchedLatencyModel;
    struct AgentUpdateTiming {
        int64_t beginMonoUs = 0;
        int64_t runtimeLockBeginMonoUs = 0;
        int64_t runtimeLockAcquiredMonoUs = 0;
        int64_t runtimeLockReleasedMonoUs = 0;
        int64_t endMonoUs = 0;
    };
    struct AgentRequestStats {
        uint64_t reportCount = 0;
        uint64_t handshakeCount = 0;
        uint64_t stoppingCount = 0;
        uint64_t receivedCount = 0;
        uint64_t executionLoadCount = 0;
        uint64_t batchFinishedCount = 0;
        uint64_t finishedCount = 0;
        uint64_t abnormalFinishedCount = 0;
        uint64_t cacheUpdateCount = 0;
        uint64_t pendingCacheUpdateCount = 0;
        uint64_t latencyModelUpdateCount = 0;
        uint64_t workloadUpdateCount = 0;
        int64_t cacheHitTokens = 0;
        int64_t cacheMissTokens = 0;
        int64_t totalLatencyUs = 0;
    };
    struct ScheduleRoundStats {
        uint64_t roundCount = 0;
        uint64_t requestCount = 0;
        size_t minRequestCount = 0;
        size_t maxRequestCount = 0;
        std::map<size_t, uint64_t> requestCountHistogram;
    };
    struct NodeDispatchStats {
        uint64_t requestCount = 0;
        uint64_t batchCount = 0;
    };
    struct ReservationStats {
        size_t batchCount = 0;
        size_t requestCount = 0;
        size_t waitingRequestCount = 0;
        size_t dispatchedRequestCount = 0;
        size_t dispatchWaiterCount = 0;
        size_t liveDispatchWaiterCount = 0;
    };
    struct CacheSyncStats {
        uint64_t agentRevision = 0;
        int64_t agentCachedBlockCount = -1;
        uint64_t agentHashChecksum = 0;
        int64_t agentFullReadyBlockCount = -1;
        uint64_t agentFullReadyHashChecksum = 0;
        int64_t agentTailReadyBlockCount = -1;
        uint64_t agentTailReadyHashChecksum = 0;
        size_t masterCachedBlockCount = 0;
        uint64_t masterHashChecksum = 0;
        size_t masterFullReadyBlockCount = 0;
        uint64_t masterFullReadyHashChecksum = 0;
        size_t masterTailReadyBlockCount = 0;
        uint64_t masterTailReadyHashChecksum = 0;
        int64_t rtpCachedBlockCount = -1;
        uint64_t revisionGapCount = 0;
        uint64_t stateMismatchCount = 0;
        uint64_t rtpMirrorSampleCount = 0;
        uint64_t rtpMirrorMismatchCount = 0;
        bool cacheStateReady = false;
        size_t pendingCacheBatchCount = 0;
        size_t pendingCacheBlockCount = 0;
        size_t pendingCacheTailReadyBlockCount = 0;
        int64_t cacheBlockSize = 0;
        int64_t cacheLinearStep = 0;
        bool cachePolicyReady = false;
        bool hybridTailRequired = false;
        uint64_t cachePolicyMismatchCount = 0;
        size_t pendingCacheScheduledBatchCount = 0;
        size_t pendingCacheAgentBatchCount = 0;
        uint64_t pendingCacheAgentRevision = 0;
        uint64_t pendingCacheAgentChecksum = 0;
        uint64_t pendingCacheRevisionGapCount = 0;
        uint64_t pendingCacheStateMismatchCount = 0;
        uint64_t pendingCacheResyncTriggerCount = 0;
        bool pendingCacheSnapshotReady = false;
    };
    struct CacheMatchBreakdown {
        int64_t staticMatchedBlockCount = 0;
        int64_t matchedBlockCount = 0;
        int64_t firstPendingBlockSourceBatchId = 0;
        int64_t pendingTailSourceBatchId = 0;
        uint64_t cacheRevision = 0;
        size_t pendingBatchCount = 0;
        size_t pendingBlockCount = 0;
        bool valid = false;
    };
    class NodeState : public NaviSchedNode {
    public:
        NaviSchedNodeInfo getInfo() const override;
        bool getScheduleSnapshot(const std::string &bizName,
                                 NaviSchedNodeSnapshot &snapshot) const override;
        int64_t getMatchedBlockCount(
            const NaviSchedBlockHashes &blockHashes) const override;
        int64_t getMatchedBlockCountForRequest(
            const NaviSchedScheduleRequest &request) const override;
        void getMatchedBlockCounts(
            const std::vector<NaviSchedScheduleItem> &requests,
            int64_t *matchedBlockCounts) const override;
        void getMatchedBlockCounts(
            const std::vector<NaviSchedScheduleItem> &requests,
            int64_t *matchedBlockCounts,
            const std::vector<size_t> &commonPrefixes) const override;
        double getExecutionUtilization() const override;
        double getExecutionBusyUtilization() const override;
        size_t getCachedBlockCount() const;
        std::shared_ptr<const LatencyInfo> getLatencyInfo() const;
        LatencyResidualModel::RuntimeConfig getResidualRuntimeConfig() const;
        NodeDispatchStats getDispatchStats() const;
        ReservationStats getReservationStats() const;
        CacheSyncStats getCacheSyncStats() const;
        bool getCachePolicy(NaviSchedCachePolicy &policy) const;
        CacheMatchBreakdown getCacheMatchBreakdown(
            const NaviSchedScheduleRequest &request) const;
        autil::legacy::json::JsonMap makePendingReplayState() const;

    private:
        friend class MasterStateR;
        friend class MasterDispatchHandle;
        friend class ::NaviSchedResourceTest;
        friend class ::NaviSchedClusterTest;
        std::atomic<double> _executionUtilization{-1.0};
        std::atomic<double> _executionBusyUtilization{-1.0};
        struct RuntimeState {
            NaviSchedNodeLoad load;
            double lastRequestLatencyMs = 0.0;
            NaviSchedAgentEvent::Type lastEvent = NaviSchedAgentEvent::UNKNOWN;
            int64_t lastEventTimeUs = 0;
            NaviSchedAgentEvent::Type lastRequestEvent = NaviSchedAgentEvent::UNKNOWN;
            int64_t lastRequestEventTimeUs = 0;
            int64_t lastReceiveTimeUs = 0;
            NaviSchedLatencyValidationStats latencyValidation;
            NaviSchedWorkloadSnapshot workload;
            bool unifiedLatencyPrediction = false;
            int64_t workloadReportReceiveTimeUs = 0;
            uint64_t workloadStaleUpdateCount = 0;
            int64_t agentStartTimeUs = 0;
        };
        struct PublishedSnapshot {
            std::shared_ptr<const NaviSchedNodeInfo> info;
            std::shared_ptr<const LatencyInfo> latencyModel;
        };
        struct PendingBatchReservation;
        using ReservationPtr = std::shared_ptr<PendingBatchReservation>;
        using ReservationList = std::list<ReservationPtr>;
        struct PendingBatchReservation {
            MasterBatchReservation batch;
            std::unordered_map<uint64_t, std::weak_ptr<navi::AsyncPipe>>
                dispatchWaiters;
            ReservationList::iterator position;
            bool attached = false;
        };
        struct PendingReplaySnapshot {
            int64_t captureTimeUs = 0;
            std::string agentId;
            uint64_t agentIdHash = 0;
            NaviSchedBlockHashes cachedBlockHashes;
            NaviSchedBlockHashes fullReadyBlockHashes;
            NaviSchedBlockHashes tailReadyBlockHashes;
            bool hybridTailRequired = false;
            int64_t blockSize = -1;
            std::vector<MasterBatchReservation> reservations;
            bool hasPendingCache = false;
            MasterPendingCacheR::ReplaySnapshot pendingCache;

            autil::legacy::json::JsonMap toJson() const;
        };
        PendingReplaySnapshot capturePendingReplaySnapshot() const;
        void rebuildInfoSnapshot();
        void setLatencyModelSlot(
            const MasterLatencyModelR::LiveModelSlotPtr &slot);
        void clearLatencyModelSlot();
        void fillDynamicInfo(NaviSchedNodeInfo &info) const;
        RuntimeState getRuntimeState() const;
        NaviSchedNodeLoad getScheduleLoad(int64_t currentTimeUs) const;
        NaviSchedNodeLoad getScheduleLoadLocked(int64_t currentTimeUs) const;
        struct ReservationTiming {
            const char *failureReason = nullptr;
            int64_t initUs = 0;
            int64_t initCpuUs = 0;
            int64_t runtimeWaitUs = 0;
            int64_t runtimeHoldUs = 0;
            int64_t runtimeHoldCpuUs = 0;
            int64_t pendingPublishUs = 0;
            int64_t pendingPublishCpuUs = 0;
            int64_t totalUs = 0;
            int64_t totalCpuUs = 0;
        };
        struct FailureTiming {
            int64_t copyUs = 0;
            int64_t markFailedUs = 0;
            int64_t predictionUs = 0;
            int64_t pendingUpdateUs = 0;
            int64_t commitUs = 0;
        };
        bool reserveBatch(
            int64_t batchId, int64_t scheduleTimeUs,
            double predictedLatencyMs,
            std::vector<MasterBatchReservation::Request> requests,
            const NaviSchedCachePolicy *cachePolicy = nullptr,
            bool *pendingCacheTracked = nullptr,
            ReservationPtr *reserved = nullptr,
            ReservationTiming *timing = nullptr);
        bool registerDispatchWaiter(int64_t batchId, uint64_t requestId,
                                    const navi::AsyncPipePtr &pipe,
                                    bool &acknowledged);
        bool unregisterDispatchWaiter(int64_t batchId, uint64_t requestId,
                                      const navi::AsyncPipePtr &pipe);
        bool cancelDispatchWaiter(int64_t batchId, uint64_t requestId,
                                  const navi::AsyncPipePtr &pipe,
                                  bool &batchReleased);
        bool reportDispatchFailed(int64_t batchId, uint64_t requestId,
                                  bool &batchReleased);
        bool reportDispatchFailures(int64_t batchId,
                                    const std::vector<uint64_t> &requestIds,
                                    bool &batchReleased);
        bool abandonUnownedDispatch(int64_t batchId, uint64_t requestId,
                                    bool &batchReleased);
        PendingBatchReservation *findReservationLocked(int64_t batchId);
        void eraseReservationLocked(PendingBatchReservation &reservation);
        void clearReservationsLocked();
        bool updateReservationPendingCacheLocked(
            const MasterBatchReservation &reservation,
            const MasterBatchReservation::Transition *transition = nullptr);
        bool refreshReservationPredictionLocked(
            MasterBatchReservation &batch, bool recomputeCacheHits);
        void clearCacheStateLocked();
        static void collectDispatchWaitersLocked(
            PendingBatchReservation &reservation,
            const std::vector<uint64_t> &requestIds,
            std::vector<navi::AsyncPipePtr> &pipes);
        bool failRequestLocked(PendingBatchReservation &reservation,
                               uint64_t requestId, bool &batchReleased);
        bool failRequestsLocked(PendingBatchReservation &reservation,
                                const std::vector<uint64_t> &requestIds,
                                bool &batchReleased);
        bool failRequestIndexesLocked(PendingBatchReservation &reservation,
                                      const std::vector<size_t> &requestIndexes,
                                      bool &batchReleased,
                                      FailureTiming *timing = nullptr);
        int64_t matchBlockCount(const NaviSchedBlockHashes &blockHashes) const;
        mutable autil::ReadWriteLock _infoLock;
        NaviSchedNodeInfo _info;
        std::shared_ptr<const NaviSchedNodeInfo> _infoSnapshot;
        std::shared_ptr<const PublishedSnapshot> _publishedSnapshot;
        mutable autil::ReadWriteLock _cacheLock;
        MasterPendingCacheR::PhysicalBlockSet _cachedBlockHashes;
        MasterPendingCacheR::PhysicalBlockSet _fullReadyBlockHashes;
        MasterPendingCacheR::PhysicalBlockSet _tailReadyBlockHashes;
        bool _hybridTailRequired = false;
        NaviSchedCachePolicy _cachePolicy;
        std::atomic<bool> _cachePolicyReady{false};
        std::atomic<bool> _cacheStateReady{false};
        mutable std::atomic<uint64_t> _cachePolicyMismatchCount{0};
        uint64_t _cacheRevision = 0;
        int64_t _agentCachedBlockCount = -1;
        uint64_t _agentCacheHashChecksum = 0;
        int64_t _agentFullReadyBlockCount = -1;
        uint64_t _agentFullReadyHashChecksum = 0;
        int64_t _agentTailReadyBlockCount = -1;
        uint64_t _agentTailReadyHashChecksum = 0;
        uint64_t _masterCacheHashChecksum = 0;
        uint64_t _masterFullReadyHashChecksum = 0;
        uint64_t _masterTailReadyHashChecksum = 0;
        int64_t _rtpCachedBlockCount = -1;
        uint64_t _cacheRevisionGapCount = 0;
        uint64_t _cacheStateMismatchCount = 0;
        uint64_t _rtpCacheMirrorSampleCount = 0;
        uint64_t _rtpCacheMirrorMismatchCount = 0;
        std::atomic<uint64_t> _pendingCacheResyncTriggerCount{0};
        MasterPendingCacheR *_pendingCacheR = nullptr;
        std::shared_ptr<MasterLatencyModelR::LiveModelSlot> _latencyModelSlot;
        std::shared_ptr<const LatencyResidualModel::RuntimeConfig>
            _residualRuntimeConfig =
                std::make_shared<const LatencyResidualModel::RuntimeConfig>();
        mutable autil::ThreadMutex _runtimeLock;
        RuntimeState _runtimeState;
        ReservationList _pendingBatchReservations;
        std::unordered_map<int64_t, ReservationPtr> _reservationsById;
        bool _acceptReservations = true;
        std::atomic<uint64_t> _dispatchedRequestCount{0};
        std::atomic<uint64_t> _dispatchedBatchCount{0};
    };
    NAVI_TYPEDEF_PTR(NodeState);
    using NodeStateConstPtr = std::shared_ptr<const NodeState>;
    using NodeStateMap = std::unordered_map<uint64_t, NodeStatePtr>;
    using NodeStateMapConstPtr = std::shared_ptr<const NodeStateMap>;

public:
    MasterStateR();

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    NodeStatePtr registerAgent(const std::string &nodeId, const std::string &bizName,
                               const std::string &address, int64_t totalBlocks);
    NodeStatePtr createAgent(const std::string &nodeId, const std::string &bizName,
                             const std::string &address, int64_t totalBlocks);
    bool removeAgent(const NodeStatePtr &state);
    bool openAgentCollector(const std::string &nodeId, const std::string &collectGraphId);
    bool closeAgentCollector(const std::string &nodeId,
                             const std::string &collectorId);
    bool stopAgentCollector(const std::string &nodeId,
                            const std::string &collectorId);
    bool bindAgentCollector(const std::string &collectorId, const std::string &nodeId,
                            const NodeStatePtr &state = nullptr);
    bool unbindAgentCollector(const std::string &collectorId);
    bool getAgentCollectorNodeId(const std::string &collectorId,
                                 std::string &nodeId) const;
    bool resetAgentInstance(const std::string &nodeId);
    bool reportAgent(const std::string &agentId,
                     const AgentUpdateToMasterDataConstPtr &update);
    bool reportAgent(const NodeStatePtr &state,
                     const AgentUpdateToMasterDataConstPtr &update,
                     bool *resyncRequired = nullptr,
                     bool *pendingCacheResyncRequired = nullptr,
                     MasterPendingCacheR::ApplyResult *pendingCacheApplyResult =
                         nullptr,
                     const std::string &latencySourceId = "",
                     MasterLatencyModelR::RecordResult *latencyRecordResult =
                         nullptr,
                     AgentUpdateTiming *timing = nullptr);
    bool receiveClientRequest(const ClientRequestToMasterData &request);
    static NaviSchedScheduleItem makeScheduleItem(
        const ClientRequestToMasterData &request, int64_t waitTimeMs);
    void snapshotScheduleNodes(const std::string &bizName,
                               NaviSchedScheduleContext &context) const;
    void reserveScheduleBatches(NaviSchedScheduleContext &context);
    void reserveScheduleNodeBatch(NaviSchedScheduleContext &context, size_t nodeIndex,
                                  const std::vector<size_t> &requestIndexes);
    bool registerDispatchWaiter(const NaviSchedTarget &target, int64_t batchId,
                                uint64_t requestId,
                                const navi::AsyncPipePtr &pipe,
                                bool &acknowledged);
    bool unregisterDispatchWaiter(const NaviSchedTarget &target, int64_t batchId,
                                  uint64_t requestId,
                                  const navi::AsyncPipePtr &pipe);
    bool cancelDispatchWaiter(const NaviSchedTarget &target, int64_t batchId,
                              uint64_t requestId,
                              const navi::AsyncPipePtr &pipe);
    bool reportDispatchFailed(const NaviSchedTarget &target, int64_t batchId,
                              uint64_t requestId);
    bool reportDispatchFailures(const NaviSchedTarget &target, int64_t batchId,
                                const std::vector<uint64_t> &requestIds);
    bool reportDispatchAbandoned(const NaviSchedTarget &target, int64_t batchId,
                                 uint64_t requestId);
    std::shared_ptr<NaviSchedScheduleReplay> makeScheduleReplay(
        const NaviSchedScheduleContext &context) const;
    NodeStateConstPtr getNodeState(uint64_t agentIdHash) const;
    std::vector<NodeStateConstPtr> getNodeStates() const;
    std::map<std::string, NaviSchedNodeInfo> getNodes() const;
    AgentRequestStats getAgentRequestStats() const;
    void recordScheduleRound(size_t requestCount);
    ScheduleRoundStats getScheduleRoundStats() const;
    uint64_t getClientRequestCount() const;
    uint64_t getAgentReportCount() const;
    std::map<std::string, std::string> getAgentCollectors() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    static std::string getHost(const std::string &address);
    static NaviSchedScheduleRequest makeScheduleRequest(
        const ClientRequestToMasterData &request);
    static bool validLatencyModel(const AgentLatencyToMasterDataPtr &latency);
    NodeStatePtr findNodeState(const NaviSchedTarget &target) const;
    NodeStatePtr getOrCreateNode(uint64_t agentIdHash, const std::string &agentId);
    bool resetAgentState(const NodeStatePtr &state,
                         MasterPendingCacheR::Stats *clearedPendingStats = nullptr);
    bool isCurrentNodeStateLocked(uint64_t agentIdHash,
                                  const NodeStatePtr &state) const;
    void updateNodeBiz(uint64_t agentIdHash, const NodeStatePtr &state,
                       const std::string &bizName);
    void refreshBizNode(const NodeStatePtr &state);
    void addBizNode(const std::string &bizName, uint64_t agentIdHash,
                    const NodeStatePtr &state);
    void removeBizNode(const std::string &bizName, uint64_t agentIdHash,
                       const NodeStatePtr &state);
    void snapshotNodeStates(std::vector<NodeStateConstPtr> &states,
                            std::unordered_set<const NodeState *> *readyStates) const;

private:
    RESOURCE_DEPEND_DECLARE();
    RESOURCE_DEPEND_ON_FALSE(kmonitor::MetricsReporterR, _metricsReporterR);
    RESOURCE_DEPEND_ON(MasterPendingCacheR, _pendingCacheR);
    RESOURCE_DEPEND_ON(MasterLatencyModelR, _masterLatencyModelR);
    mutable autil::ReadWriteLock _nodeIndexLock;
    NodeStateMap _pendingNodes;
    std::unordered_map<std::string, NodeStateMapConstPtr> _readyNodesByBiz;
    std::atomic<uint64_t> _clientRequestCount{0};
    std::atomic<uint64_t> _agentReportCount{0};
    std::atomic<uint64_t> _agentHandshakeCount{0};
    std::atomic<uint64_t> _agentStoppingCount{0};
    std::atomic<uint64_t> _agentRequestReceivedCount{0};
    std::atomic<uint64_t> _agentExecutionLoadCount{0};
    std::atomic<uint64_t> _agentBatchFinishedCount{0};
    std::atomic<uint64_t> _agentRequestFinishedCount{0};
    std::atomic<uint64_t> _agentRequestAbnormalFinishedCount{0};
    std::atomic<uint64_t> _agentCacheUpdateCount{0};
    std::atomic<uint64_t> _agentPendingCacheUpdateCount{0};
    std::atomic<uint64_t> _agentLatencyModelUpdateCount{0};
    std::atomic<uint64_t> _agentWorkloadUpdateCount{0};
    std::atomic<int64_t> _agentRequestCacheHitTokens{0};
    std::atomic<int64_t> _agentRequestCacheMissTokens{0};
    std::atomic<int64_t> _agentRequestTotalLatencyUs{0};
    mutable autil::ThreadMutex _scheduleRoundStatsLock;
    ScheduleRoundStats _scheduleRoundStats;
    mutable autil::ReadWriteLock _collectorLock;
    std::map<std::string, std::string> _agentCollectors;
    std::unordered_map<std::string, std::string> _collectorNodeIds;
    std::unordered_map<std::string, std::weak_ptr<NodeState>> _collectorNodeStates;
};

NAVI_TYPEDEF_PTR(MasterStateR);

}
