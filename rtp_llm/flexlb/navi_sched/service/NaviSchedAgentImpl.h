#pragma once

#include "rtp_llm/flexlb/navi_sched/resource/AgentBatchR.h"
#include "rtp_llm/flexlb/navi_sched/resource/RequestLatencyTracker.h"
#include "rtp_llm/flexlb/navi_sched/service/NaviSchedAgent.h"
#include "rtp_llm/flexlb/navi_sched/service/NaviSchedRole.h"
#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "navi/engine/AsyncPipe.h"

namespace navi_sched {

class AgentGraph;
class AgentLatencyInputR;
class AgentReportBridgeR;
class AgentReportR;
class AgentRequestR;
class AgentResponseR;
class AgentForwardState;

class NaviSchedAgentImpl final : public NaviSchedRole {
public:
    static NaviSchedAgentImpl *getImpl(NaviSchedAgent &agent);
    NaviSchedAgentImpl(const NaviSchedAgentConfig &agentConfig, int32_t serverPort,
                       const std::string &serverIp, int64_t totalBlocks);
    ~NaviSchedAgentImpl() override;

    bool init();
    bool addBatchRequest(const NaviSchedBatchRequest &request, int64_t batchId,
                         int64_t expectedBatchSize,
                         const NaviSchedReadyBatchCallback &callback,
                         const std::function<void()> &acceptedCallback = {});
    bool abortPendingBatch(int64_t scheduleBatchId);
    bool setForwardBatchHandler(const NaviSchedForwardBatchHandler &handler,
                                bool streamResponse = true,
                                const NaviSchedForwardRequestAcceptedHandler
                                    &requestAcceptedHandler = {});
    bool publishForwardResponse(uint64_t requestId, std::vector<std::string> frames,
                                bool terminal) const;
    bool failForwardResponse(uint64_t requestId, const std::string &errorMessage) const;
    void setForwardCancelHandler(const std::function<void(uint64_t)> &handler) const;
    bool recordObservedBatch(
        const std::vector<NaviSchedRequestStats> &requests, double latencyMs,
        int64_t eventTimeUs,
        const std::vector<NaviSchedBatchFinishReporterPtr> &finishReporters) const;
    bool reportObservedExecutionLoad(int32_t queuedBatchRequests, int32_t runningRequests,
                                     int64_t eventTimeUs) const;
    bool reportObservedWorkload(const NaviSchedWorkloadSnapshot &snapshot,
                               const std::vector<int64_t> &coveredNativeBatchIds = {}) const;
    bool beginObservedBackendExecution() const;
    bool endObservedBackendExecution() const;
    bool reportObservedExecutionStep(double elapsedMs, int64_t prefillTokens,
                                     int64_t decodeTokens) const;
    bool reportObservedCoreStep(double elapsedMs, bool futureWait) const;
    void reportCacheInserted(const NaviSchedBlockHashesConstPtr &blockHashes) const;
    void reportCacheRemoved(const NaviSchedBlockHashesConstPtr &blockHashes) const;
    void reportCacheUpdated(
        int64_t totalBlocks, int64_t blockSize, int64_t linearStep,
        const NaviSchedBlockHashesConstPtr &addedBlockHashes,
        const NaviSchedBlockHashesConstPtr &removedBlockHashes,
        const NaviSchedBlockHashesConstPtr &addedFullReadyBlockHashes,
        const NaviSchedBlockHashesConstPtr &removedFullReadyBlockHashes,
        const NaviSchedBlockHashesConstPtr &addedTailReadyBlockHashes,
        const NaviSchedBlockHashesConstPtr &removedTailReadyBlockHashes,
        int8_t hybridTailRequired, int64_t eventTimeUs,
        int64_t rtpCachedBlockCount = -1) const;
    int32_t getGrpcPort() const;
    const std::shared_ptr<navi::Navi> &getRuntimeNavi() const;
    const std::string &getRuntimeBizName() const;
    static void reportCacheHitPathSamples(
        const std::shared_ptr<kmonitor::MetricsReporter> &reporter,
        const std::vector<AgentBatchR::CacheHitPathSample> &samples);
    void stop() override;

protected:
    void onUpdating() override;
    bool onUpdated() override;
    std::string makeSnapshot() const override;
    void reportMetrics() const override;

private:
    static std::string getLatencyModelStateFile(const NaviSchedConfig &config);
    bool applyConfig(const std::string &configStr);
    int32_t getInflightRequestCount() const;
    size_t getCachedBlockCount() const;
    RequestLatencyTracker::Snapshot getInflightStats() const;
    bool addInflightRequest(uint64_t requestId, double predictedLatencyMs,
                            const NaviSchedLatencyPredictionConstPtr &latencyPrediction,
                            RequestLatencyTracker::Snapshot &snapshot) const;
    RequestLatencyTracker::Snapshot removeInflightRequests(
        const std::vector<uint64_t> &requestIds, const char *phase) const;
    void handleAcceptedBatchFailure(const AgentBatchR::ExecutionState &state) const;
    NaviSchedBatchFinishReporterPtr makeBatchFinishReporter(
        int64_t batchId, const std::vector<uint64_t> &requestIds) const;
    bool submitForwardRequest(const NaviSchedForwardRequest &request,
                              const navi::AsyncPipePtr &completionPipe);
    void abortForwardBatch(int64_t scheduleBatchId,
                           const navi::AsyncPipePtr &completionPipe);
    void handleForwardBatch(const NaviSchedBatchPtr &batch);

private:
    std::shared_ptr<AgentBatchR> _agentBatchR;
    std::shared_ptr<AgentGraph> _agentGraph;
    std::shared_ptr<AgentLatencyInputR> _agentLatencyInputR;
    std::shared_ptr<AgentReportBridgeR> _agentReportBridgeR;
    std::shared_ptr<AgentReportR> _agentReportR;
    std::shared_ptr<AgentRequestR> _agentRequestR;
    std::shared_ptr<AgentResponseR> _agentResponseR;
    NaviSchedRuntimeConfig _agentConfig;
    std::string _latencyModelFingerprint;
    std::string _latencyHistoryFingerprint;
    std::string _latencyHistoryBinaryFingerprint;
    std::string _agentBizName;
    std::string _serverIp;
    int32_t _serverPort = 0;
    int64_t _cacheTotalBlocks = 0;
    bool _pdSeparated = true;
    bool _unifiedLatencyPrediction = false;
    double _timeCompression = 1.0;
    std::shared_ptr<ConcurrentRequestLatencyTracker> _inflightState;
    std::shared_ptr<AgentForwardState> _forwardState;
    mutable NaviSchedLatencyValidationStats _lastLatencyValidationStats;
    mutable AgentBatchR::CacheHitPathStats _lastCacheHitPathStats;
    mutable int64_t _lastExecutionBusyTimeUs = 0;
    mutable int64_t _lastExecutionBusyReportTimeUs = 0;
    mutable int64_t _lastBackendExecutionBusyTimeUs = 0;
    mutable int64_t _lastBackendExecutionBusyReportTimeUs = 0;
    bool _stoppingReported = false;
};

} // namespace navi_sched
