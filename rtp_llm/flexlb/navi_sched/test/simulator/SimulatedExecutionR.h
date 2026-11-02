#pragma once

#include "aios/navi_sched/resource/AgentBatchR.h"
#include "aios/navi_sched/resource/AgentReportR.h"
#include "aios/navi_sched/resource/AgentRequestR.h"
#include "aios/navi_sched/resource/AgentLatencyInputR.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "aios/navi_sched/resource/RequestLatencyTracker.h"
#include "autil/Lock.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include <atomic>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace navi_sched {

class NaviSchedAgent;

namespace test {

class SimulatedExecutionKernel;
class SimulatedExecutionR;

class SimulatedExecutionBridgeR : public navi::ExternalResource {
public:
    SimulatedExecutionBridgeR();

public:
    void setExecutionR(const std::shared_ptr<SimulatedExecutionR> &resource);
    void clearExecutionR(const SimulatedExecutionR *resource);
    std::shared_ptr<SimulatedExecutionR> waitExecutionR(int64_t timeoutUs);

public:
    static const std::string RESOURCE_ID;

private:
    autil::ThreadCond _condition;
    std::shared_ptr<SimulatedExecutionR> _executionR;
};

NAVI_TYPEDEF_PTR(SimulatedExecutionBridgeR);

struct SimulatedRequest {
    uint64_t requestId = 0;
    std::vector<int64_t> promptTokenIds;
    NaviSchedBlockHashesConstPtr blockHashes;
    navi::AsyncPipePtr completionPipe;
    int64_t prefillTokenCount = 0;
    int64_t cacheHitBlocks = 0;
    int64_t cacheMissBlocks = 0;
    int64_t batchId = 0;
    int64_t expectedBatchSize = 0;
    int64_t receiveTimeUs = 0;
    int64_t batchReadyTimeUs = 0;
    double predictedQueueLatencyMs = 0.0;
    double predictedExecutionLatencyMs = 0.0;
    NaviSchedLatencyPredictionConstPtr latencyPrediction;
    NaviSchedLatencyTrainingPhase latencyTrainingPhase =
        NaviSchedLatencyTrainingPhase::NORMAL;
    int64_t predictedCacheHitTokenCount = 0;
    double batchWaitMs = 0.0;
    double queueWaitMs = 0.0;
};

struct SimulatedExecutionConfig {
    int64_t cacheCapacityBytes = 300LL * 1024 * 1024 * 1024;
    int64_t tokenCacheBytes = 10LL * 1024 * 1024 * 1024 / (1024 * 1024);
    int64_t blockSize = 256;
    std::vector<double> formulaParameters;
    double noiseStddevRatio = 0.01;
    uint64_t randomSeed = 1;
    double timeCompression = 1.0;
    std::string queueTraceOutputPath;
};

struct SimulatedBatchLatency {
    double predictedLatencyMs = 0.0;
    double executionLatencyMs = 0.0;
};

class SimulatedExecutionR : public navi::Resource {
public:
    SimulatedExecutionR() = default;
    ~SimulatedExecutionR() override;

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    bool submit(SimulatedRequest request);
    bool submit(const NaviSchedForwardRequest &request,
                const navi::AsyncPipePtr &completionPipe);
    bool takeSubmitted(std::vector<SimulatedRequest> &requests);
    SimulatedBatchLatency predictBatch(const std::vector<SimulatedRequest> &requests);
    bool startBatch(std::vector<SimulatedRequest> &requests);
    double predictModelLatencyMs(const std::vector<SimulatedRequest> &requests) const;
    double getWaitLatencyMs(double latencyMs) const;
    double getLogicalElapsedMs(double elapsedMs) const;
    void finishBatch(const std::vector<SimulatedRequest> &requests, double latencyMs);
    bool setRequestPipe(const navi::AsyncPipePtr &requestPipe);
    bool finishGraph();
    int32_t getInflightRequestCount() const;
    size_t getCachedBlockCount() const;
    uint64_t getEvictedBlockCount() const;
    static const std::vector<double> &getFormulaParameters();

public:
    static const std::string RESOURCE_ID;

private:
    friend class SimulatedExecutionRTest;
    friend class SimulatedExecutionKernel;

    struct CacheEntry {
        std::list<uint64_t>::iterator position;
    };
    void readCache(SimulatedRequest &request);
    NaviSchedCacheUpdate updateCache(const NaviSchedBlockHashesConstPtr &blockHashes);
    bool submitBatchRequest(SimulatedRequest request, int64_t batchId,
                            int64_t expectedBatchSize);
    bool submitRequest(SimulatedRequest request);
    bool enqueueBatch(const NaviSchedBatchPtr &batch);
    void rejectBatch(const NaviSchedBatchPtr &batch);
    void finishRequest(const SimulatedRequest &request, double latencyMs);
    RequestLatencyTracker::Snapshot removeInflightRequests(
        const std::vector<uint64_t> &requestIds, const char *phase);
    void handleAbortedBatch(const AgentBatchR::ExecutionState &state, const char *phase);
    void abortBatch(const std::vector<SimulatedRequest> &requests);
    void releaseRunningRequests(int32_t requestCount, const char *phase);
    bool finishResidentGraph();
    void reportRequestReceived(const NaviSchedBatchRequest &request);
    double getPredictedLatencyMs(const std::vector<SimulatedRequest> &requests) const;
    NaviSchedLatencySample makeLatencySample(
        const std::vector<SimulatedRequest> &requests, double latencyMs) const;
    NaviSchedRequestStats makeRequestStats(const SimulatedRequest &request) const;
    NaviSchedBlockHashesConstPtr calculateBlockHashes(
        const std::vector<int64_t> &tokenIds);
    bool writeQueueTrace() const;

    struct QueueTraceEntry {
        uint64_t requestId = 0;
        int64_t batchId = 0;
        int64_t batchReadyTimeUs = 0;
        int64_t batchStartTimeUs = 0;
        int64_t totalTokenCount = 0;
        int64_t cacheHitTokenCount = 0;
    };

private:
    RESOURCE_DEPEND_DECLARE();

private:
    RESOURCE_DEPEND_ON(AgentReportR, _agentReportR);
    RESOURCE_DEPEND_ON(AgentRequestR, _agentRequestR);
    RESOURCE_DEPEND_ON(AgentBatchR, _agentBatchR);
    RESOURCE_DEPEND_ON(AgentLatencyInputR, _agentLatencyInputR);
    RESOURCE_DEPEND_ON(LatencyModelR, _latencyModelR);
    RESOURCE_DEPEND_ON(NonLinearPrefillModelR, _nonLinearPrefillModelR);
    RESOURCE_DEPEND_ON_FALSE(SimulatedExecutionBridgeR, _executionBridgeR);
    SimulatedExecutionConfig _config;
    mutable autil::ThreadMutex _stateLock;
    std::vector<SimulatedRequest> _submitted;
    std::unordered_map<uint64_t, SimulatedRequest> _batchRequests;
    navi::AsyncPipePtr _requestPipe;
    size_t _waitingRequestCount = 0;
    bool _finishRequested = false;
    ConcurrentRequestLatencyTracker _inflightState;
    std::atomic<int32_t> _runningRequestCount{0};
    mutable autil::ThreadMutex _cacheLock;
    std::list<uint64_t> _cacheLru;
    std::unordered_map<uint64_t, CacheEntry> _cacheEntries;
    int64_t _cacheUsedBytes = 0;
    std::atomic<uint64_t> _evictedBlockCount{0};
    std::vector<QueueTraceEntry> _queueTrace;
};

}
}
