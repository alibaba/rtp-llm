#pragma once

#include "aios/navi_sched/common/NaviSchedConstants.h"
#include "aios/navi_sched/common/TPUPrefillExecutionPolicy.h"

#if defined(__GNUC__)
#define NAVI_SCHED_API __attribute__((visibility("default")))
#else
#define NAVI_SCHED_API
#endif

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "navi/pybind/NaviRuntimeConfig.h"

namespace autil {
class DataBuffer;
}

namespace navi_sched {

static const int32_t NAVI_SCHED_DATA_VERSION = 18;
static const int32_t NAVI_SCHED_LATENCY_RESIDUAL_VERSION = 2;
static const size_t NAVI_SCHED_LATENCY_RESIDUAL_PARAMETER_COUNT = 19;
NAVI_SCHED_API int64_t getNaviSchedRpcTimeoutMs(int64_t entryTimeoutMs);

enum class NaviSchedCacheKeyType : int8_t {
    BLOCK_POOL = 0,
    V6D = 1,
};

inline bool validCacheKeyType(NaviSchedCacheKeyType type)
{
    return type == NaviSchedCacheKeyType::BLOCK_POOL ||
           type == NaviSchedCacheKeyType::V6D;
}

enum class NaviSchedLatencyTrainingPhase : int8_t {
    NORMAL = 0,
    RAMP = 1,
};

inline bool validLatencyTrainingPhase(NaviSchedLatencyTrainingPhase phase)
{
    return NaviSchedLatencyTrainingPhase::NORMAL == phase ||
           NaviSchedLatencyTrainingPhase::RAMP == phase;
}

using NaviSchedBlockHashes = std::vector<uint64_t>;
using NaviSchedBlockHashesConstPtr = std::shared_ptr<const NaviSchedBlockHashes>;

struct NaviSchedCachePolicy {
    bool valid(NaviSchedCacheKeyType type = NaviSchedCacheKeyType::BLOCK_POOL) const
    {
        return validCacheKeyType(type) && blockSize > 0 && linearStep > 0 &&
               linearStep <= std::numeric_limits<int64_t>::max() / blockSize &&
               mambaRetentionInterval >= -1 && fullAttentionQueryDropTokens >= 0 &&
               (type == NaviSchedCacheKeyType::V6D
                    ? fullAttentionQueryDropTokens <= 1
                    : fullAttentionQueryDropTokens % blockSize == 0) &&
               (!mambaAlign || mambaRetentionInterval <= 0 ||
                mambaRetentionInterval % (blockSize * linearStep) == 0);
    }

    int64_t blockSize = 0;
    int64_t linearStep = 0;
    bool hybridTailRequired = false;
    // vLLM sparse Mamba retention with a one-block replay-stop shift.
    // This is a cache/scheduler policy, independent of the draft model.
    bool mambaReplayStopShift = false;
    int64_t mambaRetentionInterval = -1;
    bool mambaAlign = false;
    // Query-only margin: BlockPool subtracts whole blocks from its Full hit;
    // V6D clips the request token upper bound before matching the external suffix.
    int64_t fullAttentionQueryDropTokens = 0;
};

class NaviSchedBatchFinishReporter;
using NaviSchedBatchFinishReporterPtr = std::shared_ptr<NaviSchedBatchFinishReporter>;
struct NaviSchedLocalBatchInfo;
struct NaviSchedLatencyPrediction;

struct NaviSchedBatchRequest {
    bool valid() const;

    uint64_t requestId = 0;
    int64_t timeoutMs = 0;
    int64_t totalTokenCount = 0;
    int64_t blockSize = 256;
    NaviSchedBlockHashesConstPtr blockHashes;
    int64_t predictedCacheHitTokenCount = 0;
    double predictedQueueLatencyMs = 0.0;
    double predictedExecutionLatencyMs = 0.0;
    std::shared_ptr<const NaviSchedLatencyPrediction> latencyPrediction;
    NaviSchedLatencyTrainingPhase latencyTrainingPhase =
        NaviSchedLatencyTrainingPhase::NORMAL;
    // Local chunk observation only; execution features still use totalTokenCount.
    int64_t cachePublicationTokenCount = -1;
};

struct NaviSchedRequestStats {
    bool valid() const;

    // Local admission geometry; never serialized on the request wire.
    int64_t executionSharedPrefixBoundary = -1;
    int64_t executionExternalReadTokens = -1;
    int64_t totalTokenCount = 0;
    int64_t blockSize = 256;
    int64_t cacheHitTokenCount = 0;
    int64_t predictedCacheHitTokenCount = 0;
    uint64_t requestId = 0;
    double predictedQueueLatencyMs = 0.0;
    double predictedExecutionLatencyMs = 0.0;
    NaviSchedLatencyTrainingPhase latencyTrainingPhase =
        NaviSchedLatencyTrainingPhase::NORMAL;
    int64_t hybridCandidateBlockCount = -1;
    int64_t hybridFlatMatchedBlockCount = -1;
    int64_t hybridMinFullReuseBlockCount = -1;
    int64_t hybridSelectedReuseBlockCount = -1;
};

struct NaviSchedLatencyQuery {
    uint64_t requestId = 0;
    int64_t totalTokenCount = 0;
    int64_t cacheHitTokenCount = 0;
    int64_t hybridCandidateBlockCount = -1;
    int64_t hybridFlatMatchedBlockCount = -1;
    int64_t hybridMinFullReuseBlockCount = -1;
    int64_t hybridSelectedReuseBlockCount = -1;
};

class AgentBatchR;

class NaviSchedBatch {
public:
    NaviSchedLocalBatchInfo makeLocalBatchInfo() const;

    int64_t batchId = 0;
    int64_t scheduleBatchId = 0;
    std::vector<uint64_t> requestIds;
    int64_t expectedBatchSize = 0;
    bool timedOut = false;
    int64_t readyTimeUs = 0;
    int64_t callbackDispatchTimeUs = 0;
    int64_t readyAcceptedCallbackTimeUs = 0;
    bool pendingCacheTracked = false;
    NaviSchedBatchFinishReporterPtr finishReporter;

private:
    friend class AgentBatchR;
    std::vector<NaviSchedRequestStats> _requestStats;
};

using NaviSchedBatchPtr = std::shared_ptr<NaviSchedBatch>;
using NaviSchedReadyBatchCallback = std::function<void(const NaviSchedBatchPtr &)>;

struct NaviSchedLocalBatchInfo {
    bool empty() const;
    bool valid() const;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    int64_t scheduleBatchId = 0;
    int64_t localBatchId = 0;
    int32_t requestCount = 0;
    bool pendingCacheTracked = false;
    std::vector<uint64_t> requestIds;
};

struct NaviSchedForwardedRequest {
    uint64_t requestId = 0;
    int64_t localBatchId = 0;
    int64_t scheduleBatchSize = 0;
    int64_t localBatchSize = 0;
    bool pendingCacheTracked = false;
    std::string payload;
    NaviSchedBlockHashesConstPtr blockHashes;
    int64_t expectedBatchSize = 0;
    bool timedOut = false;
    int64_t scheduleTimeUs = 0;
    int64_t scheduleLatencyUs = 0;
    int64_t agentArrivalTimeUs = 0;
    int64_t batchReadyTimeUs = 0;
    int64_t callbackDispatchTimeUs = 0;
    int64_t readyAcceptedCallbackTimeUs = 0;
    int64_t forwardHandlerStartTimeUs = 0;
    int64_t forwardStateLockWaitUs = 0;
    int64_t forwardBatchBuildTimeUs = 0;
    int64_t predictedCacheHitTokenCount = 0;
    double predictedQueueLatencyMs = 0.0;
    double predictedExecutionLatencyMs = 0.0;
};

struct NaviSchedWorkloadSnapshot {
    bool valid() const;
    bool sameWorkload(const NaviSchedWorkloadSnapshot &other) const;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    uint64_t revision = 0;
    int64_t eventTimeUs = 0;
    int32_t batchCount = 0;
    int32_t requestCount = 0;
    int32_t remainingPrefillRequestCount = 0;
    int32_t runningRequestCount = 0;
    double runningElapsedTimeMs = 0.0;
    int32_t runningDecodeRequestCount = 0;
    int64_t runningDecodeGeneratedTokenCount = 0;
    int64_t runningDecodeRemainingTokenBudget = 0;
    int32_t waitingRequestCount = 0;
    int64_t remainingPrefillContextTokenCount = 0;
    int64_t remainingPrefillTokenCount = 0;
    double remainingPrefillTokenSquareSum = 0.0;
    double contextRemainingPrefillTokenProductSum = 0.0;
    bool prefillCapacitySplit = false;
    int32_t maxRunningRequestCount = 0;
    int32_t runningCapacitySlotCount = 0;
    int32_t runningCapacityPrefillRequestCount = 0;
    int64_t runningCapacityPrefillContextTokenCount = 0;
    int64_t runningCapacityPrefillTokenCount = 0;
    double runningCapacityPrefillTokenSquareSum = 0.0;
    double runningCapacityContextRemainingPrefillTokenProductSum = 0.0;
    int32_t waitingCapacityPrefillRequestCount = 0;
    int64_t waitingCapacityPrefillContextTokenCount = 0;
    int64_t waitingCapacityPrefillTokenCount = 0;
    double waitingCapacityPrefillTokenSquareSum = 0.0;
    double waitingCapacityContextRemainingPrefillTokenProductSum = 0.0;
    int64_t remainingTotalContextTokenCount = 0;
    int64_t remainingTotalTokenCount = 0;
    double remainingTotalTokenSquareSum = 0.0;
    double contextRemainingTotalTokenProductSum = 0.0;
    int64_t totalPromptTokenCount = 0;
    int64_t inFlightScheduledTokenCount = 0;
    NaviSchedLocalBatchInfo admittedBatch;
};

using NaviSchedWorkloadSnapshotPtr = std::shared_ptr<NaviSchedWorkloadSnapshot>;
using NaviSchedWorkloadSnapshotConstPtr =
    std::shared_ptr<const NaviSchedWorkloadSnapshot>;

using NaviSchedForwardBatchHandler =
    std::function<bool(const std::vector<NaviSchedForwardedRequest> &, int64_t,
                       const NaviSchedBatchFinishReporterPtr &)>;
using NaviSchedForwardRequestAcceptedHandler =
    std::function<void(uint64_t, const std::string &)>;

struct NaviSchedGigConfig {
    size_t heartbeatThreadNum = 5;
    size_t heartbeatQueueSize = 1000;
    std::string logPrefix;
    std::string snapshotLogDir;
    int32_t httpPort = 0;
    int32_t arpcPort = 0;
    int32_t grpcPort = 0;
    size_t grpcIoThreadNum = 20;
};

struct NaviSchedConfig {
    std::string name;
    std::string logFileName = "./logs/navi_sched.log";
    std::string logLevel = "INFO";
    bool enableKmon = true;
    bool kmonPlatformDiscovery = false;
    std::string kmonServiceName = "navi_sched";
    std::string kmonMetricsPath;
    std::map<std::string, std::string> kmonTags;
    std::string kmonMonitorNamePrefix;
    int64_t kmonReportIntervalMs = 1000;
    NaviSchedGigConfig gig;
    bool waitNoQuery = true;
    int64_t batchTimeoutMs = 10;
    int64_t closedBatchKeepMs = 60 * 1000;
    size_t latencyFitBatchSize = 4;
    size_t latencyFitInterval = 0;
    double latencyLearningRate = 0.05;
    double latencyOnlineLearningRateScale = 1.0;
    double latencyCurrentSampleWeight = 4.0;
    double latencyErrorClipMs = 5000.0;
    std::vector<double> latencyParameterLearningRateScales;
    double latencyNonlinearScale = 1500.0;
    double queueLearningRate = 0.002;
    bool latencyCheckGradient = false;
    size_t latencyColdStartFitEpochs = 100;
    size_t latencyHistoryLimit = 2000;
    size_t latencyHistoryBucketCapacity = 64;
    size_t latencyPersistInterval = 256;
    int64_t latencyPersistIntervalMs = 10 * 1000;
    size_t latencyWarmupSampleCount = 10;
    int64_t latencyModelTrustMinRevision = 256;
    double latencyZeroAnchorMaxMs = 256.0;
    size_t latencyZeroAnchorMinSamples = 8;
    bool latencyValidationEnabled = false;
    double residualLearningRate = 0.0;
    double residualUpdateErrorFloorMs = 100.0;
    double residualUpdateErrorRatio = 0.1;
    int64_t residualIdleDecayStartMs = 2 * 60 * 1000;
    int64_t residualIdleDecayEndMs = 5 * 60 * 1000;
    std::string latencyModelStateFile;
    bool enableSnapshot = true;
    std::string snapshotDir = "navi_sched";
    int64_t snapshotIntervalMs = 2 * 1000;
    size_t snapshotKeepCount = 200;
    bool enableVis = false;
};

using NaviSchedLocalNode = navi::NaviLocalNode;
using NaviSchedCm2Subscription = navi::NaviCm2Subscription;

struct NaviSchedRuntimeConfig : navi::NaviRuntimeConfig {
    std::string bizName;
};

struct NaviSchedClientConfig {
    NaviSchedConfig service;
    NaviSchedRuntimeConfig runtime;
    std::string masterBizName;
    bool enableSampleCollection = true;
    bool enableAgentForward = false;
    bool expectAgentResponseStream = true;
};

struct NaviSchedMasterConfig {
    NaviSchedConfig service;
    NaviSchedRuntimeConfig runtime;
    size_t collectorThreadNum = 0;
    size_t collectorQueueCount = 0;
    size_t maxWaitScheduleCount = 30;
    int64_t maxWaitScheduleTimeMs = 10;
    int64_t queueOverloadStartCycleMultiplier = 0;
    int64_t queueOverloadFullCycleMultiplier = 0;
    std::string schedulerName = "cost";
    std::string schedulerConfig;
    double timeCompression = 1.0;
    std::string latencyModelPoolStateFile;
    int64_t latencyModelPoolSourceMinRevision = 1024;
    size_t latencyModelPoolMaxCandidates = 4;
    int64_t latencyModelPoolPersistIntervalMs = 10000;
};

struct NaviSchedAgentConfig {
    NaviSchedConfig service;
    NaviSchedRuntimeConfig runtime;
    bool pdSeparated = true;
    bool unifiedLatencyPrediction = false;
    bool mambaReplayStopShift = false;
    int64_t mambaRetentionInterval = -1;
    bool mambaAlign = false;
    int64_t fullAttentionQueryDropTokens = 0;
    double timeCompression = 1.0;
    std::string latencyModelFingerprint;
    std::string latencyHistoryFingerprint;
    std::string latencyHistoryBinaryFingerprint;
    bool executionGeometryEnabled = false;
    bool projectedExecutionGeometryEnabled = false;
    bool executionCacheSourceEnabled = false;
    bool peerCacheEnabled = false;
    TPUPrefillExecutionPolicyConfig prefillExecutionPolicy;
};

}
