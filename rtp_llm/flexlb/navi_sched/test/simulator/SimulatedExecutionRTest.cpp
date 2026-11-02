#include "aios/navi_sched/test/simulator/SimulatedExecutionR.h"

#include "aios/navi_sched/data/AgentResponseToClientData.h"
#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/resource/AgentLatencyInputR.h"
#include "aios/navi_sched/resource/AgentRequestR.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/service/NaviSchedAgent.h"
#include "aios/navi_sched/test/simulator/SimulatedAgentAccess.h"
#include "aios/navi_sched/test/simulator/TrafficReplayR.h"
#include "aios/navi_sched/test/simulator/TrafficReplayStartR.h"
#include "navi/tester/NaviResourceHelper.h"
#include "unittest/unittest.h"
#include <cmath>
#include <cstdlib>
#include <fstream>

namespace navi_sched::test {

class SimulatedExecutionRTest : public TESTBASE {
protected:
    struct DirectExecutionContext {
        navi::NaviResourceHelper helper;
        AgentBatchRPtr batchR;
        AgentLatencyInputRPtr latencyInputR;
        AgentRequestRPtr requestR;
        SimulatedExecutionBridgeRPtr bridgeR;
        std::shared_ptr<SimulatedExecutionR> executionR;
    };

    static NaviSchedBlockHashesConstPtr makeBlockHashes(
        std::initializer_list<uint64_t> hashes)
    {
        return std::make_shared<const NaviSchedBlockHashes>(hashes);
    }

    static void setCacheConfig(SimulatedExecutionR &resource, int64_t blockCapacity)
    {
        resource._config.tokenCacheBytes = 1;
        resource._config.cacheCapacityBytes = blockCapacity * BlockHashUtil::BLOCK_SIZE;
    }

    static NaviSchedCacheUpdate updateCache(SimulatedExecutionR &resource,
                                            const NaviSchedBlockHashesConstPtr &hashes)
    {
        return resource.updateCache(hashes);
    }

    static SimulatedRequest readCache(SimulatedExecutionR &resource,
                                      const NaviSchedBlockHashesConstPtr &hashes)
    {
        SimulatedRequest request;
        request.blockHashes = hashes;
        resource.readCache(request);
        return request;
    }

    static std::unique_ptr<DirectExecutionContext> makeDirectExecution()
    {
        auto context = std::make_unique<DirectExecutionContext>();
        context->batchR = std::make_shared<AgentBatchR>(10, 1000);
        context->latencyInputR = std::make_shared<AgentLatencyInputR>();
        context->requestR = std::make_shared<AgentRequestR>();
        context->bridgeR = std::make_shared<SimulatedExecutionBridgeR>();
        auto reportBridge = std::make_shared<AgentReportBridgeR>();
        if (!context->helper.addExternalRes(context->batchR) ||
            !context->helper.addExternalRes(context->latencyInputR) ||
            !context->helper.addExternalRes(context->requestR) ||
            !context->helper.addExternalRes(context->bridgeR) ||
            !context->helper.addExternalRes(reportBridge) ||
            !context->helper.config(AgentReportR::RESOURCE_ID, "{}")) {
            return nullptr;
        }
        context->helper.kernelConfig(
            R"({"block_size":256,"noise_stddev_ratio":0.0,"time_compression":1.0})");
        context->executionR = context->helper.getOrCreateResPtr<SimulatedExecutionR>();
        return context->executionR ? std::move(context) : nullptr;
    }

    static RequestLatencyTracker::Snapshot inflight(const SimulatedExecutionR &resource)
    {
        return resource._inflightState.snapshot();
    }

    static size_t waitingRequestCount(const SimulatedExecutionR &resource)
    {
        autil::ScopedLock lock(resource._stateLock);
        return resource._waitingRequestCount;
    }

    static size_t pendingRequestCount(const SimulatedExecutionR &resource)
    {
        autil::ScopedLock lock(resource._stateLock);
        return resource._batchRequests.size();
    }

    static int32_t runningRequestCount(const SimulatedExecutionR &resource)
    {
        return resource._runningRequestCount.load(std::memory_order_acquire);
    }

    static void setRequestPipe(SimulatedExecutionR &resource,
                               const navi::AsyncPipePtr &pipe)
    {
        autil::ScopedLock lock(resource._stateLock);
        resource._requestPipe = pipe;
    }

    static void handleAbortedBatch(SimulatedExecutionR &resource,
                                   const AgentBatchR::ExecutionState &state,
                                   const char *phase)
    {
        resource.handleAbortedBatch(state, phase);
    }
};

class CompletionPipe : public navi::AsyncPipe {
public:
    CompletionPipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}

public:
    navi::ErrorCode setDataOverride(const navi::DataPtr &data, bool,
                                    bool ignoreData = false) override
    {
        if (!ignoreData) {
            auto value = std::dynamic_pointer_cast<AgentResponseToClientData>(data);
            responses.emplace_back(value);
            if (value && AgentResponseKind::ACK == value->kind) {
                response = value;
            }
            if (value && value->terminal) {
                terminalResponse = value;
            }
        }
        return navi::EC_NONE;
    }
    navi::ErrorCode setEof() override
    {
        eof = true;
        return navi::EC_NONE;
    }

public:
    std::shared_ptr<AgentResponseToClientData> response;
    std::shared_ptr<AgentResponseToClientData> terminalResponse;
    std::vector<std::shared_ptr<AgentResponseToClientData>> responses;
    bool eof = false;
};

class StartResponsePipe : public navi::AsyncPipe {
public:
    StartResponsePipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}

public:
    navi::ErrorCode setDataOverride(const navi::DataPtr &data, bool,
                                    bool ignoreData = false) override
    {
        if (!ignoreData) {
            response = std::dynamic_pointer_cast<TrafficReplayStartResponseData>(data);
        }
        return navi::EC_NONE;
    }

public:
    std::shared_ptr<TrafficReplayStartResponseData> response;
};

class RejectedRequestPipe : public navi::AsyncPipe {
public:
    RejectedRequestPipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}

public:
    navi::ErrorCode setDataOverride(const navi::DataPtr &, bool, bool = false) override
    {
        return navi::EC_ABORT;
    }
};

TEST_F(SimulatedExecutionRTest, testDefaultFormulaPreservesPhysicalWorkload)
{
    const auto &parameters = SimulatedExecutionR::getFormulaParameters();
    ASSERT_EQ(size_t(10), parameters.size());
    ASSERT_TRUE(NonLinearPrefillModelR::validateParameters(parameters));
    for (const auto count : {size_t(11), size_t(18)}) {
        EXPECT_FALSE(NonLinearPrefillModelR::validateParameters(std::vector<double>(count, 1.0)));
    }
    for (const auto hit : {int64_t(0), int64_t(2048)}) {
        for (const auto miss : {int64_t(2048), int64_t(16384)}) {
            NaviSchedLatencyFeatures features;
            features.appendBatchCount(1);
            features.appendRequest(hit, miss);
            const double h = hit / 1024.0;
            const double m = miss / 1024.0;
            const double u = (-964.0 + 10.0 + 1.4 * h + 20.0 * m +
                              0.1 * m * m + 0.09 * h * m) / 320.0;
            const double expected = 280.0 + 50.0 * (u + std::sqrt(u * u + 4.0));
            EXPECT_NEAR(expected, NonLinearPrefillModelR::predict(features, parameters), 1e-12);
        }
    }
}

TEST_F(SimulatedExecutionRTest, testTrafficReplayStartReconnectReplacesPipe)
{
    TrafficReplayStartR resource;
    TrafficReplayStartRequestData request;
    request.clientCount = 2;
    request.startDelayMs = 1000;
    request.firstRequestTimeMs = 100;
    auto oldPipe = std::make_shared<StartResponsePipe>();
    ASSERT_TRUE(resource.join(request, oldPipe));

    request.firstRequestTimeMs = 90;
    auto replacementPipe = std::make_shared<StartResponsePipe>();
    ASSERT_TRUE(resource.join(request, replacementPipe));

    request.clientIndex = 1;
    request.firstRequestTimeMs = 110;
    auto secondClientPipe = std::make_shared<StartResponsePipe>();
    ASSERT_TRUE(resource.join(request, secondClientPipe));

    EXPECT_FALSE(oldPipe->response);
    ASSERT_TRUE(replacementPipe->response);
    ASSERT_TRUE(secondClientPipe->response);
    EXPECT_EQ(90, replacementPipe->response->firstRequestTimeMs);
    EXPECT_EQ(replacementPipe->response->startTimeMs,
              secondClientPipe->response->startTimeMs);
}

TEST_F(SimulatedExecutionRTest, testTrafficReplayStartUsesIndependentGroups)
{
    TrafficReplayStartR resource;
    TrafficReplayStartRequestData request;
    request.replayGroup = "first";
    request.clientCount = 2;
    request.startDelayMs = 1000;
    request.firstRequestTimeMs = 100;
    auto firstPipe = std::make_shared<StartResponsePipe>();
    ASSERT_TRUE(resource.join(request, firstPipe));
    EXPECT_FALSE(firstPipe->response);

    request.replayGroup = "second";
    request.clientCount = 1;
    auto secondGroupPipe = std::make_shared<StartResponsePipe>();
    ASSERT_TRUE(resource.join(request, secondGroupPipe));
    EXPECT_TRUE(secondGroupPipe->response);
    EXPECT_FALSE(firstPipe->response);

    request.replayGroup = "first";
    request.clientCount = 2;
    request.clientIndex = 1;
    auto firstGroupSecondPipe = std::make_shared<StartResponsePipe>();
    ASSERT_TRUE(resource.join(request, firstGroupSecondPipe));
    EXPECT_TRUE(firstPipe->response);
    EXPECT_TRUE(firstGroupSecondPipe->response);
}

TEST_F(SimulatedExecutionRTest, testCacheMatchesOnlyContinuousPrefix)
{
    SimulatedExecutionR resource;
    setCacheConfig(resource, 4);
    updateCache(resource, makeBlockHashes({11, 12}));

    auto prefixHit = readCache(resource, makeBlockHashes({11, 12, 13}));
    EXPECT_EQ(2, prefixHit.cacheHitBlocks);
    EXPECT_EQ(1, prefixHit.cacheMissBlocks);

    auto firstBlockMiss = readCache(resource, makeBlockHashes({99, 12}));
    EXPECT_EQ(0, firstBlockMiss.cacheHitBlocks);
    EXPECT_EQ(2, firstBlockMiss.cacheMissBlocks);
}

TEST_F(SimulatedExecutionRTest, testCacheEvictsLeastRecentlyUsedBlock)
{
    SimulatedExecutionR resource;
    setCacheConfig(resource, 2);
    updateCache(resource, makeBlockHashes({11, 12}));
    ASSERT_EQ(2, resource.getCachedBlockCount());

    auto hit = readCache(resource, makeBlockHashes({11}));
    ASSERT_EQ(1, hit.cacheHitBlocks);
    auto update = updateCache(resource, makeBlockHashes({13}));
    EXPECT_EQ(NaviSchedBlockHashes({13}), BlockHashUtil::get(update.addedBlockHashes));
    EXPECT_EQ(NaviSchedBlockHashes({12}), BlockHashUtil::get(update.removedBlockHashes));
    EXPECT_EQ(1, resource.getEvictedBlockCount());
    EXPECT_EQ(1, readCache(resource, makeBlockHashes({11})).cacheHitBlocks);
    EXPECT_EQ(0, readCache(resource, makeBlockHashes({12})).cacheHitBlocks);
}

TEST_F(SimulatedExecutionRTest,
       testAdmissionRejectAndDuplicateLeaveOnlyOriginalSimulatorOwner)
{
    auto context = makeDirectExecution();
    ASSERT_TRUE(context);
    auto &execution = *context->executionR;

    SimulatedRequest first;
    first.requestId = 101;
    first.prefillTokenCount = 256;
    first.batchId = 77;
    first.expectedBatchSize = 2;
    first.predictedQueueLatencyMs = 10.0;
    first.predictedExecutionLatencyMs = 20.0;
    ASSERT_TRUE(execution.submit(first));
    const auto original = inflight(execution);
    ASSERT_EQ(1, original.requestCount);
    ASSERT_DOUBLE_EQ(30.0, original.totalPredictedLatencyMs);
    ASSERT_EQ(1, waitingRequestCount(execution));

    EXPECT_FALSE(execution.submit(first));
    auto afterDuplicate = inflight(execution);
    EXPECT_EQ(original.revision, afterDuplicate.revision);
    EXPECT_EQ(1, afterDuplicate.requestCount);
    EXPECT_DOUBLE_EQ(30.0, afterDuplicate.totalPredictedLatencyMs);

    auto rejected = first;
    rejected.requestId = 102;
    rejected.expectedBatchSize = 3;
    EXPECT_FALSE(execution.submit(rejected));
    const auto afterRejected = inflight(execution);
    EXPECT_EQ(original.revision + 2, afterRejected.revision);
    EXPECT_EQ(1, afterRejected.requestCount);
    EXPECT_DOUBLE_EQ(30.0, afterRejected.totalPredictedLatencyMs);
    EXPECT_EQ(original.missingRemoveCount, afterRejected.missingRemoveCount);
    EXPECT_EQ(1, waitingRequestCount(execution));
    EXPECT_EQ(1, pendingRequestCount(execution));

    AgentBatchR::ExecutionState aborted;
    ASSERT_TRUE(context->batchR->abortPendingBatch(77, aborted));
    handleAbortedBatch(execution, aborted, "test_cleanup");
    const auto cleaned = inflight(execution);
    EXPECT_EQ(0, cleaned.requestCount);
    EXPECT_DOUBLE_EQ(0.0, cleaned.totalPredictedLatencyMs);
    EXPECT_EQ(0, waitingRequestCount(execution));
    EXPECT_EQ(0, pendingRequestCount(execution));
}

TEST_F(SimulatedExecutionRTest,
       testEnqueueFailureClearsSimulatorBatchAndInflightOwnership)
{
    auto context = makeDirectExecution();
    ASSERT_TRUE(context);
    auto &execution = *context->executionR;
    setRequestPipe(execution, std::make_shared<RejectedRequestPipe>());

    SimulatedRequest request;
    request.requestId = 101;
    request.prefillTokenCount = 256;
    request.completionPipe = std::make_shared<CompletionPipe>();
    request.predictedQueueLatencyMs = 10.0;
    request.predictedExecutionLatencyMs = 20.0;
    ASSERT_TRUE(execution.submit(request));

    const auto cleaned = inflight(execution);
    EXPECT_EQ(0, cleaned.requestCount);
    EXPECT_DOUBLE_EQ(0.0, cleaned.totalPredictedLatencyMs);
    EXPECT_EQ(0, cleaned.missingRemoveCount);
    EXPECT_EQ(0, waitingRequestCount(execution));
    EXPECT_EQ(0, pendingRequestCount(execution));
    std::vector<SimulatedRequest> submitted;
    ASSERT_FALSE(execution.takeSubmitted(submitted));
    EXPECT_TRUE(submitted.empty());
    AgentBatchR::ExecutionState state;
    ASSERT_TRUE(context->batchR->getExecutionState(state));
    EXPECT_EQ(0, state.batchWaitingRequests);
    EXPECT_EQ(0, state.queuedBatchRequests);
    EXPECT_EQ(0, state.runningRequests);
}

TEST_F(SimulatedExecutionRTest,
       testReadyInputFailureClearsSimulatorBatchAndInflightOwnership)
{
    auto context = makeDirectExecution();
    ASSERT_TRUE(context);
    auto &execution = *context->executionR;
    ASSERT_TRUE(context->latencyInputR->finish());

    SimulatedRequest request;
    request.requestId = 101;
    request.prefillTokenCount = 256;
    request.completionPipe = std::make_shared<CompletionPipe>();
    request.predictedQueueLatencyMs = 10.0;
    request.predictedExecutionLatencyMs = 20.0;
    EXPECT_FALSE(execution.submit(request));

    const auto cleaned = inflight(execution);
    EXPECT_EQ(0, cleaned.requestCount);
    EXPECT_DOUBLE_EQ(0.0, cleaned.totalPredictedLatencyMs);
    EXPECT_EQ(0, cleaned.missingRemoveCount);
    EXPECT_EQ(0, waitingRequestCount(execution));
    EXPECT_EQ(0, pendingRequestCount(execution));
    AgentBatchR::ExecutionState state;
    ASSERT_TRUE(context->batchR->getExecutionState(state));
    EXPECT_EQ(0, state.batchWaitingRequests);
    EXPECT_EQ(0, state.queuedBatchRequests);
    EXPECT_EQ(0, state.runningRequests);
    EXPECT_EQ(1, context->batchR->getAbnormalBatchFinishCount());
}

TEST_F(SimulatedExecutionRTest, testStartFailureAbortsSimulatorBatchWithoutResidualCounts)
{
    auto context = makeDirectExecution();
    ASSERT_TRUE(context);
    auto &execution = *context->executionR;

    SimulatedRequest request;
    request.requestId = 101;
    request.prefillTokenCount = 256;
    request.predictedQueueLatencyMs = 10.0;
    request.predictedExecutionLatencyMs = 20.0;
    ASSERT_TRUE(execution.submit(request));
    std::vector<SimulatedRequest> submitted;
    ASSERT_FALSE(execution.takeSubmitted(submitted));
    ASSERT_EQ(1, submitted.size());
    submitted.front().requestId = 999;
    EXPECT_FALSE(execution.startBatch(submitted));

    const auto cleaned = inflight(execution);
    EXPECT_EQ(0, cleaned.requestCount);
    EXPECT_DOUBLE_EQ(0.0, cleaned.totalPredictedLatencyMs);
    EXPECT_EQ(0, cleaned.missingRemoveCount);
    EXPECT_EQ(0, waitingRequestCount(execution));
    EXPECT_EQ(0, runningRequestCount(execution));
    AgentBatchR::ExecutionState state;
    ASSERT_TRUE(context->batchR->getExecutionState(state));
    EXPECT_EQ(0, state.batchWaitingRequests);
    EXPECT_EQ(0, state.queuedBatchRequests);
    EXPECT_EQ(0, state.runningRequests);
}

TEST_F(SimulatedExecutionRTest,
       testMalformedFinishUsesAuthoritativeSimulatorCountAndCleansOwner)
{
    auto context = makeDirectExecution();
    ASSERT_TRUE(context);
    auto &execution = *context->executionR;

    for (uint64_t requestId : {101, 102}) {
        SimulatedRequest request;
        request.requestId = requestId;
        request.prefillTokenCount = 256;
        request.batchId = 77;
        request.expectedBatchSize = 2;
        request.predictedQueueLatencyMs = 10.0;
        request.predictedExecutionLatencyMs = 20.0;
        ASSERT_TRUE(execution.submit(request));
    }
    std::vector<SimulatedRequest> submitted;
    ASSERT_FALSE(execution.takeSubmitted(submitted));
    ASSERT_EQ(2, submitted.size());
    ASSERT_TRUE(execution.startBatch(submitted));
    ASSERT_EQ(2, runningRequestCount(execution));
    ASSERT_EQ(2, inflight(execution).requestCount);

    submitted.pop_back();
    execution.finishBatch(submitted, 1.0);

    const auto cleaned = inflight(execution);
    EXPECT_EQ(0, cleaned.requestCount);
    EXPECT_DOUBLE_EQ(0.0, cleaned.totalPredictedLatencyMs);
    EXPECT_EQ(0, cleaned.missingRemoveCount);
    EXPECT_EQ(0, waitingRequestCount(execution));
    EXPECT_EQ(0, runningRequestCount(execution));
    AgentBatchR::ExecutionState state;
    ASSERT_TRUE(context->batchR->getExecutionState(state));
    EXPECT_EQ(0, state.batchWaitingRequests);
    EXPECT_EQ(0, state.queuedBatchRequests);
    EXPECT_EQ(0, state.runningRequests);
    const auto latencyStats = context->latencyInputR->getStats();
    EXPECT_EQ(2, latencyStats.submittedAbortedRequestCount);
}

TEST_F(SimulatedExecutionRTest, testAgentStopDrainsSimulatedExecution)
{
    NaviSchedConfig config;
    config.name = "simulate_stop_drain_test";
    config.gig.grpcPort = 0;
    config.enableKmon = false;
    config.enableSnapshot = false;
    config.waitNoQuery = false;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "simulate_agent_biz";
    agentConfig.runtime.threadNum = 4;
    agentConfig.service = config;
    agentConfig.timeCompression = 100.0;
    NaviSchedAgent agent(agentConfig);
    SimulatedExecutionConfig simulateConfig;
    simulateConfig.blockSize = 1024;
    simulateConfig.formulaParameters = SimulatedExecutionR::getFormulaParameters();
    simulateConfig.noiseStddevRatio = 0.0;
    simulateConfig.timeCompression = 100.0;
    const auto *testTmpDir = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(testTmpDir);
    simulateConfig.queueTraceOutputPath =
        std::string(testTmpDir) + "/simulated_queue_trace.csv";
    auto simulateHandle = SimulatedAgentAccess::create(agent, simulateConfig);
    ASSERT_TRUE(simulateHandle);

    std::vector<std::shared_ptr<CompletionPipe>> completionPipes;
    for (uint64_t requestId = 1; requestId <= 3; ++requestId) {
        NaviSchedForwardRequest request;
        request.request.requestId = requestId;
        request.request.promptTokenCount = 4096;
        request.request.blockHashes =
            makeBlockHashes({requestId * 10 + 1, requestId * 10 + 2});
        request.decision.requestId = requestId;
        request.decision.batchId = requestId;
        request.decision.expectedBatchSize = 1;
        auto completionPipe = std::make_shared<CompletionPipe>();
        ASSERT_TRUE(simulateHandle->submit(request, completionPipe));
        completionPipes.emplace_back(std::move(completionPipe));
    }

    simulateHandle->stop();
    agent.stop();

    NaviSchedLatencyFeatures expectedFeatures;
    expectedFeatures.appendRequest(0, 4096);
    auto expectedLatencyMs = NonLinearPrefillModelR::predict(
        expectedFeatures, simulateConfig.formulaParameters);
    for (const auto &completionPipe : completionPipes) {
        ASSERT_TRUE(completionPipe->response);
        ASSERT_TRUE(completionPipe->terminalResponse);
        ASSERT_EQ(2, completionPipe->responses.size());
        EXPECT_EQ(completionPipe->response->requestId,
                  completionPipe->terminalResponse->requestId);
        EXPECT_EQ(AgentResponseKind::ACK, completionPipe->response->kind);
        EXPECT_EQ(AgentResponseKind::OUTPUT, completionPipe->terminalResponse->kind);
        EXPECT_TRUE(completionPipe->terminalResponse->terminal);
        EXPECT_NEAR(expectedLatencyMs,
                    completionPipe->response->requestTiming.executionLatencyMs, 1e-9);
    }
    std::ifstream queueTrace(simulateConfig.queueTraceOutputPath);
    ASSERT_TRUE(queueTrace.good());
    std::string line;
    size_t lineCount = 0;
    while (std::getline(queueTrace, line)) {
        ++lineCount;
    }
    EXPECT_EQ(4, lineCount);
}

TEST_F(SimulatedExecutionRTest, testAgentStopDrainsIncompleteSharedBatch)
{
    NaviSchedConfig config;
    config.name = "simulate_stop_incomplete_batch_test";
    config.gig.grpcPort = 0;
    config.enableKmon = false;
    config.enableSnapshot = false;
    config.enableVis = false;
    config.waitNoQuery = false;
    config.batchTimeoutMs = 10;
    NaviSchedAgentConfig agentConfig;
    agentConfig.runtime.bizName = "simulate_agent_biz";
    agentConfig.runtime.threadNum = 4;
    agentConfig.service = config;
    agentConfig.timeCompression = 10.0;
    NaviSchedAgent agent(agentConfig);
    SimulatedExecutionConfig simulateConfig;
    simulateConfig.noiseStddevRatio = 0.0;
    simulateConfig.timeCompression = 10.0;
    auto simulateHandle = SimulatedAgentAccess::create(agent, simulateConfig);
    ASSERT_TRUE(simulateHandle);

    NaviSchedForwardRequest request;
    request.request.requestId = 1;
    request.request.promptTokenCount = 4096;
    request.request.blockHashes = makeBlockHashes({11, 12});
    request.decision.requestId = request.request.requestId;
    request.decision.batchId = 100;
    request.decision.expectedBatchSize = 3;
    auto completionPipe = std::make_shared<CompletionPipe>();
    ASSERT_TRUE(simulateHandle->submit(request, completionPipe));

    simulateHandle->stop();
    agent.stop();

    ASSERT_TRUE(completionPipe->response);
    ASSERT_TRUE(completionPipe->terminalResponse);
    ASSERT_EQ(2, completionPipe->responses.size());
    EXPECT_EQ(1, completionPipe->response->requestId);
    EXPECT_EQ(1, completionPipe->terminalResponse->requestId);
    EXPECT_TRUE(completionPipe->terminalResponse->terminal);
    EXPECT_GT(completionPipe->response->requestTiming.executionLatencyMs, 0.0);
}

}
