#include "aios/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "autil/DataBuffer.h"
#include "aios/navi_sched/resource/MasterKVReadyWorkload.h"
#include "aios/navi_sched/config/NaviSchedConfigBuilder.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/resource/AgentReportR.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/resource/MasterLatencyModelR.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/scheduler/ScheduleContextInternal.h"
#include "navi/engine/Navi.h"
#include "navi/engine/ResourceMap.h"
#include "autil/TimeUtility.h"
#include "unittest/unittest.h"
#include <numeric>

namespace navi_sched {
AgentUpdateToMasterDataPtr makeControlUpdate(NaviSchedAgentEvent::Type event, int64_t eventTimeUs = 0) {
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = event;
    update->eventTimeUs = eventTimeUs;
    update->control = std::make_shared<AgentControlToMasterData>();
    if (NaviSchedAgentEvent::AGENT_HANDSHAKE == event) {
        update->control->agentStartTimeUs = eventTimeUs > 0 ? eventTimeUs : 1;
        update->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    }
    return update;
}

AgentUpdateToMasterDataPtr makeCachePolicyUpdate(int64_t blockSize) {
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = NaviSchedAgentEvent::CACHE_UPDATED;
    update->cache = std::make_shared<AgentCacheToMasterData>();
    auto &cache = *update->cache;
    cache.fullSnapshot = true; cache.revision = 1;
    cache.cachedBlockCount = 0; cache.hashChecksum = 0;
    cache.fullReadyBlockCount = 0; cache.fullReadyHashChecksum = 0;
    cache.tailReadyBlockCount = 0; cache.tailReadyHashChecksum = 0;
    cache.blockSize = blockSize; cache.linearStep = 1; cache.hybridTailRequired = 0;
    return update;
}

AgentUpdateToMasterDataPtr makeEmptyPendingCacheSnapshot() {
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = NaviSchedAgentEvent::PENDING_CACHE_UPDATED;
    update->pendingCache = std::make_shared<AgentPendingCacheToMasterData>();
    update->pendingCache->fullSnapshot = true;
    return update;
}

NaviSchedLatencyOptimizerState makeLatencyOptimizerState(size_t parameterCount, int64_t step = 1) {
    NaviSchedLatencyOptimizerState optimizer;
    optimizer.adamMoment1.assign(parameterCount, 0.0);
    optimizer.adamMoment2.assign(parameterCount, 0.0);
    optimizer.step = step;
    return optimizer;
}

class AgentReportTestContext {
public:
    ~AgentReportTestContext() {
        if (_navi) {
            _navi->stop();
        }
    }

    bool init(int32_t serverPort = 0,
              const std::string& serverIp = "",
              int64_t totalBlocks = 0,
              bool latencyValidationEnabled = false,
              bool pdSeparated = true,
              bool workloadPublishTimingEnabled = false,
              bool unifiedLatencyPrediction = false,
              const std::string& latencyModelFingerprint = "",
              bool pdUnifiedCapacitySplit = false) {
        NaviSchedRuntimeConfig runtimeConfig;
        runtimeConfig.bizName = "agent_report_test_biz";
        runtimeConfig.threadNum = 1;
        NaviSchedConfigBuilder::NamedConfigs resourceConfigs;
        autil::legacy::json::JsonMap cacheConfig;
        cacheConfig["total_blocks"] = totalBlocks;
        resourceConfigs[AgentCacheR::RESOURCE_ID] = autil::legacy::ToJsonString(cacheConfig, true);
        autil::legacy::json::JsonMap reportConfig;
        reportConfig["server_ip"] = serverIp;
        reportConfig["server_port"] = serverPort;
        reportConfig["latency_validation_enabled"] = latencyValidationEnabled;
        reportConfig["pd_separated"] = pdSeparated;
        reportConfig["unified_latency_prediction"] = unifiedLatencyPrediction;
        reportConfig["pd_unified_capacity_split"] = pdUnifiedCapacitySplit;
        reportConfig["workload_publish_timing_enabled"] = workloadPublishTimingEnabled;
        reportConfig["latency_model_fingerprint"] = latencyModelFingerprint;
        resourceConfigs[AgentReportR::RESOURCE_ID] = autil::legacy::ToJsonString(reportConfig, true);
        auto config =
            NaviSchedConfigBuilder::build(runtimeConfig, {}, false, {}, resourceConfigs, {AgentReportR::RESOURCE_ID});
        _bridge = std::make_shared<AgentReportBridgeR>();
        navi::ResourceMap resourceMap;
        if (config.empty() || !resourceMap.addResource(_bridge)) {
            return false;
        }
        _navi = std::make_shared<navi::Navi>("agent_report_test");
        return _navi->init("", nullptr) && _navi->update(config, resourceMap) && bool(_bridge->getAgentReportR());
    }

    AgentReportRPtr get() const {
        return _bridge ? _bridge->getAgentReportR() : nullptr;
    }

private:
    navi::NaviPtr _navi;
    AgentReportBridgeRPtr _bridge;
};

class CaptureAsyncPipe: public navi::AsyncPipe {
public:
    CaptureAsyncPipe(): navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}

public:
    navi::ErrorCode setData(const navi::DataPtr& data) override {
        ++messageCount;
        const auto update = std::dynamic_pointer_cast<AgentUpdateToMasterData>(data);
        if (update && !update->updates.empty()) {
            datas.insert(datas.end(), update->updates.begin(), update->updates.end());
        } else {
            datas.emplace_back(data);
        }
        return navi::EC_NONE;
    }
    navi::ErrorCode setDataOverride(const navi::DataPtr& data, bool overrideData, bool ignoreData = false) override {
        ++overrideCount;
        if (overrideData) {
            datas.clear();
        }
        if (!ignoreData) {
            datas.emplace_back(data);
        }
        return navi::EC_NONE;
    }
    navi::ErrorCode setEof() override {
        eof = true;
        ++eofCount;
        return navi::EC_NONE;
    }

public:
    // Keep logical events separate from actual transport messages.
    size_t messageCount = 0;
    std::vector<navi::DataPtr> datas;
    bool eof = false;
    size_t eofCount = 0;
    size_t overrideCount = 0;
};

class TestMasterStateR: public MasterStateR {
public:
    TestMasterStateR() {
        _pendingCacheR__ = &_pendingCacheRForTest;
        _masterLatencyModelR__ = &_masterLatencyModelRForTest;
    }

    using MasterStateR::reportAgent;

    bool reportAgent(const std::string& agentId, const AgentUpdateToMasterDataPtr& update) {
        completeLatencyModel(update);
        return MasterStateR::reportAgent(agentId, update);
    }

    bool reportAgent(const NodeStatePtr& state, const AgentUpdateToMasterDataPtr& update) {
        completeLatencyModel(update);
        return MasterStateR::reportAgent(state, update);
    }

    MasterLatencyModelR& latencyModelPool() {
        return _masterLatencyModelRForTest;
    }

private:
    static void completeLatencyModel(const AgentUpdateToMasterDataPtr& update) {
        if (!update || !update->latency || update->latency->modelRevision <= 0) {
            return;
        }
        auto& latency = *update->latency;
        if (!latency.optimizerState.valid(latency.parameters.size())) {
            latency.optimizerState = makeLatencyOptimizerState(latency.parameters.size());
        }
        if (NonLinearPrefillModelR::validateParameters(latency.parameters)) {
            latency.zeroPoint = NonLinearPrefillModelR::calculateZeroPoint(latency.parameters);
        }
    }

private:
    MasterPendingCacheR _pendingCacheRForTest;
    MasterLatencyModelR _masterLatencyModelRForTest;
};

NaviSchedWorkloadSnapshot executionLoadTestWorkload() {
    NaviSchedWorkloadSnapshot w;
    w.revision = 1; w.eventTimeUs = 100;
    w.batchCount = 1; w.requestCount = 2;
    w.runningRequestCount = 0; w.waitingRequestCount = 2;
    w.prefillCapacitySplit = true; w.maxRunningRequestCount = 64;
    w.runningCapacitySlotCount = 2;
    w.remainingPrefillRequestCount = w.runningCapacityPrefillRequestCount = 2;
    w.remainingPrefillContextTokenCount = w.runningCapacityPrefillContextTokenCount = 30;
    w.remainingPrefillTokenCount = w.runningCapacityPrefillTokenCount = 300;
    w.remainingPrefillTokenSquareSum = w.runningCapacityPrefillTokenSquareSum = 50000;
    w.contextRemainingPrefillTokenProductSum =
        w.runningCapacityContextRemainingPrefillTokenProductSum = 5000;
    w.remainingTotalContextTokenCount = 30; w.remainingTotalTokenCount = 300;
    w.remainingTotalTokenSquareSum = 50000; w.contextRemainingTotalTokenProductSum = 5000;
    w.totalPromptTokenCount = 330;
    w.executionLoadTracked = true;
    w.executionLoadRequests = {{1, 10, 5, 10, 100, true, NativeKVExecutionState::PREPARING},
                              {2, 10, 0, 20, 200, true, NativeKVExecutionState::READY}};
    return w;
}


TEST(ExecutionLoadResourceTest, testExecutionLoadMasterConsumesReadyViewAndDoesNotReaddPreparingReservations) {
    TestMasterStateR runtime;
    auto state = runtime.registerAgent("agent_0", "model_biz", "127.0.0.1:19000", 100);
    ASSERT_TRUE(state);
    auto handshake = makeControlUpdate(NaviSchedAgentEvent::AGENT_HANDSHAKE);
    handshake->control->pdSeparated = true;
    handshake->control->unifiedLatencyPrediction = true;
    handshake->control->serverInfo.port = 18000;
    handshake->latency = std::make_shared<AgentLatencyToMasterData>();
    handshake->latency->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    handshake->latency->binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    handshake->latency->modelRevision = 1;
    NonLinearPrefillModelR parameterModel;
    parameterModel._useCapacitySplitFeature = true;
    handshake->latency->parameters = parameterModel.getInitialParameters();
    ASSERT_EQ(19u, handshake->latency->parameters.size());
    ASSERT_TRUE(NonLinearPrefillModelR::validateParameters(handshake->latency->parameters));
    handshake->workload = std::make_shared<NaviSchedWorkloadSnapshot>(executionLoadTestWorkload());
    ASSERT_TRUE(runtime.reportAgent("agent_0", handshake));
    ASSERT_TRUE(runtime.reportAgent("agent_0", makeCachePolicyUpdate(256)));
    ASSERT_TRUE(runtime.reportAgent("agent_0", makeEmptyPendingCacheSnapshot()));
    std::vector<MasterBatchReservation::Request> requests(2);
    for (size_t i = 0; i < requests.size(); ++i) {
        requests[i].requestId = 3+i; requests[i].blockSize = 10;
        requests[i].totalTokenCount = 110*(i+1);
        requests[i].predictedCacheHitTokenCount = 10*(i+1);
        requests[i].executionLoadIncluded = i==1;
    }
    ASSERT_TRUE(state->reserveBatch(11, autil::TimeUtility::currentTime(), 0, requests));
    NaviSchedNodeSnapshot snapshot;
    ASSERT_TRUE(state->getScheduleSnapshot("model_biz", snapshot));
    EXPECT_EQ(1, snapshot.reportedWorkloadFeatures.getPrecedingRequestCount());
    EXPECT_EQ(200, snapshot.reportedWorkloadFeatures.getCacheMissTokenCount());
    EXPECT_EQ(1, snapshot.reservedWorkloadFeatures.getBatchSize());
    const auto pending = snapshot.reservedWorkloadFeatures.getExpectedKVWorkload();
    ASSERT_NE(nullptr, pending);
    EXPECT_DOUBLE_EQ(1.0, pending->precedingRequests);
    EXPECT_DOUBLE_EQ(100.0, pending->waiting[2]);
    EXPECT_DOUBLE_EQ(500.0, snapshot.workloadFeatures.getPlanningCacheMissTokenCount());
    EXPECT_EQ(400, snapshot.workloadFeatures.getCacheMissTokenCount());
    EXPECT_EQ(2, snapshot.workload.requestCount);
    EXPECT_EQ(2, snapshot.workload.waitingRequestCount);
    auto ready = std::make_shared<AgentUpdateToMasterData>();
    ready->event = NaviSchedAgentEvent::WORKLOAD_UPDATED; ready->eventTimeUs = 200;
    ready->workload = std::make_shared<NaviSchedWorkloadSnapshot>(*handshake->workload);
    ready->workload->revision = 2;
    ready->workload->executionLoadRequests[0].state = NativeKVExecutionState::READY;
    ASSERT_TRUE(runtime.reportAgent("agent_0", ready));
    snapshot = {};
    ASSERT_TRUE(state->getScheduleSnapshot("model_biz", snapshot));
    EXPECT_EQ(2, snapshot.reportedWorkloadFeatures.getPrecedingRequestCount());
    EXPECT_EQ(500, snapshot.workloadFeatures.getCacheMissTokenCount());
    auto invalid = std::make_shared<AgentUpdateToMasterData>();
    invalid->event = ready->event; invalid->eventTimeUs = ready->eventTimeUs;
    invalid->workload = std::make_shared<NaviSchedWorkloadSnapshot>(*ready->workload);
    invalid->workload->revision = 3;
    ++invalid->workload->executionLoadRequests[0].computeTokens;
    EXPECT_FALSE(runtime.reportAgent("agent_0", invalid));
    snapshot = {};
    ASSERT_TRUE(state->getScheduleSnapshot("model_biz", snapshot));
    EXPECT_EQ(2u, snapshot.workload.revision);
    EXPECT_EQ(500, snapshot.workloadFeatures.getCacheMissTokenCount());
}

TEST(ExecutionLoadResourceTest, testExecutionLoadAgentPublishesReadinessOnlyChangesWithoutChangingRawWorkload) {
    AgentReportTestContext context;
    ASSERT_TRUE(context.init(19000, "10.0.0.1", 0, false, true, false, true, "", true));
    auto resource = context.get();
    auto pipe = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource->subscribe("execution-load", pipe));
    auto workload = executionLoadTestWorkload();
    const auto first = resource->reportWorkloadUpdated(workload, {});
    ASSERT_GT(first, 0u);
    const auto reportsBefore = pipe->datas.size();
    workload.executionLoadRequests[0].state = NativeKVExecutionState::READY;
    const auto second = resource->reportWorkloadUpdated(workload, {});
    EXPECT_GT(second, first);
    ASSERT_GT(pipe->datas.size(), reportsBefore);
    auto latest = resource->getWorkloadSnapshot();
    ASSERT_TRUE(latest); EXPECT_EQ(2, latest->requestCount);
    EXPECT_EQ(2, latest->waitingRequestCount);
    EXPECT_EQ(300, latest->remainingPrefillTokenCount);
    ASSERT_EQ(2u, latest->executionLoadRequests.size());
    EXPECT_EQ(NativeKVExecutionState::READY, latest->executionLoadRequests[0].state);
    // Same readiness again must not publish or advance the revision.
    const auto reportsAfter = pipe->datas.size();
    EXPECT_EQ(second, resource->reportWorkloadUpdated(workload, {}));
    EXPECT_EQ(reportsAfter, pipe->datas.size());
}

TEST(ExecutionLoadResourceTest, testKVLoadLatencyMetricUsesEngineCoreAggregatedCompletion) {
    AgentReportTestContext context;
    ASSERT_TRUE(context.init(19000, "10.0.0.1", 0, false, true, false, true, "", true));
    auto resource = context.get();
    auto load = resource->beginKVPreparation(17, 1000000000); ASSERT_TRUE(load);
    EXPECT_EQ(0u, resource->takeKVLoadLatencyWindowStats().readyCount);
    EXPECT_TRUE(resource->completeKVPreparation(load->key,true,1200000000).terminalChanged);
    EXPECT_FALSE(resource->completeKVPreparation(load->key,true,9900000000).terminalChanged);
    auto window = resource->takeKVLoadLatencyWindowStats();
    EXPECT_EQ(1u, window.readyCount); EXPECT_EQ(200000000.0L, window.totalPreparationNs);
    EXPECT_EQ(200000000, window.maxPreparationNs);
    EXPECT_EQ(0u, resource->takeKVLoadLatencyWindowStats().readyCount);
    EXPECT_TRUE(resource->retireKVPreparation(load->key));
    EXPECT_EQ(0u, resource->getActiveKVPreparationCount());
}
TEST(ExecutionLoadResourceTest, testKVLoadLatencyMetricDoesNotTreatFailedLoadAsSuccess) {
    AgentReportTestContext context;
    ASSERT_TRUE(context.init(19000, "10.0.0.1", 0, false, true, false, true, "", true));
    auto resource = context.get(); auto load = resource->beginKVPreparation(17, 100);
    ASSERT_TRUE(load); resource->completeKVPreparation(load->key,false,200);
    resource->completeKVPreparation(load->key,true,300);
    auto stats = resource->takeKVLoadLatencyWindowStats();
    EXPECT_EQ(0u,stats.readyCount); EXPECT_EQ(1u,stats.failedCount);
    EXPECT_EQ(0,stats.maxPreparationNs);
    EXPECT_TRUE(resource->retireKVPreparation(load->key));
}

TEST(ExecutionLoadResourceTest, testWorkloadElapsedRequestCountRoundTrip) {
    EXPECT_EQ(28, NAVI_SCHED_DATA_VERSION);
    EXPECT_EQ(43, NAVI_SCHED_AGENT_UPDATE_VERSION);
    EXPECT_EQ(44, NAVI_SCHED_AGENT_UPDATE_BATCH_VERSION);
    EXPECT_NE(NAVI_SCHED_AGENT_UPDATE_VERSION, NAVI_SCHED_AGENT_UPDATE_BATCH_VERSION);
    NaviSchedWorkloadSnapshot workload;
    workload.revision = 1;
    workload.eventTimeUs = 1'000'000;
    workload.batchCount = 1;
    workload.requestCount = 32;
    workload.remainingPrefillRequestCount = 32;
    workload.runningRequestCount = 1;
    workload.runningElapsedTimeMs = 125.0;
    workload.runningElapsedRequestCount = 1;
    workload.waitingRequestCount = 31;
    workload.prefillCapacitySplit = true;
    workload.maxRunningRequestCount = 64;
    workload.runningCapacitySlotCount = 32;
    workload.runningCapacityPrefillRequestCount = 32;
    workload.remainingPrefillTokenCount = workload.runningCapacityPrefillTokenCount = 4096;
    workload.remainingPrefillTokenSquareSum = workload.runningCapacityPrefillTokenSquareSum = 524288.0;
    workload.remainingTotalTokenCount = workload.totalPromptTokenCount = 4096;
    workload.remainingTotalTokenSquareSum = 524288.0;
    ASSERT_TRUE(workload.valid());
    autil::DataBuffer buffer;
    workload.serialize(buffer);
    NaviSchedWorkloadSnapshot decoded;
    decoded.deserialize(buffer);
    ASSERT_TRUE(decoded.valid());
    EXPECT_TRUE(workload.sameWorkload(decoded));
    EXPECT_EQ(1, decoded.runningElapsedRequestCount);
    decoded.runningElapsedRequestCount = 2;
    EXPECT_FALSE(decoded.valid());
    decoded.runningElapsedRequestCount = -2;
    EXPECT_FALSE(decoded.valid());
    decoded.runningElapsedRequestCount = -1;
    EXPECT_TRUE(decoded.valid());
    decoded.runningElapsedRequestCount = 0;
    EXPECT_FALSE(decoded.valid());
    workload.runningElapsedTimeMs = decoded.runningElapsedTimeMs = 0.0;
    EXPECT_TRUE(decoded.valid());
    EXPECT_FALSE(workload.sameWorkload(decoded));
}

TEST(ExecutionLoadResourceTest, DataReadyImmediatelyPublishesWithoutANativeStepAndCannotRegress) {
    AgentReportTestContext context;
    ASSERT_TRUE(context.init(19000, "10.0.0.1", 0, false, true, false, true, "", true));
    const auto resource = context.get();
    auto pipe = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource->subscribe("ready-push", pipe));
    auto load = resource->beginKVPreparation(1, 100);
    ASSERT_TRUE(load);
    auto workload = executionLoadTestWorkload();
    workload.executionLoadRequests[0].loadGeneration = load->key.generation;
    const auto before = resource->reportWorkloadUpdated(workload, {});
    const auto reports = pipe->datas.size();
    EXPECT_EQ(before, resource->getWorkloadSnapshot()->revision);
    EXPECT_EQ(reports, pipe->datas.size());
    EXPECT_TRUE(resource->completeKVPreparation(load->key, true, 300).terminalChanged);
    auto ready = resource->getWorkloadSnapshot();
    ASSERT_TRUE(ready); EXPECT_GT(ready->revision, before);
    EXPECT_GT(pipe->datas.size(), reports);
    ASSERT_EQ(2u, ready->executionLoadRequests.size());
    EXPECT_EQ(NativeKVExecutionState::READY, ready->executionLoadRequests[0].state);
    EXPECT_EQ(workload.requestCount, ready->requestCount);
    EXPECT_EQ(workload.waitingRequestCount, ready->waitingRequestCount);
    EXPECT_EQ(workload.remainingPrefillTokenCount, ready->remainingPrefillTokenCount);
    // A later Native step still saying "loading" cannot undo physical readiness.
    EXPECT_EQ(ready->revision, resource->reportWorkloadUpdated(workload, {}));
    EXPECT_EQ(NativeKVExecutionState::READY, resource->getWorkloadSnapshot()->executionLoadRequests[0].state);
    const auto reportsAfter = pipe->datas.size();
    EXPECT_FALSE(resource->completeKVPreparation(load->key, true, 400).terminalChanged);
    EXPECT_EQ(reportsAfter, pipe->datas.size());
    EXPECT_EQ(1u, resource->takeKVLoadLatencyWindowStats().readyCount);
}

TEST(ExecutionLoadResourceTest, OldGenerationCannotMakeANewLoadReady) {
    AgentReportTestContext context;
    ASSERT_TRUE(context.init(19000, "10.0.0.1", 0, false, true, false, true, "", true));
    const auto resource = context.get();
    auto old = resource->beginKVPreparation(1, 100); ASSERT_TRUE(old);
    auto workload = executionLoadTestWorkload();
    workload.executionLoadRequests[0].loadGeneration = old->key.generation;
    ASSERT_GT(resource->reportWorkloadUpdated(workload, {}), 0u);
    resource->completeKVPreparation(old->key, true, 200);
    auto next = resource->beginKVPreparation(1, 300); ASSERT_TRUE(next);
    ASSERT_GT(next->key.generation, old->key.generation);
    workload.executionLoadRequests[0].loadGeneration = next->key.generation;
    ASSERT_GT(resource->reportWorkloadUpdated(workload, {}), 0u);
    EXPECT_FALSE(resource->completeKVPreparation(old->key, true, 400).terminalChanged);
    EXPECT_EQ(NativeKVExecutionState::PREPARING, resource->getWorkloadSnapshot()->executionLoadRequests[0].state);
    resource->completeKVPreparation(next->key, true, 500);
    EXPECT_EQ(NativeKVExecutionState::READY, resource->getWorkloadSnapshot()->executionLoadRequests[0].state);
}

TEST(ExecutionLoadResourceTest, LoadFailureDoesNotInventReadyAndNativeRecomputationRemainsExecution) {
    AgentReportTestContext context;
    ASSERT_TRUE(context.init(19000, "10.0.0.1", 0, false, true, false, true, "", true));
    const auto resource = context.get();
    auto load = resource->beginKVPreparation(1, 100); ASSERT_TRUE(load);
    auto workload = executionLoadTestWorkload();
    workload.executionLoadRequests[0].loadGeneration = load->key.generation;
    ASSERT_GT(resource->reportWorkloadUpdated(workload, {}), 0u);
    resource->completeKVPreparation(load->key, false, 200);
    EXPECT_EQ(NativeKVExecutionState::FAILED, resource->getWorkloadSnapshot()->executionLoadRequests[0].state);
    resource->completeKVPreparation(load->key, true, 300);
    EXPECT_EQ(NativeKVExecutionState::FAILED, resource->getWorkloadSnapshot()->executionLoadRequests[0].state);
    workload.executionLoadRequests[0].state = NativeKVExecutionState::EXECUTING;
    ASSERT_GT(resource->reportWorkloadUpdated(workload, {}), 0u);
    EXPECT_EQ(NativeKVExecutionState::EXECUTING, resource->getWorkloadSnapshot()->executionLoadRequests[0].state);
    const auto metrics = resource->takeKVLoadLatencyWindowStats();
    EXPECT_EQ(0u, metrics.readyCount); EXPECT_EQ(1u, metrics.failedCount);
}

TEST(ExecutionLoadResourceTest, UntrackedWorkloadDoesNotChangeWhenAReadinessEventArrives) {
    AgentReportTestContext context;
    ASSERT_TRUE(context.init(19000, "10.0.0.1", 0, false, true, false, true, "", true));
    const auto resource = context.get();
    auto workload = executionLoadTestWorkload();
    workload.executionLoadTracked = false; workload.executionLoadRequests.clear();
    const auto revision = resource->reportWorkloadUpdated(workload, {});
    ASSERT_GT(revision, 0u);
    auto load = resource->beginKVPreparation(1, 100); ASSERT_TRUE(load);
    resource->completeKVPreparation(load->key, true, 200);
    EXPECT_EQ(revision, resource->getWorkloadSnapshot()->revision);
    EXPECT_FALSE(resource->getWorkloadSnapshot()->executionLoadTracked);
    EXPECT_EQ(workload.requestCount, resource->getWorkloadSnapshot()->requestCount);
}


namespace {
class InitialKVLoadNode : public NaviSchedNode {
public:
    mutable size_t plainCalls = 0;
    mutable size_t richCalls = 0;
    std::vector<int64_t> totalBlocks{3};
    std::vector<int64_t> hbmBlocks{1};
    NaviSchedNodeInfo getInfo() const override { return {}; }
    bool getScheduleSnapshot(const std::string &, NaviSchedNodeSnapshot &) const override {
        return false;
    }
    int64_t getMatchedBlockCount(const NaviSchedBlockHashes &) const override { return 0; }
    void getMatchedBlockCounts(const std::vector<NaviSchedScheduleItem> &requests,
                              int64_t *counts, int64_t *external = nullptr) const override {
        ++plainCalls;
        for (size_t i = 0; i < requests.size(); ++i) {
            counts[i] = totalBlocks.at(i % totalBlocks.size());
            if (external) external[i] = 0;
        }
    }
    bool getExecutionCacheInputs(
        const std::vector<NaviSchedScheduleItem> &requests,
        std::vector<TPUPrefillRequestInput> &inputs, int64_t *external = nullptr,
        TPUPrefillCacheObservationBatch *observation = nullptr,
        const NaviSchedExternalCacheReadyBlocks *remote = nullptr,
        bool = true) const override {
        ++richCalls;
        EXPECT_EQ(nullptr, remote); // This source has no Tair query result.
        inputs.clear();
        if (observation && observation->readyOnlyHbmBlockCounts)
            observation->readyOnlyHbmBlockCounts->clear();
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto total = totalBlocks.at(i % totalBlocks.size());
            const auto hbm = hbmBlocks.at(i % hbmBlocks.size());
            inputs.push_back({requests[i].request.promptTokenCount,
                              total * requests[i].request.blockSize, 0, 0});
            if (external) external[i] = 0;
            if (observation && observation->readyOnlyHbmBlockCounts)
                observation->readyOnlyHbmBlockCounts->push_back(hbm);
        }
        return true;
    }
};

void initKVLoadContext(NaviSchedScheduleContext &context,
                       const NaviSchedNodeConstPtr &source,
                       bool tracking = true, size_t requests = 1) {
    NaviSchedNodeSnapshot snapshot;
    auto info = std::make_shared<NaviSchedNodeInfo>();
    info->bizName = "kv_load_test";
    info->agentId = "agent_0";
    info->serverAddress = "127.0.0.1:19001";
    info->collectorAddress = "127.0.0.1:19002";
    info->pdSeparated = true;
    snapshot.info = info;
    snapshot.workload.executionLoadTracked = tracking;
    for (size_t i = 0; i < requests; ++i) {
        NaviSchedScheduleItem item;
        item.request.requestId = 101 + i;
        item.request.promptTokenCount = 128;
        item.request.blockSize = 32;
        context._requests.emplace_back(std::move(item));
    }
    context.setNodes({source}, {snapshot});
    ASSERT_TRUE(context.prepare());
}
}

TEST(ExecutionLoadResourceTest, testMasterInitialKVLoadNoTairClassifiesMissHbmAndHostMemory) {
    for (const auto test : {std::pair<int64_t, int64_t>{0, 0}, {3, 3}, {3, 1}, {4, 4}}) {
        auto source = std::make_shared<InitialKVLoadNode>();
        source->totalBlocks = {test.first}; source->hbmBlocks = {test.second};
        NaviSchedScheduleContext context;
        initKVLoadContext(context, source);
        auto first = context.getCacheHitBlockCountsSnapshot(0);
        ASSERT_TRUE(first);
        EXPECT_EQ(test.first, first->at(0));
        // A later cache change must not change this decision's physical source.
        source->hbmBlocks = {0}; source->totalBlocks = {0};
        ASSERT_TRUE(context.selectNode(0, 0, "kv_load_test"));
        ASSERT_TRUE(context.prepareResult(context.getScheduleTimeUs()));
        ASSERT_EQ(1u, context._decisions.size());
        const auto &decision = context._decisions[0];
        ASSERT_TRUE(decision.executionLoadIncluded.has_value());
        EXPECT_EQ(test.first <= test.second, *decision.executionLoadIncluded);
        EXPECT_EQ(test.first * 32, decision.predictedCacheHitTokenCount);
        auto response = context.makeResponse(decision, 0);
        ASSERT_TRUE(response);
        EXPECT_EQ(decision.executionLoadIncluded, response->executionLoadIncluded);
        EXPECT_EQ(1u, source->richCalls);
        EXPECT_EQ(0u, source->plainCalls);
        EXPECT_EQ(0u, context._selectedNodeIndexes[0]);
    }
}

TEST(ExecutionLoadResourceTest, testMasterInitialKVLoadKeepsUntrackedNodeBehavior) {
    auto source = std::make_shared<InitialKVLoadNode>();
    NaviSchedScheduleContext context;
    initKVLoadContext(context, source, false);
    ASSERT_TRUE(context.selectNode(0, 0, "kv_load_test"));
    ASSERT_TRUE(context.prepareResult(context.getScheduleTimeUs()));
    EXPECT_FALSE(context._decisions[0].executionLoadIncluded.has_value());
    EXPECT_EQ(1u, source->plainCalls);
    EXPECT_EQ(0u, source->richCalls);
}

TEST(ExecutionLoadResourceTest, testMasterInitialKVLoadMissingHbmDoesNotDefaultReady) {
    auto source = std::make_shared<InitialKVLoadNode>();
    source->hbmBlocks = {-1};
    NaviSchedScheduleContext context;
    initKVLoadContext(context, source);
    ASSERT_TRUE(context.selectNode(0, 0, "kv_load_test"));
    EXPECT_FALSE(context.prepareResult(context.getScheduleTimeUs()));
    EXPECT_FALSE(context._decisions[0].executionLoadIncluded.has_value());
}

TEST(ExecutionLoadResourceTest, testMasterInitialKVLoadFlagsFollowRequestAfterScheduleView) {
    auto source = std::make_shared<InitialKVLoadNode>();
    source->totalBlocks = {3, 3, 0}; source->hbmBlocks = {3, 1, 0};
    NaviSchedScheduleContext context;
    initKVLoadContext(context, source, true, 3);
    ASSERT_TRUE(context.activateScheduleView({2, 0}, {0}));
    ASSERT_TRUE(context.getCacheHitBlockCountsSnapshot(0));
    context.clearScheduleView();
    for (size_t i = 0; i < 3; ++i) ASSERT_TRUE(context.selectNode(i, 0, "kv_load_test"));
    ASSERT_TRUE(context.prepareResult(context.getScheduleTimeUs()));
    EXPECT_EQ(std::optional<bool>(true), context._decisions[0].executionLoadIncluded);
    EXPECT_EQ(std::optional<bool>(false), context._decisions[1].executionLoadIncluded);
    EXPECT_EQ(std::optional<bool>(true), context._decisions[2].executionLoadIncluded);
    EXPECT_EQ(1u, source->richCalls);
}

TEST(ExecutionLoadResourceTest, testMasterInitialKVLoadUsesActualNodeV6dMemoryMatcher) {
    class CountingNode : public MasterStateR::NodeState {
    public:
        mutable size_t calls = 0;
        mutable size_t reusedPrefixCount = 0;
        using MasterStateR::NodeState::getExecutionCacheInputs;
        bool getExecutionCacheInputs(
            const std::vector<NaviSchedScheduleItem> &requests,
            std::vector<TPUPrefillRequestInput> &inputs, int64_t *external,
            TPUPrefillCacheObservationBatch *observation,
            const NaviSchedExternalCacheReadyBlocks *remote,
            bool boundary, const std::vector<size_t> *prefixes) const override {
            ++calls;
            reusedPrefixCount = prefixes ? prefixes->size() : 0;
            EXPECT_EQ(nullptr, remote);
            return MasterStateR::NodeState::getExecutionCacheInputs(
                requests, inputs, external, observation, remote, boundary, prefixes);
        }
    };
    MasterPendingCacheR pending;
    auto source = std::make_shared<CountingNode>();
    source->_pendingCacheR = &pending;
    source->_info.agentIdHash = 123;
    source->_cachePolicy.blockSize = 32;
    source->_cachePolicy.linearStep = 1;
    source->_cachePolicyReady.store(true);
    source->_cacheStateReady.store(true);
    source->_cachedBlockHashes = {10};
    auto &memory = source->_v6dCacheState;
    memory._cachePolicy.blockSize = 32;
    memory._cachePolicy.linearStep = 1;
    memory._cachePolicyReady.store(true);
    memory._cacheStateReady.store(true);
    memory._cachedBlockHashes = {20, 30};
    NaviSchedScheduleContext context;
    initKVLoadContext(context, source);
    context._requests[0].request.blockHashes = std::make_shared<const NaviSchedBlockHashes>(
        NaviSchedBlockHashes{10, 20, 30, 40});
    context.prepareCacheMatchPrefixes();
    auto counts = context.getCacheHitBlockCountsSnapshot(0);
    ASSERT_TRUE(counts);
    ASSERT_EQ(3, counts->at(0));
    source->_cachedBlockHashes = {10, 20, 30, 40};
    ASSERT_TRUE(context.selectNode(0, 0, "kv_load_test"));
    ASSERT_TRUE(context.prepareResult(context.getScheduleTimeUs()));
    EXPECT_EQ(96, context._decisions[0].predictedCacheHitTokenCount);
    EXPECT_EQ(std::optional<bool>(false), context._decisions[0].executionLoadIncluded);
    EXPECT_EQ(1u, source->calls);
    EXPECT_EQ(1u, source->reusedPrefixCount);
}

} // namespace navi_sched

namespace navi_sched {
namespace {
NaviSchedNodeSnapshot preparationProjectionSnapshot() {
    NaviSchedNodeSnapshot s;
    auto info=std::make_shared<NaviSchedNodeInfo>();info->agentId="kv_projection";info->bizName="test";s.info=info;
    auto model=std::make_shared<NaviSchedLatencyModel>();model->formula=NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    model->binaryVersion=NonLinearPrefillModelR::BINARY_VERSION;model->revision=1;
    model->parameters.assign(19,0.0);model->parameters[8]=1.0;
    model->parameters[15]=5.0/NonLinearPrefillModelR::PARAMETER_SCALES[15];
    EXPECT_TRUE(NonLinearPrefillModelR::validateParameters(model->parameters));s.latencyModel=model;
    s.unifiedLatencyPrediction=true;s.workload=executionLoadTestWorkload();
    s.reportedWorkloadFeatures=*MasterKVReadyWorkload::project(s.workload,s.workload.executionLoadRequests,true);
    s.workloadFeatures=s.reportedWorkloadFeatures;
    s.maxRunningRequestCount=64;s.workloadRunningCapacitySlotCount=2;
    s.kvPreparationSnapshotMonoUs=1000000;
    s.kvPreparationEstimates.push_back({1,5,1000000,20.0,100.0});
    return s;
}
}
TEST(ExecutionLoadResourceTest, NodeUsesFractionalPlanningLoadAndRetainsActualFacts) {
    auto snapshot=preparationProjectionSnapshot();
    NaviSchedScheduleNode node(nullptr,std::move(snapshot),1);
    const auto expected=node.getReportedWorkloadFeatures().getExpectedKVWorkload();ASSERT_NE(nullptr,expected);
    EXPECT_DOUBLE_EQ(0.20,expected->precedingRequests);
    EXPECT_DOUBLE_EQ(20.0,expected->running[2]);EXPECT_DOUBLE_EQ(2000.0,expected->running[3]);
    EXPECT_EQ(1,node.getWorkloadFeatures().getPrecedingRequestCount());
    EXPECT_DOUBLE_EQ(1.20,node.getWorkloadFeatures().getPlanningPrecedingRequestCount());
    EXPECT_EQ(200,node.getWorkloadFeatures().getCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(220.0,node.getWorkloadFeatures().getPlanningCacheMissTokenCount());
    NaviSchedLatencyFeatures floor;ASSERT_TRUE(node.getUnifiedDecayFloorFeatures(floor));
    EXPECT_DOUBLE_EQ(1.20,floor.getPlanningPrecedingRequestCount());
    EXPECT_DOUBLE_EQ(0.0,floor.getPlanningCacheMissTokenCount());
}
TEST(ExecutionLoadResourceTest, ProbabilityAuditPreservesAppliedInputsAndUnknownForecast) {
    auto snapshot = preparationProjectionSnapshot();
    snapshot.kvPreparationSnapshotMonoUs += 10000;
    NaviSchedScheduleNode node(nullptr, std::move(snapshot), 1);
    const auto &audits = node.getKVPreparationProbabilityAudits();
    ASSERT_EQ(1u, audits.size());
    const auto &audit = audits.front();
    EXPECT_EQ(1u, audit.requestId);
    EXPECT_EQ(5u, audit.generation);
    EXPECT_EQ(1010000, audit.snapshotMonoUs);
    EXPECT_EQ(1000000, audit.reportReceiveMonoUs);
    EXPECT_DOUBLE_EQ(20.0, audit.observedPreparationMs);
    EXPECT_DOUBLE_EQ(30.0, audit.elapsedAtSnapshotMs);
    EXPECT_DOUBLE_EQ(0.0, audit.queueHorizonMs);
    ASSERT_TRUE(audit.frozenTotalMs);
    EXPECT_DOUBLE_EQ(100.0, *audit.frozenTotalMs);
    EXPECT_DOUBLE_EQ(0.30, audit.probability);
    ASSERT_NE(nullptr, node.getReportedWorkloadFeatures().getExpectedKVWorkload());
    EXPECT_DOUBLE_EQ(audit.probability,
        node.getReportedWorkloadFeatures().getExpectedKVWorkload()->precedingRequests);

    auto unknown = preparationProjectionSnapshot();
    unknown.kvPreparationEstimates.front().totalMs.reset();
    NaviSchedScheduleNode unknownNode(nullptr, std::move(unknown), 1);
    const auto &unknownAudits = unknownNode.getKVPreparationProbabilityAudits();
    ASSERT_EQ(1u, unknownAudits.size());
    EXPECT_FALSE(unknownAudits.front().frozenTotalMs);
    EXPECT_DOUBLE_EQ(1.0, unknownAudits.front().probability);
}

TEST(ExecutionLoadResourceTest, ReadyAndFailureRemoveExpectedContributionAndWrongIdentityCannotDefault) {
    for(auto state:{NativeKVExecutionState::READY,NativeKVExecutionState::FAILED}) {
        auto snapshot=preparationProjectionSnapshot();snapshot.workload.executionLoadRequests[0].state=state;
        snapshot.kvPreparationEstimates.clear();snapshot.reportedWorkloadFeatures=*MasterKVReadyWorkload::project(snapshot.workload,snapshot.workload.executionLoadRequests,true);
        snapshot.workloadFeatures=snapshot.reportedWorkloadFeatures;
        NaviSchedScheduleNode node(nullptr,std::move(snapshot),1);
        EXPECT_EQ(nullptr,node.getWorkloadFeatures().getExpectedKVWorkload());
        EXPECT_TRUE(node.getKVPreparationProbabilityAudits().empty());
    }
    auto wrong=preparationProjectionSnapshot();wrong.kvPreparationEstimates[0].generation=6;
    EXPECT_THROW(NaviSchedScheduleNode(nullptr,std::move(wrong),1),std::invalid_argument);
    auto clock=preparationProjectionSnapshot();clock.kvPreparationSnapshotMonoUs=0;
    EXPECT_THROW(NaviSchedScheduleNode(nullptr,std::move(clock),1),std::invalid_argument);
}
TEST(ExecutionLoadResourceTest, FrozenSnapshotProjectionIsNotRecomputedOrDroppedAtZero) {
    for(double probability:{0.0,0.25}) {
        auto snapshot=preparationProjectionSnapshot();auto info=std::make_shared<NaviSchedNodeInfo>(*snapshot.info);info->workloadFeatureSnapshotValid=true;snapshot.info=info;
        NaviSchedExpectedKVWorkload expected;expected.precedingRequests=probability;expected.running={probability,probability*10,probability*100,probability*10000,probability*1000};
        ASSERT_TRUE(snapshot.workloadFeatures.setExpectedKVWorkload(expected));snapshot.kvPreparationEstimates.clear();snapshot.kvPreparationSnapshotMonoUs=0;
        NaviSchedScheduleNode node(nullptr,std::move(snapshot),1);
        EXPECT_DOUBLE_EQ(1.0+probability,node.getWorkloadFeatures().getPlanningPrecedingRequestCount());
        EXPECT_TRUE(node.getKVPreparationProbabilityAudits().empty());
        const auto restored=node.getReportedWorkloadFeatures().getExpectedKVWorkload();
        if(probability==0.0) EXPECT_EQ(nullptr,restored);else {ASSERT_NE(nullptr,restored);EXPECT_DOUBLE_EQ(probability,restored->precedingRequests);}
    }
}
TEST(ExecutionLoadResourceTest, UntrackedNodePredictionDoesNotAcquireExpectedWorkload) {
    auto snapshot=preparationProjectionSnapshot();snapshot.workload.executionLoadTracked=false;snapshot.workload.executionLoadRequests.clear();snapshot.kvPreparationEstimates.clear();
    snapshot.reportedWorkloadFeatures={};ASSERT_TRUE(snapshot.reportedWorkloadFeatures.appendPrefillWorkload(snapshot.workload));snapshot.workloadFeatures=snapshot.reportedWorkloadFeatures;
    const auto prediction=NonLinearPrefillModelR::predict(snapshot.workloadFeatures,snapshot.latencyModel->parameters);
    NaviSchedScheduleNode node(nullptr,std::move(snapshot),1);
    EXPECT_EQ(nullptr,node.getWorkloadFeatures().getExpectedKVWorkload());EXPECT_DOUBLE_EQ(prediction,node.predictLatency(node.getWorkloadFeatures()));
    NaviSchedLatencyFeatures floor;ASSERT_TRUE(node.getUnifiedDecayFloorFeatures(floor));
    EXPECT_FALSE(floor.usesCapacitySplit()); // Keep the pre-feature untracked path.
    EXPECT_TRUE(node.getKVPreparationProbabilityAudits().empty());
}
}

namespace navi_sched {
TEST(ExecutionLoadResourceTest, UnreportedPreparingReservationTransfersExactlyOnceWithoutInventingSlots) {
    MasterBatchReservation reservation;
    MasterBatchReservation::Request request;
    request.requestId=3;request.totalTokenCount=110;request.blockSize=10;
    request.predictedCacheHitTokenCount=10;request.executionLoadIncluded=false;
    ASSERT_TRUE(reservation.init(11,100,0,{request},nullptr));
    NaviSchedLatencyFeatures before;int32_t slots=0;
    ASSERT_TRUE(reservation.appendWaitingCapacityFeatures(before,64,slots,true,true));
    EXPECT_EQ(0,slots);EXPECT_EQ(0,before.getBatchCount());EXPECT_EQ(0,before.getBatchSize());
    EXPECT_EQ(0,before.getCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(100.0,before.getPlanningCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(1.0,before.getPlanningPrecedingRequestCount());
    ASSERT_TRUE(reservation.markAcceptedWorkloadReported(3));
    NaviSchedLatencyFeatures after;
    ASSERT_TRUE(reservation.appendWaitingCapacityFeatures(after,64,slots,true,true));
    EXPECT_EQ(nullptr,after.getExpectedKVWorkload());
    EXPECT_DOUBLE_EQ(0.0,after.getPlanningCacheMissTokenCount());
    auto raw=preparationProjectionSnapshot();raw.workload.executionLoadRequests[0].requestId=3;
    raw.kvPreparationEstimates[0].requestId=3;
    NaviSchedScheduleNode node(nullptr,std::move(raw),1);
    const auto projected=node.getReportedWorkloadFeatures().getExpectedKVWorkload();
    ASSERT_NE(nullptr,projected);EXPECT_NEAR(0.20,projected->precedingRequests,1e-12);
    EXPECT_NEAR(20.0,projected->running[2],1e-12);
}
TEST(ExecutionLoadResourceTest, ReservedAndReportedExpectedWorkloadAreCombinedAndReplayDoesNotDuplicateThem) {
    for(bool replay:{false,true}) {
        auto snapshot=preparationProjectionSnapshot();
        NaviSchedExpectedKVWorkload reserved;
        reserved.precedingRequests=1.0;reserved.waiting={1.0,30.0,500.0,250000.0,15000.0};
        ASSERT_TRUE(snapshot.reservedWorkloadFeatures.appendCapacitySplitAggregate(0,0,0,0,0,0,0,0,0,0,0));
        ASSERT_TRUE(snapshot.reservedWorkloadFeatures.setExpectedKVWorkload(reserved));
        ASSERT_TRUE(snapshot.workloadFeatures.append(snapshot.reservedWorkloadFeatures));
        if(replay) {
            auto info=std::make_shared<NaviSchedNodeInfo>(*snapshot.info);info->workloadFeatureSnapshotValid=true;snapshot.info=info;
            NaviSchedExpectedKVWorkload complete=reserved;
            complete.precedingRequests+=0.25;complete.running={0.25,2.5,25.0,2500.0,250.0};
            ASSERT_TRUE(snapshot.workloadFeatures.setExpectedKVWorkload(complete));
            snapshot.kvPreparationEstimates.clear();snapshot.kvPreparationSnapshotMonoUs=0;
        }
        NaviSchedScheduleNode node(nullptr,std::move(snapshot),1);
        const double probability = replay ? 0.25 : 0.20;
        EXPECT_NEAR(700.0+probability*100,node.getWorkloadFeatures().getPlanningCacheMissTokenCount(),1e-12);
        EXPECT_NEAR(200.0+probability*100,node.getReportedWorkloadFeatures().getPlanningCacheMissTokenCount(),1e-12);
        EXPECT_DOUBLE_EQ(500.0,node.getReservedWorkloadFeatures().getPlanningCacheMissTokenCount());
        NaviSchedLatencyFeatures floor;ASSERT_TRUE(node.getUnifiedDecayFloorFeatures(floor));
        EXPECT_DOUBLE_EQ(500.0,floor.getPlanningCacheMissTokenCount());
        EXPECT_NEAR(2.0+probability,floor.getPlanningPrecedingRequestCount(),1e-12);
    }
}
}

namespace navi_sched {
TEST(ExecutionLoadResourceTest, EmptyReservationFloorUsesTheModelFeatureFamilyAndKeepsKnownOverhead) {
    auto snapshot=preparationProjectionSnapshot();
    NaviSchedScheduleNode node(nullptr,std::move(snapshot),1);
    NaviSchedLatencyFeatures floor;ASSERT_TRUE(node.getUnifiedDecayFloorFeatures(floor));
    EXPECT_TRUE(floor.usesCapacitySplit());
    EXPECT_NEAR(6.0,node.predictLatency(floor),1e-12);
    EXPECT_NEAR(6.0,node.predictLatency(node.getWorkloadFeatures()),1e-12);
    EXPECT_DOUBLE_EQ(0.0,floor.getPlanningCacheMissTokenCount());
    // This model contains only 5ms per preceding request, with no compute cost.
    // Its fixed request overhead must not become a 5ms preparation horizon.
    const auto expected=node.getReportedWorkloadFeatures().getExpectedKVWorkload();
    ASSERT_NE(nullptr,expected);EXPECT_NEAR(0.20,expected->precedingRequests,1e-12);
}
}

namespace navi_sched {
class MasterKVPreparationLifecycleTest : public TESTBASE {
public:
    void initialize() {
        state=runtime.registerAgent("kv_lifecycle","model_biz","127.0.0.1:19000",100);
        ASSERT_TRUE(state);
        handshake=makeControlUpdate(NaviSchedAgentEvent::AGENT_HANDSHAKE,100);
        handshake->control->pdSeparated=true;handshake->control->unifiedLatencyPrediction=true;
        handshake->control->serverInfo.port=18000;
        handshake->latency=std::make_shared<AgentLatencyToMasterData>();
        handshake->latency->formula=NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        handshake->latency->binaryVersion=NonLinearPrefillModelR::BINARY_VERSION;
        handshake->latency->modelRevision=1;
        NonLinearPrefillModelR model;model._useCapacitySplitFeature=true;
        handshake->latency->parameters=model.getInitialParameters();
        handshake->workload=std::make_shared<NaviSchedWorkloadSnapshot>(executionLoadTestWorkload());
        ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",handshake));
        ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",makeCachePolicyUpdate(256)));
        ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",makeEmptyPendingCacheSnapshot()));
    }
    void reserve(int64_t batchId,uint64_t requestId) {
        MasterBatchReservation::Request r;
        r.requestId=requestId;r.totalTokenCount=110;r.blockSize=10;r.predictedCacheHitTokenCount=10;
        r.executionLoadIncluded=false;r.predictedExternalReadTokens=10;r.initialKVPreparationQueueMs=0;
        ASSERT_TRUE(state->reserveBatch(batchId,autil::TimeUtility::currentTime(),0,{r}));
    }
    AgentUpdateToMasterDataPtr observation(uint64_t revision,uint64_t requestId,
                                          NativeKVExecutionState phase,double age) {
        auto update=std::make_shared<AgentUpdateToMasterData>();
        update->event=NaviSchedAgentEvent::WORKLOAD_UPDATED;update->eventTimeUs=100+revision;
        update->workload=std::make_shared<NaviSchedWorkloadSnapshot>(executionLoadTestWorkload());
        update->workload->revision=revision;
        auto &load=update->workload->executionLoadRequests[0];
        load.requestId=requestId;load.loadGeneration=requestId;load.state=phase;load.preparationElapsedMs=age;
        auto &requests=update->workload->executionLoadRequests;
        std::sort(requests.begin(),requests.end(),[](const auto &a,const auto &b) {
            return a.requestId<b.requestId;
        });
        return update;
    }
    static void handoff(const AgentUpdateToMasterDataPtr &update,int64_t batchId,uint64_t requestId) {
        auto &batch=update->workload->admittedBatch;
        batch.scheduleBatchId=batchId;batch.localBatchId=10;batch.requestCount=1;batch.requestIds={requestId};
    }
    TestMasterStateR runtime;
    MasterStateR::NodeStatePtr state;
    AgentUpdateToMasterDataPtr handshake;
};
TEST_F(MasterKVPreparationLifecycleTest, ReservationHandoffFixedDurationStaleUpdatesFailureAndSessionReset) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_NO_FATAL_FAILURE(reserve(11,3));
    NaviSchedNodeSnapshot before;ASSERT_TRUE(state->getScheduleSnapshot("model_biz",before));
    const auto unreported=before.reservedWorkloadFeatures.getExpectedKVWorkload();
    ASSERT_NE(nullptr,unreported);EXPECT_DOUBLE_EQ(1.0,unreported->precedingRequests);
    auto preparing=observation(2,3,NativeKVExecutionState::PREPARING,0);handoff(preparing,11,3);
    ASSERT_TRUE(preparing->valid());
    ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",preparing));
    NaviSchedNodeSnapshot accepted;ASSERT_TRUE(state->getScheduleSnapshot("model_biz",accepted));
    EXPECT_EQ(nullptr,accepted.reservedWorkloadFeatures.getExpectedKVWorkload());
    ASSERT_EQ(1u,accepted.kvPreparationEstimates.size());
    EXPECT_EQ(3u,accepted.kvPreparationEstimates[0].requestId);
    EXPECT_FALSE(accepted.kvPreparationEstimates[0].totalMs);
    NaviSchedScheduleNode projected(nullptr,std::move(accepted),1);
    EXPECT_DOUBLE_EQ(2.0,projected.getWorkloadFeatures().getPlanningPrecedingRequestCount());
    EXPECT_DOUBLE_EQ(300.0,projected.getWorkloadFeatures().getPlanningCacheMissTokenCount());
    auto ready=observation(3,3,NativeKVExecutionState::READY,100);
    ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",ready));
    EXPECT_EQ(1u,state->_kvPreparationForecasts.model().samples());
    EXPECT_NEAR(100.0,*state->_kvPreparationForecasts.model().predict({1,10.0/4096.0,0,0}),1e-8);
    ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",ready));
    ASSERT_TRUE(preparing->valid());
    ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",preparing));
    EXPECT_EQ(1u,state->_kvPreparationForecasts.model().samples());
    EXPECT_EQ(3u,state->getRuntimeState().workload.revision);
    ASSERT_NO_FATAL_FAILURE(reserve(12,4));
    auto next=observation(4,4,NativeKVExecutionState::PREPARING,25);handoff(next,12,4);
    ASSERT_TRUE(next->valid());
    ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",next));
    NaviSchedNodeSnapshot forecast;ASSERT_TRUE(state->getScheduleSnapshot("model_biz",forecast));
    ASSERT_EQ(1u,forecast.kvPreparationEstimates.size());
    const auto frozen=forecast.kvPreparationEstimates[0];ASSERT_TRUE(frozen.totalMs);
    EXPECT_NEAR(100.0,*frozen.totalMs,1e-8);EXPECT_DOUBLE_EQ(25.0,frozen.observedPreparationMs);
    auto failed=observation(5,4,NativeKVExecutionState::FAILED,-1);
    ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",failed));
    EXPECT_EQ(1u,state->_kvPreparationForecasts.model().samples());
    NaviSchedNodeSnapshot failure;ASSERT_TRUE(state->getScheduleSnapshot("model_biz",failure));
    EXPECT_TRUE(failure.kvPreparationEstimates.empty());
    handshake->control->agentStartTimeUs=200;
    ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",handshake));
    EXPECT_EQ(0u,state->_kvPreparationForecasts.model().samples());
    EXPECT_EQ(0u,state->_kvPreparationForecasts.reservations());
}
TEST_F(MasterKVPreparationLifecycleTest, InvalidReservationTransitionCannotTrainOrConsumePreparationMetadata) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_NO_FATAL_FAILURE(reserve(11,3));
    const auto count=state->_kvPreparationForecasts.model().samples();
    const auto loads=state->_kvPreparationForecasts.activeLoads();
    const auto reservations=state->_kvPreparationForecasts.reservations();
    auto invalid=observation(2,3,NativeKVExecutionState::READY,100);handoff(invalid,11,9999);
    EXPECT_FALSE(runtime.reportAgent("kv_lifecycle",invalid));
    EXPECT_EQ(1u,state->getRuntimeState().workload.revision);
    EXPECT_EQ(count,state->_kvPreparationForecasts.model().samples());
    EXPECT_EQ(loads,state->_kvPreparationForecasts.activeLoads());
    EXPECT_EQ(reservations,state->_kvPreparationForecasts.reservations());
    const auto reservation=state->findReservationLocked(11);ASSERT_NE(nullptr,reservation);
    EXPECT_EQ(MasterBatchReservation::RequestState::WAITING,reservation->batch.getRequestState(3));
    const auto preparing=observation(2,3,NativeKVExecutionState::PREPARING,0);handoff(preparing,11,3);
    ASSERT_TRUE(preparing->valid());
    ASSERT_TRUE(runtime.reportAgent("kv_lifecycle",preparing));
    EXPECT_EQ(count,state->_kvPreparationForecasts.model().samples());
}
}

namespace navi_sched {
namespace {
NaviSchedNodeSnapshot nativeStepForecastSnapshot(int64_t tokenBudget = 8192) {
    auto snapshot = preparationProjectionSnapshot();
    auto info = std::make_shared<NaviSchedNodeInfo>(*snapshot.info);
    info->latencyModelFingerprint =
        R"({"unified_capacity_batch_count_unit": "native_positive_prefill_step_v1"})";
    snapshot.info = info;
    auto model = std::make_shared<NaviSchedLatencyModel>(*snapshot.latencyModel);
    model->parameters.assign(19, 0.0);
    model->parameters[8] = 1.0;
    model->parameters[6] = 100.0 / NonLinearPrefillModelR::getParameterScale(6);
    model->parameters[15] = 5.0 / NonLinearPrefillModelR::getParameterScale(15);
    EXPECT_TRUE(NonLinearPrefillModelR::validateParameters(model->parameters));
    snapshot.latencyModel = model;
    TPUPrefillExecutionPolicyConfig policy;
    policy.maxScheduledTokens = tokenBudget;
    snapshot.prefillExecutionPolicy = policy;
    snapshot.workload.executionLoadTracked = false;
    snapshot.workload.executionLoadRequests.clear();
    snapshot.kvPreparationEstimates.clear();
    return snapshot;
}
NaviSchedLatencyFeatures nativeStepInput(int64_t runningTokens, int64_t waitingTokens = 0) {
    NaviSchedLatencyFeatures features;
    EXPECT_TRUE(features.appendCapacitySplitAggregate(
        6, 1, 0, runningTokens, double(runningTokens) * runningTokens, 0,
        waitingTokens > 0 ? 1 : 0, 0, waitingTokens,
        double(waitingTokens) * waitingTokens, 0));
    return features;
}
}

TEST(ExecutionLoadResourceTest, NativeStepForecastUsesReportedBudgetAndMatchesSmoothOptimizerAtWholeSteps) {
    for (const int64_t budget : {4096, 8192}) {
        auto snapshot = nativeStepForecastSnapshot(budget);
        const auto parameters = snapshot.latencyModel->parameters;
        NaviSchedScheduleNode node(nullptr, std::move(snapshot), 1);
        EXPECT_EQ(budget, node.getNativeStepTokenBudget());
        auto features = nativeStepInput(8 * budget, 8 * budget);
        auto expected = features;
        ASSERT_TRUE(expected.setBatchCount(16));
        const auto exact = NonLinearPrefillModelR::predict(expected, parameters);
        ASSERT_GT(exact, 0.0);
        EXPECT_NEAR(exact, node.predictBaseLatency(features), 1e-8);
        EXPECT_NEAR(exact, NonLinearPrefillModelR::predict(features, node.getLatencyParameters()), 1e-8);
        EXPECT_DOUBLE_EQ(0.0, node.getLatencyParameters()[6]);
        ASSERT_TRUE(node.prepareLatencyPredictionFeatures(features));
        EXPECT_EQ(16, features.getBatchCount());
        ASSERT_TRUE(node.prepareLatencyPredictionFeatures(features));
        EXPECT_EQ(16, features.getBatchCount());
        EXPECT_NEAR(exact, node.predictBaseLatency(features), 1e-8);
    }
}

TEST(ExecutionLoadResourceTest, NativeStepForecastWeightsPreparationTokensAndDropsLogicalExpectedBatchCount) {
    auto snapshot = nativeStepForecastSnapshot();
    NaviSchedScheduleNode node(nullptr, std::move(snapshot), 1);
    auto features = nativeStepInput(8192);
    NaviSchedExpectedKVWorkload expected;
    expected.precedingRequests = 0.25;
    expected.running = {0.25, 0.0, 2048.0, 16777216.0, 0.0};
    expected.batchCount = 99.0;
    ASSERT_TRUE(features.setExpectedKVWorkload(expected));
    ASSERT_TRUE(node.prepareLatencyPredictionFeatures(features));
    EXPECT_EQ(2, features.getBatchCount());
    const auto actual = features.getExpectedKVWorkload();
    ASSERT_NE(nullptr, actual);
    EXPECT_DOUBLE_EQ(0.25, actual->precedingRequests);
    EXPECT_DOUBLE_EQ(2048.0, actual->running[2]);
    EXPECT_DOUBLE_EQ(16777216.0, actual->running[3]);
    EXPECT_DOUBLE_EQ(0.0, actual->batchCount);
    EXPECT_GT(node.predictBaseLatency(features),
              NonLinearPrefillModelR::predict(nativeStepInput(8192), node.getLatencyParameters()));
}

TEST(ExecutionLoadResourceTest, NativeStepForecastWholeStepBoundaryDoesNotAlterOtherFeatures) {
    auto snapshot = nativeStepForecastSnapshot();
    NaviSchedScheduleNode node(nullptr, std::move(snapshot), 1);
    auto features = nativeStepInput(8193);
    const auto tokens = features.getCacheMissTokenCount();
    const auto square = features.getCacheMissTokenSquareSum();
    const auto requests = features.getBatchSize();
    ASSERT_TRUE(node.prepareLatencyPredictionFeatures(features));
    EXPECT_EQ(2, features.getBatchCount());
    EXPECT_EQ(tokens, features.getCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(square, features.getCacheMissTokenSquareSum());
    EXPECT_EQ(requests, features.getBatchSize());
    EXPECT_FALSE(features.setBatchCount(-1));
    EXPECT_EQ(2, features.getBatchCount());
}

TEST(ExecutionLoadResourceTest, NativeStepForecastDoesNotChangeLegacyModelsOrGuessMissingPolicy) {
    for (const auto &fingerprint : {std::string("legacy-3.7-model"),
             std::string(R"({"unified_capacity_batch_count_unit":"logical_batch"})")}) {
        auto snapshot = nativeStepForecastSnapshot();
        auto info = std::make_shared<NaviSchedNodeInfo>(*snapshot.info);
        info->latencyModelFingerprint = fingerprint;
        snapshot.info = info;
        snapshot.prefillExecutionPolicy.reset();
        const auto parameters = snapshot.latencyModel->parameters;
        NaviSchedScheduleNode node(nullptr, std::move(snapshot), 1);
        auto features = nativeStepInput(131072);
        const auto before = NonLinearPrefillModelR::predict(features, parameters);
        EXPECT_EQ(0, node.getNativeStepTokenBudget());
        ASSERT_TRUE(node.prepareLatencyPredictionFeatures(features));
        EXPECT_EQ(6, features.getBatchCount());
        EXPECT_EQ(parameters, node.getLatencyParameters());
        EXPECT_DOUBLE_EQ(before, node.predictBaseLatency(features));
    }
    auto missing = nativeStepForecastSnapshot();
    missing.prefillExecutionPolicy.reset();
    EXPECT_THROW(NaviSchedScheduleNode(nullptr, std::move(missing), 1), std::invalid_argument);
    auto zero = nativeStepForecastSnapshot(0);
    EXPECT_THROW(NaviSchedScheduleNode(nullptr, std::move(zero), 1), std::invalid_argument);
}
}

namespace navi_sched {
TEST(ExecutionLoadResourceTest, NativeStepPredictionWirePreservesNormalizedFullAndFloorFeatures) {
    auto snapshot = nativeStepForecastSnapshot();
    const auto parameters = snapshot.latencyModel->parameters;
    NaviSchedScheduleNode node(nullptr, std::move(snapshot), 1);
    NaviSchedLatencyPrediction prediction;
    prediction.features = nativeStepInput(131073);
    prediction.floorFeatures = nativeStepInput(8193);
    NaviSchedExpectedKVWorkload expected;
    expected.precedingRequests = 0.25;
    expected.running = {0.25, 0.0, 2048.0, 16777216.0, 0.0};
    ASSERT_TRUE(prediction.features.setExpectedKVWorkload(expected));
    ASSERT_TRUE(node.prepareLatencyPredictionFeatures(prediction.features));
    ASSERT_TRUE(node.prepareLatencyPredictionFeatures(prediction.floorFeatures));
    prediction.recomputable = true;
    prediction.hasFloor = true;
    autil::DataBuffer buffer;
    prediction.serialize(buffer);
    buffer.write(int64_t(77881));
    NaviSchedLatencyPrediction decoded;
    decoded.deserialize(buffer);
    int64_t sentinel = 0;
    buffer.read(sentinel);
    ASSERT_TRUE(decoded.valid());
    EXPECT_EQ(77881, sentinel);
    EXPECT_EQ(17, decoded.features.getBatchCount());
    EXPECT_EQ(2, decoded.floorFeatures.getBatchCount());
    ASSERT_NE(nullptr, decoded.features.getExpectedKVWorkload());
    EXPECT_DOUBLE_EQ(0.25, decoded.features.getExpectedKVWorkload()->precedingRequests);
    EXPECT_DOUBLE_EQ(NonLinearPrefillModelR::predict(prediction.features, parameters),
                     NonLinearPrefillModelR::predict(decoded.features, parameters));
    EXPECT_DOUBLE_EQ(NonLinearPrefillModelR::predict(prediction.floorFeatures, parameters),
                     NonLinearPrefillModelR::predict(decoded.floorFeatures, parameters));
}
}

namespace navi_sched {
TEST(ExecutionLoadResourceTest, NativeStepOptimizerBaselineUsesContinuousWorkLikeItsCandidateCost) {
    auto snapshot = nativeStepForecastSnapshot();
    snapshot.workloadFeatures = nativeStepInput(8193);
    snapshot.reportedWorkloadFeatures = snapshot.workloadFeatures;
    NaviSchedScheduleNode node(nullptr, std::move(snapshot), 1);
    const auto &features = node.getWorkloadFeatures();
    const auto smooth = node.applyLatencyResidual(
        NonLinearPrefillModelR::predict(features, node.getLatencyParameters()));
    ASSERT_GT(smooth, 0.0);
    ASSERT_LT(smooth, node.predictLatency(features));
    node.refreshExistingUnifiedLatency();
    EXPECT_DOUBLE_EQ(smooth, node.getExistingUnifiedLatencyMs());
    EXPECT_DOUBLE_EQ(0.0, smooth - node.getExistingUnifiedLatencyMs());
}

TEST(ExecutionLoadResourceTest, LegacyOptimizerBaselineKeepsItsOriginalPrediction) {
    auto snapshot = nativeStepForecastSnapshot();
    auto info = std::make_shared<NaviSchedNodeInfo>(*snapshot.info);
    info->latencyModelFingerprint = "legacy-3.7-model";
    snapshot.info = info;
    snapshot.workloadFeatures = nativeStepInput(8193);
    snapshot.reportedWorkloadFeatures = snapshot.workloadFeatures;
    NaviSchedScheduleNode node(nullptr, std::move(snapshot), 1);
    const auto original = node.predictLatency(node.getWorkloadFeatures());
    ASSERT_GT(original, 0.0);
    node.refreshExistingUnifiedLatency();
    EXPECT_DOUBLE_EQ(original, node.getExistingUnifiedLatencyMs());
}
}

namespace navi_sched {
namespace {
void initUntrackedNativeKVLoadContext(NaviSchedScheduleContext &context,
                                     const NaviSchedNodeConstPtr &source,
                                     size_t requestCount,
                                     bool declaredNative = true) {
    auto snapshot = nativeStepForecastSnapshot();
    auto info = std::make_shared<NaviSchedNodeInfo>(*snapshot.info);
    info->bizName = "kv_load_test";
    info->agentId = "agent_0";
    info->serverAddress = "127.0.0.1:19001";
    info->pdSeparated = true;
    if (!declaredNative) info->latencyModelFingerprint = R"({"framework":"sglang"})";
    snapshot.info = info;
    snapshot.workload = {};
    snapshot.workload.executionLoadTracked = false;
    NaviSchedLatencyFeatures empty;
    ASSERT_TRUE(empty.appendCapacitySplitAggregate(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    snapshot.workloadFeatures = empty;
    snapshot.reportedWorkloadFeatures = empty;
    snapshot.reservedWorkloadFeatures = empty;
    snapshot.workloadRunningCapacitySlotCount = 0;
    snapshot.kvPreparationEstimates.clear();
    snapshot.kvPreparationSnapshotMonoUs = 0;
    for (size_t i = 0; i < requestCount; ++i) {
        NaviSchedScheduleItem item;
        item.request.requestId = 201 + i;
        item.request.promptTokenCount = 128;
        item.request.blockSize = 32;
        context._requests.emplace_back(std::move(item));
    }
    context.setNodes({source}, {snapshot});
    ASSERT_TRUE(context.prepare());
}
}

TEST(ExecutionLoadResourceTest, NativeKVLoadClassificationSurvivesUntrackedWorkload) {
    auto source = std::make_shared<InitialKVLoadNode>();
    source->totalBlocks = {0, 3, 3};
    source->hbmBlocks = {0, 3, 1};
    NaviSchedScheduleContext context;
    initUntrackedNativeKVLoadContext(context, source, 3);
    ASSERT_EQ(8192, context._scheduleNodes[0].getNativeStepTokenBudget());
    ASSERT_FALSE(context._scheduleNodes[0]._workload.executionLoadTracked);
    ASSERT_TRUE(context.getCacheHitBlockCountsSnapshot(0));
    // Freeze the physical source before another cache update arrives.
    source->totalBlocks = {0, 0, 0};
    source->hbmBlocks = {0, 0, 0};
    for (size_t i = 0; i < 3; ++i)
        ASSERT_TRUE(context.selectNode(i, 0, "kv_load_test"));
    ASSERT_TRUE(context.prepareResult(context.getScheduleTimeUs()));
    const std::vector<bool> expected = {true, true, false};
    for (size_t i = 0; i < 3; ++i) {
        const auto &decision = context._decisions[i];
        ASSERT_TRUE(decision.executionLoadIncluded.has_value()) << i;
        EXPECT_EQ(expected[i], *decision.executionLoadIncluded);
        auto response = context.makeResponse(decision, i);
        ASSERT_TRUE(response);
        autil::DataBuffer wire;
        response->serialize(wire);
        wire.write(uint64_t{817263});
        MasterResponseToClientData decoded;
        decoded.deserialize(wire);
        EXPECT_EQ(decision.executionLoadIncluded, decoded.executionLoadIncluded);
        uint64_t sentinel = 0;
        wire.read(sentinel);
        EXPECT_EQ(817263u, sentinel);
    }
    EXPECT_EQ(1u, source->richCalls);
    EXPECT_EQ(0u, source->plainCalls);
    // Classification capability does not fabricate a fully tracked observation.
    EXPECT_FALSE(context._scheduleNodes[0]._workload.executionLoadTracked);
    EXPECT_TRUE(context._scheduleNodes[0].getKVPreparationProbabilityAudits().empty());
}

TEST(ExecutionLoadResourceTest, NativeKVLoadUnknownHbmCannotDefaultToReady) {
    auto source = std::make_shared<InitialKVLoadNode>();
    source->totalBlocks = {3};
    source->hbmBlocks = {-1};
    NaviSchedScheduleContext context;
    initUntrackedNativeKVLoadContext(context, source, 1);
    ASSERT_TRUE(context.selectNode(0, 0, "kv_load_test"));
    EXPECT_FALSE(context.prepareResult(context.getScheduleTimeUs()));
    EXPECT_FALSE(context._decisions[0].executionLoadIncluded.has_value());
}

TEST(ExecutionLoadResourceTest, NativeKVLoadClassificationKeepsUndeclaredModelBehavior) {
    auto source = std::make_shared<InitialKVLoadNode>();
    NaviSchedScheduleContext context;
    initUntrackedNativeKVLoadContext(context, source, 1, false);
    ASSERT_EQ(0, context._scheduleNodes[0].getNativeStepTokenBudget());
    ASSERT_TRUE(context.selectNode(0, 0, "kv_load_test"));
    ASSERT_TRUE(context.prepareResult(context.getScheduleTimeUs()));
    EXPECT_FALSE(context._decisions[0].executionLoadIncluded.has_value());
    EXPECT_EQ(0u, source->richCalls);
    EXPECT_EQ(1u, source->plainCalls);
}
}

namespace navi_sched {
TEST(ExecutionLoadResourceTest, NativeWorkloadCaptureDefaultsOffAndBoundsSamples) {
    NaviSchedWorkloadSnapshot workload;
    workload.nativePPState = std::make_shared<const NaviSchedNativePPState>();
    EXPECT_FALSE(NaviSchedScheduleReplayJson::captureNativeWorkload(workload, 20));
    EXPECT_FALSE(NaviSchedScheduleReplayJson::captureNativeWorkload(workload, 0, true));
    EXPECT_FALSE(NaviSchedScheduleReplayJson::captureNativeWorkload(workload, 19, true));
    EXPECT_FALSE(NaviSchedScheduleReplayJson::captureNativeWorkload(workload, 20, true, 32));
    const auto captured = NaviSchedScheduleReplayJson::captureNativeWorkload(workload, 20, true);
    ASSERT_TRUE(captured);
    EXPECT_EQ(std::string::npos, captured->find("forward_payload"));
    EXPECT_EQ(std::string::npos, captured->find("cached_block_hashes"));
    EXPECT_TRUE(NaviSchedScheduleReplayJson::captureNativeWorkload(workload, 40, true));
    workload.nativePPState.reset();
    EXPECT_FALSE(NaviSchedScheduleReplayJson::captureNativeWorkload(workload, 20, true));
}

TEST(ExecutionLoadResourceTest, NativeWorkloadCapturePreservesQueueAndCompletionStates) {
    NaviSchedWorkloadSnapshot workload;
    workload.revision = 321;
    workload.eventTimeUs = 123456;
    workload.nativePPObservationAgeUs = 765;
    workload.executionLoadTracked = true;
    workload.admittedBatch.scheduleBatchId = 881;
    workload.admittedBatch.localBatchId = 71;
    workload.admittedBatch.requestCount = 2;
    workload.admittedBatch.requestIds = {502, 501};
    auto native = std::make_shared<NaviSchedNativePPState>();
    native->observationId = 991;
    auto &request = native->requests.emplace_back();
    request.requestId = 501;
    request.promptTokens = 20000;
    request.computedTokens = 4096;
    request.queue = static_cast<NativePPQueue>(1);
    request.queueIndex = 3;
    request.phase = static_cast<NativePPRequestPhase>(2);
    auto &result = native->results.emplace_back();
    result.schedulerOutputId = 192;
    result.loadRequestIds = {501, 502};
    result.executeReplyReady = true;
    result.resultReady = false;
    native->pendingDecision.emplace();
    native->pendingDecision->schedulerOutputId = 193;
    native->pendingDecision->loadRequestIds = {503};
    workload.nativePPState = native;
    const auto captured = NaviSchedScheduleReplayJson::captureNativeWorkload(workload, 20, true);
    ASSERT_TRUE(captured);
    NaviSchedWorkloadSnapshot decoded;
    NaviSchedScheduleReplayJson::workloadFromJsonString(*captured, decoded);
    EXPECT_EQ(workload.revision, decoded.revision);
    EXPECT_EQ(workload.eventTimeUs, decoded.eventTimeUs);
    EXPECT_EQ(765, decoded.nativePPObservationAgeUs);
    EXPECT_TRUE(decoded.executionLoadTracked);
    EXPECT_EQ(881, decoded.admittedBatch.scheduleBatchId);
    EXPECT_EQ(71, decoded.admittedBatch.localBatchId);
    EXPECT_EQ(2, decoded.admittedBatch.requestCount);
    EXPECT_EQ(workload.admittedBatch.requestIds, decoded.admittedBatch.requestIds);
    ASSERT_TRUE(decoded.nativePPState);
    ASSERT_EQ(1u, decoded.nativePPState->requests.size());
    const auto &copy = decoded.nativePPState->requests[0];
    EXPECT_EQ(request.requestId, copy.requestId);
    EXPECT_EQ(request.computedTokens, copy.computedTokens);
    EXPECT_EQ(request.queue, copy.queue);
    EXPECT_EQ(request.queueIndex, copy.queueIndex);
    EXPECT_EQ(request.phase, copy.phase);
    ASSERT_EQ(1u, decoded.nativePPState->results.size());
    EXPECT_EQ(result.loadRequestIds, decoded.nativePPState->results[0].loadRequestIds);
    EXPECT_TRUE(decoded.nativePPState->results[0].executeReplyReady);
    EXPECT_FALSE(decoded.nativePPState->results[0].resultReady);
    ASSERT_TRUE(decoded.nativePPState->pendingDecision);
    EXPECT_EQ(193u, decoded.nativePPState->pendingDecision->schedulerOutputId);
    EXPECT_EQ(std::vector<uint64_t>({503}), decoded.nativePPState->pendingDecision->loadRequestIds);
    EXPECT_EQ(captured, NaviSchedScheduleReplayJson::captureNativeWorkload(workload, 20, true));
}
}
