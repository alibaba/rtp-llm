#include "aios/navi_sched/resource/AgentReportR.h"
#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/kernel/AgentLatencyKernel.h"
#include "navi/tester/KernelTesterBuilder.h"
#include "aios/navi_sched/scheduler/MasterScheduleContextKernel.h"
#include "aios/navi_sched/scheduler/CostSchedulerGraph.h"
#include "aios/navi_sched/scheduler/CostSchedulerKernelInternal.h"
#include <limits>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <fstream>
#include <iomanip>
#include "autil/DataBuffer.h"
#include <cmath>
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/resource/MasterLatencyModelR.h"
#include "aios/navi_sched/data/AgentUpdateToMasterData.h"
#include "aios/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "autil/legacy/jsonizable.h"
#include "autil/legacy/json.h"
#include "aios/navi_sched/resource/AgentHistoryR.h"
#include "aios/navi_sched/resource/AgentLatencyInputR.h"
#include "aios/navi_sched/resource/AgentBatchR.h"
#include "aios/navi_sched/service/NaviSchedAgent.h"
#include "aios/navi_sched/service/NaviSchedAgentImpl.h"
#include "autil/TimeUtility.h"
#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/scheduler/ScheduleContext.h"
#include "aios/navi_sched/scheduler/ScheduleContextInternal.h"
#include "navi/tester/NaviResourceHelper.h"
#include "unittest/unittest.h"
#include "aios/navi_sched/test/DiscreteSearchBaseline.h"

namespace navi_sched {
namespace {
TPUPrefillExecutionPolicyConfig typedRuntimePolicy()
{
    TPUPrefillExecutionPolicyConfig policy;
    policy.pcpSize = 3;
    policy.interleaveSize = 32;
    policy.compileSizes = {64, 128, 256};
    policy.maxScheduledTokens = 768;
    policy.longPrefillTokenThreshold = 768;
    policy.hashBlockSize = 16;
    policy.schedulerBlockSize = 32;
    policy.mambaHasPrefillCheckpointBlocks = true;
    policy.asyncKVLookupEnabled = true;
    policy.asyncKVLookupHashBlockSize = 16;
    return policy;
}
} // namespace

class GeometryIntegrationTest : public TESTBASE {};

TEST_F(GeometryIntegrationTest, testSharedBatchQueueObservationsRemainTrainable)
{
    // A shared execution span is unsuitable for execution-model training, but
    // the batch's ready-to-first-execution wait is still directly observed.
    for (const int scenario : {0, 1, 2}) {
        const bool windows = scenario != 0;
        const bool started = scenario != 2;
        navi::NaviResourceHelper helper;
        ASSERT_TRUE(helper.addExternalRes(std::make_shared<AgentReportBridgeR>()));
        auto inputs = std::make_shared<AgentLatencyInputR>();
        if (windows) inputs->enableObservedExecutionWindowTraining();
        ASSERT_TRUE(helper.addExternalRes(inputs));
        auto batches = std::make_shared<AgentBatchR>();
        ASSERT_TRUE(helper.addExternalRes(batches));
        ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID,
                                  R"({"fit_batch_size":1,"warmup_sample_count":0})"));
        ASSERT_TRUE(helper.config(
            NonLinearPrefillModelR::RESOURCE_ID,
            R"({"pd_separated":true,"use_batch_count_feature":true,"use_capacity_split_feature":false,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":false,"execution_cache_source_enabled":true})"));
        ASSERT_TRUE(helper.config(
            AgentReportR::RESOURCE_ID,
            R"({"server_ip":"127.0.0.1","server_port":0,"pd_separated":true,"unified_latency_prediction":false})"));
        auto report = helper.getOrCreateResPtr<AgentReportR>();
        auto model = helper.getOrCreateResPtr<LatencyModelR>();
        ASSERT_TRUE(report && model);
        report->_controlConfig.executionGeometryEnabled = true;
        report->_controlConfig.prefillExecutionPolicy = typedRuntimePolicy();
        navi::KernelTesterBuilder builder;
        auto tester = builder.kernel("navi_sched.agent_latency_k")
                          .output("save")
                          .output("finished")
                          .attrs(std::string("{\"input_pipe_generation\":") +
                                 std::to_string(inputs->beginInputPipeGeneration()) + "}")
                          .resource(helper.getResourceMap())
                          .build();
        ASSERT_TRUE(tester);
        ASSERT_EQ(navi::EC_NONE, tester->getErrorCode());
        auto kernel = dynamic_cast<AgentLatencyKernel *>(tester->getKernel());
        ASSERT_TRUE(kernel);
        const auto modelRevision = model->getSnapshot()->revision;
        const auto fitStep = kernel->_queueFitStep;
        NaviSchedRequestStats request;
        request.requestId = 101;
        request.totalTokenCount = 512;
        request.blockSize = 16;
        request.executionSharedPrefixBoundary = 0;
        request.executionExternalReadTokens = 0;
        request.predictedExecutionLatencyMs = 3.0;
        NaviSchedBatchPtr batch;
        ASSERT_TRUE(batches->addRequest(
            request, 81, 1, [&](const NaviSchedBatchPtr &value) { batch = value; }));
        ASSERT_TRUE(batch);
        const auto readyUs = batch->readyTimeUs;
        ASSERT_TRUE(inputs->submitBatchReady(batch, 0, 1, 0, readyUs));
        ASSERT_TRUE(tester->compute());
        if (started) {
            AgentBatchR::ExecutionState state;
            ASSERT_TRUE(
                batches->startBatch(batch->batchId, {101}, readyUs + 2000, state));
            ASSERT_TRUE(
                inputs->submitBatchStarted(batch->batchId, 0, 0, 1, readyUs + 2000));
            ASSERT_TRUE(tester->compute());
        }
        AgentLatencyInput finished;
        finished.type = AgentLatencyInputType::BATCH_FINISHED;
        finished.batchId = batch->batchId;
        finished.eventTimeUs = readyUs + 5000;
        finished.latencyMs = 3.0;
        finished.backendQueueLatencyMs = 2.0;
        finished.latencySampleValid = false;
        finished.latencyValidationValid = false;
        finished.completedRequestCount = 1;
        finished.queries = {request};
        finished.localBatch = batch->makeLocalBatchInfo();
        ASSERT_TRUE(inputs->submit(std::move(finished)));
        ASSERT_TRUE(tester->compute());
        EXPECT_EQ(fitStep + (windows && started ? 1 : 0), kernel->_queueFitStep);
        EXPECT_EQ(modelRevision, model->getSnapshot()->revision);
        EXPECT_TRUE(kernel->_batchLatencyUs.empty());
        EXPECT_TRUE(kernel->_batchQueueFeatures.empty());
        EXPECT_TRUE(kernel->_runningBatches.empty());
    }
}

TEST_F(GeometryIntegrationTest,
       testExecutionWindowsReplaceTrainingWithoutChangingValidation)
{
    AgentLatencyInputR inputs;
    EXPECT_TRUE(inputs.shouldTrainInput(AgentLatencyInputType::BATCH_FINISHED));
    EXPECT_TRUE(inputs.shouldTrainInput(AgentLatencyInputType::OBSERVED_BATCH_FINISHED));
    EXPECT_FALSE(
        inputs.shouldTrainInput(AgentLatencyInputType::OBSERVED_EXECUTION_SAMPLE));
    inputs.enableObservedExecutionWindowTraining();
    EXPECT_FALSE(inputs.shouldTrainInput(AgentLatencyInputType::BATCH_FINISHED));
    EXPECT_FALSE(inputs.shouldTrainInput(AgentLatencyInputType::OBSERVED_BATCH_FINISHED));
    EXPECT_TRUE(
        inputs.shouldTrainInput(AgentLatencyInputType::OBSERVED_EXECUTION_SAMPLE));
    EXPECT_TRUE(AgentLatencyKernel::shouldRecordLatencyValidation(
        AgentLatencyInputType::BATCH_FINISHED, true, true));
    EXPECT_FALSE(AgentLatencyKernel::shouldRecordLatencyValidation(
        AgentLatencyInputType::OBSERVED_EXECUTION_SAMPLE, true, true));
}

TEST_F(GeometryIntegrationTest, testSharedExecutionSampleIsTrainingOnly)
{
    AgentLatencyInputR inputs;
    inputs.enableObservedExecutionWindowTraining();
    NaviSchedRequestStats first;
    first.requestId = 10;
    first.totalTokenCount = 12288;
    first.cacheHitTokenCount = 8192;
    first.executionExternalReadTokens = 8192;
    NaviSchedRequestStats second;
    second.requestId = 20;
    second.totalTokenCount = 8192;
    second.cacheHitTokenCount = 4096;
    second.executionExternalReadTokens = 0;
    ASSERT_TRUE(
        inputs.submitObservedExecutionSample({first, second}, 3.0, 1, 8192, 1000));
    EXPECT_EQ(0, inputs.getStats().submittedFinishedRequestCount);
    EXPECT_EQ(0, inputs.getStats().submittedAbortedRequestCount);
    std::vector<AgentLatencyInput> taken;
    inputs.take(taken);
    ASSERT_EQ(1, taken.size());
    const auto &input = taken.front();
    EXPECT_EQ(AgentLatencyInputType::OBSERVED_EXECUTION_SAMPLE, input.type);
    EXPECT_EQ(0, input.completedRequestCount);
    EXPECT_FALSE(AgentLatencyKernel::shouldRecordLatencyValidation(
        input.type, input.latencySampleValid, input.latencyValidationValid));
    const auto sample = input.makeLatencySample();
    EXPECT_EQ(1, sample.features.getExecutionStepCount());
    EXPECT_EQ(8192, sample.features.getGlobalPaddedTokens());
    EXPECT_EQ(12288, sample.features.getCacheHitTokenCount());
    EXPECT_EQ(8192, sample.features.getCacheMissTokenCount());
    EXPECT_TRUE(sample.features.hasExecutionCacheSource());
    EXPECT_DOUBLE_EQ(3.0, sample.latencyMs);
}

TEST_F(GeometryIntegrationTest, testSharedExecutionSampleRejectsIncompleteFacts)
{
    AgentLatencyInputR inputs;
    inputs.enableObservedExecutionWindowTraining();
    NaviSchedRequestStats request;
    request.requestId = 10;
    request.totalTokenCount = 8192;
    request.cacheHitTokenCount = 4096;
    request.executionExternalReadTokens = 4096;
    EXPECT_FALSE(inputs.submitObservedExecutionSample({request}, 3.0, 0, 8192, 1000));
    EXPECT_FALSE(inputs.submitObservedExecutionSample({request}, 3.0, 1, 0, 1000));
    EXPECT_FALSE(inputs.submitObservedExecutionSample({request}, 3.0, 1, 2048, 1000));
    EXPECT_FALSE(
        inputs.submitObservedExecutionSample({request, request}, 3.0, 1, 8192, 1000));
    request.executionExternalReadTokens = -1;
    EXPECT_FALSE(inputs.submitObservedExecutionSample({request}, 3.0, 1, 8192, 1000));
    EXPECT_EQ(0, inputs.getPendingInputCount());
}

TEST_F(GeometryIntegrationTest, testSource34LearningHotUpdateAndRollbackPreserveState)
{
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID,
                              R"({"fit_batch_size":1,"warmup_sample_count":0})"));
    ASSERT_TRUE(helper.config(
        AgentSchedulerConfigR::RESOURCE_ID,
        R"({"config_json":"{\"latency_fit_batch_size\":1,\"latency_current_sample_weight\":2,\"latency_online_learning_rate_scale\":8,\"residual_learning_rate\":0}"})"));
    ASSERT_TRUE(helper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        R"({"pd_separated":true,"use_batch_count_feature":true,"use_capacity_split_feature":false,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":true,"execution_cache_source_enabled":true})"));
    auto model = helper.getOrCreateResPtr<LatencyModelR>();
    auto history = helper.getOrCreateResPtr<AgentHistoryR>();
    auto configResource = helper.getOrCreateResPtr<AgentSchedulerConfigR>();
    ASSERT_TRUE(model && history && configResource);
    ASSERT_EQ(34, model->getSnapshot()->binaryVersion);
    for (size_t index = 0; index < 2; ++index) {
        NaviSchedLatencySample sample;
        sample.localBatchId = index + 1;
        sample.eventTimeUs = (index + 1) * 1000;
        ASSERT_TRUE(sample.features.appendBatchCount(1));
        sample.features.appendRequest(index ? 6144 : 0, index ? 18144 : 2350);
        ASSERT_TRUE(
            sample.features.setExecutionGeometry(index ? 2 : 1, index ? 16384 : 4096));
        ASSERT_TRUE(
            sample.features.setExecutionCacheSource(index ? 1 : 0, index ? 6144 : 0));
        sample.latencyMs = index ? 400.0 : 107.0;
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
        bool saveNeeded = false;
        ASSERT_TRUE(model->recordSample(sample, updated, saveNeeded));
    }
    const auto before = model->getSnapshot();
    const auto optimizerBefore = model->getOptimizerState();
    ASSERT_EQ(21u, optimizerBefore.adamMoment1.size());
    ASSERT_GT(optimizerBefore.step, 0u);
    ASSERT_TRUE(std::any_of(optimizerBefore.adamMoment1.begin(),
                            optimizerBefore.adamMoment1.end(),
                            [](double value) { return value != 0.0; }));
    autil::DataBuffer originalHistory;
    ASSERT_TRUE(history->_sampler.serializeWithExecutionCacheSource(originalHistory));
    const std::string historyBytes(originalHistory.getData(),
                                   originalHistory.getDataLen());
    const auto sessionBefore = history->getSessionSampleCount();
    const auto original = configResource->get().config;
    auto candidate = original;
    candidate.latencyCurrentSampleWeight = 16.0;
    candidate.latencyOnlineLearningRateScale = 1.0;
    for (const auto &config : {candidate, original}) {
        const auto json = AgentSchedulerConfigR::toJson(config);
        AgentSchedulerConfigR::Snapshot applied;
        std::string error;
        ASSERT_EQ(AgentSchedulerConfigR::ApplyResult::APPLIED,
                  configResource->apply(configResource->get().generation + 1,
                                        AgentSchedulerConfigR::hashJson(json), json,
                                        applied, error))
            << error;
        // These opt-in mutations exist only in this test binary. Negative
        // runs retain all assertions to prove a reset cannot pass unnoticed.
        const char *mutation = std::getenv("NAVI_TEST_LEARNING_STATE_MUTANT");
        if (config.latencyCurrentSampleWeight == 16.0 && mutation) {
            if (std::string(mutation) == "optimizer_reset") {
                auto cleared = optimizerBefore;
                cleared.step = 0;
                std::fill(cleared.adamMoment1.begin(), cleared.adamMoment1.end(), 0.0);
                std::fill(cleared.adamMoment2.begin(), cleared.adamMoment2.end(), 0.0);
                ASSERT_TRUE(
                    model->_nonLinearPrefillModelR->restoreOptimizerState(cleared));
                auto reset = std::make_shared<NaviSchedLatencyModelSnapshot>(*before);
                reset->optimizerState = cleared;
                model->publishSnapshot(reset);
            } else if (std::string(mutation) == "history_reset") {
                history->_sampler.clear();
            }
        }
        history->applyRuntimeConfig(applied.config);
        model->_nonLinearPrefillModelR->applyRuntimeConfig(applied.config);
        ASSERT_TRUE(model->applyRuntimeConfig(applied.config));
        const auto after = model->getSnapshot();
        const auto optimizer = model->getOptimizerState();
        EXPECT_EQ(before->binaryVersion, after->binaryVersion);
        EXPECT_EQ(before->revision, after->revision);
        EXPECT_EQ(before->parameters, after->parameters);
        EXPECT_EQ(before->residualParameters, after->residualParameters);
        EXPECT_EQ(before->residualLastUpdateTimeUs, after->residualLastUpdateTimeUs);
        EXPECT_EQ(before->sessionSampleCount, after->sessionSampleCount);
        EXPECT_EQ(before->overloadPredictionTrusted, after->overloadPredictionTrusted);
        EXPECT_EQ(optimizerBefore.step, optimizer.step);
        EXPECT_EQ(optimizerBefore.adamMoment1, optimizer.adamMoment1);
        EXPECT_EQ(optimizerBefore.adamMoment2, optimizer.adamMoment2);
        EXPECT_EQ(sessionBefore, history->getSessionSampleCount());
        autil::DataBuffer retainedHistory;
        ASSERT_TRUE(history->_sampler.serializeWithExecutionCacheSource(retainedHistory));
        EXPECT_EQ(historyBytes,
                  std::string(retainedHistory.getData(), retainedHistory.getDataLen()));
        EXPECT_DOUBLE_EQ(config.latencyOnlineLearningRateScale,
                         model->_fixedOnlineLearningRateScale);
        EXPECT_DOUBLE_EQ(config.latencyCurrentSampleWeight,
                         configResource->get().config.latencyCurrentSampleWeight);
    }
}

TEST_F(GeometryIntegrationTest, testGeometryLatencySampleRejectedBeforeHistoryAppend)
{
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID,
                              R"({"fit_batch_size":1,"warmup_sample_count":0})"));
    ASSERT_TRUE(helper.config(AgentSchedulerConfigR::RESOURCE_ID,
                              R"({"config_json":"{\"latency_fit_batch_size\":1}"})"));
    ASSERT_TRUE(helper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        R"({"pd_separated":true,"use_batch_count_feature":true,"use_capacity_split_feature":false,"execution_geometry_enabled":true})"));
    auto model = helper.getOrCreateResPtr<LatencyModelR>();
    auto history = helper.getOrCreateResPtr<AgentHistoryR>();
    ASSERT_TRUE(model);
    ASSERT_TRUE(history);
    ASSERT_TRUE(model->_nonLinearPrefillModelR->usesExecutionGeometry());
    const auto initial = model->getSnapshot();
    ASSERT_TRUE(initial);
    ASSERT_EQ(NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION, initial->binaryVersion);
    const auto initialSession = history->getSessionSampleCount();
    const auto initialOptimizer = model->getOptimizerState();
    const auto initialFitCount = model->_onlineFitCount.load();
    const auto initialRampFinished = model->_rampFinished;
    NaviSchedLatencySample sample;
    ASSERT_TRUE(sample.features.appendBatchCount(1));
    sample.features.appendRequest(0, 512);
    sample.latencyMs = 200.0;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated = initial;
    bool saveNeeded = true;
    EXPECT_FALSE(model->recordSample(sample, updated, saveNeeded));
    EXPECT_FALSE(updated);
    EXPECT_FALSE(saveNeeded);
    EXPECT_TRUE(history->getHistory()->empty());
    EXPECT_EQ(initialSession, history->getSessionSampleCount());
    EXPECT_EQ(initial->revision, model->getSnapshot()->revision);
    EXPECT_EQ(initial->sessionSampleCount, model->getSnapshot()->sessionSampleCount);
    EXPECT_EQ(initialOptimizer.step, model->getOptimizerState().step);
    EXPECT_EQ(initialFitCount, model->_onlineFitCount.load());
    EXPECT_EQ(initialRampFinished, model->_rampFinished);
    // One incomplete observation must not poison the next valid training batch.
    ASSERT_TRUE(sample.features.setExecutionGeometry(1, 4096));
    ASSERT_TRUE(model->recordSample(sample, updated, saveNeeded));
    ASSERT_TRUE(updated);
    ASSERT_EQ(1u, history->getHistory()->size());
    EXPECT_TRUE(history->getHistory()->front().features.hasExecutionGeometry());
    EXPECT_EQ(initialSession + 1, history->getSessionSampleCount());
    EXPECT_GT(updated->revision, initial->revision);
    EXPECT_EQ(initial->sessionSampleCount + 1, updated->sessionSampleCount);
    EXPECT_GT(model->getOptimizerState().step, initialOptimizer.step);
    EXPECT_EQ(initialFitCount + 1, model->_onlineFitCount.load());
}

namespace {
class GeometryMissingPolicyNode : public NaviSchedNode {
public:
    NaviSchedNodeInfo getInfo() const override { return {}; }
    bool getScheduleSnapshot(const std::string &, NaviSchedNodeSnapshot &) const override
    {
        return false;
    }
    int64_t getMatchedBlockCount(const NaviSchedBlockHashes &) const override
    {
        return 0;
    }
    void getMatchedBlockCounts(const std::vector<NaviSchedScheduleItem> &, int64_t *,
                               int64_t * = nullptr) const override
    {
    }
    bool getExecutionCacheInputs(const NaviSchedScheduleRequest &, int64_t &hit,
                                 int64_t &boundary) const override
    {
        hit = 0;
        boundary = 0;
        return true;
    }
    bool getExecutionCacheInputs(
        const std::vector<NaviSchedScheduleItem> &requests,
        std::vector<TPUPrefillRequestInput> &inputs,
        int64_t *externalSuffixBlockCounts = nullptr,
        ::navi_sched::TPUPrefillCacheObservationBatch * = nullptr) const override
    {
        inputs.resize(requests.size());
        for (size_t i = 0; i < requests.size(); ++i) {
            if (externalSuffixBlockCounts) {
                externalSuffixBlockCounts[i] = 0;
            }
            inputs[i].promptTokens = requests[i].request.promptTokenCount;
            inputs[i].externalReadTokens = 0;
            if (!getExecutionCacheInputs(requests[i].request, inputs[i].computedTokens,
                                         inputs[i].sharedPrefixBoundary)) {
                return false;
            }
        }
        return true;
    }
};
}

namespace {
class ConditionalQueueGeometryNode : public GeometryMissingPolicyNode {
public:
    using GeometryMissingPolicyNode::getExecutionCacheInputs;
    mutable size_t batchCalls = 0;
    void getMatchedBlockCounts(const std::vector<NaviSchedScheduleItem> &requests,
                               int64_t *counts,
                               int64_t *external = nullptr) const override
    {
        ++batchCalls;
        for (size_t i = 0; i < requests.size(); ++i) {
            counts[i] = requests[i].request.requestId % 2;
            if (external) {
                external[i] = 0;
            }
        }
    }
    bool getExecutionCacheInputs(const NaviSchedScheduleRequest &request, int64_t &hit,
                                 int64_t &boundary) const override
    {
        hit = request.requestId % 2 && request.promptTokenCount > request.blockSize
                  ? request.blockSize
                  : 0;
        boundary = 0;
        return true;
    }
};
}

TEST_F(GeometryIntegrationTest, testMissingPolicyRejectsFinalPrepareResult)
{
    auto source = std::make_shared<GeometryMissingPolicyNode>();
    NaviSchedNodeSnapshot snapshot;
    auto info = std::make_shared<NaviSchedNodeInfo>();
    info->pdSeparated = true;
    snapshot.info = info;
    auto model = std::make_shared<NaviSchedLatencyModel>();
    model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    model->binaryVersion = NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION;
    NonLinearPrefillModelR initialModel;
    initialModel._pdSeparated = true;
    model->parameters = initialModel.getInitialParameters();
    snapshot.latencyModel = model;
    // No runtime policy was supplied: selecting this node must not emit NaN.
    NaviSchedScheduleContext context;
    NaviSchedScheduleItem item;
    item.request.requestId = 1;
    item.request.promptTokenCount = 200;
    item.request.blockSize = 1;
    context._requests = {item};
    context.setNodes({source}, {snapshot});
    ASSERT_TRUE(context.prepare());
    EXPECT_FALSE(context.getNode(0).hasValidLatencyModel());
    ASSERT_TRUE(context.selectNode(0, 0, "geometry_missing_policy_test"));
    EXPECT_FALSE(context.prepareResult(context.getScheduleTimeUs()));
}

TEST_F(GeometryIntegrationTest, testReplayJsonExecutionGeometryRoundtripAndLegacyOmission)
{
    NaviSchedScheduleReplay replay;
    replay.nodes.emplace_back();
    auto &features = replay.nodes[0].workloadFeatures;
    ASSERT_TRUE(features.appendBatchCount(1));
    features.appendRequest(128, 512);
    const auto legacy = NaviSchedScheduleReplayJson::toJsonString(replay);
    EXPECT_EQ(std::string::npos, legacy.find("execution_step_count"));
    EXPECT_EQ(std::string::npos, legacy.find("global_padded_tokens"));
    NaviSchedScheduleReplay decoded;
    NaviSchedScheduleReplayJson::fromJsonString(legacy, decoded);
    ASSERT_EQ(1u, decoded.nodes.size());
    EXPECT_FALSE(decoded.nodes[0].workloadFeatures.hasExecutionGeometry());
    EXPECT_EQ(legacy, NaviSchedScheduleReplayJson::toJsonString(decoded));
    ASSERT_TRUE(features.setExecutionGeometry(2, 12288));
    const auto geometry = NaviSchedScheduleReplayJson::toJsonString(replay);
    EXPECT_NE(std::string::npos, geometry.find("execution_step_count"));
    EXPECT_NE(std::string::npos, geometry.find("global_padded_tokens"));
    NaviSchedScheduleReplayJson::fromJsonString(geometry, decoded);
    ASSERT_EQ(1u, decoded.nodes.size());
    const auto &restored = decoded.nodes[0].workloadFeatures;
    EXPECT_EQ(2, restored.getExecutionStepCount());
    EXPECT_EQ(12288, restored.getGlobalPaddedTokens());
    EXPECT_EQ(128, restored.getCacheHitTokenCount());
    EXPECT_EQ(512, restored.getCacheMissTokenCount());
    EXPECT_EQ(geometry, NaviSchedScheduleReplayJson::toJsonString(decoded));
    features.clearExecutionGeometry();
    EXPECT_EQ(legacy, NaviSchedScheduleReplayJson::toJsonString(replay));
    // Mutate only the geometry pair inside workload_features.
    auto invalidJson = [&](bool includeSteps, int64_t steps, bool includePadding,
                           int64_t padding) {
        autil::legacy::json::JsonMap root;
        autil::legacy::FromJsonString(root, geometry);
        auto nodes =
            autil::legacy::AnyCast<autil::legacy::json::JsonArray>(root.at("nodes"));
        auto node = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(nodes[0]);
        auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
            node.at("workload_features"));
        fields.erase("execution_step_count");
        fields.erase("global_padded_tokens");
        if (includeSteps) {
            fields["execution_step_count"] = steps;
        }
        if (includePadding) {
            fields["global_padded_tokens"] = padding;
        }
        node["workload_features"] = fields;
        nodes[0] = node;
        root["nodes"] = nodes;
        return autil::legacy::ToJsonString(root, true);
    };
    EXPECT_ANY_THROW(NaviSchedScheduleReplayJson::fromJsonString(
        invalidJson(true, 2, false, 0), decoded));
    EXPECT_ANY_THROW(NaviSchedScheduleReplayJson::fromJsonString(
        invalidJson(false, 0, true, 12288), decoded));
    EXPECT_ANY_THROW(NaviSchedScheduleReplayJson::fromJsonString(
        invalidJson(true, 0, true, 0), decoded));
    EXPECT_ANY_THROW(NaviSchedScheduleReplayJson::fromJsonString(
        invalidJson(true, -1, true, 12288), decoded));
    EXPECT_ANY_THROW(NaviSchedScheduleReplayJson::fromJsonString(
        invalidJson(true, 2, true, -1), decoded));
}

TEST_F(GeometryIntegrationTest, testExecutionPolicyControlSerialization)
{
    AgentControlToMasterData control;
    control.executionGeometryEnabled = true;
    control.projectedExecutionGeometryEnabled = true;
    control.prefillExecutionPolicy = typedRuntimePolicy();
    control.prefillExecutionPolicy.fixedPrefillChunkPlan = true;
    control.prefillExecutionPolicy.mambaBlockAlignedSplit = false;
    control.prefillExecutionPolicy.attentionScheduleCapacity = 1024;
    control.prefillExecutionPolicy.attentionQueryTile = 16;
    control.prefillExecutionPolicy.attentionKvTile = 64;
    control.prefillExecutionPolicy.prefillLookahead = 4;
    control.prefillExecutionPolicy.maxInputTokens = 16384;
    control.prefillExecutionPolicy.draftSlotsPerRequest = 64;
    autil::DataBuffer wire;
    control.serialize(wire);
    AgentControlToMasterData output;
    output.deserialize(wire);
    EXPECT_TRUE(output.executionGeometryEnabled);
    EXPECT_TRUE(output.projectedExecutionGeometryEnabled);
    EXPECT_EQ(control.prefillExecutionPolicy.compileSizes,
              output.prefillExecutionPolicy.compileSizes);
    EXPECT_EQ(768, output.prefillExecutionPolicy.maxScheduledTokens);
    EXPECT_EQ(16, output.prefillExecutionPolicy.asyncKVLookupHashBlockSize);
    EXPECT_TRUE(output.prefillExecutionPolicy.fixedPrefillChunkPlan);
    EXPECT_FALSE(output.prefillExecutionPolicy.mambaBlockAlignedSplit);
    EXPECT_EQ(1024, output.prefillExecutionPolicy.attentionScheduleCapacity);
    EXPECT_EQ(16, output.prefillExecutionPolicy.attentionQueryTile);
    EXPECT_EQ(64, output.prefillExecutionPolicy.attentionKvTile);
    EXPECT_EQ(4, output.prefillExecutionPolicy.prefillLookahead);
    EXPECT_EQ(16384, output.prefillExecutionPolicy.maxInputTokens);
    EXPECT_EQ(64, output.prefillExecutionPolicy.draftSlotsPerRequest);
    EXPECT_EQ(0u, wire.getDataLen());
    control.executionGeometryEnabled = false;
    control.projectedExecutionGeometryEnabled = false;
    control.serialize(wire);
    output.deserialize(wire);
    EXPECT_FALSE(output.executionGeometryEnabled);
    EXPECT_TRUE(output.prefillExecutionPolicy.compileSizes.empty());
    EXPECT_EQ(0, output.prefillExecutionPolicy.maxScheduledTokens);
}

TEST_F(GeometryIntegrationTest, testMasterReportRequiresSwitchAndBinaryVersionAgreement)
{
    // The typed control fields decide the mode; fingerprint is opaque.
    for (const bool enabled : {false, true}) {
        for (const int64_t version :
             {NonLinearPrefillModelR::PD_BINARY_VERSION,
              NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION,
              NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION,
              NonLinearPrefillModelR::CACHE_SOURCE_BINARY_VERSION}) {
            MasterPendingCacheR pending;
            MasterLatencyModelR pool;
            MasterStateR master;
            master._pendingCacheR__ = &pending;
            master._masterLatencyModelR__ = &pool;
            auto state = master.createAgent("geometry_switch_agent", "model_biz",
                                            "127.0.0.1:1000", 0);
            ASSERT_TRUE(state);
            auto update = std::make_shared<AgentUpdateToMasterData>();
            update->event = NaviSchedAgentEvent::AGENT_HANDSHAKE;
            update->eventTimeUs = 1;
            update->control = std::make_shared<AgentControlToMasterData>();
            update->control->agentStartTimeUs = 1;
            update->control->pdSeparated = true;
            update->control->serverInfo.ip = "127.0.0.1";
            update->control->serverInfo.port = 18000;
            update->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
            update->requestEvent->inflightStatsRevision = 1;
            update->latencyModelFingerprint = "opaque-runtime-policy-id";
            update->control->executionGeometryEnabled = enabled;
            update->control->projectedExecutionGeometryEnabled =
                enabled &&
                NonLinearPrefillModelR::usesProjectedGeometryForVersion(version);
            update->control->prefillExecutionPolicy = typedRuntimePolicy();
            update->latency = std::make_shared<AgentLatencyToMasterData>();
            auto &latency = *update->latency;
            latency.formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
            latency.binaryVersion = version;
            latency.modelRevision = 1;
            NonLinearPrefillModelR initialModel;
            initialModel._pdSeparated = true;
            initialModel.setExecutionGeometryEnabled(
                NonLinearPrefillModelR::usesGeometryForVersion(version));
            initialModel.setProjectedExecutionGeometryEnabled(
                NonLinearPrefillModelR::usesProjectedGeometryForVersion(version));
            initialModel.setExecutionCacheSourceEnabled(
                NonLinearPrefillModelR::usesCacheSourceForVersion(version));
            latency.parameters = initialModel.getInitialParameters();
            latency.optimizerState.adamMoment1.assign(latency.parameters.size(), 0.0);
            latency.optimizerState.adamMoment2.assign(latency.parameters.size(), 0.0);
            latency.optimizerState.step = 1;
            latency.zeroPoint = NonLinearPrefillModelR::calculateZeroPoint(
                latency.parameters, nullptr, version);
            ASSERT_TRUE(update->valid());
            ASSERT_TRUE(MasterStateR::validLatencyModel(update->latency));
            const bool agrees =
                enabled == NonLinearPrefillModelR::usesGeometryForVersion(version);
            EXPECT_EQ(agrees, master.reportAgent(state, update));
            if (agrees) {
                EXPECT_TRUE(state->getInfo().online);
                EXPECT_EQ(version, state->getInfo().latencyBinaryVersion);
            } else {
                EXPECT_FALSE(state->getInfo().online);
                EXPECT_EQ(0, state->getInfo().latencyModelRevision);
            }
        }
    }
}

TEST_F(GeometryIntegrationTest,
       testGeometryOptimizerUsesContinuousGradientsAndExactFinalScores)
{
    auto makeContext = [](int64_t version) {
        auto context = std::make_shared<NaviSchedScheduleContext>();
        std::vector<NaviSchedNodeConstPtr> nodes;
        std::vector<NaviSchedNodeSnapshot> snapshots;
        for (size_t i = 0; i < 2; ++i) {
            nodes.emplace_back(std::make_shared<GeometryMissingPolicyNode>());
            NaviSchedNodeSnapshot snapshot;
            auto info = std::make_shared<NaviSchedNodeInfo>();
            info->pdSeparated = true;
            info->agentId = "geometry_optimizer_" + std::to_string(i);
            snapshot.info = info;
            auto model = std::make_shared<NaviSchedLatencyModel>();
            model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
            model->binaryVersion = version;
            NonLinearPrefillModelR initial;
            initial._pdSeparated = true;
            model->parameters = initial.getInitialParameters();
            model->parameters[3] *= 1.0 + i;
            snapshot.latencyModel = model;
            TPUPrefillExecutionPolicyConfig policy;
            policy.pcpSize = 8;
            policy.interleaveSize = 256;
            policy.compileSizes = {512, 1024, 2048, 4096};
            policy.maxScheduledTokens = 32768;
            policy.longPrefillTokenThreshold = 32768;
            policy.hashBlockSize = 6144;
            policy.schedulerBlockSize = 6144;
            policy.useEagle = true;
            snapshot.prefillExecutionPolicy = policy;
            snapshots.emplace_back(snapshot);
        }
        for (size_t i = 0; i < 4; ++i) {
            NaviSchedScheduleItem item;
            item.request.requestId = i + 1;
            item.request.promptTokenCount = 200;
            item.request.blockSize = 1;
            context->_requests.emplace_back(item);
        }
        context->setNodes(std::move(nodes), std::move(snapshots));
        return context;
    };
    auto context = makeContext(NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION);
    ASSERT_TRUE(context->prepare());
    std::vector<int64_t> tokens(4);
    std::vector<double> inverse(4);
    const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
        *context, tokens.data(), inverse.data());
    // Poison the legacy surrogate matrix. Exact26 never reads its coefficients.
    std::vector<float> costs(16, std::numeric_limits<float>::quiet_NaN());
    std::vector<double> weights(8);
    std::vector<uint32_t> selected;
    double value = 0.0;
    CostSchedulerKernelInternal::ReplayInfo replay;
    replay.init(true, 1.0, 512.0, 1.0, 0.0, 10, 100000);
    ASSERT_TRUE(CostSchedulerKernelInternal::optimize(
        *context, costs.data(), tokens.data(), total, 1.0, 512.0, 1.0, 0.0, 10, 100000,
        replay, 0.0, 123, weights.data(), selected, value));
    ASSERT_EQ(4u, selected.size());
    CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
    const auto exact = CostSchedulerKernelInternal::forwardDiscrete(
        *context, costs.data(), selected.data(), tokens.data(), inverse.data(), total,
        1.0, 0.0, workspace);
    EXPECT_DOUBLE_EQ(exact, value);
    ASSERT_EQ(1u, replay.optimizationRuns.size());
    EXPECT_EQ(10u, replay.optimizationRuns[0].actualLoopCount);
    EXPECT_EQ("max_loop_count", replay.optimizationRuns[0].stopReason);
    EXPECT_GT(replay.optimizationRuns[0].refinementEvaluations, 0u);
    EXPECT_TRUE(std::isfinite(replay.optimizationRuns[0].finalForwardValue));

    auto graphContext = makeContext(NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION);
    ASSERT_TRUE(graphContext->prepare());
    CostSchedulerConfig config;
    config.lambda = 1.0;
    config.latencyWeightConcavity = 0.0;
    config.initialWeightSeed = 123;
    config.optimizationTimeBudgetUs = 100000;
    auto graph = CostSchedulerGraphState::create(graphContext, config, 2);
    ASSERT_TRUE(graph);
    ASSERT_TRUE(graph->prepareWorker(0));
    ASSERT_TRUE(graph->prepareWorker(1));
    ASSERT_TRUE(graph->finishPreparation());
    std::fill(graph->_costs.get(), graph->_costs.get() + 16,
              std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(graph->shouldContinue());
    for (size_t iteration = 0; graph->shouldContinue(); ++iteration) {
        ASSERT_TRUE(graph->runRow(0, iteration));
        ASSERT_TRUE(graph->runRow(1, iteration));
        ASSERT_TRUE(graph->finishRow(iteration));
        ASSERT_TRUE(graph->runProjection(0, iteration));
        ASSERT_TRUE(graph->runProjection(1, iteration));
        ASSERT_TRUE(graph->finishProjection(iteration));
    }
    ASSERT_TRUE(graph->runFinalRow(0));
    ASSERT_TRUE(graph->runFinalRow(1));
    ASSERT_TRUE(graph->finalize());
    EXPECT_EQ(10u, graph->actualLoopCount());
    EXPECT_EQ("max_loop_count", graph->stopReason());
    EXPECT_DOUBLE_EQ(value, graph->forwardValue());
    // Legacy25 still uses its unchanged continuous path and rejects poisoned costs.
    auto legacy = makeContext(NonLinearPrefillModelR::PD_BINARY_VERSION);
    ASSERT_TRUE(legacy->prepare());
    CostSchedulerKernelInternal::ReplayInfo legacyReplay;
    EXPECT_FALSE(CostSchedulerKernelInternal::optimize(
        *legacy, costs.data(), tokens.data(), total, 1.0, 512.0, 1.0, 0.0, 10, 0,
        legacyReplay, 0.0, 123, weights.data(), selected, value));
}

TEST_F(GeometryIntegrationTest, testGeometryGradientUsesIdleTailWithoutRepair)
{
    constexpr size_t NODE_COUNT = 200;
    constexpr size_t REQUEST_COUNT = 51;
    constexpr size_t WORKERS = 4;
    auto context = std::make_shared<NaviSchedScheduleContext>();
    std::vector<NaviSchedNodeConstPtr> nodes;
    std::vector<NaviSchedNodeSnapshot> snapshots;
    for (size_t i = 0; i < NODE_COUNT; ++i) {
        nodes.emplace_back(std::make_shared<GeometryMissingPolicyNode>());
        NaviSchedNodeSnapshot snapshot;
        auto info = std::make_shared<NaviSchedNodeInfo>();
        info->pdSeparated = true;
        info->agentId = "geometry_large_fleet_" + std::to_string(i);
        snapshot.info = info;
        auto model = std::make_shared<NaviSchedLatencyModel>();
        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        model->binaryVersion = NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION;
        NonLinearPrefillModelR initial;
        initial._pdSeparated = true;
        model->parameters = initial.getInitialParameters();
        snapshot.latencyModel = model;
        TPUPrefillExecutionPolicyConfig policy;
        policy.pcpSize = 8;
        policy.interleaveSize = 256;
        policy.compileSizes = {512, 1024, 2048, 4096};
        policy.maxScheduledTokens = 32768;
        policy.longPrefillTokenThreshold = 32768;
        policy.hashBlockSize = 6144;
        policy.schedulerBlockSize = 6144;
        policy.useEagle = true;
        snapshot.prefillExecutionPolicy = policy;
        snapshot.load.queueTotalPredictedLatencyMs = i + 1 == NODE_COUNT ? 0.0 : 60000.0;
        snapshots.emplace_back(snapshot);
    }
    for (size_t i = 0; i < REQUEST_COUNT; ++i) {
        NaviSchedScheduleItem item;
        item.request.requestId = i + 1;
        item.request.promptTokenCount = 443;
        item.request.blockSize = 6144;
        context->_requests.emplace_back(item);
    }
    context->setNodes(std::move(nodes), std::move(snapshots));
    ASSERT_TRUE(context->prepare());
    CostSchedulerConfig config;
    config.lambda = 0.0;
    config.latencyWeightConcavity = 128.0;
    config.optimizationTimeBudgetUs = 100000;
    // Exercise the real cached-table parallel initialization with an initial
    // maximum on a busy node. The idle candidate is at the end of the fleet.
    auto randomWeights = std::make_shared<std::vector<double>>(1 << 14, 1.0);
    std::fill(randomWeights->begin(), randomWeights->begin() + REQUEST_COUNT, 2.0);
    const auto *table = randomWeights->data();
    std::weak_ptr<const std::vector<double>> tableLifetime = randomWeights;
    auto graph = CostSchedulerGraphState::create(context, config, WORKERS, randomWeights);
    ASSERT_TRUE(graph);
    // Releasing/reseeding the publisher must not invalidate deferred workers.
    randomWeights = std::make_shared<std::vector<double>>(1 << 14, 4.0);
    EXPECT_FALSE(tableLifetime.expired());
    EXPECT_EQ(table, graph->_randomWeights->data());
    EXPECT_DOUBLE_EQ(2.0, graph->_randomWeights->front());
    for (size_t i = 0; i < WORKERS; ++i) {
        ASSERT_TRUE(graph->prepareWorker(i));
    }
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    ASSERT_TRUE(graph->finishPreparation());
    RecordProperty("initialization_us",
                   std::to_string(autil::TimeUtility::monotonicTimeUs() - beginUs));
    for (size_t iteration = 0; graph->shouldContinue(); ++iteration) {
        for (size_t i = 0; i < WORKERS; ++i) {
            ASSERT_TRUE(graph->runRow(i, iteration));
        }
        ASSERT_TRUE(graph->finishRow(iteration));
        for (size_t i = 0; i < WORKERS; ++i) {
            ASSERT_TRUE(graph->runProjection(i, iteration));
        }
        ASSERT_TRUE(graph->finishProjection(iteration));
    }
    for (size_t i = 0; i < WORKERS; ++i) {
        ASSERT_TRUE(graph->runFinalRow(i));
    }
    // No later discrete repair: gradient descent itself must use real cost.
    ASSERT_TRUE(graph->finalize(0));
    EXPECT_GT(graph->actualLoopCount(), 0u);
    ASSERT_EQ(REQUEST_COUNT, context->_selectedNodeIndexes.size());
    for (const auto index : context->_selectedNodeIndexes) {
        EXPECT_EQ(NODE_COUNT - 1, index);
    }
    EXPECT_EQ(NODE_COUNT, context->nodeCount());
    // Preparation must not restart the common optimization deadline.
    auto deadlineGraph =
        CostSchedulerGraphState::create(context, config, WORKERS, graph->_randomWeights);
    ASSERT_TRUE(deadlineGraph);
    deadlineGraph->_optimizationBeginUs =
        autil::TimeUtility::monotonicTimeUs() - config.optimizationTimeBudgetUs;
    for (size_t i = 0; i < WORKERS; ++i) {
        ASSERT_TRUE(deadlineGraph->prepareWorker(i));
    }
    ASSERT_TRUE(deadlineGraph->finishPreparation());
    for (size_t iteration = 0; deadlineGraph->shouldContinue(); ++iteration) {
        for (size_t i = 0; i < WORKERS; ++i) {
            ASSERT_TRUE(deadlineGraph->runRow(i, iteration));
        }
        ASSERT_TRUE(deadlineGraph->finishRow(iteration));
        for (size_t i = 0; i < WORKERS; ++i) {
            ASSERT_TRUE(deadlineGraph->runProjection(i, iteration));
        }
        ASSERT_TRUE(deadlineGraph->finishProjection(iteration));
    }
    EXPECT_EQ(1u, deadlineGraph->actualLoopCount());
    EXPECT_EQ("time_budget", deadlineGraph->_stopReason);
    EXPECT_FALSE(CostSchedulerKernelInternal::shouldRunNextLoop(1, 10, 100, 90, 30, 1));
    EXPECT_TRUE(CostSchedulerKernelInternal::shouldRunNextLoop(1, 10, 100, 20, 20, 1));
    graph.reset();
    EXPECT_FALSE(tableLifetime.expired());
    deadlineGraph.reset();
    EXPECT_TRUE(tableLifetime.expired());
}

TEST_F(GeometryIntegrationTest, testContinuousGeometryGradientMatchesFiniteDifference)
{
    for (const double queueCoefficient : {-80.0, 0.0, 60.0}) {
        for (const bool asyncLookup : {false, true}) {
            for (const auto &promptTokens :
                 {std::vector<int64_t>{100, 300, 800, 1100},
                  std::vector<int64_t>{1000, 2000, 65536, 98304}}) {
                for (const auto weight : {0.13, 0.42}) {
                    NaviSchedScheduleContext context;
                    std::vector<NaviSchedNodeConstPtr> nodes;
                    std::vector<NaviSchedNodeSnapshot> snapshots;
                    for (size_t i = 0; i < 2; ++i) {
                        nodes.emplace_back(
                            std::make_shared<ConditionalQueueGeometryNode>());
                        NaviSchedNodeSnapshot snapshot;
                        auto info = std::make_shared<NaviSchedNodeInfo>();
                        info->pdSeparated = true;
                        info->agentId = "geometry_gradient_" + std::to_string(i);
                        snapshot.info = info;
                        auto model = std::make_shared<NaviSchedLatencyModel>();
                        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
                        model->binaryVersion =
                            NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION;
                        NonLinearPrefillModelR initial;
                        initial._pdSeparated = true;
                        model->parameters = initial.getInitialParameters();
                        snapshot.latencyModel = model;
                        TPUPrefillExecutionPolicyConfig policy;
                        policy.pcpSize = 8;
                        policy.interleaveSize = 256;
                        policy.compileSizes = {512, 1024, 2048, 4096};
                        policy.maxScheduledTokens = 32768;
                        policy.longPrefillTokenThreshold = 32768;
                        policy.hashBlockSize = 6144;
                        policy.schedulerBlockSize = 6144;
                        policy.useEagle = true;
                        policy.asyncKVLookupEnabled = asyncLookup;
                        policy.asyncKVLookupHashBlockSize = 6144;
                        policy.asyncKVLookupUsesEagle = true;
                        snapshot.prefillExecutionPolicy = policy;
                        snapshot.load.queueTotalPredictedLatencyMs = 23.0 + i * 13.0;
                        snapshot.load.queueModel.cacheMemberWeightMs = queueCoefficient;
                        snapshot.load.queueModel.lookupMemberWeightMs =
                            queueCoefficient == 0. ? 45. : -queueCoefficient / 2.;
                        snapshots.emplace_back(snapshot);
                    }
                    for (size_t i = 0; i < promptTokens.size(); ++i) {
                        NaviSchedScheduleItem item;
                        item.request.requestId = i + 1;
                        item.request.promptTokenCount = promptTokens[i];
                        item.request.blockSize = 6144;
                        context._requests.emplace_back(item);
                    }
                    context.setNodes(std::move(nodes), std::move(snapshots));
                    ASSERT_TRUE(context.prepare());
                    std::vector<int64_t> tokens(4);
                    std::vector<double> inverse(4);
                    const auto total =
                        CostSchedulerKernelInternal::fillRequestTokenCounts(
                            context, tokens.data(), inverse.data());
                    std::vector<float> costs(16, std::numeric_limits<float>::quiet_NaN());
                    std::vector<double> weights(8, weight), gradients(8);
                    CostSchedulerKernelInternal::NodeGradient nodeGradients[2];
                    const auto evaluate = [&]() {
                        return CostSchedulerKernelInternal::forwardNodeRange(
                            context, costs.data(), weights.data(), tokens.data(),
                            inverse.data(), total, 0.35, 128.0, 0, 2, nullptr, nullptr,
                            nullptr);
                    };
                    ASSERT_TRUE(
                        std::isfinite(CostSchedulerKernelInternal::forwardNodeRange(
                            context, costs.data(), weights.data(), tokens.data(),
                            inverse.data(), total, 0.35, 128.0, 0, 2, nodeGradients,
                            nullptr, nullptr)));
                    CostSchedulerKernelInternal::backwardNodeRange(
                        context, costs.data(), weights.data(), tokens.data(),
                        inverse.data(), 4, 0, 2, nodeGradients, gradients.data());
                    for (size_t i = 0; i < weights.size(); ++i) {
                        constexpr double DELTA = 1e-6;
                        weights[i] += DELTA;
                        const auto plus = evaluate();
                        weights[i] -= 2 * DELTA;
                        const auto minus = evaluate();
                        weights[i] += DELTA;
                        const auto numerical = (plus - minus) / (2 * DELTA);
                        ASSERT_TRUE(std::isfinite(gradients[i]));
                        EXPECT_NEAR(numerical, gradients[i],
                                    1e-5 * std::max(1.0, std::abs(numerical)))
                            << "async=" << asyncLookup << " weight=" << weight
                            << " index=" << i;
                    }
                }
            }
        }
    }
    EXPECT_EQ(95000,
              CostSchedulerKernelInternal::discreteRefinementBudgetUs(100000, 5000));
}

TEST_F(GeometryIntegrationTest, testFixedChunkPlanGradientMatchesFiniteDifference)
{
    for (const double queueCoefficient : {-80.0, 0.0, 60.0}) {
        for (const bool asyncLookup : {false, true}) {
            for (const auto &promptTokens :
                 {std::vector<int64_t>{100, 300, 800, 1100},
                  std::vector<int64_t>{1000, 2000, 65536, 98304}}) {
                for (const auto weight : {0.13, 0.42, 0.63, 0.87}) {
                    NaviSchedScheduleContext context;
                    std::vector<NaviSchedNodeConstPtr> nodes;
                    std::vector<NaviSchedNodeSnapshot> snapshots;
                    for (size_t i = 0; i < 2; ++i) {
                        nodes.emplace_back(
                            std::make_shared<ConditionalQueueGeometryNode>());
                        NaviSchedNodeSnapshot snapshot;
                        auto info = std::make_shared<NaviSchedNodeInfo>();
                        info->pdSeparated = true;
                        info->agentId = "geometry_gradient_" + std::to_string(i);
                        snapshot.info = info;
                        auto model = std::make_shared<NaviSchedLatencyModel>();
                        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
                        model->binaryVersion =
                            NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION;
                        NonLinearPrefillModelR initial;
                        initial._pdSeparated = true;
                        model->parameters = initial.getInitialParameters();
                        snapshot.latencyModel = model;
                        TPUPrefillExecutionPolicyConfig policy;
                        policy.pcpSize = 8;
                        policy.interleaveSize = 256;
                        policy.compileSizes = {512, 1024, 2048, 4096};
                        policy.maxScheduledTokens = 32768;
                        policy.longPrefillTokenThreshold = 8192;
                        policy.fixedPrefillChunkPlan = true;
                        policy.attentionScheduleCapacity = 4096;
                        policy.attentionQueryTile = 128;
                        policy.attentionKvTile = 128;
                        policy.hashBlockSize = 6144;
                        policy.schedulerBlockSize = 6144;
                        policy.useEagle = true;
                        policy.asyncKVLookupEnabled = asyncLookup;
                        policy.asyncKVLookupHashBlockSize = 6144;
                        policy.asyncKVLookupUsesEagle = true;
                        snapshot.prefillExecutionPolicy = policy;
                        snapshot.load.queueTotalPredictedLatencyMs = 23.0 + i * 13.0;
                        snapshot.load.queueModel.cacheMemberWeightMs = queueCoefficient;
                        snapshot.load.queueModel.lookupMemberWeightMs =
                            queueCoefficient == 0. ? 45. : -queueCoefficient / 2.;
                        snapshots.emplace_back(snapshot);
                    }
                    for (size_t i = 0; i < promptTokens.size(); ++i) {
                        NaviSchedScheduleItem item;
                        item.request.requestId = i + 1;
                        item.request.promptTokenCount = promptTokens[i];
                        item.request.blockSize = 6144;
                        context._requests.emplace_back(item);
                    }
                    context.setNodes(std::move(nodes), std::move(snapshots));
                    ASSERT_TRUE(context.prepare());
                    std::vector<int64_t> tokens(4);
                    std::vector<double> inverse(4);
                    const auto total =
                        CostSchedulerKernelInternal::fillRequestTokenCounts(
                            context, tokens.data(), inverse.data());
                    std::vector<float> costs(16, std::numeric_limits<float>::quiet_NaN());
                    std::vector<double> weights(8, weight), gradients(8);
                    CostSchedulerKernelInternal::NodeGradient nodeGradients[2];
                    const auto evaluate = [&]() {
                        return CostSchedulerKernelInternal::forwardNodeRange(
                            context, costs.data(), weights.data(), tokens.data(),
                            inverse.data(), total, 0.35, 128.0, 0, 2, nullptr, nullptr,
                            nullptr);
                    };
                    ASSERT_TRUE(
                        std::isfinite(CostSchedulerKernelInternal::forwardNodeRange(
                            context, costs.data(), weights.data(), tokens.data(),
                            inverse.data(), total, 0.35, 128.0, 0, 2, nodeGradients,
                            nullptr, nullptr)));
                    CostSchedulerKernelInternal::backwardNodeRange(
                        context, costs.data(), weights.data(), tokens.data(),
                        inverse.data(), 4, 0, 2, nodeGradients, gradients.data());
                    for (size_t i = 0; i < weights.size(); ++i) {
                        constexpr double DELTA = 1e-6;
                        weights[i] += DELTA;
                        const auto plus = evaluate();
                        weights[i] -= 2 * DELTA;
                        const auto minus = evaluate();
                        weights[i] += DELTA;
                        const auto numerical = (plus - minus) / (2 * DELTA);
                        ASSERT_TRUE(std::isfinite(gradients[i]));
                        EXPECT_NEAR(numerical, gradients[i],
                                    1e-5 * std::max(1.0, std::abs(numerical)))
                            << "async=" << asyncLookup << " weight=" << weight
                            << " index=" << i;
                    }
                }
            }
        }
    }
    EXPECT_EQ(95000,
              CostSchedulerKernelInternal::discreteRefinementBudgetUs(100000, 5000));
}

TEST_F(GeometryIntegrationTest,
       testFixedInputBudgetAndDraftSlotsGradientMatchesFiniteDifference)
{
    for (const double queueCoefficient : {-80.0, 0.0, 60.0}) {
        for (const bool asyncLookup : {false, true}) {
            for (const auto &promptTokens :
                 {std::vector<int64_t>{100, 300, 800, 1100},
                  std::vector<int64_t>{1000, 2000, 65536, 98304}}) {
                for (const auto weight : {0.13, 0.42, 0.63, 0.87}) {
                    NaviSchedScheduleContext context;
                    std::vector<NaviSchedNodeConstPtr> nodes;
                    std::vector<NaviSchedNodeSnapshot> snapshots;
                    for (size_t i = 0; i < 2; ++i) {
                        nodes.emplace_back(
                            std::make_shared<ConditionalQueueGeometryNode>());
                        NaviSchedNodeSnapshot snapshot;
                        auto info = std::make_shared<NaviSchedNodeInfo>();
                        info->pdSeparated = true;
                        info->agentId = "geometry_gradient_" + std::to_string(i);
                        snapshot.info = info;
                        auto model = std::make_shared<NaviSchedLatencyModel>();
                        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
                        model->binaryVersion =
                            NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION;
                        NonLinearPrefillModelR initial;
                        initial._pdSeparated = true;
                        model->parameters = initial.getInitialParameters();
                        snapshot.latencyModel = model;
                        TPUPrefillExecutionPolicyConfig policy;
                        policy.pcpSize = 8;
                        policy.interleaveSize = 256;
                        policy.compileSizes = {512, 1024, 2048, 4096};
                        policy.maxScheduledTokens = 32768;
                        policy.longPrefillTokenThreshold = 8192;
                        policy.fixedPrefillChunkPlan = true;
                        policy.maxInputTokens = 4096;
                        policy.draftSlotsPerRequest = 64;
                        policy.prefillLookahead = 4;
                        policy.attentionScheduleCapacity = 0;
                        policy.attentionQueryTile = 0;
                        policy.attentionKvTile = 0;
                        policy.hashBlockSize = 6144;
                        policy.schedulerBlockSize = 6144;
                        policy.useEagle = true;
                        policy.asyncKVLookupEnabled = asyncLookup;
                        policy.asyncKVLookupHashBlockSize = 6144;
                        policy.asyncKVLookupUsesEagle = true;
                        snapshot.prefillExecutionPolicy = policy;
                        snapshot.load.queueTotalPredictedLatencyMs = 23.0 + i * 13.0;
                        snapshot.load.queueModel.cacheMemberWeightMs = queueCoefficient;
                        snapshot.load.queueModel.lookupMemberWeightMs =
                            queueCoefficient == 0. ? 45. : -queueCoefficient / 2.;
                        snapshots.emplace_back(snapshot);
                    }
                    for (size_t i = 0; i < promptTokens.size(); ++i) {
                        NaviSchedScheduleItem item;
                        item.request.requestId = i + 1;
                        item.request.promptTokenCount = promptTokens[i];
                        item.request.blockSize = 6144;
                        context._requests.emplace_back(item);
                    }
                    context.setNodes(std::move(nodes), std::move(snapshots));
                    ASSERT_TRUE(context.prepare());
                    std::vector<int64_t> tokens(4);
                    std::vector<double> inverse(4);
                    const auto total =
                        CostSchedulerKernelInternal::fillRequestTokenCounts(
                            context, tokens.data(), inverse.data());
                    std::vector<float> costs(16, std::numeric_limits<float>::quiet_NaN());
                    std::vector<double> weights(8, weight), gradients(8);
                    CostSchedulerKernelInternal::NodeGradient nodeGradients[2];
                    const auto evaluate = [&]() {
                        return CostSchedulerKernelInternal::forwardNodeRange(
                            context, costs.data(), weights.data(), tokens.data(),
                            inverse.data(), total, 0.35, 128.0, 0, 2, nullptr, nullptr,
                            nullptr);
                    };
                    ASSERT_TRUE(
                        std::isfinite(CostSchedulerKernelInternal::forwardNodeRange(
                            context, costs.data(), weights.data(), tokens.data(),
                            inverse.data(), total, 0.35, 128.0, 0, 2, nodeGradients,
                            nullptr, nullptr)));
                    CostSchedulerKernelInternal::backwardNodeRange(
                        context, costs.data(), weights.data(), tokens.data(),
                        inverse.data(), 4, 0, 2, nodeGradients, gradients.data());
                    for (size_t i = 0; i < weights.size(); ++i) {
                        constexpr double DELTA = 1e-6;
                        weights[i] += DELTA;
                        const auto plus = evaluate();
                        weights[i] -= 2 * DELTA;
                        const auto minus = evaluate();
                        weights[i] += DELTA;
                        const auto numerical = (plus - minus) / (2 * DELTA);
                        ASSERT_TRUE(std::isfinite(gradients[i]));
                        EXPECT_NEAR(numerical, gradients[i],
                                    1e-5 * std::max(1.0, std::abs(numerical)))
                            << "async=" << asyncLookup << " weight=" << weight
                            << " index=" << i;
                    }
                }
            }
        }
    }
    EXPECT_EQ(95000,
              CostSchedulerKernelInternal::discreteRefinementBudgetUs(100000, 5000));
}

TEST_F(GeometryIntegrationTest, testGeometryInputBatchPreservesViewsAndReducesRowOverhead)
{
    constexpr size_t nodeCount = 200, requestCount = 64;
    NaviSchedScheduleContext context;
    std::vector<NaviSchedNodeConstPtr> nodes;
    std::vector<NaviSchedNodeSnapshot> snapshots;
    for (size_t i = 0; i < nodeCount; ++i) {
        nodes.push_back(std::make_shared<GeometryMissingPolicyNode>());
        NaviSchedNodeSnapshot snapshot;
        auto info = std::make_shared<NaviSchedNodeInfo>();
        info->pdSeparated = true;
        snapshot.info = info;
        auto model = std::make_shared<NaviSchedLatencyModel>();
        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        model->binaryVersion = NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION;
        NonLinearPrefillModelR initial;
        initial._pdSeparated = true;
        model->parameters = initial.getInitialParameters();
        model->parameters[3] *= 1.0 + i * 0.01;
        snapshot.latencyModel = model;
        snapshot.prefillExecutionPolicy = typedRuntimePolicy();
        snapshot.residualRuntimeConfig.learningRate = 0.0;
        snapshots.push_back(snapshot);
    }
    for (size_t i = 0; i < requestCount; ++i) {
        NaviSchedScheduleItem item;
        item.request.promptTokenCount = 200 + i * 7;
        item.request.blockSize = 1;
        context._requests.push_back(item);
    }
    context.setNodes(std::move(nodes), std::move(snapshots));
    ASSERT_TRUE(context.prepare());
    std::vector<std::shared_ptr<const std::vector<NaviSchedGeometryOptimizationInput>>>
        owners;
    for (size_t n = 0; n < nodeCount; ++n) {
        const auto inputs = context.getGeometryOptimizationInputs(n);
        ASSERT_TRUE(inputs);
        ASSERT_EQ(requestCount, inputs->size());
        for (size_t i = 0; i < requestCount; ++i) {
            EXPECT_EQ(&(*inputs)[i], context.getGeometryOptimizationInput(n, i));
        }
        owners.push_back(inputs);
    }
    EXPECT_FALSE(context.getGeometryOptimizationInputs(nodeCount));
    const std::vector<size_t> viewRequests{63, 1, 31};
    ASSERT_TRUE(context.activateScheduleView(viewRequests, {199, 0}));
    EXPECT_EQ(owners[199], context.getGeometryOptimizationInputs(0));
    EXPECT_EQ(owners[0], context.getGeometryOptimizationInputs(1));
    for (size_t i = 0; i < context.requestCount(); ++i) {
        EXPECT_EQ(context.getGeometryOptimizationInput(0, i),
                  context.getGeometryOptimizationInput(*owners[199], i));
        EXPECT_EQ(&(*owners[199])[viewRequests[i]],
                  context.getGeometryOptimizationInput(*owners[199], i));
    }
    EXPECT_EQ(nullptr, context.getGeometryOptimizationInput(*owners[199], 3));
    EXPECT_EQ(nullptr, context.getGeometryOptimizationInput(
                           std::vector<NaviSchedGeometryOptimizationInput>{}, 0));
    context.clearScheduleView();
    const auto measure = [&](bool batch) {
        double checksum = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (size_t round = 0; round < 8; ++round) {
            for (size_t n = 0; n < nodeCount; ++n) {
                const auto inputs =
                    batch ? context.getGeometryOptimizationInputs(n) : nullptr;
                for (size_t i = 0; i < requestCount; ++i) {
                    const auto *input =
                        batch ? context.getGeometryOptimizationInput(*inputs, i)
                              : context.getGeometryOptimizationInput(n, i);
                    checksum += input->linearCost + input->sharedTokens +
                                (input->shared ? 1.0 : 0.0);
                }
            }
        }
        return std::make_pair(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - start)
                                  .count(),
                              checksum);
    };
    EXPECT_EQ(measure(false).second, measure(true).second);
    std::vector<int64_t> scalarTimes, batchTimes;
    for (size_t sample = 0; sample < 10; ++sample) {
        const auto first = measure(sample % 2 != 0);
        const auto second = measure(sample % 2 == 0);
        const auto scalar = sample % 2 == 0 ? first : second;
        const auto batch = sample % 2 == 0 ? second : first;
        ASSERT_EQ(scalar.second, batch.second);
        scalarTimes.push_back(scalar.first);
        batchTimes.push_back(batch.first);
        std::cout << "geometry_input_batch_perf sample=" << sample
                  << " scalar_ns=" << scalar.first << " batch_ns=" << batch.first
                  << std::endl;
    }
    std::sort(scalarTimes.begin(), scalarTimes.end());
    std::sort(batchTimes.begin(), batchTimes.end());
    EXPECT_LE(batchTimes[4] + batchTimes[5], (scalarTimes[4] + scalarTimes[5]) * 1.05);

    std::vector<int64_t> tokens(requestCount);
    std::vector<double> inverse(requestCount);
    const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
        context, tokens.data(), inverse.data());
    std::vector<float> costs(2 * nodeCount * requestCount);
    ASSERT_TRUE(CostSchedulerKernelInternal::fillCostsForNodeRange(
        context, tokens.data(), 0, nodeCount, costs.data()));
    std::vector<double> weights(nodeCount * requestCount, 1.0 / nodeCount);
    std::vector<double> gradients(weights.size());
    std::vector<CostSchedulerKernelInternal::NodeGradient> nodeGradients(nodeCount);
    const auto runGradient = [&]() {
        const auto loss = CostSchedulerKernelInternal::forwardNodeRange(
            context, costs.data(), weights.data(), tokens.data(), inverse.data(), total,
            0.35, 128.0, 0, nodeCount, nodeGradients.data(), nullptr, nullptr);
        CostSchedulerKernelInternal::backwardNodeRange(
            context, costs.data(), weights.data(), tokens.data(), inverse.data(),
            requestCount, 0, nodeCount, nodeGradients.data(), gradients.data());
        return loss;
    };
    const auto expectedLoss = runGradient();
    ASSERT_TRUE(std::isfinite(expectedLoss));
    const auto expectedGradients = gradients;
    for (const auto gradient : expectedGradients) {
        ASSERT_TRUE(std::isfinite(gradient));
    }
    uint64_t gradientHash = 14695981039346656037ULL;
    for (const auto gradient : expectedGradients) {
        uint64_t bits = 0;
        std::memcpy(&bits, &gradient, sizeof(bits));
        for (size_t byte = 0; byte < sizeof(bits); ++byte) {
            gradientHash ^= (bits >> (byte * 8)) & 0xff;
            gradientHash *= 1099511628211ULL;
        }
    }
    for (size_t sample = 0; sample < 10; ++sample) {
        constexpr size_t rounds = 32;
        const auto start = std::chrono::steady_clock::now();
        double loss = 0.0;
        for (size_t round = 0; round < rounds; ++round) {
            loss = runGradient();
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
        ASSERT_EQ(expectedLoss, loss);
        ASSERT_EQ(expectedGradients, gradients);
        std::cout << "geometry_gradient_row_perf nodes=" << nodeCount
                  << " requests=" << requestCount << " sample=" << sample
                  << " rounds=" << rounds << " elapsed_ns=" << elapsed
                  << " gradient_hash=" << gradientHash << " loss=" << std::hexfloat
                  << loss << std::defaultfloat << std::endl;
    }
}

TEST_F(GeometryIntegrationTest,
       testLargeBatchTrainedGeometryGradientAndIndependentEndpoint)
{
    class CachedNode : public GeometryMissingPolicyNode {
    public:
        bool getExecutionCacheInputs(const NaviSchedScheduleRequest &request,
                                     int64_t &hit, int64_t &boundary) const override
        {
            hit = std::min<int64_t>((request.requestId % 4) * 6144,
                                    (request.promptTokenCount - 1) / 6144 * 6144);
            boundary = hit;
            return true;
        }
    };
    // Captured node116.119 model29 revision550, in PD parameter coordinates.
    const std::vector<double> trained = {
        98.53767390761811,    1.633745404495898, 0.002044713718932428, 37.74273631790216,
        71.21778098699359,    3.020330147711656, 1.1410292074819923,   18.45487861505201,
        0.042174675678372964, 1.3528740898369398};
    for (const size_t count : {64u, 200u}) {
        for (const bool asyncLookup : {false, true}) {
            NaviSchedScheduleContext context;
            std::vector<NaviSchedNodeConstPtr> nodes;
            std::vector<NaviSchedNodeSnapshot> snapshots;
            for (size_t node = 0; node < 2; ++node) {
                nodes.emplace_back(std::make_shared<CachedNode>());
                NaviSchedNodeSnapshot snapshot;
                auto info = std::make_shared<NaviSchedNodeInfo>();
                info->pdSeparated = true;
                info->agentId = "large_geometry_" + std::to_string(node);
                snapshot.info = info;
                auto model = std::make_shared<NaviSchedLatencyModel>();
                model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
                model->binaryVersion =
                    NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION;
                model->parameters = trained;
                snapshot.latencyModel = model;
                TPUPrefillExecutionPolicyConfig policy;
                policy.pcpSize = 8;
                policy.interleaveSize = 256;
                policy.compileSizes = {512, 1024, 2048, 4096};
                policy.maxScheduledTokens = 32768;
                policy.longPrefillTokenThreshold = 32768;
                policy.hashBlockSize = 6144;
                policy.schedulerBlockSize = 6144;
                policy.useEagle = true;
                policy.asyncKVLookupEnabled = asyncLookup;
                policy.asyncKVLookupHashBlockSize = 6144;
                policy.asyncKVLookupUsesEagle = true;
                snapshot.prefillExecutionPolicy = policy;
                snapshot.load.queueTotalPredictedLatencyMs = 20.0 + node * 100.0;
                snapshots.emplace_back(snapshot);
            }
            for (size_t i = 0; i < count; ++i) {
                NaviSchedScheduleItem item;
                item.request.requestId = i + 1;
                item.request.promptTokenCount =
                    i % 3 == 0 ? 197 + i * 11 : 18433 + (i % 7) * 12288;
                item.request.blockSize = 6144;
                context._requests.emplace_back(item);
            }
            context.setNodes(std::move(nodes), std::move(snapshots));
            ASSERT_TRUE(context.prepare());
            std::vector<int64_t> tokens(count);
            std::vector<double> inverse(count);
            const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
                context, tokens.data(), inverse.data());
            std::vector<float> costs(count * 4, std::numeric_limits<float>::quiet_NaN());
            std::vector<double> weights(count * 2), gradients(count * 2);
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] = 0.013 + (i % 19) * 0.047;
            }
            CostSchedulerKernelInternal::NodeGradient nodeGradients[2];
            const auto evaluate = [&](CostSchedulerKernelInternal::NodeGradient *g,
                                      double *first) {
                return CostSchedulerKernelInternal::forwardNodeRange(
                    context, costs.data(), weights.data(), tokens.data(), inverse.data(),
                    total, 0.35, 128.0, 0, 2, g, first, nullptr);
            };
            ASSERT_TRUE(std::isfinite(evaluate(nodeGradients, nullptr)));
            CostSchedulerKernelInternal::backwardNodeRange(
                context, costs.data(), weights.data(), tokens.data(), inverse.data(),
                count, 0, 2, nodeGradients, gradients.data());
            for (size_t i = 0; i < weights.size(); ++i) {
                constexpr double DELTA = 1e-5;
                const auto original = weights[i];
                weights[i] = original + DELTA;
                const auto plus = evaluate(nullptr, nullptr);
                weights[i] = original - DELTA;
                const auto minus = evaluate(nullptr, nullptr);
                weights[i] = original;
                const auto numerical = (plus - minus) / (2 * DELTA);
                ASSERT_TRUE(std::isfinite(gradients[i]));
                EXPECT_NEAR(numerical, gradients[i],
                            1e-5 * std::max(1.0, std::abs(numerical)))
                    << "count=" << count << " async=" << asyncLookup << " index=" << i;
            }
            if (asyncLookup) {
                std::fill(weights.begin(), weights.end(), 0.0);
                std::vector<size_t> independent;
                for (size_t i = 0; i < count; ++i) {
                    if (i % 3 != 0) {
                        weights[i] = 1.0;
                        independent.push_back(i);
                    }
                }
                double firstTerms[2] = {};
                ASSERT_TRUE(std::isfinite(evaluate(nullptr, firstTerms)));
                const auto exact =
                    context.estimateBatchExecutionLatencyMs(0, independent);
                ASSERT_GT(exact, 0.0);
                EXPECT_NEAR(exact, firstTerms[0] / independent.size(), 1e-8 * exact);
                EXPECT_DOUBLE_EQ(0.0, firstTerms[1]);
            }
        }
    }
}

TEST_F(GeometryIntegrationTest, testAgentWaitingRecomputeUsesVersion26Geometry)
{
    for (const int64_t version :
         {NonLinearPrefillModelR::PD_BINARY_VERSION,
          NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION,
          NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION}) {
        AgentReportR reporter;
        auto tracker = std::make_shared<ConcurrentRequestLatencyTracker>();
        auto prediction = std::make_shared<NaviSchedLatencyPrediction>();
        ASSERT_TRUE(prediction->features.appendBatchCount(1));
        prediction->features.appendRequest(0, 512);
        ASSERT_TRUE(prediction->features.setExecutionGeometry(2, 12288));
        prediction->fixedLatencyMs = 7.0;
        prediction->recomputable = true;
        // Production descriptors cross Master→Agent serialization before the
        // waiting tracker evaluates a later online model revision.
        autil::DataBuffer descriptorWire;
        prediction->serialize(descriptorWire);
        auto decodedPrediction = std::make_shared<NaviSchedLatencyPrediction>();
        decodedPrediction->deserialize(descriptorWire);
        ASSERT_TRUE(decodedPrediction->features.hasExecutionGeometry());
        ASSERT_EQ(2, decodedPrediction->features.getExecutionStepCount());
        ASSERT_EQ(12288, decodedPrediction->features.getGlobalPaddedTokens());
        prediction = std::move(decodedPrediction);
        RequestLatencyTracker::Snapshot inflight;
        ASSERT_TRUE(tracker->add(101, 100.0, prediction, inflight));
        ASSERT_TRUE(tracker->add(102, 200.0, prediction, inflight));
        ASSERT_TRUE(tracker->markRunning({102}, inflight));
        ASSERT_TRUE(reporter.setInflightTracker(tracker));
        auto snapshot = std::make_shared<NaviSchedLatencyModelSnapshot>();
        snapshot->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        snapshot->binaryVersion = version;
        snapshot->revision = 9;
        NonLinearPrefillModelR initial;
        initial._pdSeparated = true;
        snapshot->parameters = initial.getInitialParameters();
        snapshot->optimizerState = initial.getOptimizerState();
        snapshot->onlineFitUpdate = true;
        ASSERT_TRUE(snapshot->valid());
        const auto legacy = NonLinearPrefillModelR::predict(
            prediction->features, NonLinearPrefillModelR::parametersForLegacyPrediction(
                                      snapshot->parameters, version));
        const auto exact = NonLinearPrefillModelR::predictForVersion(
            prediction->features, snapshot->parameters, version);
        if (NonLinearPrefillModelR::usesGeometryForVersion(version)) {
            // This same complete-batch geometry was ignored by the old callback.
            EXPECT_GT(std::abs(exact - legacy), 1.0);
        }
        const auto expected =
            NonLinearPrefillModelR::usesGeometryForVersion(version) ? exact : legacy;
        reporter.reportLatencyModelUpdated(snapshot);
        const auto after = tracker->snapshot();
        EXPECT_DOUBLE_EQ(expected + 7.0 + 200.0, after.totalPredictedLatencyMs);
        EXPECT_EQ(1u, after.recomputedRequestCount); // Running request remains200.
        EXPECT_EQ(1u, reporter._inflightRecomputeCount.load());
        EXPECT_EQ(snapshot.get(), reporter.getLatencyModelSnapshot().get());
        EXPECT_EQ(9, reporter.getLatencyModelSnapshot()->revision);
        auto nextSnapshot = std::make_shared<NaviSchedLatencyModelSnapshot>(*snapshot);
        nextSnapshot->revision = 10;
        reporter.reportLatencyModelUpdated(nextSnapshot);
        EXPECT_EQ(10, reporter.getLatencyModelSnapshot()->revision);
        if (NonLinearPrefillModelR::usesGeometryForVersion(version)) {
            prediction->features.clearExecutionGeometry();
            auto incomplete =
                std::make_shared<NaviSchedLatencyModelSnapshot>(*nextSnapshot);
            ++incomplete->revision;
            reporter.reportLatencyModelUpdated(incomplete);
            const auto failed = tracker->snapshot();
            EXPECT_EQ(after.revision + 1, failed.revision);
            EXPECT_DOUBLE_EQ(after.totalPredictedLatencyMs,
                             failed.totalPredictedLatencyMs);
            EXPECT_EQ(1u, reporter._inflightRecomputeFailureCount.load());
            EXPECT_EQ(nextSnapshot.get(), reporter.getLatencyModelSnapshot().get());
        }
    }
}

TEST_F(GeometryIntegrationTest, testMasterAdmissionAuditUsesVersion26Geometry)
{
    for (const int64_t version : {NonLinearPrefillModelR::PD_BINARY_VERSION,
                                  NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION}) {
        NaviSchedScheduleContext context;
        NaviSchedScheduleItem item;
        item.request.requestId = 1;
        item.request.promptTokenCount = 512;
        context._requests = {item};
        NaviSchedNodeSnapshot snapshot;
        auto info = std::make_shared<NaviSchedNodeInfo>();
        info->pdSeparated = true;
        snapshot.info = info;
        auto model = std::make_shared<NaviSchedLatencyModel>();
        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        model->binaryVersion = version;
        NonLinearPrefillModelR initial;
        initial._pdSeparated = true;
        model->parameters = initial.getInitialParameters();
        snapshot.latencyModel = model;
        context.setNodes({std::make_shared<GeometryMissingPolicyNode>()}, {snapshot});
        context._selectedNodeIndexes = {0};
        context._decisions.resize(1);
        auto prediction = std::make_shared<NaviSchedLatencyPrediction>();
        ASSERT_TRUE(prediction->features.appendBatchCount(1));
        prediction->features.appendRequest(0, 512);
        ASSERT_TRUE(prediction->features.setExecutionGeometry(2, 12288));
        prediction->recomputable = true;
        context._decisions[0].latencyPrediction = prediction;
        const auto legacy = NonLinearPrefillModelR::predict(
            prediction->features, context.getNode(0).getLatencyParameters());
        const auto exact = NonLinearPrefillModelR::predictForVersion(
            prediction->features, model->parameters, version);
        const auto expected =
            version == NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION ? exact : legacy;
        context._decisions[0].predictedExecutionLatencyMs = expected;
        EXPECT_NEAR(expected, context.getNode(0).predictBaseLatency(prediction->features),
                    1e-9);
        const auto audit =
            MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0);
        ASSERT_TRUE(audit.valid);
        EXPECT_NEAR(expected, audit.baseExecutionMs, 1e-9);
        EXPECT_NEAR(expected, audit.finalExecutionMs, 1e-9);
        if (version == NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION) {
            EXPECT_GT(std::abs(exact - legacy), 1.0);
            prediction->features.clearExecutionGeometry();
            EXPECT_FALSE(
                MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0)
                    .valid);
        } else {
            EXPECT_DOUBLE_EQ(legacy, audit.finalExecutionMs);
        }
    }
}

TEST_F(GeometryIntegrationTest,
       testExplicitCheckpointChangesGeometryWithoutChangingCacheHit)
{
    MasterPendingCacheR pending;
    MasterStateR::NodeState node;
    node._pendingCacheR = &pending;
    node._info.agentIdHash = 123;
    node._cachePolicy.blockSize = 6144;
    node._cachePolicy.linearStep = 1;
    node._cachePolicyReady.store(true);
    node._cacheStateReady.store(true);
    node._hybridTailRequired = true;
    TPUPrefillExecutionPolicyConfig policy;
    policy.pcpSize = 8;
    policy.interleaveSize = 256;
    policy.compileSizes = {512, 1024, 2048, 4096};
    policy.maxScheduledTokens = policy.longPrefillTokenThreshold = 32768;
    policy.hashBlockSize = policy.schedulerBlockSize = 6144;
    policy.useEagle = false;
    policy.asyncKVLookupEnabled = true;
    policy.asyncKVLookupHashBlockSize = 6144;
    policy.asyncKVLookupUsesEagle = true;
    policy.fixedPrefillChunkPlan = true;
    policy.mambaBlockAlignedSplit = true;
    policy.prefillLookahead = 1;
    policy.maxInputTokens = 32768;
    auto published = std::make_shared<MasterStateR::NodeState::PublishedSnapshot>();
    published->prefillExecutionPolicy = policy;
    node._publishedSnapshot = published;
    std::vector<NaviSchedScheduleItem> requests(1);
    auto &request = requests[0].request;
    request.requestId = 1;
    request.promptTokenCount = 12373;
    request.blockSize = 6144;
    request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    std::vector<TPUPrefillRequestInput> inputs;
    auto project = [&]() {
        EXPECT_TRUE(node.getExecutionCacheInputs(requests, inputs));
        TPUPrefillExecutionProjection result;
        EXPECT_TRUE(TPUPrefillExecutionPolicy::projectCacheResolvedBatch(inputs, policy,
                                                                         12373, result));
        return result;
    };
    EXPECT_EQ(2, project().stepCount);
    request.explicitCacheCheckpointPosition = 8704;
    EXPECT_EQ(3, project().stepCount);
    EXPECT_EQ(6144, inputs[0].sharedPrefixBoundary);
    EXPECT_EQ(0, inputs[0].computedTokens);
    request.explicitCacheCheckpointPosition = 0;
    EXPECT_EQ(2, project().stepCount);
    request.explicitCacheCheckpointPosition = 8704;
    EXPECT_EQ(3, project().stepCount);
    // A sub-block directive must leave an existing larger cache junction intact.
    node._fullReadyBlockHashes = {11, 22};
    request.explicitCacheCheckpointPosition = 100;
    ASSERT_TRUE(node.getExecutionCacheInputs(requests, inputs));
    EXPECT_EQ(12288, inputs[0].sharedPrefixBoundary);
    request.explicitCacheCheckpointPosition = 8704;
    ASSERT_TRUE(node.getExecutionCacheInputs(requests, inputs));
    EXPECT_EQ(6144, inputs[0].sharedPrefixBoundary);
    request.explicitCacheCheckpointPosition = 12374;
    EXPECT_FALSE(node.getExecutionCacheInputs(requests, inputs));
}

TEST_F(GeometryIntegrationTest, testCacheObservationStaysWithMatchedPrefix)
{
    static_assert(sizeof(TPUPrefillRequestInput) == 6 * sizeof(int64_t),
                  "Diagnostic metadata must stay outside hot geometry inputs");
    MasterPendingCacheR pending;
    MasterStateR::NodeState node;
    node._pendingCacheR = &pending;
    node._info.agentIdHash = 123;
    node._cachePolicy.blockSize = 6144;
    node._cachePolicy.linearStep = 1;
    node._cachePolicyReady.store(true);
    node._cacheStateReady.store(true);
    node._hybridTailRequired = true;
    node._cacheRevision = 489;
    node._fullReadyBlockHashes = {11};
    node._masterFullReadyHashChecksum = 11;
    std::vector<NaviSchedScheduleItem> requests(1);
    requests[0].request.requestId = 1;
    requests[0].request.blockSize = 6144;
    requests[0].request.promptTokenCount = 12370;
    requests[0].request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    std::vector<TPUPrefillRequestInput> inputs;
    TPUPrefillCacheObservationBatch observation;
    ASSERT_TRUE(node.getExecutionCacheInputs(requests, inputs, nullptr, &observation));
    ASSERT_EQ(1u, inputs.size());
    ASSERT_TRUE(observation.pool.has_value());
    EXPECT_EQ(489u, observation.pool->revision);
    EXPECT_TRUE(observation.pool->stateReady);
    EXPECT_EQ(1u, observation.pool->fullReadyBlocks);
    EXPECT_EQ(11u, observation.pool->fullReadyChecksum);
    EXPECT_EQ(1, observation.fullPrefixBlocksBeforeQueryDrop[0]);
    EXPECT_EQ(0, inputs[0].computedTokens);
    EXPECT_EQ(6144, inputs[0].sharedPrefixBoundary);
    node._cacheRevision = 506;
    node._fullReadyBlockHashes = {11, 22};
    node._masterFullReadyHashChecksum = 33;
    std::vector<TPUPrefillRequestInput> later;
    TPUPrefillCacheObservationBatch laterObservation;
    ASSERT_TRUE(
        node.getExecutionCacheInputs(requests, later, nullptr, &laterObservation));
    EXPECT_EQ(506u, laterObservation.pool->revision);
    EXPECT_EQ(2, laterObservation.fullPrefixBlocksBeforeQueryDrop[0]);
    EXPECT_EQ(12288, later[0].sharedPrefixBoundary);
    EXPECT_EQ(489u, observation.pool->revision);
    EXPECT_EQ(11u, observation.pool->fullReadyChecksum);
    EXPECT_EQ(6144, inputs[0].sharedPrefixBoundary);
    requests.clear();
    ASSERT_TRUE(
        node.getExecutionCacheInputs(requests, later, nullptr, &laterObservation));
    EXPECT_TRUE(later.empty());
    EXPECT_FALSE(laterObservation.pool.has_value());
    EXPECT_TRUE(laterObservation.fullPrefixBlocksBeforeQueryDrop.empty());
}

TEST_F(GeometryIntegrationTest, testSharedBoundaryUsesEffectiveLocalFullPrefix)
{
    MasterPendingCacheR pending;
    MasterStateR::NodeState node;
    node._pendingCacheR = &pending;
    node._info.agentIdHash = 123;
    node._cachePolicy.blockSize = 6144;
    node._cachePolicy.linearStep = 1;
    node._cachePolicy.fullAttentionQueryDropTokens = 6144;
    node._cachePolicyReady.store(true);
    node._hybridTailRequired = true;
    NaviSchedScheduleRequest request;
    request.requestId = 1;
    request.blockSize = 6144;
    request.promptTokenCount = 7050;
    request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    node._fullReadyBlockHashes = {11};
    int64_t hit = -1, boundary = -1;
    ASSERT_TRUE(node.getExecutionCacheInputs(request, hit, boundary));
    EXPECT_EQ(0, hit);
    EXPECT_EQ(0, boundary); // Raw Full1 is entirely discarded by Eagle.
    const auto one = pending.getMatchBreakdown(
        123, *request.blockHashes, node._cachedBlockHashes, node._fullReadyBlockHashes,
        node._tailReadyBlockHashes, true, 1, 1);
    EXPECT_EQ(1, one.fullMatchedBlockCount);
    EXPECT_EQ(0, one.effectiveFullMatchedBlockCount);
    TPUPrefillExecutionPolicyConfig policy;
    policy.pcpSize = 8;
    policy.interleaveSize = 256;
    policy.compileSizes = {512, 1024, 2048, 4096};
    policy.maxScheduledTokens = 32768;
    policy.longPrefillTokenThreshold = 32768;
    policy.hashBlockSize = 6144;
    policy.schedulerBlockSize = 6144;
    policy.useEagle = true;
    TPUPrefillExecutionProjection projection;
    ASSERT_TRUE(TPUPrefillExecutionPolicy::projectCacheResolvedBatch(
        {{7050, hit, boundary}}, policy, 2, projection));
    EXPECT_EQ(1, projection.stepCount);
    EXPECT_EQ(8192, projection.globalPaddedTokenSum);

    request.promptTokenCount = 14000;
    node._fullReadyBlockHashes = {11, 22};
    ASSERT_TRUE(node.getExecutionCacheInputs(request, hit, boundary));
    EXPECT_EQ(0, hit);
    EXPECT_EQ(6144, boundary); // Post-Eagle Full1 still exceeds sparse Tail0.
    node._tailReadyBlockHashes = {11};
    ASSERT_TRUE(node.getExecutionCacheInputs(request, hit, boundary));
    EXPECT_EQ(6144, hit);
    EXPECT_EQ(0, boundary); // Equal reconciled hit has no uncached junction.

    node._fullReadyBlockHashes.clear();
    node._tailReadyBlockHashes.clear();
    node._v6dCacheState._cachePolicy.blockSize = 6144;
    node._v6dCacheState._cachePolicy.linearStep = 1;
    node._v6dCacheState._cachePolicyReady.store(true);
    node._v6dCacheState._hybridTailRequired = true;
    node._v6dCacheState._fullReadyBlockHashes = {11, 22};
    ASSERT_TRUE(node.getExecutionCacheInputs(request, hit, boundary));
    EXPECT_EQ(0, hit);
    EXPECT_EQ(0, boundary); // External Full-only objects are not local junctions.
}

TEST_F(GeometryIntegrationTest, testBatchExecutionCacheMatchesScalarAcrossBothTiers)
{
    for (bool hybrid : {false, true}) {
        for (int64_t dropBlocks : {0, 1, 2}) {
            for (int64_t step : {1, 2, 3}) {
                for (int64_t externalDrop : {0, 1}) {
                    for (int overlay = 0; overlay < 4; ++overlay) {
                        SCOPED_TRACE(
                            std::to_string(hybrid) + ":" + std::to_string(dropBlocks) +
                            ":" + std::to_string(step) + ":" +
                            std::to_string(externalDrop) + ":" + std::to_string(overlay));
                        MasterPendingCacheR pending;
                        MasterStateR::NodeState node;
                        node._pendingCacheR = &pending;
                        node._info.agentIdHash = 123;
                        node._cachePolicy = {32, 1, hybrid};
                        node._cachePolicy.fullAttentionQueryDropTokens = dropBlocks * 32;
                        node._cachePolicyReady.store(true);
                        node._hybridTailRequired = hybrid;
                        node._cachedBlockHashes = {11, 22, 33};
                        node._fullReadyBlockHashes = {11, 44, 55, 66};
                        node._tailReadyBlockHashes = {11, 55};
                        auto &external = node._v6dCacheState;
                        external._cachePolicy = {32, step, hybrid};
                        external._cachePolicy.fullAttentionQueryDropTokens = externalDrop;
                        external._cachePolicyReady.store(true);
                        external._hybridTailRequired = hybrid;
                        external._cachedBlockHashes = {11, 22, 33, 44, 55, 66};
                        external._fullReadyBlockHashes = {11, 22, 33, 66};
                        external._tailReadyBlockHashes = {22, 66};
                        for (const auto type : {NaviSchedCacheKeyType::BLOCK_POOL,
                                                NaviSchedCacheKeyType::V6D}) {
                            NaviSchedPendingCacheReservation reservation;
                            reservation.batchId = 7;
                            reservation.scheduleBatchId = 7;
                            reservation.fullReadyBlockHashes =
                                type == NaviSchedCacheKeyType::BLOCK_POOL
                                    ? NaviSchedBlockHashes{22, 33}
                                    : NaviSchedBlockHashes{44, 55};
                            reservation.tailReadyBlockHashes =
                                type == NaviSchedCacheKeyType::BLOCK_POOL
                                    ? NaviSchedBlockHashes{33}
                                    : NaviSchedBlockHashes{55};
                            if (overlay & 1) {
                                ASSERT_TRUE(
                                    pending.updateScheduled(123, reservation, type));
                            }
                            if (overlay & 2) {
                                auto cache = pending.getOrCreateNodeCache(123, type);
                                bool changed = false;
                                ASSERT_TRUE(
                                    cache->agent.upsert(reservation, true, changed));
                            }
                        }
                        std::vector<NaviSchedScheduleItem> requests;
                        const NaviSchedBlockHashes hashes{11, 22, 33, 44, 55, 66};
                        for (int64_t prompt : {1, 32, 33, 63, 64, 96, 144, 192, 200}) {
                            NaviSchedScheduleItem item;
                            item.request.blockSize = 32;
                            item.request.promptTokenCount = prompt;
                            item.request.blockHashes =
                                std::make_shared<const NaviSchedBlockHashes>(
                                    hashes.begin(),
                                    hashes.begin() +
                                        std::min<size_t>(hashes.size(), prompt / 32));
                            requests.push_back(item);
                        }
                        auto reference = [&](const NaviSchedScheduleRequest &request,
                                             int64_t &computed, int64_t &boundary,
                                             int64_t &externalBlocks) {
                            const auto local = pending.getMatchBreakdown(
                                123, BlockHashUtil::get(request.blockHashes),
                                node._cachedBlockHashes, node._fullReadyBlockHashes,
                                node._tailReadyBlockHashes, node._hybridTailRequired,
                                BlockHashUtil::getMatchableBlockCount(
                                    request.promptTokenCount, request.blockSize,
                                    request.blockHashes),
                                dropBlocks);
                            const auto suffix = node.matchExternalSuffix(
                                BlockHashUtil::get(request.blockHashes),
                                request.promptTokenCount, request.blockSize,
                                local.matchedBlockCount);
                            externalBlocks = suffix;
                            computed =
                                (local.matchedBlockCount + suffix) * request.blockSize;
                            boundary = local.effectiveFullMatchedBlockCount >
                                               local.matchedBlockCount
                                           ? local.effectiveFullMatchedBlockCount *
                                                 request.blockSize
                                           : 0;
                        };
                        std::vector<TPUPrefillRequestInput> batch;
                        ASSERT_TRUE(node.getExecutionCacheInputs(requests, batch));
                        ASSERT_EQ(requests.size(), batch.size());
                        std::vector<int64_t> ordinary(requests.size());
                        node.getMatchedBlockCounts(requests, ordinary.data());
                        std::vector<int64_t> splitTotal(requests.size(), -1);
                        std::vector<int64_t> ordinaryExternal(requests.size(), -1);
                        std::vector<int64_t> geometryExternal(requests.size(), -1);
                        node.getMatchedBlockCounts(requests, splitTotal.data(),
                                                   ordinaryExternal.data());
                        std::vector<TPUPrefillRequestInput> splitBatch;
                        ASSERT_TRUE(node.getExecutionCacheInputs(
                            requests, splitBatch, geometryExternal.data()));
                        for (size_t i = 0; i < requests.size(); ++i) {
                            int64_t computed = -1, boundary = -1, suffix = -1;
                            reference(requests[i].request, computed, boundary, suffix);
                            EXPECT_EQ(requests[i].request.promptTokenCount,
                                      batch[i].promptTokens);
                            EXPECT_EQ(computed, batch[i].computedTokens);
                            EXPECT_EQ(boundary, batch[i].sharedPrefixBoundary);
                            EXPECT_EQ(computed / requests[i].request.blockSize,
                                      ordinary[i]);
                            EXPECT_EQ(ordinary[i], splitTotal[i]);
                            EXPECT_EQ(batch[i].computedTokens,
                                      splitBatch[i].computedTokens);
                            EXPECT_EQ(batch[i].sharedPrefixBoundary,
                                      splitBatch[i].sharedPrefixBoundary);
                            EXPECT_EQ(suffix, ordinaryExternal[i]);
                            EXPECT_EQ(suffix, geometryExternal[i]);
                            EXPECT_EQ(suffix * requests[i].request.blockSize,
                                      batch[i].externalReadTokens);
                            EXPECT_EQ(batch[i].externalReadTokens,
                                      splitBatch[i].externalReadTokens);
                            EXPECT_GE(ordinary[i], suffix);
                        }
                        external._cachePolicyReady.store(false);
                        ASSERT_TRUE(node.getExecutionCacheInputs(
                            requests, batch, geometryExternal.data()));
                        node.getMatchedBlockCounts(requests, splitTotal.data(),
                                                   ordinaryExternal.data());
                        for (size_t i = 0; i < requests.size(); ++i) {
                            int64_t computed = -1, boundary = -1, suffix = -1;
                            reference(requests[i].request, computed, boundary, suffix);
                            EXPECT_EQ(computed, batch[i].computedTokens);
                            EXPECT_EQ(boundary, batch[i].sharedPrefixBoundary);
                            EXPECT_EQ(0, suffix);
                            EXPECT_EQ(0, ordinaryExternal[i]);
                            EXPECT_EQ(0, geometryExternal[i]);
                            EXPECT_EQ(computed / requests[i].request.blockSize,
                                      splitTotal[i]);
                        }
                        requests.back().request.blockSize = 16;
                        EXPECT_FALSE(node.getExecutionCacheInputs(requests, batch));
                        node._cachePolicyReady.store(false);
                        EXPECT_FALSE(node.getExecutionCacheInputs(requests, batch));
                    }
                }
            }
        }
    }
}

TEST_F(GeometryIntegrationTest,
       testDisabledResidualSkipsValidationAndPreservesDiscreteLoss)
{
    for (const int64_t version :
         {NonLinearPrefillModelR::PD_BINARY_VERSION,
          NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION,
          NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION}) {
        auto makeContext = [&](double rate, int damage) {
            auto context = std::make_shared<NaviSchedScheduleContext>();
            std::vector<NaviSchedNodeConstPtr> nodes;
            std::vector<NaviSchedNodeSnapshot> snapshots;
            for (size_t i = 0; i < 2; ++i) {
                nodes.push_back(std::make_shared<GeometryMissingPolicyNode>());
                NaviSchedNodeSnapshot snapshot;
                auto info = std::make_shared<NaviSchedNodeInfo>();
                info->pdSeparated = true;
                snapshot.info = info;
                auto model = std::make_shared<NaviSchedLatencyModel>();
                model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
                model->binaryVersion = version;
                NonLinearPrefillModelR initial;
                initial._pdSeparated = true;
                model->parameters = initial.getInitialParameters();
                if (damage == 1) {
                    model->residualVersion = -1;
                }
                if (damage == 2) {
                    model->residualParameters.clear();
                }
                if (damage == 3) {
                    model->residualParameters[2] =
                        std::numeric_limits<double>::quiet_NaN();
                }
                if (damage == 4) {
                    model->residualLastUpdateTimeUs = -1;
                }
                if (damage == 5) {
                    model->parameters[0] = std::numeric_limits<double>::quiet_NaN();
                }
                snapshot.latencyModel = model;
                snapshot.prefillExecutionPolicy = typedRuntimePolicy();
                snapshot.residualRuntimeConfig.learningRate = rate;
                snapshots.push_back(snapshot);
            }
            for (size_t i = 0; i < 3; ++i) {
                NaviSchedScheduleItem item;
                item.request.requestId = i + 1;
                item.request.promptTokenCount = 400 + i * 200;
                item.request.blockSize = 1;
                context->_requests.push_back(item);
            }
            context->setNodes(std::move(nodes), std::move(snapshots));
            return context;
        };
        const auto baseline = makeContext(0.0, 0);
        ASSERT_TRUE(baseline->prepare());
        ASSERT_TRUE(makeContext(0.8, 0)->getNode(0).hasValidLatencyModel());
        EXPECT_FALSE(makeContext(0.0, 5)->getNode(0).hasValidLatencyModel());
        std::vector<int64_t> tokens(3);
        std::vector<double> inverse(3);
        const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
            *baseline, tokens.data(), inverse.data());
        const std::vector<float> costs(12, 3.0f);
        for (int damage : {1, 2, 3, 4}) {
            SCOPED_TRACE(version);
            SCOPED_TRACE(damage);
            auto disabled = makeContext(0.0, damage);
            ASSERT_TRUE(disabled->getNode(0).hasValidLatencyModel());
            ASSERT_TRUE(disabled->prepare());
            EXPECT_FALSE(makeContext(0.8, damage)->getNode(0).hasValidLatencyModel());
            double derivative = 0.0;
            EXPECT_EQ(123.0,
                      disabled->getNode(0).applyLatencyResidual(123.0, &derivative));
            EXPECT_EQ(1.0, derivative);
            std::vector<uint32_t> assignments{0, 0, 1};
            auto score = [&](const NaviSchedScheduleContext &context) {
                CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
                return CostSchedulerKernelInternal::forwardDiscrete(
                    context, costs.data(), assignments.data(), tokens.data(),
                    inverse.data(), total, 0.5, 0.0, workspace);
            };
            const auto initial = score(*baseline);
            ASSERT_TRUE(std::isfinite(initial));
            EXPECT_EQ(initial, score(*disabled));
            for (size_t moved = 0; moved < assignments.size(); ++moved) {
                assignments[moved] = 1 - assignments[moved];
                EXPECT_EQ(score(*baseline), score(*disabled));
                EXPECT_EQ(score(*baseline) - initial, score(*disabled) - initial);
                assignments[moved] = 1 - assignments[moved];
            }
        }
    }
}

TEST_F(GeometryIntegrationTest, testPairLossOnlyReadsAndUpdatesTwoNodes)
{
    for (int64_t version : {NonLinearPrefillModelR::PD_BINARY_VERSION,
                            NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION}) {
        NaviSchedScheduleContext context;
        std::vector<NaviSchedNodeConstPtr> nodes;
        std::vector<NaviSchedNodeSnapshot> snapshots;
        for (size_t i = 0; i < 6; ++i) {
            nodes.push_back(std::make_shared<GeometryMissingPolicyNode>());
            NaviSchedNodeSnapshot snapshot;
            auto info = std::make_shared<NaviSchedNodeInfo>();
            info->pdSeparated = true;
            snapshot.info = info;
            auto model = std::make_shared<NaviSchedLatencyModel>();
            model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
            model->binaryVersion = version;
            NonLinearPrefillModelR initial;
            initial._pdSeparated = true;
            model->parameters = initial.getInitialParameters();
            snapshot.latencyModel = model;
            snapshot.prefillExecutionPolicy = typedRuntimePolicy();
            snapshots.push_back(snapshot);
        }
        std::vector<uint32_t> original;
        for (size_t i = 0; i < 18; ++i) {
            NaviSchedScheduleItem item;
            item.request.requestId = i + 1;
            item.request.promptTokenCount = (i + 1) * 200;
            item.request.blockSize = 1;
            context._requests.push_back(item);
            original.push_back(i % nodes.size());
        }
        context.setNodes(std::move(nodes), std::move(snapshots));
        ASSERT_TRUE(context.prepare());
        std::vector<int64_t> tokens(original.size());
        std::vector<double> inverse(original.size());
        const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
            context, tokens.data(), inverse.data());
        std::vector<float> costs(context.nodeCount() * original.size() * 2, 3.0f);
        CostSchedulerKernelInternal::DiscreteForwardWorkspace baseline;
        const auto oldLoss = CostSchedulerKernelInternal::forwardDiscrete(
            context, costs.data(), original.data(), tokens.data(), inverse.data(), total,
            0.5, 0.0, baseline);
        ASSERT_TRUE(std::isfinite(oldLoss));
        for (size_t moved = 0; moved < original.size(); ++moved) {
            const auto source = original[moved];
            for (size_t target = 0; target < context.nodeCount(); ++target) {
                if (source == target) {
                    continue;
                }
                auto candidate = original;
                candidate[moved] = target;
                const auto completeCandidate = candidate;
                CostSchedulerKernelInternal::DiscreteForwardWorkspace full;
                const auto newLoss = CostSchedulerKernelInternal::forwardDiscrete(
                    context, costs.data(), candidate.data(), tokens.data(),
                    inverse.data(), total, 0.5, 0.0, full);
                CostSchedulerKernelInternal::DiscreteForwardWorkspace legacy;
                const auto legacyPair = discrete_baseline::pairLoss(
                    context, costs.data(), candidate.data(), tokens.data(),
                    inverse.data(), total, 0.5, 0.0, legacy, source, target);
                std::vector<size_t> pairRequests;
                for (size_t i = 0; i < original.size(); ++i) {
                    if (original[i] == source || original[i] == target) {
                        pairRequests.push_back(i);
                    } else {
                        candidate[i] = std::numeric_limits<uint32_t>::max();
                    }
                }
                std::vector<std::vector<size_t>> nodeRequests(context.nodeCount());
                for (const auto index : pairRequests) {
                    nodeRequests[original[index]].push_back(index);
                }
                auto pairWorkspace = baseline;
                const auto pairLoss = CostSchedulerKernelInternal::forwardDiscreteImpl(
                    context, costs.data(), candidate.data(), tokens.data(),
                    inverse.data(), total, 0.5, 0.0, pairWorkspace, source, target,
                    &nodeRequests, moved);
                EXPECT_EQ(legacyPair, pairLoss);
                EXPECT_EQ(legacy.nodeValues[source], pairWorkspace.nodeValues[source]);
                EXPECT_EQ(legacy.nodeValues[target], pairWorkspace.nodeValues[target]);
                EXPECT_EQ(
                    legacyPair - baseline.nodeValues[source] -
                        baseline.nodeValues[target],
                    pairLoss - baseline.nodeValues[source] - baseline.nodeValues[target]);
                const auto sourceOnly = CostSchedulerKernelInternal::forwardDiscreteImpl(
                    context, costs.data(), completeCandidate.data(), tokens.data(),
                    inverse.data(), total, 0.5, 0.0, pairWorkspace, source, source,
                    nullptr);
                const auto targetOnly = CostSchedulerKernelInternal::forwardDiscreteImpl(
                    context, costs.data(), completeCandidate.data(), tokens.data(),
                    inverse.data(), total, 0.5, 0.0, pairWorkspace, target, target,
                    nullptr);
                EXPECT_EQ(legacy.nodeValues[source], sourceOnly);
                EXPECT_EQ(legacy.nodeValues[target], targetOnly);
                EXPECT_EQ(legacyPair, sourceOnly + targetOnly);
                EXPECT_EQ(legacyPair - baseline.nodeValues[source] -
                              baseline.nodeValues[target],
                          sourceOnly + targetOnly - baseline.nodeValues[source] -
                              baseline.nodeValues[target]);
                EXPECT_NEAR(newLoss,
                            oldLoss + pairLoss - baseline.nodeValues[source] -
                                baseline.nodeValues[target],
                            1e-9);
                for (size_t node = 0; node < context.nodeCount(); ++node) {
                    if (node == source || node == target) {
                        continue;
                    }
                    EXPECT_DOUBLE_EQ(baseline.nodeValues[node],
                                     pairWorkspace.nodeValues[node]);
                    EXPECT_DOUBLE_EQ(baseline.requestWeightSums[node],
                                     pairWorkspace.requestWeightSums[node]);
                    EXPECT_DOUBLE_EQ(baseline.requestCostSums[node],
                                     pairWorkspace.requestCostSums[node]);
                    EXPECT_DOUBLE_EQ(
                        baseline.requestCostSumsWithoutReportedWorkload[node],
                        pairWorkspace.requestCostSumsWithoutReportedWorkload[node]);
                    EXPECT_DOUBLE_EQ(baseline.totalPromptTokenSums[node],
                                     pairWorkspace.totalPromptTokenSums[node]);
                    EXPECT_DOUBLE_EQ(baseline.inverseTokenWeightSums[node],
                                     pairWorkspace.inverseTokenWeightSums[node]);
                    EXPECT_EQ(baseline.geometryRequests[node],
                              pairWorkspace.geometryRequests[node]);
                }
            }
        }
    }
}

TEST_F(GeometryIntegrationTest, testGeometryAndOrdinaryBatchCachePerformance)
{
    constexpr size_t nodeCount = 200, requestCount = 64;
    const auto roundsEnv = std::getenv("NAVI_CACHE_PERF_ROUNDS");
    const size_t rounds = roundsEnv ? std::stoul(roundsEnv) : 8;
    ASSERT_GE(rounds, 1u);
    ASSERT_LE(rounds, 4096u);
    for (bool withExternal : {false, true}) {
        for (bool hybrid : {false, true}) {
            for (size_t prefix : {0, 4, 128}) {
                if (const auto selected = std::getenv("NAVI_CACHE_PERF_PREFIX")) {
                    if (prefix != std::stoul(selected)) {
                        continue;
                    }
                }
                SCOPED_TRACE(prefix);
                SCOPED_TRACE(hybrid);
                MasterPendingCacheR pending;
                std::vector<std::unique_ptr<MasterStateR::NodeState>> nodes;
                auto hashes = std::make_shared<NaviSchedBlockHashes>();
                for (uint64_t hash = 1; hash <= 128; ++hash) {
                    hashes->push_back(hash);
                }
                for (size_t i = 0; i < nodeCount; ++i) {
                    auto node = std::make_unique<MasterStateR::NodeState>();
                    node->_pendingCacheR = &pending;
                    node->_info.agentIdHash = i + 1;
                    node->_cachePolicy = {32, 1, hybrid};
                    node->_hybridTailRequired = hybrid;
                    node->_cachePolicyReady.store(true);
                    if (withExternal) {
                        auto &external = node->_v6dCacheState;
                        external._cachePolicy = {32, 1, hybrid};
                        external._hybridTailRequired = hybrid;
                        external._cachePolicyReady.store(true);
                        external._cachedBlockHashes.insert(hashes->begin(),
                                                           hashes->end());
                        external._fullReadyBlockHashes = external._cachedBlockHashes;
                        if (hybrid) {
                            external._tailReadyBlockHashes = external._cachedBlockHashes;
                        }
                    }
                    NaviSchedPendingCacheReservation reservation;
                    reservation.batchId = 1;
                    reservation.scheduleBatchId = 1;
                    reservation.fullReadyBlockHashes.assign(hashes->begin(),
                                                            hashes->begin() + prefix);
                    if (hybrid) {
                        reservation.tailReadyBlockHashes =
                            reservation.fullReadyBlockHashes;
                    }
                    if (prefix) {
                        ASSERT_TRUE(pending.updateScheduled(i + 1, reservation));
                    }
                    nodes.push_back(std::move(node));
                }
                std::vector<NaviSchedScheduleItem> requests(requestCount);
                for (auto &item : requests) {
                    item.request.blockSize = 32;
                    item.request.promptTokenCount = 4097;
                    item.request.blockHashes = hashes;
                }
                auto measure = [&](bool geometry, bool split = false) {
                    std::vector<int64_t> hits(requestCount);
                    std::vector<int64_t> suffix(requestCount);
                    std::vector<TPUPrefillRequestInput> inputs;
                    inputs.reserve(requestCount);
                    TPUPrefillCacheObservationBatch observation;
                    uint64_t checksum = 0;
                    uint64_t suffixChecksum = 0;
                    const auto start = std::chrono::steady_clock::now();
                    for (size_t round = 0; round < rounds; ++round) {
                        for (const auto &node : nodes) {
                            if (geometry) {
                                if (!node->getExecutionCacheInputs(
                                        requests, inputs, split ? suffix.data() : nullptr,
                                        &observation)) {
                                    return int64_t(-1);
                                }
                                for (const auto &input : inputs) {
                                    checksum += input.computedTokens / 32;
                                }
                            } else {
                                node->getMatchedBlockCounts(
                                    requests, hits.data(),
                                    split ? suffix.data() : nullptr);
                                for (const auto hit : hits) {
                                    checksum += hit;
                                }
                            }
                            if (split) {
                                for (const auto count : suffix) {
                                    suffixChecksum += count;
                                }
                            }
                        }
                    }
                    const auto elapsed =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
                    const auto total = withExternal ? 128 : prefix;
                    const auto expectedSuffix = withExternal ? 128 - prefix : 0;
                    return checksum == nodeCount * rounds * requestCount * total &&
                                   (!split || suffixChecksum == nodeCount * rounds *
                                                                    requestCount *
                                                                    expectedSuffix)
                               ? static_cast<int64_t>(elapsed)
                               : int64_t(-1);
                };
                ASSERT_GT(measure(false), 0);
                ASSERT_GT(measure(true), 0);
                std::vector<int64_t> ordinary, geometry, ordinarySplit, geometrySplit;
                for (size_t sample = 0; sample < 10; ++sample) {
                    if (sample % 2 == 0) {
                        ordinary.push_back(measure(false));
                        ordinarySplit.push_back(measure(false, true));
                        geometry.push_back(measure(true));
                        geometrySplit.push_back(measure(true, true));
                    } else {
                        geometrySplit.push_back(measure(true, true));
                        geometry.push_back(measure(true));
                        ordinarySplit.push_back(measure(false, true));
                        ordinary.push_back(measure(false));
                    }
                    ASSERT_GT(ordinary.back(), 0);
                    ASSERT_GT(geometry.back(), 0);
                    ASSERT_GT(ordinarySplit.back(), 0);
                    ASSERT_GT(geometrySplit.back(), 0);
                    std::cout << "batch_cache_perf hybrid=" << hybrid
                              << " prefix=" << prefix << " sample=" << sample
                              << " external=" << withExternal
                              << " ordinary_ns=" << ordinary.back()
                              << " geometry_ns=" << geometry.back()
                              << " ordinary_split_ns=" << ordinarySplit.back()
                              << " geometry_split_ns=" << geometrySplit.back()
                              << std::endl;
                }
                std::sort(ordinary.begin(), ordinary.end());
                std::sort(geometry.begin(), geometry.end());
                std::sort(ordinarySplit.begin(), ordinarySplit.end());
                std::sort(geometrySplit.begin(), geometrySplit.end());
                const auto ordinaryMedian = (ordinary[4] + ordinary[5]) / 2;
                const auto geometryMedian = (geometry[4] + geometry[5]) / 2;
                EXPECT_LE(geometryMedian, ordinaryMedian * 1.30)
                    << "Geometry batch cache reads must stay within 30% of ordinary "
                       "reads";
                EXPECT_LE((ordinarySplit[4] + ordinarySplit[5]) / 2,
                          ordinaryMedian * 1.30);
                EXPECT_LE((geometrySplit[4] + geometrySplit[5]) / 2,
                          geometryMedian * 1.30);
            }
        }
    }
}

TEST_F(GeometryIntegrationTest, testBatchExecutionCacheHasOnePendingObservation)
{
    MasterPendingCacheR pending;
    MasterStateR::NodeState node;
    node._pendingCacheR = &pending;
    node._info.agentIdHash = 123;
    node._cachePolicy = {32, 1, false};
    node._cachePolicyReady.store(true);
    NaviSchedPendingCacheReservation initial;
    initial.batchId = 7;
    initial.scheduleBatchId = 7;
    initial.fullReadyBlockHashes = {11};
    ASSERT_TRUE(pending.updateScheduled(123, initial));
    std::vector<NaviSchedScheduleItem> requests(64);
    for (auto &item : requests) {
        item.request.blockSize = 32;
        item.request.promptTokenCount = 129;
        item.request.blockHashes = std::make_shared<const NaviSchedBlockHashes>(
            NaviSchedBlockHashes{11, 22, 33, 44});
    }
    std::atomic<bool> stop{false}, updateFailed{false};
    std::thread writer([&] {
        auto reservation = initial;
        for (size_t i = 0; !stop.load(); ++i) {
            reservation.fullReadyBlockHashes =
                i % 2 ? NaviSchedBlockHashes{11} : NaviSchedBlockHashes{11, 22};
            if (!pending.updateScheduled(123, reservation)) {
                updateFailed.store(true);
            }
        }
    });
    for (size_t iteration = 0; iteration < 100; ++iteration) {
        std::vector<TPUPrefillRequestInput> inputs;
        const auto valid = node.getExecutionCacheInputs(requests, inputs);
        EXPECT_TRUE(valid);
        if (!valid || inputs.size() != requests.size()) {
            break;
        }
        const auto computed = inputs.front().computedTokens;
        EXPECT_TRUE(computed == 32 || computed == 64);
        for (const auto &input : inputs) {
            EXPECT_EQ(computed, input.computedTokens);
            EXPECT_EQ(0, input.sharedPrefixBoundary);
        }
    }
    stop.store(true);
    writer.join();
    EXPECT_FALSE(updateFailed.load());
}

TEST_F(GeometryIntegrationTest, testEffectiveFullPrefixPreservesBothPendingSources)
{
    for (bool agentSource : {false, true}) {
        MasterPendingCacheR pending;
        NaviSchedPendingCacheReservation reservation;
        reservation.batchId = 7;
        reservation.scheduleBatchId = 7;
        reservation.fullReadyBlockHashes = {11, 22};
        if (agentSource) {
            auto cache =
                pending.getOrCreateNodeCache(123, NaviSchedCacheKeyType::BLOCK_POOL);
            bool changed = false;
            ASSERT_TRUE(cache->agent.upsert(reservation, true, changed));
        } else {
            ASSERT_TRUE(pending.updateScheduled(123, reservation));
        }
        const NaviSchedBlockHashes hashes{11, 22};
        const MasterPendingCacheR::PhysicalBlockSet empty;
        const auto match =
            pending.getMatchBreakdown(123, hashes, empty, empty, empty, true, 2, 1);
        EXPECT_EQ(2, match.fullMatchedBlockCount);
        EXPECT_EQ(1, match.effectiveFullMatchedBlockCount);
        EXPECT_EQ(0, match.matchedBlockCount);
        EXPECT_EQ(0, match.staticMatchedBlockCount);
    }
}

TEST_F(GeometryIntegrationTest,
       testProjectedGeometryKeepsActualTelemetryAndOriginSeparate)
{
    AgentLatencyInput input;
    input.latencyMs = 217.25;
    input.executionStepCount = 3;
    input.globalPaddedTokens = 24576;
    NaviSchedRequestStats first;
    first.requestId = 1;
    first.totalTokenCount = 7050;
    first.cacheHitTokenCount = 0;
    NaviSchedRequestStats second = first;
    second.requestId = 2;
    second.totalTokenCount = 6820;
    second.cacheHitTokenCount = 6144;
    input.queries = {first, second};
    input.projectedExecutionStepCount = 2;
    input.projectedGlobalPaddedTokens = 12288;
    input.geometryFeatureOriginVersion =
        AgentLatencyInput::PROJECTED_GEOMETRY_FEATURE_VERSION;
    EXPECT_TRUE(input.hasCompatibleGeometryFeatureOrigin(
        NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION));
    EXPECT_FALSE(input.hasCompatibleGeometryFeatureOrigin(
        NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION));
    EXPECT_FALSE(input.hasCompatibleGeometryFeatureOrigin(29));
    EXPECT_FALSE(input.hasCompatibleGeometryFeatureOrigin(25));
    const auto sample = input.makeLatencySample();
    EXPECT_EQ(2, sample.features.getExecutionStepCount());
    EXPECT_EQ(12288, sample.features.getGlobalPaddedTokens());
    EXPECT_EQ(6144, sample.features.getCacheHitTokenCount());
    EXPECT_EQ(7726, sample.features.getCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(217.25, sample.latencyMs);
    EXPECT_EQ(3, input.executionStepCount);
    EXPECT_EQ(24576, input.globalPaddedTokens);
    input.globalPaddedTokens =
        0; // Missing observed padding never fabricates modeled padding.
    EXPECT_EQ(12288, input.makeLatencySample().features.getGlobalPaddedTokens());
    input.projectedExecutionStepCount = 0;
    EXPECT_FALSE(input.hasCompatibleGeometryFeatureOrigin(
        NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION));
    input.geometryFeatureOriginVersion = 26;
    input.globalPaddedTokens = 24576;
    EXPECT_TRUE(input.hasCompatibleGeometryFeatureOrigin(26));
    EXPECT_FALSE(input.hasCompatibleGeometryFeatureOrigin(
        NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION));
    EXPECT_EQ(3, input.makeLatencySample().features.getExecutionStepCount());
    EXPECT_EQ(24576, input.makeLatencySample().features.getGlobalPaddedTokens());
    input.geometryFeatureOriginVersion = 0;
    input.executionStepCount = 0;
    input.globalPaddedTokens = 0;
    EXPECT_TRUE(input.hasCompatibleGeometryFeatureOrigin(24));
    EXPECT_TRUE(input.hasCompatibleGeometryFeatureOrigin(25));
    EXPECT_FALSE(input.hasCompatibleGeometryFeatureOrigin(
        NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION));
    EXPECT_FALSE(input.makeLatencySample().features.hasExecutionGeometry());
}

TEST_F(GeometryIntegrationTest, testProjectedGeometryInputUsesDistinctCompletionChannel)
{
    AgentLatencyInputR resource;
    NaviSchedRequestStats request;
    request.requestId = 1;
    request.totalTokenCount = 7050;
    auto submit = [&](int64_t actualPad, int64_t projectedSteps, int64_t projectedPad) {
        return resource.submitBatchFinishedWithProjectedExecutionGeometry(
            {request}, 217.25, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 1, {}, -1.0,
            true, true, {}, 3, actualPad, projectedSteps, projectedPad);
    };
    EXPECT_FALSE(submit(24576, 0, 12288));
    EXPECT_FALSE(submit(24576, 3, 2));
    ASSERT_TRUE(submit(24576, 1, 8192));
    ASSERT_TRUE(submit(0, 1, 8192));
    std::vector<AgentLatencyInput> inputs;
    resource.take(inputs);
    ASSERT_EQ(2u, inputs.size());
    for (const auto &input : inputs) {
        EXPECT_EQ(29, input.geometryFeatureOriginVersion);
        EXPECT_EQ(3, input.executionStepCount);
        EXPECT_EQ(1, input.makeLatencySample().features.getExecutionStepCount());
        EXPECT_EQ(8192, input.makeLatencySample().features.getGlobalPaddedTokens());
        EXPECT_DOUBLE_EQ(217.25, input.makeLatencySample().latencyMs);
    }
    EXPECT_EQ(24576, inputs[0].globalPaddedTokens);
    EXPECT_EQ(0, inputs[1].globalPaddedTokens);
}

TEST_F(GeometryIntegrationTest, testProjectedReporterFinishesWithoutObservedPadding)
{
    NaviSchedAgentConfig config;
    config.runtime.bizName = "projected_geometry_lifecycle_biz";
    config.runtime.threadNum = 2;
    config.service.name = "projected_geometry_lifecycle_test";
    config.service.enableKmon = false;
    config.service.gig.grpcPort = 0;
    config.service.enableSnapshot = false;
    config.service.latencyValidationEnabled = true;
    config.service.latencyFitBatchSize = 1;
    config.service.latencyWarmupSampleCount = 0;
    config.executionGeometryEnabled = true;
    config.projectedExecutionGeometryEnabled = true;
    config.prefillExecutionPolicy = typedRuntimePolicy();
    NaviSchedAgent agent(config);
    ASSERT_TRUE(agent.init());
    agent.reportCacheUpdated(-1, 256, 1, nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, 0, autil::TimeUtility::currentTime());
    NaviSchedBatchRequest request;
    request.requestId = 1;
    request.totalTokenCount = 7050;
    NaviSchedBatchPtr ready;
    ASSERT_TRUE(agent.addBatchRequest(
        request, 7, 1, [&ready](const NaviSchedBatchPtr &batch) { ready = batch; }));
    ASSERT_TRUE(ready);
    ASSERT_TRUE(ready->finishReporter);
    ASSERT_TRUE(ready->finishReporter->startBatch({1}));
    const auto before = agent._impl->_agentLatencyInputR->getStats();
    NaviSchedRequestStats completed;
    completed.requestId = 1;
    completed.totalTokenCount = 7050;
    // Breakdown actual count remains3. Missing observed geometry is pad0;
    // the bridge's separate geometry-validity count0 never replaces count3.
    ASSERT_TRUE(ready->finishReporter->finishBatchWithProjectedExecutionGeometry(
        {completed}, 217.25, 200.0, 3, 7050, 0, 0, -1.0, 0, 1, 8192));
    EXPECT_EQ(before.submittedFinishedRequestCount + 1,
              agent._impl->_agentLatencyInputR->getStats().submittedFinishedRequestCount);
    EXPECT_EQ(0, agent._impl->getInflightRequestCount());
    const auto report = agent._impl->_agentReportR;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto snapshot = report->getLatencyModelSnapshot();
        if (snapshot && snapshot->sessionSampleCount == 1 &&
            report->getLatencyValidationStats().sampleCount == 1) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const auto trained = report->getLatencyModelSnapshot();
    ASSERT_TRUE(trained);
    EXPECT_EQ(NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION,
              trained->binaryVersion);
    EXPECT_EQ(1u, trained->sessionSampleCount);
    EXPECT_GT(trained->revision, 1u);
    EXPECT_EQ(1u, report->getLatencyValidationStats().sampleCount);

    const auto observed =
        agent._impl->_agentBatchR->takeBackendBatchExecutionWindowStats();
    EXPECT_EQ(3, observed.executionStepCount);
    EXPECT_FALSE(ready->finishReporter->finishBatchWithProjectedExecutionGeometry(
        {completed}, 217.25, 200.0, 3, 7050, 0, 0, -1.0, 0, 1, 8192));
    agent.stop();
}

} // namespace navi_sched

namespace navi_sched {
namespace {
std::string makeProjectedHistoryConfig(const std::string &stateFile,
                                       const std::string &formulaFingerprint,
                                       const std::string &historyFingerprint,
                                       const std::string &binaryFingerprint)
{
    return std::string(R"({"state_file":")") + stateFile +
           R"(","formula_fingerprint":")" + formulaFingerprint +
           R"(","history_fingerprint":")" + historyFingerprint +
           R"(","binary_fingerprint":")" + binaryFingerprint + R"("})";
}

AgentHistoryRPtr makeProjectedHistory(const std::string &stateFile,
                                      const std::string &formulaFingerprint,
                                      const std::string &historyFingerprint,
                                      const std::string &binaryFingerprint)
{
    navi::NaviResourceHelper helper;
    if (!helper.config(
            AgentHistoryR::RESOURCE_ID,
            makeProjectedHistoryConfig(stateFile, formulaFingerprint, historyFingerprint,
                                       binaryFingerprint))) {
        return nullptr;
    }
    return helper.getOrCreateResPtr<AgentHistoryR>();
}

}
TEST_F(GeometryIntegrationTest, testProjected29State10RoundTripAndBinaryIsolation)
{
    const auto path = GET_TEMP_DATA_PATH() + "latency-history-state/geometry29.state";
    auto history = makeProjectedHistory(path, "formula29", "history29", "binary29");
    ASSERT_TRUE(history);
    navi::NaviResourceHelper sourceHelper;
    ASSERT_TRUE(sourceHelper.config(NonLinearPrefillModelR::RESOURCE_ID,
                                    R"({"pd_separated":true})"));
    auto model = sourceHelper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(model);
    model->setExecutionGeometryEnabled(true);
    model->setProjectedExecutionGeometryEnabled(true);
    ASSERT_EQ(NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION,
              model->getBinaryVersion());
    NaviSchedLatencySample sample;
    sample.features.appendRequest(1024, 4096);
    ASSERT_TRUE(sample.features.appendBatchCount(1));
    ASSERT_TRUE(sample.features.setExecutionGeometry(3, 8192));
    sample.latencyMs = 500;
    sample.eventTimeUs = autil::TimeUtility::currentTime();
    auto parameters = model->getInitialParameters();
    std::vector<double> updated;
    ASSERT_TRUE(model->fit({sample}, parameters, updated));
    bool fitReady = false, persistReady = false;
    ASSERT_TRUE(history->append(sample, fitReady, persistReady));
    auto saved = std::make_shared<NaviSchedLatencyModelSnapshot>();
    saved->formula = model->getFormula();
    saved->binaryVersion = model->getBinaryVersion();
    saved->parameters = updated;
    saved->revision = 17;
    saved->overloadPredictionTrusted = true;
    ASSERT_TRUE(history->save(saved, model->getOptimizerState()));
    {
        std::ifstream input(path, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
        autil::DataBuffer envelope(content.data(), content.size());
        std::string magic;
        int32_t version = 0;
        envelope.read(magic);
        envelope.read(version);
        EXPECT_EQ(10, version);
    }
    auto restored = makeProjectedHistory(path, "formula29", "history29", "binary29");
    ASSERT_TRUE(restored);
    navi::NaviResourceHelper loadHelper;
    ASSERT_TRUE(loadHelper.config(NonLinearPrefillModelR::RESOURCE_ID,
                                  R"({"pd_separated":true})"));
    auto loadedModel = loadHelper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(loadedModel);
    loadedModel->setExecutionGeometryEnabled(true);
    loadedModel->setProjectedExecutionGeometryEnabled(true);
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(restored->load(*loadedModel, snapshot, fitOnStart));
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION,
              snapshot->binaryVersion);
    EXPECT_EQ(updated, snapshot->parameters);
    EXPECT_EQ(model->getOptimizerState().step, loadedModel->getOptimizerState().step);
    ASSERT_EQ(1, restored->getHistory()->size());
    const auto &restoredSample = restored->getHistory()->front();
    EXPECT_EQ(3, restoredSample.features.getExecutionStepCount());
    EXPECT_EQ(8192, restoredSample.features.getGlobalPaddedTokens());
    std::vector<double> refit;
    ASSERT_TRUE(loadedModel->fit({restoredSample}, snapshot->parameters, refit));
    // State9 from the previous projected release must not warm-start version29.
    {
        std::ifstream input(path, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
        autil::DataBuffer original(content.data(), content.size());
        std::string magic, payload;
        int32_t version = 0;
        uint32_t checksum = 0;
        original.read(magic);
        original.read(version);
        original.read(checksum);
        original.read(payload);
        autil::DataBuffer previous;
        previous.write(magic);
        previous.write(int32_t(9));
        previous.write(checksum);
        previous.write(payload);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(previous.getData(), previous.getDataLen());
    }
    ASSERT_TRUE(restored->load(*loadedModel, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(restored->getHistory()->empty());
    // Even identical fingerprints cannot restore actual26 state into projected29.
    saved->binaryVersion = NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION;
    ASSERT_TRUE(history->save(saved, model->getOptimizerState()));
    {
        std::ifstream input(path, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
        autil::DataBuffer envelope(content.data(), content.size());
        std::string magic;
        int32_t version = 0;
        envelope.read(magic);
        envelope.read(version);
        EXPECT_EQ(7, version);
    }
    ASSERT_TRUE(restored->load(*loadedModel, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(restored->getHistory()->empty());
    // Identical fingerprints cannot bypass the 25/29 envelope separation.
    loadedModel->setExecutionGeometryEnabled(false);
    ASSERT_EQ(NonLinearPrefillModelR::PD_BINARY_VERSION, loadedModel->getBinaryVersion());
    ASSERT_TRUE(restored->load(*loadedModel, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(restored->getHistory()->empty());
    saved->binaryVersion = NonLinearPrefillModelR::PD_BINARY_VERSION;
    saved->parameters = loadedModel->getInitialParameters();
    ASSERT_TRUE(history->save(saved, loadedModel->getOptimizerState()));
    {
        std::ifstream input(path, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
        autil::DataBuffer envelope(content.data(), content.size());
        std::string magic;
        int32_t version = 0;
        envelope.read(magic);
        envelope.read(version);
        EXPECT_EQ(6, version);
    }
    loadedModel->setExecutionGeometryEnabled(true);
    ASSERT_TRUE(restored->load(*loadedModel, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(restored->getHistory()->empty());
}

TEST_F(GeometryIntegrationTest, testProjected29GlobalZeroSignatureIsolation)
{
    using Signature = MasterLatencyModelR::ModelSignature;
    const Signature old25{0, 25, 10}, old26{0, 26, 10}, old27{0, 27, 10},
        old28{0, 28, 10}, projected29{0, 29, 10};
    EXPECT_TRUE(old25 < projected29 || projected29 < old25);
    EXPECT_TRUE(old26 < projected29 || projected29 < old26);
    EXPECT_TRUE(old27 < projected29 || projected29 < old27);
    EXPECT_TRUE(old28 < projected29 || projected29 < old28);
    EXPECT_FALSE(projected29 < projected29);
}
} // namespace navi_sched

namespace navi_sched {
TEST_F(GeometryIntegrationTest, testProjected29MasterFinalBatchMatchesAgentPureFeatures)
{
    class MixedLookupNode : public GeometryMissingPolicyNode {
    public:
        bool getExecutionCacheInputs(const NaviSchedScheduleRequest &request,
                                     int64_t &hit, int64_t &boundary) const override
        {
            hit = request.requestId == 3 ? 6144 : 0;
            boundary = 0;
            return true;
        }
    };
    const std::vector<TPUPrefillRequestInput> requests = {
        {604, 0, 0}, {8626, 0, 0}, {12238, 6144, 0}};
    for (const auto &mode :
         {std::pair<int64_t, bool>{
              NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION, true},
          {NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION, false},
          {NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION, true}}) {
        TPUPrefillExecutionPolicyConfig policy;
        policy.pcpSize = 8;
        policy.interleaveSize = 256;
        policy.compileSizes = {512, 1024, 2048, 4096};
        policy.maxScheduledTokens = 32768;
        policy.longPrefillTokenThreshold = 32768;
        policy.hashBlockSize = 6144;
        policy.schedulerBlockSize = 6144;
        policy.useEagle = true;
        policy.asyncKVLookupEnabled = mode.second;
        policy.asyncKVLookupHashBlockSize = 6144;
        policy.asyncKVLookupUsesEagle = true;
        TPUPrefillExecutionProjection projection;
        ASSERT_TRUE(
            (mode.first == NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION
                 ? TPUPrefillExecutionPolicy::projectLatencyFeatures
                 : TPUPrefillExecutionPolicy::projectCacheResolvedBatch)(
                requests, policy, 32768, projection, nullptr));
        EXPECT_EQ(
            mode.first == NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION &&
                    mode.second
                ? 3
                : 1,
            projection.stepCount);
        NonLinearPrefillModelR initial;
        initial._pdSeparated = true;
        auto model = std::make_shared<NaviSchedLatencyModel>();
        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        model->binaryVersion = mode.first;
        model->parameters = initial.getInitialParameters();
        model->parameters[6] = 1.0;
        NaviSchedNodeSnapshot snapshot;
        auto info = std::make_shared<NaviSchedNodeInfo>();
        info->pdSeparated = true;
        snapshot.info = info;
        snapshot.latencyModel = model;
        snapshot.prefillExecutionPolicy = policy;
        NaviSchedScheduleContext context;
        NaviSchedLatencyFeatures agentFeatures;
        ASSERT_TRUE(agentFeatures.appendBatchCount(1));
        for (size_t i = 0; i < requests.size(); ++i) {
            NaviSchedScheduleItem item;
            item.request.requestId = i + 1;
            item.request.promptTokenCount = requests[i].promptTokens;
            item.request.blockSize = 1;
            context._requests.push_back(item);
            agentFeatures.appendRequest(
                requests[i].computedTokens,
                requests[i].promptTokens - requests[i].computedTokens);
        }
        ASSERT_TRUE(agentFeatures.setExecutionGeometry(projection.stepCount,
                                                       projection.globalPaddedTokenSum));
        const auto expected = NonLinearPrefillModelR::predictForVersion(
            agentFeatures, model->parameters, model->binaryVersion);
        context.setNodes({std::make_shared<MixedLookupNode>()}, {snapshot});
        ASSERT_TRUE(context.prepare());
        NaviSchedLatencyFeatures masterFeatures;
        ASSERT_TRUE(context.buildExecutionGeometryFeatures(0, {0, 1, 2}, masterFeatures));
        EXPECT_EQ(projection.stepCount, masterFeatures.getExecutionStepCount());
        EXPECT_EQ(projection.globalPaddedTokenSum,
                  masterFeatures.getGlobalPaddedTokens());
        EXPECT_DOUBLE_EQ(expected, context.estimateBatchExecutionLatencyMs(0, {0, 1, 2}));
        for (size_t i = 0; i < requests.size(); ++i) {
            ASSERT_TRUE(context.selectNode(i, 0, "projected_master_agent_parity"));
        }
        ASSERT_TRUE(context.prepareResult(context.getScheduleTimeUs()));
        for (size_t i = 0; i < requests.size(); ++i) {
            EXPECT_DOUBLE_EQ(expected, context._decisions[i].predictedExecutionLatencyMs);
            const auto audit =
                MasterScheduleResultKernel::getAdmissionPredictionAudit(context, i);
            ASSERT_TRUE(audit.valid);
            EXPECT_DOUBLE_EQ(expected, audit.baseExecutionMs);
            EXPECT_DOUBLE_EQ(expected, audit.finalExecutionMs);
        }
    }
}
} // namespace navi_sched

namespace navi_sched {
TEST_F(GeometryIntegrationTest, testQueueReadyPartialAbortStartUsesRuntimeGeometry)
{
    for (const bool projected : {false, true}) {
        navi::NaviResourceHelper helper;
        // All mandatory ExternalResource dependencies must be supplied;
        // Navi cannot instantiate these through resource registration.
        ASSERT_TRUE(helper.addExternalRes(std::make_shared<AgentReportBridgeR>()));
        ASSERT_TRUE(helper.addExternalRes(std::make_shared<AgentLatencyInputR>()));
        auto batchResource = std::make_shared<AgentBatchR>();
        ASSERT_TRUE(helper.addExternalRes(batchResource));
        ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID,
                                  R"({"fit_batch_size":1,"warmup_sample_count":0})"));
        ASSERT_TRUE(helper.config(
            NonLinearPrefillModelR::RESOURCE_ID,
            std::string(
                R"({"pd_separated":true,"use_batch_count_feature":true,"use_capacity_split_feature":false,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":)") +
                (projected ? "true}" : "false}")));
        ASSERT_TRUE(helper.config(
            AgentReportR::RESOURCE_ID,
            R"({"server_ip":"127.0.0.1","server_port":0,"pd_separated":true,"unified_latency_prediction":false})"));
        auto report = helper.getOrCreateResPtr<AgentReportR>();
        auto input = helper.getOrCreateResPtr<AgentLatencyInputR>();
        auto model = helper.getOrCreateResPtr<LatencyModelR>();
        ASSERT_TRUE(report);
        ASSERT_TRUE(input);
        ASSERT_TRUE(model);
        report->_pdSeparated = true;
        report->_latencyModelFingerprint = "opaque-runtime-policy-id";
        report->_controlConfig.executionGeometryEnabled = true;
        report->_controlConfig.projectedExecutionGeometryEnabled = projected;
        report->_controlConfig.prefillExecutionPolicy = typedRuntimePolicy();
        navi::KernelTesterBuilder builder;
        auto tester = builder.kernel("navi_sched.agent_latency_k")
                          .output("save")
                          .output("finished")
                          .attrs(std::string("{\"input_pipe_generation\":") +
                                 std::to_string(input->beginInputPipeGeneration()) + "}")
                          .resource(helper.getResourceMap())
                          .build();
        ASSERT_TRUE(tester);
        ASSERT_EQ(navi::EC_NONE, tester->getErrorCode());
        auto kernel = dynamic_cast<AgentLatencyKernel *>(tester->getKernel());
        ASSERT_TRUE(kernel);
        auto &batches = *batchResource;
        NaviSchedBatchPtr ready;
        NaviSchedRequestStats request;
        request.totalTokenCount = 512;
        request.blockSize = 16;
        request.executionSharedPrefixBoundary = 0;
        request.requestId = 101;
        ASSERT_TRUE(batches.addRequest(
            request, 81, 2, [&](const NaviSchedBatchPtr &value) { ready = value; }));
        request.requestId = 102;
        ASSERT_TRUE(batches.addRequest(
            request, 81, 2, [&](const NaviSchedBatchPtr &value) { ready = value; }));
        ASSERT_TRUE(ready);
        const auto localId = ready->batchId;
        const auto &policy = report->getPrefillExecutionPolicy();
        ASSERT_TRUE(TPUPrefillExecutionPolicy::validate(policy));
        TPUPrefillExecutionProjection geometry;
        const std::vector<TPUPrefillRequestInput> ordered = {{512, 0, 0}, {512, 0, 0}};
        ASSERT_TRUE(projected ? TPUPrefillExecutionPolicy::projectLatencyFeatures(
                                    ordered, policy, 1024, geometry)
                              : TPUPrefillExecutionPolicy::projectCacheResolvedBatch(
                                    ordered, policy, 1024, geometry));
        NaviSchedLatencyFeatures features;
        ASSERT_TRUE(features.appendBatchCount(1));
        features.appendRequest(0, 512);
        features.appendRequest(0, 512);
        ASSERT_TRUE(features.setExecutionGeometry(geometry.stepCount,
                                                  geometry.globalPaddedTokenSum));
        const auto expectedUs = std::llround(model->predict(features) * 1000.0);
        ASSERT_GT(expectedUs, 0);
        ASSERT_TRUE(input->submitBatchReady(ready, 0, 2, 0, ready->readyTimeUs));
        ASSERT_TRUE(tester->compute());
        ASSERT_EQ(expectedUs, kernel->_batchLatencyUs.at(localId));
        EXPECT_DOUBLE_EQ(expectedUs / 1000.0,
                         kernel->getRawQueueLatencyMs(ready->readyTimeUs));
        AgentBatchR::ExecutionState state;
        std::vector<NaviSchedRequestStats> aborted;
        NaviSchedBatchPtr remaining;
        NaviSchedPendingCacheReservation reservation;
        ASSERT_TRUE(batches.abortBatchRequests(localId, {101}, state, aborted, remaining,
                                               reservation));
        ASSERT_TRUE(remaining);
        auto abortedLocal = remaining->makeLocalBatchInfo();
        abortedLocal.requestIds = {101};
        abortedLocal.requestCount = 1;
        ASSERT_TRUE(input->submitBatchRequestsAborted(aborted, remaining, localId, 1, 0,
                                                      1, 0, ready->readyTimeUs + 100,
                                                      abortedLocal));
        ASSERT_TRUE(tester->compute());
        const auto remainingUs = kernel->predictBatchLatencyUs(remaining);
        ASSERT_GT(remainingUs, 0);
        EXPECT_EQ(remainingUs, kernel->_batchLatencyUs.at(localId));
        ASSERT_TRUE(batches.startBatch(localId, {102}, state));
        ASSERT_TRUE(input->submitBatchStarted(localId, 0, 0, 1, state.eventTimeUs));
        ASSERT_TRUE(tester->compute());
        EXPECT_EQ(remainingUs, kernel->_runningBatches.at(localId).predictedLatencyUs);
        EXPECT_DOUBLE_EQ(
            0.0, kernel->getRawQueueLatencyMs(state.eventTimeUs + remainingUs + 1));
        const auto originalStats = AgentBatchR::getRequestStats(*remaining);
        auto resolvedRequests = originalStats;
        resolvedRequests.front().cacheHitTokenCount = 384;
        resolvedRequests.front().executionSharedPrefixBoundary = 32;
        AgentBatchR::ExecutionState resolvedState;
        NaviSchedBatchPtr resolvedBatch;
        auto invalid = resolvedRequests;
        invalid.front().requestId = 999;
        EXPECT_FALSE(batches.resolveExecutionCache(localId, invalid, resolvedState,
                                                   resolvedBatch));
        EXPECT_FALSE(resolvedBatch);
        invalid = resolvedRequests;
        invalid.front().totalTokenCount += 1;
        EXPECT_FALSE(batches.resolveExecutionCache(localId, invalid, resolvedState,
                                                   resolvedBatch));
        ASSERT_TRUE(batches.resolveExecutionCache(localId, resolvedRequests,
                                                  resolvedState, resolvedBatch));
        ASSERT_TRUE(resolvedBatch);
        EXPECT_EQ(0, AgentBatchR::getRequestStats(*remaining).front().cacheHitTokenCount);
        EXPECT_EQ(
            384, AgentBatchR::getRequestStats(*resolvedBatch).front().cacheHitTokenCount);
        EXPECT_DOUBLE_EQ(originalStats.front().predictedExecutionLatencyMs,
                         AgentBatchR::getRequestStats(*resolvedBatch)
                             .front()
                             .predictedExecutionLatencyMs);
        const auto recovery = batches.getGraphRecoveryState();
        ASSERT_EQ(1u, recovery.batches.size());
        EXPECT_EQ(384, AgentBatchR::getRequestStats(*recovery.batches.front().batch)
                           .front()
                           .cacheHitTokenCount);
        AgentLatencyInput correction;
        correction.eventTimeUs = resolvedState.eventTimeUs;
        correction.runningRequests = resolvedState.runningRequests;
        correction.queuedBatchRequests = resolvedState.queuedBatchRequests;
        correction.batchWaitingRequests = resolvedState.batchWaitingRequests;
        const auto queueBefore = kernel->_queueLatencyUs.load();
        const auto frozenQueue = kernel->_batchQueueFeatures.at(localId).rawLatencyMs;
        ASSERT_TRUE(input->submitBatchCacheResolved(resolvedBatch, correction));
        ASSERT_TRUE(tester->compute());
        const auto correctedUs = kernel->predictBatchLatencyUs(resolvedBatch);
        EXPECT_LT(correctedUs, remainingUs);
        EXPECT_EQ(correctedUs, kernel->_batchLatencyUs.at(localId));
        EXPECT_EQ(correctedUs, kernel->_runningBatches.at(localId).predictedLatencyUs);
        EXPECT_EQ(queueBefore - remainingUs + correctedUs,
                  kernel->_queueLatencyUs.load());
        EXPECT_DOUBLE_EQ(frozenQueue,
                         kernel->_batchQueueFeatures.at(localId).rawLatencyMs);
        // Repeated observations replace one scalar rather than adding work again.
        ASSERT_TRUE(input->submitBatchCacheResolved(resolvedBatch, correction));
        ASSERT_TRUE(tester->compute());
        EXPECT_EQ(queueBefore - remainingUs + correctedUs,
                  kernel->_queueLatencyUs.load());
        report->_controlConfig.prefillExecutionPolicy = {};
        EXPECT_EQ(0, kernel->predictBatchLatencyUs(
                         remaining)); // Explicit carried zero, not projection.
        EXPECT_GT(kernel->_queueProjectionFallbackCount, 0u);
        for (const bool invalidCarried : {false, true}) {
            NaviSchedBatchPtr fallbackBatch;
            request.requestId = invalidCarried ? 104 : 103;
            request.predictedExecutionLatencyMs = 123.0;
            const int64_t fallbackScheduleId = invalidCarried ? 83 : 82;
            ASSERT_TRUE(batches.addRequest(
                request, fallbackScheduleId, 1,
                [&](const NaviSchedBatchPtr &value) { fallbackBatch = value; }));
            ASSERT_TRUE(fallbackBatch);
            const auto fallbackId = fallbackBatch->batchId;
            if (invalidCarried) {
                fallbackBatch->_requestStats.front().predictedExecutionLatencyMs =
                    std::numeric_limits<double>::quiet_NaN();
            }
            ASSERT_TRUE(input->submitBatchReady(fallbackBatch, 0, 1, 0,
                                                fallbackBatch->readyTimeUs));
            ASSERT_TRUE(tester->compute());
            EXPECT_EQ(invalidCarried ? 0 : 123000,
                      kernel->_batchLatencyUs.at(fallbackId));
            EXPECT_EQ(invalidCarried ? 1u : 0u,
                      kernel->_unknownQueueBatches.count(fallbackId));
            ASSERT_TRUE(input->submitBatchStarted(fallbackId, 0, 0, 1,
                                                  fallbackBatch->readyTimeUs + 100));
            // Queue FINISH before the same compute: an invalid READY must never
            // abort the drained input vector and discard later lifecycle events.
            const auto finishedRequests = AgentBatchR::getRequestStats(*fallbackBatch);
            ASSERT_TRUE(input->submitBatchFinished(
                finishedRequests, 10.0, 0.1, 0.0, 123.0, 0, 0, 0, 0, 0, fallbackId, 0, 0,
                0, 0, fallbackBatch->readyTimeUs + 10100, {}, -1.0, false, false,
                fallbackBatch->makeLocalBatchInfo()));
            ASSERT_TRUE(tester->compute());
            EXPECT_EQ(0u, kernel->_batchLatencyUs.count(fallbackId));
            EXPECT_EQ(0u, kernel->_runningBatches.count(fallbackId));
            EXPECT_EQ(0u, kernel->_unknownQueueBatches.count(fallbackId));
        }
        EXPECT_GT(kernel->_queueProjectionUnknownCount, 0u);
    }
}
TEST_F(GeometryIntegrationTest, testDiscreteSearchMatchesBaselineAndDoesNotRegress)
{
    for (int64_t version : {NonLinearPrefillModelR::PD_BINARY_VERSION,
                            NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION,
                            NonLinearPrefillModelR::CACHE_SOURCE_BINARY_VERSION}) {
        if (std::getenv("NAVI_DISCRETE_PERF_PROFILE_ONLY") &&
            !NonLinearPrefillModelR::usesProjectedGeometryForVersion(version)) {
            continue;
        }
        for (const auto shape : {std::pair<size_t, size_t>{4, 8},
                                 {99, 6},
                                 {99, 18},
                                 {99, 64},
                                 {200, 6},
                                 {200, 18},
                                 {200, 64},
                                 {200, 256},
                                 {4000, 400}}) {
            if ((shape.first == 4000) !=
                (std::getenv("NAVI_DISCRETE_PERF_LARGE") != nullptr)) {
                continue;
            }
            SCOPED_TRACE(version);
            SCOPED_TRACE(shape.first);
            SCOPED_TRACE(shape.second);
            NaviSchedScheduleContext context;
            std::vector<NaviSchedNodeConstPtr> nodes;
            std::vector<NaviSchedNodeSnapshot> snapshots;
            for (size_t i = 0; i < shape.first; ++i) {
                nodes.push_back(std::make_shared<GeometryMissingPolicyNode>());
                NaviSchedNodeSnapshot snapshot;
                auto info = std::make_shared<NaviSchedNodeInfo>();
                info->pdSeparated = true;
                snapshot.info = info;
                auto model = std::make_shared<NaviSchedLatencyModel>();
                model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
                model->binaryVersion = version;
                NonLinearPrefillModelR initial;
                initial._pdSeparated = true;
                initial.setExecutionGeometryEnabled(
                    NonLinearPrefillModelR::usesGeometryForVersion(version));
                initial.setProjectedExecutionGeometryEnabled(
                    NonLinearPrefillModelR::usesProjectedGeometryForVersion(version));
                initial.setExecutionCacheSourceEnabled(
                    NonLinearPrefillModelR::usesCacheSourceForVersion(version));
                model->parameters = initial.getInitialParameters();
                model->parameters[3] *= 1.0 + i * 0.01;
                snapshot.latencyModel = model;
                auto policy = typedRuntimePolicy();
                if (std::getenv("NAVI_DISCRETE_PERF_LONG_PROMPT")) {
                    policy.pcpSize = 8;
                    policy.interleaveSize = 256;
                    policy.compileSizes = {512, 1024, 2048, 4096};
                    policy.maxScheduledTokens = 32768;
                    policy.longPrefillTokenThreshold = 32768;
                    policy.hashBlockSize = 6144;
                    policy.schedulerBlockSize = 6144;
                    policy.useEagle = true;
                    policy.asyncKVLookupHashBlockSize = 6144;
                    policy.asyncKVLookupUsesEagle = true;
                }
                snapshot.prefillExecutionPolicy = policy;
                snapshot.residualRuntimeConfig.learningRate = 0.0;
                snapshots.push_back(snapshot);
            }
            std::vector<uint32_t> original;
            for (size_t i = 0; i < shape.second; ++i) {
                NaviSchedScheduleItem item;
                item.request.requestId = i + 1;
                item.request.promptTokenCount = 200 + i % 7 * 137;
                if (std::getenv("NAVI_DISCRETE_PERF_LONG_PROMPT")) {
                    item.request.promptTokenCount = 18433 + i % 7 * 12288;
                }
                item.request.blockSize =
                    std::getenv("NAVI_DISCRETE_PERF_LONG_PROMPT") ? 6144 : 1;
                context._requests.push_back(item);
                original.push_back(i % shape.first);
            }
            context.setNodes(std::move(nodes), std::move(snapshots));
            ASSERT_TRUE(context.prepare());
            std::vector<int64_t> tokens(original.size());
            std::vector<double> inverse(original.size());
            const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
                context, tokens.data(), inverse.data());
            std::vector<float> costs(shape.first * shape.second * 2);
            for (size_t i = 0; i < costs.size(); ++i) {
                costs[i] = 1.0 + i % 31 * 0.01;
            }
            struct Result {
                bool valid = false;
                std::vector<uint32_t> assignments;
                double loss = 0;
                CostSchedulerKernelInternal::DiscreteRefinementStats stats;
                int64_t elapsedNs = 0;
            };
            const size_t maxEvaluations =
                std::getenv("NAVI_DISCRETE_PERF_EVALUATIONS")
                    ? std::stoul(std::getenv("NAVI_DISCRETE_PERF_EVALUATIONS"))
                    : 512;
            auto run = [&](bool baseline, size_t evaluationLimit = 0,
                           int64_t timeBudgetUs = -1) {
                Result result;
                result.assignments = original;
                const auto start = std::chrono::steady_clock::now();
                const auto search =
                    baseline ? discrete_baseline::refine
                             : CostSchedulerKernelInternal::refineDiscreteAssignment;
                result.valid = search(
                    context, costs.data(), result.assignments.data(), tokens.data(),
                    inverse.data(), total, 0.5, 0.0, result.loss, timeBudgetUs,
                    evaluationLimit ? evaluationLimit : maxEvaluations, result.stats);
                result.elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();
                return result;
            };
            if (std::getenv("NAVI_DISCRETE_PERF_TIME_BUDGET_US")) {
                const auto budgetUs =
                    std::stoll(std::getenv("NAVI_DISCRETE_PERF_TIME_BUDGET_US"));
                ASSERT_GT(budgetUs, 0);
                for (size_t sample = 0; sample < 10; ++sample) {
                    const auto limited = run(false, 4096, budgetUs);
                    const auto timed =
                        run(false, std::numeric_limits<size_t>::max(), budgetUs);
                    ASSERT_TRUE(limited.valid);
                    ASSERT_TRUE(timed.valid);
                    CostSchedulerKernelInternal::DiscreteForwardWorkspace full;
                    const auto exact = CostSchedulerKernelInternal::forwardDiscrete(
                        context, costs.data(), timed.assignments.data(), tokens.data(),
                        inverse.data(), total, 0.5, 0.0, full);
                    EXPECT_EQ(exact, timed.loss);
                    const auto replay = run(true, timed.stats.evaluations);
                    ASSERT_TRUE(replay.valid);
                    EXPECT_EQ(replay.assignments, timed.assignments);
                    EXPECT_EQ(replay.loss, timed.loss);
                    EXPECT_EQ(replay.stats.moves, timed.stats.moves);
                    std::cout << "discrete_timed_perf version=" << version
                              << " nodes=" << shape.first << " requests=" << shape.second
                              << " sample=" << sample << " budget_us=" << budgetUs
                              << " capped_ns=" << limited.elapsedNs
                              << " timed_ns=" << timed.elapsedNs
                              << " capped_evaluations=" << limited.stats.evaluations
                              << " timed_evaluations=" << timed.stats.evaluations
                              << " capped_moves=" << limited.stats.moves
                              << " timed_moves=" << timed.stats.moves
                              << " capped_loss=" << std::setprecision(17) << limited.loss
                              << " timed_loss=" << timed.loss
                              << " exhausted=" << timed.stats.budgetExhausted
                              << std::endl;
                }
                continue;
            }
            if (std::getenv("NAVI_DISCRETE_PERF_PROFILE_ONLY")) {
                for (size_t sample = 0; sample < 1024; ++sample) {
                    ASSERT_TRUE(run(false).valid);
                }
                continue;
            }
            const auto expected = run(true);
            ASSERT_TRUE(expected.valid);
            auto check = [&](const Result &result) {
                EXPECT_TRUE(result.valid);
                EXPECT_EQ(expected.assignments, result.assignments);
                EXPECT_EQ(expected.loss, result.loss);
                EXPECT_EQ(expected.stats.evaluations, result.stats.evaluations);
                EXPECT_EQ(expected.stats.moves, result.stats.moves);
                EXPECT_EQ(expected.stats.budgetExhausted, result.stats.budgetExhausted);
            };
            check(run(false));
            std::vector<int64_t> baselineTimes, optimizedTimes;
            for (size_t sample = 0; sample < 10; ++sample) {
                Result before, after;
                if (sample % 2 == 0) {
                    before = run(true);
                    after = run(false);
                } else {
                    after = run(false);
                    before = run(true);
                }
                check(before);
                check(after);
                baselineTimes.push_back(before.elapsedNs);
                optimizedTimes.push_back(after.elapsedNs);
                std::cout << "discrete_search_perf residual_rate=0 version=" << version
                          << " nodes=" << shape.first << " requests=" << shape.second
                          << " sample=" << sample << " baseline_ns=" << before.elapsedNs
                          << " optimized_ns=" << after.elapsedNs
                          << " evaluations=" << after.stats.evaluations
                          << " moves=" << after.stats.moves << std::endl;
            }
            if (std::getenv("NAVI_DISCRETE_PERF_FULL_SCAN")) {
                for (size_t sample = 0; sample < 3; ++sample) {
                    const auto result = run(false, shape.second * (shape.first - 1));
                    ASSERT_TRUE(result.valid);
                    std::cout << "discrete_full_scan_perf version=" << version
                              << " nodes=" << shape.first << " requests=" << shape.second
                              << " sample=" << sample
                              << " optimized_ns=" << result.elapsedNs
                              << " evaluations=" << result.stats.evaluations
                              << " moves=" << result.stats.moves << std::endl;
                }
            }
            std::sort(baselineTimes.begin(), baselineTimes.end());
            std::sort(optimizedTimes.begin(), optimizedTimes.end());
            EXPECT_LE(optimizedTimes[4] + optimizedTimes[5],
                      (baselineTimes[4] + baselineTimes[5]) * 1.05);
        }
    }
}

TEST_F(GeometryIntegrationTest, testConditionalQueueImmutableMatchesPreserveViews)
{
    NaviSchedScheduleContext context;
    std::vector<NaviSchedNodeConstPtr> nodes;
    std::vector<NaviSchedNodeSnapshot> snapshots;
    for (size_t n = 0; n < 2; ++n) {
        nodes.push_back(std::make_shared<ConditionalQueueGeometryNode>());
        NaviSchedNodeSnapshot snapshot;
        auto info = std::make_shared<NaviSchedNodeInfo>();
        info->pdSeparated = true;
        snapshot.info = info;
        snapshots.push_back(snapshot);
    }
    for (size_t r = 0; r < 4; ++r) {
        NaviSchedScheduleItem item;
        item.request.requestId = r + 1;
        item.request.promptTokenCount = 100;
        item.request.blockSize = 1;
        context._requests.push_back(item);
    }
    context.setNodes(nodes, snapshots);
    const auto first = context.getCacheHitBlockCountsSnapshot(0);
    const auto second = context.getCacheHitBlockCountsSnapshot(1);
    ASSERT_TRUE(first);
    EXPECT_EQ((std::vector<int64_t>{1, 0, 1, 0}), *first);
    ASSERT_TRUE(context.activateScheduleView({3, 0}, {1, 0}));
    EXPECT_EQ(second, context.getCacheHitBlockCountsSnapshot(0));
    EXPECT_EQ(0, context.getCacheHitBlockCount(0, 0));
    EXPECT_EQ(1, context.getCacheHitBlockCount(1, 0));
    std::vector<int64_t> view;
    ASSERT_TRUE(context.getCacheHitBlockCounts(1, view));
    EXPECT_EQ((std::vector<int64_t>{0, 1}), view);
    context.clearScheduleView();
    for (const auto &node : nodes) {
        EXPECT_EQ(1u, std::dynamic_pointer_cast<const ConditionalQueueGeometryNode>(node)
                          ->batchCalls);
    }
}

TEST_F(GeometryIntegrationTest, testConditionalQueueWholeSchedulerParityAndPerformance)
{
    for (int64_t version : {NonLinearPrefillModelR::PD_BINARY_VERSION,
                            NonLinearPrefillModelR::PROJECTED_GEOMETRY_BINARY_VERSION}) {
        for (size_t requestCount : {size_t(18), size_t(64)}) {
            constexpr size_t nodeCount = 200;
            NaviSchedScheduleContext context;
            std::vector<NaviSchedNodeConstPtr> nodes;
            std::vector<NaviSchedNodeSnapshot> snapshots;
            for (size_t n = 0; n < nodeCount; ++n) {
                nodes.push_back(std::make_shared<ConditionalQueueGeometryNode>());
                NaviSchedNodeSnapshot snapshot;
                auto info = std::make_shared<NaviSchedNodeInfo>();
                info->pdSeparated = true;
                snapshot.info = info;
                auto model = std::make_shared<NaviSchedLatencyModel>();
                model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
                model->binaryVersion = version;
                NonLinearPrefillModelR initial;
                initial._pdSeparated = true;
                model->parameters = initial.getInitialParameters();
                snapshot.latencyModel = model;
                snapshot.prefillExecutionPolicy = typedRuntimePolicy();
                snapshot.residualRuntimeConfig.learningRate = 0.;
                snapshot.load.queueTotalPredictedLatencyMs = 100.;
                snapshot.load.queueModel.cacheMemberWeightMs = n % 2 ? -40. : 60.;
                snapshot.prefillExecutionPolicy->asyncKVLookupEnabled = true;
                snapshot.prefillExecutionPolicy->asyncKVLookupHashBlockSize = 32768;
                snapshot.prefillExecutionPolicy->asyncKVLookupUsesEagle = true;
                snapshot.load.queueModel.lookupMemberWeightMs = n % 2 ? 35. : -25.;
                snapshots.push_back(snapshot);
            }
            for (size_t r = 0; r < requestCount; ++r) {
                NaviSchedScheduleItem item;
                item.request.requestId = r + 1;
                item.request.promptTokenCount = 18433 + r % 7 * 12288;
                item.request.blockSize = 1;
                context._requests.push_back(item);
            }
            context._scheduleTimeUs = 1000000;
            context.setNodes(std::move(nodes), std::move(snapshots));
            ASSERT_TRUE(context.prepare());
            for (auto &node : context._scheduleNodes) {
                node._queueDecayHeadRemainingMs = 90.;
                node._queueReportReceiveTimeUs = context.getScheduleTimeUs() - 20000;
            }
            std::vector<int64_t> tokens(requestCount);
            std::vector<double> inverse(requestCount);
            const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
                context, tokens.data(), inverse.data());
            CostSchedulerKernelInternal::ReplayInfo info;
            auto costs = CostSchedulerKernelInternal::prepare(context, tokens.data(), .37,
                                                              1., 1., 0., 1, 0, info);
            ASSERT_TRUE(costs);
            std::vector<uint32_t> assignments(requestCount);
            for (size_t r = 0; r < requestCount; ++r) {
                assignments[r] = r % 2;
            }
            CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace, full;
            auto discrete = [&](auto &ws) {
                return CostSchedulerKernelInternal::forwardDiscrete(
                    context, costs.get(), assignments.data(), tokens.data(),
                    inverse.data(), total, .37, 128., ws);
            };
            const auto before = discrete(workspace);
            ASSERT_TRUE(std::isfinite(before));
            const auto oldPair = workspace.nodeValues[0] + workspace.nodeValues[1];
            assignments[0] = 1;
            const auto newPair = CostSchedulerKernelInternal::forwardDiscreteImpl(
                context, costs.get(), assignments.data(), tokens.data(), inverse.data(),
                total, .37, 128., workspace, 0, 1);
            const auto after = discrete(full);
            EXPECT_NEAR(after - before, newPair - oldPair, 1e-12);
            EXPECT_DOUBLE_EQ(full.nodeValues[2], workspace.nodeValues[2]);
            for (size_t r = 0; r < requestCount; ++r) {
                ASSERT_TRUE(context.selectNode(r, assignments[r], "conditional"));
            }
            ASSERT_TRUE(context.prepareResult(context.getScheduleTimeUs()));
            for (size_t n = 0; n < 2; ++n) {
                size_t hits = 0, count = 0, lookups = 0;
                for (size_t r = 0; r < requestCount; ++r) {
                    if (assignments[r] == n) {
                        ++count;
                        hits += (r + 1) % 2;
                        lookups +=
                            context.getNode(n).hasQueueLookupOpportunity(tokens[r]);
                    }
                }
                const auto expected = std::max(
                    0.,
                    80. + ((n % 2 ? -40. : 60.) * hits + (n % 2 ? 35. : -25.) * lookups) /
                              count);
                for (size_t r = 0; r < requestCount; ++r) {
                    if (assignments[r] == n) {
                        EXPECT_DOUBLE_EQ(expected,
                                         context._decisions[r].predictedQueueLatencyMs);
                    }
                }
            }
            std::vector<double> weights(nodeCount * requestCount, .42),
                gradients(weights.size());
            std::vector<CostSchedulerKernelInternal::NodeGradient> ng(nodeCount);
            for (bool enabled : {false, true}) {
                for (size_t n = 0; n < nodeCount; ++n) {
                    context._scheduleNodes[n]._queueModel.cacheMemberWeightMs =
                        enabled ? (n % 2 ? -40. : 60.) : 0.;
                    context._scheduleNodes[n]._queueModel.lookupMemberWeightMs =
                        enabled && version == NonLinearPrefillModelR::
                                                  PROJECTED_GEOMETRY_BINARY_VERSION
                            ? (n % 2 ? 35. : -25.)
                            : 0.;
                    // Ordinary production snapshots do not carry a geometry policy.
                    if (version == NonLinearPrefillModelR::PD_BINARY_VERSION) {
                        context._scheduleNodes[n]._queueLookupPromptBoundary =
                            std::numeric_limits<int64_t>::max();
                    }
                }
                std::vector<int64_t> samples;
                for (size_t sample = 0; sample < 10; ++sample) {
                    const auto start = std::chrono::steady_clock::now();
                    for (size_t round = 0; round < 20; ++round) {
                        const auto value = CostSchedulerKernelInternal::forwardNodeRange(
                            context, costs.get(), weights.data(), tokens.data(),
                            inverse.data(), total, .37, 128., 0, nodeCount, ng.data(),
                            nullptr, nullptr);
                        ASSERT_TRUE(std::isfinite(value));
                        CostSchedulerKernelInternal::backwardNodeRange(
                            context, costs.get(), weights.data(), tokens.data(),
                            inverse.data(), requestCount, 0, nodeCount, ng.data(),
                            gradients.data());
                        ASSERT_TRUE(std::isfinite(discrete(workspace)));
                    }
                    samples.push_back(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count() /
                        20);
                }
                std::sort(samples.begin(), samples.end());
                std::cout << "conditional_queue_scheduler_perf version=" << version
                          << " nodes=" << nodeCount << " requests=" << requestCount
                          << " enabled=" << enabled
                          << " forward_backward_full_discrete_median_ns="
                          << (samples[4] + samples[5]) / 2 << std::endl;
            }
        }
    }
}

} // namespace navi_sched

namespace navi_sched {
TEST_F(GeometryIntegrationTest,
       testProjectedAndMeasuredCacheSourceFrozenBatchAndContinuousCostGradient)
{
    for (const bool projected : {true, false}) {
        class SourceNode : public GeometryMissingPolicyNode {
        public:
            explicit SourceNode(bool external) : _external(external) {}
            bool getExecutionCacheInputs(
                const std::vector<NaviSchedScheduleItem> &requests,
                std::vector<TPUPrefillRequestInput> &inputs, int64_t *suffix = nullptr,
                ::navi_sched::TPUPrefillCacheObservationBatch * = nullptr) const override
            {
                ++calls;
                inputs.clear();
                for (size_t i = 0; i < requests.size(); ++i) {
                    const auto hit = i == 0 ? 49152 : 86016;
                    const auto external = _external ? int64_t(i == 0 ? 12288 : 24576) : 0;
                    inputs.push_back(
                        {requests[i].request.promptTokenCount, hit, 0, external});
                    if (suffix) {
                        suffix[i] = external / requests[i].request.blockSize;
                    }
                }
                return true;
            }
            mutable int calls = 0;
            bool _external;
        };
        auto external = std::make_shared<SourceNode>(true);
        auto local = std::make_shared<SourceNode>(false);
        NonLinearPrefillModelR initial;
        initial._pdSeparated = true;
        initial.setExecutionGeometryEnabled(true);
        initial.setProjectedExecutionGeometryEnabled(projected);
        initial.setExecutionCacheSourceEnabled(true);
        auto parameters = initial.getInitialParameters();
        parameters[19] = 110.0;
        parameters[20] = 1.2;
        NaviSchedScheduleContext context;
        std::vector<NaviSchedNodeSnapshot> snapshots;
        for (size_t i = 0; i < 2; ++i) {
            NaviSchedNodeSnapshot snapshot;
            auto info = std::make_shared<NaviSchedNodeInfo>();
            info->pdSeparated = true;
            info->agentId = "cache_source_" + std::to_string(i);
            snapshot.info = info;
            auto model = std::make_shared<NaviSchedLatencyModel>();
            model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
            model->binaryVersion = initial.getBinaryVersion();
            if (!projected && std::getenv("NAVI_TEST_MEASURED_DISPATCH_SERIALIZED")) {
                model->binaryVersion =
                    NonLinearPrefillModelR::CACHE_SOURCE_BINARY_VERSION;
            }
            model->parameters = parameters;
            snapshot.latencyModel = model;
            snapshot.prefillExecutionPolicy = typedRuntimePolicy();
            snapshots.push_back(snapshot);
        }
        for (const auto prompt : {50016, 90016}) {
            NaviSchedScheduleItem item;
            item.request.requestId = context._requests.size() + 1;
            item.request.promptTokenCount = prompt;
            item.request.blockSize = 32;
            context._requests.push_back(item);
        }
        context.setNodes({external, local}, snapshots);
        ASSERT_TRUE(context.prepare());
        NaviSchedLatencyPrediction frozen;
        const auto externalMs =
            context.estimateBatchExecutionLatencyMs(0, {0, 1}, &frozen);
        const auto localMs = context.estimateBatchExecutionLatencyMs(1, {0, 1});
        EXPECT_NEAR(110.0 + 1.2 * 24, externalMs - localMs, 1e-8);
        if (!projected) {
            const std::vector<TPUPrefillRequestInput> requests = {
                {50016, 49152, 0, 12288}, {90016, 86016, 0, 24576}};
            TPUPrefillExecutionProjection joint, serialized;
            ASSERT_TRUE(TPUPrefillExecutionPolicy::projectCacheResolvedBatchSummary(
                requests, typedRuntimePolicy(), 4864, joint));
            ASSERT_TRUE(TPUPrefillExecutionPolicy::projectLatencyFeaturesSummary(
                requests, typedRuntimePolicy(), 4864, serialized));
            ASSERT_GT(serialized.stepCount, joint.stepCount);
            EXPECT_EQ(joint.stepCount, frozen.features.getExecutionStepCount());
            EXPECT_EQ(joint.globalPaddedTokenSum,
                      frozen.features.getGlobalPaddedTokens());
            ASSERT_TRUE(context.getGeometryOptimizationInput(0, 0));
            ASSERT_TRUE(context.getGeometryOptimizationInput(0, 1));
            EXPECT_TRUE(context.getGeometryOptimizationInput(0, 0)->shared);
            EXPECT_TRUE(context.getGeometryOptimizationInput(0, 1)->shared);
        }
        EXPECT_EQ(2, frozen.features.getExternalReadRequestCount());
        EXPECT_EQ(24576, frozen.features.getMaxExternalReadTokens());
        EXPECT_DOUBLE_EQ(externalMs,
                         initial.predictConfigured(frozen.features, parameters));
        autil::DataBuffer wire;
        frozen.serialize(wire);
        NaviSchedLatencyPrediction decoded;
        decoded.deserialize(wire);
        EXPECT_EQ(24576, decoded.features.getMaxExternalReadTokens());
        EXPECT_DOUBLE_EQ(externalMs,
                         initial.predictConfigured(decoded.features, parameters));

        std::vector<int64_t> tokens(2);
        std::vector<double> inverse(2);
        const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
            context, tokens.data(), inverse.data());
        std::vector<float> costs(8, std::numeric_limits<float>::quiet_NaN());
        for (const auto &start : {std::vector<double>{0.2, 0.3, 0.8, 0.7},
                                  std::vector<double>{0.8, 0.7, 0.2, 0.3}}) {
            auto weights = start;
            std::vector<double> gradients(4);
            CostSchedulerKernelInternal::NodeGradient nodeGradients[2];
            const auto evaluate = [&](CostSchedulerKernelInternal::NodeGradient *g) {
                return CostSchedulerKernelInternal::forwardNodeRange(
                    context, costs.data(), weights.data(), tokens.data(), inverse.data(),
                    total, 0.35, 128.0, 0, 2, g, nullptr, nullptr);
            };
            ASSERT_TRUE(std::isfinite(evaluate(nodeGradients)));
            CostSchedulerKernelInternal::backwardNodeRange(
                context, costs.data(), weights.data(), tokens.data(), inverse.data(), 2,
                0, 2, nodeGradients, gradients.data());
            for (size_t i = 0; i < weights.size(); ++i) {
                constexpr double DELTA = 1e-6;
                weights[i] += DELTA;
                const auto plus = evaluate(nullptr);
                weights[i] -= 2 * DELTA;
                const auto minus = evaluate(nullptr);
                weights[i] += DELTA;
                const auto numerical = (plus - minus) / (2 * DELTA);
                EXPECT_NEAR(numerical, gradients[i],
                            1e-5 * std::max(1.0, std::abs(numerical)))
                    << i;
            }
        }
        EXPECT_EQ(1, external->calls);
        EXPECT_EQ(1, local->calls);
    }
}

TEST_F(GeometryIntegrationTest,
       testFixedChunkCacheSourceFrozenBatchAndContinuousCostGradient)
{
    for (const bool projected : {true, false}) {
        class SourceNode : public GeometryMissingPolicyNode {
        public:
            explicit SourceNode(bool external) : _external(external) {}
            bool getExecutionCacheInputs(
                const std::vector<NaviSchedScheduleItem> &requests,
                std::vector<TPUPrefillRequestInput> &inputs, int64_t *suffix = nullptr,
                ::navi_sched::TPUPrefillCacheObservationBatch *observation =
                    nullptr) const override
            {
                ++calls;
                if (observation) {
                    observation->pool = TPUPrefillCacheObservation{};
                    observation->pool->revision = cacheRevision;
                    observation->fullPrefixBlocksBeforeQueryDrop.assign(requests.size(),
                                                                        8);
                }
                inputs.clear();
                for (size_t i = 0; i < requests.size(); ++i) {
                    const auto hit = i == 0 ? 49152 : 86016;
                    const auto external = _external ? int64_t(i == 0 ? 12288 : 24576) : 0;
                    inputs.push_back({requests[i].request.promptTokenCount, hit,
                                      sharedBoundary, external});
                    if (suffix) {
                        suffix[i] = external / requests[i].request.blockSize;
                    }
                }
                return true;
            }
            mutable int calls = 0;
            int64_t sharedBoundary = 0;
            uint64_t cacheRevision = 489;
            bool _external;
        };
        auto fixedPolicy = typedRuntimePolicy();
        fixedPolicy.fixedPrefillChunkPlan = true;
        fixedPolicy.longPrefillTokenThreshold =
            std::min<int64_t>(256, fixedPolicy.maxScheduledTokens);
        auto external = std::make_shared<SourceNode>(true);
        auto local = std::make_shared<SourceNode>(false);
        NonLinearPrefillModelR initial;
        initial._pdSeparated = true;
        initial.setExecutionGeometryEnabled(true);
        initial.setProjectedExecutionGeometryEnabled(projected);
        initial.setExecutionCacheSourceEnabled(true);
        auto parameters = initial.getInitialParameters();
        parameters[19] = 110.0;
        parameters[20] = 1.2;
        NaviSchedScheduleContext context;
        std::vector<NaviSchedNodeSnapshot> snapshots;
        for (size_t i = 0; i < 2; ++i) {
            NaviSchedNodeSnapshot snapshot;
            auto info = std::make_shared<NaviSchedNodeInfo>();
            info->pdSeparated = true;
            info->agentId = "cache_source_" + std::to_string(i);
            snapshot.info = info;
            auto model = std::make_shared<NaviSchedLatencyModel>();
            model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
            model->binaryVersion = initial.getBinaryVersion();
            if (!projected && std::getenv("NAVI_TEST_MEASURED_DISPATCH_SERIALIZED")) {
                model->binaryVersion =
                    NonLinearPrefillModelR::CACHE_SOURCE_BINARY_VERSION;
            }
            model->parameters = parameters;
            snapshot.latencyModel = model;
            snapshot.prefillExecutionPolicy = fixedPolicy;
            snapshots.push_back(snapshot);
        }
        for (const auto prompt : {50016, 90016}) {
            NaviSchedScheduleItem item;
            item.request.requestId = context._requests.size() + 1;
            item.request.promptTokenCount = prompt;
            item.request.blockSize = 32;
            context._requests.push_back(item);
        }
        context.setNodes({external, local}, snapshots);
        ASSERT_TRUE(context.prepare());
        NaviSchedLatencyPrediction frozen;
        const auto externalMs =
            context.estimateBatchExecutionLatencyMs(0, {0, 1}, &frozen);
        const auto localMs = context.estimateBatchExecutionLatencyMs(1, {0, 1});
        EXPECT_NEAR(110.0 + 1.2 * 24, externalMs - localMs, 1e-8);
        if (!projected) {
            const std::vector<TPUPrefillRequestInput> requests = {
                {50016, 49152, 0, 12288}, {90016, 86016, 0, 24576}};
            TPUPrefillExecutionProjection joint, serialized;
            ASSERT_TRUE(TPUPrefillExecutionPolicy::projectCacheResolvedBatchSummary(
                requests, fixedPolicy, 4864, joint));
            ASSERT_TRUE(TPUPrefillExecutionPolicy::projectLatencyFeaturesSummary(
                requests, fixedPolicy, 4864, serialized));
            ASSERT_GT(serialized.stepCount, joint.stepCount);
            EXPECT_EQ(joint.stepCount, frozen.features.getExecutionStepCount());
            EXPECT_EQ(joint.globalPaddedTokenSum,
                      frozen.features.getGlobalPaddedTokens());
            ASSERT_TRUE(context.getGeometryOptimizationInput(0, 0));
            ASSERT_TRUE(context.getGeometryOptimizationInput(0, 1));
            EXPECT_TRUE(context.getGeometryOptimizationInput(0, 0)->shared);
            EXPECT_TRUE(context.getGeometryOptimizationInput(0, 1)->shared);
        }
        EXPECT_EQ(2, frozen.features.getExternalReadRequestCount());
        EXPECT_EQ(24576, frozen.features.getMaxExternalReadTokens());
        EXPECT_DOUBLE_EQ(externalMs,
                         initial.predictConfigured(frozen.features, parameters));
        autil::DataBuffer wire;
        frozen.serialize(wire);
        NaviSchedLatencyPrediction decoded;
        decoded.deserialize(wire);
        EXPECT_EQ(24576, decoded.features.getMaxExternalReadTokens());
        EXPECT_DOUBLE_EQ(externalMs,
                         initial.predictConfigured(decoded.features, parameters));

        std::vector<int64_t> tokens(2);
        std::vector<double> inverse(2);
        const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
            context, tokens.data(), inverse.data());
        std::vector<float> costs(8, std::numeric_limits<float>::quiet_NaN());
        for (const auto &start : {std::vector<double>{0.2, 0.3, 0.8, 0.7},
                                  std::vector<double>{0.8, 0.7, 0.2, 0.3}}) {
            auto weights = start;
            std::vector<double> gradients(4);
            CostSchedulerKernelInternal::NodeGradient nodeGradients[2];
            const auto evaluate = [&](CostSchedulerKernelInternal::NodeGradient *g) {
                return CostSchedulerKernelInternal::forwardNodeRange(
                    context, costs.data(), weights.data(), tokens.data(), inverse.data(),
                    total, 0.35, 128.0, 0, 2, g, nullptr, nullptr);
            };
            ASSERT_TRUE(std::isfinite(evaluate(nodeGradients)));
            CostSchedulerKernelInternal::backwardNodeRange(
                context, costs.data(), weights.data(), tokens.data(), inverse.data(), 2,
                0, 2, nodeGradients, gradients.data());
            for (size_t i = 0; i < weights.size(); ++i) {
                constexpr double DELTA = 1e-6;
                weights[i] += DELTA;
                const auto plus = evaluate(nullptr);
                weights[i] -= 2 * DELTA;
                const auto minus = evaluate(nullptr);
                weights[i] += DELTA;
                const auto numerical = (plus - minus) / (2 * DELTA);
                EXPECT_NEAR(numerical, gradients[i],
                            1e-5 * std::max(1.0, std::abs(numerical)))
                    << i;
            }
        }
        EXPECT_EQ(1, external->calls);
        EXPECT_EQ(1, local->calls);
        context._selectedNodeIndexes = {0, 0};
        context._decisions.resize(2);
        for (size_t i = 0; i < 2; ++i) {
            context._decisions[i].requestId = i + 1;
            context._decisions[i].batchId = 77;
            context._decisions[i].latencyPrediction =
                std::make_shared<NaviSchedLatencyPrediction>(frozen);
        }
        // The live cache changes after prediction; the audit must retain the
        // exact frozen junction and external-read coordinates without rereading.
        external->sharedBoundary = 6144;
        external->_external = false;
        external->cacheRevision = 506;
        const auto audit =
            MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0, true);
        ASSERT_TRUE(audit.valid);
        ASSERT_EQ(2u, audit.frozenExecutionInputs.size());
        ASSERT_TRUE(audit.cacheObservation.has_value());
        EXPECT_EQ(489u, audit.cacheObservation->revision);
        EXPECT_EQ(8, audit.frozenExecutionInputs[0].fullPrefixBlocksBeforeQueryDrop);
        EXPECT_EQ(1u, audit.frozenExecutionInputs[0].requestId);
        EXPECT_EQ(50016, audit.frozenExecutionInputs[0].input.promptTokens);
        EXPECT_EQ(49152, audit.frozenExecutionInputs[0].input.computedTokens);
        EXPECT_EQ(0, audit.frozenExecutionInputs[0].input.sharedPrefixBoundary);
        EXPECT_EQ(12288, audit.frozenExecutionInputs[0].input.externalReadTokens);
        EXPECT_EQ(2u, audit.frozenExecutionInputs[1].requestId);
        EXPECT_EQ(86016, audit.frozenExecutionInputs[1].input.computedTokens);
        EXPECT_EQ(24576, audit.frozenExecutionInputs[1].input.externalReadTokens);
        EXPECT_EQ(1, external->calls);
        EXPECT_TRUE(MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0)
                        .frozenExecutionInputs.empty());
    }
}

TEST_F(GeometryIntegrationTest,
       testCacheSourceRejectedBeforeHistoryAndPersistentRoundtrip)
{
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID,
                              R"({"fit_batch_size":1,"warmup_sample_count":0})"));
    ASSERT_TRUE(helper.config(AgentSchedulerConfigR::RESOURCE_ID,
                              R"({"config_json":"{\"latency_fit_batch_size\":1}"})"));
    ASSERT_TRUE(helper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        R"({"pd_separated":true,"use_capacity_split_feature":false,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":true,"execution_cache_source_enabled":true})"));
    auto model = helper.getOrCreateResPtr<LatencyModelR>();
    auto history = helper.getOrCreateResPtr<AgentHistoryR>();
    ASSERT_TRUE(model);
    ASSERT_TRUE(history);
    const auto before = model->getSnapshot();
    ASSERT_TRUE(before);
    ASSERT_EQ(34, before->binaryVersion);
    NaviSchedLatencySample sample;
    ASSERT_TRUE(sample.features.appendBatchCount(1));
    sample.features.appendRequest(6144, 1024);
    ASSERT_TRUE(sample.features.setExecutionGeometry(1, 4096));
    sample.latencyMs = 220.0;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> after;
    bool save = false;
    EXPECT_FALSE(model->recordSample(sample, after, save));
    EXPECT_EQ(0u, history->getSessionSampleCount());
    EXPECT_TRUE(history->getHistory()->empty());
    ASSERT_TRUE(sample.features.setExecutionCacheSource(1, 6144));
    ASSERT_TRUE(model->recordSample(sample, after, save));
    ASSERT_TRUE(after);
    EXPECT_EQ(1u, history->getSessionSampleCount());
    LatencyHistorySampler sampler;
    sampler.configure(2000, 64);
    sampler.append(sample);
    autil::DataBuffer wire;
    ASSERT_TRUE(sampler.serializeWithExecutionCacheSource(wire));
    LatencyHistorySampler restored;
    ASSERT_TRUE(restored.deserializeWithExecutionCacheSource(wire));
    ASSERT_EQ(1u, restored.getHistory()->size());
    EXPECT_EQ(6144, restored.getHistory()->front().features.getMaxExternalReadTokens());
    sample.features.clearExecutionCacheSource();
    LatencyHistorySampler unknown;
    unknown.configure(2000, 64);
    unknown.append(sample);
    autil::DataBuffer invalid;
    EXPECT_FALSE(unknown.serializeWithExecutionCacheSource(invalid));
}
} // namespace navi_sched

namespace navi_sched {
TEST_F(GeometryIntegrationTest, testCacheSourceFirstObservationSurvivesFinishTuple)
{
    AgentBatchR batches(1000, 60000);
    NaviSchedRequestStats request;
    request.requestId = 1;
    request.totalTokenCount = 100;
    request.blockSize = 10;
    request.executionSharedPrefixBoundary = 0;
    NaviSchedBatchPtr ready;
    ASSERT_TRUE(batches.addRequest(
        request, 91, 1, [&](const NaviSchedBatchPtr &value) { ready = value; }));
    ASSERT_TRUE(ready);
    AgentBatchR::ExecutionState state;
    ASSERT_TRUE(batches.startBatch(ready->batchId, {1}, state));
    auto resolved = request;
    resolved.cacheHitTokenCount = 40;
    resolved.executionExternalReadTokens = 40;
    NaviSchedBatchPtr corrected;
    ASSERT_TRUE(
        batches.resolveExecutionCache(ready->batchId, {resolved}, state, corrected));
    ASSERT_TRUE(corrected);
    EXPECT_EQ(
        40, AgentBatchR::getRequestStats(*corrected).front().executionExternalReadTokens);
    // Later local residency after the load must not erase the original V6D read.
    auto finished = resolved;
    finished.executionExternalReadTokens = 0;
    std::vector<NaviSchedRequestStats> completed;
    ASSERT_TRUE(batches.finishBatch(ready->batchId, {finished}, state, &completed));
    ASSERT_EQ(1u, completed.size());
    EXPECT_EQ(40, completed.front().executionExternalReadTokens);
    AgentLatencyInput input;
    input.queries = completed;
    input.geometryFeatureOriginVersion =
        AgentLatencyInput::PROJECTED_GEOMETRY_FEATURE_VERSION;
    input.projectedExecutionStepCount = 1;
    input.projectedGlobalPaddedTokens = 128;
    const auto sample = input.makeLatencySample();
    EXPECT_TRUE(sample.features.hasExecutionCacheSource());
    EXPECT_EQ(1, sample.features.getExternalReadRequestCount());
    EXPECT_EQ(40, sample.features.getMaxExternalReadTokens());
}
} // namespace navi_sched

namespace navi_sched {
TEST_F(GeometryIntegrationTest,
       testMeasuredCacheSourceUsesObservedGeometryBeforeItsUpdate)
{
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID,
                              R"({"fit_batch_size":1,"warmup_sample_count":0})"));
    ASSERT_TRUE(helper.config(AgentSchedulerConfigR::RESOURCE_ID,
                              R"({"config_json":"{\"latency_fit_batch_size\":1}"})"));
    ASSERT_TRUE(helper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        R"({"pd_separated":true,"use_capacity_split_feature":false,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":false,"execution_cache_source_enabled":true})"));
    auto model = helper.getOrCreateResPtr<LatencyModelR>();
    auto nonlinear = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
    auto history = helper.getOrCreateResPtr<AgentHistoryR>();
    ASSERT_TRUE(model);
    ASSERT_TRUE(nonlinear);
    ASSERT_TRUE(history);
    const auto before = model->getSnapshot();
    ASSERT_TRUE(before);
    ASSERT_EQ(NonLinearPrefillModelR::MEASURED_CACHE_SOURCE_BINARY_VERSION,
              before->binaryVersion);
    ASSERT_EQ(21u, before->parameters.size());
    EXPECT_FALSE(
        NonLinearPrefillModelR::usesProjectedGeometryForVersion(before->binaryVersion));
    EXPECT_TRUE(NonLinearPrefillModelR::usesCacheSourceForVersion(before->binaryVersion));

    AgentLatencyInput input;
    input.latencyMs = 217.25;
    input.executionStepCount = 3;
    input.globalPaddedTokens = 24576;
    input.projectedExecutionStepCount = 2;
    input.projectedGlobalPaddedTokens = 12288;
    NaviSchedRequestStats first;
    first.requestId = 1;
    first.totalTokenCount = 7050;
    first.cacheHitTokenCount = 0;
    first.executionExternalReadTokens = 0;
    auto second = first;
    second.requestId = 2;
    second.totalTokenCount = 6820;
    second.cacheHitTokenCount = 6144;
    second.executionExternalReadTokens = 6144;
    input.queries = {first, second};
    if (const auto mutant = std::getenv("NAVI_TEST_MEASURED_GEOMETRY_MUTANT")) {
        if (std::string(mutant) == "projected") {
            input.geometryFeatureOriginVersion =
                AgentLatencyInput::PROJECTED_GEOMETRY_FEATURE_VERSION;
        }
    }
    ASSERT_TRUE(input.hasCompatibleGeometryFeatureOrigin(before->binaryVersion));
    const auto sample = input.makeLatencySample();
    ASSERT_EQ(3, sample.features.getExecutionStepCount());
    ASSERT_EQ(24576, sample.features.getGlobalPaddedTokens());
    ASSERT_TRUE(sample.features.hasExecutionCacheSource());
    EXPECT_EQ(1, sample.features.getExternalReadRequestCount());
    EXPECT_EQ(6144, sample.features.getMaxExternalReadTokens());
    EXPECT_DOUBLE_EQ(217.25, sample.latencyMs);
    const auto predictionBefore =
        nonlinear->predictConfigured(sample.features, before->parameters);
    ASSERT_TRUE(std::isfinite(predictionBefore));
    EXPECT_DOUBLE_EQ(predictionBefore,
                     NonLinearPrefillModelR::predictForVersion(
                         sample.features, before->parameters, before->binaryVersion));
    EXPECT_TRUE(nonlinear->checkGradientConfigured(sample.features, before->parameters));
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> after;
    bool save = false;
    ASSERT_TRUE(model->recordSample(sample, after, save));
    ASSERT_TRUE(after);
    ASSERT_EQ(1u, history->getSessionSampleCount());
    ASSERT_EQ(1u, history->getHistory()->size());
    EXPECT_EQ(3, history->getHistory()->front().features.getExecutionStepCount());
    EXPECT_EQ(24576, history->getHistory()->front().features.getGlobalPaddedTokens());
    // A projected-origin sample must not silently cross this model boundary.
    input.geometryFeatureOriginVersion =
        AgentLatencyInput::PROJECTED_GEOMETRY_FEATURE_VERSION;
    EXPECT_FALSE(input.hasCompatibleGeometryFeatureOrigin(before->binaryVersion));
}
} // namespace navi_sched

namespace navi_sched {
TEST_F(GeometryIntegrationTest,
       testMeasuredCacheSourceAgentRoleStartsAndReportsObservedSample)
{
    NaviSchedAgentConfig config;
    config.runtime.bizName = "measured_cache_source_startup_biz";
    config.runtime.threadNum = 2;
    config.service.name = "measured_cache_source_startup_test";
    config.service.enableKmon = false;
    config.service.gig.grpcPort = 0;
    config.service.enableSnapshot = false;
    config.service.latencyValidationEnabled = true;
    config.service.latencyFitBatchSize = 1;
    config.service.latencyWarmupSampleCount = 0;
    config.executionGeometryEnabled = true;
    config.projectedExecutionGeometryEnabled = false;
    config.executionCacheSourceEnabled = true;
    config.prefillExecutionPolicy = typedRuntimePolicy();
    NaviSchedAgent agent(config);
    // Reintroduce the exact removed source34-only prerequisite in this test,
    // never in a production process. The original native init also fails this
    // assertion without the mutation, proving the actual service path is covered.
    const bool oldPrerequisite =
        std::getenv("NAVI_TEST_REINTRODUCE_PROJECTED_AGENT_GATE") &&
        config.executionCacheSourceEnabled && !config.projectedExecutionGeometryEnabled;
    ASSERT_TRUE(!oldPrerequisite && agent.init());
    const auto report = agent._impl->_agentReportR;
    ASSERT_TRUE(report);
    auto initial = report->getLatencyModelSnapshot();
    ASSERT_TRUE(initial);
    ASSERT_EQ(NonLinearPrefillModelR::MEASURED_CACHE_SOURCE_BINARY_VERSION,
              initial->binaryVersion);
    ASSERT_EQ(21u, initial->parameters.size());
    agent.reportCacheUpdated(-1, 6144, 1, nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, 1, autil::TimeUtility::currentTime());
    NaviSchedBatchRequest request;
    request.requestId = 1;
    request.totalTokenCount = 7050;
    request.blockSize = 6144;
    NaviSchedBatchPtr ready;
    ASSERT_TRUE(agent.addBatchRequest(
        request, 7, 1, [&ready](const NaviSchedBatchPtr &batch) { ready = batch; }));
    ASSERT_TRUE(ready);
    ASSERT_TRUE(ready->finishReporter);
    ASSERT_TRUE(ready->finishReporter->startBatch({1}));
    NaviSchedRequestStats completed;
    completed.requestId = 1;
    completed.totalTokenCount = 7050;
    completed.blockSize = 6144;
    completed.executionExternalReadTokens = 0;
    ASSERT_TRUE(ready->finishReporter->finishBatchWithExecutionGeometry(
        {completed}, 217.25, 200.0, 3, 7050, 0, 0, -1.0, 24576));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto snapshot = report->getLatencyModelSnapshot();
        if (snapshot && snapshot->sessionSampleCount == 1 &&
            report->getLatencyValidationStats().sampleCount == 1)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    auto trained = report->getLatencyModelSnapshot();
    ASSERT_TRUE(trained);
    EXPECT_EQ(35, trained->binaryVersion);
    EXPECT_EQ(21u, trained->parameters.size());
    EXPECT_EQ(1u, trained->sessionSampleCount);
    EXPECT_GT(trained->revision, initial->revision);
    EXPECT_EQ(1u, report->getLatencyValidationStats().sampleCount);
    EXPECT_EQ(0, agent._impl->getInflightRequestCount());
    const auto observed =
        agent._impl->_agentBatchR->takeBackendBatchExecutionWindowStats();
    EXPECT_EQ(3, observed.executionStepCount);
    agent.stop();
    for (int mode = 0; mode < 3; ++mode) {
        auto invalid = config;
        if (mode == 0) invalid.executionGeometryEnabled = false;
        if (mode == 1) invalid.unifiedLatencyPrediction = true;
        if (mode == 2) invalid.pdSeparated = false;
        NaviSchedAgent rejected(invalid);
        EXPECT_FALSE(rejected.init());
    }
}
} // namespace navi_sched

namespace navi_sched {
TEST_F(GeometryIntegrationTest, testLiveOrderWitnessUsesCanonicalFrozenGeometry)
{
    class NoHitNode : public GeometryMissingPolicyNode {
    public:
        bool getExecutionCacheInputs(
            const std::vector<NaviSchedScheduleItem> &requests,
            std::vector<TPUPrefillRequestInput> &inputs, int64_t *suffix = nullptr,
            TPUPrefillCacheObservationBatch * = nullptr) const override
        {
            inputs.clear();
            for (size_t i = 0; i < requests.size(); ++i) {
                inputs.push_back({requests[i].request.promptTokenCount, 0, 0, 0});
                if (suffix) {
                    suffix[i] = 0;
                }
            }
            return true;
        }
    };
    auto policy = typedRuntimePolicy();
    policy.pcpSize = 8;
    policy.interleaveSize = 256;
    policy.compileSizes = {512, 1024, 2048, 4096};
    policy.maxScheduledTokens = policy.longPrefillTokenThreshold = 32768;
    policy.hashBlockSize = policy.schedulerBlockSize = 6144;
    policy.useEagle = policy.mambaHasPrefillCheckpointBlocks =
        policy.mambaPartialCacheHit = false;
    policy.fixedPrefillChunkPlan = policy.mambaBlockAlignedSplit = true;
    policy.prefillLookahead = 1;
    policy.maxInputTokens = 32768;
    policy.draftSlotsPerRequest = 0;
    policy.attentionScheduleCapacity = policy.attentionQueryTile =
        policy.attentionKvTile = 0;
    policy.asyncKVLookupEnabled = true;
    policy.asyncKVLookupHashBlockSize = 6144;
    policy.asyncKVLookupUsesEagle = true;
    TPUPrefillExecutionProjection bad, good;
    ASSERT_TRUE(TPUPrefillExecutionPolicy::projectCacheResolvedBatchSummary(
        {{5204, 0, 0, 0}, {82178, 0, 0, 0}}, policy, 87382, bad));
    ASSERT_TRUE(TPUPrefillExecutionPolicy::projectCacheResolvedBatchSummary(
        {{82178, 0, 0, 0}, {5204, 0, 0, 0}}, policy, 87382, good));
    EXPECT_EQ(5, bad.stepCount);
    EXPECT_EQ(110592, bad.globalPaddedTokenSum);
    EXPECT_EQ(4, good.stepCount);
    EXPECT_EQ(102400, good.globalPaddedTokenSum);
    NonLinearPrefillModelR model;
    model._pdSeparated = true;
    model.setExecutionGeometryEnabled(true);
    model.setProjectedExecutionGeometryEnabled(false);
    model.setExecutionCacheSourceEnabled(true);
    NaviSchedNodeSnapshot snapshot;
    auto info = std::make_shared<NaviSchedNodeInfo>();
    info->pdSeparated = true;
    info->agentId = "canonical_order";
    snapshot.info = info;
    auto latency = std::make_shared<NaviSchedLatencyModel>();
    latency->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    latency->binaryVersion = model.getBinaryVersion();
    latency->parameters = model.getInitialParameters();
    snapshot.latencyModel = latency;
    snapshot.prefillExecutionPolicy = policy;
    NaviSchedScheduleContext context;
    for (const auto &request :
         std::vector<std::pair<uint64_t, int64_t>>{{974, 5204}, {970, 82178}}) {
        NaviSchedScheduleItem item;
        item.request.requestId = request.first;
        item.request.promptTokenCount = request.second;
        item.request.blockSize = 6144;
        context._requests.push_back(item);
    }
    context.setNodes({std::make_shared<NoHitNode>()}, {snapshot});
    ASSERT_TRUE(context.prepare());
    for (const auto &indexes : std::vector<std::vector<size_t>>{{0, 1}, {1, 0}}) {
        NaviSchedLatencyPrediction frozen;
        EXPECT_GT(context.estimateBatchExecutionLatencyMs(0, indexes, &frozen), 0.0);
        EXPECT_EQ(4, frozen.features.getExecutionStepCount());
        EXPECT_EQ(102400, frozen.features.getGlobalPaddedTokens());
        EXPECT_EQ(87382, frozen.features.getCacheMissTokenCount());
    }
    context._scheduleViewActive = true;
    context._scheduleViewNodeIndexes = {0};
    context._scheduleViewRequestIndexes = {1, 0};
    NaviSchedLatencyPrediction viewed;
    EXPECT_GT(context.estimateBatchExecutionLatencyMs(0, {1, 0}, &viewed), 0.0);
    EXPECT_EQ(4, viewed.features.getExecutionStepCount());
    EXPECT_EQ(102400, viewed.features.getGlobalPaddedTokens());
}
} // namespace navi_sched
