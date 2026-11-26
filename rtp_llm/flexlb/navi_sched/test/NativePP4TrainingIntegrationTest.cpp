#include "aios/navi_sched/common/TPUPrefillPP4Timing.h"
#include "aios/navi_sched/kernel/AgentLatencyKernel.h"
#include "aios/navi_sched/resource/AgentHistoryR.h"
#include "aios/navi_sched/resource/AgentLatencyInputR.h"
#include "aios/navi_sched/resource/AgentReportR.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "navi/tester/KernelTesterBuilder.h"
#include "navi/tester/NaviResourceHelper.h"
#include "autil/DataBuffer.h"
#include "autil/TimeUtility.h"
#include "unittest/unittest.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/scheduler/CostSchedulerKernelInternal.h"
#include <algorithm>

#include "aios/navi_sched/scheduler/MasterScheduleContextKernel.h"

namespace navi_sched {
namespace {
std::shared_ptr<TPUPrefillPP4TimingObservation> observedPP4Step()
{
    auto observation = std::make_shared<TPUPrefillPP4TimingObservation>();
    observation->targetOrdinal = 0;
    for (int64_t ordinal = 0; ordinal < 3; ++ordinal) {
        TPUPrefillPipelineSlot slot;
        slot.ordinal = ordinal;
        slot.features[0] = ordinal == 0 ? 1.0 : 0.0;
        observation->slots.push_back(slot);
    }
    return observation;
}
const std::string MODEL_CONFIG =
    R"({"pd_separated":true,"use_capacity_split_feature":true,"native_prefill_pipeline_stages":4,"learning_rate":0.0005,"error_clip_ms":0})";
}

namespace {
class NativeMasterDiagnosticNode final : public NaviSchedNode {
public:
    NaviSchedNodeInfo getInfo() const override
    {
        NaviSchedNodeInfo info;
        info.agentId = "agent";
        info.bizName = "model";
        info.online = true;
        info.totalBlocks = 100;
        return info;
    }
    bool getScheduleSnapshot(const std::string &, NaviSchedNodeSnapshot &) const override
    {
        return false;
    }
    int64_t getMatchedBlockCount(const NaviSchedBlockHashes &) const override
    {
        return 1;
    }
    void getMatchedBlockCounts(const std::vector<NaviSchedScheduleItem> &requests,
                               int64_t *counts, int64_t *suffix = nullptr) const override
    {
        for (size_t i = 0; i < requests.size(); ++i) {
            counts[i] = 1;
            if (suffix) suffix[i] = 0;
        }
    }
    bool getExecutionCacheInputs(
        const std::vector<NaviSchedScheduleItem> &requests,
        std::vector<TPUPrefillRequestInput> &inputs, int64_t *suffix = nullptr,
        TPUPrefillCacheObservationBatch *observation = nullptr,
        const ::navi_sched::NaviSchedExternalCacheReadyBlocks *remoteReady = nullptr, bool = true) const override
    {
        ++cacheCalls;
        inputs.clear();
        if (observation && observation->readyOnlyHbmBlockCounts)
            observation->readyOnlyHbmBlockCounts->clear();
        for (size_t i = 0; i < requests.size(); ++i) {
            TPUPrefillRequestInput input;
            input.promptTokens = requests[i].request.promptTokenCount;
            input.computedTokens = fullHit ? input.promptTokens : cachePrefixTokens;
            input.externalReadTokens = externalReadTokenCount;
            inputs.push_back(input);
            if (observation && observation->readyOnlyHbmBlockCounts)
                observation->readyOnlyHbmBlockCounts->push_back(
                    readyHbmTokens < 0 ? -1 : readyHbmTokens / requests[i].request.blockSize);
            if (suffix) suffix[i] = 0;
        }
        return true;
    }
    mutable size_t cacheCalls = 0;
    bool fullHit = false;
    int64_t cachePrefixTokens = 16;
    int64_t externalReadTokenCount = 0;
    int64_t readyHbmTokens = -1;
};

std::vector<NaviSchedScheduleItem> nativeMasterDiagnosticRequests()
{
    std::vector<NaviSchedScheduleItem> requests(1);
    auto &request = requests[0].request;
    request.requestId = 1;
    request.nativePrefillSlotRelease = NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT;
    request.bizName = "model";
    request.promptTokenCount = 32;
    request.blockSize = 16;
    request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    request.timeoutMs = requests[0].remainingTimeoutMs = 1000;
    return requests;
}
}

class NativePP4TrainingIntegrationTest : public TESTBASE {};

TEST_F(NativePP4TrainingIntegrationTest, QueueKernelTrainsAndRestoresVersionedState)
{
    const auto path = GET_TEMP_DATA_PATH() + "native_pp4_training/model.state";
    std::filesystem::remove(path);
    const auto historyConfig =
        std::string(R"({"state_file":")") + path +
        R"(","formula_fingerprint":"native36","history_fingerprint":"native36","binary_fingerprint":"native36","history_limit":128,"fit_batch_size":1,"fit_interval":1,"warmup_sample_count":0,"persist_interval":1})";
    navi::NaviResourceHelper helper;
    auto inputs = std::make_shared<AgentLatencyInputR>();
    inputs->enableNativePP4TimingTraining();
    auto batches = std::make_shared<AgentBatchR>();
    ASSERT_TRUE(batches->setBackendExecutionPipelineStages(4));
    ASSERT_TRUE(helper.addExternalRes(inputs));
    ASSERT_TRUE(helper.addExternalRes(batches));
    ASSERT_TRUE(helper.addExternalRes(std::make_shared<AgentReportBridgeR>()));
    ASSERT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID, MODEL_CONFIG));
    ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID, historyConfig));
    ASSERT_TRUE(helper.config(LatencyModelR::RESOURCE_ID,
                              R"({"trust_min_revision":2,"cold_start_fit_epochs":5})"));
    ASSERT_TRUE(helper.config(
        AgentReportR::RESOURCE_ID,
        R"({"server_ip":"127.0.0.1","server_port":0,"pd_separated":true,"unified_latency_prediction":true,"pd_unified_capacity_split":true})"));
    auto report = helper.getOrCreateResPtr<AgentReportR>();
    auto model = helper.getOrCreateResPtr<LatencyModelR>();
    auto history = helper.getOrCreateResPtr<AgentHistoryR>();
    ASSERT_TRUE(report && model && history);
    ASSERT_EQ(36, model->getSnapshot()->binaryVersion);
    ASSERT_EQ(std::vector<double>(9, 0.0), model->getSnapshot()->parameters);
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
    const auto observation = observedPP4Step();
    for (size_t sample = 0; sample < 64; ++sample) {
        ASSERT_TRUE(inputs->submitNativePP4TimingSample(
            observation, 42.0, autil::TimeUtility::currentTime()));
        ASSERT_TRUE(tester->compute());
    }
    const auto trained = model->getSnapshot();
    EXPECT_NE(std::vector<double>(9, 0.0), trained->parameters);
    EXPECT_GT(trained->optimizerState.step, 1);
    EXPECT_GT(trained->revision, 1);
    ASSERT_EQ(64u, history->getHistory()->size());
    ASSERT_TRUE(history->getHistory()->front().nativePP4Timing);
    ASSERT_TRUE(model->saveState());
    navi::NaviResourceHelper restoredHelper;
    ASSERT_TRUE(restoredHelper.config(NonLinearPrefillModelR::RESOURCE_ID, MODEL_CONFIG));
    ASSERT_TRUE(restoredHelper.config(AgentHistoryR::RESOURCE_ID, historyConfig));
    auto restoredHistory = restoredHelper.getOrCreateResPtr<AgentHistoryR>();
    auto restoredModel = restoredHelper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(restoredHistory && restoredModel);
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> restored;
    bool fitOnStart = false;
    ASSERT_TRUE(restoredHistory->load(*restoredModel, restored, fitOnStart));
    ASSERT_TRUE(restored);
    EXPECT_EQ(trained->parameters, restored->parameters);
    EXPECT_EQ(trained->optimizerState.adamMoment1,
              restoredModel->getOptimizerState().adamMoment1);
    EXPECT_EQ(trained->optimizerState.adamMoment2,
              restoredModel->getOptimizerState().adamMoment2);
    EXPECT_EQ(trained->optimizerState.step, restoredModel->getOptimizerState().step);
    ASSERT_EQ(64u, restoredHistory->getHistory()->size());
    ASSERT_TRUE(restoredHistory->getHistory()->front().nativePP4Timing);
    EXPECT_EQ(3u, restoredHistory->getHistory()->front().nativePP4Timing->slots.size());
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeHistoryRejectsTruncationAndOversizedSupport)
{
    LatencyHistorySampler sampler(1);
    sampler.configure(2, 1);
    NaviSchedLatencySample sample;
    sample.nativePP4Timing = observedPP4Step();
    sample.latencyMs = 42.0;
    sample.eventTimeUs = 1000;
    for (int index = 0; index < 8; ++index) sampler.append(sample);
    ASSERT_EQ(2u, sampler.historySize());
    autil::DataBuffer buffer;
    ASSERT_TRUE(sampler.serializeWithNativePP4Timing(buffer));
    LatencyHistorySampler restored(1);
    restored.configure(2, 1);
    autil::DataBuffer truncated(buffer.getData(), buffer.getDataLen() - 1);
    EXPECT_FALSE(restored.deserializeWithNativePP4Timing(truncated));
    EXPECT_EQ(0u, restored.historySize());
    autil::DataBuffer malformed;
    malformed.write(uint64_t(2));
    malformed.write(uint32_t(1));
    malformed.write(uint64_t(1));
    malformed.write(int64_t(0));
    malformed.write(uint32_t(7));
    EXPECT_FALSE(restored.deserializeWithNativePP4Timing(malformed));
    EXPECT_EQ(0u, restored.historySize());
    AgentLatencyInputR inputs;
    inputs.enableNativePP4TimingTraining();
    auto impossible = observedPP4Step();
    impossible->slots.back().dispatchTimeMs = 100.0;
    EXPECT_FALSE(inputs.submitNativePP4TimingSample(impossible, 42.0, 1000));
    EXPECT_FALSE(inputs.shouldTrainInput(AgentLatencyInputType::BATCH_FINISHED));
}

TEST_F(NativePP4TrainingIntegrationTest, ColdRecordedGeometryUsesOriginalFixedAdam)
{
    const auto corpus = std::getenv("NAVI_SCHED_PP4_TRAINING_CORPUS");
    if (!corpus) GTEST_SKIP() << "optional recorded geometry corpus was not supplied";
    std::ifstream file(corpus);
    ASSERT_TRUE(file);
    std::array<std::deque<NaviSchedLatencySample>, 4> cohorts;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream input(line);
        int group;
        size_t count;
        NaviSchedLatencySample sample;
        auto observation = std::make_shared<TPUPrefillPP4TimingObservation>();
        ASSERT_TRUE(bool(input >> group >> observation->targetOrdinal >>
                         sample.latencyMs >> count));
        ASSERT_GE(group, 0);
        ASSERT_LT(group, 4);
        ASSERT_LE(count, 6u);
        observation->slots.resize(count);
        for (auto &slot : observation->slots) {
            ASSERT_TRUE(bool(input >> slot.ordinal >> slot.dispatchTimeMs));
            for (auto &feature : slot.features) ASSERT_TRUE(bool(input >> feature));
        }
        std::string extra;
        ASSERT_FALSE(bool(input >> extra));
        ASSERT_TRUE(observation->validLatencyMs(sample.latencyMs));
        sample.nativePP4Timing = std::move(observation);
        sample.eventTimeUs = 1000;
        cohorts[group].push_back(std::move(sample));
    }
    ASSERT_EQ(2285u, cohorts[0].size());
    ASSERT_EQ(756u, cohorts[1].size());
    ASSERT_EQ(765u, cohorts[2].size());
    ASSERT_EQ(1531u, cohorts[3].size());
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID, MODEL_CONFIG));
    auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(model);
    const auto initial = model->getInitialParameters();
    ASSERT_EQ(std::vector<double>(9, 0.0), initial);
    ASSERT_EQ(1, model->getOptimizerState().step);
    std::vector<double> trained;
    const auto begin = std::chrono::steady_clock::now();
    ASSERT_TRUE(model->fitHistory(cohorts[0], 64, 100, initial, trained));
    const auto trainUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
    const auto frozenOptimizer = model->getOptimizerState();
    std::cout << "PP4_COLD_TRAIN_US " << trainUs << "\nPP4_COLD_PARAMETERS";
    for (double parameter : trained)
        std::cout << " " << std::setprecision(17) << parameter;
    std::cout << "\n";
    for (size_t group = 0; group < cohorts.size(); ++group) {
        double error = 0.0;
        size_t spikes = 0;
        for (const auto &sample : cohorts[group]) {
            const auto predicted = model->predictSampleConfigured(sample, trained);
            ASSERT_TRUE(std::isfinite(predicted));
            const auto relative =
                std::abs(predicted - sample.latencyMs) / sample.latencyMs;
            error += relative;
            spikes += relative > 0.3;
        }
        std::cout << "PP4_COLD_SCORE " << group << " " << cohorts[group].size() << " "
                  << std::setprecision(17) << 100.0 * error / cohorts[group].size() << " "
                  << spikes << "\n";
    }
    EXPECT_EQ(frozenOptimizer.step, model->getOptimizerState().step);
    EXPECT_EQ(frozenOptimizer.adamMoment1, model->getOptimizerState().adamMoment1);
    EXPECT_EQ(frozenOptimizer.adamMoment2, model->getOptimizerState().adamMoment2);
}

TEST_F(NativePP4TrainingIntegrationTest, MasterRetainsNativeRequestLifetimeAndCheckpoint)
{
    ClientRequestToMasterData request;
    request.requestId = 123;
    request.bizName = "prefill";
    request.timeoutMs = 1000;
    request.promptTokenCount = 32;
    request.blockSize = 16;
    request.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    request.explicitCacheCheckpointPosition = 16;
    request.nativePrefillSlotRelease = NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT;
    ASSERT_TRUE(request.valid());
    const auto item = MasterStateR::makeScheduleItem(request, 5);
    ASSERT_EQ(request.nativePrefillSlotRelease, item.request.nativePrefillSlotRelease);
    ASSERT_EQ(16, item.request.explicitCacheCheckpointPosition);
    ASSERT_EQ(32, item.request.promptTokenCount);
    MasterBatchReservation::Request reserved{
        123, 32, 16, 16, request.blockHashes, 4.0, 16, request.nativePrefillSlotRelease};
    MasterBatchReservation reservation;
    ASSERT_TRUE(reservation.init(10, 1000, 4.0, {reserved}, nullptr));
    ASSERT_EQ(request.nativePrefillSlotRelease,
              reservation.getWaitingRequests()[0].nativePrefillSlotRelease);
    const auto replay = reservation.makeReplayState();
    MasterBatchReservation restored;
    ASSERT_TRUE(restored.restoreReplayState(replay));
    const auto retained = restored.getWaitingRequests();
    ASSERT_EQ(1u, retained.size());
    ASSERT_EQ(16, retained[0].explicitCacheCheckpointPosition);
    ASSERT_EQ(request.nativePrefillSlotRelease, retained[0].nativePrefillSlotRelease);
    EXPECT_TRUE(reservation.sameDefinition(restored));
    reserved.nativePrefillSlotRelease = NativePPPrefillSlotRelease::RETAINS_SLOT;
    MasterBatchReservation changed;
    ASSERT_TRUE(changed.init(10, 1000, 4.0, {reserved}, nullptr));
    EXPECT_FALSE(reservation.sameDefinition(changed));
    reserved.nativePrefillSlotRelease = static_cast<NativePPPrefillSlotRelease>(99);
    MasterBatchReservation invalid;
    EXPECT_FALSE(invalid.init(10, 1000, 4.0, {reserved}, nullptr));
}

namespace {
MasterBatchReservation::Request nativeReservedRequest(uint64_t id)
{
    return {id,
            32,
            16,
            16,
            std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22}),
            0.0,
            16,
            NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT};
}

NaviSchedLocalBatchInfo nativeReservationBatch(int64_t batchId, uint64_t requestId)
{
    NaviSchedLocalBatchInfo batch;
    batch.scheduleBatchId = batchId;
    batch.localBatchId = batchId;
    batch.requestCount = 1;
    batch.requestIds = {requestId};
    return batch;
}

struct NativeMasterSnapshotFixture {
    MasterPendingCacheR pending;
    std::shared_ptr<MasterStateR::NodeState> node =
        std::make_shared<MasterStateR::NodeState>();
    std::shared_ptr<MasterStateR::NodeState::PublishedSnapshot> published =
        std::make_shared<MasterStateR::NodeState::PublishedSnapshot>();

    NativeMasterSnapshotFixture()
    {
        auto info = std::make_shared<NaviSchedNodeInfo>();
        info->agentId = "native_master_snapshot";
        info->agentIdHash = 123;
        info->bizName = "prefill";
        info->online = true;
        info->serverAddress = "127.0.0.1:18000";
        info->pdSeparated = true;
        info->nativePrefillPipelineStages = 4;
        published->info = info;
        setVersion(NonLinearPrefillModelR::NATIVE_PP4_BINARY_VERSION);
        node->_info = *info;
        node->_publishedSnapshot = published;
        node->_pendingCacheR = &pending;
        node->_acceptReservations = true;
        node->_cachePolicyReady.store(true);
        node->_cacheStateReady.store(true);
        AgentPendingCacheToMasterData update;
        update.fullSnapshot = true;
        pending.applyAgentUpdate(info->agentIdHash, update);
        node->_runtimeState.unifiedLatencyPrediction = true;
        auto &workload = node->_runtimeState.workload;
        workload.revision = workload.eventTimeUs = 1;
        workload.prefillCapacitySplit = true;
        workload.maxRunningRequestCount = 64;
        workload.nativePPObservationAgeUs = 0;
        auto native = std::make_shared<NaviSchedNativePPState>();
        native->observationId = 1;
        native->occupiedRequestSlotCount = 0;
        workload.nativePPState = native;
    }

    void setVersion(int64_t version)
    {
        auto model = std::make_shared<NaviSchedLatencyModel>();
        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        model->binaryVersion = version;
        model->parameters.assign(
            NonLinearPrefillModelR::usesNativePP4ForVersion(version) ? 9 : 19, 0.0);
        published->latencyModel = model;
    }
};
}

TEST_F(NativePP4TrainingIntegrationTest,
       UnfinishedReservationsKeepAcceptedAndDropFinished)
{
    MasterBatchReservation batch;
    ASSERT_TRUE(batch.init(10, 1000, 0.0,
                           {nativeReservedRequest(9), nativeReservedRequest(3),
                            nativeReservedRequest(7), nativeReservedRequest(11)},
                           nullptr));
    ASSERT_TRUE(batch.markAcceptedWorkloadReported(7));
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(batch.markAdmitted(nativeReservationBatch(10, 9), transition));
    ASSERT_TRUE(batch.markFailed(3, transition));
    std::vector<MasterBatchReservation::UnfinishedRequest> result;
    result.push_back(
        {nativeReservedRequest(100), MasterBatchReservation::RequestState::WAITING});
    batch.appendUnfinishedRequests(result);
    ASSERT_EQ(4u, result.size());
    EXPECT_EQ(100u, result[0].request.requestId);
    EXPECT_EQ(7u, result[1].request.requestId);
    EXPECT_EQ(9u, result[2].request.requestId);
    EXPECT_EQ(11u, result[3].request.requestId);
    EXPECT_EQ(MasterBatchReservation::RequestState::DISPATCHED, result[1].state);
    EXPECT_EQ(MasterBatchReservation::RequestState::DISPATCHED, result[2].state);
    EXPECT_EQ(MasterBatchReservation::RequestState::WAITING, result[3].state);
    EXPECT_EQ(16, result[1].request.explicitCacheCheckpointPosition);
    EXPECT_EQ(NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT,
              result[1].request.nativePrefillSlotRelease);
    const auto frozen = result;
    ASSERT_TRUE(batch.markCompleted(nativeReservationBatch(10, 9), true, transition));
    ASSERT_TRUE(batch.markFailed(7, transition));
    result.clear();
    batch.appendUnfinishedRequests(result);
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(11u, result[0].request.requestId);
    EXPECT_EQ(4u, frozen.size());
    EXPECT_EQ(MasterBatchReservation::RequestState::DISPATCHED, frozen[1].state);
    auto replay = batch.makeReplayState();
    MasterBatchReservation restored;
    ASSERT_TRUE(restored.restoreReplayState(replay));
    std::vector<MasterBatchReservation::UnfinishedRequest> replayed;
    restored.appendUnfinishedRequests(replayed);
    ASSERT_EQ(1u, replayed.size());
    EXPECT_EQ(11u, replayed[0].request.requestId);
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeMasterSnapshotRetainsAcceptedRequestMetadata)
{
    NativeMasterSnapshotFixture fixture;
    auto node = fixture.node;
    ASSERT_TRUE(node->_runtimeState.workload.valid());
    ASSERT_TRUE(fixture.pending.getStats(123).agentSnapshotReady);
    ASSERT_TRUE(node->reserveBatch(
        10, 1000, 0.0, {nativeReservedRequest(9), nativeReservedRequest(3)}, nullptr));
    auto &batch = node->_pendingBatchReservations.front()->batch;
    ASSERT_TRUE(batch.markAcceptedWorkloadReported(3));
    NaviSchedNodeSnapshot snapshot;
    ASSERT_TRUE(node->getScheduleSnapshot("prefill", snapshot));
    ASSERT_EQ(2u, snapshot.nativePrefillReservations.size());
    EXPECT_EQ(3u, snapshot.nativePrefillReservations[0].request.requestId);
    EXPECT_EQ(MasterBatchReservation::RequestState::DISPATCHED,
              snapshot.nativePrefillReservations[0].state);
    EXPECT_EQ(9u, snapshot.nativePrefillReservations[1].request.requestId);
    EXPECT_EQ(32, snapshot.nativePrefillReservations[0].request.totalTokenCount);
    EXPECT_EQ(
        16,
        snapshot.nativePrefillReservations[0].request.explicitCacheCheckpointPosition);
    EXPECT_EQ(NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT,
              snapshot.nativePrefillReservations[0].request.nativePrefillSlotRelease);
    EXPECT_EQ(snapshot.workload.nativePPState,
              node->_runtimeState.workload.nativePPState);
    const auto hashes = snapshot.nativePrefillReservations[0].request.blockHashes;
    NaviSchedScheduleNode scheduled(node, std::move(snapshot), 1000);
    ASSERT_EQ(2u, scheduled.getNativePrefillReservations().size());
    EXPECT_EQ(hashes, scheduled.getNativePrefillReservations()[0].request.blockHashes);
    MasterBatchReservation::Transition transition;
    ASSERT_TRUE(batch.markFailed(3, transition));
    ASSERT_TRUE(node->getScheduleSnapshot("prefill", snapshot));
    ASSERT_EQ(1u, snapshot.nativePrefillReservations.size());
    EXPECT_EQ(9u, snapshot.nativePrefillReservations[0].request.requestId);
    EXPECT_EQ(2u, scheduled.getNativePrefillReservations().size());
    fixture.setVersion(NonLinearPrefillModelR::BINARY_VERSION);
    ASSERT_TRUE(node->getScheduleSnapshot("prefill", snapshot));
    EXPECT_TRUE(snapshot.nativePrefillReservations.empty());
}

TEST_F(NativePP4TrainingIntegrationTest, NativeMasterReservationSnapshotBenchmark)
{
    if (!std::getenv("NAVI_E256_RESERVATION_BENCHMARK")) return;
    for (size_t count : {size_t(16), size_t(64), size_t(512)}) {
        NativeMasterSnapshotFixture fixture;
        fixture.node->_runtimeState.unifiedLatencyPrediction = false;
        std::vector<MasterBatchReservation::Request> requests;
        for (size_t index = 0; index < count; ++index)
            requests.push_back(nativeReservedRequest(count - index));
        ASSERT_TRUE(
            fixture.node->reserveBatch(10, 1000, 0.0, std::move(requests), nullptr));
        for (bool native : {false, true, true, false}) {
            fixture.setVersion(native ? NonLinearPrefillModelR::NATIVE_PP4_BINARY_VERSION
                                      : NonLinearPrefillModelR::BINARY_VERSION);
            for (int sample = 0; sample < 7; ++sample) {
                const auto begin = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < 2000; ++iteration) {
                    NaviSchedNodeSnapshot snapshot;
                    ASSERT_TRUE(fixture.node->getScheduleSnapshot("prefill", snapshot));
                    ASSERT_EQ(native ? count : 0,
                              snapshot.nativePrefillReservations.size());
                }
                const double ns = std::chrono::duration<double, std::nano>(
                                      std::chrono::steady_clock::now() - begin)
                                      .count() /
                                  2000;
                if (sample > 0)
                    std::cout << "E256_SNAPSHOT_BENCH count=" << count
                              << " native=" << native << " ns=" << std::setprecision(17)
                              << ns << "\n";
            }
        }
    }
}

TEST_F(NativePP4TrainingIntegrationTest, Native9MasterCostAndFinalPredictionUsePPState)
{
    NonLinearPrefillModelR legacyModel;
    legacyModel._useCapacitySplitFeature = true;
    auto legacyParameters = legacyModel.getInitialParameters();
    legacyParameters[NonLinearPrefillModelR::ZERO_POINT_PARAMETER_INDEX] =
        100.0 / NonLinearPrefillModelR::getParameterScale(
                    NonLinearPrefillModelR::ZERO_POINT_PARAMETER_INDEX);
    ASSERT_EQ(19u, legacyParameters.size());
    ASSERT_EQ(NonLinearPrefillModelR::BINARY_VERSION, legacyModel.getBinaryVersion());
    ASSERT_TRUE(NonLinearPrefillModelR::validateParametersForVersion(
        legacyParameters, legacyModel.getBinaryVersion()));
    TPUPrefillExecutionPolicyConfig policy;
    policy.pcpSize = 8;
    policy.interleaveSize = 128;
    policy.compileSizes = {128, 256};
    policy.maxScheduledTokens = 1024;
    policy.hashBlockSize = policy.schedulerBlockSize = 16;
    ASSERT_TRUE(TPUPrefillExecutionPolicy::validate(policy));
    TPUPrefillPP4Timing::Parameters parameters{};
    parameters[1] = 10.0;
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    TPUPrefillPP4Request incoming;
    incoming.requestId = 1;
    incoming.contextTokens = incoming.remainingTokens = 16;
    incoming.owned = true;
    incoming.releaseSlotAfterPrefill = true;
    const auto expected =
        TPUPrefillPP4Forecast::predict(state, {incoming}, policy, parameters, 1000);
    ASSERT_EQ(1u, expected.ownedSteps);
    ASSERT_EQ(16, expected.scheduledTokens.at(1));
    ASSERT_GT(expected.totalMs, 0.0);

    // The old 19-parameter model is the control on the same real Master path.
    for (bool native : {false, true}) {
        auto node = std::make_shared<NativeMasterDiagnosticNode>();
        auto info = std::make_shared<NaviSchedNodeInfo>(node->getInfo());
        info->pdSeparated = info->pdUnifiedCapacitySplit = true;
        info->unifiedLatencyPrediction = true;
        info->nativePrefillPipelineStages = 4;
        info->prefillExecutionPolicy = policy;
        auto model = std::make_shared<NaviSchedLatencyModel>();
        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        model->binaryVersion = native ? NonLinearPrefillModelR::NATIVE_PP4_BINARY_VERSION
                                      : NonLinearPrefillModelR::BINARY_VERSION;
        model->revision = 1;
        model->parameters =
            native ? std::vector<double>(parameters.begin(), parameters.end())
                   : legacyParameters;
        ASSERT_TRUE(NonLinearPrefillModelR::validateParametersForVersion(
            model->parameters, model->binaryVersion));
        NaviSchedNodeSnapshot snapshot;
        snapshot.info = info;
        snapshot.latencyModel = model;
        snapshot.prefillExecutionPolicy = policy;
        snapshot.residualRuntimeConfig.learningRate = 0.0;
        snapshot.unifiedLatencyPrediction = true;
        snapshot.maxRunningRequestCount = 64;
        snapshot.load.totalBlocks = 100;
        auto observed = std::make_shared<NaviSchedNativePPState>();
        observed->observationId = observed->burst = 1;
        observed->occupiedRequestSlotCount = 0;
        ASSERT_TRUE(observed->valid());
        snapshot.workload.revision = snapshot.workload.eventTimeUs = 1;
        snapshot.workload.prefillCapacitySplit = true;
        snapshot.workload.maxRunningRequestCount = 64;
        snapshot.workload.nativePPObservationAgeUs = 0;
        snapshot.workload.nativePPState = observed;
        ASSERT_TRUE(snapshot.workload.valid());
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        context._requests.resize(1);
        context.setNodes({node}, {snapshot});
        ASSERT_EQ(1u, context.nodeCount());
        ASSERT_TRUE(context.getNode(0).hasValidLatencyModel());
        const auto finalPrediction = context.estimateBatchExecutionLatencyMs(0, {0});
        int64_t tokens[] = {32};
        float costs[2] = {};
        ASSERT_TRUE(CostSchedulerKernelInternal::fillCostsForNodeRange(context, tokens, 0,
                                                                       1, costs));
        double weights[] = {1.0}, inverseTokens[] = {1.0 / 32.0};
        CostSchedulerKernelInternal::NodeGradient gradient;
        const auto loss = CostSchedulerKernelInternal::forwardNodeRange(
            context, costs, weights, tokens, inverseTokens, 32.0, 0.5, 0.0, 0, 1,
            &gradient, nullptr, nullptr);
        uint32_t selected[] = {0};
        CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
        const auto discreteLoss = CostSchedulerKernelInternal::forwardDiscrete(
            context, costs, selected, tokens, inverseTokens, 32.0, 0.5, 0.0, workspace);
        double weightGradient = 0.0;
        CostSchedulerKernelInternal::backwardNodeRange(context, costs, weights, tokens,
                                                       inverseTokens, 1, 0, 1, &gradient,
                                                       &weightGradient);
        std::cout << "NATIVE_MASTER_VERTICAL native=" << native
                  << " expected_pp_total_ms=" << expected.totalMs
                  << " final_ms=" << finalPrediction << " running_cost=" << costs[0]
                  << " waiting_cost=" << costs[1] << " continuous_loss=" << loss
                  << " discrete_loss=" << discreteLoss
                  << " weight_gradient=" << weightGradient
                  << " gradient_cost=" << gradient.cost << std::endl;
        EXPECT_TRUE(std::isfinite(finalPrediction));
        EXPECT_TRUE(std::isfinite(costs[0]));
        EXPECT_TRUE(std::isfinite(costs[1]));
        EXPECT_TRUE(std::isfinite(loss));
        EXPECT_TRUE(std::isfinite(discreteLoss));
        EXPECT_TRUE(std::isfinite(gradient.cost));
        EXPECT_TRUE(std::isfinite(weightGradient));
        if (native) {
            EXPECT_NEAR(expected.totalMs, finalPrediction, 1e-8);
            EXPECT_NEAR(expected.totalMs / 32.0, loss, 1e-8);
            EXPECT_NEAR(expected.totalMs / 32.0, discreteLoss, 1e-8);
            EXPECT_GT(weightGradient, 0.0);
        }
    }
}

TEST_F(NativePP4TrainingIntegrationTest, MasterNativeExactForecastAndDiscreteCost)
{
    NonLinearPrefillModelR legacyModel;
    legacyModel._useCapacitySplitFeature = true;
    auto legacyParameters = legacyModel.getInitialParameters();
    legacyParameters[NonLinearPrefillModelR::ZERO_POINT_PARAMETER_INDEX] =
        100.0 / NonLinearPrefillModelR::getParameterScale(
                    NonLinearPrefillModelR::ZERO_POINT_PARAMETER_INDEX);
    ASSERT_EQ(19u, legacyParameters.size());
    ASSERT_EQ(NonLinearPrefillModelR::BINARY_VERSION, legacyModel.getBinaryVersion());
    ASSERT_TRUE(NonLinearPrefillModelR::validateParametersForVersion(
        legacyParameters, legacyModel.getBinaryVersion()));
    TPUPrefillExecutionPolicyConfig policy;
    policy.pcpSize = 8;
    policy.interleaveSize = 128;
    policy.compileSizes = {128, 256};
    policy.maxScheduledTokens = 1024;
    policy.hashBlockSize = policy.schedulerBlockSize = 16;
    ASSERT_TRUE(TPUPrefillExecutionPolicy::validate(policy));
    TPUPrefillPP4Timing::Parameters parameters{};
    parameters[1] = 10.0;
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    TPUPrefillPP4Request incoming;
    incoming.requestId = 1;
    incoming.contextTokens = incoming.remainingTokens = 16;
    incoming.owned = true;
    incoming.releaseSlotAfterPrefill = true;
    const auto expected =
        TPUPrefillPP4Forecast::predict(state, {incoming}, policy, parameters, 1000);
    ASSERT_EQ(1u, expected.ownedSteps);
    ASSERT_EQ(16, expected.scheduledTokens.at(1));
    ASSERT_GT(expected.totalMs, 0.0);

    // The old 19-parameter model is the control on the same real Master path.
    for (bool native : {false, true}) {
        auto node = std::make_shared<NativeMasterDiagnosticNode>();
        auto info = std::make_shared<NaviSchedNodeInfo>(node->getInfo());
        info->pdSeparated = info->pdUnifiedCapacitySplit = true;
        info->unifiedLatencyPrediction = true;
        info->nativePrefillPipelineStages = 4;
        info->prefillExecutionPolicy = policy;
        auto model = std::make_shared<NaviSchedLatencyModel>();
        model->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        model->binaryVersion = native ? NonLinearPrefillModelR::NATIVE_PP4_BINARY_VERSION
                                      : NonLinearPrefillModelR::BINARY_VERSION;
        model->revision = 1;
        model->parameters =
            native ? std::vector<double>(parameters.begin(), parameters.end())
                   : legacyParameters;
        ASSERT_TRUE(NonLinearPrefillModelR::validateParametersForVersion(
            model->parameters, model->binaryVersion));
        NaviSchedNodeSnapshot snapshot;
        snapshot.info = info;
        snapshot.latencyModel = model;
        snapshot.prefillExecutionPolicy = policy;
        snapshot.residualRuntimeConfig.learningRate = 0.0;
        snapshot.unifiedLatencyPrediction = true;
        snapshot.maxRunningRequestCount = 64;
        snapshot.load.totalBlocks = 100;
        auto observed = std::make_shared<NaviSchedNativePPState>();
        observed->observationId = observed->burst = 1;
        observed->occupiedRequestSlotCount = 0;
        ASSERT_TRUE(observed->valid());
        snapshot.workload.revision = snapshot.workload.eventTimeUs = 1;
        snapshot.workload.prefillCapacitySplit = true;
        snapshot.workload.maxRunningRequestCount = 64;
        snapshot.workload.nativePPObservationAgeUs = 0;
        snapshot.workload.nativePPState = observed;
        ASSERT_TRUE(snapshot.workload.valid());
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        context._requests.resize(1);
        context.setNodes({node}, {snapshot});
        ASSERT_EQ(1u, context.nodeCount());
        ASSERT_TRUE(context.getNode(0).hasValidLatencyModel());
        const auto finalPrediction = context.estimateBatchExecutionLatencyMs(0, {0});
        int64_t tokens[] = {32};
        float costs[2] = {};
        ASSERT_TRUE(CostSchedulerKernelInternal::fillCostsForNodeRange(context, tokens, 0,
                                                                       1, costs));
        double weights[] = {1.0}, inverseTokens[] = {1.0 / 32.0};
        CostSchedulerKernelInternal::NodeGradient gradient;
        const auto loss = CostSchedulerKernelInternal::forwardNodeRange(
            context, costs, weights, tokens, inverseTokens, 32.0, 0.5, 0.0, 0, 1,
            &gradient, nullptr, nullptr);
        uint32_t selected[] = {0};
        CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
        const auto discreteLoss = CostSchedulerKernelInternal::forwardDiscrete(
            context, costs, selected, tokens, inverseTokens, 32.0, 0.5, 0.0, workspace);
        double weightGradient = 0.0;
        CostSchedulerKernelInternal::backwardNodeRange(context, costs, weights, tokens,
                                                       inverseTokens, 1, 0, 1, &gradient,
                                                       &weightGradient);
        std::cout << "NATIVE_MASTER_VERTICAL native=" << native
                  << " expected_pp_total_ms=" << expected.totalMs
                  << " final_ms=" << finalPrediction << " running_cost=" << costs[0]
                  << " waiting_cost=" << costs[1] << " continuous_loss=" << loss
                  << " discrete_loss=" << discreteLoss
                  << " weight_gradient=" << weightGradient
                  << " gradient_cost=" << gradient.cost << std::endl;
        EXPECT_TRUE(std::isfinite(finalPrediction));
        EXPECT_TRUE(std::isfinite(costs[0]));
        EXPECT_TRUE(std::isfinite(costs[1]));
        EXPECT_TRUE(std::isfinite(loss));
        EXPECT_TRUE(std::isfinite(discreteLoss));
        EXPECT_TRUE(std::isfinite(gradient.cost));
        EXPECT_TRUE(std::isfinite(weightGradient));
        if (native) {
            EXPECT_NEAR(expected.totalMs, finalPrediction, 1e-8);
            // Continuous relaxation is checked by the separate full-consumer regression.
            (void)loss;
            EXPECT_NEAR(expected.totalMs / 32.0, discreteLoss, 1e-8);
            EXPECT_GT(weightGradient, 0.0);
        }
    }
}

namespace {
NaviSchedNodeSnapshot nativeConsumerSnapshot()
{
    NativeMasterSnapshotFixture fixture;
    auto snapshot = NaviSchedNodeSnapshot();
    snapshot.info = fixture.published->info;
    snapshot.latencyModel = fixture.published->latencyModel;
    auto model = std::make_shared<NaviSchedLatencyModel>(*snapshot.latencyModel);
    model->revision = 1;
    model->parameters[1] = 10.0;
    snapshot.latencyModel = model;
    TPUPrefillExecutionPolicyConfig policy;
    policy.pcpSize = 8;
    policy.interleaveSize = 128;
    policy.compileSizes = {128, 256};
    policy.maxScheduledTokens = 1024;
    policy.hashBlockSize = policy.schedulerBlockSize = 16;
    snapshot.prefillExecutionPolicy = policy;
    snapshot.workload = fixture.node->_runtimeState.workload;
    snapshot.maxRunningRequestCount = 64;
    snapshot.unifiedLatencyPrediction = true;
    snapshot.residualRuntimeConfig.learningRate = 0.0;
    return snapshot;
}

NaviSchedNativePPRequest readyNativeRequest(uint64_t id, int64_t index)
{
    NaviSchedNativePPRequest request;
    request.requestId = id;
    request.queue = NativePPQueue::WAITING;
    request.queueIndex = index;
    request.phase = NativePPRequestPhase::CACHE_KNOWN;
    request.promptTokens = 32;
    request.cacheHitTokens = 16;
    request.prefillSlotRelease = NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT;
    return request;
}

void setNativeConsumerRows(NaviSchedNodeSnapshot &snapshot,
                           std::vector<NaviSchedNativePPRequest> requests)
{
    auto state =
        std::make_shared<NaviSchedNativePPState>(*snapshot.workload.nativePPState);
    state->requests = std::move(requests);
    snapshot.workload.nativePPState = state;
    auto &w = snapshot.workload;
    w.requestCount = w.waitingRequestCount = w.remainingPrefillRequestCount =
        w.runningCapacityPrefillRequestCount = w.runningCapacitySlotCount =
            state->requests.size();
    w.batchCount = state->requests.empty() ? 0 : 1;
    w.remainingPrefillContextTokenCount = w.remainingPrefillTokenCount = 0;
    w.remainingPrefillTokenSquareSum = w.contextRemainingPrefillTokenProductSum = 0;
    w.totalPromptTokenCount = 0;
    for (const auto &r : state->requests) {
        const auto hit = std::max<int64_t>(0, r.cacheHitTokens);
        const auto remaining = r.promptTokens - hit;
        w.remainingPrefillContextTokenCount += hit;
        w.remainingPrefillTokenCount += remaining;
        w.remainingPrefillTokenSquareSum += double(remaining) * remaining;
        w.contextRemainingPrefillTokenProductSum += double(hit) * remaining;
        w.totalPromptTokenCount += r.promptTokens;
    }
    w.runningCapacityPrefillContextTokenCount = w.remainingPrefillContextTokenCount;
    w.runningCapacityPrefillTokenCount = w.remainingPrefillTokenCount;
    w.runningCapacityPrefillTokenSquareSum = w.remainingPrefillTokenSquareSum;
    w.runningCapacityContextRemainingPrefillTokenProductSum =
        w.contextRemainingPrefillTokenProductSum;
    w.remainingTotalContextTokenCount = w.remainingPrefillContextTokenCount;
    w.remainingTotalTokenCount = w.remainingPrefillTokenCount;
    w.remainingTotalTokenSquareSum = w.remainingPrefillTokenSquareSum;
    w.contextRemainingTotalTokenProductSum = w.contextRemainingPrefillTokenProductSum;
}
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeMasterJoinUsesNativeOrderAndNormalizedPreparedFacts)
{
    auto snapshot = nativeConsumerSnapshot();
    auto prepared = readyNativeRequest(7, 0);
    prepared.queue = NativePPQueue::PREPARED;
    prepared.phase = NativePPRequestPhase::NOT_REGISTERED;
    prepared.promptTokens = 33;
    prepared.cacheHitTokens = -1;
    prepared.prefillSlotRelease = NativePPPrefillSlotRelease::UNKNOWN;
    setNativeConsumerRows(snapshot,
                          {readyNativeRequest(9, 0), readyNativeRequest(3, 1), prepared});
    ASSERT_TRUE(snapshot.workload.valid());
    auto three = nativeReservedRequest(3);
    three.predictedCacheHitTokenCount = 0;
    auto nine = nativeReservedRequest(9);
    nine.predictedCacheHitTokenCount = 0;
    snapshot.nativePrefillReservations = {
        {three, MasterBatchReservation::RequestState::DISPATCHED},
        {nativeReservedRequest(7), MasterBatchReservation::RequestState::DISPATCHED},
        {nine, MasterBatchReservation::RequestState::WAITING}};
    const auto raw = snapshot.workload.nativePPState;
    auto transport = std::make_shared<NativeMasterDiagnosticNode>();
    NaviSchedScheduleNode node(transport, snapshot, 1000);
    const auto first = node.getNativePrefillObservation();
    EXPECT_EQ(first, node.getNativePrefillObservation());
    ASSERT_EQ(3u, first->requests.size());
    EXPECT_EQ(9u, first->requests[0].requestId);
    EXPECT_EQ(3u, first->requests[1].requestId);
    EXPECT_EQ(7u, first->requests[2].requestId);
    for (const auto &r : first->requests) {
        EXPECT_EQ(16, r.contextTokens);
        EXPECT_EQ(16, r.remainingTokens);
        ASSERT_TRUE(r.releaseSlotAfterPrefill);
        EXPECT_TRUE(*r.releaseSlotAfterPrefill);
    }
    EXPECT_EQ(16, first->requests[2].sharedPrefixBoundary);
    EXPECT_EQ(0, first->requests[0].sharedPrefixBoundary);
    EXPECT_EQ(33, raw->requests[2].promptTokens);
    EXPECT_EQ(-1, raw->requests[2].cacheHitTokens);
    EXPECT_EQ(NativePPPrefillSlotRelease::UNKNOWN, raw->requests[2].prefillSlotRelease);
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeMasterJoinDoesNotResurrectDispatchedOrDuplicatePreparedWork)
{
    auto snapshot = nativeConsumerSnapshot();
    setNativeConsumerRows(snapshot, {readyNativeRequest(3, 0)});
    snapshot.nativePrefillReservations = {
        {nativeReservedRequest(1), MasterBatchReservation::RequestState::WAITING},
        {nativeReservedRequest(2), MasterBatchReservation::RequestState::DISPATCHED}};
    NaviSchedScheduleNode node(std::make_shared<NativeMasterDiagnosticNode>(), snapshot,
                               1000);
    TPUPrefillPP4ObservationPredictions predictions;
    TPUPrefillPP4Request incoming;
    incoming.requestId = 999;
    incoming.remainingTokens = 1;
    incoming.owned = true;
    incoming.releaseSlotAfterPrefill = true;
    predictions.incoming = {incoming};
    const auto value = node.prepareNativePrefillObservation(predictions);
    ASSERT_EQ(3u, value.requests.size());
    EXPECT_EQ(3u, value.requests[0].requestId);
    EXPECT_EQ(1u, value.requests[1].requestId);
    EXPECT_FALSE(value.requests[1].owned);
    EXPECT_EQ(999u, value.requests[2].requestId);
    EXPECT_TRUE(value.requests[2].owned);
    predictions.incoming[0].requestId = 1;
    EXPECT_THROW(node.prepareNativePrefillObservation(predictions),
                 std::invalid_argument);
    snapshot.nativePrefillReservations.push_back(
        snapshot.nativePrefillReservations.front());
    NaviSchedScheduleNode duplicate(std::make_shared<NativeMasterDiagnosticNode>(),
                                    snapshot, 1000);
    EXPECT_THROW(duplicate.prepareNativePrefillObservation({}), std::invalid_argument);
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeMasterJoinNeedsFutureReadinessAndExplicitLifetime)
{
    auto snapshot = nativeConsumerSnapshot();
    auto loading = readyNativeRequest(3, 0);
    loading.phase = NativePPRequestPhase::KV_LOADING;
    setNativeConsumerRows(snapshot, {loading});
    NaviSchedScheduleNode node(std::make_shared<NativeMasterDiagnosticNode>(), snapshot,
                               1000);
    EXPECT_THROW(node.prepareNativePrefillObservation({}), std::invalid_argument);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.requests[3].readyAfterObservationMs = 20.0;
    const auto value = node.prepareNativePrefillObservation(predictions);
    EXPECT_DOUBLE_EQ(20.0, value.requests[0].readyTimeMs);
    snapshot = nativeConsumerSnapshot();
    auto reserved = nativeReservedRequest(1);
    reserved.nativePrefillSlotRelease = NativePPPrefillSlotRelease::UNKNOWN;
    snapshot.nativePrefillReservations = {
        {reserved, MasterBatchReservation::RequestState::WAITING}};
    NaviSchedScheduleNode unknown(std::make_shared<NativeMasterDiagnosticNode>(),
                                  snapshot, 1000);
    EXPECT_THROW(unknown.prepareNativePrefillObservation({}), std::invalid_argument);
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeMasterForecastKeepsFullHitRecomputeAndOneCacheObservation)
{
    auto snapshot = nativeConsumerSnapshot();
    auto transport = std::make_shared<NativeMasterDiagnosticNode>();
    transport->fullHit = true;
    NaviSchedScheduleContext context;
    context._scheduleTimeUs = 1000;
    context._requests = nativeMasterDiagnosticRequests();
    context.setNodes({transport}, {snapshot});
    const auto counts = context.getCacheHitBlockCountsSnapshot(0);
    ASSERT_TRUE(counts);
    EXPECT_EQ(2, (*counts)[0]);
    EXPECT_DOUBLE_EQ(40.0, context.estimateBatchExecutionLatencyMs(0, {0}));
    EXPECT_DOUBLE_EQ(40.0, context.estimateBatchExecutionLatencyMs(0, {0}));
    EXPECT_EQ(1u, transport->cacheCalls);
    snapshot.prefillExecutionPolicy->asyncKVLookupEnabled = true;
    snapshot.prefillExecutionPolicy->asyncKVLookupHashBlockSize = 16;
    context.setNodes({transport}, {snapshot});
    EXPECT_TRUE(std::isnan(context.estimateBatchExecutionLatencyMs(0, {0})));
}

TEST_F(NativePP4TrainingIntegrationTest, NativeAsyncCompleteHBMPrefixSkipsQuery)
{
    const auto predict = [](int64_t prompt, int64_t hbmOrTotalHit, int64_t externalRead,
                            bool eagle) {
        auto snapshot = nativeConsumerSnapshot();
        snapshot.prefillExecutionPolicy->asyncKVLookupEnabled = true;
        snapshot.prefillExecutionPolicy->asyncKVLookupHashBlockSize = 16;
        snapshot.prefillExecutionPolicy->asyncKVLookupUsesEagle = eagle;
        auto transport = std::make_shared<NativeMasterDiagnosticNode>();
        transport->cachePrefixTokens = hbmOrTotalHit;
        transport->externalReadTokenCount = externalRead;
        transport->readyHbmTokens = externalRead == 0 ? hbmOrTotalHit : -1;
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        context._requests[0].request.promptTokenCount = prompt;
        context.setNodes({transport}, {snapshot});
        const auto result = context.estimateBatchExecutionLatencyMs(0, {0});
        EXPECT_EQ(1u, transport->cacheCalls);
        return result;
    };
    EXPECT_DOUBLE_EQ(40.0, predict(33, 32, 0, false));
    EXPECT_DOUBLE_EQ(40.0, predict(33, 32, 0, true));
    EXPECT_DOUBLE_EQ(40.0, predict(32, 16, 0, true));
    EXPECT_TRUE(std::isnan(predict(33, 16, 0, false)));
    EXPECT_TRUE(std::isnan(predict(33, 32, 16, false)));
    EXPECT_TRUE(std::isnan(predict(33, 32, -1, false)));
    EXPECT_TRUE(std::isnan(predict(48, 32, 0, false)));
    EXPECT_TRUE(std::isnan(predict(32, 32, 0, false)));
    {
        auto snapshot = nativeConsumerSnapshot();
        auto &policy = *snapshot.prefillExecutionPolicy;
        policy.asyncKVLookupEnabled = true;
        policy.asyncKVLookupHashBlockSize = 4096;
        policy.hashBlockSize = policy.schedulerBlockSize = 4096;
        policy.interleaveSize = 256;
        policy.compileSizes = {256, 512, 1024};
        policy.maxScheduledTokens = 8192;
        auto transport = std::make_shared<NativeMasterDiagnosticNode>();
        transport->cachePrefixTokens = 106496;
        transport->readyHbmTokens = 106496;
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        context._requests[0].request.promptTokenCount = 107658;
        context._requests[0].request.blockSize = 4096;
        auto hashes = std::make_shared<NaviSchedBlockHashes>();
        for (uint64_t block = 1; block <= 26; ++block) hashes->push_back(block);
        context._requests[0].request.blockHashes = hashes;
        context.setNodes({transport}, {snapshot});
        EXPECT_DOUBLE_EQ(40.0, context.estimateBatchExecutionLatencyMs(0, {0}));
        EXPECT_EQ(1u, transport->cacheCalls);
    }
    if (std::getenv("NAVI_E274_HBM_QUERY_BENCHMARK")) {
        for (bool asyncLookup : {false, true}) {
            std::vector<double> firstUs, repeatedUs;
            double sum = 0.0;
            for (int iteration = 0; iteration < 16; ++iteration) {
                auto snapshot = nativeConsumerSnapshot();
                snapshot.prefillExecutionPolicy->asyncKVLookupEnabled = asyncLookup;
                snapshot.prefillExecutionPolicy->asyncKVLookupHashBlockSize = 16;
                auto transport = std::make_shared<NativeMasterDiagnosticNode>();
                transport->cachePrefixTokens = 32;
                transport->readyHbmTokens = 32;
                NaviSchedScheduleContext context;
                context._scheduleTimeUs = 1000;
                context._requests = nativeMasterDiagnosticRequests();
                context._requests[0].request.promptTokenCount = 33;
                context.setNodes({transport}, {snapshot});
                auto begin = std::chrono::steady_clock::now();
                EXPECT_DOUBLE_EQ(40.0, context.estimateBatchExecutionLatencyMs(0, {0}));
                firstUs.push_back(std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - begin)
                                      .count());
                begin = std::chrono::steady_clock::now();
                for (int sample = 0; sample < 2000; ++sample) {
                    sum += context.estimateBatchExecutionLatencyMs(0, {0});
                }
                repeatedUs.push_back(std::chrono::duration<double, std::micro>(
                                         std::chrono::steady_clock::now() - begin)
                                         .count() /
                                     2000.0);
                EXPECT_EQ(1u, transport->cacheCalls);
            }
            EXPECT_DOUBLE_EQ(16.0 * 2000.0 * 40.0, sum);
            std::sort(firstUs.begin(), firstUs.end());
            std::sort(repeatedUs.begin(), repeatedUs.end());
            std::cout << "NATIVE_HBM_QUERY_COST async=" << asyncLookup
                      << " first_p50_us=" << firstUs[8]
                      << " repeated_p50_us=" << repeatedUs[8]
                      << " repeated_p95_us=" << repeatedUs[15] << "\n";
        }
    }
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeWaitingReservationUsesPredictedCacheWithoutAsyncLookup)
{
    auto snapshot = nativeConsumerSnapshot();
    auto waiting = readyNativeRequest(3, 0);
    waiting.phase = NativePPRequestPhase::QUERY_NOT_STARTED;
    waiting.cacheHitTokens = -1;
    setNativeConsumerRows(snapshot, {waiting});
    snapshot.nativePrefillReservations = {
        {nativeReservedRequest(3), MasterBatchReservation::RequestState::DISPATCHED}};
    auto transport = std::make_shared<NativeMasterDiagnosticNode>();
    NaviSchedScheduleNode node(transport, snapshot, 1000);
    const auto input = node.getNativePrefillObservation();
    ASSERT_EQ(1u, input->requests.size());
    EXPECT_EQ(16, input->requests[0].contextTokens);
    EXPECT_EQ(16, input->requests[0].remainingTokens);
    EXPECT_FALSE(input->requests[0].needsFirstQuery);
    EXPECT_DOUBLE_EQ(0.0, input->requests[0].readyTimeMs);
    EXPECT_EQ(-1, snapshot.workload.nativePPState->requests[0].cacheHitTokens);
    EXPECT_EQ(0u, transport->cacheCalls);
    NaviSchedScheduleContext context;
    context._scheduleTimeUs = 1000;
    context._requests = nativeMasterDiagnosticRequests();
    context.setNodes({transport}, {snapshot});
    EXPECT_DOUBLE_EQ(40.0, context.estimateBatchExecutionLatencyMs(0, {0}));
    EXPECT_EQ(1u, transport->cacheCalls);
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeWaitingReservationPreservesCacheAndReadinessInputs)
{
    auto snapshot = nativeConsumerSnapshot();
    auto waiting = readyNativeRequest(3, 0);
    waiting.phase = NativePPRequestPhase::QUERY_NOT_STARTED;
    waiting.cacheHitTokens = -1;
    setNativeConsumerRows(snapshot, {waiting});
    snapshot.nativePrefillReservations = {
        {nativeReservedRequest(3), MasterBatchReservation::RequestState::DISPATCHED}};
    snapshot.prefillExecutionPolicy->asyncKVLookupEnabled = true;
    snapshot.prefillExecutionPolicy->asyncKVLookupHashBlockSize = 16;
    NaviSchedScheduleNode unknown(std::make_shared<NativeMasterDiagnosticNode>(),
                                  snapshot, 1000);
    EXPECT_THROW(unknown.prepareNativePrefillObservation({}), std::invalid_argument);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.requests[3].cacheHitTokens = 0;
    predictions.requests[3].readyAfterObservationMs = 7.0;
    predictions.requests[3].needsFirstQuery = true;
    const auto explicitInput = unknown.prepareNativePrefillObservation(predictions);
    ASSERT_EQ(1u, explicitInput.requests.size());
    EXPECT_EQ(0, explicitInput.requests[0].contextTokens);
    EXPECT_TRUE(explicitInput.requests[0].needsFirstQuery);
    EXPECT_DOUBLE_EQ(7.0, explicitInput.requests[0].readyTimeMs);

    snapshot.prefillExecutionPolicy->asyncKVLookupEnabled = false;
    waiting.phase = NativePPRequestPhase::QUERY_PENDING;
    setNativeConsumerRows(snapshot, {waiting});
    NaviSchedScheduleNode pending(std::make_shared<NativeMasterDiagnosticNode>(),
                                  snapshot, 1000);
    EXPECT_THROW(pending.prepareNativePrefillObservation({}), std::invalid_argument);

    waiting.phase = NativePPRequestPhase::CACHE_KNOWN;
    waiting.cacheHitTokens = 0;
    setNativeConsumerRows(snapshot, {waiting});
    NaviSchedScheduleNode resolved(std::make_shared<NativeMasterDiagnosticNode>(),
                                   snapshot, 1000);
    const auto nativeInput = resolved.prepareNativePrefillObservation({});
    EXPECT_EQ(0, nativeInput.requests[0].contextTokens);
}

TEST_F(NativePP4TrainingIntegrationTest, NativeWaitingReservationCacheBenchmark)
{
    if (!std::getenv("NAVI_E277_WAITING_CACHE_BENCHMARK")) GTEST_SKIP();
    for (size_t count : {size_t(1), size_t(16), size_t(64)}) {
        for (bool unresolved : {false, true}) {
            std::vector<double> firstUs, repeatedUs;
            for (int iteration = 0; iteration < 16; ++iteration) {
                auto snapshot = nativeConsumerSnapshot();
                std::vector<NaviSchedNativePPRequest> requests;
                for (size_t i = 0; i < count; ++i) {
                    auto waiting = readyNativeRequest(i + 2, i);
                    waiting.phase = NativePPRequestPhase::QUERY_NOT_STARTED;
                    if (unresolved) waiting.cacheHitTokens = -1;
                    requests.push_back(waiting);
                    snapshot.nativePrefillReservations.push_back(
                        {nativeReservedRequest(i + 2),
                         MasterBatchReservation::RequestState::DISPATCHED});
                }
                setNativeConsumerRows(snapshot, std::move(requests));
                auto transport = std::make_shared<NativeMasterDiagnosticNode>();
                NaviSchedScheduleNode node(transport, snapshot, 1000);
                auto begin = std::chrono::steady_clock::now();
                const auto input = node.getNativePrefillObservation();
                firstUs.push_back(std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - begin)
                                      .count());
                ASSERT_EQ(count, input->requests.size());
                for (const auto &r : input->requests) {
                    EXPECT_EQ(16, r.contextTokens);
                    EXPECT_FALSE(r.needsFirstQuery);
                }
                begin = std::chrono::steady_clock::now();
                for (int sample = 0; sample < 2000; ++sample)
                    EXPECT_EQ(input, node.getNativePrefillObservation());
                repeatedUs.push_back(std::chrono::duration<double, std::micro>(
                                         std::chrono::steady_clock::now() - begin)
                                         .count() /
                                     2000.0);
                EXPECT_EQ(0u, transport->cacheCalls);
            }
            std::sort(firstUs.begin(), firstUs.end());
            std::sort(repeatedUs.begin(), repeatedUs.end());
            std::cout << "NATIVE_WAITING_CACHE_COST count=" << count
                      << " unresolved=" << unresolved << " first_p50_us=" << firstUs[8]
                      << " repeated_p50_us=" << repeatedUs[8]
                      << " repeated_p95_us=" << repeatedUs[15] << "\n";
        }
    }
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeMasterObservationAgeAndConflictingPromptAreExplicit)
{
    auto snapshot = nativeConsumerSnapshot();
    snapshot.workloadReportReceiveTimeUs = 1000;
    NaviSchedScheduleNode node(std::make_shared<NativeMasterDiagnosticNode>(), snapshot,
                               2500);
    const auto input = node.getNativePrefillObservation();
    EXPECT_DOUBLE_EQ(1.5, input->state.nowMs);
    snapshot = nativeConsumerSnapshot();
    setNativeConsumerRows(snapshot, {readyNativeRequest(1, 0)});
    auto reserved = nativeReservedRequest(1);
    reserved.totalTokenCount = 33;
    snapshot.nativePrefillReservations = {
        {reserved, MasterBatchReservation::RequestState::DISPATCHED}};
    NaviSchedScheduleNode inconsistent(std::make_shared<NativeMasterDiagnosticNode>(),
                                       snapshot, 1000);
    EXPECT_THROW(inconsistent.prepareNativePrefillObservation({}), std::invalid_argument);
}

TEST_F(NativePP4TrainingIntegrationTest, NativeMasterExactForecastBenchmark)
{
    if (!std::getenv("NAVI_E257_FORECAST_BENCHMARK")) return;
    for (size_t count : {size_t(1), size_t(16), size_t(64)}) {
        auto snapshot = nativeConsumerSnapshot();
        auto transport = std::make_shared<NativeMasterDiagnosticNode>();
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        context._requests.resize(count, context._requests.front());
        std::vector<size_t> indexes;
        std::vector<TPUPrefillPP4Request> requests;
        for (size_t index = 0; index < count; ++index) {
            context._requests[index].request.requestId = index + 1;
            indexes.push_back(index);
            TPUPrefillPP4Request r;
            r.requestId = index + 1;
            r.contextTokens = r.remainingTokens = 16;
            r.owned = true;
            r.releaseSlotAfterPrefill = true;
            requests.push_back(r);
        }
        context.setNodes({transport}, {snapshot});
        TPUPrefillPP4State state;
        state.maxRunningRequestCount = 64;
        TPUPrefillPP4Timing::Parameters parameters{};
        parameters[1] = 10.0;
        const size_t bound = 8 * (requests.size() * 17 + 1);
        for (bool master : {false, true, true, false}) {
            for (size_t sample = 0; sample < 7; ++sample) {
                double sum = 0.0;
                const auto start = std::chrono::steady_clock::now();
                for (size_t iteration = 0; iteration < 1000; ++iteration)
                    sum += master ? context.estimateBatchExecutionLatencyMs(0, indexes)
                                  : TPUPrefillPP4Forecast::predict(
                                        state, requests, *snapshot.prefillExecutionPolicy,
                                        parameters, bound, false)
                                        .totalMs;
                const double ns = std::chrono::duration<double, std::nano>(
                                      std::chrono::steady_clock::now() - start)
                                      .count() /
                                  1000;
                ASSERT_DOUBLE_EQ(40000.0, sum);
                if (sample > 0)
                    std::cout << "E257_NATIVE_FORECAST_BENCH count=" << count
                              << " master=" << master << " ns=" << std::setprecision(17)
                              << ns << "\n";
            }
        }
        EXPECT_EQ(1u, transport->cacheCalls);
    }
}

namespace {
void setupNativeContinuousContext(NaviSchedScheduleContext &context,
                                  const std::vector<int64_t> &prompts, size_t nodes,
                                  bool biasOnly = false)
{
    context._scheduleTimeUs = 1000;
    for (size_t index = 0; index < prompts.size(); ++index) {
        auto item = nativeMasterDiagnosticRequests().front();
        item.request.requestId = index + 1;
        item.request.promptTokenCount = prompts[index];
        item.request.blockHashes =
            std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes(
                static_cast<size_t>((prompts[index] + 15) / 16), index + 100));
        context._requests.push_back(std::move(item));
    }
    std::vector<std::shared_ptr<const NaviSchedNode>> transports;
    std::vector<NaviSchedNodeSnapshot> snapshots;
    for (size_t node = 0; node < nodes; ++node) {
        auto snapshot = nativeConsumerSnapshot();
        auto model = std::make_shared<NaviSchedLatencyModel>(*snapshot.latencyModel);
        model->parameters =
            biasOnly ? std::vector<double>{0, 10, 0, 0, 0, 0, 0, 0, 0}
                     : std::vector<double>{3.5, 1.7, .12, .09, .07, .05, .03, .25, .06};
        for (auto &parameter : model->parameters) parameter *= 1.0 + .13 * node;
        snapshot.latencyModel = model;
        transports.push_back(std::make_shared<NativeMasterDiagnosticNode>());
        snapshots.push_back(snapshot);
    }
    context.setNodes(transports, snapshots);
}

void checkNativeContinuousGradient(NaviSchedScheduleContext &context,
                                   std::vector<double> weights, double concavity,
                                   double lambda = .37)
{
    const auto n = context.requestCount(), m = context.nodeCount();
    std::vector<int64_t> tokens(n);
    std::vector<double> inverse(n);
    CostSchedulerKernelInternal::fillRequestTokenCounts(context, tokens.data(),
                                                        inverse.data());
    const auto denominator =
        CostSchedulerKernelInternal::calculateTokenNormalizationDenominator(tokens.data(),
                                                                            n, 1.0);
    std::vector<float> costs(n * m * 2);
    ASSERT_TRUE(CostSchedulerKernelInternal::fillCostsForNodeRange(context, tokens.data(),
                                                                   0, m, costs.data()));
    for (float cost : costs) ASSERT_EQ(0.0f, cost);
    std::vector<CostSchedulerKernelInternal::NodeGradient> nodeGradients(m);
    const auto forward = [&](CostSchedulerKernelInternal::NodeGradient *gradients) {
        return CostSchedulerKernelInternal::forwardNodeRange(
            context, costs.data(), weights.data(), tokens.data(), inverse.data(),
            denominator, lambda, concavity, 0, m, gradients, nullptr, nullptr);
    };
    ASSERT_TRUE(std::isfinite(forward(nodeGradients.data())));
    std::vector<double> gradients(n * m);
    CostSchedulerKernelInternal::backwardNodeRange(
        context, costs.data(), weights.data(), tokens.data(), inverse.data(), n, 0, m,
        nodeGradients.data(), gradients.data());
    constexpr double epsilon = 1e-5;
    for (size_t i = 0; i < weights.size(); ++i) {
        const double saved = weights[i];
        weights[i] = saved + epsilon;
        const double plus = forward(nullptr);
        weights[i] = saved - epsilon;
        const double minus = forward(nullptr);
        weights[i] = saved;
        const auto difference = (plus - minus) / (2 * epsilon);
        EXPECT_NEAR(difference, gradients[i], 1e-6 * std::max(1.0, std::abs(difference)))
            << "coordinate=" << i;
    }
}
}

TEST_F(NativePP4TrainingIntegrationTest, NativeContinuousGradientMatchesFiniteDifference)
{
    NaviSchedScheduleContext context;
    setupNativeContinuousContext(context, {240, 1488, 3536}, 2);
    const auto before = context.getNativeContinuousInput(0);
    ASSERT_TRUE(before);
    for (double concavity : {0.0, 128.0})
        checkNativeContinuousGradient(context, {.19, .37, .61, .81, .63, .39}, concavity);
    EXPECT_EQ(before, context.getNativeContinuousInput(0));
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeContinuousSharedStepMatchesSubsetExpectation)
{
    NaviSchedScheduleContext context;
    setupNativeContinuousContext(context, {32, 32, 32, 32}, 1, true);
    std::vector<int64_t> tokens(4, 32);
    std::vector<double> inverse(4, 1.0 / 32.0), weights(4, .2);
    std::vector<float> costs(8);
    ASSERT_TRUE(CostSchedulerKernelInternal::fillCostsForNodeRange(context, tokens.data(),
                                                                   0, 1, costs.data()));
    double expected = 0;
    for (unsigned mask = 0; mask < 16; ++mask) {
        double probability = 1.0;
        std::vector<size_t> selected;
        for (size_t i = 0; i < 4; ++i) {
            const bool present = mask & (1u << i);
            probability *= present ? .2 : .8;
            if (present) selected.push_back(i);
        }
        if (!selected.empty())
            expected +=
                probability * context.estimateBatchExecutionLatencyMs(0, selected);
    }
    double first = 0;
    auto evaluate = [&] {
        return CostSchedulerKernelInternal::forwardNodeRange(
            context, costs.data(), weights.data(), tokens.data(), inverse.data(), 128.0,
            1.0, 0.0, 0, 1, nullptr, &first, nullptr);
    };
    ASSERT_TRUE(std::isfinite(evaluate()));
    EXPECT_NEAR(23.616, expected, 1e-9);
    EXPECT_NEAR(expected, first, 1e-9);
    weights = {1, 0, 1, 0};
    ASSERT_TRUE(std::isfinite(evaluate()));
    EXPECT_NEAR(40.0, first, 1e-9);
    weights = {0, 0, 0, 0};
    ASSERT_TRUE(std::isfinite(evaluate()));
    EXPECT_DOUBLE_EQ(0.0, first);
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeContinuousEarlierWorkChangesLaterCompletion)
{
    NaviSchedScheduleContext context;
    setupNativeContinuousContext(context, {20016, 21040}, 1);
    checkNativeContinuousGradient(context, {.47, .63}, 0.0, 1.0);
    const auto plan = context.getNativeContinuousInput(0);
    ASSERT_TRUE(plan);
    ASSERT_GT(plan->slots.size(), 6u);
    bool earlyInTail = false;
    for (size_t i = plan->slots.size() - 6; i < plan->slots.size(); ++i)
        for (const auto &span : plan->costs[i].spans)
            earlyInTail |= span.requestIndex == 0;
    EXPECT_FALSE(earlyInTail);
    int64_t tokens[] = {20016, 21040};
    double inverse[] = {1.0 / 20016, 1.0 / 21040}, weights[] = {.47, .63};
    float costs[4] = {};
    CostSchedulerKernelInternal::NodeGradient gradient;
    ASSERT_TRUE(std::isfinite(CostSchedulerKernelInternal::forwardNodeRange(
        context, costs, weights, tokens, inverse, 41056, 1.0, 0.0, 0, 1, &gradient,
        nullptr, nullptr)));
    ASSERT_EQ(2u, gradient.nativeRequestGradients.size());
    EXPECT_GT(gradient.nativeRequestGradients[0], 0.0);
}

TEST_F(NativePP4TrainingIntegrationTest, NativeContinuousViewRebuildsRequestCoordinates)
{
    NaviSchedScheduleContext context;
    setupNativeContinuousContext(context, {240, 1488, 3536}, 2);
    const auto before = context.getNativeContinuousInput(1);
    ASSERT_TRUE(before);
    ASSERT_TRUE(context.activateScheduleView({2, 0}, {1}));
    const auto cropped = context.getNativeContinuousInput(0);
    ASSERT_TRUE(cropped);
    EXPECT_NE(before, cropped);
    checkNativeContinuousGradient(context, {.31, .59}, 128.0);
    context.clearScheduleView();
    const auto restored = context.getNativeContinuousInput(1);
    ASSERT_TRUE(restored);
    EXPECT_NE(cropped, restored);
    EXPECT_NE(before, restored);
    checkNativeContinuousGradient(context, {.19, .37, .61, .81, .63, .39}, 0.0);
}

TEST_F(NativePP4TrainingIntegrationTest,
       NativeContinuousOriginalOptimizerUsesExactDiscreteCost)
{
    NaviSchedScheduleContext context;
    setupNativeContinuousContext(context, {240, 1488, 3536}, 2);
    std::vector<int64_t> tokens(3);
    const auto total =
        CostSchedulerKernelInternal::fillRequestTokenCounts(context, tokens.data());
    context._scheduleBeginTimeUs = autil::TimeUtility::monotonicTimeUs();
    context._scheduleDeadlineUs = context._scheduleBeginTimeUs + 16000;
    std::vector<float> costs(12);
    ASSERT_TRUE(CostSchedulerKernelInternal::fillCostsForNodeRange(context, tokens.data(),
                                                                   0, 2, costs.data()));
    std::vector<double> weights(6);
    std::vector<uint32_t> selected;
    double result = 0;
    CostSchedulerKernelInternal::ReplayInfo replay;
    replay.init(true, .37, .1, .99, .0001, 50, 16000);
    ASSERT_TRUE(CostSchedulerKernelInternal::optimize(
        context, costs.data(), tokens.data(), total, .37, .1, .99, .0001, 50, 16000,
        replay, 128.0, 42, weights.data(), selected, result));
    ASSERT_EQ(3u, selected.size());
    ASSERT_EQ(1u, replay.optimizationRuns.size());
    EXPECT_GT(replay.optimizationRuns.front().actualLoopCount, 0u);
    std::vector<double> inverse(3);
    CostSchedulerKernelInternal::fillRequestTokenCounts(context, tokens.data(),
                                                        inverse.data());
    const auto denominator =
        CostSchedulerKernelInternal::calculateTokenNormalizationDenominator(tokens.data(),
                                                                            3, 1.0);
    CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
    const auto exact = CostSchedulerKernelInternal::forwardDiscrete(
        context, costs.data(), selected.data(), tokens.data(), inverse.data(),
        denominator, .37, 128.0, workspace);
    EXPECT_DOUBLE_EQ(exact, result);
    EXPECT_EQ(16000, replay.optimizationTimeBudgetUs);
    EXPECT_EQ(context._scheduleBeginTimeUs + 16000, context._scheduleDeadlineUs);
    std::cout << "E258_NATIVE_OPTIMIZER loops="
              << replay.optimizationRuns.front().actualLoopCount
              << " elapsed_us=" << replay.optimizationRuns.front().elapsedUs
              << " exact_loss=" << exact << "\n";
}

TEST_F(NativePP4TrainingIntegrationTest, NativeContinuousCostBenchmark)
{
    if (!std::getenv("NAVI_E258_NATIVE_COST_BENCHMARK")) return;
    for (const auto configuration : std::vector<std::pair<size_t, int64_t>>{
             {1, 32}, {16, 32}, {64, 32}, {16, 8192}}) {
        const auto n = configuration.first;
        const auto prompt = configuration.second;
        for (int sample = 0; sample < 7; ++sample) {
            NaviSchedScheduleContext context;
            setupNativeContinuousContext(context, std::vector<int64_t>(n, prompt), 3);
            std::vector<int64_t> tokens(n);
            std::vector<double> inverse(n);
            const auto total = CostSchedulerKernelInternal::fillRequestTokenCounts(
                context, tokens.data(), inverse.data());
            std::vector<float> costs(n * 3 * 2);
            std::vector<double> weights(n * 3, 1.0 / 3.0), gradients(n * 3);
            std::vector<CostSchedulerKernelInternal::NodeGradient> nodeGradients(3);
            const auto prepStart = autil::TimeUtility::monotonicTimeUs();
            context._scheduleBeginTimeUs = prepStart;
            context._scheduleDeadlineUs = prepStart + 16000;
            ASSERT_TRUE(CostSchedulerKernelInternal::fillCostsForNodeRange(
                context, tokens.data(), 0, 3, costs.data()));
            const auto preparation = autil::TimeUtility::monotonicTimeUs() - prepStart;
            const auto denominator =
                CostSchedulerKernelInternal::calculateTokenNormalizationDenominator(
                    tokens.data(), n, 1.0);
            CostSchedulerKernelInternal::ReplayInfo replay;
            replay.init(true, .37, .1, .99, .0001, 100, 16000);
            std::vector<uint32_t> selected;
            double exact = 0;
            const auto optimizeStart = autil::TimeUtility::monotonicTimeUs();
            ASSERT_TRUE(CostSchedulerKernelInternal::optimize(
                context, costs.data(), tokens.data(), total, .37, .1, .99, .0001, 100,
                16000, replay, 128.0, 42, weights.data(), selected, exact));
            const auto optimizerUs =
                autil::TimeUtility::monotonicTimeUs() - optimizeStart;
            const auto roundUs = autil::TimeUtility::monotonicTimeUs() - prepStart;
            const auto begin = std::chrono::steady_clock::now();
            double sum = 0.0;
            for (int iteration = 0; iteration < 100; ++iteration) {
                sum += CostSchedulerKernelInternal::forwardNodeRange(
                    context, costs.data(), weights.data(), tokens.data(), inverse.data(),
                    denominator, .37, 128.0, 0, 3, nodeGradients.data(), nullptr,
                    nullptr);
                CostSchedulerKernelInternal::backwardNodeRange(
                    context, costs.data(), weights.data(), tokens.data(), inverse.data(),
                    n, 0, 3, nodeGradients.data(), gradients.data());
            }
            const auto stepNs = std::chrono::duration<double, std::nano>(
                                    std::chrono::steady_clock::now() - begin)
                                    .count() /
                                100.0;
            ASSERT_TRUE(std::isfinite(sum));
            if (sample > 0)
                std::cout << "E258_NATIVE_COST_BENCH count=" << n << " prompt=" << prompt
                          << " preparation_us=" << preparation
                          << " forward_backward_ns=" << std::setprecision(17) << stepNs
                          << " optimizer_us=" << optimizerUs << " round_us=" << roundUs
                          << " shared_budget_us=" << replay.optimizationTimeBudgetUs
                          << " loops=" << replay.optimizationRuns.front().actualLoopCount
                          << " exact_loss=" << exact << "\n";
        }
    }
}

TEST_F(NativePP4TrainingIntegrationTest, NativeContinuousKeepsSeparateStepActivation)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    NaviSchedScheduleContext context;
    setupNativeContinuousContext(context, {1040, 1040}, 1, true);
    const auto native = context.getNativeContinuousInput(0);
    ASSERT_TRUE(native);
    ASSERT_EQ(1, native->targetOrdinal);
    ASSERT_EQ(40.0, context.estimateBatchExecutionLatencyMs(0, {0, 1}));
    // Isolate activation on a fixed two-step reference, with no host handoff
    // delays. Actual Native dispatch/FIFO is covered by the Master cases above.
    auto reference = std::make_shared<NaviSchedNativeContinuousInput>(*native);
    for (auto &slot : reference->slots) slot.dispatchTimeMs = 0.0;
    reference->referenceTotalMs = 40.0;
    ASSERT_TRUE(
        CostSchedulerKernelInternal::initializeNativeContinuousReference(*reference, 2));
    ASSERT_DOUBLE_EQ(0.0, reference->fixedLatencyMs);
    std::shared_ptr<const NaviSchedNativeContinuousInput> frozen = reference;
    std::atomic_store(&context._scheduleNodes[0]._nativeContinuousInput, frozen);
    int64_t tokens[] = {1040, 1040};
    double inverse[] = {1.0 / 1040, 1.0 / 1040}, weights[] = {.2, .7};
    float costs[4] = {};
    double first = 0;
    const auto loss = CostSchedulerKernelInternal::forwardNodeRange(
        context, costs, weights, tokens, inverse, 2080.0, 1.0, 0.0, 0, 1, nullptr, &first,
        nullptr);
    ASSERT_TRUE(std::isfinite(loss));
    // Two 10ms steps become 2ms and 7ms. Each of the three PP handoff
    // cycles is bounded by 7ms, then the target's own stage costs 7ms.
    EXPECT_NEAR(3 * 7.0 + 7.0, first, 1e-9);
}

TEST_F(NativePP4TrainingIntegrationTest, NativeMasterFinalForecastUsesRequestArrivalTime)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    for (int64_t elapsedUs : {int64_t(0), int64_t(100000), int64_t(1000000)}) {
        auto snapshot = nativeConsumerSnapshot();
        auto prior = readyNativeRequest(9000, 0);
        prior.promptTokens = 1040;
        setNativeConsumerRows(snapshot, {prior});
        auto transport = std::make_shared<NativeMasterDiagnosticNode>();
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        context.setNodes({transport}, {snapshot});
        context._decisions.resize(1);
        context._selectedNodeIndexes = {0};
        const auto reference = context.getNativeContinuousInput(0);
        ASSERT_TRUE(reference);
        const auto observed = context.getNode(0).getNativePrefillObservation();
        ASSERT_NEAR(40.0, context.estimateBatchExecutionLatencyMs(0, {0}), 1e-9);
        ASSERT_TRUE(context.prepareResult(context._scheduleTimeUs + elapsedUs));
        const double expected = 40.0;
        const auto actual = context._decisions.front().predictedExecutionLatencyMs;
        std::cout << "NATIVE_FINAL_TIME busy=1 elapsed_us=" << elapsedUs
                  << " expected_ms=" << expected << " predicted_ms=" << actual << "\n";
        EXPECT_NEAR(expected, actual, 1e-9);
        // Result preparation must not move the frozen cost inputs to its later
        // clock or mutate the already observed FIFO and unfinished replies.
        EXPECT_NEAR(40.0, context.estimateBatchExecutionLatencyMs(0, {0}), 1e-9);
        EXPECT_EQ(reference, context.getNativeContinuousInput(0));
        EXPECT_EQ(observed, context.getNode(0).getNativePrefillObservation());
        EXPECT_EQ(1u, transport->cacheCalls);
        EXPECT_TRUE(context.prepareResult(context._scheduleTimeUs + elapsedUs + 1000000));
        EXPECT_EQ(actual, context._decisions.front().predictedExecutionLatencyMs);
    }
}

TEST_F(NativePP4TrainingIntegrationTest, NativeFinalDelayDoesNotSubtractUnsubmittedWork)
{
    for (int64_t elapsedUs : {int64_t(0), int64_t(100000), int64_t(1000000)}) {
        auto snapshot = nativeConsumerSnapshot();
        auto transport = std::make_shared<NativeMasterDiagnosticNode>();
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        context.setNodes({transport}, {snapshot});
        context._decisions.resize(1);
        context._selectedNodeIndexes = {0};
        ASSERT_TRUE(context.prepareResult(context._scheduleTimeUs + elapsedUs));
        EXPECT_NEAR(40.0, context._decisions.front().predictedExecutionLatencyMs, 1e-9);
        EXPECT_NEAR(40.0, context.estimateBatchExecutionLatencyMs(0, {0}), 1e-9);
    }
}

TEST_F(NativePP4TrainingIntegrationTest, NativeFinalResultClockBenchmark)
{
    if (!std::getenv("NAVI_E259_FINAL_TIME_BENCHMARK")) return;
    for (size_t count : {size_t(1), size_t(16), size_t(64)}) {
        auto snapshot = nativeConsumerSnapshot();
        auto prior = readyNativeRequest(9000, 0);
        prior.promptTokens = 1040;
        setNativeConsumerRows(snapshot, {prior});
        auto transport = std::make_shared<NativeMasterDiagnosticNode>();
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        while (context._requests.size() < count) {
            auto item = context._requests.front();
            item.request.requestId = context._requests.size() + 1;
            context._requests.push_back(std::move(item));
        }
        context.setNodes({transport}, {snapshot});
        context._decisions.resize(count);
        context._selectedNodeIndexes.assign(count, 0);
        for (int sample = 0; sample < 13; ++sample) {
            const auto order = sample % 2 ? std::vector<int64_t>{0, 1000000}
                                          : std::vector<int64_t>{1000000, 0};
            for (auto delay : order) {
                const auto begin = std::chrono::steady_clock::now();
                double sum = 0;
                for (int iteration = 0; iteration < 1000; ++iteration) {
                    context._resultPrepared = false;
                    ASSERT_TRUE(context.prepareResult(context._scheduleTimeUs + delay));
                    sum += context._decisions.front().predictedExecutionLatencyMs;
                }
                const auto ns = std::chrono::duration<double, std::nano>(
                                    std::chrono::steady_clock::now() - begin)
                                    .count() /
                                1000.0;
                ASSERT_TRUE(std::isfinite(sum));
                ASSERT_GT(sum, 0.0);
                if (sample > 0)
                    std::cout << "E259_NATIVE_FINAL_BENCH count=" << count
                              << " delay_us=" << delay << " ns=" << std::setprecision(17)
                              << ns << "\n";
            }
        }
        EXPECT_EQ(1u, transport->cacheCalls);
    }
}

TEST_F(NativePP4TrainingIntegrationTest, NativeFinalAdmissionAuditUsesActualPPResult)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    for (int64_t elapsedUs : {int64_t(0), int64_t(100000)}) {
        auto snapshot = nativeConsumerSnapshot();
        auto prior = readyNativeRequest(9000, 0);
        prior.promptTokens = 1040;
        setNativeConsumerRows(snapshot, {prior});
        auto transport = std::make_shared<NativeMasterDiagnosticNode>();
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        context.setNodes({transport}, {snapshot});
        context._decisions.resize(1);
        context._selectedNodeIndexes = {0};
        ASSERT_TRUE(context.prepareResult(context._scheduleTimeUs + elapsedUs));
        const double expected = 40.0;
        ASSERT_DOUBLE_EQ(expected,
                         context._decisions.front().predictedExecutionLatencyMs);
        const auto audit =
            MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0, true);
        std::cout << "NATIVE_FINAL_AUDIT elapsed_us=" << elapsedUs
                  << " expected_ms=" << expected << " valid=" << audit.valid << "\n";
        EXPECT_TRUE(audit.valid);
        EXPECT_DOUBLE_EQ(expected, audit.baseExecutionMs);
        EXPECT_DOUBLE_EQ(expected, audit.finalExecutionMs);
        EXPECT_DOUBLE_EQ(0.0, audit.correctionMs);
        EXPECT_EQ(1u, transport->cacheCalls);
    }
}

TEST_F(NativePP4TrainingIntegrationTest, NativeAuditFreezesInputsAndDoesNotRequery)
{
    auto snapshot = nativeConsumerSnapshot();
    auto transport = std::make_shared<NativeMasterDiagnosticNode>();
    NaviSchedScheduleContext context;
    context._scheduleTimeUs = 1000;
    context._requests = nativeMasterDiagnosticRequests();
    context.setNodes({transport}, {snapshot});
    context._decisions.resize(1);
    context._selectedNodeIndexes = {0};
    ASSERT_TRUE(context.prepareResult(101000));
    const auto audit =
        MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0, true);
    ASSERT_TRUE(audit.valid);
    ASSERT_TRUE(audit.nativePrediction);
    ASSERT_EQ(1u, audit.frozenExecutionInputs.size());
    const auto frozen = audit.nativePrediction;
    EXPECT_DOUBLE_EQ(40.0, frozen->baseExecutionMs);
    EXPECT_DOUBLE_EQ(40.0, frozen->recomputeBaseLatencyMs(frozen->parameters));
    auto weights = frozen->parameters;
    weights[1] = 20.0;
    EXPECT_DOUBLE_EQ(80.0, frozen->recomputeBaseLatencyMs(weights));
    EXPECT_DOUBLE_EQ(100.0, frozen->arrivalDelayMs);
    ASSERT_EQ(1u, frozen->input.requests.size());
    EXPECT_DOUBLE_EQ(100.0, frozen->input.requests.front().readyTimeMs);
    EXPECT_EQ(16, frozen->input.requests.front().contextTokens);
    EXPECT_EQ(16, frozen->input.requests.front().remainingTokens);
    // A cost query uses the original scheduling clock and does not replace the
    // final result snapshot. Native and wire-level legacy features stay separate.
    EXPECT_DOUBLE_EQ(40.0, context.estimateBatchExecutionLatencyMs(0, {0}));
    EXPECT_FALSE(context._decisions.front().latencyPrediction);
    const auto again =
        MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0);
    EXPECT_EQ(frozen, again.nativePrediction);
    EXPECT_DOUBLE_EQ(40.0, again.finalExecutionMs);
    EXPECT_EQ(1u, transport->cacheCalls);
    context._queuePredictionTimeUs += 1;
    EXPECT_FALSE(
        MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0).valid);
    context._queuePredictionTimeUs -= 1;
    context.setNodes({transport}, {snapshot});
    EXPECT_FALSE(
        MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0).valid);
    EXPECT_DOUBLE_EQ(40.0, frozen->recomputeBaseLatencyMs(frozen->parameters));
}

TEST_F(NativePP4TrainingIntegrationTest, NativeFinalAuditBenchmark)
{
    if (!std::getenv("NAVI_E261_FINAL_AUDIT_BENCHMARK")) return;
    for (size_t count : {size_t(1), size_t(16), size_t(64)}) {
        auto snapshot = nativeConsumerSnapshot();
        auto transport = std::make_shared<NativeMasterDiagnosticNode>();
        NaviSchedScheduleContext context;
        context._scheduleTimeUs = 1000;
        context._requests = nativeMasterDiagnosticRequests();
        while (context._requests.size() < count) {
            auto request = context._requests.front();
            request.request.requestId = context._requests.size() + 1;
            context._requests.push_back(request);
        }
        context.setNodes({transport}, {snapshot});
        context._decisions.resize(count);
        context._selectedNodeIndexes.assign(count, 0);
        ASSERT_TRUE(context.prepareResult(101000));
        for (int sample = 0; sample < 13; ++sample) {
            double sum = 0.0;
            const auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < 1000; ++iteration) {
                const auto audit =
                    MasterScheduleResultKernel::getAdmissionPredictionAudit(context, 0,
                                                                            false);
                ASSERT_TRUE(audit.valid);
                sum += audit.finalExecutionMs;
            }
            const double ns = std::chrono::duration<double, std::nano>(
                                  std::chrono::steady_clock::now() - start)
                                  .count() /
                              1000;
            ASSERT_DOUBLE_EQ(40000.0, sum);
            if (sample > 0)
                std::cout << "E261_NATIVE_AUDIT_BENCH count=" << count
                          << " ns=" << std::setprecision(17) << ns << "\n";
        }
        EXPECT_EQ(1u, transport->cacheCalls);
    }
}

TEST_F(NativePP4TrainingIntegrationTest,
       QueryServicePreservesFirstPollClockForObservedAndUnreportedReservations)
{
    for (bool registered : {false, true}) {
        auto snapshot = nativeConsumerSnapshot();
        snapshot.prefillExecutionPolicy->asyncKVLookupEnabled = true;
        snapshot.prefillExecutionPolicy->asyncKVLookupHashBlockSize = 16;
        auto waiting = readyNativeRequest(3, 0);
        waiting.phase = NativePPRequestPhase::QUERY_NOT_STARTED;
        waiting.cacheHitTokens = -1;
        setNativeConsumerRows(snapshot, registered
            ? std::vector<NaviSchedNativePPRequest>{waiting}
            : std::vector<NaviSchedNativePPRequest>{});
        snapshot.nativePrefillReservations = {
            {nativeReservedRequest(3), registered
                ? MasterBatchReservation::RequestState::DISPATCHED
                : MasterBatchReservation::RequestState::WAITING}};
        NaviSchedScheduleNode node(std::make_shared<NativeMasterDiagnosticNode>(),
                                   snapshot, 1000);
        TPUPrefillPP4ObservationPredictions predictions;
        auto &prediction = predictions.requests[3];
        prediction.cacheHitTokens = 0;
        prediction.readyAfterObservationMs = 7.0;
        prediction.needsFirstQuery = true;
        prediction.firstQueryServiceMs = 25.0;
        auto input = node.prepareNativePrefillObservation(predictions);
        ASSERT_EQ(size_t(1), input.requests.size());
        ASSERT_TRUE(input.requests[0].firstQueryServiceMs.has_value());
        ASSERT_DOUBLE_EQ(25.0, *input.requests[0].firstQueryServiceMs);
        input.requests[0].owned = true;
        const auto output = TPUPrefillPP4Forecast::predict(
            input.state, input.requests, *snapshot.prefillExecutionPolicy,
            {0., 10., 0., 0., 0., 0., 0., 0., 0.}, 1000, true);
        ASSERT_GE(output.firstQueryTimeMs.at(3), input.state.nowMs + 7.0);
        ASSERT_DOUBLE_EQ(25.0, output.firstQueryReadyTimeMs.at(3) -
                                  output.firstQueryTimeMs.at(3));
    }
}

} // namespace navi_sched
