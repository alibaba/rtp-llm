#include "aios/navi_sched/resource/AgentHistoryR.h"
#include "aios/navi_sched/packaging/dashllm_r50/ExecutionCacheSourceStateMigration.h"
#include "autil/legacy/jsonizable.h"
#include <cstdlib>
#include "aios/navi_sched/resource/MasterLatencyModelR.h"
#include "aios/navi_sched/resource/LatencyResidualModel.h"
#include <sys/wait.h>
#include <unistd.h>

#include "aios/navi_sched/resource/LatencyModelR.h"
#include "autil/CRC32C.h"
#include "autil/DataBuffer.h"
#include "autil/TimeUtility.h"
#include "navi/tester/NaviResourceHelper.h"
#include "unittest/unittest.h"
#include <filesystem>
#include <fstream>

namespace navi_sched {
namespace {

std::string makeHistoryConfig(const std::string &stateFile,
                              const std::string &formulaFingerprint,
                              const std::string &historyFingerprint,
                              const std::string &binaryFingerprint)
{
    return std::string(R"({"state_file":")") + stateFile +
           R"(","formula_fingerprint":")" + formulaFingerprint +
           R"(","history_fingerprint":")" + historyFingerprint +
           R"(","binary_fingerprint":")" + binaryFingerprint + R"("})";
}

AgentHistoryRPtr makeHistory(const std::string &stateFile,
                             const std::string &formulaFingerprint,
                             const std::string &historyFingerprint,
                             const std::string &binaryFingerprint)
{
    navi::NaviResourceHelper helper;
    if (!helper.config(AgentHistoryR::RESOURCE_ID,
                       makeHistoryConfig(stateFile, formulaFingerprint,
                                         historyFingerprint, binaryFingerprint))) {
        return nullptr;
    }
    return helper.getOrCreateResPtr<AgentHistoryR>();
}

std::vector<double> makeNegativeWorkloadIncidentParameters()
{
    return {
        16.3819395199594,
        -10.3540165663599,
        -12.6186200001777,
        1.35509418410749,
        38.8582262144234,
        1.08282721097769,
        0.0,
        187.050116825021,
        0.000378280769895689,
        0.00658702178106546,
        8.1547032780751,
        7.98917265282236,
        0.022083145192398,
        -8.93681084000853,
        0.0156816955369439,
        141.473652382685,
        1.38375202315852,
        0.0,
        0.0,
    };
}

NaviSchedLatencyFeatures makeNegativeWorkloadIncidentFeatures()
{
    NaviSchedLatencyFeatures features;
    EXPECT_TRUE(features.appendCapacitySplitAggregate(
        424, 0, 0, 0, 0.0, 0.0, 541, 2'078'064, 40'022, 14'084'028.0, 153'692'768.0));
    EXPECT_TRUE(features.appendPrecedingRequestCount(792));
    return features;
}

void recordMatchingSamples(const LatencyModelRPtr &latencyModel,
                           const std::vector<double> &parameters, size_t count,
                           std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updated,
                           bool &saveNeeded, int64_t firstEventTimeUs = 1)
{
    for (size_t index = 0; index < count; ++index) {
        NaviSchedLatencySample sample;
        sample.features.appendRequest(0, 1024 + index);
        sample.latencyMs = latencyModel->predict(sample.features, parameters);
        sample.eventTimeUs = firstEventTimeUs + index;
        ASSERT_TRUE(latencyModel->recordSample(sample, updated, saveNeeded));
    }
}

struct SavedState {
    std::string stateFile;
    std::vector<double> parameters;
    NonLinearPrefillModelR::OptimizerState optimizer;
    std::vector<double> residualParameters;
    int64_t residualLastUpdateTimeUs = 0;
};

SavedState saveState(const std::string &tempDataPath, const std::string &name)
{
    SavedState result;
    result.stateFile = tempDataPath + "latency-history-state/" + name;
    std::filesystem::remove(result.stateFile);
    auto history = makeHistory(result.stateFile, "formula-a", "history-a", "binary-a");
    EXPECT_TRUE(history);
    NonLinearPrefillModelR model;
    result.parameters = model.getInitialParameters();
    result.parameters[NonLinearPrefillModelR::BASE_LATENCY_PARAMETER_INDEX] = 7.0;
    for (size_t index = 0; index < result.parameters.size(); ++index) {
        result.optimizer.adamMoment1[index] = index + 0.25;
        result.optimizer.adamMoment2[index] = index + 1.5;
    }
    result.optimizer.step = 73;
    EXPECT_TRUE(model.restoreOptimizerState(result.optimizer));
    NaviSchedLatencySample sample;
    sample.features.appendRequest(11, 13);
    sample.latencyMs = 17.0;
    sample.eventTimeUs = 19;
    bool fitReady = false;
    bool persistReady = false;
    EXPECT_TRUE(history->append(sample, fitReady, persistReady));
    auto snapshot = std::make_shared<NaviSchedLatencyModelSnapshot>();
    snapshot->formula = model.getFormula();
    snapshot->binaryVersion = model.getBinaryVersion();
    snapshot->revision = 211;
    snapshot->parameters = result.parameters;
    result.residualParameters = LatencyResidualModel::makeInitialParameters();
    result.residualParameters[3] = 37.5;
    result.residualLastUpdateTimeUs = autil::TimeUtility::currentTime();
    snapshot->residualParameters = result.residualParameters;
    snapshot->residualLastUpdateTimeUs = result.residualLastUpdateTimeUs;
    snapshot->overloadPredictionTrusted = true;
    EXPECT_TRUE(history->save(snapshot, model.getOptimizerState()));
    return result;
}

SavedState saveControlledFormulaState(const std::string &tempDataPath,
                                      const std::string &name,
                                      int64_t binaryVersionOffset = -1,
                                      int workloadIndex = -1, double workloadValue = -1.0)
{
    SavedState result;
    result.stateFile = tempDataPath + "latency-history-state/" + name;
    std::filesystem::create_directories(
        std::filesystem::path(result.stateFile).parent_path());
    navi::NaviResourceHelper helper;
    EXPECT_TRUE(helper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        workloadIndex >= 10 ? R"({"use_capacity_split_feature":true})" : "{}"));
    auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
    EXPECT_TRUE(model);
    result.parameters = model->getInitialParameters();
    result.parameters[NonLinearPrefillModelR::BASE_LATENCY_PARAMETER_INDEX] = 7.0;
    if (workloadIndex >= 0) {
        result.parameters.at(workloadIndex) = workloadValue;
    }
    for (size_t index = 0; index < result.parameters.size(); ++index) {
        result.optimizer.adamMoment1[index] = index + 0.25;
        result.optimizer.adamMoment2[index] = index + 1.5;
    }
    result.optimizer.step = 73;
    result.residualParameters = LatencyResidualModel::makeInitialParameters();
    result.residualParameters[3] = 37.5;
    result.residualLastUpdateTimeUs = autil::TimeUtility::currentTime();
    LatencyHistorySampler sampler(43);
    NaviSchedLatencySample sample;
    sample.features.appendRequest(11, 13);
    sample.latencyMs = 17.0;
    sample.eventTimeUs = 19;
    sampler.append(sample);

    const auto writePayload = [&](autil::DataBuffer &buffer) {
        buffer.write(std::string("formula-a"));
        buffer.write(std::string("history-a"));
        buffer.write(std::string("binary-a"));
        buffer.write(model->getFormula());
        buffer.write(model->getBinaryVersion() + binaryVersionOffset);
        buffer.write(int64_t(211));
        buffer.write(true);
        buffer.write(false);
        buffer.write(result.parameters);
        buffer.write(std::vector<double>(
            result.optimizer.adamMoment1.begin(),
            result.optimizer.adamMoment1.begin() + result.parameters.size()));
        buffer.write(std::vector<double>(
            result.optimizer.adamMoment2.begin(),
            result.optimizer.adamMoment2.begin() + result.parameters.size()));
        buffer.write(result.optimizer.step);
        buffer.write(LatencyResidualModel::VERSION);
        buffer.write(result.residualParameters);
        buffer.write(result.residualLastUpdateTimeUs);
        sampler.serialize(buffer);
    };

    autil::DataBuffer buffer;
    buffer.write(std::string("navi_sched_latency_state"));
    buffer.write(int32_t(6));
    autil::DataBuffer payload;
    writePayload(payload);
    const std::string payloadString(payload.getData(), payload.getDataLen());
    const auto checksum = autil::CRC32C::Mask(
        autil::CRC32C::Value(payloadString.data(), payloadString.size()));
    buffer.write(checksum);
    buffer.write(payloadString);
    std::ofstream output(result.stateFile, std::ios::binary | std::ios::trunc);
    EXPECT_TRUE(output.write(buffer.getData(), buffer.getDataLen()));
    return result;
}

SavedState saveInvalidResidualState(const std::string &tempDataPath,
                                    const std::string &name)
{
    SavedState result;
    result.stateFile = tempDataPath + "latency-history-state/" + name;
    std::filesystem::create_directories(
        std::filesystem::path(result.stateFile).parent_path());
    NonLinearPrefillModelR model;
    result.parameters = model.getInitialParameters();
    result.parameters[NonLinearPrefillModelR::BASE_LATENCY_PARAMETER_INDEX] = 7.0;
    result.optimizer.step = 73;
    result.residualParameters = LatencyResidualModel::makeInitialParameters();
    result.residualParameters[3] = 37.5;
    result.residualLastUpdateTimeUs = autil::TimeUtility::currentTime();

    LatencyHistorySampler sampler(43);
    NaviSchedLatencySample sample;
    sample.features.appendRequest(11, 13);
    sample.latencyMs = 17.0;
    sample.eventTimeUs = 19;
    sampler.append(sample);

    autil::DataBuffer payload;
    payload.write(std::string("formula-a"));
    payload.write(std::string("history-a"));
    payload.write(std::string("binary-a"));
    payload.write(model.getFormula());
    payload.write(model.getBinaryVersion());
    payload.write(int64_t(211));
    payload.write(true);
    payload.write(false);
    payload.write(result.parameters);
    payload.write(std::vector<double>(result.parameters.size(), 0.0));
    payload.write(std::vector<double>(result.parameters.size(), 0.0));
    payload.write(result.optimizer.step);
    payload.write(LatencyResidualModel::VERSION - 1);
    payload.write(result.residualParameters);
    payload.write(result.residualLastUpdateTimeUs);
    sampler.serialize(payload);
    const std::string payloadString(payload.getData(), payload.getDataLen());
    const auto checksum = autil::CRC32C::Mask(
        autil::CRC32C::Value(payloadString.data(), payloadString.size()));

    autil::DataBuffer buffer;
    buffer.write(std::string("navi_sched_latency_state"));
    buffer.write(int32_t(6));
    buffer.write(checksum);
    buffer.write(payloadString);
    std::ofstream output(result.stateFile, std::ios::binary | std::ios::trunc);
    EXPECT_TRUE(output.write(buffer.getData(), buffer.getDataLen()));
    return result;
}

} // namespace

class AgentHistoryStateTest : public TESTBASE {};

TEST_F(AgentHistoryStateTest, testExactFingerprintsRestoreEverything)
{
    const auto saved = saveState(GET_TEMP_DATA_PATH(), "exact.state");
    auto history = makeHistory(saved.stateFile, "formula-a", "history-a", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(history->load(model, snapshot, fitOnStart));
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(saved.parameters, snapshot->parameters);
    EXPECT_EQ(model.getBinaryVersion(), snapshot->binaryVersion);
    EXPECT_EQ(211, snapshot->revision);
    EXPECT_TRUE(snapshot->overloadPredictionTrusted);
    EXPECT_EQ(1, history->getHistory()->size());
    EXPECT_EQ(saved.optimizer.adamMoment1, model.getOptimizerState().adamMoment1);
    EXPECT_EQ(saved.optimizer.adamMoment2, model.getOptimizerState().adamMoment2);
    EXPECT_EQ(saved.optimizer.step, model.getOptimizerState().step);
    EXPECT_EQ(saved.residualParameters, snapshot->residualParameters);
    EXPECT_EQ(saved.residualLastUpdateTimeUs, snapshot->residualLastUpdateTimeUs);
    EXPECT_FALSE(history->requiresParameterValidation());
}

TEST_F(AgentHistoryStateTest,
       testPersistedUnsafeWorkloadStateRecoversAndStaysSafeAfterRestart)
{
    const auto stateFile =
        GET_TEMP_DATA_PATH() + "latency-history-state/unsafe-workload.state";
    std::filesystem::remove(stateFile);
    const auto historyConfig =
        std::string(R"({"state_file":")") + stateFile +
        R"(","formula_fingerprint":"formula-a","history_fingerprint":"history-a","binary_fingerprint":"binary-a","fit_batch_size":1,"fit_interval":1,"persist_interval":1,"warmup_sample_count":0})";
    const auto modelConfig =
        R"({"use_batch_count_feature":true,"use_capacity_split_feature":true})";
    const auto latencyConfig =
        R"({"latency_model_fingerprint":"formula-a","trust_min_revision":2})";
    const auto incidentParameters = makeNegativeWorkloadIncidentParameters();
    const auto incidentFeatures = makeNegativeWorkloadIncidentFeatures();

    {
        navi::NaviResourceHelper helper;
        ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID, historyConfig));
        ASSERT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID, modelConfig));
        auto history = helper.getOrCreateResPtr<AgentHistoryR>();
        auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
        ASSERT_TRUE(history);
        ASSERT_TRUE(model);

        NonLinearPrefillModelR::OptimizerState dirtyOptimizer;
        dirtyOptimizer.step = 17;
        dirtyOptimizer.adamMoment1[13] = 2.0;
        dirtyOptimizer.adamMoment2[13] = 3.0;
        ASSERT_TRUE(model->restoreOptimizerState(dirtyOptimizer));
        NaviSchedLatencySample sample;
        sample.features = incidentFeatures;
        sample.latencyMs = 240'000.0;
        sample.eventTimeUs = 1;
        bool fitReady = false;
        bool persistReady = false;
        ASSERT_TRUE(history->append(sample, fitReady, persistReady));

        auto snapshot = std::make_shared<NaviSchedLatencyModelSnapshot>();
        snapshot->formula = model->getFormula();
        snapshot->binaryVersion = model->getBinaryVersion();
        snapshot->revision = 211;
        snapshot->parameters = incidentParameters;
        snapshot->optimizerState = dirtyOptimizer;
        snapshot->residualParameters = LatencyResidualModel::makeInitialParameters();
        snapshot->residualLastUpdateTimeUs = autil::TimeUtility::currentTime();
        snapshot->overloadPredictionTrusted = true;
        EXPECT_FALSE(history->save(snapshot, dirtyOptimizer));
    }

    const auto invalidState = saveControlledFormulaState(
        GET_TEMP_DATA_PATH(), "unsafe-workload.state", 0, 13, incidentParameters[13]);
    ASSERT_EQ(stateFile, invalidState.stateFile);
    std::vector<double> expectedRecoveredParameters;
    NonLinearPrefillModelR::OptimizerState expectedRecoveredOptimizer;
    {
        navi::NaviResourceHelper helper;
        ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID, historyConfig));
        ASSERT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID, modelConfig));
        ASSERT_TRUE(helper.config(LatencyModelR::RESOURCE_ID, latencyConfig));
        auto latencyModel = helper.getOrCreateResPtr<LatencyModelR>();
        auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
        ASSERT_TRUE(latencyModel);
        ASSERT_TRUE(model);
        const auto initialParameters = model->getInitialParameters();

        auto restored = latencyModel->getSnapshot();
        ASSERT_TRUE(restored);
        EXPECT_EQ(initialParameters, restored->parameters);
        EXPECT_EQ(1, model->getOptimizerState().step);
        EXPECT_DOUBLE_EQ(0.0, model->getOptimizerState().adamMoment1[13]);
        EXPECT_DOUBLE_EQ(0.0, model->getOptimizerState().adamMoment2[13]);
        EXPECT_FALSE(restored->overloadPredictionTrusted);

        NaviSchedLatencySample sample;
        sample.features = incidentFeatures;
        sample.latencyMs = 240'000.0;
        sample.eventTimeUs = 2;
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
        bool saveNeeded = false;
        ASSERT_TRUE(latencyModel->recordSample(sample, updated, saveNeeded,
                                               NaviSchedLatencyTrainingPhase::NORMAL));
        ASSERT_TRUE(updated);
        ASSERT_TRUE(NonLinearPrefillModelR::validateParameters(updated->parameters));
        EXPECT_NE(initialParameters, updated->parameters);
        expectedRecoveredParameters = updated->parameters;
        expectedRecoveredOptimizer = updated->optimizerState;
        EXPECT_TRUE(updated->overloadPredictionTrusted);
        EXPECT_EQ(2, updated->optimizerState.step);
        EXPECT_TRUE(saveNeeded);
        ASSERT_TRUE(latencyModel->saveState());
    }

    {
        navi::NaviResourceHelper helper;
        ASSERT_TRUE(helper.config(AgentHistoryR::RESOURCE_ID, historyConfig));
        ASSERT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID, modelConfig));
        ASSERT_TRUE(helper.config(LatencyModelR::RESOURCE_ID, latencyConfig));
        auto latencyModel = helper.getOrCreateResPtr<LatencyModelR>();
        auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
        ASSERT_TRUE(latencyModel);
        ASSERT_TRUE(model);
        auto restored = latencyModel->getSnapshot();
        ASSERT_TRUE(restored);
        EXPECT_EQ(expectedRecoveredParameters, restored->parameters);
        EXPECT_EQ(expectedRecoveredOptimizer.step, model->getOptimizerState().step);
        EXPECT_EQ(expectedRecoveredOptimizer.adamMoment1,
                  model->getOptimizerState().adamMoment1);
        EXPECT_EQ(expectedRecoveredOptimizer.adamMoment2,
                  model->getOptimizerState().adamMoment2);
    }
}

TEST_F(AgentHistoryStateTest, testInvalidResidualStateOnlyResetsResidual)
{
    const auto saved =
        saveInvalidResidualState(GET_TEMP_DATA_PATH(), "invalid-residual.state");
    auto history = makeHistory(saved.stateFile, "formula-a", "history-a", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = false;
    ASSERT_TRUE(history->load(model, snapshot, fitOnStart));
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(saved.parameters, snapshot->parameters);
    EXPECT_EQ(211, snapshot->revision);
    EXPECT_TRUE(snapshot->overloadPredictionTrusted);
    EXPECT_EQ(1, history->getHistory()->size());
    EXPECT_EQ(saved.optimizer.step, model.getOptimizerState().step);
    EXPECT_EQ(LatencyResidualModel::VERSION, snapshot->residualVersion);
    EXPECT_EQ(LatencyResidualModel::makeInitialParameters(),
              snapshot->residualParameters);
    EXPECT_EQ(0, snapshot->residualLastUpdateTimeUs);
    EXPECT_FALSE(fitOnStart);
}

TEST_F(AgentHistoryStateTest, testUntrustedParametersAreNotRestored)
{
    const auto stateFile = GET_TEMP_DATA_PATH() + "latency-history-state/untrusted.state";
    std::filesystem::remove(stateFile);
    auto history = makeHistory(stateFile, "formula-a", "history-a", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    auto parameters = model.getInitialParameters();
    parameters[NonLinearPrefillModelR::BASE_LATENCY_PARAMETER_INDEX] = 17.0;
    auto saved = std::make_shared<NaviSchedLatencyModelSnapshot>();
    saved->formula = model.getFormula();
    saved->binaryVersion = model.getBinaryVersion();
    saved->revision = 100;
    saved->parameters = parameters;
    saved->overloadPredictionTrusted = false;
    ASSERT_TRUE(history->save(saved, model.getOptimizerState()));

    auto loadedHistory = makeHistory(stateFile, "formula-a", "history-a", "binary-a");
    ASSERT_TRUE(loadedHistory);
    NonLinearPrefillModelR loadedModel;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> loaded;
    bool fitOnStart = true;
    ASSERT_TRUE(loadedHistory->load(loadedModel, loaded, fitOnStart));
    EXPECT_FALSE(loaded);
    EXPECT_TRUE(loadedHistory->getHistory()->empty());
    EXPECT_EQ(1, loadedModel.getOptimizerState().step);
    EXPECT_FALSE(fitOnStart);
}

TEST_F(AgentHistoryStateTest, testStateBitFlipDropsEverything)
{
    const auto saved = saveState(GET_TEMP_DATA_PATH(), "bit-flip.state");
    std::ifstream input(saved.stateFile, std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(input)),
                        std::istreambuf_iterator<char>());
    ASSERT_GT(content.size(), 16);
    content.back() ^= 1;
    std::ofstream output(saved.stateFile, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output.write(content.data(), content.size()));
    output.close();

    auto history = makeHistory(saved.stateFile, "formula-a", "history-a", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    const auto optimizerBefore = model.getOptimizerState();
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(history->load(model, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(history->getHistory()->empty());
    EXPECT_FALSE(history->requiresParameterValidation());
    EXPECT_EQ(optimizerBefore.step, model.getOptimizerState().step);
    EXPECT_FALSE(fitOnStart);
}

TEST_F(AgentHistoryStateTest, testTruncatedStateDropsEverything)
{
    const auto saved = saveState(GET_TEMP_DATA_PATH(), "truncated.state");
    std::ifstream input(saved.stateFile, std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(input)),
                        std::istreambuf_iterator<char>());
    ASSERT_GT(content.size(), 16);
    content.resize(content.size() - 3);
    std::ofstream output(saved.stateFile, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output.write(content.data(), content.size()));
    output.close();

    auto history = makeHistory(saved.stateFile, "formula-a", "history-a", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    const auto optimizerBefore = model.getOptimizerState();
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(history->load(model, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(history->getHistory()->empty());
    EXPECT_FALSE(history->requiresParameterValidation());
    EXPECT_EQ(optimizerBefore.step, model.getOptimizerState().step);
    EXPECT_FALSE(fitOnStart);
}

TEST_F(AgentHistoryStateTest, testHistoryFingerprintChangeKeepsUntrustedParameters)
{
    const auto saved = saveState(GET_TEMP_DATA_PATH(), "history-change.state");
    auto history = makeHistory(saved.stateFile, "formula-a", "history-b", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(history->load(model, snapshot, fitOnStart));
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(saved.parameters, snapshot->parameters);
    EXPECT_TRUE(snapshot->overloadPredictionTrusted);
    EXPECT_FALSE(snapshot->warmStarted);
    EXPECT_TRUE(history->getHistory()->empty());
    EXPECT_EQ(1, model.getOptimizerState().step);
    EXPECT_TRUE(history->requiresParameterValidation());
}

TEST_F(AgentHistoryStateTest, testBinaryChangeKeepsUntrustedParameters)
{
    const auto saved = saveState(GET_TEMP_DATA_PATH(), "binary-change.state");
    auto history = makeHistory(saved.stateFile, "formula-a", "history-a", "binary-b");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(history->load(model, snapshot, fitOnStart));
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(saved.parameters, snapshot->parameters);
    EXPECT_TRUE(history->getHistory()->empty());
    EXPECT_EQ(1, model.getOptimizerState().step);
    EXPECT_TRUE(history->requiresParameterValidation());
}

TEST_F(AgentHistoryStateTest, testFormulaBinaryChangeDropsEverything)
{
    const auto saved =
        saveControlledFormulaState(GET_TEMP_DATA_PATH(), "formula-binary-change.state");
    auto history = makeHistory(saved.stateFile, "formula-a", "history-a", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    const auto optimizerBefore = model.getOptimizerState();
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(history->load(model, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(history->getHistory()->empty());
    EXPECT_EQ(optimizerBefore.adamMoment1, model.getOptimizerState().adamMoment1);
    EXPECT_EQ(optimizerBefore.adamMoment2, model.getOptimizerState().adamMoment2);
    EXPECT_EQ(optimizerBefore.step, model.getOptimizerState().step);
    EXPECT_FALSE(history->requiresParameterValidation());
    EXPECT_FALSE(fitOnStart);
}

TEST_F(AgentHistoryStateTest, testNegativeWorkloadStateRejectedWithZeroBoundaryControl)
{
    for (const auto index : {5, 13}) {
        for (const auto value : {-1e-12, 0.0}) {
            SCOPED_TRACE(index);
            SCOPED_TRACE(value);
            const auto saved = saveControlledFormulaState(
                GET_TEMP_DATA_PATH(),
                "workload-" + std::to_string(index) +
                    (value < 0.0 ? "-negative.state" : "-zero.state"),
                0, index, value);
            navi::NaviResourceHelper helper;
            ASSERT_TRUE(helper.config(
                NonLinearPrefillModelR::RESOURCE_ID,
                index >= 10 ? R"({"use_capacity_split_feature":true})" : "{}"));
            auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
            ASSERT_TRUE(model);
            const auto optimizerBefore = model->getOptimizerState();
            auto history =
                makeHistory(saved.stateFile, "formula-a", "history-a", "binary-a");
            ASSERT_TRUE(history);
            std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
            bool fitOnStart = true;
            ASSERT_TRUE(history->load(*model, snapshot, fitOnStart));
            EXPECT_FALSE(fitOnStart);
            EXPECT_FALSE(history->requiresParameterValidation());
            const auto optimizerAfter = model->getOptimizerState();
            if (value < 0.0) {
                EXPECT_FALSE(snapshot);
                EXPECT_TRUE(history->getHistory()->empty());
                EXPECT_EQ(optimizerBefore.step, optimizerAfter.step);
                EXPECT_EQ(optimizerBefore.adamMoment1, optimizerAfter.adamMoment1);
                EXPECT_EQ(optimizerBefore.adamMoment2, optimizerAfter.adamMoment2);
            } else {
                ASSERT_TRUE(snapshot);
                EXPECT_EQ(saved.parameters, snapshot->parameters);
                EXPECT_EQ(211, snapshot->revision);
                EXPECT_TRUE(snapshot->overloadPredictionTrusted);
                EXPECT_FALSE(snapshot->warmStarted);
                EXPECT_EQ(1u, history->getHistory()->size());
                EXPECT_EQ(saved.optimizer.step, optimizerAfter.step);
                EXPECT_EQ(saved.optimizer.adamMoment1, optimizerAfter.adamMoment1);
                EXPECT_EQ(saved.optimizer.adamMoment2, optimizerAfter.adamMoment2);
            }
        }
    }
}

TEST_F(AgentHistoryStateTest, testFormulaFingerprintChangeDropsEverything)
{
    const auto saved = saveState(GET_TEMP_DATA_PATH(), "formula-change.state");
    auto history = makeHistory(saved.stateFile, "formula-b", "history-a", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(history->load(model, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(history->getHistory()->empty());
    EXPECT_EQ(1, model.getOptimizerState().step);
    EXPECT_FALSE(history->requiresParameterValidation());
}

TEST_F(AgentHistoryStateTest, testRestoredParametersStayCandidateUntilValidated)
{
    const auto saved = saveState(GET_TEMP_DATA_PATH(), "candidate-selected.state");
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.config(
        AgentHistoryR::RESOURCE_ID,
        makeHistoryConfig(saved.stateFile, "formula-a", "history-b", "binary-a")));
    ASSERT_TRUE(helper.config(
        LatencyModelR::RESOURCE_ID,
        R"({"latency_model_fingerprint":"formula-a","trust_min_revision":1024})"));
    auto latencyModel = helper.getOrCreateResPtr<LatencyModelR>();
    ASSERT_TRUE(latencyModel);

    auto initial = latencyModel->getSnapshot();
    ASSERT_TRUE(initial);
    EXPECT_NE(saved.parameters, initial->parameters);
    EXPECT_FALSE(initial->overloadPredictionTrusted);

    std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
    bool saveNeeded = false;
    recordMatchingSamples(latencyModel, saved.parameters, 31, updated, saveNeeded);
    EXPECT_FALSE(updated);
    EXPECT_EQ(initial->parameters, latencyModel->getSnapshot()->parameters);

    recordMatchingSamples(latencyModel, saved.parameters, 1, updated, saveNeeded, 32);
    ASSERT_TRUE(updated);
    EXPECT_EQ(saved.parameters, updated->parameters);
    EXPECT_FALSE(updated->overloadPredictionTrusted);
    EXPECT_TRUE(updated->warmStarted);
    EXPECT_LT(updated->revision, 1024);
    EXPECT_TRUE(saveNeeded);
}

TEST_F(AgentHistoryStateTest, testRestoredParametersAreRejectedByStartupSamples)
{
    const auto saved = saveState(GET_TEMP_DATA_PATH(), "candidate-rejected.state");
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.config(
        AgentHistoryR::RESOURCE_ID,
        std::string(R"({"state_file":")") + saved.stateFile +
            R"(","formula_fingerprint":"formula-a","history_fingerprint":"history-b","binary_fingerprint":"binary-a","fit_batch_size":1,"fit_interval":1})"));
    ASSERT_TRUE(helper.config(
        LatencyModelR::RESOURCE_ID,
        R"({"latency_model_fingerprint":"formula-a","trust_min_revision":2})"));
    auto latencyModel = helper.getOrCreateResPtr<LatencyModelR>();
    ASSERT_TRUE(latencyModel);

    auto initial = latencyModel->getSnapshot();
    ASSERT_TRUE(initial);
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
    bool saveNeeded = false;
    recordMatchingSamples(latencyModel, initial->parameters, 31, updated, saveNeeded);
    EXPECT_FALSE(updated);
    recordMatchingSamples(latencyModel, initial->parameters, 1, updated, saveNeeded, 32);
    ASSERT_TRUE(updated);
    EXPECT_EQ(initial->parameters, updated->parameters);
    EXPECT_EQ(2, updated->revision);
    EXPECT_TRUE(updated->overloadPredictionTrusted);
    EXPECT_FALSE(latencyModel->getSnapshot()->warmStarted);

    NaviSchedLatencySample sample;
    sample.features.appendRequest(0, 1024);
    sample.latencyMs = latencyModel->predict(sample.features, initial->parameters);
    sample.latencyMs += 100.0;
    sample.eventTimeUs = 33;
    ASSERT_TRUE(latencyModel->recordSample(sample, updated, saveNeeded));
    ASSERT_TRUE(updated);
    EXPECT_EQ(3, updated->revision);
    EXPECT_TRUE(updated->overloadPredictionTrusted);
}

TEST_F(AgentHistoryStateTest,
       testSubthresholdMasterCandidateDoesNotReplaceRestoredValidationCandidate)
{
    const auto saved = saveState(GET_TEMP_DATA_PATH(), "merged-candidates.state");
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.config(
        AgentHistoryR::RESOURCE_ID,
        makeHistoryConfig(saved.stateFile, "formula-a", "history-b", "binary-a")));
    ASSERT_TRUE(helper.config(
        LatencyModelR::RESOURCE_ID,
        R"({"latency_model_fingerprint":"formula-a","trust_min_revision":1024})"));
    auto latencyModel = helper.getOrCreateResPtr<LatencyModelR>();
    auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(latencyModel);
    ASSERT_TRUE(model);

    auto masterParameters = model->getInitialParameters();
    masterParameters[NonLinearPrefillModelR::BASE_LATENCY_PARAMETER_INDEX] = 5.0;
    MasterLatencyModelWarmupToAgentData warmup;
    warmup.fingerprint = "formula-a";
    warmup.formula = model->getFormula();
    warmup.binaryVersion = model->getBinaryVersion();
    NaviSchedLatencyModelCandidate candidate;
    candidate.sourceRevision = 300;
    candidate.parameters = masterParameters;
    candidate.optimizerState = NonLinearPrefillModelR::makeOptimizerStateData(
        model->getOptimizerState(), masterParameters.size());
    warmup.candidates = {candidate};
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
    ASSERT_TRUE(latencyModel->applyColdStartCandidates(warmup, updated));
    EXPECT_FALSE(updated);

    bool saveNeeded = false;
    recordMatchingSamples(latencyModel, masterParameters, 31, updated, saveNeeded);
    EXPECT_FALSE(updated);
    recordMatchingSamples(latencyModel, masterParameters, 1, updated, saveNeeded, 32);
    ASSERT_TRUE(updated);
    EXPECT_NE(masterParameters, updated->parameters);
    EXPECT_FALSE(updated->overloadPredictionTrusted);
    EXPECT_TRUE(updated->warmStarted);
    EXPECT_LT(updated->revision, 1024);
}

TEST_F(AgentHistoryStateTest, testGeometry26State7RoundTripAndBinaryIsolation)
{
    const auto path = GET_TEMP_DATA_PATH() + "latency-history-state/geometry26.state";
    auto history = makeHistory(path, "formula26", "history26", "binary26");
    ASSERT_TRUE(history);
    navi::NaviResourceHelper sourceHelper;
    ASSERT_TRUE(sourceHelper.config(NonLinearPrefillModelR::RESOURCE_ID,
                                    R"({"pd_separated":true})"));
    auto model = sourceHelper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(model);
    model->setExecutionGeometryEnabled(true);
    ASSERT_EQ(NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION, model->getBinaryVersion());
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
        EXPECT_EQ(7, version);
    }
    auto restored = makeHistory(path, "formula26", "history26", "binary26");
    ASSERT_TRUE(restored);
    navi::NaviResourceHelper loadHelper;
    ASSERT_TRUE(loadHelper.config(NonLinearPrefillModelR::RESOURCE_ID,
                                  R"({"pd_separated":true})"));
    auto loadedModel = loadHelper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(loadedModel);
    loadedModel->setExecutionGeometryEnabled(true);
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(restored->load(*loadedModel, snapshot, fitOnStart));
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION, snapshot->binaryVersion);
    EXPECT_EQ(updated, snapshot->parameters);
    EXPECT_EQ(model->getOptimizerState().step, loadedModel->getOptimizerState().step);
    ASSERT_EQ(1, restored->getHistory()->size());
    const auto &restoredSample = restored->getHistory()->front();
    EXPECT_EQ(3, restoredSample.features.getExecutionStepCount());
    EXPECT_EQ(8192, restoredSample.features.getGlobalPaddedTokens());
    std::vector<double> refit;
    ASSERT_TRUE(loadedModel->fit({restoredSample}, snapshot->parameters, refit));
    // Identical fingerprints cannot bypass the 25/26 envelope separation.
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

TEST_F(AgentHistoryStateTest, testGeometry26SaveRejectsHistoryWithoutShape)
{
    const auto path =
        GET_TEMP_DATA_PATH() + "latency-history-state/geometry26-invalid.state";
    auto history = makeHistory(path, "formula26", "history26", "binary26");
    ASSERT_TRUE(history);
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(
        helper.config(NonLinearPrefillModelR::RESOURCE_ID, R"({"pd_separated":true})"));
    auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(model);
    model->setExecutionGeometryEnabled(true);
    NaviSchedLatencySample sample;
    sample.features.appendRequest(0, 1024);
    sample.latencyMs = 500;
    sample.eventTimeUs = autil::TimeUtility::currentTime();
    bool fitReady = false, persistReady = false;
    ASSERT_TRUE(history->append(sample, fitReady, persistReady));
    auto saved = std::make_shared<NaviSchedLatencyModelSnapshot>();
    saved->formula = model->getFormula();
    saved->binaryVersion = NonLinearPrefillModelR::GEOMETRY_BINARY_VERSION;
    saved->parameters = model->getInitialParameters();
    saved->revision = 1;
    saved->overloadPredictionTrusted = true;
    EXPECT_FALSE(history->save(saved, model->getOptimizerState()));
}

TEST_F(AgentHistoryStateTest, testCacheSource34State11PreservesOptimizerAndKnownHistory)
{
    const auto path = GET_TEMP_DATA_PATH() + "latency-history-state/source34.state";
    std::filesystem::remove(path);
    auto history = makeHistory(path, "formula34", "history34", "binary34");
    ASSERT_TRUE(history);
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        R"({"pd_separated":true,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":true,"execution_cache_source_enabled":true})"));
    auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(model);
    NaviSchedLatencySample sample;
    sample.features.appendRequest(6144, 1024);
    ASSERT_TRUE(sample.features.setExecutionGeometry(1, 8192));
    ASSERT_TRUE(sample.features.setExecutionCacheSource(1, 6144));
    sample.latencyMs = 750;
    sample.eventTimeUs = autil::TimeUtility::currentTime();
    bool fitReady = false, persistReady = false;
    ASSERT_TRUE(history->append(sample, fitReady, persistReady));
    auto parameters = model->getInitialParameters();
    std::vector<double> fitted;
    ASSERT_TRUE(model->fit({sample}, parameters, fitted));
    auto saved = std::make_shared<NaviSchedLatencyModelSnapshot>();
    saved->formula = model->getFormula();
    saved->binaryVersion = 34;
    saved->parameters = fitted;
    saved->revision = 317;
    saved->overloadPredictionTrusted = true;
    saved->warmStarted = true;
    const auto optimizer = model->getOptimizerState();
    ASSERT_TRUE(history->save(saved, optimizer));
    {
        std::ifstream input(path, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(input)), {});
        autil::DataBuffer envelope(content.data(), content.size());
        std::string magic;
        int32_t version = 0;
        envelope.read(magic);
        envelope.read(version);
        EXPECT_EQ(11, version);
    }
    auto restored = makeHistory(path, "formula34", "history34", "binary34");
    ASSERT_TRUE(restored);
    navi::NaviResourceHelper loadedHelper;
    ASSERT_TRUE(loadedHelper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        R"({"pd_separated":true,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":true,"execution_cache_source_enabled":true})"));
    auto loadedPtr = loadedHelper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(loadedPtr);
    auto &loaded = *loadedPtr;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = false;
    ASSERT_TRUE(restored->load(loaded, snapshot, fitOnStart));
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(34, snapshot->binaryVersion);
    EXPECT_EQ(317, snapshot->revision);
    EXPECT_EQ(fitted, snapshot->parameters);
    EXPECT_TRUE(snapshot->overloadPredictionTrusted);
    EXPECT_TRUE(snapshot->warmStarted);
    EXPECT_EQ(optimizer.step, loaded.getOptimizerState().step);
    EXPECT_EQ(optimizer.adamMoment1, loaded.getOptimizerState().adamMoment1);
    EXPECT_EQ(optimizer.adamMoment2, loaded.getOptimizerState().adamMoment2);
    ASSERT_EQ(1u, restored->getHistory()->size());
    const auto &sample2 = restored->getHistory()->front();
    EXPECT_EQ(1, sample2.features.getExternalReadRequestCount());
    EXPECT_EQ(6144, sample2.features.getMaxExternalReadTokens());
    std::vector<double> firstNext, restoredNext;
    ASSERT_TRUE(model->fit({sample}, fitted, firstNext));
    ASSERT_TRUE(loaded.fit({sample2}, snapshot->parameters, restoredNext));
    EXPECT_EQ(firstNext, restoredNext);
    EXPECT_EQ(model->getOptimizerState().adamMoment1,
              loaded.getOptimizerState().adamMoment1);
    // The distinct source schema must not be silently interpreted as geometry-only
    // history.
    loaded.setExecutionCacheSourceEnabled(false);
    ASSERT_EQ(33, loaded.getBinaryVersion());
    ASSERT_TRUE(restored->load(loaded, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(restored->getHistory()->empty());
}

static void verifyExplicitSourceMigration(int64_t sourceVersion,
                                          const std::string &basePath,
                                          bool overloadTrusted = true)
{
    using JsonMap = autil::legacy::json::JsonMap;
    SCOPED_TRACE(sourceVersion);
    const auto root = basePath + "latency-history-state/source-migration-" +
                      std::to_string(sourceVersion) +
                      (overloadTrusted ? "-trusted/" : "-untrusted/");
    std::filesystem::create_directories(root);
    const auto oldPath = root + "old" + std::to_string(sourceVersion) + ".state";
    const auto targetVersion = sourceVersion >= 34 ? 35 : 34;
    const auto newPath = root + "new" + std::to_string(targetVersion) + ".state";
    std::filesystem::remove(oldPath);
    std::filesystem::remove(newPath);
    std::filesystem::remove(newPath + ".source33");
    std::filesystem::remove(newPath + ".source34");
    std::filesystem::remove(newPath + ".source35");
    std::filesystem::remove(newPath + ".source29");
    JsonMap formulaValues;
    formulaValues["navi_latency_mode"] =
        sourceVersion == 29
            ? std::string("split_output_complete_lookup_projected_geometry_v29")
            : std::string("split_continuous_lookup_projected_geometry_v33");
    formulaValues["hardware"] = std::string("r52-fixture");
    formulaValues["model"] = std::string("same-physical-model");
    formulaValues["dtype"] = std::string("bfloat16");
    formulaValues["fingerprint_version"] = int64_t{3};
    formulaValues["tensor_parallel_size"] = int64_t{1};
    formulaValues["pipeline_parallel_size"] = int64_t{1};
    formulaValues["data_parallel_size"] = int64_t{1};
    formulaValues["prefill_context_parallel_size"] = int64_t{8};
    formulaValues["decode_context_parallel_size"] = int64_t{1};
    JsonMap cachePolicy;
    cachePolicy["block_size"] = int64_t{6144};
    formulaValues["effective_cache_policy"] = cachePolicy;
    formulaValues["navi_sched_model_version"] = int64_t{3};
    JsonMap executionPolicy;
    executionPolicy["use_eagle"] = sourceVersion == 29;
    executionPolicy["block_size"] = int64_t{6144};
    formulaValues["runtime_prefill_execution_policy"] = executionPolicy;
    if (sourceVersion == 33) {
        formulaValues["execution_completion_boundary"] =
            std::string("scheduler_update_complete_v1");
    }
    if (sourceVersion == 34) {
        formulaValues["navi_latency_mode"] =
            std::string("split_native_start_cache_source_geometry_v34");
        formulaValues["execution_cache_source_feature_protocol"] = int64_t{1};
        formulaValues["execution_start_boundary"] = std::string("native_start_call_v1");
        formulaValues["execution_completion_boundary"] =
            std::string("scheduler_update_complete_v1");
        formulaValues["projected_execution_geometry_enabled"] = true;
        formulaValues["prefill_execution_feature_protocol"] = int64_t{1};
    }
    if (sourceVersion == 35) {
        formulaValues["navi_latency_mode"] =
            std::string("split_native_start_measured_cache_source_geometry_v35");
        formulaValues["execution_cache_source_feature_protocol"] = int64_t{1};
        formulaValues["execution_start_boundary"] = std::string("native_start_call_v1");
        formulaValues["execution_completion_boundary"] =
            std::string("scheduler_update_complete_v1");
        formulaValues["projected_execution_geometry_enabled"] = false;
        formulaValues["training_geometry_origin"] = std::string("observed_runner");
        formulaValues["dispatch_geometry_origin"] =
            std::string("all_cache_resolved_forecast");
    }
    const auto formula33 = autil::legacy::ToJsonString(formulaValues, true);
    JsonMap historyValues;
    historyValues["formula_fingerprint"] = formula33;
    historyValues["history_limit"] = int64_t{2000};
    const auto history33 = autil::legacy::ToJsonString(historyValues, true);
    JsonMap binaryValues;
    binaryValues["navi_sched_commit"] = std::string(40, 'a');
    binaryValues["framework_binary"] = std::string("unchanged-r52-framework");
    const auto binary33 = autil::legacy::ToJsonString(binaryValues, true);
    formulaValues["navi_latency_mode"] =
        std::string("split_native_start_cache_source_geometry_v34");
    formulaValues["execution_cache_source_feature_protocol"] = int64_t{1};
    formulaValues["execution_start_boundary"] = std::string("native_start_call_v1");
    formulaValues["execution_completion_boundary"] =
        std::string("scheduler_update_complete_v1");
    formulaValues["navi_sched_model_version"] = std::string("native-start-fixture");
    if (sourceVersion == 29) {
        executionPolicy["use_eagle"] = false;
        formulaValues["runtime_prefill_execution_policy"] = executionPolicy;
    }
    if (sourceVersion == 34) {
        formulaValues["navi_latency_mode"] =
            std::string("split_native_start_measured_cache_source_geometry_v35");
        formulaValues["projected_execution_geometry_enabled"] = false;
        formulaValues.erase("prefill_execution_feature_protocol");
        formulaValues["training_geometry_origin"] = std::string("observed_runner");
        formulaValues["dispatch_geometry_origin"] =
            std::string("all_cache_resolved_forecast");
    }
    if (sourceVersion == 35) {
        autil::legacy::FromJsonString(formulaValues, formula33);
    }
    ExecutionCacheSourceStateTarget target;
    target.formulaFingerprint = autil::legacy::ToJsonString(formulaValues, true);
    historyValues["formula_fingerprint"] = target.formulaFingerprint;
    target.historyFingerprint = autil::legacy::ToJsonString(historyValues, true);
    binaryValues["navi_sched_commit"] = std::string(40, 'b');
    target.binaryFingerprint = autil::legacy::ToJsonString(binaryValues, true);
    const auto historyResource = [&](const std::string &path, const std::string &formula,
                                     const std::string &history,
                                     const std::string &binary) {
        JsonMap config;
        config["state_file"] = path;
        config["formula_fingerprint"] = formula;
        config["history_fingerprint"] = history;
        config["binary_fingerprint"] = binary;
        navi::NaviResourceHelper helper;
        if (!helper.config(AgentHistoryR::RESOURCE_ID,
                           autil::legacy::ToJsonString(config, true))) {
            return AgentHistoryRPtr();
        }
        return helper.getOrCreateResPtr<AgentHistoryR>();
    };
    auto history = historyResource(oldPath, formula33, history33, binary33);
    ASSERT_TRUE(history);
    navi::NaviResourceHelper modelHelper;
    ASSERT_TRUE(modelHelper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        sourceVersion == 35
            ? R"({"pd_separated":true,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":false,"execution_cache_source_enabled":true})"
        : sourceVersion == 34
            ? R"({"pd_separated":true,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":true,"execution_cache_source_enabled":true})"
            : R"({"pd_separated":true,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":true})"));
    auto model = modelHelper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(model);
    NaviSchedLatencySample sample;
    sample.features.appendRequest(6144, 1024);
    ASSERT_TRUE(sample.features.setExecutionGeometry(1, 8192));
    if (sourceVersion >= 34) {
        ASSERT_TRUE(sample.features.setExecutionCacheSource(1, 6144));
    }
    sample.latencyMs = 650;
    sample.eventTimeUs = autil::TimeUtility::currentTime();
    bool fitReady = false, persistReady = false;
    ASSERT_TRUE(history->append(sample, fitReady, persistReady));
    std::vector<double> parameters;
    ASSERT_TRUE(model->fit({sample}, model->getInitialParameters(), parameters));
    auto saved = std::make_shared<NaviSchedLatencyModelSnapshot>();
    saved->formula = model->getFormula();
    saved->binaryVersion = sourceVersion >= 34 ? sourceVersion : 33;
    saved->parameters = parameters;
    saved->revision = 713;
    saved->overloadPredictionTrusted = overloadTrusted;
    saved->warmStarted = true;
    saved->residualParameters[3] = 7.5;
    saved->residualLastUpdateTimeUs = autil::TimeUtility::currentTime();
    const auto optimizer = model->getOptimizerState();
    ASSERT_TRUE(history->save(saved, optimizer));
    const auto readFile = [](const std::string &path) {
        std::ifstream input(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    };
    const auto rewriteVersion = [](const std::string &raw, int64_t version) {
        autil::DataBuffer envelope(const_cast<char *>(raw.data()), raw.size());
        std::string magic, payload;
        int32_t stateVersion = 0;
        uint32_t checksum = 0;
        envelope.read(magic);
        envelope.read(stateVersion);
        envelope.read(checksum);
        envelope.read(payload);
        autil::DataBuffer input(payload.data(), payload.size());
        std::string formula, history, binary;
        int32_t formulaId = 0;
        int64_t sourceVersion = 0;
        input.read(formula);
        input.read(history);
        input.read(binary);
        input.read(formulaId);
        input.read(sourceVersion);
        autil::DataBuffer output;
        output.write(formula);
        output.write(history);
        output.write(binary);
        output.write(formulaId);
        output.write(version);
        output.writeBytes(input.getData(), input.getDataLen());
        autil::DataBuffer wrapped;
        wrapped.write(magic);
        wrapped.write(stateVersion);
        wrapped.write(autil::CRC32C::Mask(
            autil::CRC32C::Value(output.getData(), output.getDataLen())));
        wrapped.write(std::string(output.getData(), output.getDataLen()));
        return std::string(wrapped.getData(), wrapped.getDataLen());
    };
    if (sourceVersion == 29) {
        // State10's geometry history layout is identical. The serving loader is
        // deliberately not taught to accept29; only this offline tool converts it.
        const auto old29 = rewriteVersion(readFile(oldPath), 29);
        std::ofstream output(oldPath, std::ios::binary);
        output.write(old29.data(), old29.size());
    }
    EXPECT_FALSE(NonLinearPrefillModelR::supportsBinaryVersion(29));
    const auto original = readFile(oldPath);
    ExecutionCacheSourceStateMigration migrated;
    std::string error;
    if (!overloadTrusted) {
        // The existing serving loader refuses such snapshots before loading
        // parameters/history. Refuse this upgrade before any source is changed.
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> sourceSnapshot;
        bool sourceFit = false;
        auto sourceControl = historyResource(oldPath, formula33, history33, binary33);
        ASSERT_TRUE(sourceControl);
        ASSERT_TRUE(sourceControl->load(*model, sourceSnapshot, sourceFit));
        EXPECT_FALSE(sourceSnapshot);
        EXPECT_FALSE(migrateExecutionCacheSourceState(original, target, migrated, error));
        EXPECT_TRUE(migrated.state.empty());
        EXPECT_EQ(0, migrated.sourceModelVersion);
        EXPECT_EQ(original, readFile(oldPath));
        return;
    }
    ASSERT_TRUE(migrateExecutionCacheSourceState(original, target, migrated, error));
    EXPECT_EQ(sourceVersion, migrated.sourceModelVersion);
    EXPECT_EQ(targetVersion, migrated.targetModelVersion);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(713, migrated.revision);
    EXPECT_EQ(optimizer.step, migrated.optimizerStep);
    EXPECT_EQ(1u, migrated.archivedHistorySamples);
    EXPECT_GE(migrated.archivedHistoryBuckets, 1u);
    {
        std::ofstream output(newPath, std::ios::binary);
        output.write(migrated.state.data(), migrated.state.size());
    }
    auto restored = historyResource(newPath, target.formulaFingerprint,
                                    target.historyFingerprint, target.binaryFingerprint);
    ASSERT_TRUE(restored);
    navi::NaviResourceHelper loadedHelper;
    ASSERT_TRUE(loadedHelper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        sourceVersion >= 34
            ? R"({"pd_separated":true,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":false,"execution_cache_source_enabled":true})"
            : R"({"pd_separated":true,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":true,"execution_cache_source_enabled":true})"));
    auto loaded = loadedHelper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(loaded);
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = false;
    ASSERT_TRUE(restored->load(*loaded, snapshot, fitOnStart));
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(targetVersion, snapshot->binaryVersion);
    EXPECT_EQ(saved->revision, snapshot->revision);
    auto expected = parameters;
    expected.resize(21, 0.0);
    EXPECT_EQ(expected, snapshot->parameters);
    EXPECT_EQ(optimizer.step, loaded->getOptimizerState().step);
    EXPECT_EQ(optimizer.adamMoment1, loaded->getOptimizerState().adamMoment1);
    EXPECT_EQ(optimizer.adamMoment2, loaded->getOptimizerState().adamMoment2);
    EXPECT_EQ(saved->residualParameters, snapshot->residualParameters);
    EXPECT_EQ(saved->residualLastUpdateTimeUs, snapshot->residualLastUpdateTimeUs);
    EXPECT_TRUE(snapshot->warmStarted);
    EXPECT_EQ(saved->overloadPredictionTrusted, snapshot->overloadPredictionTrusted);
    if (sourceVersion == 35) {
        if (std::getenv("NAVI_TEST_CURRENT35_HISTORY_LOSS_MUTANT")) {
            const_cast<AgentHistoryR::History *>(restored->getHistory().get())->clear();
        }
        ASSERT_EQ(1u, restored->getHistory()->size());
        EXPECT_EQ(650, restored->getHistory()->front().latencyMs);
        const auto stateTail = [](const std::string &raw) {
            autil::DataBuffer envelope(const_cast<char *>(raw.data()), raw.size());
            std::string magic, payload, fingerprint;
            int32_t version = 0;
            uint32_t checksum = 0;
            envelope.read(magic);
            envelope.read(version);
            envelope.read(checksum);
            envelope.read(payload);
            autil::DataBuffer body(payload.data(), payload.size());
            body.read(fingerprint);
            body.read(fingerprint);
            body.read(fingerprint);
            return std::string(body.getData(), body.getDataLen());
        };
        EXPECT_EQ(stateTail(original), stateTail(migrated.state));
        EXPECT_TRUE(migrated.preservedStateTail);
    } else {
        EXPECT_TRUE(restored->getHistory()->empty());
        EXPECT_FALSE(migrated.preservedStateTail);
    }
    EXPECT_FALSE(restored->requiresParameterValidation());
    if (sourceVersion < 34) {
        ASSERT_TRUE(sample.features.setExecutionCacheSource(0, 0));
    } else {
        EXPECT_NE(0.0, optimizer.adamMoment1[20]);
        EXPECT_NE(0.0, optimizer.adamMoment2[20]);
        if (sourceVersion == 34 && std::getenv("NAVI_TEST_MEASURED_HISTORY_MUTANT")) {
            // Deliberately relabel one projected sample as observed. The unchanged
            // empty-history assertion below must catch the invalid migration.
            bool fit = false, persist = false;
            ASSERT_TRUE(restored->append(sample, fit, persist));
        }
        if (sourceVersion == 34) {
            EXPECT_TRUE(restored->getHistory()->empty());
        }
    }
    EXPECT_DOUBLE_EQ(model->predictConfigured(sample.features, parameters),
                     loaded->predictConfigured(sample.features, snapshot->parameters));
    EXPECT_EQ(original, readFile(oldPath));

    const auto badTarget = [&](JsonMap changed) {
        auto bad = target;
        bad.formulaFingerprint = autil::legacy::ToJsonString(changed, true);
        EXPECT_FALSE(migrateExecutionCacheSourceState(original, bad, migrated, error));
        EXPECT_TRUE(migrated.state.empty());
        EXPECT_EQ(0, migrated.sourceModelVersion);
    };
    auto changed = formulaValues;
    changed["hardware"] = std::string("different-hardware");
    badTarget(changed);
    changed = formulaValues;
    changed["model"] = std::string("different-model");
    badTarget(changed);
    changed = formulaValues;
    changed["pipeline_parallel_size"] = int64_t{2};
    badTarget(changed);
    changed = formulaValues;
    changed.erase("model");
    badTarget(changed);
    changed = formulaValues;
    changed.erase("execution_cache_source_feature_protocol");
    badTarget(changed);
    changed = formulaValues;
    changed.erase("execution_start_boundary");
    badTarget(changed);
    changed = formulaValues;
    changed["execution_completion_boundary"] = std::string("future_callback_not_output");
    badTarget(changed);
    changed = formulaValues;
    auto badPolicy = executionPolicy;
    badPolicy["block_size"] = int64_t{32};
    changed["runtime_prefill_execution_policy"] = badPolicy;
    badTarget(changed);
    changed = formulaValues;
    badPolicy = executionPolicy;
    badPolicy["use_eagle"] = true;
    changed["runtime_prefill_execution_policy"] = badPolicy;
    badTarget(changed);
    if (sourceVersion >= 34) {
        changed = formulaValues;
        changed["projected_execution_geometry_enabled"] = true;
        badTarget(changed);
        changed = formulaValues;
        changed["training_geometry_origin"] = std::string("projected_dispatch");
        badTarget(changed);
        changed = formulaValues;
        changed["navi_latency_mode"] =
            std::string("split_native_start_cache_source_geometry_v34");
        badTarget(changed);
    }
    if (sourceVersion == 35) {
        changed = formulaValues;
        changed["navi_sched_model_version"] = std::string("different-valid-epoch");
        badTarget(changed);
        auto alteredHistory = target;
        auto cadence = historyValues;
        cadence["history_limit"] = int64_t{2001};
        alteredHistory.historyFingerprint = autil::legacy::ToJsonString(cadence, true);
        EXPECT_FALSE(
            migrateExecutionCacheSourceState(original, alteredHistory, migrated, error));
        EXPECT_TRUE(migrated.state.empty());
    }
    changed = formulaValues;
    changed["navi_sched_model_version"] = JsonMap();
    badTarget(changed);
    auto wrongFramework = target;
    auto wrongBinary = binaryValues;
    wrongBinary["framework_binary"] = std::string("unapproved-framework");
    wrongFramework.binaryFingerprint = autil::legacy::ToJsonString(wrongBinary, true);
    EXPECT_FALSE(
        migrateExecutionCacheSourceState(original, wrongFramework, migrated, error));
    EXPECT_TRUE(migrated.state.empty());
    EXPECT_FALSE(migrateExecutionCacheSourceState(rewriteVersion(original, 32), target,
                                                  migrated, error));
    EXPECT_TRUE(migrated.state.empty());
    EXPECT_FALSE(migrateExecutionCacheSourceState(
        rewriteVersion(original, sourceVersion == 29 ? 33 : 29), target, migrated,
        error));
    EXPECT_TRUE(migrated.state.empty());
    auto corrupt = original;
    corrupt.back() ^= 1;
    EXPECT_FALSE(migrateExecutionCacheSourceState(corrupt, target, migrated, error));
    EXPECT_TRUE(migrated.state.empty());

    // Exercise the real standalone binary's no-overwrite and byte-exact archive contract.
    std::filesystem::remove(newPath);
    JsonMap config;
    config["formula_fingerprint"] = target.formulaFingerprint;
    config["history_fingerprint"] = target.historyFingerprint;
    config["binary_fingerprint"] = target.binaryFingerprint;
    const auto configPath = root + "target.json";
    {
        std::ofstream output(configPath);
        output << autil::legacy::ToJsonString(config, true);
    }
    const auto runfiles = std::getenv("TEST_SRCDIR");
    const auto workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(nullptr, runfiles);
    ASSERT_NE(nullptr, workspace);
    const auto executable = std::string(runfiles) + "/" + workspace +
                            "/aios/navi_sched/migrate_execution_cache_source_state";
    const auto run = [&]() {
        const auto pid = ::fork();
        if (pid == 0) {
            ::execl(executable.c_str(), executable.c_str(), oldPath.c_str(),
                    newPath.c_str(), configPath.c_str(), static_cast<char *>(nullptr));
            ::_exit(127);
        }
        if (pid < 0) {
            return -1;
        }
        int status = 0;
        if (::waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) {
            return -1;
        }
        return WEXITSTATUS(status);
    };
    ASSERT_EQ(0, run());
    const auto archivePath = newPath + ".source" + std::to_string(sourceVersion);
    EXPECT_EQ(original, readFile(archivePath));
    EXPECT_EQ(original, readFile(oldPath));
    const auto firstOutput = readFile(newPath);
    ASSERT_EQ(2, run());
    EXPECT_EQ(firstOutput, readFile(newPath));
    EXPECT_EQ(original, readFile(archivePath));
    // An archive of the other known source version also prevents ambiguous reuse.
    std::filesystem::remove(newPath);
    std::filesystem::remove(archivePath);
    const auto otherArchive = newPath + (sourceVersion == 29 ? ".source33" : ".source29");
    {
        std::ofstream output(otherArchive);
        output << "preserved-other-source";
    }
    EXPECT_EQ(2, run());
    EXPECT_FALSE(std::filesystem::exists(newPath));
    EXPECT_EQ("preserved-other-source", readFile(otherArchive));
}

TEST_F(AgentHistoryStateTest,
       testExplicitSourceMigrationPreservesOldModelAndArchivesUnknownHistory)
{
    verifyExplicitSourceMigration(33, GET_TEMP_DATA_PATH());
}

TEST_F(AgentHistoryStateTest,
       testExplicitModel29MigrationPreservesAdamAndRejectsUnapprovedChanges)
{
    verifyExplicitSourceMigration(29, GET_TEMP_DATA_PATH());
}

TEST_F(AgentHistoryStateTest,
       testExplicitMeasuredGeometryMigrationPreservesAll21Coordinates)
{
    verifyExplicitSourceMigration(34, GET_TEMP_DATA_PATH());
}

TEST_F(AgentHistoryStateTest,
       testExplicitCurrent35BinaryRebindPreservesFullHistoryAndStateTail)
{
    verifyExplicitSourceMigration(35, GET_TEMP_DATA_PATH());
}

TEST_F(AgentHistoryStateTest,
       testExplicitCurrent35RebindRejectsStatesTheLoaderCannotRestore)
{
    verifyExplicitSourceMigration(35, GET_TEMP_DATA_PATH(), false);
}

TEST_F(AgentHistoryStateTest, testLegacySavePayloadAbiPreservesEnvelopeBytes)
{
    const auto path =
        GET_TEMP_DATA_PATH() + "latency-history-state/legacy-envelope-abi.state";
    auto history = makeHistory(path, "formula-a", "history-a", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    auto snapshot = std::make_shared<NaviSchedLatencyModelSnapshot>();
    snapshot->formula = model.getFormula();
    snapshot->binaryVersion = model.getBinaryVersion();
    snapshot->revision = 1;
    snapshot->parameters = model.getInitialParameters();
    snapshot->overloadPredictionTrusted = true;
    std::string payload;
    ASSERT_TRUE(history->makeSavePayload(snapshot, model.getOptimizerState(), payload));
    auto readFile = [&]() {
        std::ifstream input(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    };
    ASSERT_TRUE(history->savePayload(payload));
    const auto original = readFile();
    ASSERT_TRUE(history->savePayload(payload, 24));
    EXPECT_EQ(original, readFile());
    ASSERT_TRUE(history->savePayload(payload, NonLinearPrefillModelR::PD_BINARY_VERSION));
    EXPECT_EQ(original, readFile());
}

// Exercise the real serving resource, rather than inferring successful restore
// from an init return code. Optional file inputs extend this same test to a
// private frozen production pool; they never touch the source files.
TEST_F(AgentHistoryStateTest, testMasterPoolLegacy29VersusExplicit34Load)
{
    auto inspect = [&](const std::string &raw, size_t expectedCount,
                       uint64_t expectedFailures, int iteration) {
        const auto path = GET_TEMP_DATA_PATH() + "latency-history-state/pool-load-" +
                          std::to_string(iteration) + ".state";
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(raw.data(), raw.size());
        }
        navi::NaviResourceHelper helper;
        ASSERT_TRUE(
            helper.config(MasterLatencyModelR::RESOURCE_ID,
                          std::string(R"({"state_file":")") + path +
                              R"(","source_min_revision":2,"max_candidates":1024})"));
        auto resource = helper.getOrCreateResPtr<MasterLatencyModelR>();
        ASSERT_TRUE(resource);
        const auto stats = resource->getStats();
        EXPECT_EQ(expectedCount, stats.candidateCount);
        EXPECT_EQ(expectedFailures, stats.persistFailureCount);
        if (expectedCount) {
            autil::DataBuffer envelope(const_cast<char *>(raw.data()), raw.size());
            std::string magic, payload, fingerprint;
            int32_t version = 0;
            uint32_t checksum = 0, domains = 0;
            envelope.read(magic);
            envelope.read(version);
            envelope.read(checksum);
            envelope.read(payload);
            autil::DataBuffer body(payload.data(), payload.size());
            body.read(domains);
            body.read(fingerprint);
            ASSERT_EQ(1u, domains);
            auto warmup = resource->makeWarmup(fingerprint, 1, 34, 21);
            ASSERT_TRUE(warmup);
            ASSERT_TRUE(warmup->valid());
            ASSERT_EQ(expectedCount, warmup->candidates.size());
            for (const auto &candidate : warmup->candidates) {
                EXPECT_EQ(21u, candidate.parameters.size());
                EXPECT_TRUE(candidate.optimizerState.valid(21));
            }
        }
    };
    auto makePool = [&](int64_t binaryVersion) {
        navi::NaviResourceHelper helper;
        EXPECT_TRUE(helper.config(
            NonLinearPrefillModelR::RESOURCE_ID,
            R"({"pd_separated":true,"execution_geometry_enabled":true,"projected_execution_geometry_enabled":true,"execution_cache_source_enabled":true})"));
        auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
        auto parameters = model->getInitialParameters();
        parameters.resize(binaryVersion == 29 ? 10 : 21);
        NaviSchedLatencyOptimizerState optimizer;
        optimizer.adamMoment1.resize(parameters.size(), 0.0);
        optimizer.adamMoment2.resize(parameters.size(), 0.0);
        optimizer.step = 99;
        autil::DataBuffer payload;
        payload.write(uint32_t{1});
        payload.write(std::string("explicit-pool-domain"));
        payload.write(int32_t{1});
        payload.write(binaryVersion);
        payload.write(static_cast<uint32_t>(parameters.size()));
        payload.write(uint32_t{4});
        for (int i = 0; i < 4; ++i) {
            payload.write(std::string("candidate-") + std::to_string(i));
            payload.write(int64_t{100 + i});
            payload.write(int64_t{123456 + i});
            // Distinct candidates: warmup intentionally deduplicates equal predictions.
            auto candidateParameters = parameters;
            candidateParameters[0] += i * 0.001;
            payload.write(candidateParameters);
            payload.write(optimizer);
            payload.write(int32_t{LatencyResidualModel::VERSION});
            payload.write(LatencyResidualModel::makeInitialParameters());
            payload.write(int64_t{0});
        }
        autil::DataBuffer out;
        out.write(std::string("navi_sched_master_latency_models"));
        out.write(int32_t{5});
        out.write(autil::CRC32C::Mask(
            autil::CRC32C::Value(payload.getData(), payload.getDataLen())));
        out.write(std::string(payload.getData(), payload.getDataLen()));
        return std::string(out.getData(), out.getDataLen());
    };
    const auto old = makePool(29), current = makePool(34);
    inspect(old, 0, 0, 0);
    inspect(current, 4, 0, 1);
    inspect(old, 0, 0, 2);
    auto corrupt = current;
    corrupt.back() ^= 1;
    inspect(corrupt, 0, 1, 3);
    const auto realOld = std::getenv("NAVI_SCHED_POOL_REAL_OLD");
    const auto realNew = std::getenv("NAVI_SCHED_POOL_REAL_NEW");
    ASSERT_EQ(bool(realOld), bool(realNew));
    if (realOld && realNew) {
        auto read = [](const char *path) {
            std::ifstream in(path, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(in), {});
        };
        const auto source = read(realOld), target = read(realNew);
        ASSERT_FALSE(source.empty());
        ASSERT_FALSE(target.empty());
        inspect(source, 0, 0, 4);
        inspect(target, 4, 0, 5);
        inspect(source, 0, 0, 6);
        auto damaged = target;
        damaged.back() ^= 1;
        inspect(damaged, 0, 1, 7);
    }
}

TEST_F(AgentHistoryStateTest, testMasterPool34And35StayDistinctAcrossSaveAndReload)
{
    const auto path = GET_TEMP_DATA_PATH() + "latency-history-state/mixed34-35.state";
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    navi::NaviResourceHelper modelHelper;
    ASSERT_TRUE(modelHelper.config(
        NonLinearPrefillModelR::RESOURCE_ID,
        R"({"pd_separated":true,"execution_geometry_enabled":true,"execution_cache_source_enabled":true})"));
    auto model = modelHelper.getOrCreateResPtr<NonLinearPrefillModelR>();
    ASSERT_TRUE(model);
    ASSERT_EQ(35, model->getBinaryVersion());
    autil::DataBuffer payload;
    payload.write(uint32_t{2});
    for (const auto version : {int64_t{34}, int64_t{35}}) {
        payload.write(std::string("domain-") + std::to_string(version));
        payload.write(int32_t{1});
        payload.write(version);
        payload.write(uint32_t{21});
        payload.write(uint32_t{1});
        payload.write(std::string("candidate-") + std::to_string(version));
        payload.write(int64_t{version});
        payload.write(int64_t{123456});
        auto parameters = model->getInitialParameters();
        parameters[0] += version * 0.001;
        NaviSchedLatencyOptimizerState optimizer;
        optimizer.adamMoment1.resize(21, 0.001 * version);
        optimizer.adamMoment2.resize(21, 0.002 * version);
        optimizer.step = 99;
        payload.write(parameters);
        payload.write(optimizer);
        payload.write(int32_t{LatencyResidualModel::VERSION});
        payload.write(LatencyResidualModel::makeInitialParameters());
        payload.write(int64_t{0});
    }
    autil::DataBuffer envelope;
    envelope.write(std::string("navi_sched_master_latency_models"));
    envelope.write(int32_t{5});
    envelope.write(autil::CRC32C::Mask(
        autil::CRC32C::Value(payload.getData(), payload.getDataLen())));
    envelope.write(std::string(payload.getData(), payload.getDataLen()));
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(envelope.getData(), envelope.getDataLen());
    }
    const auto inspect = [&](bool save) {
        navi::NaviResourceHelper helper;
        ASSERT_TRUE(
            helper.config(MasterLatencyModelR::RESOURCE_ID,
                          std::string(R"({"state_file":")") + path +
                              R"(","source_min_revision":2,"max_candidates":1024})"));
        auto pool = helper.getOrCreateResPtr<MasterLatencyModelR>();
        ASSERT_TRUE(pool);
        EXPECT_EQ(2u, pool->getStats().candidateCount);
        for (const auto version : {int64_t{34}, int64_t{35}}) {
            auto warmup =
                pool->makeWarmup("domain-" + std::to_string(version), 1, version, 21);
            ASSERT_TRUE(warmup);
            ASSERT_TRUE(warmup->valid());
            ASSERT_EQ(1u, warmup->candidates.size());
            const auto &candidate = warmup->candidates.front();
            EXPECT_EQ(version, candidate.sourceRevision);
            EXPECT_DOUBLE_EQ(model->getInitialParameters()[0] + version * 0.001,
                             candidate.parameters[0]);
            EXPECT_EQ(99, candidate.optimizerState.step);
            EXPECT_DOUBLE_EQ(0.001 * version, candidate.optimizerState.adamMoment1[20]);
            EXPECT_DOUBLE_EQ(0.002 * version, candidate.optimizerState.adamMoment2[20]);
            auto wrong = pool->makeWarmup("domain-" + std::to_string(version), 1,
                                          version == 34 ? 35 : 34, 21);
            EXPECT_FALSE(wrong);
        }
        if (save) {
            ASSERT_TRUE(pool->save());
        }
    };
    inspect(true);
    inspect(false);
}

} // namespace navi_sched
