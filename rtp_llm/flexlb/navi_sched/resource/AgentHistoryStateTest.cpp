#include "aios/navi_sched/resource/AgentHistoryR.h"

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
        424, 0, 0, 0, 0.0, 0.0, 541, 2'078'064, 40'022,
        14'084'028.0, 153'692'768.0));
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
                                      int workloadIndex = -1,
                                      double workloadValue = -1.0)
{
    SavedState result;
    result.stateFile = tempDataPath + "latency-history-state/" + name;
    std::filesystem::create_directories(
        std::filesystem::path(result.stateFile).parent_path());
    navi::NaviResourceHelper helper;
    EXPECT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID,
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
    EXPECT_EQ(saved.residualLastUpdateTimeUs,
              snapshot->residualLastUpdateTimeUs);
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
        ASSERT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID,
                                  modelConfig));
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
        snapshot->residualParameters =
            LatencyResidualModel::makeInitialParameters();
        snapshot->residualLastUpdateTimeUs =
            autil::TimeUtility::currentTime();
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
        ASSERT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID,
                                  modelConfig));
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
        ASSERT_TRUE(latencyModel->recordSample(
            sample, updated, saveNeeded,
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
        ASSERT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID,
                                  modelConfig));
        ASSERT_TRUE(helper.config(LatencyModelR::RESOURCE_ID, latencyConfig));
        auto latencyModel = helper.getOrCreateResPtr<LatencyModelR>();
        auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
        ASSERT_TRUE(latencyModel);
        ASSERT_TRUE(model);
        auto restored = latencyModel->getSnapshot();
        ASSERT_TRUE(restored);
        EXPECT_EQ(expectedRecoveredParameters, restored->parameters);
        EXPECT_EQ(expectedRecoveredOptimizer.step, model->getOptimizerState().step);
        EXPECT_EQ(expectedRecoveredOptimizer.adamMoment1, model->getOptimizerState().adamMoment1);
        EXPECT_EQ(expectedRecoveredOptimizer.adamMoment2, model->getOptimizerState().adamMoment2);
    }
}

TEST_F(AgentHistoryStateTest, testInvalidResidualStateOnlyResetsResidual)
{
    const auto saved = saveInvalidResidualState(
        GET_TEMP_DATA_PATH(), "invalid-residual.state");
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
    const auto saved = saveControlledFormulaState(
        GET_TEMP_DATA_PATH(), "formula-binary-change.state");
    auto history = makeHistory(saved.stateFile, "formula-a", "history-a", "binary-a");
    ASSERT_TRUE(history);
    NonLinearPrefillModelR model;
    const auto optimizerBefore = model.getOptimizerState();
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    bool fitOnStart = true;
    ASSERT_TRUE(history->load(model, snapshot, fitOnStart));
    EXPECT_FALSE(snapshot);
    EXPECT_TRUE(history->getHistory()->empty());
    EXPECT_EQ(optimizerBefore.adamMoment1,
              model.getOptimizerState().adamMoment1);
    EXPECT_EQ(optimizerBefore.adamMoment2,
              model.getOptimizerState().adamMoment2);
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
                GET_TEMP_DATA_PATH(), "workload-" + std::to_string(index) +
                    (value < 0.0 ? "-negative.state" : "-zero.state"),
                0, index, value);
            navi::NaviResourceHelper helper;
            ASSERT_TRUE(helper.config(NonLinearPrefillModelR::RESOURCE_ID,
                index >= 10 ? R"({"use_capacity_split_feature":true})" : "{}"));
            auto model = helper.getOrCreateResPtr<NonLinearPrefillModelR>();
            ASSERT_TRUE(model);
            const auto optimizerBefore = model->getOptimizerState();
            auto history = makeHistory(saved.stateFile, "formula-a", "history-a", "binary-a");
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
    ASSERT_TRUE(helper.config(LatencyModelR::RESOURCE_ID,
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
    ASSERT_TRUE(helper.config(LatencyModelR::RESOURCE_ID,
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

}
