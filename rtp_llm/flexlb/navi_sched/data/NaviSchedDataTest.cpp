#include "rtp_llm/flexlb/navi_sched/data/AgentCacheToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentControlToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentLatencyToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentUpdateToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentRequestEventToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"
#include "rtp_llm/flexlb/navi_sched/data/ClientRequestToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterPendingCacheSyncToAgentData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterLatencyModelWarmupToAgentData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterScheduleRequestData.h"
#include "rtp_llm/flexlb/navi_sched/data/NaviSchedAgentEvent.h"
#include "rtp_llm/flexlb/navi_sched/data/SummaryUtil.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyModelR.h"
#include "autil/legacy/jsonizable.h"
#include "autil/HashAlgorithm.h"
#include "autil/TimeUtility.h"
#include "unittest/unittest.h"
#include <cmath>
#include <fstream>
#include <limits>

using namespace std;
using namespace navi_sched;

TEST(NativeAcceptedWorkloadDataTest, ReceiveReportCarriesAtomicWorkloadSnapshot)
{
    AgentUpdateToMasterData input;
    input.event = NaviSchedAgentEvent::REQUEST_RECEIVED;
    input.eventTimeUs = 100;
    input.requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    input.requestEvent->requestId = 11;
    input.requestEvent->inflightRequests = 1;
    input.workload = std::make_shared<NaviSchedWorkloadSnapshot>();
    auto &workload = *input.workload;
    workload.revision = 1;
    workload.eventTimeUs = 100;
    workload.prefillCapacitySplit = true;
    workload.requestCount = workload.waitingRequestCount = 1;
    workload.remainingPrefillRequestCount = workload.waitingCapacityPrefillRequestCount = 1;
    workload.totalPromptTokenCount = workload.remainingTotalTokenCount =
        workload.remainingPrefillTokenCount = workload.waitingCapacityPrefillTokenCount = 32;
    workload.remainingTotalTokenSquareSum = workload.remainingPrefillTokenSquareSum =
        workload.waitingCapacityPrefillTokenSquareSum = 1024;
    ASSERT_TRUE(input.valid());
    autil::DataBuffer buffer;
    input.serialize(buffer);
    AgentUpdateToMasterData output;
    output.deserialize(buffer);
    ASSERT_TRUE(output.valid());
    ASSERT_TRUE(output.workload);
    EXPECT_EQ(11u, output.requestEvent->requestId);
    EXPECT_EQ(1, output.workload->requestCount);
    EXPECT_EQ(0, output.workload->batchCount);
    EXPECT_EQ(32, output.workload->remainingPrefillTokenCount);
    output.workload->admittedBatch.scheduleBatchId = 101;
    output.workload->admittedBatch.localBatchId = 7;
    output.workload->admittedBatch.requestCount = 1;
    EXPECT_FALSE(output.valid());
    output.workload.reset();
    EXPECT_TRUE(output.valid());
}

class NaviSchedDataTest : public TESTBASE {};

TEST_F(NaviSchedDataTest, testAgentUpdateReceiveTimingIsLocalAndMonotonic)
{
    AgentUpdateToMasterData input;
    input.event = NaviSchedAgentEvent::WORKLOAD_UPDATED;
    input.eventTimeUs = 123;
    input.workload = std::make_shared<NaviSchedWorkloadSnapshot>();
    input.workload->revision = 7;
    autil::DataBuffer original;
    input.serialize(original);
    input.deserializeBeginMonoUs = std::numeric_limits<int64_t>::max();
    input.deserializeEndMonoUs = std::numeric_limits<int64_t>::max();
    autil::DataBuffer traced;
    input.serialize(traced);
    ASSERT_EQ(std::string(original.getData(), original.getDataLen()),
              std::string(traced.getData(), traced.getDataLen()));

    AgentUpdateToMasterData output;
    const auto before = autil::TimeUtility::monotonicTimeUs();
    output.deserialize(traced);
    const auto after = autil::TimeUtility::monotonicTimeUs();
    EXPECT_LE(before, output.deserializeBeginMonoUs);
    EXPECT_LE(output.deserializeBeginMonoUs, output.deserializeEndMonoUs);
    EXPECT_LE(output.deserializeEndMonoUs, after);
    EXPECT_EQ(123, output.eventTimeUs);
    ASSERT_TRUE(output.workload);
    EXPECT_EQ(7, output.workload->revision);
}

TEST_F(NaviSchedDataTest, testRpcTimeoutReservesTimeForEntryError)
{
    EXPECT_EQ(0, getNaviSchedRpcTimeoutMs(0));
    EXPECT_EQ(1, getNaviSchedRpcTimeoutMs(1));
    EXPECT_EQ(1, getNaviSchedRpcTimeoutMs(2));
    EXPECT_EQ(9, getNaviSchedRpcTimeoutMs(10));
    EXPECT_EQ(90, getNaviSchedRpcTimeoutMs(100));
    EXPECT_EQ(970, getNaviSchedRpcTimeoutMs(1000));
}

namespace {

uint64_t hashString(const std::string &value)
{
    return autil::HashAlgorithm::hashString64(value.data(), value.size());
}

NaviSchedLatencyOptimizerState makeOptimizerState(size_t parameterCount, int64_t step = 1,
                                                  double offset = 0.0)
{
    NaviSchedLatencyOptimizerState state;
    state.adamMoment1.resize(parameterCount);
    state.adamMoment2.resize(parameterCount);
    for (size_t index = 0; index < parameterCount; ++index) {
        state.adamMoment1[index] = offset + index * 0.01;
        state.adamMoment2[index] = offset + index * 0.02 + 0.1;
    }
    state.step = step;
    return state;
}

struct RequestTokenCounts {
    int64_t cacheHitTokenCount = 0;
    int64_t cacheMissTokenCount = 0;
};

RequestTokenCounts makeLatencyFeatures(int64_t totalTokenCount, int64_t reuseTokenCount)
{
    totalTokenCount = std::max<int64_t>(0, totalTokenCount);
    reuseTokenCount = std::max<int64_t>(0, std::min(reuseTokenCount, totalTokenCount));
    return {reuseTokenCount, totalTokenCount - reuseTokenCount};
}

NaviSchedLatencySample makeLatencySample(const std::vector<RequestTokenCounts> &queries,
                                         double latencyMs)
{
    NaviSchedLatencySample sample;
    sample.features.appendBatchCount(1);
    for (const auto &query : queries) {
        sample.features.appendRequest(query.cacheHitTokenCount,
                                      query.cacheMissTokenCount);
    }
    sample.latencyMs = latencyMs;
    return sample;
}

double predictBatch(const NonLinearPrefillModelR &model,
                    const std::vector<RequestTokenCounts> &requests,
                    const std::vector<double> &parameters)
{
    return model.predict(makeLatencySample(requests, 0.0).features, parameters);
}

struct LearningPredictorStream : public autil::legacy::Jsonizable {
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("input", input);
        json.Jsonize("reuse", reuse);
    }

    int64_t input = 0;
    int64_t reuse = 0;
};

struct LearningPredictorRecord : public autil::legacy::Jsonizable {
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("streams", streams);
        json.Jsonize("forward_ms", forwardMs);
    }

    std::vector<LearningPredictorStream> streams;
    double forwardMs = 0.0;
};

std::vector<LearningPredictorRecord> loadLearningPredictorRecords()
{
    std::ifstream input(
        "./aios/navi_sched/testdata/latency/learning_predictor_data.json");
    std::string content{std::istreambuf_iterator<char>(input),
                        std::istreambuf_iterator<char>()};
    std::vector<LearningPredictorRecord> records;
    autil::legacy::FromJsonString(records, content);
    return records;
}

std::vector<RequestTokenCounts> makeLatencyBatch(const LearningPredictorRecord &record)
{
    std::vector<RequestTokenCounts> batch;
    batch.reserve(record.streams.size());
    for (const auto &stream : record.streams) {
        batch.emplace_back(makeLatencyFeatures(stream.input, stream.reuse));
    }
    return batch;
}

std::vector<double> getJavaInitialParameters()
{
    std::vector<double> parameters = {
        -37.45139450397258,
        0.36016924559279639,
        0.051491658565753815,
        0.73254528192972401,
        0.0038151034028042394,
        0.0028430735006826171,
        0.0,
        318.15118889329102,
        0.004686616538115581,
        0.0,
    };
    for (size_t index = 0; index < parameters.size(); ++index) {
        parameters[index] /=
            NonLinearPrefillModelR::getParameterScale(index);
    }
    return parameters;
}

// Independent scalar calculus for the ten-coordinate contract. No production
// prediction, input builder, AutoGrad, fit, or projection is used here.
struct ReferenceLogAdam {
    std::vector<double> weights = getJavaInitialParameters();
    std::array<double, 10> first{}, second{}, gradient{}, candidate{};
    int64_t step = 1;

    double predict(const NaviSchedLatencyFeatures &features,
                   std::array<double, 10> *jacobian = nullptr) const
    {
        const auto &scales = NonLinearPrefillModelR::PARAMETER_SCALES;
        const std::array<double, 7> basis = {
            1.0, double(features.getBatchSize()),
            features.getCacheHitTokenCount() / 1024.0,
            features.getCacheMissTokenCount() / 1024.0,
            features.getCacheMissTokenSquareSum() / (1024.0 * 1024.0),
            features.getCacheHitMissTokenProductSum() / (1024.0 * 1024.0),
            double(features.getBatchCount()),
        };
        double u = 0.0;
        for (size_t i = 0; i < basis.size(); ++i) {
            u += basis[i] * weights[i] * scales[i];
        }
        u /= 320.0;
        const double z = weights[0] * scales[0] / 320.0;
        const double k = weights[8] * scales[8];
        const auto shape = [k](double value) {
            const double root = std::sqrt(value * value + k);
            return value >= 0.0 ? value + root : k / (root - value);
        };
        const double root = std::sqrt(u * u + k);
        const double zeroRoot = std::sqrt(z * z + k);
        const double total = basis[2] + basis[3];
        const double result = weights[7] * scales[7] +
            1500.0 * (shape(u) - shape(z)) + weights[9] * scales[9] * total;
        if (jacobian) {
            jacobian->fill(0.0);
            if (result >= 0.0) {
                for (size_t i = 0; i < basis.size(); ++i) {
                    (*jacobian)[i] = 1500.0 * (1.0 + u / root) * basis[i] * scales[i] / 320.0;
                }
                (*jacobian)[0] -= 1500.0 * (1.0 + z / zeroRoot) * scales[0] / 320.0;
                (*jacobian)[7] = scales[7];
                (*jacobian)[8] = 750.0 * (1.0 / root - 1.0 / zeroRoot) * scales[8];
                (*jacobian)[9] = scales[9] * total;
            }
        }
        return std::max(0.0, result);
    }

    void fit(const std::vector<NaviSchedLatencySample> &samples)
    {
        gradient.fill(0.0);
        for (const auto &sample : samples) {
            std::array<double, 10> jacobian;
            const auto prediction = predict(sample.features, &jacobian);
            const double derivative = (std::log1p(prediction) - std::log1p(sample.latencyMs)) / (1.0 + prediction);
            for (size_t i = 0; i < weights.size(); ++i) {
                gradient[i] += derivative * jacobian[i];
            }
        }
        const double correction = 0.05 * std::sqrt(1.0 - std::pow(0.95, step)) / (1.0 - std::pow(0.9, step));
        for (size_t i = 0; i < weights.size(); ++i) {
            gradient[i] /= samples.size();
            first[i] = 0.9 * first[i] + (1.0 - 0.9) * gradient[i];
            second[i] = 0.95 * second[i] + (1.0 - 0.95) * gradient[i] * gradient[i];
            weights[i] -= correction * first[i] / std::sqrt(second[i] + 1e-20);
            candidate[i] = weights[i];
            const double floor = i == 8 ? 1e-8 / NonLinearPrefillModelR::PARAMETER_SCALES[8] : 0.0;
            if (i != 0 && weights[i] < floor) {
                weights[i] = floor;
                if (first[i] > 0.0) {
                    first[i] = 0.0;
                }
            }
        }
        ++step;
    }

    void expectState(const NonLinearPrefillModelR &model,
                     const std::vector<double> &actual) const
    {
        const auto state = model.getOptimizerState();
        const auto diagnostics = model.getOptimizerStepDiagnostics();
        EXPECT_EQ(step, state.step);
        ASSERT_EQ(weights.size(), actual.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            EXPECT_NEAR(weights[i], actual[i], 1e-12) << "weight " << i;
            EXPECT_NEAR(first[i], state.adamMoment1[i], 1e-12) << "moment1 " << i;
            EXPECT_NEAR(second[i], state.adamMoment2[i], 1e-12) << "moment2 " << i;
            EXPECT_NEAR(gradient[i], diagnostics.rawGradient[i], 1e-12) << "gradient " << i;
        }
    }
};

} // namespace

TEST_F(NaviSchedDataTest, testIndependentLogAdamJacobian)
{
    ReferenceLogAdam reference;
    const auto sample = makeLatencySample(
        {makeLatencyFeatures(4096, 1024), makeLatencyFeatures(16384, 2048)}, 430.0);
    std::array<double, 10> jacobian;
    const double prediction = reference.predict(sample.features, &jacobian);
    EXPECT_NEAR(prediction, NonLinearPrefillModelR::predict(sample.features, reference.weights), 1e-12);
    for (size_t i = 0; i < reference.weights.size(); ++i) {
        const double initial = reference.weights[i];
        const double epsilon = 1e-4;
        reference.weights[i] = initial + epsilon;
        const auto plus = reference.predict(sample.features);
        reference.weights[i] = initial - epsilon;
        const auto minus = reference.predict(sample.features);
        reference.weights[i] = initial + 2.0 * epsilon;
        const auto plusTwo = reference.predict(sample.features);
        reference.weights[i] = initial - 2.0 * epsilon;
        const auto minusTwo = reference.predict(sample.features);
        reference.weights[i] = initial;
        EXPECT_NEAR(jacobian[i], (-plusTwo + 8.0 * plus - 8.0 * minus + minusTwo) / (12.0 * epsilon), 1e-6) << i;
    }
}

TEST_F(NaviSchedDataTest, testClientDataSerialize)
{
    ClientRequestToMasterData data;
    data.bizName = "target_biz";
    data.timeoutMs = 1234;
    data.promptTokenCount = 3;
    data.requestId = 1;
    data.collectReplay = true;
    data.forwardPayload = std::string("pb\0payload", 10);
    data.deserializeBeginTimeUs = 123;
    data.deserializeEndTimeUs = 456;
    data.blockHashes = std::make_shared<const NaviSchedBlockHashes>(
        NaviSchedBlockHashes{11, UINT64_MAX});

    autil::DataBuffer buffer;
    data.serialize(buffer);
    ClientRequestToMasterData output;
    output.deserialize(buffer);

    ASSERT_EQ(data.requestId, output.requestId);
    ASSERT_EQ(NAVI_SCHED_DATA_VERSION, output.version);
    ASSERT_EQ(data.bizName, output.bizName);
    ASSERT_EQ(data.timeoutMs, output.timeoutMs);
    ASSERT_EQ(data.promptTokenCount, output.promptTokenCount);
    ASSERT_EQ(data.collectReplay, output.collectReplay);
    ASSERT_EQ(data.forwardPayload, output.forwardPayload);
    ASSERT_EQ(*data.blockHashes, *output.blockHashes);
    ASSERT_EQ(0, output.deserializeBeginTimeUs);
    ASSERT_EQ(0, output.deserializeEndTimeUs);
    autil::DataBuffer untracedBuffer;
    data.deserializeBeginTimeUs = 0;
    data.deserializeEndTimeUs = 0;
    data.serialize(untracedBuffer);
    autil::DataBuffer tracedBuffer;
    data.deserializeBeginTimeUs = 789;
    data.deserializeEndTimeUs = 987;
    data.serialize(tracedBuffer);
    ASSERT_EQ(untracedBuffer.getDataLen(), tracedBuffer.getDataLen());
    ASSERT_EQ(std::string(untracedBuffer.getData(), untracedBuffer.getDataLen()),
              std::string(tracedBuffer.getData(), tracedBuffer.getDataLen()));
}

TEST_F(NaviSchedDataTest, testAgentResponseBatchSerialize)
{
    AgentResponseToClientData input;
    input.requestId = 17;
    input.kind = AgentResponseKind::OUTPUT;
    input.frames = {std::string("frame\0one", 9), "frame-two"};
    input.terminal = true;
    input.requestTiming.batchWaitMs = 1.25;
    input.requestTiming.queueWaitMs = 2.5;
    input.requestTiming.executionLatencyMs = 3.75;

    autil::DataBuffer buffer;
    input.serialize(buffer);
    AgentResponseToClientData output;
    output.deserialize(buffer);

    ASSERT_EQ(17, output.requestId);
    ASSERT_EQ(AgentResponseKind::OUTPUT, output.kind);
    ASSERT_EQ(input.frames, output.frames);
    ASSERT_TRUE(output.terminal);
    ASSERT_TRUE(output.errorMessage.empty());
    ASSERT_DOUBLE_EQ(0.0, output.requestTiming.batchWaitMs);
    ASSERT_DOUBLE_EQ(0.0, output.requestTiming.queueWaitMs);
    ASSERT_DOUBLE_EQ(0.0, output.requestTiming.executionLatencyMs);
    ASSERT_TRUE(output.valid());
}

TEST_F(NaviSchedDataTest, testAgentResponseSerializesOnlyKindPayload)
{
    AgentResponseToClientData ack;
    ack.requestId = 18;
    ack.kind = AgentResponseKind::ACK;
    ack.terminal = true;
    ack.requestTiming.queueWaitMs = 2.5;
    autil::DataBuffer ackBuffer;
    ack.serialize(ackBuffer);
    AgentResponseToClientData decodedAck;
    decodedAck.deserialize(ackBuffer);
    ASSERT_TRUE(decodedAck.valid());
    ASSERT_TRUE(decodedAck.terminal);
    ASSERT_DOUBLE_EQ(2.5, decodedAck.requestTiming.queueWaitMs);
    ASSERT_TRUE(decodedAck.frames.empty());
    ASSERT_TRUE(decodedAck.errorMessage.empty());

    AgentResponseToClientData error;
    error.requestId = 19;
    error.kind = AgentResponseKind::ERROR;
    error.terminal = true;
    error.errorMessage = "engine failed";
    autil::DataBuffer errorBuffer;
    error.serialize(errorBuffer);
    AgentResponseToClientData decodedError;
    decodedError.deserialize(errorBuffer);
    ASSERT_TRUE(decodedError.valid());
    ASSERT_EQ("engine failed", decodedError.errorMessage);
    ASSERT_TRUE(decodedError.frames.empty());
    ASSERT_DOUBLE_EQ(0.0, decodedError.requestTiming.queueWaitMs);
}

TEST_F(NaviSchedDataTest, testClientRequestKeepsBlockHashStorage)
{
    auto blockHashes = std::make_shared<const NaviSchedBlockHashes>(
        NaviSchedBlockHashes{11, UINT64_MAX});
    ClientRequestToMasterData request;
    request.requestId = 1;
    request.bizName = "biz";
    request.promptTokenCount = 32;
    request.blockHashes = blockHashes;
    ASSERT_EQ(blockHashes.get(), request.blockHashes.get());
    ASSERT_EQ(blockHashes->data(), request.blockHashes->data());
}

TEST_F(NaviSchedDataTest, testRtpMatchableBlockCountKeepsAlignedTail)
{
    auto threeHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22, 33});
    auto twoHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});

    EXPECT_EQ(2, BlockHashUtil::getMatchableBlockCount(48, 16, threeHashes));
    EXPECT_EQ(3, BlockHashUtil::getMatchableBlockCount(49, 16, threeHashes));
    EXPECT_EQ(2, BlockHashUtil::getMatchableBlockCount(48, 16, twoHashes));
    EXPECT_EQ(0, BlockHashUtil::getMatchableBlockCount(0, 16, nullptr));
}

TEST_F(NaviSchedDataTest, testRequestValidationRejectsInvalidTokenCounts)
{
    NaviSchedScheduleRequest scheduleRequest;
    scheduleRequest.requestId = 1;
    scheduleRequest.bizName = "biz";
    scheduleRequest.promptTokenCount = 512;
    scheduleRequest.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    scheduleRequest.timeoutMs = 1000;
    ASSERT_TRUE(scheduleRequest.valid());
    scheduleRequest.promptTokenCount = -1;
    ASSERT_FALSE(scheduleRequest.valid());
    scheduleRequest.promptTokenCount = 511;
    ASSERT_FALSE(scheduleRequest.valid());

    NaviSchedBatchRequest batchRequest;
    batchRequest.requestId = 1;
    batchRequest.totalTokenCount = 256;
    batchRequest.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11});
    ASSERT_TRUE(batchRequest.valid());
    batchRequest.latencyTrainingPhase =
        static_cast<NaviSchedLatencyTrainingPhase>(127);
    ASSERT_FALSE(batchRequest.valid());
    batchRequest.latencyTrainingPhase = NaviSchedLatencyTrainingPhase::NORMAL;
    batchRequest.totalTokenCount = -1;
    ASSERT_FALSE(batchRequest.valid());

    NaviSchedRequestStats stats;
    stats.requestId = 1;
    stats.totalTokenCount = 256;
    stats.cacheHitTokenCount = 256;
    ASSERT_TRUE(stats.valid());
    stats.latencyTrainingPhase =
        static_cast<NaviSchedLatencyTrainingPhase>(127);
    ASSERT_FALSE(stats.valid());
    stats.latencyTrainingPhase = NaviSchedLatencyTrainingPhase::NORMAL;
    stats.cacheHitTokenCount = 257;
    ASSERT_FALSE(stats.valid());
}

TEST_F(NaviSchedDataTest, testMasterDataSerialize)
{
    MasterResponseToClientData data;
    data.requestId = 1;
    data.scheduleId = 7;
    data.scheduleLoss = 1.25;
    data.scheduleLatencyUs = 125;
    data.scheduleTimeUs = 123456789;
    data.batchId = 100;
    data.expectedBatchSize = 3;
    data.predictedQueueLatencyMs = 12.5;
    data.predictedExecutionLatencyMs = 34.5;
    data.latencyTrainingPhase = NaviSchedLatencyTrainingPhase::RAMP;
    auto latencyPrediction = std::make_shared<NaviSchedLatencyPrediction>();
    ASSERT_TRUE(latencyPrediction->features.appendBatchCount(1));
    latencyPrediction->features.appendRequest(128, 384);
    ASSERT_TRUE(latencyPrediction->floorFeatures.appendBatchCount(1));
    latencyPrediction->floorFeatures.appendRequest(64, 448);
    latencyPrediction->fixedLatencyMs = -7.25;
    latencyPrediction->hasFloor = true;
    latencyPrediction->recomputable = true;
    data.latencyPrediction = latencyPrediction;
    data.reason = "not_scheduled";
    data.requestTiming = std::make_shared<NaviSchedRequestTiming>();
    data.requestTiming->batchWaitMs = 3.0;
    data.requestTiming->queueWaitMs = 4.0;
    data.requestTiming->executionLatencyMs = 5.0;
    data.requestTiming->clientTotalLatencyMs = 6.0;
    data.target.bizName = "biz";
    data.target.address = "127.0.0.1:1234";
    data.latencyModelGlobal.masterEpoch = 1001;
    data.latencyModelGlobal.version = 7;
    data.latencyModelGlobal.formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    data.latencyModelGlobal.binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    data.latencyModelGlobal.parameterCount =
        NonLinearPrefillModelR::CAPACITY_SPLIT_PARAMETER_COUNT;
    data.latencyModelGlobal.globalZeroPoint = 12.5;

    autil::DataBuffer buffer;
    data.serialize(buffer);
    MasterResponseToClientData output;
    output.deserialize(buffer);

    ASSERT_EQ(data.requestId, output.requestId);
    ASSERT_EQ(data.scheduleId, output.scheduleId);
    ASSERT_DOUBLE_EQ(data.scheduleLoss, output.scheduleLoss);
    ASSERT_EQ(data.scheduleLatencyUs, output.scheduleLatencyUs);
    ASSERT_EQ(data.scheduleTimeUs, output.scheduleTimeUs);
    ASSERT_EQ(NAVI_SCHED_DATA_VERSION, output.version);
    ASSERT_EQ(data.batchId, output.batchId);
    ASSERT_EQ(data.expectedBatchSize, output.expectedBatchSize);
    ASSERT_DOUBLE_EQ(data.predictedQueueLatencyMs, output.predictedQueueLatencyMs);
    ASSERT_DOUBLE_EQ(data.predictedExecutionLatencyMs,
                     output.predictedExecutionLatencyMs);
    ASSERT_EQ(NaviSchedLatencyTrainingPhase::RAMP,
              output.latencyTrainingPhase);
    ASSERT_TRUE(output.latencyPrediction);
    EXPECT_EQ(1, output.latencyPrediction->features.getBatchCount());
    EXPECT_EQ(128, output.latencyPrediction->features.getCacheHitTokenCount());
    EXPECT_EQ(384, output.latencyPrediction->features.getCacheMissTokenCount());
    EXPECT_EQ(64, output.latencyPrediction->floorFeatures.getCacheHitTokenCount());
    EXPECT_EQ(448, output.latencyPrediction->floorFeatures.getCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(-7.25, output.latencyPrediction->fixedLatencyMs);
    EXPECT_TRUE(output.latencyPrediction->hasFloor);
    EXPECT_TRUE(output.latencyPrediction->recomputable);
    ASSERT_EQ(data.reason, output.reason);
    ASSERT_TRUE(output.requestTiming);
    ASSERT_EQ(data.requestTiming->executionLatencyMs,
              output.requestTiming->executionLatencyMs);
    ASSERT_EQ("127.0.0.1:1234", output.target.address);
    ASSERT_TRUE(output.latencyModelGlobal.valid());
    EXPECT_EQ(data.latencyModelGlobal.masterEpoch, output.latencyModelGlobal.masterEpoch);
    EXPECT_EQ(data.latencyModelGlobal.version, output.latencyModelGlobal.version);
    EXPECT_EQ(data.latencyModelGlobal.formula, output.latencyModelGlobal.formula);
    EXPECT_EQ(data.latencyModelGlobal.binaryVersion,
              output.latencyModelGlobal.binaryVersion);
    EXPECT_EQ(data.latencyModelGlobal.parameterCount,
              output.latencyModelGlobal.parameterCount);
    EXPECT_DOUBLE_EQ(data.latencyModelGlobal.globalZeroPoint,
                     output.latencyModelGlobal.globalZeroPoint);

    auto copy = data.makeCopy();
    ASSERT_TRUE(copy);
    ASSERT_EQ(data.latencyPrediction, copy->latencyPrediction);
    EXPECT_EQ(NaviSchedLatencyTrainingPhase::RAMP,
              copy->latencyTrainingPhase);
    EXPECT_EQ(data.latencyModelGlobal.version, copy->latencyModelGlobal.version);
    EXPECT_DOUBLE_EQ(data.latencyModelGlobal.globalZeroPoint,
                     copy->latencyModelGlobal.globalZeroPoint);
    auto forwarded = data.makeForwardResponse();
    ASSERT_TRUE(forwarded);
    EXPECT_EQ(NaviSchedLatencyTrainingPhase::RAMP,
              forwarded->latencyTrainingPhase);
    EXPECT_FALSE(forwarded->latencyModelGlobal.valid());
}

TEST_F(NaviSchedDataTest, testLatencyModelUpdateCarriesInflightSnapshot)
{
    AgentUpdateToMasterData update;
    update.event = NaviSchedAgentEvent::LATENCY_MODEL_UPDATED;
    update.latency = std::make_shared<AgentLatencyToMasterData>();
    update.latency->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    update.latency->binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    update.latency->modelRevision = 19;
    update.latency->parameters.assign(NonLinearPrefillModelR::PARAMETER_COUNT, 1.0);
    update.latency->optimizerState =
        makeOptimizerState(update.latency->parameters.size(), 20);
    update.latency->zeroPoint =
        NonLinearPrefillModelR::calculateZeroPoint(update.latency->parameters);
    update.requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    update.requestEvent->inflightStatsRevision = 23;
    update.requestEvent->inflightRequests = 3;
    update.requestEvent->inflightPredictedLatencySumMs = 45.0;
    ASSERT_TRUE(update.valid());

    update.requestEvent.reset();
    EXPECT_FALSE(update.valid());
}

TEST_F(NaviSchedDataTest, testAgentUpdateSerialize)
{
    auto control = std::make_shared<AgentControlToMasterData>();
    control->serverInfo.port = 18000;
    control->agentStartTimeUs = 100;
    auto latency = std::make_shared<AgentLatencyToMasterData>();
    latency->formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    latency->binaryVersion = 1;
    latency->modelRevision = 7;
    latency->parameters.assign(NonLinearPrefillModelR::PARAMETER_COUNT, 1.0);
    latency->optimizerState = makeOptimizerState(latency->parameters.size(), 8);
    latency->residualParameters[3] = 12.5;
    latency->residualLastUpdateTimeUs = 117;
    latency->zeroPoint = NonLinearPrefillModelR::calculateZeroPoint(latency->parameters);
    latency->sessionSampleCount = 29;
    AgentUpdateToMasterData message;
    message.event = NaviSchedAgentEvent::AGENT_HANDSHAKE;
    message.eventTimeUs = 123;
    message.control = control;
    message.requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    message.requestEvent->inflightStatsRevision = 17;
    message.requestEvent->inflightRequests = 2;
    message.requestEvent->inflightPredictedLatencySumMs = 45.5;
    message.latency = latency;
    message.latencyModelFingerprint = "{\"framework\":\"vllm\"}";
    message.latencyModelTrusted = true;
    message.latencyModelWarmStarted = true;

    autil::DataBuffer buffer;
    message.serialize(buffer);
    auto wire = std::string(buffer.getData(), buffer.getDataLen());
    AgentUpdateToMasterData output;
    output.deserialize(buffer);

    ASSERT_EQ(message.event, output.event);
    ASSERT_EQ(message.eventTimeUs, output.eventTimeUs);
    ASSERT_TRUE(output.control);
    ASSERT_TRUE(output.requestEvent);
    ASSERT_EQ(17, output.requestEvent->inflightStatsRevision);
    ASSERT_EQ(2, output.requestEvent->inflightRequests);
    ASSERT_DOUBLE_EQ(45.5, output.requestEvent->inflightPredictedLatencySumMs);
    ASSERT_FALSE(output.cache);
    ASSERT_TRUE(output.latency);
    ASSERT_EQ(control->serverInfo.port, output.control->serverInfo.port);
    ASSERT_EQ(control->agentStartTimeUs, output.control->agentStartTimeUs);
    ASSERT_EQ(7, output.latency->modelRevision);
    ASSERT_EQ(latency->parameters, output.latency->parameters);
    ASSERT_EQ(latency->optimizerState.adamMoment1,
              output.latency->optimizerState.adamMoment1);
    ASSERT_EQ(latency->optimizerState.adamMoment2,
              output.latency->optimizerState.adamMoment2);
    ASSERT_EQ(latency->optimizerState.step, output.latency->optimizerState.step);
    ASSERT_EQ(latency->residualVersion, output.latency->residualVersion);
    ASSERT_EQ(latency->residualParameters,
              output.latency->residualParameters);
    ASSERT_EQ(latency->residualLastUpdateTimeUs,
              output.latency->residualLastUpdateTimeUs);
    ASSERT_DOUBLE_EQ(latency->zeroPoint, output.latency->zeroPoint);
    ASSERT_EQ(latency->sessionSampleCount, output.latency->sessionSampleCount);
    ASSERT_EQ(message.latencyModelFingerprint, output.latencyModelFingerprint);
    ASSERT_TRUE(output.latencyModelTrusted);
    ASSERT_TRUE(output.latencyModelWarmStarted);
    ASSERT_EQ(std::string::npos, wire.find("prefill_0"));

    message.version = NAVI_SCHED_AGENT_UPDATE_VERSION - 1;
    autil::DataBuffer oldBuffer;
    message.serialize(oldBuffer);
    AgentUpdateToMasterData oldOutput;
    oldOutput.deserialize(oldBuffer);
    ASSERT_FALSE(oldOutput.valid());
    ASSERT_EQ(NAVI_SCHED_AGENT_UPDATE_VERSION - 1, oldOutput.version);
}

TEST_F(NaviSchedDataTest, testLatencyModelWarmupRoundTrip)
{
    MasterLatencyModelWarmupToAgentData input;
    input.fingerprint = "{\"framework\":\"vllm\",\"hardware\":\"rtx\"}";
    input.formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    input.binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    input.candidates = {
        {300, std::vector<double>(NonLinearPrefillModelR::PARAMETER_COUNT, 1.0),
         makeOptimizerState(NonLinearPrefillModelR::PARAMETER_COUNT, 301, 1.0),
         LatencyResidualModel::VERSION,
         LatencyResidualModel::makeInitialParameters(), 123},
        {200, std::vector<double>(NonLinearPrefillModelR::PARAMETER_COUNT, 2.0),
         makeOptimizerState(NonLinearPrefillModelR::PARAMETER_COUNT, 201, 2.0),
         LatencyResidualModel::VERSION,
         LatencyResidualModel::makeInitialParameters(), 456},
    };
    ASSERT_TRUE(input.valid());

    autil::DataBuffer buffer;
    input.serialize(buffer);
    MasterLatencyModelWarmupToAgentData output;
    output.deserialize(buffer);
    ASSERT_TRUE(output.valid());
    EXPECT_EQ(input.fingerprint, output.fingerprint);
    EXPECT_EQ(input.formula, output.formula);
    EXPECT_EQ(input.binaryVersion, output.binaryVersion);
    ASSERT_EQ(2, output.candidates.size());
    EXPECT_EQ(300, output.candidates[0].sourceRevision);
    EXPECT_EQ(input.candidates[1].parameters, output.candidates[1].parameters);
    EXPECT_EQ(input.candidates[0].optimizerState.adamMoment1,
              output.candidates[0].optimizerState.adamMoment1);
    EXPECT_EQ(input.candidates[1].optimizerState.adamMoment2,
              output.candidates[1].optimizerState.adamMoment2);
    EXPECT_EQ(201, output.candidates[1].optimizerState.step);
    EXPECT_EQ(input.candidates[0].residualParameters,
              output.candidates[0].residualParameters);
    EXPECT_EQ(456, output.candidates[1].residualLastUpdateTimeUs);
    const auto validCandidates = output.candidates;

    output.candidates[1].parameters.pop_back();
    EXPECT_FALSE(output.valid());

    output.candidates = validCandidates;
    output.candidates[0].residualVersion =
        NAVI_SCHED_LATENCY_RESIDUAL_VERSION + 1;
    EXPECT_FALSE(output.valid());

    output.candidates = validCandidates;
    output.candidates[0].residualParameters.pop_back();
    EXPECT_FALSE(output.valid());

    output.candidates = validCandidates;
    output.candidates[0].residualParameters[3] =
        std::numeric_limits<double>::infinity();
    EXPECT_FALSE(output.valid());
}

TEST_F(NaviSchedDataTest, testNodeLatencyModelMetadataRoundTrip)
{
    NaviSchedNodeInfo input;
    input.agentId = "agent_0";
    input.latencyModelFingerprint = R"({"framework":"vllm","hardware":"rtx"})";
    input.latencyModelTrusted = true;
    input.latencyModelWarmStarted = true;
    input.latencyResidualVersion = LatencyResidualModel::VERSION;
    input.latencyResidualParameters[4] = -17.5;
    input.latencyResidualLastUpdateTimeUs = 123456;
    input.inflightStatsRevision = 29;
    input.inflightRequests = 3;
    input.inflightPredictedLatencySumMs = 67.25;

    autil::DataBuffer buffer;
    input.serialize(buffer);
    NaviSchedNodeInfo output;
    output.deserialize(buffer);

    EXPECT_EQ(input.agentId, output.agentId);
    EXPECT_EQ(input.latencyModelFingerprint, output.latencyModelFingerprint);
    EXPECT_TRUE(output.latencyModelTrusted);
    EXPECT_TRUE(output.latencyModelWarmStarted);
    EXPECT_EQ(input.latencyResidualVersion, output.latencyResidualVersion);
    EXPECT_EQ(input.latencyResidualParameters,
              output.latencyResidualParameters);
    EXPECT_EQ(input.latencyResidualLastUpdateTimeUs,
              output.latencyResidualLastUpdateTimeUs);
    EXPECT_EQ(input.inflightStatsRevision, output.inflightStatsRevision);
    EXPECT_EQ(input.inflightRequests, output.inflightRequests);
    EXPECT_DOUBLE_EQ(input.inflightPredictedLatencySumMs,
                     output.inflightPredictedLatencySumMs);
}

TEST_F(NaviSchedDataTest, testAgentCacheStateSerialize)
{
    AgentCacheToMasterData input;
    input.totalBlocks = 1024;
    input.fullSnapshot = true;
    input.addedBlockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    input.addedFullReadyBlockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    input.addedTailReadyBlockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{22});
    input.blockSize = 256;
    input.linearStep = 16;
    input.hybridTailRequired = 1;
    input.revision = 17;
    input.cachedBlockCount = 2;
    input.hashChecksum = 11 ^ 22;
    input.fullReadyBlockCount = 2;
    input.fullReadyHashChecksum = 11 ^ 22;
    input.tailReadyBlockCount = 1;
    input.tailReadyHashChecksum = 22;
    input.rtpCachedBlockCount = 123;

    autil::DataBuffer buffer;
    input.serialize(buffer);
    AgentCacheToMasterData output;
    output.deserialize(buffer);

    ASSERT_EQ(input.totalBlocks, output.totalBlocks);
    ASSERT_EQ(input.fullSnapshot, output.fullSnapshot);
    ASSERT_EQ(*input.addedBlockHashes, *output.addedBlockHashes);
    ASSERT_EQ(*input.addedFullReadyBlockHashes, *output.addedFullReadyBlockHashes);
    ASSERT_EQ(*input.addedTailReadyBlockHashes, *output.addedTailReadyBlockHashes);
    ASSERT_EQ(input.blockSize, output.blockSize);
    ASSERT_EQ(input.linearStep, output.linearStep);
    ASSERT_EQ(input.hybridTailRequired, output.hybridTailRequired);
    ASSERT_EQ(input.revision, output.revision);
    ASSERT_EQ(input.cachedBlockCount, output.cachedBlockCount);
    ASSERT_EQ(input.hashChecksum, output.hashChecksum);
    ASSERT_EQ(input.fullReadyBlockCount, output.fullReadyBlockCount);
    ASSERT_EQ(input.fullReadyHashChecksum, output.fullReadyHashChecksum);
    ASSERT_EQ(input.tailReadyBlockCount, output.tailReadyBlockCount);
    ASSERT_EQ(input.tailReadyHashChecksum, output.tailReadyHashChecksum);
    ASSERT_EQ(input.rtpCachedBlockCount, output.rtpCachedBlockCount);
}

TEST_F(NaviSchedDataTest, testAgentUpdateUsesHashesOnWire)
{
    auto requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    requestEvent->requestId = hashString("request_sensitive_value");
    requestEvent->inflightStatsRevision = 31;
    requestEvent->inflightRequests = 6;
    requestEvent->inflightPredictedLatencySumMs = 78.5;
    requestEvent->batchWaitingRequests = 7;
    requestEvent->queuedBatchRequests = 5;
    requestEvent->runningRequests = 3;
    requestEvent->completedRequestCount = 4;
    requestEvent->batchTimeoutCount = 9;
    requestEvent->abnormalBatchFinishCount = 3;
    requestEvent->abnormalBatchFinished = true;
    requestEvent->latencyMs = 12.5;
    requestEvent->queueTotalPredictedLatencyMs = 45.5;
    requestEvent->latencyValidation = true;
    requestEvent->localBatch.scheduleBatchId = 456;
    requestEvent->localBatch.localBatchId = 12;
    requestEvent->localBatch.requestCount = 2;
    requestEvent->localBatch.pendingCacheTracked = true;
    requestEvent->localBatch.requestIds = {requestEvent->requestId, 789};
    requestEvent->agentPredictedExecutionLatencyMs = 11.25;
    requestEvent->backendQueueLatencyMs = 2.5;
    requestEvent->latencyValidationStats.sampleCount = 8;
    requestEvent->latencyValidationStats.agentExecutionMapePct = 1.75;
    requestEvent->latencyValidationStats.agentQueueMaeMs = 2.25;
    requestEvent->latencyValidationStats.queueModelWeights = {1.0, 2.0, 3.0};
    requestEvent->latencyValidationStats.queueModelFitStep = 17;
    requestEvent->latencyValidationStats.executionLatencyBuckets.resize(7);
    auto &bucket = requestEvent->latencyValidationStats.executionLatencyBuckets[6];
    bucket.sampleCount = 3;
    bucket.masterExecutionMaeMs = 4.5;
    bucket.masterExecutionMapePct = 5.5;
    bucket.agentExecutionMaeMs = 6.5;
    bucket.agentExecutionMapePct = 7.5;
    AgentUpdateToMasterData message;
    message.event = NaviSchedAgentEvent::REQUEST_RECEIVED;
    message.eventTimeUs = 123;
    message.requestEvent = requestEvent;

    autil::DataBuffer buffer;
    message.serialize(buffer);
    auto wire = std::string(buffer.getData(), buffer.getDataLen());
    AgentUpdateToMasterData output;
    output.deserialize(buffer);

    ASSERT_TRUE(output.valid());
    ASSERT_EQ(std::string::npos, wire.find("prefill_0"));
    ASSERT_EQ(std::string::npos, wire.find("request_sensitive_value"));
    ASSERT_TRUE(output.requestEvent);
    ASSERT_EQ(requestEvent->requestId, output.requestEvent->requestId);
    ASSERT_EQ(NaviSchedAgentEvent::REQUEST_RECEIVED, output.event);
    ASSERT_EQ(31, output.requestEvent->inflightStatsRevision);
    ASSERT_EQ(6, output.requestEvent->inflightRequests);
    ASSERT_DOUBLE_EQ(78.5, output.requestEvent->inflightPredictedLatencySumMs);
    ASSERT_EQ(7, output.requestEvent->batchWaitingRequests);
    ASSERT_EQ(5, output.requestEvent->queuedBatchRequests);
    ASSERT_EQ(3, output.requestEvent->runningRequests);
    ASSERT_EQ(4, output.requestEvent->completedRequestCount);
    ASSERT_EQ(9, output.requestEvent->batchTimeoutCount);
    ASSERT_EQ(3, output.requestEvent->abnormalBatchFinishCount);
    ASSERT_TRUE(output.requestEvent->abnormalBatchFinished);
    ASSERT_DOUBLE_EQ(12.5, output.requestEvent->latencyMs);
    ASSERT_DOUBLE_EQ(45.5, output.requestEvent->queueTotalPredictedLatencyMs);
    ASSERT_TRUE(output.requestEvent->latencyValidation);
    ASSERT_EQ(456, output.requestEvent->localBatch.scheduleBatchId);
    ASSERT_EQ(12, output.requestEvent->localBatch.localBatchId);
    ASSERT_EQ(2, output.requestEvent->localBatch.requestCount);
    ASSERT_TRUE(output.requestEvent->localBatch.pendingCacheTracked);
    ASSERT_EQ((std::vector<uint64_t>{requestEvent->requestId, 789}),
              output.requestEvent->localBatch.requestIds);
    ASSERT_DOUBLE_EQ(11.25, output.requestEvent->agentPredictedExecutionLatencyMs);
    ASSERT_DOUBLE_EQ(2.5, output.requestEvent->backendQueueLatencyMs);
    ASSERT_EQ(8, output.requestEvent->latencyValidationStats.sampleCount);
    ASSERT_DOUBLE_EQ(1.75,
                     output.requestEvent->latencyValidationStats.agentExecutionMapePct);
    ASSERT_DOUBLE_EQ(2.25, output.requestEvent->latencyValidationStats.agentQueueMaeMs);
    ASSERT_EQ(std::vector<double>({1.0, 2.0, 3.0}),
              output.requestEvent->latencyValidationStats.queueModelWeights);
    ASSERT_EQ(17, output.requestEvent->latencyValidationStats.queueModelFitStep);
    ASSERT_EQ(7,
              output.requestEvent->latencyValidationStats.executionLatencyBuckets.size());
    const auto &outputBucket =
        output.requestEvent->latencyValidationStats.executionLatencyBuckets[6];
    ASSERT_EQ(3, outputBucket.sampleCount);
    ASSERT_DOUBLE_EQ(4.5, outputBucket.masterExecutionMaeMs);
    ASSERT_DOUBLE_EQ(5.5, outputBucket.masterExecutionMapePct);
    ASSERT_DOUBLE_EQ(6.5, outputBucket.agentExecutionMaeMs);
    ASSERT_DOUBLE_EQ(7.5, outputBucket.agentExecutionMapePct);
    output.requestEvent->requestId = 0;
    EXPECT_FALSE(output.valid());
}

TEST_F(NaviSchedDataTest, testExecutionLoadEventName)
{
    ASSERT_EQ(NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED,
              NaviSchedAgentEvent::fromName("execution_load_updated"));
    ASSERT_EQ("execution_load_updated",
              NaviSchedAgentEvent::getName(NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED));
}

TEST_F(NaviSchedDataTest, testBatchRequestsAbortedRoundTripAndValidation)
{
    AgentUpdateToMasterData input;
    input.event = NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED;
    input.eventTimeUs = 123;
    input.requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    auto &event = *input.requestEvent;
    event.inflightStatsRevision = 19;
    event.inflightRequests = 3;
    event.inflightPredictedLatencySumMs = 42.25;
    event.batchWaitingRequests = 2;
    event.queuedBatchRequests = 1;
    event.runningRequests = 4;
    event.completedRequestCount = 2;
    event.abnormalBatchFinished = true;
    event.queueTotalPredictedLatencyMs = 25.0;
    event.queueDecayHeadRemainingMs = 10.0;
    event.localBatch.scheduleBatchId = 101;
    event.localBatch.localBatchId = 1001;
    event.localBatch.requestCount = 2;
    event.localBatch.pendingCacheTracked = true;
    event.localBatch.requestIds = {11, 12};
    ASSERT_TRUE(input.valid());
    EXPECT_EQ(NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED,
              NaviSchedAgentEvent::fromName("batch_requests_aborted"));
    EXPECT_EQ("batch_requests_aborted",
              NaviSchedAgentEvent::getName(NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED));

    autil::DataBuffer buffer;
    input.serialize(buffer);
    AgentUpdateToMasterData output;
    output.deserialize(buffer);
    ASSERT_TRUE(output.valid());
    ASSERT_TRUE(output.requestEvent);
    EXPECT_EQ(123, output.eventTimeUs);
    EXPECT_EQ(19, output.requestEvent->inflightStatsRevision);
    EXPECT_EQ(3, output.requestEvent->inflightRequests);
    EXPECT_DOUBLE_EQ(42.25, output.requestEvent->inflightPredictedLatencySumMs);
    EXPECT_EQ(2, output.requestEvent->completedRequestCount);
    EXPECT_EQ((std::vector<uint64_t>{11, 12}),
              output.requestEvent->localBatch.requestIds);
    EXPECT_TRUE(output.requestEvent->localBatch.pendingCacheTracked);

    output.requestEvent->localBatch.requestIds = {11, 11};
    EXPECT_FALSE(output.valid());
    output.requestEvent->localBatch.requestIds = {11, 12};
    output.requestEvent->completedRequestCount = 1;
    EXPECT_FALSE(output.valid());
    output.requestEvent->completedRequestCount = 2;
    output.requestEvent->queueDecayHeadRemainingMs = 26.0;
    EXPECT_FALSE(output.valid());
    output.requestEvent->queueDecayHeadRemainingMs = 10.0;
    output.requestEvent->inflightRequests = 0;
    EXPECT_FALSE(output.valid());
    output.requestEvent->inflightPredictedLatencySumMs = 0.0;
    output.requestEvent->inflightRequests = -1;
    EXPECT_FALSE(output.valid());
    output.requestEvent->inflightRequests = 1;
    output.requestEvent->inflightPredictedLatencySumMs =
        std::numeric_limits<double>::infinity();
    EXPECT_FALSE(output.valid());
}

TEST_F(NaviSchedDataTest, testProtocolVersionsAndPendingCacheSyncRoundTrip)
{
    EXPECT_EQ(10, NAVI_SCHED_DATA_VERSION);
    EXPECT_EQ(10, NAVI_SCHED_AGENT_UPDATE_VERSION);
    EXPECT_EQ(24, NonLinearPrefillModelR::BINARY_VERSION);
    EXPECT_EQ(2, NAVI_SCHED_LATENCY_RESIDUAL_VERSION);
    EXPECT_EQ(19, NAVI_SCHED_LATENCY_RESIDUAL_PARAMETER_COUNT);

    MasterPendingCacheSyncToAgentData input;
    input.collectorId = "collector_7";
    input.reason = "revision_gap";
    input.fullSnapshot = true;
    input.snapshotReadyBefore = true;
    input.candidateStateComputed = true;
    input.masterRevisionBefore = 17;
    input.expectedRevision = 18;
    input.rejectedRevision = 19;
    input.masterChecksumBefore = 20;
    input.candidateChecksum = 21;
    input.receivedChecksum = 22;
    input.masterBatchCountBefore = 23;
    input.candidateBatchCount = 24;
    input.receivedBatchCount = 25;
    input.masterBlockCountBefore = 26;
    input.candidateBlockCount = 27;
    input.receivedBlockCount = 28;
    input.masterTailBlockCountBefore = 29;
    input.candidateTailBlockCount = 30;
    input.receivedTailBlockCount = 31;
    autil::DataBuffer buffer;
    input.serialize(buffer);
    MasterPendingCacheSyncToAgentData output;
    output.deserialize(buffer);
    EXPECT_EQ(input.collectorId, output.collectorId);
    EXPECT_EQ(input.reason, output.reason);
    EXPECT_EQ(input.fullSnapshot, output.fullSnapshot);
    EXPECT_EQ(input.snapshotReadyBefore, output.snapshotReadyBefore);
    EXPECT_EQ(input.candidateStateComputed, output.candidateStateComputed);
    EXPECT_EQ(input.masterRevisionBefore, output.masterRevisionBefore);
    EXPECT_EQ(input.expectedRevision, output.expectedRevision);
    EXPECT_EQ(input.rejectedRevision, output.rejectedRevision);
    EXPECT_EQ(input.masterChecksumBefore, output.masterChecksumBefore);
    EXPECT_EQ(input.candidateChecksum, output.candidateChecksum);
    EXPECT_EQ(input.receivedChecksum, output.receivedChecksum);
    EXPECT_EQ(input.masterBatchCountBefore, output.masterBatchCountBefore);
    EXPECT_EQ(input.candidateBatchCount, output.candidateBatchCount);
    EXPECT_EQ(input.receivedBatchCount, output.receivedBatchCount);
    EXPECT_EQ(input.masterBlockCountBefore, output.masterBlockCountBefore);
    EXPECT_EQ(input.candidateBlockCount, output.candidateBlockCount);
    EXPECT_EQ(input.receivedBlockCount, output.receivedBlockCount);
    EXPECT_EQ(input.masterTailBlockCountBefore, output.masterTailBlockCountBefore);
    EXPECT_EQ(input.candidateTailBlockCount, output.candidateTailBlockCount);
    EXPECT_EQ(input.receivedTailBlockCount, output.receivedTailBlockCount);
}

TEST_F(NaviSchedDataTest, testWorkloadRejectsElapsedTimeWithoutRunningRequest)
{
    NaviSchedWorkloadSnapshot workload;
    workload.revision = 1;
    workload.eventTimeUs = 1;
    workload.runningElapsedTimeMs = 1.0;
    EXPECT_FALSE(workload.valid());

    workload.requestCount = 1;
    workload.runningRequestCount = 1;
    EXPECT_TRUE(workload.valid());

    workload.requestCount = 0;
    workload.runningRequestCount = 0;
    workload.runningElapsedTimeMs = 0.0;
    EXPECT_TRUE(workload.valid());
}

TEST_F(NaviSchedDataTest, testIncompatibleVersionStopsBeforePayload)
{
    constexpr int32_t OLD_DATA_VERSION = NAVI_SCHED_DATA_VERSION - 1;
    constexpr int32_t UNSUPPORTED_AGENT_UPDATE_VERSION =
        NAVI_SCHED_AGENT_UPDATE_VERSION + 1;
    constexpr uint64_t PAYLOAD_MARKER = 0x123456789abcdef0UL;

    {
        autil::DataBuffer buffer;
        buffer.write(OLD_DATA_VERSION);
        buffer.write(PAYLOAD_MARKER);
        ClientRequestToMasterData output;
        output.deserialize(buffer);
        uint64_t marker = 0;
        buffer.read(marker);
        EXPECT_EQ(OLD_DATA_VERSION, output.version);
        EXPECT_EQ(PAYLOAD_MARKER, marker);
        EXPECT_FALSE(output.valid());
    }
    {
        autil::DataBuffer buffer;
        buffer.write(OLD_DATA_VERSION);
        buffer.write(PAYLOAD_MARKER);
        NaviSchedScheduleRequest output;
        output.deserialize(buffer);
        uint64_t marker = 0;
        buffer.read(marker);
        EXPECT_EQ(OLD_DATA_VERSION, output.version);
        EXPECT_EQ(PAYLOAD_MARKER, marker);
        EXPECT_FALSE(output.valid());
    }
    {
        autil::DataBuffer buffer;
        buffer.write(UNSUPPORTED_AGENT_UPDATE_VERSION);
        buffer.write(PAYLOAD_MARKER);
        AgentUpdateToMasterData output;
        output.deserialize(buffer);
        uint64_t marker = 0;
        buffer.read(marker);
        EXPECT_EQ(UNSUPPORTED_AGENT_UPDATE_VERSION, output.version);
        EXPECT_EQ(PAYLOAD_MARKER, marker);
        EXPECT_FALSE(output.valid());
    }
    {
        constexpr int32_t OLD_AGENT_UPDATE_VERSION = NAVI_SCHED_AGENT_UPDATE_VERSION - 1;
        autil::DataBuffer buffer;
        buffer.write(OLD_AGENT_UPDATE_VERSION);
        buffer.write(PAYLOAD_MARKER);
        AgentUpdateToMasterData output;
        output.deserialize(buffer);
        uint64_t marker = 0;
        buffer.read(marker);
        EXPECT_EQ(OLD_AGENT_UPDATE_VERSION, output.version);
        EXPECT_EQ(PAYLOAD_MARKER, marker);
        EXPECT_FALSE(output.valid());
    }
    {
        NaviSchedDecision oldDecision;
        oldDecision.version = OLD_DATA_VERSION;
        autil::DataBuffer buffer;
        buffer.write(oldDecision);
        buffer.write(PAYLOAD_MARKER);
        MasterResponseToClientData output;
        output.deserialize(buffer);
        uint64_t marker = 0;
        buffer.read(marker);
        EXPECT_EQ(OLD_DATA_VERSION, output.version);
        EXPECT_EQ(PAYLOAD_MARKER, marker);
        EXPECT_EQ(0, output.requestId);
    }
    {
        autil::DataBuffer buffer;
        buffer.write(OLD_DATA_VERSION);
        buffer.write(PAYLOAD_MARKER);
        NaviSchedScheduleReplay output;
        output.deserialize(buffer);
        uint64_t marker = 0;
        buffer.read(marker);
        EXPECT_EQ(OLD_DATA_VERSION, output.version);
        EXPECT_EQ(PAYLOAD_MARKER, marker);
        EXPECT_TRUE(output.requests.empty());
    }
}

TEST_F(NaviSchedDataTest, testWorkloadUpdateRoundTripAndPayloadIsolation)
{
    auto workload = std::make_shared<NaviSchedWorkloadSnapshot>();
    workload->revision = 7;
    workload->eventTimeUs = 123456;
    workload->batchCount = 2;
    workload->requestCount = 5;
    workload->remainingPrefillRequestCount = 3;
    workload->runningRequestCount = 2;
    workload->runningElapsedTimeMs = 37.25;
    workload->runningDecodeRequestCount = 2;
    workload->runningDecodeGeneratedTokenCount = 47;
    workload->runningDecodeRemainingTokenBudget = 81;
    workload->waitingRequestCount = 3;
    workload->remainingPrefillContextTokenCount = 100;
    workload->remainingPrefillTokenCount = 200;
    workload->remainingPrefillTokenSquareSum = 30000.0;
    workload->contextRemainingPrefillTokenProductSum = 4000.0;
    workload->prefillCapacitySplit = true;
    workload->maxRunningRequestCount = 4;
    workload->runningCapacitySlotCount = 4;
    workload->runningCapacityPrefillRequestCount = 2;
    workload->runningCapacityPrefillContextTokenCount = 60;
    workload->runningCapacityPrefillTokenCount = 120;
    workload->runningCapacityPrefillTokenSquareSum = 20000.0;
    workload->runningCapacityContextRemainingPrefillTokenProductSum = 2500.0;
    workload->waitingCapacityPrefillRequestCount = 1;
    workload->waitingCapacityPrefillContextTokenCount = 40;
    workload->waitingCapacityPrefillTokenCount = 80;
    workload->waitingCapacityPrefillTokenSquareSum = 10000.0;
    workload->waitingCapacityContextRemainingPrefillTokenProductSum = 1500.0;
    workload->remainingTotalContextTokenCount = 150;
    workload->remainingTotalTokenCount = 280;
    workload->remainingTotalTokenSquareSum = 50000.0;
    workload->contextRemainingTotalTokenProductSum = 6000.0;
    workload->totalPromptTokenCount = 350;
    workload->inFlightScheduledTokenCount = 32;
    workload->admittedBatch.scheduleBatchId = 91;
    workload->admittedBatch.localBatchId = 7;
    workload->admittedBatch.requestCount = 2;
    workload->admittedBatch.pendingCacheTracked = true;
    workload->admittedBatch.requestIds = {11, 12};
    ASSERT_TRUE(workload->valid());

    AgentUpdateToMasterData input;
    input.event = NaviSchedAgentEvent::WORKLOAD_UPDATED;
    input.eventTimeUs = workload->eventTimeUs;
    input.workload = workload;
    ASSERT_TRUE(input.valid());

    autil::DataBuffer buffer;
    input.serialize(buffer);
    AgentUpdateToMasterData output;
    output.deserialize(buffer);
    ASSERT_TRUE(output.valid());
    ASSERT_TRUE(output.workload);
    ASSERT_EQ(7, output.workload->revision);
    ASSERT_EQ(2, output.workload->batchCount);
    ASSERT_EQ(5, output.workload->requestCount);
    ASSERT_EQ(3, output.workload->remainingPrefillRequestCount);
    ASSERT_DOUBLE_EQ(37.25, output.workload->runningElapsedTimeMs);
    ASSERT_EQ(2, output.workload->runningDecodeRequestCount);
    ASSERT_EQ(47, output.workload->runningDecodeGeneratedTokenCount);
    ASSERT_EQ(81, output.workload->runningDecodeRemainingTokenBudget);
    ASSERT_EQ(100, output.workload->remainingPrefillContextTokenCount);
    ASSERT_EQ(200, output.workload->remainingPrefillTokenCount);
    ASSERT_DOUBLE_EQ(30000.0, output.workload->remainingPrefillTokenSquareSum);
    ASSERT_TRUE(output.workload->prefillCapacitySplit);
    ASSERT_EQ(4, output.workload->maxRunningRequestCount);
    ASSERT_EQ(4, output.workload->runningCapacitySlotCount);
    ASSERT_EQ(2, output.workload->runningCapacityPrefillRequestCount);
    ASSERT_EQ(60, output.workload->runningCapacityPrefillContextTokenCount);
    ASSERT_EQ(120, output.workload->runningCapacityPrefillTokenCount);
    ASSERT_DOUBLE_EQ(20000.0, output.workload->runningCapacityPrefillTokenSquareSum);
    ASSERT_DOUBLE_EQ(
        2500.0, output.workload->runningCapacityContextRemainingPrefillTokenProductSum);
    ASSERT_EQ(1, output.workload->waitingCapacityPrefillRequestCount);
    ASSERT_EQ(40, output.workload->waitingCapacityPrefillContextTokenCount);
    ASSERT_EQ(80, output.workload->waitingCapacityPrefillTokenCount);
    ASSERT_DOUBLE_EQ(10000.0, output.workload->waitingCapacityPrefillTokenSquareSum);
    ASSERT_DOUBLE_EQ(
        1500.0, output.workload->waitingCapacityContextRemainingPrefillTokenProductSum);
    ASSERT_EQ(150, output.workload->remainingTotalContextTokenCount);
    ASSERT_EQ(280, output.workload->remainingTotalTokenCount);
    ASSERT_EQ(32, output.workload->inFlightScheduledTokenCount);
    ASSERT_EQ(91, output.workload->admittedBatch.scheduleBatchId);
    ASSERT_EQ(7, output.workload->admittedBatch.localBatchId);
    ASSERT_EQ(2, output.workload->admittedBatch.requestCount);
    ASSERT_TRUE(output.workload->admittedBatch.pendingCacheTracked);
    ASSERT_EQ((std::vector<uint64_t>{11, 12}), output.workload->admittedBatch.requestIds);

    input.requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    ASSERT_FALSE(input.valid());
}

TEST_F(NaviSchedDataTest, testCapacitySplitWorkloadValidationAndEquality)
{
    NaviSchedWorkloadSnapshot workload;
    workload.revision = 1;
    workload.eventTimeUs = 2;
    workload.batchCount = 2;
    workload.requestCount = 5;
    workload.remainingPrefillRequestCount = 3;
    workload.runningRequestCount = 3;
    workload.runningElapsedTimeMs = 125.5;
    workload.runningDecodeRequestCount = 1;
    workload.runningDecodeGeneratedTokenCount = 17;
    workload.runningDecodeRemainingTokenBudget = 31;
    workload.waitingRequestCount = 2;
    workload.remainingPrefillContextTokenCount = 70;
    workload.remainingPrefillTokenCount = 100;
    workload.remainingPrefillTokenSquareSum = 4600.0;
    workload.contextRemainingPrefillTokenProductSum = 2300.0;
    workload.remainingTotalContextTokenCount = 90;
    workload.remainingTotalTokenCount = 130;
    workload.remainingTotalTokenSquareSum = 7000.0;
    workload.contextRemainingTotalTokenProductSum = 3200.0;
    workload.totalPromptTokenCount = 220;
    workload.prefillCapacitySplit = true;
    workload.maxRunningRequestCount = 3;
    workload.runningCapacitySlotCount = 3;
    workload.runningCapacityPrefillRequestCount = 2;
    workload.runningCapacityPrefillContextTokenCount = 50;
    workload.runningCapacityPrefillTokenCount = 60;
    workload.runningCapacityPrefillTokenSquareSum = 2600.0;
    workload.runningCapacityContextRemainingPrefillTokenProductSum = 1500.0;
    workload.waitingCapacityPrefillRequestCount = 1;
    workload.waitingCapacityPrefillContextTokenCount = 20;
    workload.waitingCapacityPrefillTokenCount = 40;
    workload.waitingCapacityPrefillTokenSquareSum = 2000.0;
    workload.waitingCapacityContextRemainingPrefillTokenProductSum = 800.0;
    ASSERT_TRUE(workload.valid());

    auto invalidDecode = workload;
    invalidDecode.runningDecodeRequestCount =
        workload.runningRequestCount + 1;
    EXPECT_FALSE(invalidDecode.valid());
    invalidDecode = workload;
    invalidDecode.runningDecodeGeneratedTokenCount = 0;
    EXPECT_FALSE(invalidDecode.valid());
    invalidDecode = workload;
    invalidDecode.runningDecodeRequestCount = 0;
    EXPECT_FALSE(invalidDecode.valid());
    invalidDecode = workload;
    invalidDecode.runningDecodeRemainingTokenBudget = -1;
    EXPECT_FALSE(invalidDecode.valid());

    NaviSchedWorkloadSnapshot pdSeparatedWorkload;
    pdSeparatedWorkload.revision = 1;
    pdSeparatedWorkload.eventTimeUs = 2;
    pdSeparatedWorkload.runningRequestCount = 1;
    pdSeparatedWorkload.runningDecodeRequestCount = 1;
    pdSeparatedWorkload.runningDecodeGeneratedTokenCount = 1;
    pdSeparatedWorkload.runningDecodeRemainingTokenBudget = 1;
    EXPECT_FALSE(pdSeparatedWorkload.valid());

    NaviSchedLatencyFeatures workloadFeatures;
    ASSERT_TRUE(workloadFeatures.appendPrefillWorkload(workload));
    EXPECT_EQ(5, workloadFeatures.getPrecedingRequestCount());
    EXPECT_DOUBLE_EQ(125.5, workloadFeatures.getRunningElapsedTimeMs());
    EXPECT_EQ(1, workloadFeatures.getRunningDecodeRequestCount());
    EXPECT_EQ(17, workloadFeatures.getRunningDecodeGeneratedTokenCount());
    EXPECT_EQ(31, workloadFeatures.getRunningDecodeRemainingTokenBudget());

    auto same = workload;
    EXPECT_TRUE(workload.sameWorkload(same));
    same.runningElapsedTimeMs = 126.5;
    EXPECT_FALSE(workload.sameWorkload(same));
    same.runningElapsedTimeMs = workload.runningElapsedTimeMs;
    ++same.runningDecodeGeneratedTokenCount;
    EXPECT_FALSE(workload.sameWorkload(same));
    same.runningDecodeGeneratedTokenCount =
        workload.runningDecodeGeneratedTokenCount;
    ++same.runningDecodeRemainingTokenBudget;
    EXPECT_FALSE(workload.sameWorkload(same));
    same.runningDecodeRemainingTokenBudget =
        workload.runningDecodeRemainingTokenBudget;
    ++same.waitingCapacityPrefillTokenCount;
    EXPECT_FALSE(workload.sameWorkload(same));

    auto invalid = workload;
    invalid.runningCapacitySlotCount = 2;
    EXPECT_FALSE(invalid.valid());
    invalid = workload;
    --invalid.remainingPrefillTokenCount;
    EXPECT_FALSE(invalid.valid());
    invalid = workload;
    invalid.prefillCapacitySplit = false;
    EXPECT_FALSE(invalid.valid());

    NaviSchedWorkloadSnapshot emptySplit;
    emptySplit.revision = 1;
    emptySplit.eventTimeUs = 2;
    emptySplit.prefillCapacitySplit = true;
    EXPECT_TRUE(emptySplit.valid());
}

TEST_F(NaviSchedDataTest, testCapacitySplitLatencyFeaturesRoundTripAndArithmetic)
{
    NaviSchedLatencyFeatures first;
    ASSERT_TRUE(first.appendCapacitySplitAggregate(2, 3, 100, 200, 14000.0, 7000.0, 4,
                                                   300, 400, 44000.0, 23000.0));
    ASSERT_TRUE(first.appendPrecedingRequestCount(7));
    ASSERT_TRUE(first.appendRunningElapsedTimeMs(125.5));
    ASSERT_TRUE(first.appendRunningDecodeProgress(3, 51, 149));
    EXPECT_TRUE(first.usesCapacitySplit());
    EXPECT_EQ(2, first.getBatchCount());
    EXPECT_EQ(7, first.getPrecedingRequestCount());
    EXPECT_DOUBLE_EQ(125.5, first.getRunningElapsedTimeMs());
    EXPECT_EQ(3, first.getRunningDecodeRequestCount());
    EXPECT_EQ(51, first.getRunningDecodeGeneratedTokenCount());
    EXPECT_EQ(149, first.getRunningDecodeRemainingTokenBudget());
    EXPECT_EQ(7, first.getBatchSize());
    EXPECT_EQ(400, first.getCacheHitTokenCount());
    EXPECT_EQ(600, first.getCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(58000.0, first.getCacheMissTokenSquareSum());
    EXPECT_DOUBLE_EQ(30000.0, first.getCacheHitMissTokenProductSum());
    EXPECT_EQ(3, first.getRunningBatchSize());
    EXPECT_EQ(100, first.getRunningCacheHitTokenCount());
    EXPECT_EQ(200, first.getRunningCacheMissTokenCount());
    EXPECT_EQ(4, first.getWaitingBatchSize());
    EXPECT_EQ(300, first.getWaitingCacheHitTokenCount());
    EXPECT_EQ(400, first.getWaitingCacheMissTokenCount());

    autil::DataBuffer buffer;
    first.serialize(buffer);
    NaviSchedLatencyFeatures restored;
    restored.deserialize(buffer);
    EXPECT_TRUE(restored.usesCapacitySplit());
    EXPECT_EQ(first.getBatchCount(), restored.getBatchCount());
    EXPECT_EQ(first.getPrecedingRequestCount(), restored.getPrecedingRequestCount());
    EXPECT_DOUBLE_EQ(first.getRunningElapsedTimeMs(),
                     restored.getRunningElapsedTimeMs());
    EXPECT_EQ(first.getRunningDecodeRequestCount(),
              restored.getRunningDecodeRequestCount());
    EXPECT_EQ(first.getRunningDecodeGeneratedTokenCount(),
              restored.getRunningDecodeGeneratedTokenCount());
    EXPECT_EQ(first.getRunningDecodeRemainingTokenBudget(),
              restored.getRunningDecodeRemainingTokenBudget());
    EXPECT_EQ(first.getRunningBatchSize(), restored.getRunningBatchSize());
    EXPECT_EQ(first.getRunningCacheHitTokenCount(),
              restored.getRunningCacheHitTokenCount());
    EXPECT_EQ(first.getRunningCacheMissTokenCount(),
              restored.getRunningCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(first.getRunningCacheMissTokenSquareSum(),
                     restored.getRunningCacheMissTokenSquareSum());
    EXPECT_DOUBLE_EQ(first.getRunningCacheHitMissTokenProductSum(),
                     restored.getRunningCacheHitMissTokenProductSum());
    EXPECT_EQ(first.getWaitingBatchSize(), restored.getWaitingBatchSize());
    EXPECT_EQ(first.getWaitingCacheHitTokenCount(),
              restored.getWaitingCacheHitTokenCount());
    EXPECT_EQ(first.getWaitingCacheMissTokenCount(),
              restored.getWaitingCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(first.getWaitingCacheMissTokenSquareSum(),
                     restored.getWaitingCacheMissTokenSquareSum());
    EXPECT_DOUBLE_EQ(first.getWaitingCacheHitMissTokenProductSum(),
                     restored.getWaitingCacheHitMissTokenProductSum());

    NaviSchedLatencyFeatures delta;
    ASSERT_TRUE(delta.appendCapacitySplitAggregate(1, 1, 5, 7, 49.0, 35.0, 2, 11, 13,
                                                   85.0, 71.0));
    ASSERT_TRUE(delta.appendPrecedingRequestCount(2));
    ASSERT_TRUE(delta.appendRunningElapsedTimeMs(24.5));
    ASSERT_TRUE(delta.appendRunningDecodeProgress(1, 13, 27));
    auto combined = first;
    ASSERT_TRUE(combined.append(delta));
    EXPECT_EQ(3, combined.getBatchCount());
    EXPECT_EQ(9, combined.getPrecedingRequestCount());
    EXPECT_DOUBLE_EQ(150.0, combined.getRunningElapsedTimeMs());
    EXPECT_EQ(4, combined.getRunningDecodeRequestCount());
    EXPECT_EQ(64, combined.getRunningDecodeGeneratedTokenCount());
    EXPECT_EQ(176, combined.getRunningDecodeRemainingTokenBudget());
    EXPECT_EQ(4, combined.getRunningBatchSize());
    EXPECT_EQ(6, combined.getWaitingBatchSize());
    ASSERT_TRUE(combined.subtract(delta));
    EXPECT_EQ(first.getBatchCount(), combined.getBatchCount());
    EXPECT_EQ(first.getPrecedingRequestCount(), combined.getPrecedingRequestCount());
    EXPECT_DOUBLE_EQ(first.getRunningElapsedTimeMs(),
                     combined.getRunningElapsedTimeMs());
    EXPECT_EQ(first.getRunningDecodeRequestCount(),
              combined.getRunningDecodeRequestCount());
    EXPECT_EQ(first.getRunningDecodeGeneratedTokenCount(),
              combined.getRunningDecodeGeneratedTokenCount());
    EXPECT_EQ(first.getRunningDecodeRemainingTokenBudget(),
              combined.getRunningDecodeRemainingTokenBudget());
    EXPECT_EQ(first.getBatchSize(), combined.getBatchSize());
    EXPECT_EQ(first.getCacheHitTokenCount(), combined.getCacheHitTokenCount());
    EXPECT_EQ(first.getCacheMissTokenCount(), combined.getCacheMissTokenCount());
    EXPECT_DOUBLE_EQ(first.getCacheMissTokenSquareSum(),
                     combined.getCacheMissTokenSquareSum());
    EXPECT_DOUBLE_EQ(first.getCacheHitMissTokenProductSum(),
                     combined.getCacheHitMissTokenProductSum());

    NaviSchedLatencyFeatures aggregate;
    ASSERT_TRUE(aggregate.appendAggregate(1, 1, 1, 2, 4.0, 2.0));
    EXPECT_FALSE(first.append(aggregate));
    EXPECT_FALSE(first.subtract(aggregate));
    EXPECT_FALSE(
        delta.appendCapacitySplitAggregate(-1, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0));
    EXPECT_FALSE(delta.appendPrecedingRequestCount(-1));
    EXPECT_FALSE(delta.appendRunningElapsedTimeMs(-1.0));
    EXPECT_FALSE(delta.appendRunningDecodeProgress(-1, 0));
    EXPECT_FALSE(delta.appendRunningDecodeProgress(2, 1));
    EXPECT_FALSE(delta.appendRunningDecodeProgress(0, 0, 1));
    EXPECT_FALSE(delta.appendRunningDecodeProgress(1, 1, -1));

    NaviSchedLatencyFeatures invalidDecodeRemainder;
    ASSERT_TRUE(invalidDecodeRemainder.appendRunningDecodeProgress(2, 3, 7));
    NaviSchedLatencyFeatures highProgressSubset;
    ASSERT_TRUE(highProgressSubset.appendRunningDecodeProgress(1, 3, 7));
    EXPECT_FALSE(invalidDecodeRemainder.subtract(highProgressSubset));
    EXPECT_EQ(2, invalidDecodeRemainder.getRunningDecodeRequestCount());
    EXPECT_EQ(3,
              invalidDecodeRemainder.getRunningDecodeGeneratedTokenCount());

    NaviSchedLatencyFeatures maximum;
    ASSERT_TRUE(maximum.appendPrecedingRequestCount(std::numeric_limits<int64_t>::max()));
    EXPECT_FALSE(maximum.appendPrecedingRequestCount(1));
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), maximum.getPrecedingRequestCount());

    auto unchanged = first;
    EXPECT_FALSE(unchanged.append(maximum));
    EXPECT_EQ(first.getBatchCount(), unchanged.getBatchCount());
    EXPECT_EQ(first.getPrecedingRequestCount(), unchanged.getPrecedingRequestCount());
    EXPECT_EQ(first.getBatchSize(), unchanged.getBatchSize());
}

TEST_F(NaviSchedDataTest, testHandshakeCarriesPdModeAndLatestWorkload)
{
    AgentUpdateToMasterData input;
    input.event = NaviSchedAgentEvent::AGENT_HANDSHAKE;
    input.eventTimeUs = 123;
    input.control = std::make_shared<AgentControlToMasterData>();
    input.control->serverInfo.port = 18000;
    input.control->agentStartTimeUs = 100;
    input.control->pdSeparated = false;
    input.control->unifiedLatencyPrediction = true;
    input.requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    input.requestEvent->inflightStatsRevision = 7;
    input.requestEvent->inflightRequests = 2;
    input.requestEvent->inflightPredictedLatencySumMs = 25.0;
    input.workload = std::make_shared<NaviSchedWorkloadSnapshot>();
    input.workload->revision = 1;
    input.workload->eventTimeUs = 122;
    ASSERT_TRUE(input.valid());

    autil::DataBuffer buffer;
    input.serialize(buffer);
    AgentUpdateToMasterData output;
    output.deserialize(buffer);
    ASSERT_TRUE(output.valid());
    ASSERT_FALSE(output.control->pdSeparated);
    ASSERT_TRUE(output.control->unifiedLatencyPrediction);
    ASSERT_EQ(1, output.workload->revision);
    ASSERT_EQ(7, output.requestEvent->inflightStatsRevision);
    ASSERT_EQ(2, output.requestEvent->inflightRequests);
    ASSERT_DOUBLE_EQ(25.0, output.requestEvent->inflightPredictedLatencySumMs);

    output.control->unifiedLatencyPrediction = false;
    ASSERT_FALSE(output.valid());
    output.control->unifiedLatencyPrediction = true;

    output.event = NaviSchedAgentEvent::AGENT_STOPPING;
    ASSERT_FALSE(output.valid());
    output.workload.reset();
    output.requestEvent.reset();
    ASSERT_TRUE(output.valid());
}

TEST_F(NaviSchedDataTest, testSummaryLevel)
{
    ASSERT_EQ(1, SummaryUtil::getSummaryLevel(""));
    ASSERT_EQ(1, SummaryUtil::getSummaryLevel("invalid"));
    ASSERT_EQ(0, SummaryUtil::getSummaryLevel("-1"));
    ASSERT_EQ(2, SummaryUtil::getSummaryLevel("2"));
    ASSERT_EQ(64, SummaryUtil::getSummaryLimit(2, 100));
    ASSERT_EQ(100, SummaryUtil::getSummaryLimit(3, 100));
    ClientRequestToMasterData data;
    data.requestId = 1;
    data.promptTokenCount = 4;
    data.blockHashes =
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    navi::SummaryBuilder basicBuilder;
    ASSERT_EQ(navi::SEC_OK, data.getSummary("0", basicBuilder));
    auto basicSummary = basicBuilder.toString();
    ASSERT_NE(std::string::npos, basicSummary.find("ClientRequestToMasterData"));
    ASSERT_EQ(std::string::npos, basicSummary.find("ClientBlockHashes"));

    navi::SummaryBuilder detailBuilder;
    ASSERT_EQ(navi::SEC_OK, data.getSummary("2", detailBuilder));
    auto detailSummary = detailBuilder.toString();
    ASSERT_NE(std::string::npos, detailSummary.find("ClientBlockHashes"));
}

TEST_F(NaviSchedDataTest, testAllDataSummaries)
{
    auto scheduleRequest =
        std::make_shared<MasterScheduleRequestData>(nullptr, nullptr);
    std::vector<std::pair<navi::DataPtr, std::string>> datas = {
        {std::make_shared<AgentUpdateToMasterData>(), "AgentUpdateToMasterData"},
        {std::make_shared<ClientRequestToMasterData>(), "ClientRequestToMasterData"},
        {std::make_shared<MasterResponseToClientData>(), "MasterResponseToClientData"},
        {scheduleRequest, "MasterScheduleRequestData"},
    };
    for (const auto &entry : datas) {
        for (const auto &query : {"0", "1", "2", "3"}) {
            navi::SummaryBuilder builder;
            ASSERT_EQ(navi::SEC_OK, entry.first->getSummary(query, builder));
            ASSERT_NE(std::string::npos, builder.toString().find(entry.second));
        }
    }
}

TEST_F(NaviSchedDataTest, testNonLinearPrefillModelLog1pAdamSequence)
{
    NonLinearPrefillModelR model;
    ReferenceLogAdam reference;
    auto parameters = getJavaInitialParameters();
    ASSERT_EQ(NonLinearPrefillModelR::BINARY_VERSION, model.getBinaryVersion());
    ASSERT_EQ(NonLinearPrefillModelR::PARAMETER_COUNT, parameters.size());
    ASSERT_TRUE(model.validateParameters(parameters));
    ASSERT_FALSE(model.validateParameters(
        std::vector<double>(NonLinearPrefillModelR::PARAMETER_COUNT - 1)));
    for (auto index : {NonLinearPrefillModelR::BASE_LATENCY_PARAMETER_INDEX,
                       NonLinearPrefillModelR::TOTAL_PROMPT_PARAMETER_INDEX}) {
        auto invalidParameters = parameters;
        invalidParameters[index] = -1e-12;
        ASSERT_FALSE(model.validateParameters(invalidParameters));
    }
    auto invalidCurvature = parameters;
    invalidCurvature[NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX] =
        0.0;
    ASSERT_FALSE(model.validateParameters(invalidCurvature));
    navi::SummaryBuilder basicBuilder;
    ASSERT_EQ(navi::SEC_OK, model.getSummary("1", basicBuilder));
    auto basicSummary = basicBuilder.toString();
    ASSERT_NE(std::string::npos, basicSummary.find("NonLinearPrefillModelOverview"));
    ASSERT_NE(std::string::npos, basicSummary.find("nonlinear_prefill"));
    ASSERT_EQ(std::string::npos, basicSummary.find("NonLinearPrefillOptimizer"));
    navi::SummaryBuilder detailBuilder;
    ASSERT_EQ(navi::SEC_OK, model.getSummary("2", detailBuilder));
    ASSERT_NE(std::string::npos,
              detailBuilder.toString().find("NonLinearPrefillOptimizer"));
    std::vector<std::vector<RequestTokenCounts>> batches = {
        {{makeLatencyFeatures(1613, 0), makeLatencyFeatures(1729, 0),
          makeLatencyFeatures(1126, 0), makeLatencyFeatures(2034, 1024),
          makeLatencyFeatures(471, 0), makeLatencyFeatures(16413, 0),
          makeLatencyFeatures(1448, 0), makeLatencyFeatures(921, 0),
          makeLatencyFeatures(16833, 1024), makeLatencyFeatures(4538, 0),
          makeLatencyFeatures(5, 0), makeLatencyFeatures(2130, 0),
          makeLatencyFeatures(27419, 26624)}},
        {{makeLatencyFeatures(2912, 2048), makeLatencyFeatures(4698, 0),
          makeLatencyFeatures(5170, 5120), makeLatencyFeatures(12959, 11264)}},
        {{makeLatencyFeatures(1168, 0), makeLatencyFeatures(11328, 0),
          makeLatencyFeatures(6167, 6144), makeLatencyFeatures(2154, 1024)}},
        {{makeLatencyFeatures(2801, 2048), makeLatencyFeatures(1120, 0),
          makeLatencyFeatures(485, 0), makeLatencyFeatures(739, 0),
          makeLatencyFeatures(3370, 1024), makeLatencyFeatures(6, 0),
          makeLatencyFeatures(26388, 17408)}}};
    const std::vector<double> expectedPredictions = {
        419.9161288905335,
        324.0928273291982,
        328.4824270856956,
        331.6345195219027,
    };
    std::vector<NaviSchedLatencySample> samples;
    const std::vector<double> actualLatencyMs = {430, 320, 323, 325};
    for (size_t index = 0; index < batches.size(); ++index) {
        ASSERT_NEAR(expectedPredictions[index],
                    predictBatch(model, batches[index], parameters), 1e-12);
        auto sample = makeLatencySample(batches[index], actualLatencyMs[index]);
        ASSERT_DOUBLE_EQ(predictBatch(model, batches[index], parameters),
                         model.predict(sample.features, parameters));
        samples.emplace_back(std::move(sample));
    }
    NonLinearPrefillModelR exactModel;
    auto exactInitialParameters = exactModel.getInitialParameters();
    std::vector<NaviSchedLatencySample> exactSamples;
    for (const auto &batch : batches) {
        NaviSchedLatencySample sample;
        sample = makeLatencySample(
            batch, predictBatch(exactModel, batch, exactInitialParameters));
        exactSamples.emplace_back(std::move(sample));
    }
    std::vector<double> exactParameters;
    ASSERT_TRUE(exactModel.fit(exactSamples, exactInitialParameters, exactParameters));
    ASSERT_EQ(exactInitialParameters, exactParameters);
    std::vector<double> updatedParameters;
    reference.fit(samples);
    ASSERT_TRUE(model.fit(samples, parameters, updatedParameters));
    reference.expectState(model, updatedParameters);
    EXPECT_NEAR(-0.049854683294997183, reference.candidate[5], 1e-12);
    EXPECT_DOUBLE_EQ(0.0, updatedParameters[5]);
    EXPECT_DOUBLE_EQ(0.0, model.getOptimizerState().adamMoment1[5]);
    EXPECT_GT(model.getOptimizerState().adamMoment2[5], 0.0);
    parameters = updatedParameters;
    auto javaBatches = batches;
    batches = {{{makeLatencyFeatures(560, 0), makeLatencyFeatures(1557, 1024),
                 makeLatencyFeatures(3950, 1024), makeLatencyFeatures(462, 0),
                 makeLatencyFeatures(5526, 5120), makeLatencyFeatures(11980, 5120),
                 makeLatencyFeatures(177, 0), makeLatencyFeatures(16629, 0),
                 makeLatencyFeatures(838, 0), makeLatencyFeatures(975, 0)}},
               {{makeLatencyFeatures(311, 0), makeLatencyFeatures(1031, 0),
                 makeLatencyFeatures(30744, 6144), makeLatencyFeatures(5287, 0),
                 makeLatencyFeatures(1810, 0), makeLatencyFeatures(446, 0),
                 makeLatencyFeatures(2037, 1024)}},
               {{makeLatencyFeatures(6030, 3072), makeLatencyFeatures(14669, 3072),
                 makeLatencyFeatures(3471, 3072), makeLatencyFeatures(1961, 1024),
                 makeLatencyFeatures(920, 0), makeLatencyFeatures(16780, 3072),
                 makeLatencyFeatures(74, 0), makeLatencyFeatures(77, 0)}},
               {{makeLatencyFeatures(3831, 0), makeLatencyFeatures(5646, 3072),
                 makeLatencyFeatures(1873, 0), makeLatencyFeatures(6473, 0),
                 makeLatencyFeatures(455, 0), makeLatencyFeatures(2808, 2048),
                 makeLatencyFeatures(1521, 1024), makeLatencyFeatures(7278, 0),
                 makeLatencyFeatures(6281, 0), makeLatencyFeatures(17806, 4096),
                 makeLatencyFeatures(5166, 0), makeLatencyFeatures(4310, 1024),
                 makeLatencyFeatures(343, 0)}}};
    const std::vector<double> secondActualLatencyMs = {343, 399, 334, 420};
    samples.clear();
    for (size_t index = 0; index < batches.size(); ++index) {
        EXPECT_NEAR(reference.predict(makeLatencySample(batches[index], 0.0).features),
                    predictBatch(model, batches[index], parameters), 1e-12)
            << "prediction index " << index;
        samples.emplace_back(
            makeLatencySample(batches[index], secondActualLatencyMs[index]));
    }
    reference.fit(samples);
    ASSERT_TRUE(model.fit(samples, parameters, updatedParameters));
    reference.expectState(model, updatedParameters);
    parameters = updatedParameters;
    javaBatches.insert(javaBatches.end(), batches.begin(), batches.end());
    const std::vector<double> javaActualLatencyMs = {430, 320, 323, 325,
                                                     343, 399, 334, 420};
    for (size_t round = 1; round < 8; ++round) {
        for (size_t offset = 0; offset < javaBatches.size(); offset += 4) {
            samples.clear();
            for (size_t index = 0; index < 4; ++index) {
                auto recordIndex = (round + offset + index) % javaBatches.size();
                samples.emplace_back(makeLatencySample(javaBatches[recordIndex],
                                                       javaActualLatencyMs[recordIndex]));
            }
            reference.fit(samples);
            ASSERT_TRUE(model.fit(samples, parameters, updatedParameters));
            reference.expectState(model, updatedParameters);
            parameters = updatedParameters;
        }
    }
    reference.expectState(model, parameters);
}

TEST_F(NaviSchedDataTest, testNonLinearPrefillRequestLinearCost)
{
    std::vector<double> weights = {
        100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    };
    for (size_t index = 0; index < weights.size(); ++index) {
        weights[index] /=
            NonLinearPrefillModelR::getParameterScale(index);
    }
    EXPECT_DOUBLE_EQ(
        45.0, NonLinearPrefillModelR::calculateRequestLinearCost(weights, 3072, 1024));
    EXPECT_DOUBLE_EQ(0.0,
                     NonLinearPrefillModelR::calculateRequestLinearCost({}, 1024, 0));
}

TEST_F(NaviSchedDataTest, testLatencyFromRequestCostSumMatchesBatchPrediction)
{
    NonLinearPrefillModelR model;
    auto parameters = model.getInitialParameters();
    parameters[NonLinearPrefillModelR::TOTAL_PROMPT_PARAMETER_INDEX] =
        0.75 / NonLinearPrefillModelR::getParameterScale(
                   NonLinearPrefillModelR::TOTAL_PROMPT_PARAMETER_INDEX);
    const std::vector<RequestTokenCounts> requests = {
        makeLatencyFeatures(3072, 1024),
        makeLatencyFeatures(8192, 0),
        makeLatencyFeatures(5120, 4096),
    };
    AutoGradTape autoGrad(requests.size(), 32);
    AutoGradTape::Value requestCostSum;
    AutoGradTape::Value totalPromptTokenCount;
    NaviSchedLatencyFeatures features;
    features.appendBatchCount(1);
    for (size_t index = 0; index < requests.size(); ++index) {
        const auto &request = requests[index];
        features.appendRequest(request.cacheHitTokenCount, request.cacheMissTokenCount);
        auto totalTokenCount = request.cacheHitTokenCount + request.cacheMissTokenCount;
        auto requestCost = NonLinearPrefillModelR::calculateRequestLinearCost(
                parameters, totalTokenCount, request.cacheHitTokenCount);
        auto assignment = autoGrad.parameter(1.0, index);
        requestCostSum = requestCostSum + assignment * requestCost;
        totalPromptTokenCount = totalPromptTokenCount + assignment * totalTokenCount;
    }
    auto latency = NonLinearPrefillModelR::calculateLatencyFromRequestCostSum(
            parameters, requestCostSum, totalPromptTokenCount);
    EXPECT_NEAR(model.predict(features, parameters), latency.value(), 1e-12);

    std::vector<double> gradients;
    ASSERT_TRUE(autoGrad.backward(latency, gradients));
    ASSERT_EQ(requests.size(), gradients.size());
    auto linear =
        (requestCostSum.value() +
         parameters[0] * NonLinearPrefillModelR::getParameterScale(0) +
         parameters[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] *
             NonLinearPrefillModelR::getParameterScale(
                 NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX)) /
        320.0;
    auto requestCostDerivative =
        NonLinearPrefillModelR::NONLINEAR_SCALE *
        (1.0 + linear /
                   std::sqrt(
                       linear * linear +
                       parameters[NonLinearPrefillModelR::
                                      NONLINEAR_CURVATURE_PARAMETER_INDEX] *
                           NonLinearPrefillModelR::getParameterScale(
                               NonLinearPrefillModelR::
                                   NONLINEAR_CURVATURE_PARAMETER_INDEX))) /
        320.0;
    for (size_t index = 0; index < gradients.size(); ++index) {
        const auto &request = requests[index];
        auto totalTokenCount = request.cacheHitTokenCount + request.cacheMissTokenCount;
        auto requestCost = NonLinearPrefillModelR::calculateRequestLinearCost(
                parameters, totalTokenCount, request.cacheHitTokenCount);
        auto expectedGradient =
            requestCostDerivative * requestCost +
            parameters[NonLinearPrefillModelR::TOTAL_PROMPT_PARAMETER_INDEX] *
                NonLinearPrefillModelR::getParameterScale(
                    NonLinearPrefillModelR::TOTAL_PROMPT_PARAMETER_INDEX) *
                totalTokenCount / 1024.0;
        auto gradient = gradients[index];
        EXPECT_NEAR(expectedGradient, gradient, 1e-12);
    }
}

TEST_F(NaviSchedDataTest, testLatencyModelSnapshotValidation)
{
    NaviSchedLatencyModelSnapshot snapshot;
    snapshot.formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    snapshot.binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    snapshot.revision = 1;
    NonLinearPrefillModelR model;
    snapshot.parameters = model.getInitialParameters();
    ASSERT_TRUE(snapshot.valid());

    snapshot.binaryVersion = NonLinearPrefillModelR::BINARY_VERSION - 1;
    EXPECT_FALSE(snapshot.valid());
    snapshot.binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;

    for (const auto legacyParameterCount : {size_t(11), size_t(18)}) {
        snapshot.parameters.assign(legacyParameterCount, 1.0);
        EXPECT_FALSE(snapshot.valid());
    }

    snapshot.parameters = model.getInitialParameters();
    snapshot.parameters.resize(NonLinearPrefillModelR::PARAMETER_COUNT - 1);
    EXPECT_FALSE(snapshot.valid());
    snapshot.parameters = model.getInitialParameters();
    snapshot.binaryVersion = -1;
    EXPECT_FALSE(snapshot.valid());
    snapshot.formula = 2;
    snapshot.binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    snapshot.parameters = model.getInitialParameters();
    EXPECT_FALSE(snapshot.valid());
}

TEST_F(NaviSchedDataTest, testLegacyOptimizerStateRejected)
{
    for (const auto legacyParameterCount : {size_t(11), size_t(18)}) {
        const auto legacyState = makeOptimizerState(legacyParameterCount);
        ASSERT_TRUE(legacyState.valid(legacyParameterCount));
        NonLinearPrefillModelR::OptimizerState optimizerState;
        EXPECT_FALSE(
            NonLinearPrefillModelR::makeOptimizerState(legacyState, optimizerState));
    }
}

TEST_F(NaviSchedDataTest, testNonLinearPrefillModelLog1pFullData)
{
    const auto records = loadLearningPredictorRecords();
    ASSERT_EQ(1500, records.size());
    std::vector<std::vector<RequestTokenCounts>> batches;
    batches.reserve(records.size());
    for (const auto &record : records) {
        batches.emplace_back(makeLatencyBatch(record));
    }

    NonLinearPrefillModelR model;
    auto parameters = getJavaInitialParameters();
    std::deque<NaviSchedLatencySample> history;
    for (size_t index = 0; index < records.size(); ++index) {
        history.emplace_back(makeLatencySample(
            batches[index], static_cast<int64_t>(records[index].forwardMs)));
    }
    std::vector<double> updatedParameters;
    ReferenceLogAdam reference;
    NonLinearPrefillModelR replayModel;
    auto replayParameters = parameters;
    std::vector<NaviSchedLatencySample> referenceBatch;
    for (size_t epoch = 0; epoch < 100; ++epoch) {
        for (size_t offset = 0; offset < history.size(); ++offset) {
            referenceBatch.push_back(history[(epoch + offset) % history.size()]);
            if (referenceBatch.size() == 4) {
                // Audit each local step independently. Feeding the verified
                // state avoids amplifying roundoff across 37,500 nonlinear steps.
                reference.weights = replayParameters;
                const auto state = replayModel.getOptimizerState();
                std::copy_n(state.adamMoment1.begin(), 10, reference.first.begin());
                std::copy_n(state.adamMoment2.begin(), 10, reference.second.begin());
                reference.step = state.step;
                reference.fit(referenceBatch);
                std::vector<double> next;
                ASSERT_TRUE(replayModel.fit(referenceBatch, replayParameters, next));
                reference.expectState(replayModel, next);
                ASSERT_FALSE(HasFailure()) << "epoch " << epoch << " offset " << offset;
                replayParameters = std::move(next);
                referenceBatch.clear();
            }
        }
    }
    ASSERT_TRUE(referenceBatch.empty());
    ASSERT_TRUE(model.fitHistory(history, 4, 100, parameters, updatedParameters));
    parameters = std::move(updatedParameters);
    EXPECT_EQ(replayParameters, parameters);
    const auto expectedState = replayModel.getOptimizerState();
    const auto actualState = model.getOptimizerState();
    EXPECT_EQ(expectedState.step, actualState.step);
    EXPECT_EQ(expectedState.adamMoment1, actualState.adamMoment1);
    EXPECT_EQ(expectedState.adamMoment2, actualState.adamMoment2);
    reference.weights = parameters;

    double errorSum = 0.0;
    double referenceErrorSum = 0.0;
    for (size_t index = 0; index < records.size(); ++index) {
        errorSum += std::abs(predictBatch(model, batches[index], parameters) /
                                 records[index].forwardMs -
                             1.0);
        referenceErrorSum += std::abs(reference.predict(history[index].features) /
                                      records[index].forwardMs - 1.0);
    }
    ASSERT_NEAR(referenceErrorSum, errorSum, 1e-10);
    ASSERT_NEAR(referenceErrorSum / records.size(), errorSum / records.size(), 1e-12);
}
