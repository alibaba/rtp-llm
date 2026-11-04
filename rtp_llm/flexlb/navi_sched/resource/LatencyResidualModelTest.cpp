#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"

#include "unittest/unittest.h"

#include <cmath>
#include <limits>

using namespace navi_sched;

namespace {
LatencyResidualModel::RuntimeConfig baselineUpdateConfig()
{
    LatencyResidualModel::RuntimeConfig config;
    config.learningRate = 0.03;
    return config;
}

double evaluateWithoutMonotonicValidation(
    double basePredictionMs, const std::vector<double> &parameters)
{
    const auto position = std::log2(basePredictionMs);
    const auto lowerBucket = static_cast<int32_t>(std::floor(position));
    const auto lowerIndex = static_cast<size_t>(
        lowerBucket - LatencyResidualModel::MIN_BUCKET);
    const auto fraction = position - lowerBucket;
    const auto upperWeight = fraction * fraction * fraction *
        (fraction * (fraction * 6.0 - 15.0) + 10.0);
    return basePredictionMs + parameters[lowerIndex] +
        upperWeight * (parameters[lowerIndex + 1] - parameters[lowerIndex]);
}
} // namespace

class LatencyResidualModelTest : public TESTBASE {};

TEST_F(LatencyResidualModelTest, testZeroParametersAreIdentity)
{
    const auto parameters = LatencyResidualModel::makeInitialParameters();
    for (double prediction : {16.0, 24.0, 1024.0, 4194304.0}) {
        const auto result = LatencyResidualModel::evaluate(prediction, parameters);
        EXPECT_DOUBLE_EQ(prediction, result.predictionMs);
        EXPECT_DOUBLE_EQ(0.0, result.correctionMs);
        EXPECT_DOUBLE_EQ(1.0, result.derivativeFactor);
        EXPECT_TRUE(result.applied);
    }
}

TEST_F(LatencyResidualModelTest, testZeroLearningRateDisablesExistingResidual)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    parameters[3] = 30.0;
    ASSERT_TRUE(LatencyResidualModel::validateParameters(parameters));
    const auto previousParameters = parameters;
    int64_t lastUpdateTimeUs = 100;
    LatencyResidualModel::RuntimeConfig disabled;
    disabled.learningRate = 0.0;

    EXPECT_TRUE(disabled.valid());
    EXPECT_DOUBLE_EQ(30.0, LatencyResidualModel::evaluate(128.0, parameters).correctionMs);
    EXPECT_DOUBLE_EQ(0.0, LatencyResidualModel::decayFactor(lastUpdateTimeUs, 101, disabled));
    const auto prediction = LatencyResidualModel::evaluate(
        128.0, parameters, lastUpdateTimeUs, 101, disabled);
    EXPECT_DOUBLE_EQ(128.0, prediction.predictionMs);
    EXPECT_DOUBLE_EQ(0.0, prediction.correctionMs);
    EXPECT_DOUBLE_EQ(1.0, prediction.derivativeFactor);
    EXPECT_FALSE(prediction.applied);

    ASSERT_TRUE(LatencyResidualModel::applyIdleDecay(
        parameters, lastUpdateTimeUs, 101, disabled));
    EXPECT_EQ(previousParameters, parameters);
    ASSERT_TRUE(LatencyResidualModel::update(
        128.0, 200.0, parameters, lastUpdateTimeUs, 101, disabled));
    EXPECT_EQ(previousParameters, parameters);
    EXPECT_EQ(100, lastUpdateTimeUs);
    EXPECT_DOUBLE_EQ(30.0, LatencyResidualModel::evaluate(128.0, parameters).correctionMs);
}

TEST_F(LatencyResidualModelTest, testSmootherStepInterpolationAndDerivative)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    parameters[1] = 8.0;
    const auto midpoint = std::sqrt(16.0 * 32.0);
    const auto result = LatencyResidualModel::evaluate(midpoint, parameters);
    EXPECT_NEAR(4.0, result.correctionMs, 1e-10);

    const auto epsilon = midpoint * 1e-6;
    const auto left = LatencyResidualModel::evaluate(midpoint - epsilon, parameters);
    const auto right = LatencyResidualModel::evaluate(midpoint + epsilon, parameters);
    const auto numerical =
        (right.predictionMs - left.predictionMs) / (2.0 * epsilon);
    EXPECT_NEAR(numerical, result.derivativeFactor, 1e-7);
}

TEST_F(LatencyResidualModelTest, testRejectsNonMonotonicProductionParameters)
{
    // Captured from 11.225.198.48 at 2026-09-23 20:43, when load was
    // concentrated on this node while its corrected prediction decreased.
    const std::vector<double> parameters = {
        0.0,
        0.0,
        4.525707623468009,
        -28.8877894922789,
        -35.001980457401494,
        143.63054792577185,
        -369.18164093692627,
        447.2544156963677,
        -188.04473168337296,
        2689.6532523708706,
        -5899.21551647106,
        8912.258341352184,
        14261.144760242472,
        -90879.51284619785,
        -55914.49814996725,
        0.0,
        0.0,
        0.0,
        0.0,
    };
    ASSERT_FALSE(LatencyResidualModel::validateParameters(parameters));

    const auto oldAt80Seconds =
        evaluateWithoutMonotonicValidation(80000.0, parameters);
    const auto oldAt100Seconds =
        evaluateWithoutMonotonicValidation(100000.0, parameters);
    EXPECT_NEAR(78783.53079913935, oldAt80Seconds, 1e-8);
    EXPECT_NEAR(40759.3095244678, oldAt100Seconds, 1e-8);
    EXPECT_GT(oldAt80Seconds, oldAt100Seconds);

    const auto at80Seconds = LatencyResidualModel::evaluate(80000.0, parameters);
    const auto at100Seconds = LatencyResidualModel::evaluate(100000.0, parameters);
    EXPECT_FALSE(at80Seconds.applied);
    EXPECT_FALSE(at100Seconds.applied);
    EXPECT_DOUBLE_EQ(80000.0, at80Seconds.predictionMs);
    EXPECT_DOUBLE_EQ(100000.0, at100Seconds.predictionMs);
    EXPECT_LE(at80Seconds.predictionMs, at100Seconds.predictionMs);
    EXPECT_GE(at80Seconds.derivativeFactor, 0.0);
    EXPECT_GE(at100Seconds.derivativeFactor, 0.0);
}

TEST_F(LatencyResidualModelTest, testDerivativeIsContinuousAtKnotsAndBounds)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    parameters[1] = 8.0;
    parameters[2] = -4.0;
    ASSERT_TRUE(LatencyResidualModel::validateParameters(parameters));
    for (double knot : {16.0, 32.0, 64.0, 4194304.0}) {
        const auto epsilon = knot * 1e-8;
        const auto left = LatencyResidualModel::evaluate(knot - epsilon, parameters);
        const auto at = LatencyResidualModel::evaluate(knot, parameters);
        const auto right = LatencyResidualModel::evaluate(knot + epsilon, parameters);
        EXPECT_NEAR(1.0, left.derivativeFactor, 1e-10);
        EXPECT_NEAR(1.0, at.derivativeFactor, 1e-12);
        EXPECT_NEAR(1.0, right.derivativeFactor, 1e-10);
    }
}

TEST_F(LatencyResidualModelTest, testOutsideRangeIsIdentity)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    parameters[3] = 50.0;
    ASSERT_TRUE(LatencyResidualModel::validateParameters(parameters));
    EXPECT_FALSE(LatencyResidualModel::evaluate(15.0, parameters).applied);
    EXPECT_FALSE(LatencyResidualModel::evaluate(4194305.0, parameters).applied);
}

TEST_F(LatencyResidualModelTest, testLowBasePredictionIsValidNoOpForUpdate)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    parameters[3] = 50.0;
    const auto original = parameters;
    int64_t lastUpdateTimeUs = 17;
    ASSERT_TRUE(LatencyResidualModel::update(
        1.0, 25.0, parameters, lastUpdateTimeUs, 23));
    EXPECT_EQ(original, parameters);
    EXPECT_EQ(17, lastUpdateTimeUs);
    const auto result = LatencyResidualModel::evaluate(1.0, parameters);
    EXPECT_FALSE(result.applied);
    EXPECT_DOUBLE_EQ(1.0, result.predictionMs);
}

TEST_F(LatencyResidualModelTest, testUpdateMovesCurrentPointByLearningRate)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    int64_t lastUpdateTimeUs = 0;
    const auto prediction = std::sqrt(128.0 * 256.0);
    ASSERT_TRUE(LatencyResidualModel::update(prediction, prediction + 100.0,
                                             parameters, lastUpdateTimeUs, 1,
                                             baselineUpdateConfig()));
    const auto result = LatencyResidualModel::evaluate(prediction, parameters);
    EXPECT_NEAR(3.0, result.correctionMs, 1e-10);
}

TEST_F(LatencyResidualModelTest, testUpdateClipsSingleSampleAmplitude)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    int64_t lastUpdateTimeUs = 0;
    constexpr double prediction = 10000.0;
    ASSERT_TRUE(LatencyResidualModel::update(
        prediction, 1e9, parameters, lastUpdateTimeUs, 1,
        baselineUpdateConfig()));
    const auto result = LatencyResidualModel::evaluate(prediction, parameters);
    EXPECT_NEAR(30.0, result.correctionMs, 1e-10);
}

TEST_F(LatencyResidualModelTest, testRejectsInvalidStateWithoutMutation)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    const auto original = parameters;
    int64_t lastUpdateTimeUs = 10;
    EXPECT_FALSE(LatencyResidualModel::update(
        128.0, std::numeric_limits<double>::infinity(), parameters,
        lastUpdateTimeUs, 20));
    EXPECT_EQ(original, parameters);
    EXPECT_EQ(10, lastUpdateTimeUs);
    parameters[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(LatencyResidualModel::validateParameters(parameters));
    const auto result = LatencyResidualModel::evaluate(128.0, parameters);
    EXPECT_FALSE(result.applied);
    EXPECT_DOUBLE_EQ(128.0, result.predictionMs);
}

TEST_F(LatencyResidualModelTest, testDoesNotHideInvalidBasePrediction)
{
    const auto parameters = LatencyResidualModel::makeInitialParameters();
    const auto infinity = LatencyResidualModel::evaluate(
        std::numeric_limits<double>::infinity(), parameters);
    EXPECT_FALSE(infinity.applied);
    EXPECT_TRUE(std::isinf(infinity.predictionMs));
    const auto nan = LatencyResidualModel::evaluate(
        std::numeric_limits<double>::quiet_NaN(), parameters);
    EXPECT_FALSE(nan.applied);
    EXPECT_TRUE(std::isnan(nan.predictionMs));
}

TEST_F(LatencyResidualModelTest, testIdleDecayBoundaries)
{
    constexpr int64_t secondUs = 1000 * 1000;
    constexpr int64_t lastUpdateTimeUs = 10 * secondUs;
    EXPECT_DOUBLE_EQ(1.0, LatencyResidualModel::decayFactor(
                              lastUpdateTimeUs, lastUpdateTimeUs + 119 * secondUs));
    EXPECT_DOUBLE_EQ(1.0, LatencyResidualModel::decayFactor(
                              lastUpdateTimeUs, lastUpdateTimeUs + 120 * secondUs));
    EXPECT_DOUBLE_EQ(0.5, LatencyResidualModel::decayFactor(
                              lastUpdateTimeUs, lastUpdateTimeUs + 210 * secondUs));
    EXPECT_DOUBLE_EQ(0.0, LatencyResidualModel::decayFactor(
                              lastUpdateTimeUs, lastUpdateTimeUs + 300 * secondUs));
    EXPECT_TRUE(LatencyResidualModel::expired(
        lastUpdateTimeUs, lastUpdateTimeUs + 300 * secondUs));
}

TEST_F(LatencyResidualModelTest, testIdleDecayScalesCorrectionAndDerivative)
{
    constexpr int64_t secondUs = 1000 * 1000;
    constexpr int64_t lastUpdateTimeUs = 10 * secondUs;
    auto parameters = LatencyResidualModel::makeInitialParameters();
    parameters[3] = 40.0;
    parameters[4] = 100.0;
    ASSERT_TRUE(LatencyResidualModel::validateParameters(parameters));
    const auto prediction = std::sqrt(128.0 * 256.0);
    const auto full = LatencyResidualModel::evaluate(
        prediction, parameters, lastUpdateTimeUs,
        lastUpdateTimeUs + 120 * secondUs);
    const auto half = LatencyResidualModel::evaluate(
        prediction, parameters, lastUpdateTimeUs,
        lastUpdateTimeUs + 210 * secondUs);
    const auto zero = LatencyResidualModel::evaluate(
        prediction, parameters, lastUpdateTimeUs,
        lastUpdateTimeUs + 300 * secondUs);
    EXPECT_NEAR(full.correctionMs * 0.5, half.correctionMs, 1e-12);
    EXPECT_NEAR((full.derivativeFactor - 1.0) * 0.5,
                half.derivativeFactor - 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(0.0, zero.correctionMs);
    EXPECT_DOUBLE_EQ(prediction, zero.predictionMs);
    EXPECT_DOUBLE_EQ(1.0, zero.derivativeFactor);
}

TEST_F(LatencyResidualModelTest, testUpdatePermanentlyAppliesIdleDecay)
{
    constexpr int64_t secondUs = 1000 * 1000;
    auto parameters = LatencyResidualModel::makeInitialParameters();
    parameters[3] = 50.0;
    int64_t lastUpdateTimeUs = 10 * secondUs;
    const auto updateTimeUs = lastUpdateTimeUs + 210 * secondUs;
    ASSERT_TRUE(LatencyResidualModel::update(
        128.0, 153.0, parameters, lastUpdateTimeUs, updateTimeUs));
    EXPECT_EQ(updateTimeUs, lastUpdateTimeUs);
    EXPECT_NEAR(25.0, parameters[3], 1e-12);
    const auto result = LatencyResidualModel::evaluate(
        128.0, parameters, lastUpdateTimeUs, updateTimeUs);
    EXPECT_DOUBLE_EQ(25.0, result.correctionMs);
}

TEST_F(LatencyResidualModelTest, testUpdatesPreserveMonotonicPrediction)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    int64_t lastUpdateTimeUs = 0;
    for (size_t round = 0; round < 20; ++round) {
        for (int32_t bucket = LatencyResidualModel::MIN_BUCKET;
             bucket < LatencyResidualModel::MAX_BUCKET; ++bucket) {
            const auto basePrediction = std::exp2(bucket + 0.37);
            const auto actualLatency =
                (round + bucket) % 2 == 0 ? 0.0 : basePrediction * 10.0;
            ASSERT_TRUE(LatencyResidualModel::update(
                basePrediction, actualLatency, parameters, lastUpdateTimeUs,
                static_cast<int64_t>(round * 100 + bucket)));
            ASSERT_TRUE(LatencyResidualModel::validateParameters(parameters));
        }
    }

    auto previous = LatencyResidualModel::evaluate(
        std::exp2(LatencyResidualModel::MIN_BUCKET), parameters);
    ASSERT_TRUE(previous.applied);
    for (double position = LatencyResidualModel::MIN_BUCKET + 0.001;
         position <= LatencyResidualModel::MAX_BUCKET; position += 0.001) {
        const auto basePrediction = std::exp2(position);
        const auto current =
            LatencyResidualModel::evaluate(basePrediction, parameters);
        ASSERT_TRUE(current.applied);
        EXPECT_GE(current.predictionMs + 1e-8, previous.predictionMs);
        EXPECT_GE(current.derivativeFactor, 0.0);
        EXPECT_TRUE(std::isfinite(current.predictionMs));
        EXPECT_TRUE(std::isfinite(current.derivativeFactor));
        previous = current;
    }
}

TEST_F(LatencyResidualModelTest, testUpdateSupportsSignedResidual)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    int64_t lastUpdateTimeUs = 0;
    ASSERT_TRUE(LatencyResidualModel::update(
        128.0, 28.0, parameters, lastUpdateTimeUs, 1,
        baselineUpdateConfig()));
    EXPECT_NEAR(-3.0,
                LatencyResidualModel::evaluate(128.0, parameters).correctionMs,
                1e-12);
}

TEST_F(LatencyResidualModelTest, testUpdateWithoutTimestampSkipsIdleDecay)
{
    auto parameters = LatencyResidualModel::makeInitialParameters();
    int64_t lastUpdateTimeUs = 0;
    ASSERT_TRUE(LatencyResidualModel::update(
        128.0, 228.0, parameters, lastUpdateTimeUs, 0,
        baselineUpdateConfig()));
    EXPECT_EQ(0, lastUpdateTimeUs);
    EXPECT_NEAR(3.0,
                LatencyResidualModel::evaluate(128.0, parameters).correctionMs,
                1e-12);
}

TEST_F(LatencyResidualModelTest, testRuntimeConfigsAreIndependent)
{
    EXPECT_DOUBLE_EQ(0.8, LatencyResidualModel::RuntimeConfig().learningRate);
    LatencyResidualModel::RuntimeConfig fast;
    fast.learningRate = 0.5;
    fast.idleDecayStartUs = 10;
    fast.idleDecayEndUs = 20;
    LatencyResidualModel::RuntimeConfig slow;
    slow.learningRate = 0.01;
    slow.idleDecayStartUs = 100;
    slow.idleDecayEndUs = 200;
    ASSERT_TRUE(fast.valid());
    ASSERT_TRUE(slow.valid());

    auto fastParameters = LatencyResidualModel::makeInitialParameters();
    auto slowParameters = fastParameters;
    int64_t fastUpdateTimeUs = 0;
    int64_t slowUpdateTimeUs = 0;
    ASSERT_TRUE(LatencyResidualModel::update(
        128.0, 228.0, fastParameters, fastUpdateTimeUs, 1, fast));
    ASSERT_TRUE(LatencyResidualModel::update(
        128.0, 228.0, slowParameters, slowUpdateTimeUs, 1, slow));
    EXPECT_NEAR(50.0,
                LatencyResidualModel::evaluate(
                    128.0, fastParameters, fastUpdateTimeUs, 1, fast)
                    .correctionMs,
                1e-12);
    EXPECT_NEAR(1.0,
                LatencyResidualModel::evaluate(
                    128.0, slowParameters, slowUpdateTimeUs, 1, slow)
                    .correctionMs,
                1e-12);
    EXPECT_DOUBLE_EQ(
        0.0, LatencyResidualModel::decayFactor(fastUpdateTimeUs, 21, fast));
    EXPECT_DOUBLE_EQ(
        1.0, LatencyResidualModel::decayFactor(slowUpdateTimeUs, 21, slow));
}
