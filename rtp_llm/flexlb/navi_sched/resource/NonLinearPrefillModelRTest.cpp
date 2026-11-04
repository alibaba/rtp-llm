#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterLatencyModelR.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContext.h"

#include "unittest/unittest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

using namespace navi_sched;

class NonLinearPrefillModelRTest : public TESTBASE {};

TEST_F(NonLinearPrefillModelRTest, testValueOnlyLatencyExactlyMatchesDerivativeEvaluator)
{
    std::mt19937_64 generator(20261027);
    std::uniform_real_distribution<double> coefficient(-1000.0, 1000.0);
    for (const auto capacitySplit : {false, true}) {
        NonLinearPrefillModelR model;
        model._useCapacitySplitFeature = capacitySplit;
        for (size_t sample = 0; sample < 128; ++sample) {
            auto parameters = model.getInitialParameters();
            if (sample > 0) {
                for (auto &parameter : parameters) {
                    parameter = coefficient(generator);
                }
                parameters[NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX] =
                    std::abs(parameters[NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX]) +
                    NonLinearPrefillModelR::MIN_NONLINEAR_CURVATURE /
                        NonLinearPrefillModelR::PARAMETER_SCALES[
                            NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX];
            }
            const auto zeroLinearCost =
                -parameters[0] * NonLinearPrefillModelR::PARAMETER_SCALES[0] -
                parameters[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] *
                    NonLinearPrefillModelR::PARAMETER_SCALES[
                        NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX];
            for (const auto cost : {-1e100, -1e6, -1.0, 0.0, 1e-9, 1.0, 1e6, 1e100,
                                    zeroLinearCost,
                                    std::nextafter(zeroLinearCost, -INFINITY),
                                    std::nextafter(zeroLinearCost, INFINITY)}) {
                for (const auto prompt : {0.0, 3072.0, 1e8}) {
                    for (const auto preceding : {0.0, 1.0, 10000.0}) {
                        double costDerivative = 0.0;
                        double promptDerivative = 0.0;
                        const auto expected = NonLinearPrefillModelR::
                            calculateLatencyAndDerivativesFromRequestCostSum(
                                parameters, cost, prompt, preceding,
                                costDerivative, promptDerivative);
                        const auto actual = NonLinearPrefillModelR::
                            calculateLatencyFromRequestCostSum(
                                parameters, cost, prompt, preceding);
                        ASSERT_EQ(expected, actual)
                            << "capacitySplit=" << capacitySplit << " sample=" << sample
                            << " cost=" << cost << " prompt=" << prompt
                            << " preceding=" << preceding;
                    }
                }
            }
        }
    }
}

TEST_F(NonLinearPrefillModelRTest, testPairedRequestCostsMatchIndependentFeatureEvaluation)
{
    std::mt19937_64 generator(601026);
    std::uniform_real_distribution<double> coefficient(-1000.0, 1000.0);
    for (size_t parameterCount = 0; parameterCount <= 22; ++parameterCount) {
        std::vector<double> parameters(parameterCount);
        for (auto &parameter : parameters) {
            parameter = coefficient(generator);
        }
        for (const int64_t prompt : {0, 1, 255, 256, 3072, 131072, 1048576}) {
            for (const int64_t hit : {int64_t(0), prompt / 3, prompt}) {
                const auto costs = NonLinearPrefillModelR::calculateRequestLinearCosts(
                    parameters, prompt, hit);
                EXPECT_EQ(NonLinearPrefillModelR::calculateRequestLinearCost(
                              parameters, prompt, hit), costs.first);
                EXPECT_EQ(NonLinearPrefillModelR::calculateWaitingRequestLinearCost(
                              parameters, prompt, hit), costs.second);
            }
        }
    }
    for (const size_t count : {size_t(10), size_t(19)}) {
        for (size_t index = 0; index < count; ++index) {
            for (const double special : {std::numeric_limits<double>::infinity(),
                                          std::numeric_limits<double>::quiet_NaN()}) {
                std::vector<double> parameters(count, 1.0);
                parameters[index] = special;
                const auto costs = NonLinearPrefillModelR::calculateRequestLinearCosts(
                    parameters, 3072, 2048);
                const auto running = NonLinearPrefillModelR::calculateRequestLinearCost(
                    parameters, 3072, 2048);
                const auto waiting = NonLinearPrefillModelR::calculateWaitingRequestLinearCost(
                    parameters, 3072, 2048);
                if (std::isnan(running)) {
                    EXPECT_TRUE(std::isnan(costs.first));
                } else {
                    EXPECT_EQ(running, costs.first);
                }
                if (std::isnan(waiting)) {
                    EXPECT_TRUE(std::isnan(costs.second));
                } else {
                    EXPECT_EQ(waiting, costs.second);
                }
            }
        }
    }
}

namespace {

double parameterForFormulaCoefficient(size_t index, double value)
{
    return value / NonLinearPrefillModelR::getParameterScale(index);
}

std::vector<double> makeNegativeWaitingCostIncidentParameters()
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

NaviSchedLatencyFeatures makeNegativeWaitingCostIncidentFeatures()
{
    NaviSchedLatencyFeatures features;
    EXPECT_TRUE(features.appendCapacitySplitAggregate(
        424, 0, 0, 0, 0.0, 0.0, 541, 2'078'064, 40'022,
        14'084'028.0, 153'692'768.0));
    EXPECT_TRUE(features.appendPrecedingRequestCount(792));
    return features;
}

} // namespace

TEST_F(NonLinearPrefillModelRTest, testNonlinearOperatingRegions)
{
    NonLinearPrefillModelR model;
    NaviSchedLatencyFeatures features;
    features.appendRequest(0, 0);
    auto parameters = model.getInitialParameters();
    for (auto &parameter : parameters) {
        parameter = 0.0;
    }
    for (size_t index : {1u, 2u, 3u, 4u, 5u, 7u, 9u}) {
        parameters[index] = model.positiveParameterFloor(index, 1.0);
    }
    parameters[NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX] =
        parameterForFormulaCoefficient(
            NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX, 1.0);

    for (const auto &test : {std::pair<double, int>{-3.01, -1}, {-3.0, 0},
                             {0.0, 0}, {3.0, 0}, {3.01, 1}}) {
        parameters[0] = parameterForFormulaCoefficient(
            0, test.first * 320.0 -
                   parameters[1] * NonLinearPrefillModelR::getParameterScale(1));
        const auto point = model.getNonlinearOperatingPointConfigured(
            features, parameters);
        ASSERT_TRUE(point.has_value());
        EXPECT_NEAR(test.first, point->u, 1e-12);
        EXPECT_DOUBLE_EQ(1.0, point->curvature);
        EXPECT_NEAR(test.first, point->r, 1e-12);
        EXPECT_EQ(test.second, point->region);
    }

    parameters[NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX] =
        parameterForFormulaCoefficient(
            NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX, 4.0);
    parameters[0] = parameterForFormulaCoefficient(
        0, 8.0 * 320.0 -
               parameters[1] * NonLinearPrefillModelR::getParameterScale(1));
    const auto scaled = model.getNonlinearOperatingPointConfigured(
        features, parameters);
    ASSERT_TRUE(scaled.has_value());
    EXPECT_DOUBLE_EQ(8.0, scaled->u);
    EXPECT_DOUBLE_EQ(4.0, scaled->r);
    EXPECT_EQ(1, scaled->region);
}

TEST_F(NonLinearPrefillModelRTest, testRejectsInvalidParameters)
{
    NonLinearPrefillModelR model;
    NaviSchedLatencyFeatures features;
    auto parameters = model.getInitialParameters();
    parameters[NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX] = 0.0;
    EXPECT_FALSE(model.getNonlinearOperatingPointConfigured(
        features, parameters).has_value());
}

TEST_F(NonLinearPrefillModelRTest, testMonotoneCoefficientsHaveNonNegativeFloor)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    const auto initial = model.getInitialParameters();
    NonLinearPrefillModelR::Values weights{}, m1{}, m2{};
    std::copy(initial.begin(), initial.end(), weights.begin());
    const std::array<size_t, 12> indexes = {1, 2, 3, 4, 5, 7, 9,
                                             10, 11, 12, 13, 14};
    for (auto index : indexes) {
        SCOPED_TRACE(index);
        auto invalid = initial;
        invalid[index] = 0.0;
        EXPECT_TRUE(NonLinearPrefillModelR::validateParameters(invalid));
        weights[index] = -1.0;
    }
    model.projectOutputParameters(weights, m1, m2, 1e-6);
    for (auto index : indexes) {
        SCOPED_TRACE(index);
        EXPECT_DOUBLE_EQ(0.0, weights[index]);
    }
}

TEST_F(NonLinearPrefillModelRTest, testEnabledCountCoefficientsHaveNonNegativeFloor)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    model._useBatchCountFeature = true;
    model._useRunningDecodeRequestCountFeature = true;
    NonLinearPrefillModelR::Values weights{}, m1{}, m2{};
    const auto initial = model.getInitialParameters();
    std::copy(initial.begin(), initial.end(), weights.begin());
    weights[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] = -1.0;
    weights[NonLinearPrefillModelR::RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX] = -1.0;
    model.projectOutputParameters(weights, m1, m2, 1e-6);
    EXPECT_DOUBLE_EQ(0.0,
                     weights[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX]);
    EXPECT_DOUBLE_EQ(
        0.0,
        weights[NonLinearPrefillModelR::RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX]);

    NaviSchedLatencyFeatures features;
    ASSERT_TRUE(features.appendCapacitySplitAggregate(
        2, 2, 2048, 4096, 16'777'216.0, 8'388'608.0,
        2, 2048, 4096, 16'777'216.0, 8'388'608.0));
    ASSERT_TRUE(features.appendRunningDecodeProgress(3, 3));
    const auto baseline = model.predictConfigured(features, initial);
    for (auto index : {NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX,
                       NonLinearPrefillModelR::RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX}) {
        auto increased = initial;
        increased[index] += 1.0;
        EXPECT_GT(model.predictConfigured(features, increased), baseline);
    }

    model._useBatchCountFeature = false;
    model._useRunningDecodeRequestCountFeature = false;
    model.projectOutputParameters(weights, m1, m2, 1e-6);
    EXPECT_DOUBLE_EQ(0.0, weights[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX]);
    EXPECT_DOUBLE_EQ(0.0,
                     weights[NonLinearPrefillModelR::RUNNING_DECODE_REQUEST_COUNT_PARAMETER_INDEX]);
}

TEST_F(NonLinearPrefillModelRTest, testPrecedingCountFloorFollowsNonlinearScale)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    NonLinearPrefillModelR::Values weights{}, m1{}, m2{};
    const auto index = NonLinearPrefillModelR::PRECEDING_REQUEST_COUNT_PARAMETER_INDEX;
    model.projectOutputParameters(weights, m1, m2, 1e-6, 2.0);
    EXPECT_GE(weights[index] * NonLinearPrefillModelR::getParameterScale(index),
              2.0 * NonLinearPrefillModelR::MIN_PRECEDING_REQUEST_COEFFICIENT_MS);
}

TEST_F(NonLinearPrefillModelRTest, testPositiveWorkloadCoefficientsIncreasePrediction)
{
    for (bool split : {false, true}) {
        NonLinearPrefillModelR model;
        model._useCapacitySplitFeature = split;
        NaviSchedLatencyFeatures features;
        if (split) {
            ASSERT_TRUE(features.appendCapacitySplitAggregate(
                2, 2, 2048, 4096, 16'777'216.0, 8'388'608.0,
                2, 2048, 4096, 16'777'216.0, 8'388'608.0));
            ASSERT_TRUE(features.appendPrecedingRequestCount(3));
        } else {
            features.appendRequest(2048, 4096);
        }
        const auto parameters = model.getInitialParameters();
        const auto baseline = model.predictConfigured(features, parameters);
        for (auto index : {1u, 2u, 3u, 4u, 5u, 7u, 9u,
                           10u, 11u, 12u, 13u, 14u}) {
            if (index >= parameters.size()) {
                continue;
            }
            SCOPED_TRACE(index);
            auto increased = parameters;
            increased[index] += 0.1;
            EXPECT_GT(model.predictConfigured(features, increased), baseline);
        }
    }
}

TEST_F(NonLinearPrefillModelRTest, testFirstFitProjectsRestoredZeroCoefficients)
{
    for (double nonlinearScale : {750.0, 1500.0, 3000.0}) {
        NonLinearPrefillModelR model;
        model._useCapacitySplitFeature = true;
        model._useRunningDecodeRequestCountFeature = true;
        model._nonlinearScale = nonlinearScale;
        auto parameters = model.getInitialParameters();
        for (auto index : NonLinearPrefillModelR::POSITIVE_PARAMETER_INDEXES) {
            parameters[index] = 0.0;
        }
        ASSERT_TRUE(NonLinearPrefillModelR::validateParameters(parameters));
        NaviSchedLatencySample sample;
        ASSERT_TRUE(sample.features.appendCapacitySplitAggregate(
            2, 2, 2048, 4096, 16'777'216.0, 8'388'608.0,
            2, 2048, 4096, 16'777'216.0, 8'388'608.0));
        sample.latencyMs = 1000.0;
        std::vector<double> updated;
        ASSERT_TRUE(model.fit({sample}, parameters, updated));
        for (auto index : NonLinearPrefillModelR::POSITIVE_PARAMETER_INDEXES) {
            EXPECT_GE(updated[index], model.positiveParameterFloor(index, 1.0))
                << index << " nonlinearScale=" << nonlinearScale;
        }
    }
}

TEST_F(NonLinearPrefillModelRTest, testCapacitySplitUsesWaitingAndRunningElapsed)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    NaviSchedLatencyFeatures features;
    ASSERT_TRUE(features.appendCapacitySplitAggregate(
        1, 1, 0, 0, 0.0, 0.0, 1, 0, 0, 0.0, 0.0));
    ASSERT_TRUE(features.appendRunningElapsedTimeMs(2'000'000.0));
    auto parameters = model.getInitialParameters();
    for (auto &parameter : parameters) {
        parameter = 0.0;
    }
    for (size_t index : {1u, 2u, 3u, 4u, 5u, 7u, 9u,
                         10u, 11u, 12u, 13u, 14u}) {
        parameters[index] = model.positiveParameterFloor(index, 1.0);
    }
    parameters[1] = parameterForFormulaCoefficient(1, 320.0);
    parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET] =
        parameterForFormulaCoefficient(NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET,
                                   640.0);
    parameters[NonLinearPrefillModelR::RUNNING_ELAPSED_TIME_PARAMETER_INDEX] =
        parameterForFormulaCoefficient(
            NonLinearPrefillModelR::RUNNING_ELAPSED_TIME_PARAMETER_INDEX, 160.0);
    parameters[NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX] =
        parameterForFormulaCoefficient(
            NonLinearPrefillModelR::NONLINEAR_CURVATURE_PARAMETER_INDEX, 4.0);
    parameters[NonLinearPrefillModelR::PRECEDING_REQUEST_COUNT_PARAMETER_INDEX] =
        parameterForFormulaCoefficient(
            NonLinearPrefillModelR::PRECEDING_REQUEST_COUNT_PARAMETER_INDEX,
            NonLinearPrefillModelR::MIN_PRECEDING_REQUEST_COEFFICIENT_MS);

    const auto point = model.getNonlinearOperatingPointConfigured(
        features, parameters);
    ASSERT_TRUE(point.has_value());
    EXPECT_NEAR(4.0, point->u, 1e-3);
    EXPECT_NEAR(2.0, point->r, 1e-3);
    EXPECT_EQ(0, point->region);
}

TEST_F(NonLinearPrefillModelRTest, testRejectsNegativeWaitingWorkloadCostFromIncident)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    const auto features = makeNegativeWaitingCostIncidentFeatures();
    const auto incidentParameters = makeNegativeWaitingCostIncidentParameters();

    double waitingCost = 0.0;
    EXPECT_FALSE(NonLinearPrefillModelR::hasNonNegativeWaitingWorkloadCost(
        incidentParameters, features, &waitingCost));
    EXPECT_DOUBLE_EQ(0.0, waitingCost);
    EXPECT_NEAR(-6950.317548237581,
                NonLinearPrefillModelR::calculateLinearCost(incidentParameters, features),
                1e-9);
    EXPECT_TRUE(NonLinearPrefillModelR::hasNonNegativeWaitingWorkloadCost(
        model.getInitialParameters(), features));

    auto safeParameters = incidentParameters;
    safeParameters[10] = 3.7331160591272701;
    safeParameters[11] = 3.4794434928013009;
    safeParameters[12] = 0.077691744104586583;
    safeParameters[13] = -0.21142905740360265;
    safeParameters[14] = 0.072042782697160815;
    ASSERT_LT(safeParameters[13], 0.0);
    EXPECT_FALSE(NonLinearPrefillModelR::hasNonNegativeWaitingWorkloadCost(
        safeParameters, features, &waitingCost));
    EXPECT_DOUBLE_EQ(0.0, waitingCost);
    EXPECT_NEAR(10570.357553180886,
                NonLinearPrefillModelR::calculateLinearCost(safeParameters, features),
                1e-9);
}

TEST_F(NonLinearPrefillModelRTest, testFitDoesNotPublishNegativeWaitingCostCandidate)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    const auto parameters = makeNegativeWaitingCostIncidentParameters();
    NaviSchedLatencySample sample;
    sample.features = makeNegativeWaitingCostIncidentFeatures();
    sample.latencyMs = 240'000.0;
    NonLinearPrefillModelR::OptimizerState dirtyOptimizer;
    dirtyOptimizer.step = 17;
    dirtyOptimizer.adamMoment1[13] = 2.0;
    dirtyOptimizer.adamMoment2[13] = 3.0;
    ASSERT_TRUE(model.restoreOptimizerState(dirtyOptimizer));

    std::vector<double> updatedParameters = {123.0};
    EXPECT_FALSE(model.fit({sample}, parameters, updatedParameters));
    EXPECT_EQ(std::vector<double>({123.0}), updatedParameters);
    EXPECT_EQ(0, model._unsafeWaitingCandidateRecoveryCount.load());
    EXPECT_EQ(0, model._unsafeWaitingStateResetCount.load());
    EXPECT_EQ(dirtyOptimizer.step, model.getOptimizerState().step);
    EXPECT_EQ(dirtyOptimizer.adamMoment1, model.getOptimizerState().adamMoment1);
    EXPECT_EQ(dirtyOptimizer.adamMoment2, model.getOptimizerState().adamMoment2);
}

TEST_F(NonLinearPrefillModelRTest, testRejectedWaitingParameterDoesNotFreezeOtherParameters)
{
    for (double nonlinearScale : {750.0, 1500.0, 3000.0}) {
        for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
            NonLinearPrefillModelR model;
            model._useCapacitySplitFeature = true;
            model._nonlinearScale = nonlinearScale;
            auto parameters = model.getInitialParameters();
            const auto index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset;
            const auto healthyIndex = NonLinearPrefillModelR::PRECEDING_REQUEST_COUNT_PARAMETER_INDEX;
            parameters[index] = 1e-10;
            auto optimizer = model.getOptimizerState();
            optimizer.step = 17;
            optimizer.adamMoment1[index] = 2000.0;
            optimizer.adamMoment2[index] = 3.0;
            optimizer.adamMoment1[healthyIndex] = 0.3;
            optimizer.adamMoment2[healthyIndex] = 3.0;
            ASSERT_TRUE(model.restoreOptimizerState(optimizer));
            for (size_t attempt = 0; attempt < 4; ++attempt) {
                SCOPED_TRACE(::testing::Message() << "scale=" << nonlinearScale
                                                 << " offset=" << offset
                                                 << " attempt=" << attempt);
                NaviSchedLatencySample sample;
                sample.features = makeNegativeWaitingCostIncidentFeatures();
                sample.latencyMs = model.predictConfigured(sample.features, parameters);
                const auto before = model.getOptimizerState();
                NonLinearPrefillModelR reference;
                reference._useCapacitySplitFeature = true;
                reference._nonlinearScale = nonlinearScale;
                ASSERT_TRUE(reference.restoreOptimizerState(before));
                std::vector<double> expected;
                ASSERT_TRUE(reference.fit({sample}, parameters, expected));
                std::vector<double> updated;
                ASSERT_TRUE(model.fit({sample}, parameters, updated, 1.0, 1.0, nullptr, true));
                const auto after = model.getOptimizerState();
                if (attempt == 0) {
                    EXPECT_DOUBLE_EQ(0.0, expected[index]);
                    EXPECT_LT(updated[healthyIndex], parameters[healthyIndex]);
                }
                if (expected[index] == 0.0 && parameters[index] > 0.0) {
                    EXPECT_DOUBLE_EQ(parameters[index], updated[index]);
                    EXPECT_DOUBLE_EQ(before.adamMoment1[index], after.adamMoment1[index]);
                    EXPECT_DOUBLE_EQ(before.adamMoment2[index], after.adamMoment2[index]);
                } else {
                    EXPECT_DOUBLE_EQ(expected[index], updated[index]);
                }
                EXPECT_EQ(before.step + 1, after.step);
                EXPECT_DOUBLE_EQ(expected[healthyIndex], updated[healthyIndex]);
                EXPECT_DOUBLE_EQ(reference.getOptimizerState().adamMoment1[healthyIndex],
                                 after.adamMoment1[healthyIndex]);
                EXPECT_DOUBLE_EQ(reference.getOptimizerState().adamMoment2[healthyIndex],
                                 after.adamMoment2[healthyIndex]);
                parameters = std::move(updated);
            }
        }
    }
}

TEST_F(NonLinearPrefillModelRTest, testTrustedWaitingGuardPreservesPositiveCoefficientsAndExistingZeros)
{
    for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
        for (const bool exactlyZero : {false, true}) {
            NonLinearPrefillModelR model;
            model._useCapacitySplitFeature = true;
            auto parameters = model.getInitialParameters();
            const auto index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset;
            parameters[index] = exactlyZero ? 0.0 : 1e-10;
            NonLinearPrefillModelR::OptimizerState before;
            before.step = 17;
            if (exactlyZero) {
                model._learningRate = 0.0;
            } else {
                before.adamMoment1[index] = 2.0;
                before.adamMoment2[index] = 3.0;
            }
            ASSERT_TRUE(model.restoreOptimizerState(before));
            NaviSchedLatencySample sample;
            sample.features = makeNegativeWaitingCostIncidentFeatures();
            sample.latencyMs = NonLinearPrefillModelR::predict(sample.features, parameters);
            std::vector<double> updated = {123.0};
            ASSERT_TRUE(model.fit({sample}, parameters, updated, 1.0, 1.0, nullptr, true))
                << "index=" << index << " exactlyZero=" << exactlyZero;
            const auto after = model.getOptimizerState();
            EXPECT_EQ(parameters, updated);
            EXPECT_EQ(before.step + 1, after.step);
            EXPECT_EQ(before.adamMoment1, after.adamMoment1);
            EXPECT_EQ(before.adamMoment2, after.adamMoment2);
        }
    }
}

TEST_F(NonLinearPrefillModelRTest, testFitRetainsWaitingCoefficientsFromSafeModel)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    model._learningRate = 20.0;
    const auto parameters = model.getInitialParameters();
    const auto before = model.getOptimizerState();
    NaviSchedLatencySample sample;
    sample.features = makeNegativeWaitingCostIncidentFeatures();
    sample.latencyMs = 0.0;
    std::vector<double> updated = {123.0};
    ASSERT_TRUE(model.fit({sample}, parameters, updated, 1.0, 1.0, nullptr, true));
    EXPECT_NE(parameters, updated);
    const auto after = model.getOptimizerState();
    EXPECT_EQ(before.step + 1, after.step);
    for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
        const auto index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset;
        EXPECT_DOUBLE_EQ(parameters[index], updated[index]);
        EXPECT_DOUBLE_EQ(before.adamMoment1[index], after.adamMoment1[index]);
        EXPECT_DOUBLE_EQ(before.adamMoment2[index], after.adamMoment2[index]);
    }
}

TEST_F(NonLinearPrefillModelRTest, testTrustedFitCanUpdateWithExistingZeroWaitingCoefficient)
{
    for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
        NonLinearPrefillModelR model;
        model._useCapacitySplitFeature = true;
        auto parameters = model.getInitialParameters();
        for (size_t i = 0; i < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++i) {
            parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + i] = 100.0;
        }
        const auto index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset;
        const auto otherIndex = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET +
                                (offset + 1) % NonLinearPrefillModelR::WAITING_PARAMETER_COUNT;
        parameters[index] = 0.0;
        NonLinearPrefillModelR::OptimizerState before;
        before.step = 17;
        before.adamMoment1[index] = 2.0;
        before.adamMoment2[index] = 3.0;
        before.adamMoment1[otherIndex] = 2.0;
        before.adamMoment2[otherIndex] = 3.0;
        ASSERT_TRUE(model.restoreOptimizerState(before));
        NaviSchedLatencySample sample;
        sample.features = makeNegativeWaitingCostIncidentFeatures();
        sample.latencyMs = NonLinearPrefillModelR::predict(sample.features, parameters);
        std::vector<double> updated;
        ASSERT_TRUE(model.fit({sample}, parameters, updated, 1.0, 1.0, nullptr, true))
            << "zero coefficient index=" << index;
        EXPECT_DOUBLE_EQ(0.0, updated[index]);
        EXPECT_GT(updated[otherIndex], 0.0);
        EXPECT_LT(updated[otherIndex], parameters[otherIndex]);
        EXPECT_EQ(18, model.getOptimizerState().step);
        EXPECT_DOUBLE_EQ(0.0, model.getOptimizerState().adamMoment1[index]);
        EXPECT_DOUBLE_EQ(0.95 * before.adamMoment2[index],
                         model.getOptimizerState().adamMoment2[index]);
    }
}

TEST_F(NonLinearPrefillModelRTest, testTrustedZeroWaitingCoefficientDoesNotFreezePublishedRevision)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    auto parameters = model.getInitialParameters();
    for (size_t i = 0; i < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++i) {
        parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + i] = 0.01;
    }
    const auto index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET +
                       NonLinearPrefillModelR::WAITING_PARAMETER_COUNT - 1;
    parameters[index] = 0.0;
    NonLinearPrefillModelR::OptimizerState optimizer;
    optimizer.step = 1146;
    optimizer.adamMoment1[index] = 2.0;
    optimizer.adamMoment2[index] = 3.0;
    ASSERT_TRUE(model.restoreOptimizerState(optimizer));
    AgentHistoryR history;
    history._fitBatchSize = 1;
    history._fitInterval = 1;
    LatencyModelR latency;
    latency._nonLinearPrefillModelR__ = &model;
    latency._agentHistoryR__ = &history;
    latency._rampFinished = true;
    auto initial = std::make_shared<NaviSchedLatencyModelSnapshot>();
    initial->formula = model.getFormula();
    initial->binaryVersion = model.getBinaryVersion();
    initial->revision = 256;
    initial->parameters = parameters;
    initial->optimizerState = optimizer;
    initial->overloadPredictionTrusted = true;
    ASSERT_TRUE(initial->valid());
    latency.publishSnapshot(initial);
    NaviSchedLatencySample sample;
    sample.features = makeNegativeWaitingCostIncidentFeatures();
    sample.latencyMs = NonLinearPrefillModelR::predict(sample.features, parameters);
    ASSERT_GE(sample.latencyMs, 0.0);
    ASSERT_LT(sample.latencyMs, NonLinearPrefillModelR::MAX_FIT_PREDICTION_MS);
    sample.eventTimeUs = 1;
    bool saveNeeded = false;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
    ASSERT_TRUE(latency.recordSample(sample, updated, saveNeeded));
    ASSERT_TRUE(updated);
    EXPECT_EQ(257, updated->revision);
    EXPECT_TRUE(updated->overloadPredictionTrusted);
    EXPECT_DOUBLE_EQ(0.0, updated->parameters[index]);
    EXPECT_EQ(1147, model.getOptimizerState().step);
    EXPECT_EQ(0, history._pendingFitSampleCount);

    sample.eventTimeUs = 2;
    sample.latencyMs += 1e6;
    ASSERT_TRUE(latency.recordSample(sample, updated, saveNeeded));
    ASSERT_TRUE(updated);
    EXPECT_EQ(258, updated->revision);
    EXPECT_TRUE(updated->overloadPredictionTrusted);
    EXPECT_GT(updated->parameters[index], 0.0);
    EXPECT_EQ(1148, model.getOptimizerState().step);
    EXPECT_EQ(0, history._pendingFitSampleCount);
}

TEST_F(NonLinearPrefillModelRTest, testOnlineWaitingGuardUsesPublishedTrust)
{
    for (const bool trusted : {false, true}) {
        NonLinearPrefillModelR model;
        model._useCapacitySplitFeature = true;
        auto parameters = model.getInitialParameters();
        for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
            parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset] = 1e-10;
        }
        NonLinearPrefillModelR::OptimizerState before;
        before.step = 17;
        for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
            const auto index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset;
            before.adamMoment1[index] = 2.0;
            before.adamMoment2[index] = 3.0;
        }
        ASSERT_TRUE(model.restoreOptimizerState(before));
        AgentHistoryR history;
        LatencyModelR latency;
        latency._nonLinearPrefillModelR__ = &model;
        latency._agentHistoryR__ = &history;
        latency._rampFinished = trusted;
        auto initial = std::make_shared<NaviSchedLatencyModelSnapshot>();
        initial->formula = model.getFormula();
        initial->binaryVersion = model.getBinaryVersion();
        initial->revision = 1024;
        initial->parameters = parameters;
        initial->optimizerState = before;
        initial->overloadPredictionTrusted = trusted;
        ASSERT_TRUE(initial->valid());
        latency.publishSnapshot(initial);
        NaviSchedLatencySample sample;
        sample.features = makeNegativeWaitingCostIncidentFeatures();
        sample.latencyMs = NonLinearPrefillModelR::predict(sample.features, parameters);
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
        ASSERT_TRUE(latency.fit(std::vector<NaviSchedLatencySample>{sample}, updated));
        ASSERT_TRUE(updated);
        EXPECT_EQ(1025, updated->revision);
        EXPECT_EQ(18, model.getOptimizerState().step);
        if (trusted) {
            EXPECT_EQ(parameters, latency.getSnapshot()->parameters);
            EXPECT_EQ(before.adamMoment1, model.getOptimizerState().adamMoment1);
            EXPECT_EQ(before.adamMoment2, model.getOptimizerState().adamMoment2);
        } else {
            EXPECT_FALSE(updated->overloadPredictionTrusted);
            for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
                EXPECT_DOUBLE_EQ(0.0, updated->parameters[
                    NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset]);
            }
        }
    }
}

TEST_F(NonLinearPrefillModelRTest, testGuardedOnlineFitConsumesAttemptedSamples)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    auto parameters = model.getInitialParameters();
    NonLinearPrefillModelR::OptimizerState before;
    before.step = 17;
    for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
        const auto index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset;
        parameters[index] = 1e-10;
        before.adamMoment1[index] = 1e-6;
        before.adamMoment2[index] = 3.0;
    }
    ASSERT_TRUE(model.restoreOptimizerState(before));
    AgentHistoryR history;
    history._fitBatchSize = 1;
    history._fitInterval = 1;
    LatencyModelR latency;
    latency._nonLinearPrefillModelR__ = &model;
    latency._agentHistoryR__ = &history;
    latency._rampFinished = true;
    auto initial = std::make_shared<NaviSchedLatencyModelSnapshot>();
    initial->formula = model.getFormula();
    initial->binaryVersion = model.getBinaryVersion();
    initial->revision = 1024;
    initial->parameters = parameters;
    initial->optimizerState = before;
    initial->overloadPredictionTrusted = true;
    ASSERT_TRUE(initial->valid());
    latency.publishSnapshot(initial);
    NaviSchedLatencySample sample;
    sample.features = makeNegativeWaitingCostIncidentFeatures();
    sample.latencyMs = NonLinearPrefillModelR::predict(sample.features, parameters);
    sample.eventTimeUs = 1;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
    bool saveNeeded = false;
    ASSERT_TRUE(latency.recordSample(sample, updated, saveNeeded));
    ASSERT_TRUE(updated);
    EXPECT_EQ(1025, updated->revision);
    EXPECT_EQ(parameters, updated->parameters);
    EXPECT_EQ(before.step + 1, model.getOptimizerState().step);
    EXPECT_EQ(before.adamMoment1, model.getOptimizerState().adamMoment1);
    EXPECT_EQ(before.adamMoment2, model.getOptimizerState().adamMoment2);
    EXPECT_EQ(0, history._pendingFitSampleCount);
    EXPECT_TRUE(history.getOnlineFitBatches().empty());

    sample.eventTimeUs = 2;
    AgentHistoryR::AppendResult appended;
    ASSERT_TRUE(history.append(sample, NaviSchedLatencyTrainingPhase::NORMAL, appended));
    ASSERT_TRUE(appended.fitReady);
    const auto batches = history.getOnlineFitBatches();
    ASSERT_EQ(1, batches.size());
    EXPECT_EQ(2, batches[0].back().eventTimeUs);
    history.clearPendingFitSamples();
    sample.eventTimeUs = 3;
    sample.latencyMs += 1e6;
    ASSERT_TRUE(latency.recordSample(sample, updated, saveNeeded));
    ASSERT_TRUE(updated);
    EXPECT_EQ(1026, updated->revision);
    EXPECT_EQ(19, model.getOptimizerState().step);
    EXPECT_EQ(0, history._pendingFitSampleCount);
    for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
        EXPECT_GT(updated->parameters[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset], 0.0);
    }
}

TEST_F(NonLinearPrefillModelRTest, testFitAcceptsPositiveWaitingCandidate)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    const auto parameters = model.getInitialParameters();
    NaviSchedLatencySample sample;
    sample.features = makeNegativeWaitingCostIncidentFeatures();
    sample.latencyMs = NonLinearPrefillModelR::predict(sample.features, parameters);
    std::vector<double> updated;
    ASSERT_TRUE(model.fit({sample}, parameters, updated, 1.0, 1.0, nullptr, true));
    EXPECT_EQ(parameters, updated);
    EXPECT_EQ(2, model.getOptimizerState().step);
    for (size_t offset = 0; offset < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++offset) {
        EXPECT_GT(updated[NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + offset], 0.0);
    }
}

TEST_F(NonLinearPrefillModelRTest, testNegativeWorkloadRejectedBySnapshotAndScheduling)
{
    for (bool split : {false, true}) {
        NonLinearPrefillModelR model;
        model._useCapacitySplitFeature = split;
        const auto initial = model.getInitialParameters();
        for (size_t index : {1u, 2u, 3u, 4u, 5u, 10u, 11u, 12u, 13u, 14u}) {
            if (index >= initial.size()) {
                continue;
            }
            SCOPED_TRACE(index);
            NaviSchedLatencyModelSnapshot snapshot;
            snapshot.formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
            snapshot.binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
            snapshot.revision = 1024;
            snapshot.parameters = initial;
            ASSERT_TRUE(snapshot.valid());
            snapshot.parameters[index] = -1e-12;
            EXPECT_FALSE(NonLinearPrefillModelR::validateParameters(snapshot.parameters));
            EXPECT_FALSE(snapshot.valid());
            auto received = std::make_shared<NaviSchedLatencyModel>();
            received->formula = snapshot.formula;
            received->binaryVersion = snapshot.binaryVersion;
            received->revision = snapshot.revision;
            received->parameters = snapshot.parameters;
            received->overloadPredictionTrusted = true;
            NaviSchedNodeSnapshot nodeSnapshot;
            nodeSnapshot.info = std::make_shared<NaviSchedNodeInfo>();
            nodeSnapshot.latencyModel = received;
            NaviSchedScheduleNode node(nullptr, nodeSnapshot, 1);
            EXPECT_FALSE(node.hasValidLatencyModel());
            EXPECT_FALSE(node.isLatencyModelTrusted());
            received->parameters[index] = initial[index];
            EXPECT_TRUE(node.hasValidLatencyModel());
            EXPECT_TRUE(node.isLatencyModelTrusted());
        }
    }
}

TEST_F(NonLinearPrefillModelRTest, testMasterRetainsValidModelAfterNegativeWorkloadReport)
{
    for (size_t index : {5u, 13u}) {
        NonLinearPrefillModelR model;
        model._useCapacitySplitFeature = true;
        AgentLatencyToMasterData initial;
        initial.formula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
        initial.binaryVersion = model.getBinaryVersion();
        initial.modelRevision = 1024;
        initial.parameters = model.getInitialParameters();
        initial.optimizerState = NonLinearPrefillModelR::makeOptimizerStateData(
            model.getOptimizerState(), initial.parameters.size());
        initial.zeroPoint = NonLinearPrefillModelR::calculateZeroPoint(initial.parameters);
        MasterLatencyModelR pool;
        ASSERT_TRUE(pool.activateAgent("agent", "collector"));
        bool accepted = false;
        ASSERT_TRUE(pool.record("agent", "hardware", initial, true, 1, "collector", &accepted));
        ASSERT_TRUE(accepted);
        const auto slot = pool.getLiveModelSlot("agent", "collector");
        ASSERT_TRUE(slot);
        ASSERT_EQ(1024, slot->getModel()->revision);
        auto invalid = initial;
        invalid.modelRevision = 1025;
        invalid.parameters[index] = -1e-12;
        EXPECT_FALSE(pool.record("agent", "hardware", invalid, true, 2, "collector", &accepted));
        EXPECT_FALSE(accepted);
        ASSERT_TRUE(slot->getModel());
        EXPECT_EQ(1024, slot->getModel()->revision);
        EXPECT_EQ(initial.parameters, slot->getModel()->parameters);
        const auto warmup = pool.makeWarmup("hardware", initial.formula,
                                            initial.binaryVersion, initial.parameters.size());
        ASSERT_TRUE(warmup);
        ASSERT_EQ(1, warmup->candidates.size());
        EXPECT_EQ(initial.parameters, warmup->candidates[0].parameters);
    }
}

TEST_F(NonLinearPrefillModelRTest, testAgentRejectsNegativeWarmupWithoutChangingState)
{
    for (bool split : {false, true}) {
        for (size_t index : {1u, 2u, 3u, 4u, 5u, 7u, 9u, 10u, 11u, 12u, 13u, 14u}) {
            NonLinearPrefillModelR model;
            model._useCapacitySplitFeature = split;
            const auto initialParameters = model.getInitialParameters();
            if (index >= initialParameters.size()) {
                continue;
            }
            SCOPED_TRACE(index);
            AgentHistoryR history;
            LatencyModelR latency;
            latency._nonLinearPrefillModelR__ = &model;
            latency._agentHistoryR__ = &history;
            latency._latencyModelFingerprint = "incident-workload";
            NonLinearPrefillModelR::OptimizerState initialOptimizer;
            initialOptimizer.step = 7;
            initialOptimizer.adamMoment1[index] = 0.25;
            initialOptimizer.adamMoment2[index] = 0.5;
            ASSERT_TRUE(model.restoreOptimizerState(initialOptimizer));
            auto initial = std::make_shared<NaviSchedLatencyModelSnapshot>();
            initial->formula = model.getFormula();
            initial->binaryVersion = model.getBinaryVersion();
            initial->revision = 1;
            initial->parameters = initialParameters;
            initial->optimizerState = initialOptimizer;
            ASSERT_TRUE(initial->valid());
            latency.publishSnapshot(initial);

            MasterLatencyModelWarmupToAgentData warmup;
            warmup.fingerprint = "incident-workload";
            warmup.formula = model.getFormula();
            warmup.binaryVersion = model.getBinaryVersion();
            NaviSchedLatencyModelCandidate candidate;
            candidate.sourceRevision = 1024;
            candidate.parameters = initialParameters;
            candidate.parameters[index] = -1e-12;
            NonLinearPrefillModelR::OptimizerState importedOptimizer;
            importedOptimizer.step = 41;
            importedOptimizer.adamMoment1[index] = -0.125;
            importedOptimizer.adamMoment2[index] = 0.75;
            candidate.optimizerState = NonLinearPrefillModelR::makeOptimizerStateData(
                importedOptimizer, initialParameters.size());
            warmup.candidates = {candidate};
            // Wire-format validity alone does not validate the model's workload response.
            ASSERT_TRUE(warmup.valid());
            std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
            EXPECT_FALSE(latency.applyColdStartCandidates(warmup, updated));
            EXPECT_FALSE(updated);
            EXPECT_EQ(initial, latency.getSnapshot());
            EXPECT_EQ(initialParameters, latency.getSnapshot()->parameters);
            EXPECT_EQ(initialOptimizer.step, model.getOptimizerState().step);
            EXPECT_EQ(initialOptimizer.adamMoment1, model.getOptimizerState().adamMoment1);
            EXPECT_EQ(initialOptimizer.adamMoment2, model.getOptimizerState().adamMoment2);
            EXPECT_EQ(0, latency._coldStartAppliedCount.load());
            EXPECT_EQ(0, latency._coldStartSelectedCount.load());

            warmup.candidates[0].parameters[index] = initialParameters[index];
            ASSERT_TRUE(latency.applyColdStartCandidates(warmup, updated));
            ASSERT_TRUE(updated);
            EXPECT_EQ(warmup.candidates[0].parameters, updated->parameters);
            EXPECT_EQ(1, updated->revision);
            EXPECT_TRUE(updated->warmStarted);
            EXPECT_FALSE(updated->overloadPredictionTrusted);
            EXPECT_EQ(importedOptimizer.step, model.getOptimizerState().step);
            EXPECT_EQ(importedOptimizer.adamMoment1, model.getOptimizerState().adamMoment1);
            EXPECT_EQ(importedOptimizer.adamMoment2, model.getOptimizerState().adamMoment2);
            EXPECT_EQ(1, latency._coldStartAppliedCount.load());
            EXPECT_EQ(1, latency._coldStartSelectedCount.load());
        }
    }
}

TEST_F(NonLinearPrefillModelRTest, testWorkloadUpdateCannotCrossIntoNegativeCost)
{
    const std::array<size_t, 10> indexes = {1, 2, 3, 4, 5, 10, 11, 12, 13, 14};
    for (double learningRate : {0.001, 0.05, 2.0}) {
        NonLinearPrefillModelR model;
        model._useCapacitySplitFeature = true;
        model._useBatchCountFeature = false;
        model._learningRate = learningRate;
        auto parameters = model.getInitialParameters();
        auto optimizer = model.getOptimizerState();
        optimizer.step = 41;
        for (auto index : indexes) {
            parameters[index] = std::max(1e-10, model.positiveParameterFloor(index, 1.0) * 1.001);
            optimizer.adamMoment1[index] = 1.0;
            optimizer.adamMoment2[index] = 1.0;
        }
        ASSERT_TRUE(model.restoreOptimizerState(optimizer));
        NaviSchedLatencySample sample;
        ASSERT_TRUE(sample.features.appendCapacitySplitAggregate(
            1, 1, 3840, 64, 4096, 245760, 1, 3840, 64, 4096, 245760));
        ASSERT_TRUE(sample.features.appendPrecedingRequestCount(256));
        sample.latencyMs = 0.0;
        std::vector<double> updated;
        ASSERT_TRUE(model.fit({sample}, parameters, updated, 1.0, 1.0, nullptr, true));
        const auto after = model.getOptimizerState();
        EXPECT_EQ(optimizer.step + 1, after.step);
        for (size_t index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET;
             index < NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET +
                         NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++index) {
            EXPECT_DOUBLE_EQ(parameters[index], updated[index]);
            EXPECT_DOUBLE_EQ(optimizer.adamMoment1[index], after.adamMoment1[index]);
            EXPECT_DOUBLE_EQ(optimizer.adamMoment2[index], after.adamMoment2[index]);
        }
        for (int64_t prompt : {0, 3890, 5754, 8044, 65536}) {
            for (int64_t hit : {int64_t{0}, prompt / 2, prompt}) {
                EXPECT_GE(NonLinearPrefillModelR::calculateRequestLinearCost(updated, prompt, hit), 0.0);
                EXPECT_GE(NonLinearPrefillModelR::calculateWaitingRequestLinearCost(updated, prompt, hit), 0.0);
            }
        }
    }
}

TEST_F(NonLinearPrefillModelRTest, testWorkloadCoefficientsHaveZeroFloor)
{
    NonLinearPrefillModelR model;
    for (const auto index : NonLinearPrefillModelR::POSITIVE_PARAMETER_INDEXES) {
        EXPECT_DOUBLE_EQ(0.0, model.positiveParameterFloor(index, 1.0)) << index;
        EXPECT_DOUBLE_EQ(0.0, model.positiveParameterFloor(index, 2.0)) << index;
    }
    EXPECT_GT(NonLinearPrefillModelR::MIN_PRECEDING_REQUEST_COEFFICIENT_MS, 0.0);
}

TEST_F(NonLinearPrefillModelRTest, testWorkloadProjectionPreservesUsefulMomentumAndOtherDimensions)
{
    NonLinearPrefillModelR model;
    model._useCapacitySplitFeature = true;
    model._useBatchCountFeature = false;
    NonLinearPrefillModelR::Values weights{}, m1{}, m2{};
    weights.fill(3.0);
    m1.fill(-2.0);
    m2.fill(7.0);
    const std::array<size_t, 10> indexes = {1, 2, 3, 4, 5, 10, 11, 12, 13, 14};
    for (auto index : indexes) {
        weights[index] = -0.01;
    }
    weights[0] = -10.0;
    weights[16] = -20.0;
    model.projectOutputParameters(weights, m1, m2, 1e-6);
    for (auto index : indexes) {
        EXPECT_DOUBLE_EQ(model.positiveParameterFloor(index, 1.0),
                         weights[index]) << index;
        EXPECT_DOUBLE_EQ(-2.0, m1[index]) << index;
        EXPECT_DOUBLE_EQ(7.0, m2[index]) << index;
    }
    EXPECT_DOUBLE_EQ(-10.0, weights[0]);
    EXPECT_DOUBLE_EQ(-20.0, weights[16]);
    EXPECT_DOUBLE_EQ(0.0, weights[6]);
    EXPECT_DOUBLE_EQ(0.0, m1[6]);
    EXPECT_DOUBLE_EQ(0.0, m2[6]);
}

TEST_F(NonLinearPrefillModelRTest, testParameterScalesRoundTripEveryFormulaCoefficient)
{
    const std::array<double, NonLinearPrefillModelR::MAX_PARAMETER_COUNT>
        formulaCoefficients = {
            -37.25, 0.125, 2.5, -8.0, 31.0, 0.0075, 4.0, 310.0, 0.004,
            0.75,  -1.25, 6.5, 128.0, 0.03125, -0.5, 5.0, 42.0, 3.0,
            2048.0,
        };
    for (size_t index = 0; index < formulaCoefficients.size(); ++index) {
        const auto scale = NonLinearPrefillModelR::getParameterScale(index);
        ASSERT_TRUE(std::isfinite(scale)) << index;
        ASSERT_GT(scale, 0.0) << index;
        const auto parameter =
            parameterForFormulaCoefficient(index, formulaCoefficients[index]);
        EXPECT_NEAR(formulaCoefficients[index], parameter * scale,
                    1e-13 * std::max(1.0, std::abs(formulaCoefficients[index])))
            << index;
    }
    EXPECT_DOUBLE_EQ(
        0.0,
        NonLinearPrefillModelR::getParameterScale(
            NonLinearPrefillModelR::MAX_PARAMETER_COUNT));
}

TEST_F(NonLinearPrefillModelRTest, testScaledFormulaMatchesOfflineFixedSample)
{
    NaviSchedLatencyFeatures features;
    ASSERT_TRUE(features.appendCapacitySplitAggregate(
        2, 3, 1024, 2048, 4194304.0, 2097152.0, 4, 512, 3072,
        9437184.0, 1572864.0));
    ASSERT_TRUE(features.appendPrecedingRequestCount(5));
    ASSERT_TRUE(features.appendRunningElapsedTimeMs(250000.0));
    const std::vector<double> parameters = {
        4.0, 20.0, 4.0, 3.0, 2.0, 1.0, 0.0, 100.0, 10.0,
        1.0, 12.0, 5.0, 3.0, 2.0, 1.0,
        parameterForFormulaCoefficient(15, 1.0), 2.0, 0.0, 0.0,
    };
    EXPECT_NEAR(66741.79706653491,
                NonLinearPrefillModelR::predict(features, parameters), 1e-8);
}

TEST_F(NonLinearPrefillModelRTest,
       testFull19ParameterFixedSampleAndJacobian)
{
    NaviSchedLatencyFeatures features;
    ASSERT_TRUE(features.appendCapacitySplitAggregate(
        2, 3, 1024, 2048, 4194304.0, 2097152.0, 4, 512, 3072,
        9437184.0, 1572864.0));
    ASSERT_TRUE(features.appendPrecedingRequestCount(5));
    ASSERT_TRUE(features.appendRunningElapsedTimeMs(250000.0));
    ASSERT_TRUE(features.appendRunningDecodeProgress(6, 8192));
    const std::vector<double> parameters = {
        4.0, 20.0, 4.0, 3.0, 2.0, 1.0, 0.75, 100.0, 10.0,
        1.0, 12.0, 5.0, 3.0, 2.0, 1.0,
        parameterForFormulaCoefficient(15, 1.0), 2.0, 7.0, 11.0,
    };
    // Independently evaluated with 70-digit Decimal arithmetic and differences.
    const std::vector<double> expectedGradient = {
        3.821889975911229, 3.348373534738624, 0.826014753940311,
        45.99139701676822, 34.43808249247954, 366.8333775144461,
        18.74978529136965, 1.23972355201, -11.26598670418203,
        1.371576322558, 41.1584918557375, 4.119294428471385,
        5454.204820414845, 18756.81025484373, 2917.711744062623,
        0.121433822941, 4434.542467222233, 1.684116606731721,
        30.29069415424376,
    };
    const auto input = NonLinearPrefillModelR::makeInput(features);
    for (const auto value : input) {
        ASSERT_GT(value, 0.0);
    }
    ASSERT_EQ(19, parameters.size());
    auto weights = NonLinearPrefillModelR::toValues(parameters);
    NonLinearPrefillModelR::Values gradient{};
    EXPECT_NEAR(67100.84583536643,
                NonLinearPrefillModelR::predict(input, weights, 19, &gradient),
                1e-8);
    EXPECT_NEAR(67100.84583536643,
                NonLinearPrefillModelR::predict(features, parameters), 1e-8);
    for (size_t index = 0; index < parameters.size(); ++index) {
        SCOPED_TRACE(index);
        ASSERT_NE(0.0, parameters[index]);
        EXPECT_NEAR(expectedGradient[index], gradient[index],
                    1e-10 * std::max(1.0, std::abs(expectedGradient[index])));
        const auto delta = 1e-4 * std::max(1.0, std::abs(weights[index]));
        auto plus = weights;
        auto minus = weights;
        plus[index] += delta;
        minus[index] -= delta;
        const auto numerical =
            (NonLinearPrefillModelR::predict(input, plus, 19, nullptr) -
             NonLinearPrefillModelR::predict(input, minus, 19, nullptr)) /
            (2.0 * delta);
        EXPECT_NEAR(gradient[index], numerical,
                    1e-6 * std::max(1.0, std::abs(gradient[index])));
    }
    for (size_t index : {17, 18}) {
        auto disabled = parameters;
        disabled[index] = 0.0;
        EXPECT_LT(NonLinearPrefillModelR::predict(features, disabled),
                  NonLinearPrefillModelR::predict(features, parameters));
    }
}

TEST_F(NonLinearPrefillModelRTest,
       testEquivalentEffectiveLearningRatesProduceIdenticalAdamTrajectory)
{
    NonLinearPrefillModelR directModel;
    NonLinearPrefillModelR factoredModel;
    directModel._learningRate = 0.05;
    factoredModel._learningRate = 0.01;
    auto directParameters = directModel.getInitialParameters();
    auto factoredParameters = factoredModel.getInitialParameters();

    for (size_t step = 0; step < 8; ++step) {
        std::vector<NaviSchedLatencySample> samples(4);
        for (size_t index = 0; index < samples.size(); ++index) {
            samples[index].features.appendRequest(
                128 * (step + index), 1024 + 257 * step + 131 * index);
            samples[index].latencyMs = 400.0 + 73.0 * step + 29.0 * index;
        }
        std::vector<double> directUpdated;
        std::vector<double> factoredUpdated;
        ASSERT_TRUE(directModel.fit(samples, directParameters, directUpdated, 1.0));
        ASSERT_TRUE(factoredModel.fit(samples, factoredParameters, factoredUpdated,
                                      5.0));
        EXPECT_EQ(directUpdated, factoredUpdated) << step;
        const auto directOptimizer = directModel.getOptimizerState();
        const auto factoredOptimizer = factoredModel.getOptimizerState();
        EXPECT_EQ(directOptimizer.step, factoredOptimizer.step) << step;
        EXPECT_EQ(directOptimizer.adamMoment1, factoredOptimizer.adamMoment1) << step;
        EXPECT_EQ(directOptimizer.adamMoment2, factoredOptimizer.adamMoment2) << step;
        directParameters = std::move(directUpdated);
        factoredParameters = std::move(factoredUpdated);
    }
}

TEST_F(NonLinearPrefillModelRTest,
       testDifferentAndInvalidEffectiveLearningRatesDoNotAliasBaseline)
{
    NonLinearPrefillModelR baselineModel;
    NonLinearPrefillModelR fasterModel;
    baselineModel._learningRate = 0.05;
    fasterModel._learningRate = 0.05;
    const auto initialParameters = baselineModel.getInitialParameters();
    NaviSchedLatencySample sample;
    sample.features.appendRequest(2048, 8192);
    sample.latencyMs = 20'000.0;

    std::vector<double> baseline;
    std::vector<double> faster;
    ASSERT_TRUE(baselineModel.fit({sample}, initialParameters, baseline, 1.0));
    ASSERT_TRUE(fasterModel.fit({sample}, initialParameters, faster, 1.1));
    EXPECT_NE(baseline, faster);
    EXPECT_NE(NonLinearPrefillModelR::predict(sample.features, baseline),
              NonLinearPrefillModelR::predict(sample.features, faster));

    NonLinearPrefillModelR rejectedModel;
    const auto optimizerBefore = rejectedModel.getOptimizerState();
    std::vector<double> untouched = {123.0};
    EXPECT_FALSE(rejectedModel.fit({sample}, initialParameters, untouched, 0.0));
    EXPECT_EQ(std::vector<double>({123.0}), untouched);
    EXPECT_FALSE(rejectedModel.fit(
        {sample}, initialParameters, untouched,
        std::numeric_limits<double>::quiet_NaN()));
    EXPECT_EQ(std::vector<double>({123.0}), untouched);
    const auto optimizerAfter = rejectedModel.getOptimizerState();
    EXPECT_EQ(optimizerBefore.step, optimizerAfter.step);
    EXPECT_EQ(optimizerBefore.adamMoment1, optimizerAfter.adamMoment1);
    EXPECT_EQ(optimizerBefore.adamMoment2, optimizerAfter.adamMoment2);
}

TEST_F(NonLinearPrefillModelRTest, testOptimizerStepDiagnosticsMatchAdamUpdate)
{
    NonLinearPrefillModelR model;
    const auto initialParameters = model.getInitialParameters();
    NaviSchedLatencySample sample;
    sample.features.appendRequest(1024, 2048);
    sample.latencyMs = 900.0;

    const auto input = NonLinearPrefillModelR::makeInput(sample.features);
    const auto weights = NonLinearPrefillModelR::toValues(initialParameters);
    NonLinearPrefillModelR::Values predictionGradient{};
    const auto prediction = NonLinearPrefillModelR::predict(
        input, weights, initialParameters.size(), &predictionGradient);
    const auto lossDerivative =
        (std::log1p(prediction) - std::log1p(sample.latencyMs)) /
        (1.0 + prediction);

    std::vector<double> updatedParameters;
    ASSERT_TRUE(model.fit({sample}, initialParameters, updatedParameters));
    const auto diagnostics = model.getOptimizerStepDiagnostics();
    ASSERT_EQ(initialParameters.size(), diagnostics.parameterCount);
    EXPECT_EQ(1, diagnostics.step);

    constexpr double BETA_1 = 0.9;
    constexpr double BETA_2 = 0.95;
    constexpr double EPSILON = 1e-20;
    const auto correction =
        model._learningRate * std::sqrt(1.0 - BETA_2) / (1.0 - BETA_1);
    for (size_t index = 0; index < diagnostics.parameterCount; ++index) {
        const auto expectedGradient = lossDerivative * predictionGradient[index];
        const auto expectedMoment1 = (1.0 - BETA_1) * expectedGradient;
        const auto expectedMoment2 =
            (1.0 - BETA_2) * expectedGradient * expectedGradient;
        EXPECT_NEAR(expectedGradient, diagnostics.rawGradient[index],
                    1e-12 * std::max(1.0, std::abs(expectedGradient)))
            << index;
        EXPECT_NEAR(expectedMoment1, diagnostics.adamMoment1[index],
                    1e-12 * std::max(1.0, std::abs(expectedMoment1)))
            << index;
        EXPECT_NEAR(expectedMoment2, diagnostics.adamMoment2[index],
                    1e-12 * std::max(1.0, std::abs(expectedMoment2)))
            << index;
        const auto expectedNormalized =
            expectedMoment1 / std::sqrt(expectedMoment2 + EPSILON);
        EXPECT_NEAR(expectedNormalized, diagnostics.normalizedGradient[index],
                    1e-12 * std::max(1.0, std::abs(expectedNormalized)))
            << index;
        EXPECT_NEAR(-correction * expectedNormalized,
                    diagnostics.candidateUpdate[index], 1e-12)
            << index;
        EXPECT_DOUBLE_EQ(updatedParameters[index] - initialParameters[index],
                         diagnostics.actualUpdate[index])
            << index;

        auto oneParameterUpdated = initialParameters;
        oneParameterUpdated[index] = updatedParameters[index];
        EXPECT_NEAR(NonLinearPrefillModelR::predict(sample.features,
                                                    oneParameterUpdated) -
                        prediction,
                    diagnostics.predictionShiftMs[index], 1e-9)
            << index;
    }
}

TEST_F(NonLinearPrefillModelRTest, testRestoringOptimizerClearsStepDiagnostics)
{
    NonLinearPrefillModelR model;
    const auto parameters = model.getInitialParameters();
    NaviSchedLatencySample sample;
    sample.features.appendRequest(0, 1024);
    sample.latencyMs = 700.0;
    std::vector<double> updatedParameters;
    ASSERT_TRUE(model.fit({sample}, parameters, updatedParameters));
    ASSERT_NE(0, model.getOptimizerStepDiagnostics().parameterCount);

    ASSERT_TRUE(model.restoreOptimizerState(model.getOptimizerState()));
    EXPECT_EQ(0, model.getOptimizerStepDiagnostics().parameterCount);
    EXPECT_EQ(0, model.getOptimizerStepDiagnostics().step);
}
