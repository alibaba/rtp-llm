#include <limits>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "aios/navi_sched/common/TPUPrefillPP4Timing.h"
#include "unittest/unittest.h"

namespace navi_sched {
namespace {

std::vector<TPUPrefillPipelineSlot> fourSlots()
{
    std::vector<TPUPrefillPipelineSlot> result;
    for (int64_t ordinal = 0; ordinal < 4; ++ordinal) {
        TPUPrefillPipelineSlot slot;
        slot.ordinal = ordinal;
        slot.dispatchTimeMs = ordinal * 10.0;
        slot.features[0] = 1.0;
        result.push_back(slot);
    }
    return result;
}

TPUPrefillPP4Timing::Parameters constantStageCost()
{
    return {3.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

} // namespace

class TPUPrefillPP4TimingTest : public TESTBASE {};

TEST_F(TPUPrefillPP4TimingTest, emptyPrologueUsesThreeHandoffsAndOneFinalStage)
{
    TPUPrefillPP4Timing model(constantStageCost(), fourSlots());
    EXPECT_EQ(0.0, model.stepCostMs(-1));
    EXPECT_EQ(5.0, model.stepCostMs(0));
    EXPECT_EQ(25.0, model.lastStageReplyTimeMs(0));
    EXPECT_EQ(33.0, model.resultReadyTimeMs(0));
}

TEST_F(TPUPrefillPP4TimingTest, observedReplyAndPendingReplyBoundUseDifferentMilestones)
{
    TPUPrefillPP4Timing model(constantStageCost(), fourSlots(), {{0, 100.0}},
                              {{1, 120.0}});
    EXPECT_EQ(100.0, model.lastStageReplyTimeMs(0));
    EXPECT_EQ(120.0, model.lastStageReplyTimeMs(1));
    EXPECT_EQ(128.0, model.resultReadyTimeMs(1));
}

TEST_F(TPUPrefillPP4TimingTest, futureSlotsExtendTheSameCapturedPipeline)
{
    TPUPrefillPP4Timing model(constantStageCost(), fourSlots(), {{0, 100.0}},
                              {{1, 120.0}});
    EXPECT_EQ(128.0, model.resultReadyTimeMs(1));
    for (int64_t ordinal : {4, 5}) {
        TPUPrefillPipelineSlot slot;
        slot.ordinal = ordinal;
        slot.dispatchTimeMs = ordinal * 10.0;
        slot.features[0] = 1.0;
        model.appendSlot(slot);
    }
    EXPECT_EQ(130.0, model.lastStageReplyTimeMs(3));
    EXPECT_EQ(138.0, model.resultReadyTimeMs(3));
    EXPECT_EQ(120.0, model.lastStageReplyTimeMs(1));
}

TEST_F(TPUPrefillPP4TimingTest, aPushHasZeroComputeCost)
{
    auto slots = fourSlots();
    slots[2].features = {};
    TPUPrefillPP4Timing model(constantStageCost(), slots);
    EXPECT_EQ(0.0, model.stepCostMs(2));
    EXPECT_EQ(5.0, model.stepCostMs(1));
}

TEST_F(TPUPrefillPP4TimingTest, rejectsMissingHistoryRatherThanInventingZeroWork)
{
    TPUPrefillPP4Timing model(constantStageCost(), fourSlots());
    EXPECT_THROW(model.resultReadyTimeMs(2), std::out_of_range);
    EXPECT_THROW(model.stepCostMs(7), std::out_of_range);
    auto slots = fourSlots();
    slots.erase(slots.begin());
    TPUPrefillPP4Timing partial(constantStageCost(), slots);
    EXPECT_THROW(partial.lastStageReplyTimeMs(1), std::out_of_range);
}

TEST_F(TPUPrefillPP4TimingTest, anInvalidAppendDoesNotChangeTheForecast)
{
    TPUPrefillPP4Timing model(constantStageCost(), fourSlots());
    TPUPrefillPipelineSlot slot;
    slot.ordinal = 5;
    slot.dispatchTimeMs = 50.0;
    slot.features[0] = 1.0;
    EXPECT_THROW(model.appendSlot(slot), std::invalid_argument);
    EXPECT_EQ(3, model.lastOrdinal());
    EXPECT_EQ(33.0, model.resultReadyTimeMs(0));
}

TEST_F(TPUPrefillPP4TimingTest,
       rejectsInvalidCoefficientsFeaturesAndContradictoryObservations)
{
    auto parameters = constantStageCost();
    parameters[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((TPUPrefillPP4Timing(parameters, fourSlots())), std::invalid_argument);
    auto slots = fourSlots();
    slots[1].features[4] = -1.0;
    EXPECT_THROW((TPUPrefillPP4Timing(constantStageCost(), slots)),
                 std::invalid_argument);
    EXPECT_THROW((TPUPrefillPP4Timing(constantStageCost(), fourSlots(), {{0, 100.0}},
                                      {{0, 101.0}})),
                 std::invalid_argument);
}

TEST_F(TPUPrefillPP4TimingTest, unobservedEmptyWaveAcceptsItsFirstDispatch)
{
    TPUPrefillPP4Timing model(constantStageCost(), {});
    EXPECT_EQ(-1, model.lastOrdinal());
    auto slots = fourSlots();
    model.appendSlot(slots[0]);
    EXPECT_EQ(0, model.lastOrdinal());
    EXPECT_EQ(5.0, model.stepCostMs(0));
    EXPECT_THROW(model.resultReadyTimeMs(0), std::out_of_range);
}

TEST_F(TPUPrefillPP4TimingTest, CompleteTrainingSupportGradientMatchesFiniteDifferences)
{
    TPUPrefillPP4TimingObservation observation;
    observation.targetOrdinal = 3;
    for (int64_t ordinal = 0; ordinal < 6; ++ordinal) {
        TPUPrefillPipelineSlot slot;
        slot.ordinal = ordinal;
        slot.dispatchTimeMs = ordinal * 0.3;
        slot.features = {
            1., 1. + ordinal, 2. + ordinal, 0.2 * ordinal, 0.1 * ordinal, 0.3 * ordinal,
            1., 0.4 * ordinal};
        observation.slots.push_back(slot);
    }
    ASSERT_TRUE(observation.valid());
    TPUPrefillPP4Timing::Parameters weights{2., 10., 1., 0.5, 0.2, 0.1, 0.3, 0.4, 0.5},
        gradient;
    const auto value = observation.predictLatencyMs(weights, &gradient);
    TPUPrefillPP4Timing reference(weights, observation.slots);
    EXPECT_DOUBLE_EQ(reference.resultReadyTimeMs(3) - observation.slots[3].dispatchTimeMs,
                     value);
    for (size_t index = 0; index < weights.size(); ++index) {
        auto lower = weights, upper = weights;
        lower[index] -= 1e-5;
        upper[index] += 1e-5;
        const double numerical =
            (observation.predictLatencyMs(upper) - observation.predictLatencyMs(lower)) /
            2e-5;
        EXPECT_NEAR(numerical, gradient[index], 1e-6) << index;
    }
    observation.slots.erase(observation.slots.begin());
    EXPECT_FALSE(observation.valid());
}

TEST_F(TPUPrefillPP4TimingTest, NativeOverlapLatePushDoesNotChargeOlderStagesTwice)
{
    auto slots = fourSlots();
    slots[2].dispatchTimeMs = 100.0; slots[2].features = {};
    slots[3].dispatchTimeMs = 200.0; slots[3].features = {};
    TPUPrefillPP4Timing model(constantStageCost(), slots);
    // Stages processing older forwards finish before the delayed push arrives.
    EXPECT_DOUBLE_EQ(100.0, model.lastStageReplyTimeMs(0));
    EXPECT_DOUBLE_EQ(108.0, model.resultReadyTimeMs(0));
}

TEST_F(TPUPrefillPP4TimingTest, NativeOverlapObservedReplyRetainsConcurrentOlderWork)
{
    auto slots = fourSlots();
    slots[2].dispatchTimeMs = 100.0; slots[2].features = {};
    slots[3].dispatchTimeMs = 200.0; slots[3].features = {};
    TPUPrefillPP4Timing model(constantStageCost(), slots, {{0, 100.0}});
    EXPECT_DOUBLE_EQ(200.0, model.lastStageReplyTimeMs(1));
    EXPECT_DOUBLE_EQ(208.0, model.resultReadyTimeMs(1));
}

TEST_F(TPUPrefillPP4TimingTest, NativeOverlapTrainingGradientFollowsDelayedStageZero)
{
    TPUPrefillPP4TimingObservation sample;
    sample.targetOrdinal = 0; sample.slots = fourSlots(); sample.slots.resize(3);
    sample.slots[2].dispatchTimeMs = 100.0; sample.slots[2].features = {};
    const auto weights = constantStageCost(); TPUPrefillPP4Timing::Parameters gradient;
    ASSERT_TRUE(sample.valid());
    EXPECT_DOUBLE_EQ(108.0, sample.predictLatencyMs(weights, &gradient));
    // The final stage still costs five milliseconds; older stages overlap
    // the late dispatch, so their coefficients have no additional contribution.
    EXPECT_DOUBLE_EQ(1.0, gradient[0]); EXPECT_DOUBLE_EQ(1.0, gradient[1]);
    for (size_t index = 2; index < gradient.size(); ++index) EXPECT_DOUBLE_EQ(0.0, gradient[index]);
    for (size_t index : {size_t(0), size_t(1)}) {
        auto lower=weights, upper=weights; lower[index]-=1e-5; upper[index]+=1e-5;
        EXPECT_NEAR((sample.predictLatencyMs(upper)-sample.predictLatencyMs(lower))/2e-5,
                    gradient[index], 1e-6);
    }
}

TEST_F(TPUPrefillPP4TimingTest, NativeOverlapExpressionAndGradientAcrossDispatchGaps)
{
    // Independent reference from vllm_torchtpu/core/pp_lazy_step.py::_handoff
    // and _others (serving source e53fb9d273c208d36ae0d9620b3b649392ae24b33b6fdb29e6a5f6c7c896d697).
    // Keep host dispatch gaps separate from the other stages' concurrent work.
    const auto nativeLatency = [](const TPUPrefillPP4TimingObservation& observation,
                                  const TPUPrefillPP4Timing::Parameters& weights) {
        std::vector<double> costs;
        for (const auto& slot : observation.slots) {
            double cost = 0.0;
            for (size_t i = 0; i < slot.features.size(); ++i)
                cost += slot.features[i] * weights[i + 1];
            costs.push_back(cost);
        }
        const auto target = observation.targetOrdinal;
        const double start = observation.slots[target].dispatchTimeMs;
        double clock = start;
        for (int64_t j = target; j < target + 3; ++j) {
            double others = 0.0;
            for (int64_t at = std::max<int64_t>(0, j - 3); at < j; ++at)
                others = std::max(others, costs[at]);
            clock = std::max(clock + others,
                             std::max(clock, observation.slots[j].dispatchTimeMs) + costs[j]);
        }
        return clock + costs[target] + weights[0] - start;
    };
    std::mt19937 generator(20261125);
    std::uniform_real_distribution<double> positive(0.1, 5.0), gap(0.0, 100.0);
    size_t checkedGradients = 0;
    for (int sample = 0; sample < 1000; ++sample) {
        TPUPrefillPP4TimingObservation observation;
        observation.targetOrdinal = generator() % 4;
        double clock = 0.0;
        for (int64_t ordinal = 0; ordinal < observation.targetOrdinal + 3; ++ordinal) {
            clock += generator() % 2 ? gap(generator) : 0.0;
            TPUPrefillPipelineSlot slot;
            slot.ordinal = ordinal;
            slot.dispatchTimeMs = clock;
            if (ordinal == observation.targetOrdinal || generator() % 5) {
                slot.features[0] = 1.0;
                for (size_t i = 1; i < slot.features.size(); ++i)
                    slot.features[i] = positive(generator);
            }
            observation.slots.push_back(slot);
        }
        TPUPrefillPP4Timing::Parameters weights, gradient;
        for (double& weight : weights) weight = positive(generator);
        ASSERT_TRUE(observation.valid());
        const double expected = nativeLatency(observation, weights);
        EXPECT_NEAR(expected, observation.predictLatencyMs(weights, &gradient), 1e-8)
            << "sample=" << sample;
        for (size_t i = 0; i < weights.size(); ++i) {
            auto low = weights, high = weights;
            low[i] -= 1e-5; high[i] += 1e-5;
            const double left = (expected - nativeLatency(observation, low)) / 1e-5;
            const double right = (nativeLatency(observation, high) - expected) / 1e-5;
            if (std::abs(left - right) > 1e-4) continue; // max() has no unique gradient at a kink.
            ++checkedGradients;
            EXPECT_NEAR((left + right) / 2.0, gradient[i], 1e-4)
                << "sample=" << sample << " parameter=" << i;
        }
    }
    EXPECT_GT(checkedGradients, size_t(500));
}

} // namespace navi_sched
