#include "aios/navi_sched/common/TPUPrefillPP4Forecast.h"
#include "unittest/unittest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace navi_sched {
namespace {

TPUPrefillExecutionPolicyConfig policy()
{
    TPUPrefillExecutionPolicyConfig value;
    value.pcpSize = 8;
    value.interleaveSize = 256;
    value.compileSizes = {256, 512, 1024};
    value.maxScheduledTokens = 8192;
    value.hashBlockSize = value.schedulerBlockSize = 4096;
    return value;
}

TPUPrefillPP4Request request(uint64_t id, int64_t remaining, bool owned = true)
{
    TPUPrefillPP4Request value;
    value.requestId = id;
    value.remainingTokens = remaining;
    value.owned = owned;
    value.releaseSlotAfterPrefill = true;
    return value;
}

const TPUPrefillPP4Timing::Parameters PARAMETERS{0.0, 10.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0,  0.0, 0.0};

} // namespace

TEST(TPUPrefillPP4ForecastTest, ClosedWaveOpensWithComputeAndWaitsForReadiness)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 1);
    own.readyTimeMs = 100.0;
    auto result =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 100, true);
    ASSERT_EQ(size_t(3), result.dispatches.size());
    ASSERT_TRUE(result.dispatches[0].real);
    ASSERT_EQ(0, result.dispatches[0].ordinal);
    ASSERT_DOUBLE_EQ(100.0, result.dispatches[0].timeMs);
    ASSERT_DOUBLE_EQ(115.0, result.dispatches[1].timeMs);
    ASSERT_DOUBLE_EQ(125.0, result.dispatches[2].timeMs);
    ASSERT_DOUBLE_EQ(100.0, result.queueMs);
    ASSERT_DOUBLE_EQ(40.0, result.executionMs);
    ASSERT_DOUBLE_EQ(140.0, result.totalMs);
    ASSERT_EQ(int64_t(1), result.scheduledTokens.at(1));
    ASSERT_TRUE(state.slots.empty());
}

TEST(TPUPrefillPP4ForecastTest, ExplicitFutureReportersCanDeliverSameOrLaterCalls)
{
    std::vector<double> starts;
    for (const int64_t reporter : {int64_t(1), int64_t(2), int64_t(3)}) {
        TPUPrefillPP4State state;
        state.maxRunningRequestCount = 64;
        state.measuredForwardMs = 10.0;
        auto prior = request(1, 8193);
        prior.owned = false;
        prior.queue = TPUPrefillForecastQueue::RUNNING;
        auto own = request(2, 8192);
        own.receiveDependency = TPUPrefillKVReceiveDependency{-1, reporter, -1};
        auto first = request(3, 1);
        first.owned = false;
        first.readyTimeMs = 1.0;
        auto second = request(4, 1);
        second.owned = false;
        second.readyTimeMs = 2.0;
        TPUPrefillPP4ForecastOutput result;
        ASSERT_NO_THROW(
            result = TPUPrefillPP4Forecast::predict(state, {prior, own, first, second},
                                                    policy(), PARAMETERS, 1000, true));
        ASSERT_EQ(8192, result.scheduledTokens.at(2));
        bool carrierFound = false, reporterFound = false;
        for (const auto &call : result.dispatches) {
            if (call.loadMetadataRequestIds == std::vector<uint64_t>{2}) {
                EXPECT_EQ(1, call.ordinal);
                carrierFound = true;
            }
            if (call.ordinal == reporter) {
                ASSERT_TRUE(call.real);
                EXPECT_GT(result.firstScheduledTimeMs.at(2), call.timeMs);
                reporterFound = true;
            }
        }
        ASSERT_TRUE(carrierFound);
        ASSERT_TRUE(reporterFound);
        starts.push_back(result.firstScheduledTimeMs.at(2));
        ASSERT_TRUE(state.slots.empty());
        ASSERT_TRUE(own.receiveDependency->decidedProposalFence.has_value());
    }
    ASSERT_LT(starts[0], starts[1]);
    ASSERT_LT(starts[1], starts[2]);
}

TEST(TPUPrefillPP4ForecastTest, FutureReporterMustActuallyCarryALoadAndBeReal)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto prior = request(1, 8193);
    prior.owned = false;
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    auto own = request(2, 8192);
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, 0, -1};
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {prior, own}, policy(), PARAMETERS,
                                                1000, true),
                 std::invalid_argument);
    own.receiveDependency->reporterOrdinal = 2;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {prior, own}, policy(), PARAMETERS,
                                                1000, true),
                 std::invalid_argument);
    own.receiveDependency->reporterOrdinal = 1;
    state.results.push_back({std::nullopt, false, 500.0});
    TPUPrefillPP4ForecastOutput result;
    ASSERT_NO_THROW(result = TPUPrefillPP4Forecast::predict(state, {prior, own}, policy(),
                                                            PARAMETERS, 1000, true));
    ASSERT_GE(result.firstScheduledTimeMs.at(2), 500.0);
}

TEST(TPUPrefillPP4ForecastTest, QueryLoadWithoutRealCarrierUsesMetadataResultFIFO)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 8192);
    own.readyTimeMs = 12.0;
    own.needsFirstQuery = true;
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
    const auto result =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 200, true);
    ASSERT_DOUBLE_EQ(56.0, result.firstScheduledTimeMs.at(1));
    ASSERT_EQ(1u, result.realSteps);
    ASSERT_EQ(1u, result.ownedSteps);
    ASSERT_EQ(8192, result.scheduledTokens.at(1));
    ASSERT_EQ(3u, result.dispatches.size());
    ASSERT_EQ(0, result.dispatches.front().ordinal);
    ASSERT_DOUBLE_EQ(56.0, result.dispatches.front().timeMs);
    ASSERT_FALSE(result.controlDispatches.empty());
    ASSERT_EQ(std::vector<uint64_t>{1}, result.controlDispatches.front().loadRequestIds);
    size_t notifications = 0;
    for (const auto &call : result.controlDispatches) {
        EXPECT_EQ(-1, call.ppOrdinal);
        notifications += call.finishedLoadRequestIds.size();
    }
    ASSERT_EQ(1u, notifications);
    ASSERT_DOUBLE_EQ(12.0, result.controlDispatches.front().timeMs);
    ASSERT_DOUBLE_EQ(56.0, result.controlDispatches.back().readyTimeMs);
    ASSERT_TRUE(state.results.empty());
    ASSERT_TRUE(state.slots.empty());
}

TEST(TPUPrefillPP4ForecastTest, MetadataNotificationCannotBypassEarlierControlResult)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.results.push_back({std::nullopt, false, 100.0});
    auto own = request(1, 8192);
    own.readyTimeMs = 12.0;
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
    const auto result =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 200, true);
    EXPECT_GE(result.firstScheduledTimeMs.at(1), 100.0);
    EXPECT_EQ(1u, result.realSteps);
    EXPECT_EQ(8192, result.scheduledTokens.at(1));
}

TEST(TPUPrefillPP4ForecastTest, WorkerLoadCompletionDoesNotWakeNativeWithoutAResult)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 8192);
    own.readyTimeMs = 3.0;
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{5.0, 2.0};
    const auto result =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 200, true);
    EXPECT_DOUBLE_EQ(11.0, result.firstScheduledTimeMs.at(1));
    ASSERT_EQ(4u, result.controlDispatches.size());
    EXPECT_DOUBLE_EQ(9.0, result.controlDispatches.back().timeMs);
    EXPECT_DOUBLE_EQ(11.0, result.controlDispatches.back().readyTimeMs);
}

TEST(TPUPrefillPP4ForecastTest, MetadataLoadRequiresWaitingVisitAndNativeCapacity)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    state.otherRequestReleaseTimesMs = {100.0};
    state.measuredForwardMs = 10.0;
    auto own = request(1, 8192);
    own.readyTimeMs = 12.0;
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
    const auto result =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 200, true);
    ASSERT_FALSE(result.controlDispatches.empty());
    EXPECT_GE(result.controlDispatches.front().timeMs, 100.0);
    EXPECT_GE(result.firstScheduledTimeMs.at(1), 144.0);
}

TEST(TPUPrefillPP4ForecastTest, MetadataCarrierWaitsForFourComputeResultGate)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.lastRealOrdinal = 3;
    for (int64_t ordinal = 0; ordinal < 4; ++ordinal) {
        TPUPrefillPipelineSlot slot;
        slot.ordinal = ordinal;
        slot.features[0] = 1.0;
        state.slots.push_back(slot);
        state.expectedForwardMs[ordinal] = 10.0;
        state.results.push_back({ordinal, false, 0.0});
    }
    auto own = request(1, 8192);
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, 3};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
    const auto result =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 500, true);
    ASSERT_FALSE(result.controlDispatches.empty());
    EXPECT_GE(result.controlDispatches.front().timeMs, 40.0);
    EXPECT_EQ(1u, result.realSteps);
    EXPECT_EQ(4u, state.results.size());
}

TEST(TPUPrefillPP4ForecastTest, MetadataCarrierWaitsForEightTotalResultGate)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    for (int index = 0; index < 8; ++index)
        state.results.push_back({std::nullopt, false, double(index + 1)});
    auto own = request(1, 8192);
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
    const auto result =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 500, true);
    ASSERT_FALSE(result.controlDispatches.empty());
    EXPECT_GE(result.controlDispatches.front().timeMs, 1.0);
    EXPECT_EQ(1u, result.realSteps);
    EXPECT_EQ(8u, state.results.size());
}

TEST(TPUPrefillPP4ForecastTest,
     MetadataTracingDoesNotChangePredictionOrRepeatNotifications)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 16 * 8192);
    own.readyTimeMs = 12.0;
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
    const auto traced =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 1000, true);
    const auto quiet =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 1000, false);
    EXPECT_DOUBLE_EQ(traced.totalMs, quiet.totalMs);
    EXPECT_DOUBLE_EQ(traced.queueMs, quiet.queueMs);
    EXPECT_TRUE(quiet.dispatches.empty());
    EXPECT_TRUE(quiet.controlDispatches.empty());
    EXPECT_EQ(16 * 8192, quiet.scheduledTokens.at(1));
    EXPECT_GE(quiet.totalMs,
              traced.dispatches[traced.dispatches.size() - 3].timeMs + 10.0);
    size_t notifications = 0;
    for (const auto &call : traced.controlDispatches)
        notifications += call.finishedLoadRequestIds.size();
    for (const auto &call : traced.dispatches)
        notifications += call.finishedLoadRequestIds.size();
    EXPECT_EQ(1u, notifications);
}

TEST(TPUPrefillPP4ForecastTest, InvalidMetadataTimingIsRejectedWithoutChangingInputs)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    auto own = request(1, 8192);
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    for (const auto timing :
         {TPUPrefillMetadataKVLoad{1.0, 0.0}, TPUPrefillMetadataKVLoad{-1.0, 1.0},
          TPUPrefillMetadataKVLoad{1.0, std::numeric_limits<double>::infinity()}}) {
        own.metadataOnlyLoad = timing;
        EXPECT_THROW(
            TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 100),
            std::invalid_argument);
    }
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
    EXPECT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 1),
                 std::length_error);
    EXPECT_TRUE(state.results.empty());
    EXPECT_EQ(8192, own.remainingTokens);
}

TEST(TPUPrefillPP4ForecastTest, LoadingPriorRequestDoesNotBlockReadyShortRequest)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto prior = request(1, 16384, false);
    prior.readyTimeMs = 1000.0;
    const auto own = request(2, 1);
    const auto overtaking = TPUPrefillPP4Forecast::predict(state, {prior, own}, policy(),
                                                           PARAMETERS, 100, true);
    prior.readyTimeMs = 0.0;
    const auto blocking = TPUPrefillPP4Forecast::predict(state, {prior, own}, policy(),
                                                         PARAMETERS, 100, true);
    ASSERT_EQ(uint64_t(2), overtaking.dispatches.front().spans.front().requestId);
    ASSERT_EQ(uint64_t(1), blocking.dispatches.front().spans.front().requestId);
    ASSERT_LT(overtaking.totalMs, blocking.totalMs);
    ASSERT_EQ(size_t(1), overtaking.ownedSteps);
}

TEST(TPUPrefillPP4ForecastTest, SkippedWaitersRunBeforeOrdinaryWaiters)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto prior = request(1, 8192, false);
    auto own = request(2, 1);
    own.queue = TPUPrefillForecastQueue::SKIPPED_WAITING;
    const auto output = TPUPrefillPP4Forecast::predict(state, {prior, own}, policy(),
                                                       PARAMETERS, 100, true);
    ASSERT_EQ(uint64_t(2), output.dispatches.front().spans.front().requestId);
}

TEST(TPUPrefillPP4ForecastTest, ReservedTokensAreNotScheduledAgain)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 8193);
    own.reservedTokens = 8192;
    const auto output =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 100, true);
    ASSERT_EQ(8192, output.dispatches.front().spans.front().computedTokens);
    ASSERT_EQ(1, output.dispatches.front().spans.front().scheduledTokens);
    ASSERT_EQ(int64_t(1), output.scheduledTokens.at(1));
}

TEST(TPUPrefillPP4ForecastTest, NoTraceAllocationsUnlessRequested)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    const auto output =
        TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(), PARAMETERS, 100);
    ASSERT_TRUE(output.dispatches.empty());
    ASSERT_EQ(size_t(1), output.realSteps);
    ASSERT_EQ(size_t(1), output.realPrefixSteps);
}

TEST(TPUPrefillPP4ForecastTest,
     ControlResultKeepsItsReadinessBeforeComputeCapacityIsReleased)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    for (size_t index = 0; index < 8; ++index) {
        state.results.push_back({std::nullopt, false, 100.0});
    }
    const auto output =
        TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(), PARAMETERS, 100);
    ASSERT_DOUBLE_EQ(100.0, output.queueMs);
}

TEST(TPUPrefillPP4ForecastTest, RejectsPartialForecastAndInvalidRequestState)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {request(1, 8192)}, policy(),
                                                PARAMETERS, 1),
                 std::length_error);
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {request(1, 1), request(1, 2)},
                                                policy(), PARAMETERS, 100),
                 std::invalid_argument);
    auto own = request(1, 1);
    own.queue = TPUPrefillForecastQueue::RUNNING;
    own.readyTimeMs = 1.0;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 100),
                 std::invalid_argument);
    own.queue = TPUPrefillForecastQueue::WAITING;
    own.readyTimeMs = std::numeric_limits<double>::quiet_NaN();
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 100),
                 std::invalid_argument);
}

TEST(TPUPrefillPP4ForecastTest, StreamingChunkUsesExistingBlockAndSharedPrefixRules)
{
    auto config = policy();
    TPUPrefillRequestInput input;
    input.promptTokens = 16385;
    input.computedTokens = 100;
    int64_t tokens = 0;
    ASSERT_TRUE(TPUPrefillExecutionPolicy::clipStreamingPrefillChunk(input, 8192, config,
                                                                     tokens));
    ASSERT_EQ(3996, tokens);
    input.computedTokens = 4096;
    input.sharedPrefixBoundary = 8192;
    ASSERT_TRUE(TPUPrefillExecutionPolicy::clipStreamingPrefillChunk(input, 8192, config,
                                                                     tokens));
    ASSERT_EQ(4096, tokens);
    ASSERT_TRUE(
        TPUPrefillExecutionPolicy::clipStreamingPrefillChunk(input, 0, config, tokens));
    ASSERT_EQ(0, tokens);
}

TEST(TPUPrefillPP4ForecastTest, KVReceiveWaitsForItsReporterResultToBeProcessed)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    for (int64_t ordinal = 0; ordinal < 4; ++ordinal) {
        TPUPrefillPipelineSlot slot;
        slot.ordinal = ordinal;
        slot.features[0] = 1.0;
        state.slots.push_back(slot);
        state.expectedForwardMs.emplace(ordinal, 10.0);
        state.results.push_back({ordinal, true, 0.0});
    }
    state.lastRealOrdinal = 3;
    auto own = request(1, 1);
    own.receiveDependency = TPUPrefillKVReceiveDependency{0, 0, std::nullopt};
    const auto output =
        TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 100, true);
    ASSERT_EQ(4, output.dispatches.front().ordinal);
    ASSERT_DOUBLE_EQ(0.0, output.firstScheduledTimeMs.at(1));
    ASSERT_EQ(uint64_t(1), output.dispatches.front().spans.front().requestId);
    // Input state remains one immutable observation.
    ASSERT_EQ(size_t(4), state.results.size());
    ASSERT_TRUE(own.receiveDependency.has_value());
}

TEST(TPUPrefillPP4ForecastTest, PendingQueryCannotAttachToAnAlreadyDecidedOutput)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto prior = request(1, 1, false);
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    auto own = request(2, 1);
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    const auto noDecision = TPUPrefillPP4Forecast::predict(
        state, {prior, own, request(3, 20000, false)}, policy(), PARAMETERS, 200, true);
    own.receiveDependency->decidedProposalFence = 0;
    const auto decided = TPUPrefillPP4Forecast::predict(
        state, {prior, own, request(3, 20000, false)}, policy(), PARAMETERS, 200, true);
    ASSERT_GT(decided.firstScheduledTimeMs.at(2), noDecision.firstScheduledTimeMs.at(2));
    ASSERT_EQ(int64_t(1), decided.scheduledTokens.at(1));
    ASSERT_EQ(int64_t(1), decided.scheduledTokens.at(2));
}

TEST(TPUPrefillPP4ForecastTest, RejectsOrdinalOverflowBeforeForecastArithmetic)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    TPUPrefillPipelineSlot slot;
    slot.ordinal = std::numeric_limits<int64_t>::max() - 2;
    state.slots.push_back(slot);
    state.expectedForwardMs.emplace(slot.ordinal, 1.0);
    state.lastRealOrdinal = slot.ordinal;
    ASSERT_THROW(
        TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(), PARAMETERS, 100),
        std::invalid_argument);
}

TEST(TPUPrefillPP4ForecastTest, RejectsFutureExpectationsOutsideTheObservedSlots)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    TPUPrefillPipelineSlot slot;
    slot.features[0] = 1.0;
    state.slots.push_back(slot);
    state.expectedForwardMs.emplace(0, 10.0);
    state.expectedForwardMs.emplace(1, 99.0);
    state.lastRealOrdinal = 0;
    ASSERT_THROW(
        TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(), PARAMETERS, 100),
        std::invalid_argument);
}

TEST(TPUPrefillPP4ForecastTest, DecidedDispatchKeepsItsOriginAndReservedTokens)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.nowMs = 5.0;
    state.measuredForwardMs = 10.0;
    TPUPrefillPP4PendingDispatch pending;
    pending.dispatchTimeMs = 1.0;
    pending.expectedForwardMs = 10.0;
    pending.features = {1.0, 8.0, 8.0, 0.0, 0.0, 64.0, 1.0, 0.0};
    pending.spans.push_back({1, 0, 8192});
    state.pendingDispatch = pending;
    auto prior = request(1, 8193, false);
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    prior.reservedTokens = 8192;
    const auto output = TPUPrefillPP4Forecast::predict(state, {prior, request(2, 1)},
                                                       policy(), PARAMETERS, 100, true);
    ASSERT_EQ(1, output.dispatches.front().ordinal);
    ASSERT_DOUBLE_EQ(5.0, output.dispatches.front().timeMs);
    ASSERT_EQ(8192, output.dispatches.front().spans.front().computedTokens);
    ASSERT_EQ(1, output.scheduledTokens.at(1));
    ASSERT_EQ(1, output.scheduledTokens.at(2));
    ASSERT_DOUBLE_EQ(1.0, state.pendingDispatch->dispatchTimeMs);
}

TEST(TPUPrefillPP4ForecastTest, FutureDecidedDispatchWaitIsIncludedInLatency)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    TPUPrefillPP4PendingDispatch pending;
    pending.dispatchTimeMs = 100.0;
    pending.controlReadyTimeMs = 150.0;
    state.pendingDispatch = pending;
    const auto output = TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(),
                                                       PARAMETERS, 100, true);
    ASSERT_DOUBLE_EQ(100.0, output.queueMs);
    ASSERT_DOUBLE_EQ(100.0, output.firstScheduledTimeMs.at(1));
    ASSERT_GE(output.totalMs, 150.0);
}

TEST(TPUPrefillPP4ForecastTest, DecidedMetadataDoesNotCreateAComputeSlot)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    TPUPrefillPP4PendingDispatch pending;
    pending.controlReadyTimeMs = 200.0;
    state.pendingDispatch = pending;
    const auto output = TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(),
                                                       PARAMETERS, 100, true);
    ASSERT_EQ(0, output.dispatches.front().ordinal);
    ASSERT_EQ(size_t(1), output.realSteps);
    ASSERT_GE(output.totalMs, 200.0);
}

TEST(TPUPrefillPP4ForecastTest, RejectsDecidedDispatchWithoutPipelineCapacity)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    for (int i = 0; i < 8; ++i) state.results.push_back({std::nullopt, false, 100.0});
    state.pendingDispatch = TPUPrefillPP4PendingDispatch{};
    ASSERT_THROW(
        TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(), PARAMETERS, 100),
        std::invalid_argument);
    state.results.clear();
    state.pendingDispatch->dispatchTimeMs = 1.0;
    ASSERT_THROW(
        TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(), PARAMETERS, 100),
        std::invalid_argument);
}

TEST(TPUPrefillPP4ForecastTest, OneNativeSlotWaitsUntilThePriorOutputIsProcessed)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    state.measuredForwardMs = 10.0;
    const auto output = TPUPrefillPP4Forecast::predict(
        state, {request(1, 1, false), request(2, 1)}, policy(), PARAMETERS, 100, true);
    ASSERT_DOUBLE_EQ(0.0, output.firstScheduledTimeMs.at(1));
    ASSERT_DOUBLE_EQ(40.0, output.firstScheduledTimeMs.at(2));
    ASSERT_DOUBLE_EQ(40.0, output.queueMs);
    ASSERT_EQ(size_t(1), output.dispatches.front().spans.size());
}

TEST(TPUPrefillPP4ForecastTest, OnlyTheFinalResultReleasesAPartiallyComputedRequest)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    state.measuredForwardMs = 10.0;
    for (int64_t ordinal = 0; ordinal < 2; ++ordinal) {
        state.slots.push_back({ordinal, 0.0, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});
        state.expectedForwardMs.emplace(ordinal, 10.0);
        state.results.push_back({ordinal, false, 0.0});
    }
    state.lastRealOrdinal = 1;
    state.requestCompletionSlots.emplace(1, 1);
    auto prior = request(1, 2, false);
    prior.reservedTokens = 2;
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    const auto output = TPUPrefillPP4Forecast::predict(state, {prior, request(2, 1)},
                                                       policy(), PARAMETERS, 100, true);
    ASSERT_DOUBLE_EQ(45.0, output.firstScheduledTimeMs.at(2));
    ASSERT_EQ(4, output.dispatches[2].ordinal);
}

TEST(TPUPrefillPP4ForecastTest, ReadySnapshotResultReleasesItsSlotBeforeScheduling)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    state.nowMs = 100.0;
    state.measuredForwardMs = 10.0;
    state.lastRealOrdinal = 0;
    for (int64_t ordinal = 0; ordinal < 3; ++ordinal) {
        TPUPrefillPipelineSlot slot;
        slot.ordinal = ordinal;
        if (!ordinal) slot.features[0] = 1.0;
        state.slots.push_back(slot);
        state.expectedForwardMs.emplace(ordinal, ordinal ? 0.0 : 10.0);
    }
    state.results.push_back({0, true, 0.0});
    state.requestCompletionSlots.emplace(1, 0);
    auto prior = request(1, 1, false);
    prior.reservedTokens = 1;
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    const auto output = TPUPrefillPP4Forecast::predict(state, {prior, request(2, 1)},
                                                       policy(), PARAMETERS, 100, true);
    ASSERT_DOUBLE_EQ(0.0, output.queueMs);
    ASSERT_DOUBLE_EQ(100.0, output.firstScheduledTimeMs.at(2));
}

TEST(TPUPrefillPP4ForecastTest, NonPrefillAndPausedSlotsKeepTheirReleaseEvents)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 2;
    state.measuredForwardMs = 10.0;
    state.otherRequestReleaseTimesMs = {200.0, 100.0};
    const auto output = TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(),
                                                       PARAMETERS, 100, true);
    ASSERT_DOUBLE_EQ(100.0, output.queueMs);
    ASSERT_DOUBLE_EQ(100.0, output.firstScheduledTimeMs.at(1));
    ASSERT_DOUBLE_EQ(200.0, state.otherRequestReleaseTimesMs.front());
}

TEST(TPUPrefillPP4ForecastTest, AFullSlotWithUnknownCompletionIsNotSilentlyReleased)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    auto prior = request(1, 0, false);
    prior.contextTokens = 100;
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {prior, request(2, 1)}, policy(),
                                                PARAMETERS, 100),
                 std::runtime_error);
}

TEST(TPUPrefillPP4ForecastTest, RejectsMissingCapacityAndImpossibleSlotOccupancy)
{
    TPUPrefillPP4State state;
    ASSERT_THROW(
        TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(), PARAMETERS, 100),
        std::invalid_argument);
    state.maxRunningRequestCount = 1;
    state.otherRequestReleaseTimesMs = {10.0, 20.0};
    ASSERT_THROW(
        TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(), PARAMETERS, 100),
        std::invalid_argument);
    state.otherRequestReleaseTimesMs = {std::numeric_limits<double>::infinity()};
    ASSERT_THROW(
        TPUPrefillPP4Forecast::predict(state, {request(1, 1)}, policy(), PARAMETERS, 100),
        std::invalid_argument);
}

TEST(TPUPrefillPP4ForecastTest, QueueAboveMaxSeqRunsInSeparateNativeAdmissionWaves)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    std::vector<TPUPrefillPP4Request> requests;
    for (uint64_t id = 1; id <= 129; ++id) requests.push_back(request(id, 1, id == 129));
    const auto output =
        TPUPrefillPP4Forecast::predict(state, requests, policy(), PARAMETERS, 1000, true);
    ASSERT_DOUBLE_EQ(80.0, output.firstScheduledTimeMs.at(129));
    ASSERT_DOUBLE_EQ(80.0, output.queueMs);
    ASSERT_EQ(size_t(3), output.realSteps);
    std::vector<size_t> batchSizes;
    for (const auto &dispatch : output.dispatches)
        if (dispatch.real) batchSizes.push_back(dispatch.spans.size());
    ASSERT_EQ(std::vector<size_t>({64, 64, 1}), batchSizes);
}

TEST(TPUPrefillPP4ForecastTest, DecidedComputeKeepsItsNativeAdmissionSlot)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    state.measuredForwardMs = 10.0;
    TPUPrefillPP4PendingDispatch pending;
    pending.features = {1.0, 1.0 / 1024.0, 2.0, 0.0, 0.0, 1.0 / 1048576.0, 1.0, 0.0};
    pending.spans.push_back({1, 0, 1});
    pending.expectedForwardMs = 10.0;
    state.pendingDispatch = pending;
    state.requestCompletionSlots.emplace(1, 0);
    auto prior = request(1, 1, false);
    prior.reservedTokens = 1;
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    const auto output = TPUPrefillPP4Forecast::predict(state, {prior, request(2, 1)},
                                                       policy(), PARAMETERS, 100, true);
    ASSERT_DOUBLE_EQ(40.0, output.firstScheduledTimeMs.at(2));
}

TEST(TPUPrefillPP4ForecastTest, UnknownPrefillReleaseConditionIsNotAssumedTrue)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    auto own = request(1, 1);
    own.releaseSlotAfterPrefill.reset();
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS, 100),
                 std::invalid_argument);
}

TEST(TPUPrefillPP4ForecastTest, FutureNonterminalPrefillKeepsItsNativeSlot)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    state.measuredForwardMs = 10.0;
    auto prior = request(1, 1, false);
    prior.releaseSlotAfterPrefill = false;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {prior, request(2, 1)}, policy(),
                                                PARAMETERS, 100),
                 std::runtime_error);
    state.maxRunningRequestCount = 2;
    const auto result = TPUPrefillPP4Forecast::predict(state, {prior, request(2, 1)},
                                                       policy(), PARAMETERS, 100, true);
    ASSERT_DOUBLE_EQ(0.0, result.queueMs);
    ASSERT_EQ(size_t(1), result.realSteps);
}

TEST(TPUPrefillPP4ForecastTest, ReadyObservedNonterminalOutputDoesNotFreeARequestSlot)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    state.measuredForwardMs = 10.0;
    state.lastRealOrdinal = 0;
    state.slots.push_back(
        {0, 0.0, {1.0, 1.0 / 1024.0, 2.0, 0.0, 0.0, 1.0 / 1048576.0, 1.0, 0.0}});
    state.expectedForwardMs[0] = 10.0;
    state.results.push_back({0, true, 0.0});
    state.requestCompletionSlots[1] = 0;
    auto prior = request(1, 0, false);
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    prior.contextTokens = 1;
    prior.releaseSlotAfterPrefill = false;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {prior, request(2, 1)}, policy(),
                                                PARAMETERS, 100),
                 std::runtime_error);
    prior.releaseSlotAfterPrefill = true;
    const auto result = TPUPrefillPP4Forecast::predict(state, {prior, request(2, 1)},
                                                       policy(), PARAMETERS, 100, true);
    ASSERT_EQ(int64_t(1), result.scheduledTokens.at(2));
}

TEST(TPUPrefillPP4ForecastTest, PendingLoadMetadataWaitsForNativeQueueVisit)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto running = request(1, 8193, false);
    running.queue = TPUPrefillForecastQueue::RUNNING;
    auto own = request(2, 1);
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    const auto result =
        TPUPrefillPP4Forecast::predict(state, {running, own, request(3, 20000, false)},
                                       policy(), PARAMETERS, 1000, true);
    size_t carriers = 0;
    for (const auto &dispatch : result.dispatches) {
        for (const auto id : dispatch.loadMetadataRequestIds) {
            if (id == 2) {
                ASSERT_EQ(1, dispatch.ordinal);
                ++carriers;
            }
        }
    }
    ASSERT_EQ(size_t(1), carriers);
    ASSERT_EQ(int64_t(1), result.scheduledTokens.at(2));
}

TEST(TPUPrefillPP4ForecastTest, PendingLoadMetadataWaitsForQueryReadiness)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto running = request(1, 1, false);
    running.queue = TPUPrefillForecastQueue::RUNNING;
    auto own = request(2, 1);
    own.readyTimeMs = 100.0;
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    const auto result = TPUPrefillPP4Forecast::predict(
        state, {running, own, request(3, 200000, false), request(4, 20000, false)},
        policy(), PARAMETERS, 10000, true);
    size_t carriers = 0;
    for (const auto &dispatch : result.dispatches) {
        for (const auto id : dispatch.loadMetadataRequestIds) {
            if (id == 2) {
                ASSERT_GE(dispatch.timeMs, 100.0);
                ++carriers;
            }
        }
    }
    ASSERT_EQ(size_t(1), carriers);
    ASSERT_EQ(int64_t(1), result.scheduledTokens.at(2));
}

TEST(TPUPrefillPP4ForecastTest, MetadataReceivePredictionWaitsForItsNativeFifoResult)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.results = {{std::nullopt, false, 100.0}, {std::nullopt, true, 0.0}};
    TPUPrefillPP4Request request;
    request.requestId = 1;
    request.remainingTokens = 8192;
    request.releaseSlotAfterPrefill = true;
    request.owned = true;
    request.receiveHostResultIndex = 1;
    const auto value = TPUPrefillPP4Forecast::predict(
        state, {request}, policy(), {0., 10., 0., 0., 0., 0., 0., 0., 0.}, 1000, true);
    ASSERT_DOUBLE_EQ(100.0, value.firstScheduledTimeMs.at(1));
    ASSERT_DOUBLE_EQ(100.0, value.queueMs);
    ASSERT_DOUBLE_EQ(140.0, value.totalMs);
    ASSERT_EQ(1u, value.realSteps);
    ASSERT_TRUE(value.controlDispatches.empty());
}

TEST(TPUPrefillPP4ForecastTest, MetadataReceiveCanUseFirstOrPendingControlResult)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    TPUPrefillPP4Request request;
    request.requestId = 1;
    request.remainingTokens = 8192;
    request.releaseSlotAfterPrefill = true;
    request.owned = true;
    request.receiveHostResultIndex = 0;
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.results = {{std::nullopt, true, 0.0}};
    const TPUPrefillPP4Timing::Parameters parameters{0., 10., 0., 0., 0., 0., 0., 0., 0.};
    auto value =
        TPUPrefillPP4Forecast::predict(state, {request}, policy(), parameters, 1000);
    ASSERT_DOUBLE_EQ(0.0, value.queueMs);
    ASSERT_DOUBLE_EQ(40.0, value.totalMs);
    state.results.clear();
    TPUPrefillPP4PendingDispatch pending;
    pending.dispatchTimeMs = 5.0;
    pending.controlReadyTimeMs = 25.0;
    state.pendingDispatch = pending;
    value = TPUPrefillPP4Forecast::predict(state, {request}, policy(), parameters, 1000);
    ASSERT_DOUBLE_EQ(25.0, value.queueMs);
    ASSERT_DOUBLE_EQ(65.0, value.totalMs);
    request.receiveHostResultIndex = 1;
    ASSERT_THROW(
        TPUPrefillPP4Forecast::predict(state, {request}, policy(), parameters, 1000),
        std::invalid_argument);
    request.receiveHostResultIndex = 0;
    request.needsFirstQuery = true;
    ASSERT_THROW(
        TPUPrefillPP4Forecast::predict(state, {request}, policy(), parameters, 1000),
        std::invalid_argument);
}

TEST(TPUPrefillPP4ForecastTest, QueryServiceStartsAtFirstPollAfterFullNativeFifo)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    for (size_t i = 0; i < 8; ++i)
        state.results.push_back({std::nullopt, false, i == 0 ? 100.0 : 0.0});
    auto own = request(1, 8192);
    own.needsFirstQuery = true;
    own.firstQueryServiceMs = 30.0;
    const auto output = TPUPrefillPP4Forecast::predict(
        state, {own}, policy(), PARAMETERS, 1000, true);
    ASSERT_DOUBLE_EQ(100.0, output.firstQueryTimeMs.at(1));
    ASSERT_DOUBLE_EQ(130.0, output.firstQueryReadyTimeMs.at(1));
    ASSERT_GE(output.firstScheduledTimeMs.at(1), 130.0);
    ASSERT_TRUE(own.needsFirstQuery);
    ASSERT_DOUBLE_EQ(30.0, *own.firstQueryServiceMs);
    ASSERT_DOUBLE_EQ(0.0, own.readyTimeMs);
    ASSERT_EQ(size_t(8), state.results.size());
}

TEST(TPUPrefillPP4ForecastTest, QueryServiceDoesNotStartBeforeArrivalGate)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 8192);
    own.needsFirstQuery = true;
    own.readyTimeMs = 200.0;
    own.firstQueryServiceMs = 30.0;
    const auto output = TPUPrefillPP4Forecast::predict(
        state, {own}, policy(), PARAMETERS, 1000, true);
    ASSERT_DOUBLE_EQ(200.0, output.firstQueryTimeMs.at(1));
    ASSERT_DOUBLE_EQ(230.0, output.firstQueryReadyTimeMs.at(1));
    ASSERT_DOUBLE_EQ(230.0, output.firstScheduledTimeMs.at(1));
}

TEST(TPUPrefillPP4ForecastTest, QueryServiceRespectsNativeRunningRequestCapacity)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    state.measuredForwardMs = 10.0;
    state.otherRequestReleaseTimesMs = {100.0};
    auto own = request(1, 8192);
    own.needsFirstQuery = true;
    own.firstQueryServiceMs = 30.0;
    const auto output = TPUPrefillPP4Forecast::predict(
        state, {own}, policy(), PARAMETERS, 1000, true);
    ASSERT_DOUBLE_EQ(100.0, output.firstQueryTimeMs.at(1));
    ASSERT_DOUBLE_EQ(130.0, output.firstQueryReadyTimeMs.at(1));
    ASSERT_DOUBLE_EQ(130.0, output.firstScheduledTimeMs.at(1));
}

TEST(TPUPrefillPP4ForecastTest, QueryServiceCompletionPrecedesMetadataLoadAndReporter)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 8192);
    own.needsFirstQuery = true;
    own.firstQueryServiceMs = 25.0;
    own.readyTimeMs = 100.0;
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{5.0, 2.0};
    const auto output = TPUPrefillPP4Forecast::predict(
        state, {own}, policy(), PARAMETERS, 1000, true);
    ASSERT_DOUBLE_EQ(100.0, output.firstQueryTimeMs.at(1));
    ASSERT_DOUBLE_EQ(125.0, output.firstQueryReadyTimeMs.at(1));
    bool carrier = false, reporter = false;
    for (const auto &call : output.controlDispatches) {
        if (!call.loadRequestIds.empty()) {
            ASSERT_GE(call.timeMs, 125.0);
            carrier = true;
        }
        if (!call.finishedLoadRequestIds.empty()) {
            ASSERT_GE(call.timeMs, 130.0);
            ASSERT_GE(output.firstScheduledTimeMs.at(1), call.readyTimeMs);
            reporter = true;
        }
    }
    ASSERT_TRUE(carrier);
    ASSERT_TRUE(reporter);
}

TEST(TPUPrefillPP4ForecastTest, QueryServiceZeroDurationAndTraceOffPreserveInputs)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 8192);
    own.needsFirstQuery = true;
    own.firstQueryServiceMs = 0.0;
    const auto traced = TPUPrefillPP4Forecast::predict(
        state, {own}, policy(), PARAMETERS, 1000, true);
    const auto plain = TPUPrefillPP4Forecast::predict(
        state, {own}, policy(), PARAMETERS, 1000);
    ASSERT_DOUBLE_EQ(traced.totalMs, plain.totalMs);
    ASSERT_DOUBLE_EQ(0.0, traced.firstQueryReadyTimeMs.at(1));
    ASSERT_TRUE(plain.firstQueryTimeMs.empty());
    ASSERT_TRUE(plain.firstQueryReadyTimeMs.empty());
    ASSERT_TRUE(own.needsFirstQuery);
    ASSERT_TRUE(own.firstQueryServiceMs.has_value());
}

TEST(TPUPrefillPP4ForecastTest, QueryServiceRejectsInvalidOrAlreadyIssuedInputs)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 8192);
    own.needsFirstQuery = true;
    for (double invalid : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
        own.firstQueryServiceMs = invalid;
        ASSERT_THROW(TPUPrefillPP4Forecast::predict(
                         state, {own}, policy(), PARAMETERS, 1000),
                     std::invalid_argument);
    }
    own.firstQueryServiceMs = 25.0;
    own.needsFirstQuery = false;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(
                     state, {own}, policy(), PARAMETERS, 1000),
                 std::invalid_argument);
    own.needsFirstQuery = true;
    own.queue = TPUPrefillForecastQueue::RUNNING;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(
                     state, {own}, policy(), PARAMETERS, 1000),
                 std::invalid_argument);
    own.queue = TPUPrefillForecastQueue::WAITING;
    own.receiveDependency = TPUPrefillKVReceiveDependency{0, std::nullopt, std::nullopt};
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(
                     state, {own}, policy(), PARAMETERS, 1000),
                 std::invalid_argument);
}

TEST(TPUPrefillPP4ForecastTest, QueryServiceRejectsCompletionClockOverflow)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.nowMs = std::numeric_limits<double>::max() * 0.5;
    auto own = request(1, 8192);
    own.readyTimeMs = state.nowMs;
    own.needsFirstQuery = true;
    own.firstQueryServiceMs = std::numeric_limits<double>::max() * 0.75;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(
                     state, {own}, policy(), PARAMETERS, 1000),
                 std::overflow_error);
}


TEST(TPUPrefillPP4ForecastTest, ComputeCarrierLoadWaitsForWorkerAndNativeNotification)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto prior = request(1, 1, false);
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    auto own = request(2, 8192);
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{100.0, 4.0};
    TPUPrefillPP4ForecastOutput result;
    ASSERT_NO_THROW(result = TPUPrefillPP4Forecast::predict(
        state, {prior, own}, policy(), PARAMETERS, 5000, true));
    ASSERT_EQ(std::vector<uint64_t>{2}, result.dispatches.front().loadMetadataRequestIds);
    EXPECT_TRUE(result.dispatches.front().finishedLoadRequestIds.empty());
    EXPECT_GE(result.firstScheduledTimeMs.at(2), 100.0);
    size_t loads = 0, notifications = 0;
    for (const auto &call : result.dispatches) {
        loads += call.loadMetadataRequestIds.size();
        notifications += call.finishedLoadRequestIds.size();
        if (!call.finishedLoadRequestIds.empty()) EXPECT_GE(call.timeMs, 100.0);
    }
    for (const auto &call : result.controlDispatches) {
        EXPECT_TRUE(call.loadRequestIds.empty());
        notifications += call.finishedLoadRequestIds.size();
        if (!call.finishedLoadRequestIds.empty()) EXPECT_GE(call.timeMs, 100.0);
    }
    EXPECT_EQ(1u, loads);
    EXPECT_EQ(1u, notifications);
    EXPECT_EQ(8192, result.scheduledTokens.at(2));
    EXPECT_TRUE(own.receiveDependency->decidedProposalFence.has_value());
    EXPECT_TRUE(state.results.empty());
}

TEST(TPUPrefillPP4ForecastTest, ComputeCarrierLoadsCompleteIndependently)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto prior = request(1, 1, false);
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    auto early = request(2, 1), late = request(3, 1);
    early.receiveDependency = late.receiveDependency =
        TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    early.metadataOnlyLoad = TPUPrefillMetadataKVLoad{30.0, 4.0};
    late.metadataOnlyLoad = TPUPrefillMetadataKVLoad{160.0, 4.0};
    TPUPrefillPP4ForecastOutput result;
    ASSERT_NO_THROW(result = TPUPrefillPP4Forecast::predict(
        state, {prior, early, late}, policy(), PARAMETERS, 5000, true));
    EXPECT_EQ((std::vector<uint64_t>{2, 3}), result.dispatches.front().loadMetadataRequestIds);
    EXPECT_GE(result.firstScheduledTimeMs.at(2), 30.0);
    EXPECT_LT(result.firstScheduledTimeMs.at(2), 160.0);
    EXPECT_GE(result.firstScheduledTimeMs.at(3), 160.0);
    std::map<uint64_t, size_t> notifications;
    for (const auto &call : result.dispatches)
        for (auto id : call.finishedLoadRequestIds) ++notifications[id];
    for (const auto &call : result.controlDispatches)
        for (auto id : call.finishedLoadRequestIds) ++notifications[id];
    EXPECT_EQ(1u, notifications[2]);
    EXPECT_EQ(1u, notifications[3]);
}

TEST(TPUPrefillPP4ForecastTest, ComputeCarrierLoadTracingDoesNotChangePrediction)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto prior = request(1, 8193, false);
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    auto own = request(2, 8192);
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
    TPUPrefillPP4ForecastOutput traced, quiet;
    ASSERT_NO_THROW(traced = TPUPrefillPP4Forecast::predict(
        state, {prior, own}, policy(), PARAMETERS, 5000, true));
    ASSERT_NO_THROW(quiet = TPUPrefillPP4Forecast::predict(
        state, {prior, own}, policy(), PARAMETERS, 5000, false));
    EXPECT_DOUBLE_EQ(traced.totalMs, quiet.totalMs);
    EXPECT_DOUBLE_EQ(traced.queueMs, quiet.queueMs);
    EXPECT_TRUE(quiet.dispatches.empty());
    EXPECT_TRUE(quiet.controlDispatches.empty());
    const auto carrier = std::find_if(traced.dispatches.begin(), traced.dispatches.end(),
        [](const auto &call) { return !call.loadMetadataRequestIds.empty(); });
    ASSERT_NE(traced.dispatches.end(), carrier);
    EXPECT_GE(traced.firstScheduledTimeMs.at(2), carrier->timeMs + 40.0);
}

TEST(TPUPrefillPP4ForecastTest, ComputeCarrierLoadRejectsConflictingExplicitReporter)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto prior = request(1, 8193, false);
    prior.queue = TPUPrefillForecastQueue::RUNNING;
    auto own = request(2, 8192);
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, 2, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{100.0, 4.0};
    EXPECT_THROW(TPUPrefillPP4Forecast::predict(
        state, {prior, own}, policy(), PARAMETERS, 5000, true), std::invalid_argument);
}


TEST(TPUPrefillPP4ForecastTest, ComputeCarrierLoadLegacyBenchmark)
{
    for (size_t count : {size_t(1), size_t(8), size_t(32)}) {
        TPUPrefillPP4State state;
        state.maxRunningRequestCount = 64;
        state.measuredForwardMs = 10.0;
        std::vector<TPUPrefillPP4Request> requests;
        for (size_t i = 0; i < count; ++i) requests.push_back(request(i + 1, 8192));
        std::vector<double> us;
        double sink = 0;
        for (size_t rep = 0; rep < 9; ++rep) {
            const auto start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < 2000; ++i)
                sink += TPUPrefillPP4Forecast::predict(
                    state, requests, policy(), PARAMETERS, 10000, false).totalMs;
            us.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / 2000);
        }
        std::cout << "COMPUTE_CARRIER_LOAD_BENCH {\"requests\":" << count
                  << ",\"sink\":" << sink << ",\"us\":[";
        for (size_t i = 0; i < us.size(); ++i) std::cout << (i ? "," : "") << us[i];
        std::cout << "]}" << std::endl;
    }
}


TEST(TPUPrefillPP4ForecastTest, QueryBacklogTracksQueriesIssuedEarlierInSamePoll)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{10.0, 0.0, 10.0, 0.0, 0.0};
    auto first = request(1, 1), second = request(2, 1);
    first.needsFirstQuery = second.needsFirstQuery = true;
    first.queryBlockCount = 4;
    second.queryBlockCount = 8;
    const auto result = TPUPrefillPP4Forecast::predict(
        state, {first, second}, policy(), PARAMETERS, 1000, true);
    ASSERT_DOUBLE_EQ(0.0, result.firstQueryTimeMs.at(1));
    ASSERT_DOUBLE_EQ(0.0, result.firstQueryTimeMs.at(2));
    ASSERT_DOUBLE_EQ(10.0, result.firstQueryReadyTimeMs.at(1));
    ASSERT_DOUBLE_EQ(20.0, result.firstQueryReadyTimeMs.at(2));
    const auto &features = result.firstQueryServiceFeatures.at(2);
    ASSERT_DOUBLE_EQ(1.0, features[2]);
    ASSERT_DOUBLE_EQ(std::log1p(4.0), features[3]);
    ASSERT_DOUBLE_EQ(0.0, features[4]);
    ASSERT_TRUE(first.needsFirstQuery);
}

TEST(TPUPrefillPP4ForecastTest, QueryBacklogRetainsPendingQueryAgeWithoutReissue)
{
    TPUPrefillPP4State state;
    state.nowMs = 100.0;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{10.0, 0.0, 1.0, 0.0, 1.0};
    auto pending = request(1, 1, false), own = request(2, 1);
    pending.readyTimeMs = 180.0;
    pending.pendingQueryReadyTimeMs = 180.0;
    pending.pendingQueryAgeMs = 30.0;
    pending.queryBlockCount = 7;
    own.needsFirstQuery = true;
    own.queryBlockCount = 5;
    const auto result = TPUPrefillPP4Forecast::predict(
        state, {pending, own}, policy(), PARAMETERS, 1000, true);
    ASSERT_EQ(size_t(1), result.firstQueryTimeMs.size());
    const auto &features = result.firstQueryServiceFeatures.at(2);
    ASSERT_DOUBLE_EQ(1.0, features[2]);
    ASSERT_DOUBLE_EQ(std::log1p(7.0), features[3]);
    ASSERT_DOUBLE_EQ(std::log1p(30.0), features[4]);
    ASSERT_NEAR(111.0 + std::log1p(30.0),
                result.firstQueryReadyTimeMs.at(2), 1e-10);
}

TEST(TPUPrefillPP4ForecastTest, QueryBacklogExpiresAtIssueAndIgnoresOtherWaits)
{
    TPUPrefillPP4State state;
    state.nowMs = 100.0;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{1.0, 0.0, 1.0, 0.0, 0.0};
    auto expired = request(1, 1, false), waiting = request(2, 1, false);
    auto pending = request(3, 1, false), own = request(4, 1);
    expired.readyTimeMs = 100.0;
    expired.pendingQueryReadyTimeMs = 100.0;
    expired.pendingQueryAgeMs = 30.0;
    expired.queryBlockCount = 99;
    waiting.readyTimeMs = 200.0; // A load/arrival wait is not an active query.
    pending.readyTimeMs = 200.0;
    pending.pendingQueryReadyTimeMs = 200.0;
    pending.pendingQueryAgeMs = 10.0;
    pending.queryBlockCount = 3;
    own.needsFirstQuery = true;
    own.queryBlockCount = 5;
    const auto result = TPUPrefillPP4Forecast::predict(
        state, {expired, waiting, pending, own}, policy(), PARAMETERS, 1000, true);
    const auto &features = result.firstQueryServiceFeatures.at(4);
    ASSERT_DOUBLE_EQ(1.0, features[2]);
    ASSERT_DOUBLE_EQ(std::log1p(3.0), features[3]);
    ASSERT_DOUBLE_EQ(std::log1p(10.0), features[4]);
}

TEST(TPUPrefillPP4ForecastTest, QueryBacklogRespectsCapacityBeforeFirstIssue)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 1;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{10.0, 0.0, 1.0, 0.0, 0.0};
    auto running = request(1, 8192, false), own = request(2, 1);
    running.queue = TPUPrefillForecastQueue::RUNNING;
    own.needsFirstQuery = true;
    own.queryBlockCount = 2;
    const auto result = TPUPrefillPP4Forecast::predict(
        state, {running, own}, policy(), PARAMETERS, 1000, true);
    ASSERT_GT(result.firstQueryTimeMs.at(2), 0.0);
    ASSERT_DOUBLE_EQ(result.firstQueryTimeMs.at(2) + 10.0,
                     result.firstQueryReadyTimeMs.at(2));
    ASSERT_DOUBLE_EQ(0.0, result.firstQueryServiceFeatures.at(2)[2]);
}

TEST(TPUPrefillPP4ForecastTest, QueryBacklogIssueGateDoesNotRunQueryAtAdmission)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{10.0, 0.0, 1.0, 0.0, 0.0};
    auto own = request(1, 1);
    own.needsFirstQuery = true;
    own.queryBlockCount = 2;
    own.readyTimeMs = 100.0;
    own.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    own.metadataOnlyLoad = TPUPrefillMetadataKVLoad{5.0, 2.0};
    const auto result = TPUPrefillPP4Forecast::predict(
        state, {own}, policy(), PARAMETERS, 1000, true);
    ASSERT_DOUBLE_EQ(100.0, result.firstQueryTimeMs.at(1));
    ASSERT_DOUBLE_EQ(110.0, result.firstQueryReadyTimeMs.at(1));
    ASSERT_GE(result.firstScheduledTimeMs.at(1), 115.0);
}

TEST(TPUPrefillPP4ForecastTest, QueryBacklogRejectsConflictingOrInvalidInputs)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{10.0, 0.0, 0.0, 0.0, 0.0};
    auto own = request(1, 1);
    own.queryBlockCount = -1;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
    own.queryBlockCount = 1;
    own.pendingQueryAgeMs = -1.0;
    own.pendingQueryReadyTimeMs = 1.0;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
    own.pendingQueryAgeMs = 1.0;
    own.needsFirstQuery = true;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
    own.pendingQueryAgeMs.reset();
    own.pendingQueryReadyTimeMs.reset();
    own.firstQueryServiceMs = 10.0;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
    own.firstQueryServiceMs.reset();
    state.queryServiceParameters->at(0) = -1.0;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
    state.queryServiceParameters->at(0) = std::numeric_limits<double>::infinity();
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
}

TEST(TPUPrefillPP4ForecastTest, QueryBacklogDoesNotAllocateTraceWhenDisabled)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{10.0, 1.0, 1.0, 1.0, 1.0};
    auto own = request(1, 1);
    own.needsFirstQuery = true;
    own.queryBlockCount = 3;
    const auto result = TPUPrefillPP4Forecast::predict(
        state, {own}, policy(), PARAMETERS, 1000, false);
    ASSERT_TRUE(result.firstQueryTimeMs.empty());
    ASSERT_TRUE(result.firstQueryReadyTimeMs.empty());
    ASSERT_TRUE(result.firstQueryServiceFeatures.empty());
    ASSERT_GT(result.totalMs, 45.0);
}

TEST(TPUPrefillPP4ForecastTest, QueryBacklogRejectsServiceOverflow)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{0.0, 0.0,
        std::numeric_limits<double>::max(), 0.0, 0.0};
    auto pending1 = request(1, 1, false), pending2 = request(2, 1, false);
    for (auto *pending : {&pending1, &pending2}) {
        pending->readyTimeMs = 100.0;
        pending->pendingQueryReadyTimeMs = 100.0;
        pending->pendingQueryAgeMs = 1.0;
        pending->queryBlockCount = 1;
    }
    auto own = request(3, 1);
    own.needsFirstQuery = true;
    own.queryBlockCount = 1;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {pending1, pending2, own},
        policy(), PARAMETERS, 1000), std::overflow_error);
}

TEST(TPUPrefillPP4ForecastTest, QueryBacklogModelBenchmark)
{
    for (const int count : {16, 64, 128}) {
        TPUPrefillPP4State state;
        state.maxRunningRequestCount = 64;
        state.measuredForwardMs = 10.0;
        state.queryServiceParameters = std::array<double, 5>{10.0, 1.0, 1.0, 1.0, 1.0};
        std::vector<TPUPrefillPP4Request> requests;
        for (int i = 0; i < count; ++i) {
            auto item = request(i + 1, 1);
            item.needsFirstQuery = true;
            item.queryBlockCount = i + 1;
            requests.push_back(item);
        }
        std::vector<double> us;
        for (int rep = 0; rep < 9; ++rep) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 200; ++i) {
                const auto result = TPUPrefillPP4Forecast::predict(
                    state, requests, policy(), PARAMETERS, 100000, false);
                ASSERT_TRUE(result.firstQueryServiceFeatures.empty());
                ASSERT_EQ(size_t(count), result.scheduledTokens.size());
            }
            us.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / 200);
        }
        std::cout << "QUERY_BACKLOG_MODEL_BENCH {\"requests\":" << count
                  << ",\"us\":[";
        for (size_t i = 0; i < us.size(); ++i)
            std::cout << (i ? "," : "") << us[i];
        std::cout << "]}" << std::endl;
    }
}


TEST(TPUPrefillPP4ForecastTest, QueryBacklogEndsBeforeTheKVLoadAndNotification)
{
    TPUPrefillPP4State state;
    state.nowMs = 100.0;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{10.0, 0.0, 10.0, 0.0, 0.0};
    auto loading = request(1, 1, false), own = request(2, 1);
    loading.readyTimeMs = 200.0;
    loading.pendingQueryAgeMs = 50.0;
    loading.pendingQueryReadyTimeMs = 100.0;
    loading.queryBlockCount = 7;
    own.needsFirstQuery = true;
    own.queryBlockCount = 3;
    const auto result = TPUPrefillPP4Forecast::predict(
        state, {loading, own}, policy(), PARAMETERS, 1000, true);
    ASSERT_DOUBLE_EQ(0.0, result.firstQueryServiceFeatures.at(2)[2]);
    ASSERT_DOUBLE_EQ(110.0, result.firstQueryReadyTimeMs.at(2));
    ASSERT_DOUBLE_EQ(200.0, loading.readyTimeMs);
}

TEST(TPUPrefillPP4ForecastTest, QueryBacklogRequiresExplicitPendingCompletion)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    auto own = request(1, 1);
    own.queryBlockCount = 7;
    own.pendingQueryAgeMs = 50.0;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
    own.pendingQueryAgeMs.reset();
    own.pendingQueryReadyTimeMs = 100.0;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
}


TEST(TPUPrefillPP4ForecastTest, QueryBacklogMissingOwnSizeIsNotAnInstantQuery)
{
    TPUPrefillPP4State state;
    state.maxRunningRequestCount = 64;
    state.measuredForwardMs = 10.0;
    state.queryServiceParameters = std::array<double, 5>{10., 1., 1., 1., 1.};
    auto own = request(1, 1);
    own.needsFirstQuery = true;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
    own.firstQueryServiceMs = 10.0;
    ASSERT_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                              1000), std::invalid_argument);
    state.queryServiceParameters.reset();
    ASSERT_NO_THROW(TPUPrefillPP4Forecast::predict(state, {own}, policy(), PARAMETERS,
                                                 1000));
}

} // namespace navi_sched
