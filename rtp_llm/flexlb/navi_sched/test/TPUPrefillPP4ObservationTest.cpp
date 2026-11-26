#include "aios/navi_sched/common/TPUPrefillPP4Observation.h"
#include "unittest/unittest.h"

#include <limits>
#include <cmath>
#include <chrono>
#include <iostream>
#include <autil/DataBuffer.h>
#include <stdexcept>

namespace navi_sched {
namespace {

NaviSchedNativePPState emptyObservation()
{
    NaviSchedNativePPState value;
    value.observationId = 1;
    value.occupiedRequestSlotCount = 0;
    return value;
}

NaviSchedNativePPRequest nativeRequest(uint64_t id, NativePPQueue queue, int64_t index)
{
    NaviSchedNativePPRequest value;
    value.requestId = id;
    value.prefillSlotRelease = NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT;
    value.queue = queue;
    value.queueIndex = index;
    value.phase = queue == NativePPQueue::RUNNING ? NativePPRequestPhase::RUNNING
                                                  : NativePPRequestPhase::CACHE_KNOWN;
    value.promptTokens = 32768;
    value.cacheHitTokens = 0;
    return value;
}

NaviSchedNativePPSlot realSlot(int64_t ordinal, uint64_t id, int64_t context,
                               int64_t tokens, int64_t ageUs)
{
    NaviSchedNativePPSlot value;
    value.ordinal = ordinal;
    value.dispatchAgeUs = ageUs;
    value.expectedForwardMs = 10.0;
    value.spans.push_back({id, context, tokens});
    value.features = {1.0,
                      tokens / 1024.0,
                      8.0,
                      double(context) * tokens / 1048576.0,
                      context / 1024.0,
                      double(tokens) * tokens / 1048576.0,
                      1.0,
                      context / 1024.0};
    return value;
}

void addResult(NaviSchedNativePPState &state, const NaviSchedNativePPSlot &slot,
               bool executeReady = false, bool ready = false)
{
    state.waveOpen = true;
    state.burst = 1;
    state.lastRealOrdinal = slot.ordinal;
    state.slots.push_back(slot);
    state.results.push_back({uint64_t(100 + slot.ordinal), slot.dispatchAgeUs,
                             slot.ordinal, executeReady, ready});
}

TPUPrefillPP4Request incoming(uint64_t id = 999)
{
    TPUPrefillPP4Request value;
    value.requestId = id;
    value.remainingTokens = 1;
    value.owned = true;
    value.releaseSlotAfterPrefill = true;
    return value;
}

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

TPUPrefillPP4ObservationInput prepare(
    const NaviSchedNativePPState &observed, int64_t age,
    const TPUPrefillPP4ObservationPredictions &predictions,
    TPUPrefillPP4ObservedCapacity capacity = {64, 0})
{
    return TPUPrefillPP4Observation::prepare(observed, age, capacity, predictions);
}

NaviSchedWorkloadSnapshot loadingWorkload(size_t count = 1)
{
    NaviSchedWorkloadSnapshot workload;
    workload.revision = workload.eventTimeUs = 1;
    workload.prefillCapacitySplit = true;
    workload.maxRunningRequestCount = 64;
    workload.requestCount = workload.batchCount = workload.waitingRequestCount = count;
    workload.remainingPrefillRequestCount = workload.runningCapacityPrefillRequestCount =
        count;
    workload.runningCapacitySlotCount = count;
    workload.remainingPrefillContextTokenCount =
        workload.runningCapacityPrefillContextTokenCount = 4096 * count;
    workload.remainingPrefillTokenCount = workload.runningCapacityPrefillTokenCount =
        4096 * count;
    workload.remainingPrefillTokenSquareSum =
        workload.runningCapacityPrefillTokenSquareSum = 16777216.0 * count;
    workload.contextRemainingPrefillTokenProductSum =
        workload.runningCapacityContextRemainingPrefillTokenProductSum =
            16777216.0 * count;
    workload.remainingTotalContextTokenCount = workload.remainingPrefillContextTokenCount;
    workload.remainingTotalTokenCount = workload.remainingPrefillTokenCount;
    workload.remainingTotalTokenSquareSum = workload.remainingPrefillTokenSquareSum;
    workload.contextRemainingTotalTokenProductSum =
        workload.contextRemainingPrefillTokenProductSum;
    workload.totalPromptTokenCount = 8192 * count;
    workload.nativePPObservationAgeUs = 5000;
    auto observed = std::make_shared<NaviSchedNativePPState>(emptyObservation());
    observed->measuredForwardMs = 10.0;
    for (size_t i = 0; i < count; ++i) {
        auto request = nativeRequest(i + 1, NativePPQueue::WAITING, i);
        request.promptTokens = 8192;
        request.computedTokens = request.cacheHitTokens = 4096;
        request.phase = NativePPRequestPhase::KV_LOADING;
        observed->requests.push_back(request);
        NaviSchedNativeCacheQuery query;
        query.requestId = i + 1;
        query.generation = 1;
        query.inputSource = NativeCacheQueryInputSource::RESOLVED_QUERY;
        query.localPrefixTokenCount = 0;
        query.loadState = NativeCacheLoadState::IDLE;
        query.receiveState = NativeCacheReceiveState::RECEIVED;
        workload.nativeCacheQueries.push_back(query);
    }
    workload.nativePPState = std::move(observed);
    return workload;
}

} // namespace

TEST(TPUPrefillPP4ObservationTest, FutureBoundsKeepTheirOriginalObservationClock)
{
    auto observed = emptyObservation();
    auto first = realSlot(0, 1, 0, 4096, 2000);
    first.lastStageReplyAgeUs = 500;
    addResult(observed, first, true, false);
    addResult(observed, realSlot(1, 2, 0, 4096, 1000));
    ASSERT_TRUE(observed.valid());
    const auto value = prepare(observed, 50000, {});
    ASSERT_DOUBLE_EQ(52.0, value.state.nowMs);
    ASSERT_DOUBLE_EQ(0.0, value.state.slots.front().dispatchTimeMs);
    ASSERT_DOUBLE_EQ(1.0, value.state.slots.back().dispatchTimeMs);
    ASSERT_DOUBLE_EQ(1.5, value.state.observedReplies.at(0));
    ASSERT_DOUBLE_EQ(2.0, value.state.replyLowerBounds.at(1));
    ASSERT_FALSE(value.state.results[0].ready);
    ASSERT_EQ(2000, observed.slots.front().dispatchAgeUs);
}

TEST(TPUPrefillPP4ObservationTest, ReadyUnconsumedResultsRemainReservedExactlyOnce)
{
    auto observed = emptyObservation();
    addResult(observed, realSlot(0, 1, 4096, 4096, 2000), true, true);
    addResult(observed, realSlot(1, 1, 8192, 4096, 1000), true, true);
    auto request = nativeRequest(1, NativePPQueue::RUNNING, 0);
    request.computedTokens = 12288;
    request.cacheHitTokens = 4096;
    observed.requests.push_back(request);
    observed.occupiedRequestSlotCount = 1;
    const auto value = prepare(observed, 5000, {});
    ASSERT_EQ(4096, value.requests.front().contextTokens);
    ASSERT_EQ(8192, value.requests.front().reservedTokens);
    ASSERT_EQ(28672, value.requests.front().remainingTokens);
    ASSERT_EQ(12288, value.requests.front().contextTokens +
                         value.requests.front().reservedTokens);
}

TEST(TPUPrefillPP4ObservationTest, NativeQueueOrderIsIndependentOfRequestIdAndVectorOrder)
{
    auto observed = emptyObservation();
    observed.requests = {nativeRequest(9, NativePPQueue::WAITING, 0),
                         nativeRequest(90, NativePPQueue::SKIPPED_WAITING, 0),
                         nativeRequest(999, NativePPQueue::RUNNING, 0),
                         nativeRequest(1, NativePPQueue::WAITING, 1)};
    observed.occupiedRequestSlotCount = 1;
    const auto value = prepare(observed, 0, {});
    ASSERT_EQ(uint64_t(999), value.requests[0].requestId);
    ASSERT_EQ(uint64_t(90), value.requests[1].requestId);
    ASSERT_EQ(uint64_t(9), value.requests[2].requestId);
    ASSERT_EQ(uint64_t(1), value.requests[3].requestId);
}

TEST(TPUPrefillPP4ObservationTest,
     UnknownCacheAndQueryReadinessRequireExplicitPredictions)
{
    auto observed = emptyObservation();
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.cacheHitTokens = -1;
    request.phase = NativePPRequestPhase::QUERY_PENDING;
    observed.requests.push_back(request);
    ASSERT_THROW(prepare(observed, 0, {}), std::invalid_argument);
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.requests[1].readyAfterObservationMs = 100.0;
    ASSERT_THROW(prepare(observed, 0, predicted), std::invalid_argument);
    predicted.requests[1].readyAfterObservationMs.reset();
    predicted.requests[1].cacheHitTokens = 8192;
    ASSERT_THROW(prepare(observed, 0, predicted), std::invalid_argument);
    predicted.requests[1].readyAfterObservationMs = 100.0;
    const auto value = prepare(observed, 5000, predicted);
    ASSERT_EQ(8192, value.requests.front().contextTokens);
    ASSERT_DOUBLE_EQ(100.0, value.requests.front().readyTimeMs);
    ASSERT_EQ(-1, observed.requests.front().cacheHitTokens);
}

TEST(TPUPrefillPP4ObservationTest, CompletedQueryPrefixDoesNotMakeItsLoadReady)
{
    auto observed = emptyObservation();
    observed.measuredForwardMs = 10.0;
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.cacheHitTokens = 12288;
    request.phase = NativePPRequestPhase::QUERY_READY_NOT_CONSUMED;
    observed.requests.push_back(request);
    ASSERT_THROW(prepare(observed, 0, {}), std::invalid_argument);
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.requests[1].readyAfterObservationMs = 100.0;
    auto value = prepare(observed, 0, predicted);
    ASSERT_EQ(12288, value.requests.front().contextTokens);
    ASSERT_EQ(20480, value.requests.front().remainingTokens);
    ASSERT_DOUBLE_EQ(100.0, value.requests.front().readyTimeMs);
    value.requests.front().owned = true;
    const auto forecast = TPUPrefillPP4Forecast::predict(
        value.state, value.requests, policy(),
        {0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1000);
    ASSERT_DOUBLE_EQ(100.0, forecast.firstScheduledTimeMs.at(1));
}

TEST(TPUPrefillPP4ObservationTest, FullExternalPrefixStillRequiresTheLastToken)
{
    for (const auto phase : {NativePPRequestPhase::QUERY_READY_NOT_CONSUMED,
                             NativePPRequestPhase::KV_LOADING}) {
        for (const int64_t prompt : {int64_t(1), int64_t(8192)}) {
            for (const int64_t cursor : {int64_t(0), prompt}) {
                auto observed = emptyObservation();
                observed.measuredForwardMs = 10.0;
                auto request = nativeRequest(1, NativePPQueue::SKIPPED_WAITING, 0);
                request.promptTokens = prompt;
                request.computedTokens = cursor;
                request.cacheHitTokens = prompt;
                request.phase = phase;
                observed.requests.push_back(request);
                TPUPrefillPP4ObservationPredictions predicted;
                predicted.requests[1].readyAfterObservationMs = 100.0;
                auto value = prepare(observed, 0, predicted);
                ASSERT_EQ(prompt - 1, value.requests.front().contextTokens);
                ASSERT_EQ(1, value.requests.front().remainingTokens);
                ASSERT_EQ(prompt, observed.requests.front().cacheHitTokens);
                ASSERT_EQ(cursor, observed.requests.front().computedTokens);
                value.requests.front().owned = true;
                const auto forecast = TPUPrefillPP4Forecast::predict(
                    value.state, value.requests, policy(),
                    {0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1000);
                ASSERT_DOUBLE_EQ(100.0, forecast.firstScheduledTimeMs.at(1));
                ASSERT_EQ(1, forecast.scheduledTokens.at(1));
            }
        }
    }
}

TEST(TPUPrefillPP4ObservationTest, NativeResolvedCacheOverridesTheEarlierMasterEstimate)
{
    auto observed = emptyObservation();
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.cacheHitTokens = 12288;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.requests[1].cacheHitTokens = 4096;
    predicted.requests[1].readyAfterObservationMs = 1000.0;
    const auto value = prepare(observed, 100, predicted);
    ASSERT_EQ(12288, value.requests.front().contextTokens);
    ASSERT_DOUBLE_EQ(0.0, value.requests.front().readyTimeMs);
}

TEST(TPUPrefillPP4ObservationTest, MetadataFutureDoesNotBecomeReadyWhenTheReportAges)
{
    auto observed = emptyObservation();
    observed.results.push_back({101, 1000, -1, false, false});
    ASSERT_THROW(prepare(observed, 200000, {}), std::invalid_argument);
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.controlReadyAfterObservationMs[101] = 300.0;
    const auto value = prepare(observed, 200000, predicted);
    ASSERT_FALSE(value.state.results.front().ready);
    ASSERT_FALSE(value.state.results.front().slotOrdinal);
    ASSERT_DOUBLE_EQ(300.0, value.state.results.front().controlReadyTimeMs);
}

TEST(TPUPrefillPP4ObservationTest,
     PreparedPromptNormalizationIsExplicitAndDoesNotFreshenTheClock)
{
    auto observed = emptyObservation();
    auto request = nativeRequest(1, NativePPQueue::PREPARED, 0);
    request.phase = NativePPRequestPhase::NOT_REGISTERED;
    request.prefillSlotRelease = NativePPPrefillSlotRelease::UNKNOWN;
    request.cacheHitTokens = -1;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.requests[1].cacheHitTokens = 4096;
    predicted.requests[1].readyAfterObservationMs = 10.0;
    predicted.requests[1].needsFirstQuery = true;
    ASSERT_THROW(prepare(observed, 100000, predicted), std::invalid_argument);
    predicted.requests[1].nativePromptTokens = 32767;
    ASSERT_THROW(prepare(observed, 100000, predicted), std::invalid_argument);
    predicted.requests[1].releaseSlotAfterPrefill = true;
    const auto value = prepare(observed, 100000, predicted);
    ASSERT_EQ(28671, value.requests.front().remainingTokens);
    ASSERT_TRUE(value.requests.front().needsFirstQuery);
    ASSERT_DOUBLE_EQ(10.0, value.requests.front().readyTimeMs);
    ASSERT_DOUBLE_EQ(100.0, value.state.nowMs);
}

TEST(TPUPrefillPP4ObservationTest, DecidedSpansAreReservedBeforeARealSlotExists)
{
    auto observed = emptyObservation();
    auto shape = realSlot(0, 1, 4096, 8192, 0);
    observed.pendingDecision =
        NaviSchedNativePPDecision{500, shape.features, shape.spans};
    auto request = nativeRequest(1, NativePPQueue::RUNNING, 0);
    request.computedTokens = 12288;
    request.cacheHitTokens = 4096;
    observed.requests.push_back(request);
    observed.occupiedRequestSlotCount = 1;
    TPUPrefillPP4ObservationPredictions predicted;
    ASSERT_THROW(prepare(observed, 1000, predicted), std::invalid_argument);
    predicted.pendingDispatchAfterObservationMs = 0.25;
    const auto value = prepare(observed, 1000, predicted);
    ASSERT_TRUE(value.state.slots.empty());
    ASSERT_TRUE(value.state.pendingDispatch);
    ASSERT_DOUBLE_EQ(0.25, value.state.pendingDispatch->dispatchTimeMs);
    ASSERT_EQ(8192, value.requests.front().reservedTokens);
    ASSERT_EQ(4096, value.requests.front().contextTokens);
}

TEST(TPUPrefillPP4ObservationTest, PendingQueryRetainsItsDecidedProposalFence)
{
    auto observed = emptyObservation();
    addResult(observed, realSlot(0, 10, 0, 8192, 1000));
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.cacheHitTokens = -1;
    request.phase = NativePPRequestPhase::QUERY_PENDING;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.requests[1].cacheHitTokens = 4096;
    predicted.requests[1].receiveDependency =
        TPUPrefillKVReceiveDependency{0, std::nullopt, 1};
    predicted.requests[1].readyAfterObservationMs = 0.0;
    const auto value = prepare(observed, 100, predicted);
    ASSERT_TRUE(value.requests.front().receiveDependency);
    ASSERT_EQ(1, *value.requests.front().receiveDependency->decidedProposalFence);
    ASSERT_FALSE(value.requests.front().receiveDependency->reporterOrdinal);
}

TEST(TPUPrefillPP4ObservationTest, PreemptedCursorDoesNotReserveAnObsoleteSpanAgain)
{
    auto observed = emptyObservation();
    addResult(observed, realSlot(0, 1, 4096, 4096, 1000));
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.phase = NativePPRequestPhase::PREEMPTED;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.requests[1].readyAfterObservationMs = 0.0;
    predicted.requests[1].needsFirstQuery = false;
    const auto value = prepare(observed, 0, predicted);
    ASSERT_EQ(0, value.requests.front().reservedTokens);
    ASSERT_EQ(0, value.requests.front().contextTokens);
}

TEST(TPUPrefillPP4ObservationTest, RejectsGapsRatherThanHidingMissingReservationWork)
{
    auto observed = emptyObservation();
    addResult(observed, realSlot(0, 1, 0, 4096, 2000));
    addResult(observed, realSlot(1, 1, 8192, 4096, 1000));
    auto request = nativeRequest(1, NativePPQueue::RUNNING, 0);
    request.computedTokens = 12288;
    observed.requests.push_back(request);
    observed.occupiedRequestSlotCount = 1;
    ASSERT_THROW(prepare(observed, 0, {}), std::invalid_argument);
}

TEST(TPUPrefillPP4ObservationTest, RejectsContradictoryReplyFactsAndUnrelatedPredictions)
{
    auto observed = emptyObservation();
    auto slot = realSlot(0, 1, 0, 4096, 1000);
    slot.lastStageReplyAgeUs = 0;
    addResult(observed, slot, false, false);
    ASSERT_THROW(prepare(observed, 0, {}), std::invalid_argument);
    observed = emptyObservation();
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.controlReadyAfterObservationMs[999] = 1.0;
    ASSERT_THROW(prepare(observed, 0, predicted), std::invalid_argument);
    predicted = {};
    predicted.requests[999].cacheHitTokens = 0;
    ASSERT_THROW(prepare(observed, 0, predicted), std::invalid_argument);
    ASSERT_THROW(prepare(observed, -1, {}), std::invalid_argument);
}

TEST(TPUPrefillPP4ObservationTest,
     NormalizedIncomingWorkCannotDuplicatePreparedOrNativeRequests)
{
    auto observed = emptyObservation();
    observed.requests.push_back(nativeRequest(999, NativePPQueue::WAITING, 0));
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.incoming.push_back(incoming());
    ASSERT_THROW(prepare(observed, 0, predicted), std::invalid_argument);
    observed.requests.clear();
    const auto value = prepare(observed, 1000, predicted);
    ASSERT_TRUE(value.requests.front().owned);
    ASSERT_DOUBLE_EQ(1.0, value.state.nowMs);
    ASSERT_DOUBLE_EQ(0.0, value.requests.front().readyTimeMs);
}

TEST(TPUPrefillPP4ObservationTest, MetadataReadinessIsKeptThroughConversionAndPrediction)
{
    auto observed = emptyObservation();
    observed.results.push_back({101, 1000, -1, false, false});
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.controlReadyAfterObservationMs[101] = 200.0;
    predicted.incoming.push_back(incoming());
    const auto value = prepare(observed, 0, predicted);
    TPUPrefillPP4Timing::Parameters parameters{0.0, 10.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0,  0.0, 0.0};
    const auto output = TPUPrefillPP4Forecast::predict(value.state, value.requests,
                                                       policy(), parameters, 100);
    ASSERT_GE(output.totalMs, 200.0);
}

TEST(TPUPrefillPP4ObservationTest, FullyReservedNativeRequestKeepsItsFinalOutputOrdinal)
{
    auto observed = emptyObservation();
    addResult(observed, realSlot(0, 1, 0, 4096, 2000), true, true);
    addResult(observed, realSlot(1, 1, 4096, 4096, 1000), true, true);
    auto request = nativeRequest(1, NativePPQueue::RUNNING, 0);
    request.promptTokens = request.computedTokens = 8192;
    observed.requests.push_back(request);
    observed.occupiedRequestSlotCount = 1;
    const auto value = prepare(observed, 0, {});
    ASSERT_EQ(1, value.state.requestCompletionSlots.at(1));
    ASSERT_EQ(8192, value.requests.front().reservedTokens);
    ASSERT_EQ(8192, value.requests.front().remainingTokens);
}

TEST(TPUPrefillPP4ObservationTest, PendingFinalComputeSuppliesTheReleaseOrdinal)
{
    auto observed = emptyObservation();
    auto slot = realSlot(0, 1, 0, 8192, 0);
    observed.pendingDecision = NaviSchedNativePPDecision{500, slot.features, slot.spans};
    auto request = nativeRequest(1, NativePPQueue::RUNNING, 0);
    request.promptTokens = request.computedTokens = 8192;
    observed.requests.push_back(request);
    observed.occupiedRequestSlotCount = 1;
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.pendingDispatchAfterObservationMs = 0.1;
    const auto value = prepare(observed, 0, predictions);
    ASSERT_EQ(0, value.state.requestCompletionSlots.at(1));
}

TEST(TPUPrefillPP4ObservationTest, OtherNativeOccupancyRequiresExplicitReleasePredictions)
{
    auto observed = emptyObservation();
    TPUPrefillPP4ObservationPredictions predictions;
    ASSERT_THROW(prepare(observed, 50000, predictions, {64, 2}), std::invalid_argument);
    predictions.otherRequestReleaseAfterObservationMs = {100.0, 200.0};
    observed.occupiedRequestSlotCount = 2;
    const auto value = prepare(observed, 50000, predictions, {64, 2});
    ASSERT_EQ(64, value.state.maxRunningRequestCount);
    ASSERT_DOUBLE_EQ(100.0, value.state.otherRequestReleaseTimesMs.front());
    ASSERT_DOUBLE_EQ(50.0, value.state.nowMs);
    ASSERT_THROW(prepare(observed, 0, predictions, {0, 2}), std::invalid_argument);
}

TEST(TPUPrefillPP4ObservationTest, ObservedOccupancyCannotExceedNativeCapacity)
{
    auto observed = emptyObservation();
    observed.requests.push_back(nativeRequest(1, NativePPQueue::RUNNING, 0));
    observed.occupiedRequestSlotCount = 1;
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.otherRequestReleaseAfterObservationMs = {10.0};
    ASSERT_THROW(prepare(observed, 0, predictions, {1, 1}), std::invalid_argument);
}

TEST(TPUPrefillPP4ObservationTest, NativeReceiptOverridesAStaleLoadingPhase)
{
    const auto workload = loadingWorkload();
    ASSERT_TRUE(workload.valid());
    ASSERT_THROW(prepare(*workload.nativePPState, 5000, {}), std::invalid_argument);
    autil::DataBuffer buffer;
    workload.serialize(buffer);
    NaviSchedWorkloadSnapshot received;
    received.deserialize(buffer);
    ASSERT_TRUE(received.valid());
    ASSERT_EQ(NativeCacheReceiveState::RECEIVED,
              received.nativeCacheQueries[0].receiveState);
    TPUPrefillPP4ObservationInput input;
    ASSERT_NO_THROW(input = TPUPrefillPP4Observation::prepare(received, {}));
    ASSERT_DOUBLE_EQ(0.0, input.requests.front().readyTimeMs);
    ASSERT_DOUBLE_EQ(5.0, input.state.nowMs);
    ASSERT_FALSE(input.requests.front().receiveDependency);
    ASSERT_FALSE(input.requests.front().receiveHostResultIndex);
    ASSERT_FALSE(input.requests.front().metadataOnlyLoad);
    ASSERT_EQ(4096, input.requests.front().contextTokens);
    ASSERT_EQ(4096, input.requests.front().remainingTokens);
    ASSERT_EQ(NativePPRequestPhase::KV_LOADING,
              received.nativePPState->requests.front().phase);
    ASSERT_EQ(4096, received.nativePPState->requests.front().computedTokens);
    input.requests.front().owned = true;
    const auto forecast = TPUPrefillPP4Forecast::predict(
        input.state, input.requests, policy(),
        {0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1000);
    ASSERT_DOUBLE_EQ(5.0, forecast.firstScheduledTimeMs.at(1));
    ASSERT_DOUBLE_EQ(0.0, forecast.queueMs);
}

TEST(TPUPrefillPP4ObservationTest, ReceiptDoesNotReleaseOtherNativeRequestSlots)
{
    auto workload = loadingWorkload();
    workload.maxRunningRequestCount = 1;
    auto observed = std::make_shared<NaviSchedNativePPState>(*workload.nativePPState);
    observed->occupiedRequestSlotCount = 1;
    workload.nativePPState = observed;
    ASSERT_TRUE(workload.valid());
    TPUPrefillPP4ObservationPredictions predicted;
    predicted.otherRequestReleaseAfterObservationMs = {100.0};
    TPUPrefillPP4ObservationInput input;
    ASSERT_NO_THROW(input = TPUPrefillPP4Observation::prepare(workload, predicted));
    input.requests.front().owned = true;
    const auto forecast = TPUPrefillPP4Forecast::predict(
        input.state, input.requests, policy(),
        {0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1000);
    ASSERT_DOUBLE_EQ(100.0, forecast.firstScheduledTimeMs.at(1));
}

TEST(TPUPrefillPP4ObservationTest, FailedMissingOrOtherReceiptsDoNotMakeLoadingReady)
{
    for (int mode = 0; mode < 5; ++mode) {
        auto workload = loadingWorkload();
        if (mode == 0) workload.nativeCacheQueries.clear();
        if (mode == 1)
            workload.nativeCacheQueries[0].receiveState =
                NativeCacheReceiveState::NOT_REPORTED;
        if (mode == 2)
            workload.nativeCacheQueries[0].receiveState = NativeCacheReceiveState::FAILED;
        if (mode == 3)
            workload.nativeCacheQueries[0].loadState = NativeCacheLoadState::FAILED;
        if (mode == 4) workload.nativeCacheQueries[0].requestId = 2;
        ASSERT_TRUE(workload.valid());
        ASSERT_THROW(TPUPrefillPP4Observation::prepare(workload, {}),
                     std::invalid_argument);
    }
}

TEST(TPUPrefillPP4ObservationTest, ReceiptSupersedesEarlierLoadReadinessPredictions)
{
    const auto workload = loadingWorkload();
    for (int mode = 0; mode < 3; ++mode) {
        TPUPrefillPP4ObservationPredictions predicted;
        auto &p = predicted.requests[1];
        p.readyAfterObservationMs = 500.0;
        if (mode == 1) p.receiveSchedulerOutputId = 999;
        if (mode == 2) {
            p.receiveDependency = TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
            p.metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
        }
        TPUPrefillPP4ObservationInput input;
        ASSERT_NO_THROW(input = TPUPrefillPP4Observation::prepare(workload, predicted));
        ASSERT_DOUBLE_EQ(0.0, input.requests.front().readyTimeMs);
        ASSERT_FALSE(input.requests.front().receiveDependency);
        ASSERT_FALSE(input.requests.front().receiveHostResultIndex);
        ASSERT_FALSE(input.requests.front().metadataOnlyLoad);
    }
}

TEST(TPUPrefillPP4ObservationTest, ReceiptConversionMicrobenchmark)
{
    for (const size_t count : {size_t(1), size_t(16), size_t(64)}) {
        for (int mode = 0; mode < 3; ++mode) {
            auto workload = loadingWorkload(count);
            if (mode == 0) workload.nativeCacheQueries.clear();
            if (mode == 1)
                for (auto &q : workload.nativeCacheQueries)
                    q.receiveState = NativeCacheReceiveState::NOT_REPORTED;
            ASSERT_TRUE(workload.valid());
            TPUPrefillPP4ObservationPredictions predicted;
            for (size_t id = 1; id <= count; ++id)
                predicted.requests[id].readyAfterObservationMs = 0.0;
            for (int i = 0; i < 100; ++i)
                TPUPrefillPP4Observation::prepare(workload, predicted);
            int64_t checksum = 0;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 2000; ++i) {
                const auto input = TPUPrefillPP4Observation::prepare(workload, predicted);
                checksum += input.requests.size() + input.requests.front().contextTokens;
            }
            const auto elapsed = std::chrono::duration<double, std::micro>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
            ASSERT_EQ(2000 * int64_t(count + 4096), checksum);
            std::cout << "NAVI_RECEIPT_BENCH count=" << count << " mode=" << mode
                      << " mean_us=" << elapsed / 2000 << " checksum=" << checksum
                      << std::endl;
        }
    }
}

TEST(TPUPrefillPP4ObservationTest, WorkloadUsesNativeOccupancyInsteadOfProjectedCapacity)
{
    NaviSchedWorkloadSnapshot workload;
    workload.revision = workload.eventTimeUs = 1;
    workload.prefillCapacitySplit = true;
    workload.maxRunningRequestCount = 2;
    auto observed = std::make_shared<NaviSchedNativePPState>(emptyObservation());
    observed->occupiedRequestSlotCount = 2;
    workload.nativePPState = observed;
    workload.nativePPObservationAgeUs = 5000;
    ASSERT_TRUE(workload.valid());
    // No prefill RUNNING row, but two Native slots are held by other requests.
    ASSERT_EQ(0, workload.runningCapacitySlotCount);
    TPUPrefillPP4ObservationPredictions predictions;
    ASSERT_THROW(TPUPrefillPP4Observation::prepare(workload, predictions),
                 std::invalid_argument);
    predictions.otherRequestReleaseAfterObservationMs = {20.0, 40.0};
    const auto input = TPUPrefillPP4Observation::prepare(workload, predictions);
    ASSERT_EQ(2, input.state.maxRunningRequestCount);
    ASSERT_DOUBLE_EQ(5.0, input.state.nowMs);
    ASSERT_EQ(std::vector<double>({20.0, 40.0}), input.state.otherRequestReleaseTimesMs);
    observed->occupiedRequestSlotCount = 3;
    ASSERT_FALSE(workload.valid());
    ASSERT_THROW(TPUPrefillPP4Observation::prepare(workload, predictions),
                 std::invalid_argument);
}

TEST(TPUPrefillPP4ObservationTest, ExplicitCapacityCannotReplaceAnObservedOccupiedSlot)
{
    auto observed = emptyObservation();
    observed.occupiedRequestSlotCount = 1;
    ASSERT_THROW(prepare(observed, 0, {}, {64, 0}), std::invalid_argument);
}

TEST(TPUPrefillPP4ObservationTest, NativeSlotRetentionCannotBeOverriddenByAPrediction)
{
    auto observed = emptyObservation();
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.prefillSlotRelease = NativePPPrefillSlotRelease::RETAINS_SLOT;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predictions;
    const auto value = prepare(observed, 0, predictions);
    ASSERT_TRUE(value.requests.front().releaseSlotAfterPrefill.has_value());
    ASSERT_FALSE(*value.requests.front().releaseSlotAfterPrefill);
    predictions.requests[1].releaseSlotAfterPrefill = true;
    ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
}

TEST(TPUPrefillPP4ObservationTest, IncomingRequestNeedsItsNormalizedReleaseCondition)
{
    auto observed = emptyObservation();
    TPUPrefillPP4ObservationPredictions predictions;
    auto value = incoming();
    value.releaseSlotAfterPrefill.reset();
    predictions.incoming.push_back(value);
    ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
}

TEST(TPUPrefillPP4ObservationTest, PendingLoadKeepsItsQueryReadinessPrediction)
{
    auto observed = emptyObservation();
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.phase = NativePPRequestPhase::QUERY_PENDING;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.requests[1].receiveDependency =
        TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    predictions.requests[1].readyAfterObservationMs = 30.0;
    const auto value = prepare(observed, 100, predictions);
    ASSERT_DOUBLE_EQ(30.0, value.requests.front().readyTimeMs);
}

TEST(TPUPrefillPP4ObservationTest, PendingLoadRequiresAQueryReadinessPrediction)
{
    auto observed = emptyObservation();
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.phase = NativePPRequestPhase::QUERY_PENDING;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.requests[1].receiveDependency =
        TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    ASSERT_THROW(prepare(observed, 100, predictions), std::invalid_argument);
}

TEST(TPUPrefillPP4ObservationTest, MetadataLoadPredictionReachesTheNativeForecast)
{
    auto observed = emptyObservation();
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.promptTokens = 8192;
    request.phase = NativePPRequestPhase::QUERY_PENDING;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.requests[1].receiveDependency =
        TPUPrefillKVReceiveDependency{-1, std::nullopt, -1};
    predictions.requests[1].readyAfterObservationMs = 12.0;
    predictions.requests[1].metadataOnlyLoad = TPUPrefillMetadataKVLoad{40.0, 4.0};
    auto value = prepare(observed, 1000, predictions);
    ASSERT_TRUE(value.requests.front().metadataOnlyLoad);
    EXPECT_DOUBLE_EQ(40.0, value.requests.front().metadataOnlyLoad->localLoadMs);
    EXPECT_DOUBLE_EQ(4.0, value.requests.front().metadataOnlyLoad->controlReplyMs);
    value.requests.front().owned = true;
    TPUPrefillPP4Timing::Parameters parameters{};
    parameters[1] = 10.0;
    const auto result = TPUPrefillPP4Forecast::predict(value.state, value.requests,
                                                       policy(), parameters, 200, true);
    EXPECT_DOUBLE_EQ(56.0, result.firstScheduledTimeMs.at(1));
    EXPECT_DOUBLE_EQ(55.0, result.queueMs);
    EXPECT_EQ(8192, result.scheduledTokens.at(1));
}

TEST(TPUPrefillPP4ObservationTest, BindsMetadataReceivePredictionToObservedCallIdentity)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    auto observed = emptyObservation();
    observed.measuredForwardMs = 10.0;
    observed.results = {{10, 1000, -1, false, false}, {11, 500, -1, true, true}};
    observed.results[1].loadRequestIds = {1};
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.phase = NativePPRequestPhase::KV_LOADING;
    request.promptTokens = 8192;
    request.loadAgeUs = 500;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.controlReadyAfterObservationMs = {{10, 100.0}};
    predictions.requests[1].receiveSchedulerOutputId = 11;
    auto input = prepare(observed, 0, predictions);
    ASSERT_EQ(std::optional<size_t>(1), input.requests[0].receiveHostResultIndex);
    input.requests[0].owned = true;
    const auto value =
        TPUPrefillPP4Forecast::predict(input.state, input.requests, policy(),
                                       {0., 10., 0., 0., 0., 0., 0., 0., 0.}, 1000);
    ASSERT_DOUBLE_EQ(100.0, value.queueMs);
    ASSERT_DOUBLE_EQ(140.0, value.totalMs);
}

TEST(TPUPrefillPP4ObservationTest,
     ReceivePredictionRequiresAnObservedLoadAndUnconsumedCall)
{
    // Native handoff: older PP stages progress while stage 0 awaits dispatch.
    auto observed = emptyObservation();
    observed.measuredForwardMs = 10.0;
    observed.results = {{10, 1000, -1, true, true}};
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.phase = NativePPRequestPhase::KV_LOADING;
    request.promptTokens = 8192;
    observed.requests.push_back(request);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.requests[1].receiveSchedulerOutputId = 10;
    ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    observed.results[0].loadRequestIds = {1};
    ASSERT_EQ(std::optional<size_t>(0),
              prepare(observed, 0, predictions).requests[0].receiveHostResultIndex);
    predictions.requests[1].receiveSchedulerOutputId = 404;
    ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    predictions.requests[1].receiveSchedulerOutputId = 10;
    predictions.requests[1].readyAfterObservationMs = 2.0;
    ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    predictions.requests[1].readyAfterObservationMs.reset();
    observed.results[0].loadRequestIds.clear();
    observed.pendingDecision = NaviSchedNativePPDecision{};
    observed.pendingDecision->schedulerOutputId = 12;
    observed.pendingDecision->loadRequestIds = {1};
    predictions.pendingDispatchAfterObservationMs = 5.0;
    predictions.controlReadyAfterObservationMs = {{12, 25.0}};
    ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    predictions.requests[1].receiveSchedulerOutputId = 12;
    auto input = prepare(observed, 0, predictions);
    ASSERT_EQ(std::optional<size_t>(1), input.requests[0].receiveHostResultIndex);
    input.requests[0].owned = true;
    const auto value =
        TPUPrefillPP4Forecast::predict(input.state, input.requests, policy(),
                                       {0., 10., 0., 0., 0., 0., 0., 0., 0.}, 1000);
    ASSERT_DOUBLE_EQ(25.0, value.queueMs);
    ASSERT_DOUBLE_EQ(65.0, value.totalMs);
}

TEST(TPUPrefillPP4ObservationTest, QueryServiceRemainsRelativeWhenObservationAges)
{
    for (int64_t age : {int64_t(0), int64_t(5000), int64_t(10000000)}) {
        auto observed = emptyObservation();
        auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
        request.phase = NativePPRequestPhase::QUERY_NOT_STARTED;
        observed.requests.push_back(request);
        TPUPrefillPP4ObservationPredictions predictions;
        auto &prediction = predictions.requests[1];
        prediction.needsFirstQuery = true;
        prediction.readyAfterObservationMs = 10.0;
        prediction.firstQueryServiceMs = 25.0;
        auto input = prepare(observed, age, predictions);
        ASSERT_EQ(size_t(1), input.requests.size());
        ASSERT_TRUE(input.requests[0].firstQueryServiceMs.has_value());
        ASSERT_DOUBLE_EQ(25.0, *input.requests[0].firstQueryServiceMs);
        ASSERT_DOUBLE_EQ(10.0, input.requests[0].readyTimeMs);
        input.requests[0].owned = true;
        const auto output = TPUPrefillPP4Forecast::predict(
            input.state, input.requests, policy(),
            {0., 10., 0., 0., 0., 0., 0., 0., 0.}, 1000, true);
        ASSERT_DOUBLE_EQ(input.state.nowMs > 10.0 ? input.state.nowMs : 10.0,
                         output.firstQueryTimeMs.at(1));
        ASSERT_DOUBLE_EQ(25.0, output.firstQueryReadyTimeMs.at(1) -
                                  output.firstQueryTimeMs.at(1));
    }
}

TEST(TPUPrefillPP4ObservationTest, QueryServiceRejectsAlreadyPendingOrKnownCachePhase)
{
    for (auto phase : {NativePPRequestPhase::QUERY_PENDING,
                       NativePPRequestPhase::CACHE_KNOWN}) {
        auto observed = emptyObservation();
        auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
        request.phase = phase;
        observed.requests.push_back(request);
        TPUPrefillPP4ObservationPredictions predictions;
        predictions.requests[1].readyAfterObservationMs = 10.0;
        predictions.requests[1].firstQueryServiceMs = 25.0;
        ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    }
}

TEST(TPUPrefillPP4ObservationTest, QueryServiceRejectsInvalidServiceBeforeForecast)
{
    auto observed = emptyObservation();
    auto request = nativeRequest(1, NativePPQueue::WAITING, 0);
    request.phase = NativePPRequestPhase::QUERY_NOT_STARTED;
    observed.requests.push_back(request);
    for (double invalid : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
        TPUPrefillPP4ObservationPredictions predictions;
        predictions.requests[1].needsFirstQuery = true;
        predictions.requests[1].readyAfterObservationMs = 0.0;
        predictions.requests[1].firstQueryServiceMs = invalid;
        ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    }
}


TEST(TPUPrefillPP4ObservationTest, QueryBacklogPreservesObservedAgeAndOwnCompletion)
{
    auto observed = emptyObservation();
    auto pending = nativeRequest(1, NativePPQueue::WAITING, 0);
    pending.phase = NativePPRequestPhase::QUERY_PENDING;
    pending.queryAgeUs = 30000;
    observed.requests.push_back(pending);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.queryServiceParameters = std::array<double, 5>{10., 0., 1., 0., 1.};
    auto &prediction = predictions.requests[1];
    prediction.readyAfterObservationMs = 200.0;
    prediction.queryReadyAfterObservationMs = 80.0;
    prediction.queryBlockCount = 7;
    const auto input = prepare(observed, 10000, predictions);
    ASSERT_EQ(predictions.queryServiceParameters, input.state.queryServiceParameters);
    ASSERT_DOUBLE_EQ(40.0, *input.requests[0].pendingQueryAgeMs);
    ASSERT_DOUBLE_EQ(80.0, *input.requests[0].pendingQueryReadyTimeMs);
    ASSERT_DOUBLE_EQ(200.0, input.requests[0].readyTimeMs);
    ASSERT_FALSE(input.requests[0].needsFirstQuery);
    ASSERT_EQ(30000, observed.requests[0].queryAgeUs);
}

TEST(TPUPrefillPP4ObservationTest, QueryBacklogRejectsMissingClockAndWrongPhase)
{
    for (auto phase : {NativePPRequestPhase::QUERY_PENDING,
                       NativePPRequestPhase::QUERY_READY_NOT_CONSUMED,
                       NativePPRequestPhase::KV_LOADING}) {
        auto observed = emptyObservation();
        auto pending = nativeRequest(1, NativePPQueue::WAITING, 0);
        pending.phase = phase;
        pending.queryAgeUs = phase == NativePPRequestPhase::QUERY_PENDING ? -1 : 30000;
        observed.requests.push_back(pending);
        TPUPrefillPP4ObservationPredictions predictions;
        auto &prediction = predictions.requests[1];
        prediction.readyAfterObservationMs = 200.0;
        prediction.queryReadyAfterObservationMs = 80.0;
        prediction.queryBlockCount = 7;
        ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    }
}

TEST(TPUPrefillPP4ObservationTest, QueryBacklogDoesNotSubstituteWholeCacheReadiness)
{
    auto observed = emptyObservation();
    auto pending = nativeRequest(1, NativePPQueue::WAITING, 0);
    pending.phase = NativePPRequestPhase::QUERY_PENDING;
    pending.queryAgeUs = 30000;
    observed.requests.push_back(pending);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.queryServiceParameters = std::array<double, 5>{10., 0., 1., 0., 1.};
    predictions.requests[1].readyAfterObservationMs = 200.0;
    predictions.requests[1].queryBlockCount = 7;
    ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    predictions.requests[1].queryReadyAfterObservationMs = 80.0;
    ASSERT_NO_THROW(prepare(observed, 0, predictions));
}

TEST(TPUPrefillPP4ObservationTest, QueryBacklogUnissuedInputsReachIssueTimeModel)
{
    auto observed = emptyObservation();
    observed.measuredForwardMs = 10.0;
    auto pending = nativeRequest(1, NativePPQueue::WAITING, 0);
    pending.phase = NativePPRequestPhase::QUERY_NOT_STARTED;
    pending.promptTokens = 1;
    observed.requests.push_back(pending);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.queryServiceParameters = std::array<double, 5>{10., 1., 0., 0., 0.};
    auto &prediction = predictions.requests[1];
    prediction.needsFirstQuery = true;
    prediction.readyAfterObservationMs = 5.0;
    prediction.queryBlockCount = 3;
    auto input = prepare(observed, 10000, predictions);
    input.requests[0].owned = true;
    ASSERT_FALSE(input.requests[0].pendingQueryAgeMs);
    const auto output = TPUPrefillPP4Forecast::predict(
        input.state, input.requests, policy(),
        {0., 10., 0., 0., 0., 0., 0., 0., 0.}, 1000, true);
    ASSERT_DOUBLE_EQ(10.0, output.firstQueryTimeMs.at(1));
    ASSERT_NEAR(20.0 + std::log1p(3.0), output.firstQueryReadyTimeMs.at(1), 1e-10);
}


TEST(TPUPrefillPP4ObservationTest, QueryBacklogMissingPendingSizeIsNotZeroBacklog)
{
    auto observed = emptyObservation();
    auto pending = nativeRequest(1, NativePPQueue::WAITING, 0);
    pending.phase = NativePPRequestPhase::QUERY_PENDING;
    pending.queryAgeUs = 30000;
    observed.requests.push_back(pending);
    TPUPrefillPP4ObservationPredictions predictions;
    predictions.queryServiceParameters = std::array<double, 5>{10., 1., 1., 1., 1.};
    predictions.requests[1].readyAfterObservationMs = 200.0;
    ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    predictions.requests[1].queryReadyAfterObservationMs = 80.0;
    ASSERT_THROW(prepare(observed, 0, predictions), std::invalid_argument);
    predictions.requests[1].queryBlockCount = 7;
    ASSERT_NO_THROW(prepare(observed, 0, predictions));
}

} // namespace navi_sched
