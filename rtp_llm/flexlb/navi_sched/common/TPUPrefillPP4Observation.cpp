#include "aios/navi_sched/common/TPUPrefillPP4Observation.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace navi_sched {
namespace {

double predictionTime(double value)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument("invalid PP4 readiness prediction");
    return value;
}

using Intervals = std::vector<std::pair<int64_t, int64_t>>;

int64_t reservedTail(Intervals intervals, int64_t cursor)
{
    std::sort(intervals.begin(), intervals.end());
    int64_t first = -1, end = -1;
    for (const auto &[start, stop] : intervals) {
        const auto clippedEnd = std::min(stop, cursor);
        if (start >= clippedEnd) continue;
        if (first < 0) {
            first = start;
            end = clippedEnd;
        } else if (start <= end) {
            end = std::max(end, clippedEnd);
        } else {
            throw std::invalid_argument("noncontiguous PP4 unfinished request spans");
        }
    }
    if (first < 0) return 0;
    if (end != cursor)
        throw std::invalid_argument(
            "PP4 unfinished spans do not reach the Native cursor");
    return cursor - first;
}

bool readyAtObservation(const NaviSchedNativePPRequest &request)
{
    return request.queue == NativePPQueue::RUNNING ||
           request.phase == NativePPRequestPhase::CACHE_KNOWN ||
           request.phase == NativePPRequestPhase::LOOKUP_SKIPPED;
}

} // namespace

TPUPrefillPP4ObservationInput TPUPrefillPP4Observation::prepare(
    const NaviSchedWorkloadSnapshot &workload,
    const TPUPrefillPP4ObservationPredictions &predictions)
{
    if (!workload.valid() || !workload.nativePPState)
        throw std::invalid_argument("PP4 prediction requires a valid Native workload");
    const auto &observed = *workload.nativePPState;
    const auto running = std::count_if(
        observed.requests.begin(), observed.requests.end(),
        [](const auto &request) { return request.queue == NativePPQueue::RUNNING; });
    return prepare(
        observed, workload.nativePPObservationAgeUs,
        {workload.maxRunningRequestCount, observed.occupiedRequestSlotCount - running},
        predictions, &workload.nativeCacheQueries);
}

TPUPrefillPP4ObservationInput TPUPrefillPP4Observation::prepare(
    const NaviSchedNativePPState &observed, int64_t observationAgeUs,
    const TPUPrefillPP4ObservedCapacity &capacity,
    const TPUPrefillPP4ObservationPredictions &predictions)
{
    return prepare(observed, observationAgeUs, capacity, predictions, nullptr);
}

TPUPrefillPP4ObservationInput TPUPrefillPP4Observation::prepare(
    const NaviSchedNativePPState &observed, int64_t observationAgeUs,
    const TPUPrefillPP4ObservedCapacity &capacity,
    const TPUPrefillPP4ObservationPredictions &predictions,
    const std::vector<NaviSchedNativeCacheQuery> *cacheQueries)
{
    if (!observed.valid() || observationAgeUs < 0 ||
        capacity.maxRunningRequestCount <= 0 || capacity.otherOccupiedRequestCount < 0 ||
        uint64_t(capacity.otherOccupiedRequestCount) !=
            predictions.otherRequestReleaseAfterObservationMs.size())
        throw std::invalid_argument("invalid PP4 observation input");
    const auto running = std::count_if(
        observed.requests.begin(), observed.requests.end(),
        [](const auto &request) { return request.queue == NativePPQueue::RUNNING; });
    if (capacity.otherOccupiedRequestCount !=
            observed.occupiedRequestSlotCount - running ||
        capacity.otherOccupiedRequestCount > capacity.maxRunningRequestCount ||
        uint64_t(running) > uint64_t(capacity.maxRunningRequestCount -
                                     capacity.otherOccupiedRequestCount))
        throw std::invalid_argument("Native occupied slots exceed the observed capacity");
    TPUPrefillPP4ObservationInput result;
    auto &state = result.state;
    // The earliest dispatch is zero, avoiding negative pipeline origins and
    // retaining sub-millisecond precision without any cross-host clock value.
    const int64_t originAgeUs =
        observed.slots.empty() ? 0 : observed.slots.front().dispatchAgeUs;
    const double observationMs = originAgeUs / 1000.0;
    state.nowMs = observationMs + observationAgeUs / 1000.0;
    state.queryServiceParameters = predictions.queryServiceParameters;
    if (state.queryServiceParameters)
        for (double value : *state.queryServiceParameters) predictionTime(value);
    state.measuredForwardMs = observed.measuredForwardMs;
    state.lastRealOrdinal = observed.lastRealOrdinal;
    state.maxRunningRequestCount = capacity.maxRunningRequestCount;
    for (double when : predictions.otherRequestReleaseAfterObservationMs)
        state.otherRequestReleaseTimesMs.push_back(observationMs + predictionTime(when));
    std::unordered_map<int64_t, const NaviSchedNativePPSlot *> slots;
    for (const auto &slot : observed.slots) {
        state.slots.push_back(
            {slot.ordinal, (originAgeUs - slot.dispatchAgeUs) / 1000.0, slot.features});
        state.expectedForwardMs.emplace(slot.ordinal, slot.expectedForwardMs);
        slots.emplace(slot.ordinal, &slot);
        if (slot.lastStageReplyAgeUs >= 0)
            state.observedReplies.emplace(
                slot.ordinal, (originAgeUs - slot.lastStageReplyAgeUs) / 1000.0);
    }
    std::unordered_map<uint64_t, Intervals> reservations;
    std::unordered_map<uint64_t, std::vector<std::pair<int64_t, int64_t>>>
        completionCandidates;
    auto reserve = [&](const auto &spans, int64_t ordinal) {
        for (const auto &span : spans) {
            reservations[span.requestId].emplace_back(
                span.computedTokens, span.computedTokens + span.scheduledTokens);
            completionCandidates[span.requestId].emplace_back(
                span.computedTokens + span.scheduledTokens, ordinal);
        }
    };
    std::unordered_map<uint64_t, size_t> outputIndices;
    for (size_t index = 0; index < observed.results.size(); ++index)
        outputIndices.emplace(observed.results[index].schedulerOutputId, index);
    if (observed.pendingDecision)
        outputIndices.emplace(observed.pendingDecision->schedulerOutputId,
                              observed.results.size());
    std::set<uint64_t> controlIds;
    auto controlTime = [&](uint64_t id) {
        const auto found = predictions.controlReadyAfterObservationMs.find(id);
        if (found == predictions.controlReadyAfterObservationMs.end())
            throw std::invalid_argument("missing PP4 metadata-call readiness prediction");
        controlIds.insert(id);
        return observationMs + predictionTime(found->second);
    };
    for (const auto &pending : observed.results) {
        TPUPrefillPP4HostResult host;
        host.ready = pending.resultReady;
        if (pending.slotOrdinal >= 0) {
            host.slotOrdinal = pending.slotOrdinal;
            reserve(slots.at(pending.slotOrdinal)->spans, pending.slotOrdinal);
            if (!pending.executeReplyReady) {
                const auto *slot = slots.at(pending.slotOrdinal);
                if (slot->lastStageReplyAgeUs >= 0)
                    throw std::invalid_argument(
                        "contradictory PP4 execute-reply observation");
                state.replyLowerBounds.emplace(pending.slotOrdinal, observationMs);
            }
        } else if (!host.ready) {
            host.controlReadyTimeMs = controlTime(pending.schedulerOutputId);
        }
        state.results.push_back(host);
    }
    if (observed.pendingDecision) {
        if (!predictions.pendingDispatchAfterObservationMs)
            throw std::invalid_argument("missing decided PP4 dispatch-time prediction");
        const auto &decision = *observed.pendingDecision;
        TPUPrefillPP4PendingDispatch dispatch;
        dispatch.dispatchTimeMs =
            observationMs +
            predictionTime(*predictions.pendingDispatchAfterObservationMs);
        dispatch.features = decision.features;
        for (const auto &span : decision.spans)
            dispatch.spans.push_back(
                {span.requestId, span.computedTokens, span.scheduledTokens});
        if (!decision.spans.empty()) {
            reserve(decision.spans,
                    state.lastRealOrdinal < 0 ? 0 : state.slots.back().ordinal + 1);
            dispatch.expectedForwardMs = observed.measuredForwardMs;
        } else {
            dispatch.controlReadyTimeMs = controlTime(decision.schedulerOutputId);
        }
        state.pendingDispatch = std::move(dispatch);
    } else if (predictions.pendingDispatchAfterObservationMs) {
        throw std::invalid_argument("PP4 dispatch prediction has no decided output");
    }
    for (const auto &[id, time] : predictions.controlReadyAfterObservationMs) {
        if (!controlIds.count(id))
            throw std::invalid_argument("PP4 control readiness has no unresolved call");
    }
    std::vector<const NaviSchedNativePPRequest *> ordered;
    std::set<uint64_t> ids;
    for (const auto &request : observed.requests) {
        ordered.push_back(&request);
        ids.insert(request.requestId);
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto *a, const auto *b) {
        return std::make_pair(int32_t(a->queue), a->queueIndex) <
               std::make_pair(int32_t(b->queue), b->queueIndex);
    });
    for (const auto *request : ordered) {
        const auto found = predictions.requests.find(request->requestId);
        const TPUPrefillPP4RequestPrediction empty;
        const auto &prediction =
            found == predictions.requests.end() ? empty : found->second;
        int64_t prompt = request->promptTokens;
        if (request->queue == NativePPQueue::PREPARED) {
            if (!prediction.nativePromptTokens)
                throw std::invalid_argument(
                    "prepared PP4 request needs its normalized Native prompt");
            prompt = *prediction.nativePromptTokens;
            if (prompt <= 0 || prompt > request->promptTokens)
                throw std::invalid_argument("invalid normalized PP4 prompt");
        } else if (prediction.nativePromptTokens &&
                   *prediction.nativePromptTokens != prompt) {
            throw std::invalid_argument(
                "registered PP4 prompt cannot be normalized again");
        }
        TPUPrefillPP4Request value;
        value.requestId = request->requestId;
        value.metadataOnlyLoad = prediction.metadataOnlyLoad;
        value.firstQueryServiceMs = prediction.firstQueryServiceMs;
        value.queryBlockCount = prediction.queryBlockCount;
        if (value.queryBlockCount && *value.queryBlockCount < 0)
            throw std::invalid_argument("invalid PP4 query block prediction");
        if (prediction.queryReadyAfterObservationMs) {
            if (request->phase != NativePPRequestPhase::QUERY_PENDING ||
                request->queryAgeUs < 0 || !value.queryBlockCount)
                throw std::invalid_argument("query readiness lacks a pending query");
            value.pendingQueryAgeMs = request->queryAgeUs / 1000.0 +
                                      observationAgeUs / 1000.0;
            value.pendingQueryReadyTimeMs = observationMs +
                predictionTime(*prediction.queryReadyAfterObservationMs);
        } else if (state.queryServiceParameters &&
                   request->phase == NativePPRequestPhase::QUERY_PENDING) {
            throw std::invalid_argument("missing PP4 background query readiness");
        }
        value.releaseSlotAfterPrefill =
            request->prefillSlotRelease == NativePPPrefillSlotRelease::UNKNOWN
                ? prediction.releaseSlotAfterPrefill
                : std::optional<bool>(request->prefillSlotRelease ==
                                      NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT);
        if (!value.releaseSlotAfterPrefill)
            throw std::invalid_argument(
                "PP4 request needs an explicit prefill slot release condition");
        if (prediction.releaseSlotAfterPrefill &&
            *prediction.releaseSlotAfterPrefill != *value.releaseSlotAfterPrefill)
            throw std::invalid_argument(
                "PP4 prediction contradicts Native prefill slot release");
        const int64_t cursor = std::min(prompt, request->computedTokens);
        value.reservedTokens = reservedTail(reservations[request->requestId], cursor);
        value.contextTokens = cursor - value.reservedTokens;
        if (request->queue != NativePPQueue::RUNNING && value.reservedTokens == 0) {
            const auto hit = request->cacheHitTokens >= 0
                                 ? std::optional<int64_t>(request->cacheHitTokens)
                                 : prediction.cacheHitTokens;
            if (!hit || *hit < 0 || *hit > prompt)
                throw std::invalid_argument(
                    "unknown PP4 cache prefix needs an explicit prediction");
            value.contextTokens =
                std::min(prompt - 1, std::max(value.contextTokens, *hit));
        }
        value.remainingTokens = prompt - value.contextTokens;
        if (request->queue == NativePPQueue::RUNNING &&
            value.remainingTokens == value.reservedTokens) {
            std::optional<int64_t> completion;
            for (const auto &[end, ordinal] : completionCandidates[request->requestId]) {
                if (end >= prompt)
                    completion = completion ? std::min(*completion, ordinal) : ordinal;
            }
            if (!completion)
                throw std::invalid_argument(
                    "fully scheduled Native PP request has no observed completion");
            state.requestCompletionSlots.emplace(request->requestId, *completion);
        }
        value.sharedPrefixBoundary = request->sharedPrefixBoundary;
        value.queue = request->queue == NativePPQueue::RUNNING
                          ? TPUPrefillForecastQueue::RUNNING
                      : request->queue == NativePPQueue::SKIPPED_WAITING
                          ? TPUPrefillForecastQueue::SKIPPED_WAITING
                          : TPUPrefillForecastQueue::WAITING;
        if (request->queue == NativePPQueue::NOT_QUEUED)
            throw std::invalid_argument("PP4 request has no observed queue position");
        value.readyTimeMs = observationMs;
        bool nativeReceived = false;
        if (cacheQueries && request->phase == NativePPRequestPhase::KV_LOADING) {
            const auto query = std::lower_bound(
                cacheQueries->begin(), cacheQueries->end(), request->requestId,
                [](const auto &entry, uint64_t id) { return entry.requestId < id; });
            nativeReceived = query != cacheQueries->end() &&
                             query->requestId == request->requestId &&
                             query->receiveState == NativeCacheReceiveState::RECEIVED &&
                             query->loadState != NativeCacheLoadState::FAILED;
        }
        if (nativeReceived) value.metadataOnlyLoad.reset();
        if (prediction.receiveSchedulerOutputId && !nativeReceived) {
            const auto reporter =
                outputIndices.find(*prediction.receiveSchedulerOutputId);
            if (reporter == outputIndices.end() ||
                request->phase != NativePPRequestPhase::KV_LOADING ||
                prediction.receiveDependency || prediction.metadataOnlyLoad ||
                prediction.readyAfterObservationMs ||
                prediction.needsFirstQuery.value_or(false))
                throw std::invalid_argument(
                    "invalid observed PP4 KV receive call prediction");
            const auto carriesLoad = [&](const auto &call) {
                return std::binary_search(call.loadRequestIds.begin(),
                                          call.loadRequestIds.end(), request->requestId);
            };
            const bool pendingCarrier =
                observed.pendingDecision && carriesLoad(*observed.pendingDecision);
            const bool issuedCarrier = std::any_of(observed.results.begin(),
                                                   observed.results.end(), carriesLoad);
            if (request->loadAgeUs < 0 && !issuedCarrier && !pendingCarrier)
                throw std::invalid_argument(
                    "PP4 receive prediction has no observed load issue");
            if (pendingCarrier && reporter->second < observed.results.size() &&
                observed.results[reporter->second].resultReady)
                throw std::invalid_argument(
                    "ready PP4 result cannot report a later pending load");
            value.receiveHostResultIndex = reporter->second;
        }
        if (!readyAtObservation(*request) && !value.receiveHostResultIndex &&
            !nativeReceived) {
            value.receiveDependency = prediction.receiveDependency;
            const bool pendingQuery =
                value.receiveDependency && value.receiveDependency->decidedProposalFence;
            if (!value.receiveDependency ||
                (pendingQuery &&
                 request->phase != NativePPRequestPhase::QUERY_READY_NOT_CONSUMED)) {
                if (!prediction.readyAfterObservationMs)
                    throw std::invalid_argument(
                        "waiting PP4 request needs a readiness prediction");
                value.readyTimeMs =
                    observationMs + predictionTime(*prediction.readyAfterObservationMs);
            }
        }
        if (request->phase == NativePPRequestPhase::QUERY_NOT_STARTED ||
            request->phase == NativePPRequestPhase::NOT_REGISTERED ||
            request->phase == NativePPRequestPhase::PREEMPTED ||
            request->phase == NativePPRequestPhase::CACHE_UNKNOWN ||
            request->phase == NativePPRequestPhase::UNKNOWN) {
            if (!prediction.needsFirstQuery)
                throw std::invalid_argument(
                    "PP4 first-query phase needs an explicit policy prediction");
            value.needsFirstQuery = *prediction.needsFirstQuery;
        }
        if (value.firstQueryServiceMs) {
            predictionTime(*value.firstQueryServiceMs);
            if (!value.needsFirstQuery)
                throw std::invalid_argument("query service has no unissued Native query");
        }
        result.requests.push_back(std::move(value));
    }
    for (const auto &[id, prediction] : predictions.requests) {
        if (!ids.count(id))
            throw std::invalid_argument("PP4 prediction refers to an unobserved request");
    }
    for (auto incoming : predictions.incoming) {
        if (!incoming.releaseSlotAfterPrefill || !ids.insert(incoming.requestId).second)
            throw std::invalid_argument("invalid or duplicate incoming PP4 request");
        if (incoming.firstQueryServiceMs) {
            predictionTime(*incoming.firstQueryServiceMs);
            if (!incoming.needsFirstQuery)
                throw std::invalid_argument("query service has no unissued incoming query");
        }
        incoming.readyTimeMs = observationMs + predictionTime(incoming.readyTimeMs);
        result.requests.push_back(std::move(incoming));
    }
    return result;
}

} // namespace navi_sched
