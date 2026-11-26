#include "aios/navi_sched/common/TPUPrefillPP4Forecast.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>
#include <unordered_map>

namespace navi_sched {
namespace {

constexpr double NEVER = std::numeric_limits<double>::infinity();

bool validTime(double value) { return std::isfinite(value) && value >= 0.0; }

class Forecast {
public:
    Forecast(const TPUPrefillPP4State &state,
             const std::vector<TPUPrefillPP4Request> &requests,
             const TPUPrefillExecutionPolicyConfig &policy,
             const TPUPrefillPP4Timing::Parameters &parameters, size_t maxTransitions,
             bool captureDispatches)
        : _timing(parameters, state.slots, state.observedReplies, state.replyLowerBounds)
        , _expected(state.expectedForwardMs)
        , _results(state.results)
        , _nextResultId(state.results.size())
        , _requests(requests)
        , _policy(policy)
        , _now(state.nowMs)
        , _begin(state.nowMs)
        , _measured(state.measuredForwardMs)
        , _lastReal(state.lastRealOrdinal)
        , _pending(state.pendingDispatch)
        , _maxRunning(state.maxRunningRequestCount)
        , _otherReleases(state.otherRequestReleaseTimesMs)
        , _remainingTransitions(maxTransitions)
        , _captureDispatches(captureDispatches)
        , _queryServiceParameters(state.queryServiceParameters)
    {
        if (_queryServiceParameters)
            for (double value : *_queryServiceParameters)
                if (!validTime(value))
                    throw std::invalid_argument("invalid PP4 query service parameter");
        std::string error;
        if (!TPUPrefillExecutionPolicy::validate(policy, &error) ||
            policy.fixedPrefillChunkPlan || !validTime(_now) || !validTime(_measured) ||
            !maxTransitions || _maxRunning <= 0 || state.slots.size() > 8 ||
            _results.size() > 8 || realResults() > 4 || _lastReal < -1 ||
            _lastReal > _timing.lastOrdinal() ||
            _timing.lastOrdinal() >= std::numeric_limits<int64_t>::max() - 3) {
            throw std::invalid_argument("invalid continuous PP4 forecast state: " +
                                        error);
        }
        if (state.slots.empty() && (_lastReal != -1 || !_expected.empty())) {
            throw std::invalid_argument("closed PP4 wave has timing history");
        }
        if (_pending) {
            const auto &pending = *_pending;
            if (!validTime(pending.dispatchTimeMs) ||
                !validTime(pending.expectedForwardMs) ||
                !validTime(pending.controlReadyTimeMs) || _results.size() >= 8 ||
                (pending.spans.empty() &&
                 pending.controlReadyTimeMs < pending.dispatchTimeMs) ||
                (!pending.spans.empty() && realResults() >= 4) ||
                (!state.slots.empty() &&
                 pending.dispatchTimeMs < state.slots.back().dispatchTimeMs)) {
                throw std::invalid_argument("invalid decided PP4 dispatch");
            }
            std::set<uint64_t> ids;
            for (const auto &span : pending.spans) {
                if (!span.requestId || !ids.insert(span.requestId).second ||
                    span.computedTokens < 0 || span.scheduledTokens <= 0 ||
                    span.scheduledTokens >
                        std::numeric_limits<int64_t>::max() - span.computedTokens) {
                    throw std::invalid_argument("invalid decided PP4 span");
                }
            }
            for (double feature : pending.features) {
                if (!validTime(feature))
                    throw std::invalid_argument("invalid decided PP4 feature");
            }
            if (pending.features[0] != double(!pending.spans.empty()) ||
                (pending.spans.empty() &&
                 (pending.expectedForwardMs != 0.0 ||
                  std::any_of(pending.features.begin(), pending.features.end(),
                              [](double value) { return value != 0.0; })))) {
                throw std::invalid_argument("decided PP4 shape does not match its work");
            }
        }
        for (const auto &[ordinal, expected] : _expected) {
            if (state.slots.empty() || ordinal < state.slots.front().ordinal ||
                ordinal > _timing.lastOrdinal() || !validTime(expected)) {
                throw std::invalid_argument(
                    "PP4 expectation is outside observed history");
            }
        }
        for (const auto &slot : state.slots) {
            const auto expected = _expected.find(slot.ordinal);
            if (slot.dispatchTimeMs > _now ||
                (expected != _expected.end() && !validTime(expected->second))) {
                throw std::invalid_argument("invalid observed PP4 slot timing");
            }
            _dispatchTimes.emplace(slot.ordinal, slot.dispatchTimeMs);
            if (slot.ordinal == 0) {
                _firstDispatchTime = slot.dispatchTimeMs;
            }
        }
        for (int64_t ordinal = std::max<int64_t>(0, _timing.lastOrdinal() - 2);
             ordinal <= _timing.lastOrdinal(); ++ordinal) {
            if (_expected.find(ordinal) == _expected.end()) {
                throw std::invalid_argument(
                    "PP4 push is missing a recent Native expectation");
            }
        }
        for (const auto *replies : {&state.observedReplies, &state.replyLowerBounds}) {
            for (const auto &[ordinal, value] : *replies) {
                if (value > _now) {
                    throw std::invalid_argument("PP4 observation is after its capture");
                }
            }
        }
        std::set<int64_t> resultSlots;
        for (const auto &result : _results) {
            if (!validTime(result.controlReadyTimeMs) ||
                (result.slotOrdinal &&
                 (!resultSlots.insert(*result.slotOrdinal).second ||
                  *result.slotOrdinal <
                      (state.slots.empty() ? 0 : state.slots.front().ordinal) ||
                  *result.slotOrdinal > _timing.lastOrdinal()))) {
                throw std::invalid_argument("invalid PP4 outstanding result");
            }
        }
        if (_pending && !_pending->spans.empty())
            resultSlots.insert(_timing.lastOrdinal() + 1);
        std::sort(_otherReleases.begin(), _otherReleases.end());
        for (double when : _otherReleases) {
            if (!validTime(when))
                throw std::invalid_argument("invalid other-request slot release");
        }
        _activeRequests.resize(_requests.size(), false);
        _pendingQueryReadyTimes.resize(_requests.size());
        bool ownedWork = false;
        for (size_t index = 0; index < _requests.size(); ++index) {
            auto &request = _requests[index];
            if (!request.releaseSlotAfterPrefill || !request.requestId ||
                !_requestIndices.emplace(request.requestId, index).second ||
                request.contextTokens < 0 || request.remainingTokens < 0 ||
                request.remainingTokens >
                    std::numeric_limits<int64_t>::max() - request.contextTokens ||
                request.reservedTokens < 0 ||
                request.reservedTokens > request.remainingTokens ||
                request.sharedPrefixBoundary < 0 ||
                request.sharedPrefixBoundary >
                    request.contextTokens + request.remainingTokens ||
                !validTime(request.readyTimeMs)) {
                throw std::invalid_argument("invalid PP4 forecast request");
            }
            if ((request.queryBlockCount && *request.queryBlockCount < 0) ||
                (bool(request.pendingQueryAgeMs) !=
                 bool(request.pendingQueryReadyTimeMs)) ||
                (request.pendingQueryAgeMs &&
                 (!validTime(*request.pendingQueryAgeMs) ||
                  !validTime(*request.pendingQueryReadyTimeMs) ||
                  !request.queryBlockCount || request.needsFirstQuery ||
                  request.queue == TPUPrefillForecastQueue::RUNNING ||
                  request.receiveHostResultIndex)))
                throw std::invalid_argument("invalid PP4 query backlog observation");
            if (_queryServiceParameters && request.needsFirstQuery &&
                !request.queryBlockCount)
                throw std::invalid_argument("missing PP4 first-query size prediction");
            if (_queryServiceParameters && request.queryBlockCount &&
                request.needsFirstQuery && request.firstQueryServiceMs)
                throw std::invalid_argument("conflicting PP4 query service predictions");
            if (_queryServiceParameters && request.pendingQueryAgeMs)
                addActiveQuery(*request.pendingQueryReadyTimeMs,
                               _now - *request.pendingQueryAgeMs,
                               *request.queryBlockCount);
            if (request.firstQueryServiceMs &&
                (!validTime(*request.firstQueryServiceMs) || !request.needsFirstQuery ||
                 request.queue == TPUPrefillForecastQueue::RUNNING ||
                 request.receiveHostResultIndex ||
                 (request.receiveDependency &&
                  !request.receiveDependency->decidedProposalFence)))
                throw std::invalid_argument("invalid unissued PP4 query service");
            request.contextTokens += request.reservedTokens;
            request.remainingTokens -= request.reservedTokens;
            ownedWork |= request.owned && request.remainingTokens > 0;
            if (request.owned) {
                _ownedRequestIds.insert(request.requestId);
            }
            if (request.queue == TPUPrefillForecastQueue::RUNNING) {
                if (request.readyTimeMs > _now || request.receiveDependency) {
                    throw std::invalid_argument(
                        "running PP4 request has pending KV readiness");
                }
                _running.push_back(index);
                _activeRequests[index] = true;
            } else if (request.queue != TPUPrefillForecastQueue::SKIPPED_WAITING &&
                       request.queue != TPUPrefillForecastQueue::WAITING) {
                throw std::invalid_argument("invalid PP4 request queue");
            }
            const auto completion = state.requestCompletionSlots.find(request.requestId);
            if (completion != state.requestCompletionSlots.end() &&
                (request.queue != TPUPrefillForecastQueue::RUNNING ||
                 request.remainingTokens != 0 || !resultSlots.count(completion->second)))
                throw std::invalid_argument(
                    "invalid observed PP4 request completion slot");
            if (completion != state.requestCompletionSlots.end() &&
                *request.releaseSlotAfterPrefill)
                _releasesBySlot[completion->second].push_back(index);
            if (request.receiveHostResultIndex) {
                if (*request.receiveHostResultIndex >= _results.size() + bool(_pending) ||
                    request.queue == TPUPrefillForecastQueue::RUNNING ||
                    request.receiveDependency || request.metadataOnlyLoad ||
                    request.needsFirstQuery)
                    throw std::invalid_argument(
                        "invalid PP4 host receive result prediction");
                _metadataReports[*request.receiveHostResultIndex].push_back(index);
                request.readyTimeMs = NEVER;
            }
            if (request.receiveDependency) {
                const auto &dependency = *request.receiveDependency;
                if ((dependency.reporterOrdinal &&
                     (*dependency.reporterOrdinal <
                          std::max<int64_t>(0, dependency.carrierOrdinal) ||
                      *dependency.reporterOrdinal >=
                          std::numeric_limits<int64_t>::max() - 3 ||
                      (!resultSlots.count(*dependency.reporterOrdinal) &&
                       *dependency.reporterOrdinal <= _timing.lastOrdinal()))) ||
                    (dependency.decidedProposalFence &&
                     (*dependency.decidedProposalFence < _timing.lastOrdinal() ||
                      (dependency.reporterOrdinal &&
                       *dependency.reporterOrdinal <=
                           *dependency.decidedProposalFence)))) {
                    throw std::invalid_argument("invalid PP4 KV receive dependency");
                }
                if (dependency.decidedProposalFence)
                    _pendingQueryReadyTimes[index] = request.readyTimeMs;
                request.readyTimeMs = NEVER;
            }
            if (request.metadataOnlyLoad &&
                (!request.receiveDependency ||
                 !request.receiveDependency->decidedProposalFence ||
                 request.receiveDependency->reporterOrdinal ||
                 !validTime(request.metadataOnlyLoad->localLoadMs) ||
                 !validTime(request.metadataOnlyLoad->controlReplyMs) ||
                 request.metadataOnlyLoad->controlReplyMs == 0.0))
                throw std::invalid_argument("invalid PP4 metadata load prediction");
        }
        for (const auto &[id, ordinal] : state.requestCompletionSlots) {
            if (!_requestIndices.count(id))
                throw std::invalid_argument(
                    "PP4 completion refers to an unknown request");
        }
        if (_running.size() > size_t(_maxRunning) ||
            _otherReleases.size() > size_t(_maxRunning) - _running.size())
            throw std::invalid_argument(
                "observed PP4 running slots exceed Native capacity");
        // Native tries skipped waiters before ordinary waiting requests.
        for (const auto queue : {TPUPrefillForecastQueue::SKIPPED_WAITING,
                                 TPUPrefillForecastQueue::WAITING}) {
            for (size_t index = 0; index < _requests.size(); ++index) {
                if (_requests[index].queue == queue) {
                    _waiting.push_back(index);
                }
            }
        }
        if (!ownedWork) {
            throw std::invalid_argument(
                "PP4 forecast requires owned work not already reserved");
        }
    }

    TPUPrefillPP4ForecastOutput run()
    {
        dispatchPending();
        while (hasWork(true) || !_lastOwned || _timing.lastOrdinal() < *_lastOwned + 2) {
            transition();
            drainReady();
            if (hasWork(false)) {
                ensureCapacity();
                bool queried = false, blocked = false;
                auto spans = schedule(queried, blocked);
                if (!spans.empty()) {
                    append(true, _now, std::move(spans));
                    continue;
                }
                if (queried) {
                    continue;
                }
                if (!blocked) {
                    throw std::runtime_error(
                        "PP4 forecast cannot schedule or wait for KV readiness");
                }
                dispatchMetadataOnly();
                double ready = NEVER;
                for (size_t index = 0; index < _requests.size(); ++index) {
                    const auto &request = _requests[index];
                    if (request.remainingTokens > 0) {
                        if (request.readyTimeMs > _now)
                            ready = std::min(ready, request.readyTimeMs);
                        const auto queryReady = _pendingQueryReadyTimes[index];
                        if (queryReady && *queryReady > _now)
                            ready = std::min(ready, *queryReady);
                    }
                }
                if (_otherReleaseIndex < _otherReleases.size())
                    ready = std::min(ready, _otherReleases[_otherReleaseIndex]);
                // Worker-local completion is not an EngineCore wakeup. The
                // next host result advances the poll that can report it.
                const double done = _results.empty() ? NEVER : headFinish();
                const double push = pushTime();
                if (push < std::min(done, ready)) {
                    append(false, push, {});
                } else if (done <= ready) {
                    if (!validTime(done))
                        throw std::runtime_error(
                            "PP4 capacity has no finite release event");
                    _now = done;
                } else {
                    _now = ready;
                }
            } else {
                drainReady();
                append(false, pushTime(), {});
            }
        }
        // Native consumes results in FIFO order, including calls without tokens.
        // A ready compute reply cannot bypass an unresolved earlier control call.
        while (!_lastOwnedProcessedMs) {
            if (_results.empty())
                throw std::runtime_error("PP4 owned result is absent from its FIFO");
            processHead(headFinish());
        }
        const double end = *_lastOwnedProcessedMs;
        if (!_firstOwned || end < *_firstOwned || !validTime(end)) {
            throw std::runtime_error("invalid PP4 forecast completion");
        }
        _output.queueMs = *_firstOwned - _begin;
        _output.executionMs = end - *_firstOwned;
        _output.totalMs = end - _begin;
        return std::move(_output);
    }

private:
    void dispatchPending()
    {
        if (!_pending) return;
        transition();
        _now = std::max(_now, _pending->dispatchTimeMs);
        if (_pending->spans.empty()) {
            enqueueResult(
                {std::nullopt, false, std::max(_now, _pending->controlReadyTimeMs)});
            return;
        }
        TPUPrefillPipelineSlot slot;
        slot.ordinal = _timing.lastOrdinal() + 1;
        slot.dispatchTimeMs = _pending->dispatchTimeMs;
        slot.features = _pending->features;
        if (slot.ordinal == 0) _firstDispatchTime = slot.dispatchTimeMs;
        _timing.appendSlot(slot);
        _expected.emplace(slot.ordinal, _pending->expectedForwardMs);
        _dispatchTimes.emplace(slot.ordinal, slot.dispatchTimeMs);
        _lastReal = slot.ordinal;
        enqueueResult({slot.ordinal, false, 0.0});
    }

    size_t enqueueResult(const TPUPrefillPP4HostResult &result)
    {
        if (_nextResultId == std::numeric_limits<size_t>::max())
            throw std::length_error("PP4 result sequence exhausted");
        _results.push_back(result);
        return _nextResultId++;
    }

    std::vector<uint64_t> attachMetadataReports(size_t resultId)
    {
        std::vector<uint64_t> ids;
        for (const auto &[index, when] : _metadataLoadReadyTimes) {
            if (when > _now || !_metadataReportReserved.insert(index).second) continue;
            _metadataReports[resultId].push_back(index);
            if (_captureDispatches) ids.push_back(_requests[index].requestId);
        }
        return ids;
    }

    void startWorkerLoad(size_t index)
    {
        auto &request = _requests[index];
        const double loaded = _now + request.metadataOnlyLoad->localLoadMs;
        if (!validTime(loaded))
            throw std::overflow_error("invalid PP4 local load prediction");
        _metadataLoadReadyTimes.emplace(index, loaded);
        request.receiveDependency.reset();
    }

    void dispatchMetadataOnly()
    {
        std::vector<size_t> carriers;
        double controlMs = 0.0;
        for (size_t index = 0; index < _requests.size(); ++index) {
            auto &request = _requests[index];
            if (request.metadataOnlyLoad && request.receiveDependency &&
                request.receiveDependency->decidedProposalFence &&
                *request.receiveDependency->decidedProposalFence <=
                    _timing.lastOrdinal() &&
                !_pendingQueryReadyTimes[index]) {
                carriers.push_back(index);
                controlMs = std::max(controlMs, request.metadataOnlyLoad->controlReplyMs);
            }
        }
        for (const auto &[index, when] : _metadataLoadReadyTimes)
            if (!_metadataReportReserved.count(index))
                controlMs = std::max(controlMs,
                                     _requests[index].metadataOnlyLoad->controlReplyMs);
        if (controlMs == 0.0) return;
        // Native metadata calls occupy its host-result FIFO, without opening a
        // PP wave or advancing compute ordinals. Poll turnaround is an explicit
        // caller prediction; it is never substituted for local load duration.
        if (realResults() >= 4 || _results.size() >= 8)
            throw std::logic_error("metadata call exceeds Native FIFO capacity");
        transition();
        const double completed = _now + controlMs;
        if (!validTime(completed) || completed <= _now)
            throw std::overflow_error("invalid PP4 control reply prediction");
        TPUPrefillPP4ForecastOutput::ControlDispatch trace;
        if (_captureDispatches) {
            trace.timeMs = _now;
            trace.readyTimeMs = completed;
            trace.ppOrdinal = _timing.lastOrdinal();
        }
        for (const auto index : carriers) {
            startWorkerLoad(index);
            if (_captureDispatches) trace.loadRequestIds.push_back(_requests[index].requestId);
        }
        auto finished =
            attachMetadataReports(enqueueResult({std::nullopt, false, completed}));
        if (_captureDispatches) {
            trace.finishedLoadRequestIds = std::move(finished);
            _output.controlDispatches.push_back(std::move(trace));
        }
    }

    void transition()
    {
        if (!_remainingTransitions) {
            throw std::length_error("PP4 forecast transition budget exhausted");
        }
        --_remainingTransitions;
    }

    bool hasWork(bool ownedOnly) const
    {
        return std::any_of(
            _requests.begin(), _requests.end(), [ownedOnly](const auto &request) {
                return request.remainingTokens > 0 && (!ownedOnly || request.owned);
            });
    }

    size_t realResults() const
    {
        return std::count_if(_results.begin(), _results.end(), [](const auto &result) {
            return result.slotOrdinal.has_value();
        });
    }

    double headFinish()
    {
        const auto &head = _results.front();
        if (head.ready) {
            return _now;
        }
        if (!head.slotOrdinal) {
            return std::max(_now, head.controlReadyTimeMs);
        }
        if (*head.slotOrdinal > _timing.lastOrdinal() - 2) {
            return NEVER;
        }
        return std::max(_now, _timing.resultReadyTimeMs(*head.slotOrdinal));
    }

    double expectedMaximum(int64_t first, int64_t end) const
    {
        double maximum = 0.0;
        for (int64_t index = std::max<int64_t>(0, first); index < end; ++index) {
            maximum = std::max(maximum, _expected.at(index));
        }
        return maximum;
    }

    double pushTime()
    {
        if (_lastReal < 0 || _timing.lastOrdinal() + 1 > _lastReal + 2) {
            return NEVER;
        }
        const int64_t next = _timing.lastOrdinal() + 1;
        const int64_t observed = next - 3;
        double handoff;
        if (observed >= 0) {
            handoff = _timing.lastStageReplyTimeMs(observed);
        } else {
            handoff = _firstDispatchTime;
            for (int64_t index = 0; index < next; ++index) {
                handoff = std::max(
                    handoff + expectedMaximum(index - 3, index),
                    std::max(handoff, _dispatchTimes.at(index)) + _expected.at(index));
            }
        }
        const double when =
            std::max(_now, handoff + 0.5 * expectedMaximum(next - 3, next));
        if (!validTime(when)) {
            throw std::overflow_error("invalid PP4 predicted push time");
        }
        return when;
    }

    void processHead(double when)
    {
        transition();
        if (_results.empty() || !validTime(when)) {
            throw std::runtime_error(
                "PP4 FIFO cannot advance without a result or readiness event");
        }
        _now = std::max(_now, when);
        const auto result = _results.front();
        _results.pop_front();
        const auto metadataReports = _metadataReports.find(_resultHeadId++);
        if (metadataReports != _metadataReports.end()) {
            for (const auto index : metadataReports->second) {
                _requests[index].readyTimeMs = _now;
                _requests[index].receiveHostResultIndex.reset();
                _metadataLoadReadyTimes.erase(index);
                _metadataReportReserved.erase(index);
            }
            _metadataReports.erase(metadataReports);
        }
        if (result.slotOrdinal && result.slotOrdinal == _lastOwned)
            _lastOwnedProcessedMs = _now;
        if (result.slotOrdinal) {
            const auto released = _releasesBySlot.find(*result.slotOrdinal);
            if (released != _releasesBySlot.end()) {
                for (const auto index : released->second) _activeRequests[index] = false;
                _running.erase(
                    std::remove_if(_running.begin(), _running.end(),
                                   [&](size_t index) { return !_activeRequests[index]; }),
                    _running.end());
                _releasesBySlot.erase(released);
            }
            for (auto &request : _requests) {
                if (request.receiveDependency &&
                    request.receiveDependency->reporterOrdinal == result.slotOrdinal) {
                    request.readyTimeMs = _now;
                    request.receiveDependency.reset();
                }
            }
        }
    }

    void drainReady()
    {
        if (!_results.empty() && headFinish() <= _now) {
            processHead(_now);
        }
    }

    void ensureCapacity()
    {
        while (realResults() >= 4 || _results.size() >= 8) {
            const double done = headFinish();
            const double push = pushTime();
            if (push < done) {
                append(false, push, {});
            } else {
                processHead(done);
            }
        }
    }

    int64_t scheduleRequest(size_t index, int64_t budget,
                            std::vector<TPUPrefillPP4ForecastSpan> &spans)
    {
        auto &request = _requests[index];
        TPUPrefillRequestInput input;
        input.promptTokens = request.contextTokens + request.remainingTokens;
        input.computedTokens = request.contextTokens;
        input.sharedPrefixBoundary = request.sharedPrefixBoundary;
        int64_t tokens = 0;
        std::string error;
        if (!TPUPrefillExecutionPolicy::clipStreamingPrefillChunk(input, budget, _policy,
                                                                  tokens, &error)) {
            throw std::invalid_argument(error);
        }
        if (tokens) {
            spans.push_back({request.requestId, request.contextTokens, tokens});
            request.contextTokens += tokens;
            request.remainingTokens -= tokens;
        }
        return tokens;
    }

    void issueFirstQuery(size_t index, double &ready, bool &queried)
    {
        auto &request = _requests[index];
        request.needsFirstQuery = false;
        queried = true;
        if (_captureDispatches)
            _output.firstQueryTimeMs.emplace(request.requestId, _now);
        auto service = request.firstQueryServiceMs;
        if (_queryServiceParameters && request.queryBlockCount) {
            pruneActiveQueries();
            const auto elapsed = std::max<long double>(
                0.0L, static_cast<long double>(_activeQueries.size()) * _now -
                          _activeQueryIssuedSum);
            std::array<double, 5> features{
                1.0, std::log1p(static_cast<double>(*request.queryBlockCount)),
                static_cast<double>(_activeQueries.size()),
                std::log1p(static_cast<double>(_activeQueryBlocks)),
                std::log1p(static_cast<double>(elapsed))};
            long double value = 0.0L;
            for (size_t i = 0; i < features.size(); ++i) {
                if (!validTime(features[i]))
                    throw std::overflow_error("invalid PP4 query service feature");
                value += (*_queryServiceParameters)[i] *
                         static_cast<long double>(features[i]);
            }
            service = static_cast<double>(value);
            if (_captureDispatches)
                _output.firstQueryServiceFeatures.emplace(request.requestId, features);
        }
        if (service) {
            if (!validTime(*service))
                throw std::overflow_error("invalid PP4 query service prediction");
            ready = _now + *service;
            if (!validTime(ready))
                throw std::overflow_error("invalid PP4 query completion time");
            if (_queryServiceParameters && request.queryBlockCount)
                addActiveQuery(ready, _now, *request.queryBlockCount);
            request.firstQueryServiceMs.reset();
            if (_captureDispatches)
                _output.firstQueryReadyTimeMs.emplace(request.requestId, ready);
        }
    }

    struct ActiveQuery {
        double readyMs, issuedMs;
        int64_t blocks;
        bool operator>(const ActiveQuery &other) const { return readyMs > other.readyMs; }
    };

    void addActiveQuery(double ready, double issued, int64_t blocks)
    {
        _activeQueries.push({ready, issued, blocks});
        _activeQueryBlocks += blocks;
        _activeQueryIssuedSum += issued;
    }

    void pruneActiveQueries()
    {
        while (!_activeQueries.empty() && _activeQueries.top().readyMs <= _now) {
            const auto done = _activeQueries.top();
            _activeQueryBlocks -= done.blocks;
            _activeQueryIssuedSum -= done.issuedMs;
            _activeQueries.pop();
        }
        if (_activeQueries.empty()) {
            _activeQueryBlocks = 0.0L;
            _activeQueryIssuedSum = 0.0L;
        }
    }

    std::vector<TPUPrefillPP4ForecastSpan> schedule(bool &queried, bool &blocked)
    {
        while (_otherReleaseIndex < _otherReleases.size() &&
               _otherReleases[_otherReleaseIndex] <= _now)
            ++_otherReleaseIndex;
        std::vector<TPUPrefillPP4ForecastSpan> spans;
        int64_t budget = _policy.maxScheduledTokens;
        for (const auto index : _running) {
            const auto &request = _requests[index];
            if (request.remainingTokens <= 0) {
                continue;
            }
            budget -= scheduleRequest(index, budget, spans);
            if (!budget) {
                break;
            }
        }
        std::deque<size_t> deferred;
        while (!_waiting.empty() && budget) {
            if (_running.size() + _otherReleases.size() - _otherReleaseIndex >=
                size_t(_maxRunning)) {
                blocked = true;
                break;
            }
            const auto index = _waiting.front();
            auto &request = _requests[index];
            if (request.remainingTokens <= 0) {
                _waiting.pop_front();
                continue;
            }
            if (request.receiveDependency &&
                request.receiveDependency->decidedProposalFence) {
                auto &queryReady = _pendingQueryReadyTimes[index];
                if (request.needsFirstQuery) {
                    if (!(request.firstQueryServiceMs ||
                          (_queryServiceParameters && request.queryBlockCount)) ||
                        *queryReady <= _now)
                        issueFirstQuery(index, *queryReady, queried);
                } else if (queryReady && *queryReady <= _now) {
                    queryReady.reset();
                }
                blocked = true;
                _waiting.pop_front();
                deferred.push_back(index);
                continue;
            }
            if (request.readyTimeMs > _now || request.needsFirstQuery) {
                if (request.readyTimeMs > _now) {
                    blocked = true;
                } else {
                    issueFirstQuery(index, request.readyTimeMs, queried);
                }
                _waiting.pop_front();
                deferred.push_back(index);
                continue;
            }
            const auto tokens = scheduleRequest(index, budget, spans);
            if (!tokens) {
                break;
            }
            _waiting.pop_front();
            _running.push_back(index);
            _activeRequests[index] = true;
            budget -= tokens;
        }
        deferred.insert(deferred.end(), _waiting.begin(), _waiting.end());
        _waiting = std::move(deferred);
        return spans;
    }

    void append(bool real, double when, std::vector<TPUPrefillPP4ForecastSpan> spans)
    {
        transition();
        if (!validTime(when)) {
            throw std::runtime_error("PP4 forecast has no finite dispatch opportunity");
        }
        if (_timing.lastOrdinal() >= std::numeric_limits<int64_t>::max() - 3) {
            throw std::overflow_error("PP4 slot ordinal exceeds forecast range");
        }
        _now = std::max(_now, when);
        std::vector<uint64_t> loadMetadataRequestIds;
        std::vector<uint64_t> finishedLoadRequestIds;
        TPUPrefillPipelineSlot slot;
        slot.ordinal = _timing.lastOrdinal() + 1;
        slot.dispatchTimeMs = _now;
        if (slot.ordinal == 0) {
            _firstDispatchTime = _now;
        }
        if (real) {
            std::vector<int64_t> counts;
            int64_t scheduled = 0;
            double context = 0.0, product = 0.0, squares = 0.0, maximum = 0.0;
            bool owned = false;
            for (const auto &span : spans) {
                counts.push_back(span.scheduledTokens);
                scheduled += span.scheduledTokens;
                context += span.computedTokens;
                product +=
                    static_cast<double>(span.computedTokens) * span.scheduledTokens;
                squares +=
                    static_cast<double>(span.scheduledTokens) * span.scheduledTokens;
                maximum = std::max(maximum, static_cast<double>(span.computedTokens));
                _output.firstScheduledTimeMs.emplace(span.requestId, _now);
                _output.scheduledTokens[span.requestId] += span.scheduledTokens;
                owned |= _ownedRequestIds.count(span.requestId) != 0;
                const auto index = _requestIndices.at(span.requestId);
                if (_requests[index].remainingTokens == 0 &&
                    *_requests[index].releaseSlotAfterPrefill)
                    _releasesBySlot[slot.ordinal].push_back(index);
            }
            TPUPrefillStepGeometry geometry;
            std::string error;
            if (!TPUPrefillExecutionPolicy::projectStepGeometry(counts, _policy, geometry,
                                                                std::nullopt, &error)) {
                throw std::invalid_argument(error);
            }
            slot.features = {1.0,
                             scheduled / 1024.0,
                             geometry.globalPaddedTokens / 1024.0,
                             product / 1048576.0,
                             context / 1024.0,
                             squares / 1048576.0,
                             static_cast<double>(spans.size()),
                             maximum / 1024.0};
            _lastReal = slot.ordinal;
            const auto resultId = enqueueResult({slot.ordinal, false, 0.0});
            finishedLoadRequestIds = attachMetadataReports(resultId);
            ++_output.realSteps;
            if (owned) {
                ++_output.ownedSteps;
                _lastOwned = slot.ordinal;
                _lastOwnedProcessedMs.reset();
                if (!_firstOwned) {
                    _firstOwned = _now;
                }
            }
            if (!_output.realPrefixSteps && !hasWork(true)) {
                _output.realPrefixSteps = _output.realSteps;
            }
            for (size_t index = 0; index < _requests.size(); ++index) {
                auto &request = _requests[index];
                if (!request.receiveDependency) {
                    continue;
                }
                auto &dependency = *request.receiveDependency;
                if (dependency.decidedProposalFence && !_pendingQueryReadyTimes[index] &&
                    slot.ordinal > *dependency.decidedProposalFence) {
                    dependency.carrierOrdinal = slot.ordinal;
                    dependency.decidedProposalFence.reset();
                    if (_captureDispatches)
                        loadMetadataRequestIds.push_back(request.requestId);
                    if (request.metadataOnlyLoad) {
                        // A compute output carries the same async KV-load
                        // metadata as a control-only output. Worker completion
                        // and a later FIFO notification remain separate events.
                        startWorkerLoad(index);
                        continue;
                    }
                }
                if (dependency.reporterOrdinal == slot.ordinal &&
                    (dependency.decidedProposalFence ||
                     dependency.carrierOrdinal > slot.ordinal)) {
                    throw std::invalid_argument(
                        "PP4 reporter precedes its load metadata");
                }
                if (!dependency.decidedProposalFence && !dependency.reporterOrdinal &&
                    slot.ordinal > dependency.carrierOrdinal) {
                    dependency.reporterOrdinal = slot.ordinal;
                }
            }
        }
        if (!real) {
            for (const auto &request : _requests)
                if (request.receiveDependency &&
                    request.receiveDependency->reporterOrdinal == slot.ordinal)
                    throw std::invalid_argument("PP4 reporter cannot be a pure push");
        }
        _timing.appendSlot(slot);
        _expected.emplace(slot.ordinal, real ? _measured : 0.0);
        _dispatchTimes.emplace(slot.ordinal, _now);
        if (_captureDispatches) {
            _output.dispatches.push_back({slot.ordinal, _now, real, std::move(spans),
                                          std::move(loadMetadataRequestIds),
                                          std::move(finishedLoadRequestIds)});
        }
    }

private:
    TPUPrefillPP4Timing _timing;
    std::map<int64_t, double> _expected;
    std::deque<TPUPrefillPP4HostResult> _results;
    size_t _nextResultId = 0;
    size_t _resultHeadId = 0;
    std::map<size_t, double> _metadataLoadReadyTimes;
    std::set<size_t> _metadataReportReserved;
    std::map<size_t, std::vector<size_t>> _metadataReports;
    std::vector<TPUPrefillPP4Request> _requests;
    const TPUPrefillExecutionPolicyConfig &_policy;
    double _now, _begin, _measured;
    int64_t _lastReal;
    std::optional<TPUPrefillPP4PendingDispatch> _pending;
    int64_t _maxRunning;
    std::vector<double> _otherReleases;
    size_t _otherReleaseIndex = 0;
    std::map<int64_t, std::vector<size_t>> _releasesBySlot;
    std::vector<bool> _activeRequests;
    // Present until the waiting loop visits this request with a ready query.
    std::vector<std::optional<double>> _pendingQueryReadyTimes;
    std::unordered_map<uint64_t, size_t> _requestIndices;
    size_t _remainingTransitions;
    bool _captureDispatches;
    std::optional<std::array<double, 5>> _queryServiceParameters;
    std::priority_queue<ActiveQuery, std::vector<ActiveQuery>,
                        std::greater<ActiveQuery>> _activeQueries;
    long double _activeQueryBlocks = 0.0L, _activeQueryIssuedSum = 0.0L;
    double _firstDispatchTime = 0.0;
    std::map<int64_t, double> _dispatchTimes;
    std::vector<size_t> _running;
    std::deque<size_t> _waiting;
    std::set<uint64_t> _ownedRequestIds;
    std::optional<int64_t> _lastOwned;
    std::optional<double> _lastOwnedProcessedMs;
    std::optional<double> _firstOwned;
    TPUPrefillPP4ForecastOutput _output;
};

} // namespace

TPUPrefillPP4ForecastOutput TPUPrefillPP4Forecast::predict(
    const TPUPrefillPP4State &state,
    const std::vector<TPUPrefillPP4Request> &orderedRequests,
    const TPUPrefillExecutionPolicyConfig &policy,
    const TPUPrefillPP4Timing::Parameters &parameters, size_t maxTransitions,
    bool captureDispatches)
{
    return Forecast(state, orderedRequests, policy, parameters, maxTransitions,
                    captureDispatches)
        .run();
}

} // namespace navi_sched
