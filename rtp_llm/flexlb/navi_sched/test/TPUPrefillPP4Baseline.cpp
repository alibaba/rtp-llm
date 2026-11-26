// Frozen test-only reference for paired prediction-overhead measurements.
#include "aios/navi_sched/test/TPUPrefillPP4Baseline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

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
        , _requests(requests)
        , _policy(policy)
        , _now(state.nowMs)
        , _begin(state.nowMs)
        , _measured(state.measuredForwardMs)
        , _lastReal(state.lastRealOrdinal)
        , _pending(state.pendingDispatch)
        , _remainingTransitions(maxTransitions)
        , _captureDispatches(captureDispatches)
    {
        std::string error;
        if (!TPUPrefillExecutionPolicy::validate(policy, &error) ||
            policy.fixedPrefillChunkPlan || !validTime(_now) || !validTime(_measured) ||
            !maxTransitions || state.slots.size() > 8 || _results.size() > 8 ||
            realResults() > 4 || _lastReal < -1 || _lastReal > _timing.lastOrdinal() ||
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
        bool ownedWork = false;
        std::set<uint64_t> requestIds;
        for (size_t index = 0; index < _requests.size(); ++index) {
            auto &request = _requests[index];
            if (!request.requestId || !requestIds.insert(request.requestId).second ||
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
            } else if (request.queue != TPUPrefillForecastQueue::SKIPPED_WAITING &&
                       request.queue != TPUPrefillForecastQueue::WAITING) {
                throw std::invalid_argument("invalid PP4 request queue");
            }
            if (request.receiveDependency) {
                const auto &dependency = *request.receiveDependency;
                if ((dependency.reporterOrdinal &&
                     !resultSlots.count(*dependency.reporterOrdinal)) ||
                    (dependency.decidedProposalFence &&
                     (*dependency.decidedProposalFence < _timing.lastOrdinal() ||
                      dependency.reporterOrdinal))) {
                    throw std::invalid_argument("invalid PP4 KV receive dependency");
                }
                request.readyTimeMs = NEVER;
            }
        }
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
                drainReady();
                double ready = NEVER;
                for (const auto &request : _requests) {
                    if (request.remainingTokens > 0 && request.readyTimeMs > _now) {
                        ready = std::min(ready, request.readyTimeMs);
                    }
                }
                const double done = _results.empty() ? NEVER : headFinish();
                const double push = pushTime();
                if (push < std::min(done, ready)) {
                    append(false, push, {});
                } else if (done <= ready) {
                    processHead(done);
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
            _results.push_back(
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
        _results.push_back({slot.ordinal, false, 0.0});
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
        if (result.slotOrdinal && result.slotOrdinal == _lastOwned)
            _lastOwnedProcessedMs = _now;
        if (result.slotOrdinal) {
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
        while (!_results.empty() && headFinish() <= _now) {
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

    std::vector<TPUPrefillPP4ForecastSpan> schedule(bool &queried, bool &blocked)
    {
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
            const auto index = _waiting.front();
            auto &request = _requests[index];
            if (request.remainingTokens <= 0) {
                _waiting.pop_front();
                continue;
            }
            if (request.readyTimeMs > _now || request.needsFirstQuery) {
                if (request.readyTimeMs > _now) {
                    blocked = true;
                } else {
                    request.needsFirstQuery = false;
                    queried = true;
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
            _results.push_back({slot.ordinal, false, 0.0});
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
            for (auto &request : _requests) {
                if (!request.receiveDependency) {
                    continue;
                }
                auto &dependency = *request.receiveDependency;
                if (dependency.decidedProposalFence &&
                    slot.ordinal > *dependency.decidedProposalFence) {
                    dependency.carrierOrdinal = slot.ordinal;
                    dependency.decidedProposalFence.reset();
                }
                if (!dependency.decidedProposalFence && !dependency.reporterOrdinal &&
                    slot.ordinal > dependency.carrierOrdinal) {
                    dependency.reporterOrdinal = slot.ordinal;
                }
            }
        }
        _timing.appendSlot(slot);
        _expected.emplace(slot.ordinal, real ? _measured : 0.0);
        _dispatchTimes.emplace(slot.ordinal, _now);
        if (_captureDispatches) {
            _output.dispatches.push_back({slot.ordinal, _now, real, std::move(spans)});
        }
    }

private:
    TPUPrefillPP4Timing _timing;
    std::map<int64_t, double> _expected;
    std::deque<TPUPrefillPP4HostResult> _results;
    std::vector<TPUPrefillPP4Request> _requests;
    const TPUPrefillExecutionPolicyConfig &_policy;
    double _now, _begin, _measured;
    int64_t _lastReal;
    std::optional<TPUPrefillPP4PendingDispatch> _pending;
    size_t _remainingTransitions;
    bool _captureDispatches;
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

TPUPrefillPP4ForecastOutput TPUPrefillPP4ForecastBaseline::predict(
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
