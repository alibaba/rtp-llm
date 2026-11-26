#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <vector>
#include <stdexcept>

namespace navi_sched {

struct TPUPrefillPipelineSlot {
    int64_t ordinal = 0;
    double dispatchTimeMs = 0.0;
    // Real-step bias, scheduled/1024, padded/1024, context*scheduled/1024^2,
    // context/1024, scheduled^2/1024^2, request count, max context/1024.
    // A Native push has eight zero features.
    std::array<double, 8> features{};
};

// Pure PP4 host-protocol timing. Reply milestones are host observations, not
// GPU execution clocks. Slot shapes/origins may be observed or forecast by the
// caller; this class neither schedules requests nor estimates KV readiness.
// Each forecast owns one instance; append and cached timing calls are not shared
// between threads. The first parameter is the final sample-reply overhead;
// the other eight multiply one stage's slot features.
class TPUPrefillPP4Timing {
public:
    using Parameters = std::array<double, 9>;
    using ReplyTimes = std::map<int64_t, double>;

    TPUPrefillPP4Timing(Parameters parameters, std::vector<TPUPrefillPipelineSlot> slots,
                        ReplyTimes observedLastStageReplies = {},
                        ReplyTimes lastStageReplyLowerBounds = {});

    void appendSlot(const TPUPrefillPipelineSlot& slot);
    double stepCostMs(int64_t ordinal) const;
    double lastStageReplyTimeMs(int64_t ordinal);
    double resultReadyTimeMs(int64_t ordinal);
    int64_t lastOrdinal() const { return _lastOrdinal; }

    // Shared recurrence for ordinary prediction and continuous-cost AD. Callers
    // provide scalar costs/dispatches; observations retain their original clock.
    template <class Scalar, class DispatchAt, class CostAt, class Maximum, class Validate>
    static Scalar lastStageReplyFromCosts(int64_t ordinal, int64_t first, int64_t last,
                                          const ReplyTimes& observed,
                                          const ReplyTimes& lower,
                                          std::map<int64_t, Scalar>& cache,
                                          DispatchAt dispatchAt, CostAt costAt,
                                          Maximum maximum, Validate validate)
    {
        if (ordinal < first || ordinal > last - 2)
            throw std::out_of_range("PP4 completion requires its next two handoff slots");
        const auto saved = cache.find(ordinal);
        if (saved != cache.end()) return saved->second;
        const auto advanceHandoff = [&](Scalar clock, Scalar arrival, int64_t cycle) {
            Scalar others(0.0);
            for (int offset = 0; offset < 3; ++offset)
                others = maximum(others, costAt(cycle - 3 + offset));
            // Only stage 0 waits for its new dispatch. Other stages have
            // already been executing since the preceding handoff.
            return maximum(clock + others, maximum(clock, arrival) + costAt(cycle));
        };
        const auto original = [&](int64_t target) {
            const auto start = dispatchAt(target);
            Scalar elapsed(0.0);
            for (int offset = 0; offset < 3; ++offset)
                elapsed = advanceHandoff(elapsed, dispatchAt(target + offset) - start,
                                         target + offset);
            return validate(start + elapsed);
        };
        int64_t previous;
        Scalar end;
        auto at = observed.upper_bound(ordinal);
        if (at != observed.begin()) {
            --at;
            previous = at->first;
            end = Scalar(at->second);
        } else {
            const auto bound = lower.upper_bound(ordinal);
            if (bound == lower.begin()) {
                auto result = original(ordinal);
                cache.emplace(ordinal, result);
                return result;
            }
            previous = lower.begin()->first;
            end = maximum(original(previous), Scalar(lower.begin()->second));
        }
        auto prefix = cache.upper_bound(ordinal);
        if (prefix != cache.begin()) {
            --prefix;
            if (prefix->first >= previous) {
                previous = prefix->first;
                end = prefix->second;
            }
        }
        for (int64_t step = previous + 1; step <= ordinal; ++step) {
            const auto cycle = step + 2;
            end = validate(advanceHandoff(end, dispatchAt(cycle), cycle));
            const auto bound = lower.find(step);
            if (bound != lower.end()) end = maximum(end, Scalar(bound->second));
            cache[step] = end;
        }
        const auto bound = lower.find(ordinal);
        if (bound != lower.end()) end = maximum(end, Scalar(bound->second));
        cache[ordinal] = end;
        return end;
    }

private:
    static void validateSlot(const TPUPrefillPipelineSlot& slot);
    const TPUPrefillPipelineSlot& slotAt(int64_t ordinal) const;

private:
    Parameters _parameters;
    std::vector<TPUPrefillPipelineSlot> _slots;
    int64_t _firstOrdinal = 0;
    int64_t _lastOrdinal = -1;
    ReplyTimes _observedReplies;
    ReplyTimes _replyLowerBounds;
    ReplyTimes _replyCache;
};

// Actual dispatches supporting one successful sample reply. This is training
// data, not a prediction input: the following two slots are known at completion.
struct TPUPrefillPP4TimingObservation {
    int64_t targetOrdinal = -1;
    std::vector<TPUPrefillPipelineSlot> slots;

    bool valid() const;
    bool validLatencyMs(double latencyMs) const;
    double predictLatencyMs(const TPUPrefillPP4Timing::Parameters& parameters,
                            TPUPrefillPP4Timing::Parameters* gradient = nullptr) const;
};

} // namespace navi_sched
