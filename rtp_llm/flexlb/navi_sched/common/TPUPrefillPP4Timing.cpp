#include "aios/navi_sched/common/TPUPrefillPP4Timing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace navi_sched {
namespace {

double finiteTime(double value)
{
    if (!std::isfinite(value) || value < 0.0) {
        throw std::overflow_error("invalid PP4 forecast time");
    }
    return value;
}

} // namespace

TPUPrefillPP4Timing::TPUPrefillPP4Timing(Parameters parameters,
                                         std::vector<TPUPrefillPipelineSlot> slots,
                                         ReplyTimes observedLastStageReplies,
                                         ReplyTimes lastStageReplyLowerBounds)
    : _parameters(std::move(parameters))
    , _observedReplies(std::move(observedLastStageReplies))
    , _replyLowerBounds(std::move(lastStageReplyLowerBounds))
{
    for (const double parameter : _parameters) {
        if (!std::isfinite(parameter) || parameter < 0.0) {
            throw std::invalid_argument("invalid PP4 timing parameter");
        }
    }
    if (!slots.empty()) {
        validateSlot(slots.front());
        _firstOrdinal = slots.front().ordinal;
        _lastOrdinal = _firstOrdinal - 1;
        _slots.reserve(slots.size());
        for (const auto& slot : slots) {
            appendSlot(slot);
        }
    }
    for (const auto* replies : {&_observedReplies, &_replyLowerBounds}) {
        for (const auto& [ordinal, timeMs] : *replies) {
            if (ordinal < _firstOrdinal || ordinal > _lastOrdinal ||
                !std::isfinite(timeMs) || timeMs < 0.0) {
                throw std::invalid_argument("invalid PP4 reply observation");
            }
        }
    }
    for (const auto& [ordinal, timeMs] : _observedReplies) {
        const auto bound = _replyLowerBounds.find(ordinal);
        if (bound != _replyLowerBounds.end() && bound->second > timeMs) {
            throw std::invalid_argument("contradictory PP4 reply observation");
        }
    }
}

void TPUPrefillPP4Timing::validateSlot(const TPUPrefillPipelineSlot& slot)
{
    if (slot.ordinal < 0 || !std::isfinite(slot.dispatchTimeMs) ||
        slot.dispatchTimeMs < 0.0) {
        throw std::invalid_argument("invalid PP4 slot origin");
    }
    for (const double feature : slot.features) {
        if (!std::isfinite(feature) || feature < 0.0) {
            throw std::invalid_argument("invalid PP4 slot feature");
        }
    }
}

void TPUPrefillPP4Timing::appendSlot(const TPUPrefillPipelineSlot& slot)
{
    validateSlot(slot);
    if (_lastOrdinal == std::numeric_limits<int64_t>::max() ||
        slot.ordinal != _lastOrdinal + 1 ||
        (!_slots.empty() && slot.dispatchTimeMs < _slots.back().dispatchTimeMs)) {
        throw std::invalid_argument(
            "PP4 slots must be contiguous and ordered by dispatch time");
    }
    _slots.push_back(slot);
    _lastOrdinal = slot.ordinal;
}

const TPUPrefillPipelineSlot& TPUPrefillPP4Timing::slotAt(int64_t ordinal) const
{
    if (ordinal < _firstOrdinal || ordinal > _lastOrdinal) {
        throw std::out_of_range("missing PP4 slot history");
    }
    return _slots[static_cast<size_t>(ordinal - _firstOrdinal)];
}

double TPUPrefillPP4Timing::stepCostMs(int64_t ordinal) const
{
    // Negative ordinals are the empty prologue before this burst's slot zero.
    if (ordinal < 0) {
        return 0.0;
    }
    const auto& features = slotAt(ordinal).features;
    long double sum = 0.0L;
    for (size_t index = 0; index < features.size(); ++index) {
        const double product = features[index] * _parameters[index + 1];
        sum += product;
    }
    return finiteTime(static_cast<double>(sum));
}

double TPUPrefillPP4Timing::lastStageReplyTimeMs(int64_t ordinal)
{
    return lastStageReplyFromCosts<double>(
        ordinal, _firstOrdinal, _lastOrdinal, _observedReplies, _replyLowerBounds,
        _replyCache, [this](int64_t at) { return slotAt(at).dispatchTimeMs; },
        [this](int64_t at) { return stepCostMs(at); },
        [](double left, double right) { return std::max(left, right); },
        [](double value) { return finiteTime(value); });
}

double TPUPrefillPP4Timing::resultReadyTimeMs(int64_t ordinal)
{
    return finiteTime(lastStageReplyTimeMs(ordinal) + stepCostMs(ordinal) +
                      _parameters[0]);
}

bool TPUPrefillPP4TimingObservation::valid() const
{
    if (targetOrdinal < 0 || targetOrdinal > std::numeric_limits<int64_t>::max() - 2 ||
        slots.empty() || slots.size() > 6 ||
        slots.front().ordinal != std::max<int64_t>(0, targetOrdinal - 3) ||
        slots.back().ordinal != targetOrdinal + 2)
        return false;
    double previous = 0.0;
    for (size_t index = 0; index < slots.size(); ++index) {
        const auto& slot = slots[index];
        if (slot.ordinal != slots.front().ordinal + static_cast<int64_t>(index) ||
            !std::isfinite(slot.dispatchTimeMs) || slot.dispatchTimeMs < previous ||
            (slot.features[0] != 0.0 && slot.features[0] != 1.0))
            return false;
        for (double feature : slot.features)
            if (!std::isfinite(feature) || feature < 0.0 ||
                (slot.features[0] == 0.0 && feature != 0.0))
                return false;
        previous = slot.dispatchTimeMs;
    }
    return slots[static_cast<size_t>(targetOrdinal - slots.front().ordinal)]
               .features[0] == 1.0;
}

bool TPUPrefillPP4TimingObservation::validLatencyMs(double latencyMs) const
{
    if (!valid() || !std::isfinite(latencyMs) || latencyMs <= 0.0) return false;
    const auto own = static_cast<size_t>(targetOrdinal - slots.front().ordinal);
    return latencyMs >= slots.back().dispatchTimeMs - slots[own].dispatchTimeMs;
}

double TPUPrefillPP4TimingObservation::predictLatencyMs(
    const TPUPrefillPP4Timing::Parameters& parameters,
    TPUPrefillPP4Timing::Parameters* gradient) const
{
    if (!valid()) throw std::invalid_argument("invalid PP4 timing training support");
    TPUPrefillPP4Timing timing(parameters, slots);
    const auto at = [&](int64_t ordinal) -> const TPUPrefillPipelineSlot& {
        return slots[static_cast<size_t>(ordinal - slots.front().ordinal)];
    };
    const double start = at(targetOrdinal).dispatchTimeMs;
    if (gradient) {
        gradient->fill(0.0);
        double elapsed = 0.0;
        for (int cycleOffset = 0; cycleOffset < 3; ++cycleOffset) {
            const int64_t cycle = targetOrdinal + cycleOffset;
            const double arrival = at(cycle).dispatchTimeMs - start;
            double others = 0.0;
            int64_t selected = -1;
            for (int slotOffset = 0; slotOffset < 3; ++slotOffset) {
                const int64_t ordinal = cycle - 3 + slotOffset;
                const double cost = timing.stepCostMs(ordinal);
                if (cost > others) {
                    others = cost;
                    selected = ordinal;
                }
            }
            const double otherEnd = elapsed + others;
            const double ownEnd = std::max(elapsed, arrival) + timing.stepCostMs(cycle);
            if (ownEnd > otherEnd) {
                if (arrival > elapsed) gradient->fill(0.0);
                for (size_t index = 0; index < 8; ++index)
                    (*gradient)[index + 1] += at(cycle).features[index];
                elapsed = ownEnd;
            } else {
                if (selected >= 0)
                    for (size_t index = 0; index < 8; ++index)
                        (*gradient)[index + 1] += at(selected).features[index];
                elapsed = otherEnd;
            }
        }
        (*gradient)[0] = 1.0;
        for (size_t index = 0; index < 8; ++index)
            (*gradient)[index + 1] += at(targetOrdinal).features[index];
    }
    return timing.resultReadyTimeMs(targetOrdinal) - start;
}

} // namespace navi_sched
