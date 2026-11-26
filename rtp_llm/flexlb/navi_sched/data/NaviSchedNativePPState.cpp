#include "aios/navi_sched/data/NaviSchedNativePPState.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <autil/DataBuffer.h>

namespace navi_sched {
namespace {

bool nonnegative(double value) { return std::isfinite(value) && value >= 0.0; }

bool validLoadRequestIds(const std::vector<uint64_t> &ids)
{
    for (size_t index = 0; index < ids.size(); ++index)
        if (ids[index] == 0 || (index && ids[index] <= ids[index - 1])) return false;
    return true;
}

bool sameFeature(double left, double right)
{
    return std::abs(left - right) <= 8 * std::numeric_limits<double>::epsilon() *
                                         std::max({1.0, std::abs(left), std::abs(right)});
}

bool validShape(const std::array<double, 8> &features,
                const std::vector<NaviSchedNativePPSpan> &spans)
{
    if (!std::all_of(features.begin(), features.end(), nonnegative)) return false;
    if (spans.empty()) {
        return std::all_of(features.begin(), features.end(),
                           [](double v) { return v == 0.0; });
    }
    int64_t scheduled = 0;
    double context = 0.0, product = 0.0, squares = 0.0, maximum = 0.0;
    std::vector<uint64_t> ids;
    ids.reserve(spans.size());
    for (const auto &span : spans) {
        if (!span.valid() ||
            span.scheduledTokens > std::numeric_limits<int64_t>::max() - scheduled)
            return false;
        scheduled += span.scheduledTokens;
        context += span.computedTokens;
        product += static_cast<double>(span.computedTokens) * span.scheduledTokens;
        squares += static_cast<double>(span.scheduledTokens) * span.scheduledTokens;
        maximum = std::max(maximum, static_cast<double>(span.computedTokens));
        ids.push_back(span.requestId);
    }
    std::sort(ids.begin(), ids.end());
    return std::adjacent_find(ids.begin(), ids.end()) == ids.end() &&
           features[0] == 1.0 && sameFeature(features[1], scheduled / 1024.0) &&
           features[2] >= features[1] && sameFeature(features[3], product / 1048576.0) &&
           sameFeature(features[4], context / 1024.0) &&
           sameFeature(features[5], squares / 1048576.0) &&
           features[6] == static_cast<double>(spans.size()) &&
           sameFeature(features[7], maximum / 1024.0);
}

void writeShape(autil::DataBuffer &buffer, const std::array<double, 8> &features)
{
    for (double value : features) buffer.write(value);
}

void readShape(autil::DataBuffer &buffer, std::array<double, 8> &features)
{
    for (double &value : features) buffer.read(value);
}

} // namespace

bool NaviSchedNativePPSpan::valid() const
{
    return requestId > 0 && computedTokens >= 0 && scheduledTokens > 0 &&
           scheduledTokens <= std::numeric_limits<int64_t>::max() - computedTokens;
}

bool NaviSchedNativePPSpan::operator==(const NaviSchedNativePPSpan &other) const
{
    return requestId == other.requestId && computedTokens == other.computedTokens &&
           scheduledTokens == other.scheduledTokens;
}

void NaviSchedNativePPSpan::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(requestId);
    buffer.write(computedTokens);
    buffer.write(scheduledTokens);
}

void NaviSchedNativePPSpan::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(requestId);
    buffer.read(computedTokens);
    buffer.read(scheduledTokens);
}

bool NaviSchedNativePPSlot::valid() const
{
    return ordinal >= 0 && ordinal < std::numeric_limits<int64_t>::max() - 3 &&
           dispatchAgeUs >= 0 && lastStageReplyAgeUs >= -1 &&
           lastStageReplyAgeUs <= dispatchAgeUs && nonnegative(expectedForwardMs) &&
           (!spans.empty() || expectedForwardMs == 0.0) && validShape(features, spans);
}

bool NaviSchedNativePPSlot::operator==(const NaviSchedNativePPSlot &other) const
{
    return ordinal == other.ordinal && dispatchAgeUs == other.dispatchAgeUs &&
           expectedForwardMs == other.expectedForwardMs &&
           lastStageReplyAgeUs == other.lastStageReplyAgeUs &&
           features == other.features && spans == other.spans;
}

void NaviSchedNativePPSlot::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(ordinal);
    buffer.write(dispatchAgeUs);
    buffer.write(expectedForwardMs);
    buffer.write(lastStageReplyAgeUs);
    writeShape(buffer, features);
    buffer.write(spans);
}

void NaviSchedNativePPSlot::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(ordinal);
    buffer.read(dispatchAgeUs);
    buffer.read(expectedForwardMs);
    buffer.read(lastStageReplyAgeUs);
    readShape(buffer, features);
    buffer.read(spans);
}

bool NaviSchedNativePPResult::valid() const
{
    return schedulerOutputId > 0 && dispatchAgeUs >= 0 && slotOrdinal >= -1 &&
           validLoadRequestIds(loadRequestIds);
}

bool NaviSchedNativePPResult::operator==(const NaviSchedNativePPResult &other) const
{
    return schedulerOutputId == other.schedulerOutputId &&
           dispatchAgeUs == other.dispatchAgeUs && slotOrdinal == other.slotOrdinal &&
           executeReplyReady == other.executeReplyReady &&
           resultReady == other.resultReady && loadRequestIds == other.loadRequestIds;
}

void NaviSchedNativePPResult::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(schedulerOutputId);
    buffer.write(dispatchAgeUs);
    buffer.write(slotOrdinal);
    buffer.write(executeReplyReady);
    buffer.write(resultReady);
    buffer.write(loadRequestIds);
}

void NaviSchedNativePPResult::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(schedulerOutputId);
    buffer.read(dispatchAgeUs);
    buffer.read(slotOrdinal);
    buffer.read(executeReplyReady);
    buffer.read(resultReady);
    buffer.read(loadRequestIds);
}

bool NaviSchedNativePPRequest::valid() const
{
    const auto kind = int32_t(queue), progress = int32_t(phase);
    const bool indexed = queue != NativePPQueue::NOT_QUEUED;
    if (!validNativePrefillSlotRelease(prefillSlotRelease) ||
        (queue != NativePPQueue::PREPARED &&
         prefillSlotRelease == NativePPPrefillSlotRelease::UNKNOWN))
        return false;
    return requestId > 0 && kind >= 0 && kind <= 4 && progress >= 0 && progress <= 11 &&
           (indexed ? queueIndex >= 0 : queueIndex == -1) && promptTokens > 0 &&
           computedTokens >= 0 && cacheHitTokens >= -1 &&
           cacheHitTokens <= promptTokens && sharedPrefixBoundary >= 0 &&
           sharedPrefixBoundary <= promptTokens && queryAgeUs >= -1 && loadAgeUs >= -1;
}

bool NaviSchedNativePPRequest::operator==(const NaviSchedNativePPRequest &other) const
{
    return requestId == other.requestId && queue == other.queue &&
           queueIndex == other.queueIndex && phase == other.phase &&
           promptTokens == other.promptTokens && computedTokens == other.computedTokens &&
           cacheHitTokens == other.cacheHitTokens &&
           sharedPrefixBoundary == other.sharedPrefixBoundary &&
           queryAgeUs == other.queryAgeUs && loadAgeUs == other.loadAgeUs &&
           prefillSlotRelease == other.prefillSlotRelease;
}

void NaviSchedNativePPRequest::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(requestId);
    buffer.write(int32_t(queue));
    buffer.write(queueIndex);
    buffer.write(int32_t(phase));
    buffer.write(promptTokens);
    buffer.write(computedTokens);
    buffer.write(cacheHitTokens);
    buffer.write(sharedPrefixBoundary);
    buffer.write(queryAgeUs);
    buffer.write(loadAgeUs);
    buffer.write(int32_t(prefillSlotRelease));
}

void NaviSchedNativePPRequest::deserialize(autil::DataBuffer &buffer)
{
    int32_t kind, progress;
    buffer.read(requestId);
    buffer.read(kind);
    buffer.read(queueIndex);
    buffer.read(progress);
    queue = static_cast<NativePPQueue>(kind);
    phase = static_cast<NativePPRequestPhase>(progress);
    buffer.read(promptTokens);
    buffer.read(computedTokens);
    buffer.read(cacheHitTokens);
    buffer.read(sharedPrefixBoundary);
    buffer.read(queryAgeUs);
    buffer.read(loadAgeUs);
    int32_t release;
    buffer.read(release);
    prefillSlotRelease = static_cast<NativePPPrefillSlotRelease>(release);
}

bool NaviSchedNativePPDecision::valid() const
{
    return schedulerOutputId > 0 && validShape(features, spans) &&
           validLoadRequestIds(loadRequestIds);
}

bool NaviSchedNativePPDecision::operator==(const NaviSchedNativePPDecision &other) const
{
    return schedulerOutputId == other.schedulerOutputId && features == other.features &&
           spans == other.spans && loadRequestIds == other.loadRequestIds;
}

void NaviSchedNativePPDecision::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(schedulerOutputId);
    writeShape(buffer, features);
    buffer.write(spans);
    buffer.write(loadRequestIds);
}

void NaviSchedNativePPDecision::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(schedulerOutputId);
    readShape(buffer, features);
    buffer.read(spans);
    buffer.read(loadRequestIds);
}

bool NaviSchedNativePPState::valid() const
{
    if (!observationId || occupiedRequestSlotCount < 0 ||
        !nonnegative(measuredForwardMs) || slots.size() > 8 || results.size() > 8)
        return false;
    for (size_t index = 0; index < slots.size(); ++index) {
        const auto &slot = slots[index];
        if (!slot.valid() ||
            (index && (slot.ordinal != slots[index - 1].ordinal + 1 ||
                       slot.dispatchAgeUs > slots[index - 1].dispatchAgeUs)))
            return false;
    }
    if (waveOpen ? (!burst || slots.empty() || lastRealOrdinal < slots.front().ordinal ||
                    lastRealOrdinal > slots.back().ordinal ||
                    slots.back().ordinal > lastRealOrdinal + 2)
                 : (!slots.empty() || lastRealOrdinal != -1))
        return false;
    if (waveOpen &&
        slots[static_cast<size_t>(lastRealOrdinal - slots.front().ordinal)].spans.empty())
        return false;
    std::vector<uint64_t> outputIds;
    outputIds.reserve(results.size() + bool(pendingDecision));
    size_t realResults = 0;
    int64_t previousOrdinal = -1;
    for (const auto &result : results) {
        if (!result.valid()) return false;
        outputIds.push_back(result.schedulerOutputId);
        if (result.slotOrdinal >= 0) {
            if (!waveOpen || result.slotOrdinal <= previousOrdinal ||
                result.slotOrdinal < slots.front().ordinal ||
                result.slotOrdinal > slots.back().ordinal ||
                slots[static_cast<size_t>(result.slotOrdinal - slots.front().ordinal)]
                    .spans.empty())
                return false;
            previousOrdinal = result.slotOrdinal;
            ++realResults;
        }
    }
    if (realResults > 4) return false;
    if (pendingDecision) {
        if (!pendingDecision->valid() || results.size() == 8 ||
            (!pendingDecision->spans.empty() && realResults == 4))
            return false;
        outputIds.push_back(pendingDecision->schedulerOutputId);
    }
    std::sort(outputIds.begin(), outputIds.end());
    if (std::adjacent_find(outputIds.begin(), outputIds.end()) != outputIds.end())
        return false;
    std::vector<uint64_t> requestIds;
    requestIds.reserve(requests.size());
    int64_t runningRequests = 0;
    std::array<int64_t, 5> lastQueueIndex{-1, -1, -1, -1, -1};
    for (const auto &request : requests) {
        if (!request.valid()) return false;
        runningRequests += request.queue == NativePPQueue::RUNNING;
        if (request.queue != NativePPQueue::NOT_QUEUED) {
            auto &last = lastQueueIndex[static_cast<size_t>(request.queue)];
            if (request.queueIndex <= last) return false;
            last = request.queueIndex;
        }
        requestIds.push_back(request.requestId);
    }
    std::sort(requestIds.begin(), requestIds.end());
    return runningRequests <= occupiedRequestSlotCount &&
           std::adjacent_find(requestIds.begin(), requestIds.end()) == requestIds.end();
}

bool NaviSchedNativePPState::operator==(const NaviSchedNativePPState &other) const
{
    return observationId == other.observationId && burst == other.burst &&
           waveOpen == other.waveOpen && lastRealOrdinal == other.lastRealOrdinal &&
           measuredForwardMs == other.measuredForwardMs &&
           occupiedRequestSlotCount == other.occupiedRequestSlotCount &&
           slots == other.slots && results == other.results &&
           requests == other.requests && pendingDecision == other.pendingDecision;
}

void NaviSchedNativePPState::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(observationId);
    buffer.write(burst);
    buffer.write(waveOpen);
    buffer.write(lastRealOrdinal);
    buffer.write(measuredForwardMs);
    buffer.write(occupiedRequestSlotCount);
    buffer.write(slots);
    buffer.write(results);
    buffer.write(requests);
    buffer.write(bool(pendingDecision));
    if (pendingDecision) pendingDecision->serialize(buffer);
}

void NaviSchedNativePPState::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(observationId);
    buffer.read(burst);
    buffer.read(waveOpen);
    buffer.read(lastRealOrdinal);
    buffer.read(measuredForwardMs);
    buffer.read(occupiedRequestSlotCount);
    buffer.read(slots);
    buffer.read(results);
    buffer.read(requests);
    bool hasDecision;
    buffer.read(hasDecision);
    pendingDecision.reset();
    if (hasDecision) {
        pendingDecision.emplace();
        pendingDecision->deserialize(buffer);
    }
}

} // namespace navi_sched
