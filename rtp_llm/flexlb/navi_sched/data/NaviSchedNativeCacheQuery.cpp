#include "aios/navi_sched/data/NaviSchedNativeCacheQuery.h"
#include <algorithm>

namespace navi_sched {

bool NaviSchedNativeCacheQuery::valid() const
{
    const auto source = static_cast<int32_t>(inputSource);
    const auto future = static_cast<int32_t>(futureState);
    const auto load = static_cast<int32_t>(loadState);
    const auto receive = static_cast<int32_t>(receiveState);
    if (requestId == 0 || source < 0 || source > 3 || future < 0 || future > 3 ||
        load < 0 || load > 3 || receive < 0 || receive > 3)
        return false;
    switch (inputSource) {
        case NativeCacheQueryInputSource::UNOBSERVED:
            return localPrefixTokenCount == -1 &&
                   futureState == NativeCacheQueryFutureState::UNOBSERVED;
        case NativeCacheQueryInputSource::ARRIVAL_PREQUERY:
            return generation > 0 && localPrefixTokenCount == -1 &&
                   futureState != NativeCacheQueryFutureState::UNOBSERVED &&
                   loadState != NativeCacheLoadState::UNOBSERVED;
        case NativeCacheQueryInputSource::ACTIVE_QUERY:
            return generation > 0 && localPrefixTokenCount >= 0 &&
                   futureState != NativeCacheQueryFutureState::UNOBSERVED &&
                   loadState != NativeCacheLoadState::UNOBSERVED;
        case NativeCacheQueryInputSource::RESOLVED_QUERY:
            return generation > 0 && localPrefixTokenCount >= 0 &&
                   futureState == NativeCacheQueryFutureState::UNOBSERVED &&
                   loadState != NativeCacheLoadState::UNOBSERVED;
    }
    return false;
}

bool NaviSchedNativeCacheQuery::operator==(const NaviSchedNativeCacheQuery &other) const
{
    return requestId == other.requestId && generation == other.generation &&
           localPrefixTokenCount == other.localPrefixTokenCount &&
           inputSource == other.inputSource && futureState == other.futureState &&
           loadState == other.loadState && receiveState == other.receiveState;
}

std::optional<int64_t> NaviSchedNativeCacheQuery::expectedExternalTokenCount(
    int64_t predictedTotalHitTokens) const
{
    if (predictedTotalHitTokens < 0 || localPrefixTokenCount < 0 ||
        futureState == NativeCacheQueryFutureState::FAILED) return std::nullopt;
    return std::max<int64_t>(0, predictedTotalHitTokens - localPrefixTokenCount);
}

void NaviSchedNativeCacheQuery::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(requestId);
    buffer.write(generation);
    buffer.write(localPrefixTokenCount);
    buffer.write(static_cast<int32_t>(inputSource));
    buffer.write(static_cast<int32_t>(futureState));
    buffer.write(static_cast<int32_t>(loadState));
    buffer.write(static_cast<int32_t>(receiveState));
}

void NaviSchedNativeCacheQuery::deserialize(autil::DataBuffer &buffer)
{
    int32_t source, future, load, receive;
    buffer.read(requestId);
    buffer.read(generation);
    buffer.read(localPrefixTokenCount);
    buffer.read(source);
    buffer.read(future);
    buffer.read(load);
    buffer.read(receive);
    inputSource = static_cast<NativeCacheQueryInputSource>(source);
    futureState = static_cast<NativeCacheQueryFutureState>(future);
    loadState = static_cast<NativeCacheLoadState>(load);
    receiveState = static_cast<NativeCacheReceiveState>(receive);
}

} // namespace navi_sched
