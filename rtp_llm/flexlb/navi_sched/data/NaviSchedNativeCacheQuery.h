#pragma once

#include "autil/DataBuffer.h"
#include <cstdint>
#include <optional>

namespace navi_sched {

enum class NativeCacheQueryInputSource : int32_t {
    UNOBSERVED = 0,
    ACTIVE_QUERY = 1,
    ARRIVAL_PREQUERY = 2,
    RESOLVED_QUERY = 3,
};

enum class NativeCacheQueryFutureState : int32_t {
    UNOBSERVED = 0,
    PENDING = 1,
    READY = 2,
    // Completed with an exception, cancellation or failed object fetch. This is
    // not an observed cache miss and does not imply a load has completed.
    FAILED = 3,
};

enum class NativeCacheLoadState : int32_t {
    UNOBSERVED = 0,
    IDLE = 1,
    IN_FLIGHT = 2,
    FAILED = 3,
};

enum class NativeCacheReceiveState : int32_t {
    UNOBSERVED = 0,
    NOT_REPORTED = 1,
    RECEIVED = 2,
    FAILED = 3,
};

// PP scheduler-thread observations. An unobserved prefix remains -1; Native's
// planned token cursor and total predicted cache hit are not query arguments.
struct NaviSchedNativeCacheQuery {
    bool valid() const;
    bool operator==(const NaviSchedNativeCacheQuery &other) const;
    void serialize(autil::DataBuffer &buffer) const;
    void deserialize(autil::DataBuffer &buffer);
    std::optional<int64_t> expectedExternalTokenCount(
        int64_t predictedTotalHitTokens) const;

    uint64_t requestId = 0;
    uint64_t generation = 0;
    int64_t localPrefixTokenCount = -1;
    NativeCacheQueryInputSource inputSource = NativeCacheQueryInputSource::UNOBSERVED;
    NativeCacheQueryFutureState futureState = NativeCacheQueryFutureState::UNOBSERVED;
    NativeCacheLoadState loadState = NativeCacheLoadState::UNOBSERVED;
    NativeCacheReceiveState receiveState = NativeCacheReceiveState::UNOBSERVED;
};

} // namespace navi_sched
