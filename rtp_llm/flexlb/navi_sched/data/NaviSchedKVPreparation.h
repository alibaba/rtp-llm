#pragma once
#include "autil/DataBuffer.h"
#include <cstdint>
#include <cmath>

namespace navi_sched {

enum class NativeKVExecutionState : int32_t {
    UNKNOWN = -1,
    PREPARING = 0,
    READY = 1,
    EXECUTING = 2,
    FAILED = 3,
};

// Per-request facts at the SAME boundary as the aggregate workload snapshot.
// Context is the current computed prefix, not necessarily a cache hit prefix.
struct NaviSchedKVPreparation {
    uint64_t requestId = 0;
    int64_t localBatchId = 0;
    uint64_t loadGeneration = 0;
    int64_t contextTokens = 0;
    int64_t computeTokens = 0;
    bool runningCapacity = false;
    NativeKVExecutionState state = NativeKVExecutionState::UNKNOWN;
    // Agent-local elapsed preparation time at report construction, or -1 when
    // unobserved. READY/EXECUTING use the fixed completion duration. This is not
    // a cross-host monotonic timestamp. PREPARING age ticks do not change identity;
    // the fixed terminal duration does.
    double preparationElapsedMs = -1.0;

    bool valid() const {
        const auto value = static_cast<int32_t>(state);
        return requestId != 0 && localBatchId >= 0 && contextTokens >= 0 &&
               computeTokens >= 0 && value >= 0 && value <= 3 &&
               (preparationElapsedMs == -1.0 ||
                (loadGeneration > 0 && std::isfinite(preparationElapsedMs) && preparationElapsedMs >= 0.0));
    }
    bool operator==(const NaviSchedKVPreparation &other) const {
        return requestId == other.requestId && localBatchId == other.localBatchId &&
               loadGeneration == other.loadGeneration && contextTokens == other.contextTokens &&
               computeTokens == other.computeTokens && runningCapacity == other.runningCapacity &&
               state == other.state &&
               // PREPARING age is extrapolated at Master and cannot trigger a
               // report every clock tick. Terminal duration is a fixed fact.
               (state == NativeKVExecutionState::PREPARING ||
                preparationElapsedMs == other.preparationElapsedMs);
    }
    bool executable() const {
        return state == NativeKVExecutionState::READY ||
               state == NativeKVExecutionState::EXECUTING;
    }
    void serialize(autil::DataBuffer &buffer) const {
        buffer.write(requestId);
        buffer.write(localBatchId);
        buffer.write(loadGeneration);
        buffer.write(contextTokens);
        buffer.write(computeTokens);
        buffer.write(runningCapacity);
        buffer.write(static_cast<int32_t>(state));
        buffer.write(preparationElapsedMs);
    }
    void deserialize(autil::DataBuffer &buffer) {
        int32_t value;
        buffer.read(requestId);
        buffer.read(localBatchId);
        buffer.read(loadGeneration);
        buffer.read(contextTokens);
        buffer.read(computeTokens);
        buffer.read(runningCapacity);
        buffer.read(value);
        state = static_cast<NativeKVExecutionState>(value);
        buffer.read(preparationElapsedMs);
    }
};
}
