#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace navi_sched {

// Explicit offline conversion: state10/model29or33 ->34; state11/model34 ->35.
// Current35 may only rebind its binary commit, preserving its full state tail.
// Serving loaders never infer this transition or accept historical model29.
struct ExecutionCacheSourceStateTarget {
    std::string formulaFingerprint;
    std::string historyFingerprint;
    std::string binaryFingerprint;
};

struct ExecutionCacheSourceStateMigration {
    std::string state;
    int64_t sourceModelVersion = 0;
    int64_t targetModelVersion = 0;
    int64_t revision = 0;
    int64_t optimizerStep = 0;
    size_t archivedHistorySamples = 0;
    size_t archivedHistoryBuckets = 0;
    bool preservedStateTail = false;
};

bool migrateExecutionCacheSourceState(
    const std::string &source,
    const ExecutionCacheSourceStateTarget &target,
    ExecutionCacheSourceStateMigration &result,
    std::string &error);

} // namespace navi_sched
