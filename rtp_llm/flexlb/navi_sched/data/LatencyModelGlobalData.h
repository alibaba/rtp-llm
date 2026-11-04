#pragma once

#include <autil/DataBuffer.h>

#include <cstdint>

namespace navi_sched {

struct LatencyModelGlobalData {
    bool valid() const;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    int64_t masterEpoch = 0;
    uint64_t version = 0;
    int32_t formula = 0;
    int64_t binaryVersion = 0;
    uint32_t parameterCount = 0;
    double globalZeroPoint = 0.0;
};

} // namespace navi_sched
