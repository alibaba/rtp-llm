#pragma once

#include "autil/DataBuffer.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace navi_sched {

struct NaviSchedPendingCacheReservation {
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    int64_t batchId = 0;
    int64_t scheduleBatchId = 0;
    std::vector<uint64_t> fullReadyBlockHashes;
    std::vector<uint64_t> tailReadyBlockHashes;
};

struct AgentPendingCacheToMasterData {
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    bool fullSnapshot = false;
    std::vector<NaviSchedPendingCacheReservation> addedReservations;
    std::vector<int64_t> removedBatchIds;
    uint64_t revision = 0;
    int64_t batchCount = 0;
    int64_t uniqueBlockCount = 0;
    int64_t uniqueTailReadyBlockCount = 0;
    uint64_t checksum = 0;
};

using AgentPendingCacheToMasterDataPtr =
    std::shared_ptr<AgentPendingCacheToMasterData>;

}
