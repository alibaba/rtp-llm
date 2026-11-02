#pragma once

#include "aios/navi_sched/common/NaviSchedCommon.h"
#include <autil/DataBuffer.h>
#include "navi/common.h"
#include <vector>

namespace navi_sched {

struct AgentCacheToMasterData {
public:
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

public:
    int64_t totalBlocks = -1;
    bool fullSnapshot = false;
    NaviSchedBlockHashesConstPtr addedBlockHashes;
    NaviSchedBlockHashesConstPtr removedBlockHashes;
    NaviSchedBlockHashesConstPtr addedFullReadyBlockHashes;
    NaviSchedBlockHashesConstPtr removedFullReadyBlockHashes;
    NaviSchedBlockHashesConstPtr addedTailReadyBlockHashes;
    NaviSchedBlockHashesConstPtr removedTailReadyBlockHashes;
    int64_t blockSize = -1;
    int64_t linearStep = -1;
    int8_t hybridTailRequired = -1;
    uint64_t revision = 0;
    int64_t cachedBlockCount = -1;
    uint64_t hashChecksum = 0;
    int64_t fullReadyBlockCount = -1;
    uint64_t fullReadyHashChecksum = 0;
    int64_t tailReadyBlockCount = -1;
    uint64_t tailReadyHashChecksum = 0;
    int64_t rtpCachedBlockCount = -1;
};

NAVI_TYPEDEF_PTR(AgentCacheToMasterData);

}
