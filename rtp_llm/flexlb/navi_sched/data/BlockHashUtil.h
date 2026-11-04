#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedCommon.h"
#include <autil/DataBuffer.h>

namespace navi_sched {

class BlockHashUtil {
public:
    static constexpr int64_t BLOCK_SIZE = 256;
    static const NaviSchedBlockHashes &get(const NaviSchedBlockHashesConstPtr &hashes);
    static size_t size(const NaviSchedBlockHashesConstPtr &hashes);
    static bool validTokenBlocks(int64_t tokenCount, int64_t blockSize,
                                 const NaviSchedBlockHashesConstPtr &hashes);
    static size_t getMatchableBlockCount(
        int64_t tokenCount, int64_t blockSize,
        const NaviSchedBlockHashesConstPtr &hashes);
    static void write(autil::DataBuffer &dataBuffer,
                      const NaviSchedBlockHashesConstPtr &hashes);
    static void read(autil::DataBuffer &dataBuffer, NaviSchedBlockHashesConstPtr &hashes);
};

} // namespace navi_sched
