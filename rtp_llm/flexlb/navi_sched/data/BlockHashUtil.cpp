#include "aios/navi_sched/data/BlockHashUtil.h"

namespace navi_sched {
const NaviSchedBlockHashes EMPTY_BLOCK_HASHES;

const NaviSchedBlockHashes &BlockHashUtil::get(const NaviSchedBlockHashesConstPtr &hashes)
{
    return hashes ? *hashes : EMPTY_BLOCK_HASHES;
}

size_t BlockHashUtil::size(const NaviSchedBlockHashesConstPtr &hashes)
{
    return hashes ? hashes->size() : 0;
}

bool BlockHashUtil::validTokenBlocks(
    int64_t tokenCount, int64_t blockSize,
    const NaviSchedBlockHashesConstPtr &hashes)
{
    if (tokenCount < 0 || blockSize <= 0) {
        return false;
    }
    return size(hashes) <= static_cast<size_t>(tokenCount / blockSize);
}

size_t BlockHashUtil::getMatchableBlockCount(
    int64_t tokenCount, int64_t blockSize,
    const NaviSchedBlockHashesConstPtr &hashes)
{
    const auto hashCount = size(hashes);
    if (tokenCount <= 0 || blockSize <= 0 || tokenCount % blockSize != 0 ||
        hashCount != static_cast<size_t>(tokenCount / blockSize) ||
        hashCount == 0) {
        return hashCount;
    }
    return hashCount - 1;
}

void BlockHashUtil::write(autil::DataBuffer &dataBuffer,
                          const NaviSchedBlockHashesConstPtr &hashes)
{
    dataBuffer.write(get(hashes));
}

void BlockHashUtil::read(autil::DataBuffer &dataBuffer,
                         NaviSchedBlockHashesConstPtr &hashes)
{
    auto values = std::make_shared<NaviSchedBlockHashes>();
    dataBuffer.read(*values);
    hashes = std::move(values);
}

}
