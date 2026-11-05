#include "aios/navi_sched/data/AgentCacheToMasterData.h"
#include "aios/navi_sched/data/BlockHashUtil.h"

namespace navi_sched {

void AgentCacheToMasterData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(cacheKeyType);
    dataBuffer.write(totalBlocks);
    dataBuffer.write(fullSnapshot);
    BlockHashUtil::write(dataBuffer, addedBlockHashes);
    BlockHashUtil::write(dataBuffer, removedBlockHashes);
    BlockHashUtil::write(dataBuffer, addedFullReadyBlockHashes);
    BlockHashUtil::write(dataBuffer, removedFullReadyBlockHashes);
    BlockHashUtil::write(dataBuffer, addedTailReadyBlockHashes);
    BlockHashUtil::write(dataBuffer, removedTailReadyBlockHashes);
    dataBuffer.write(blockSize);
    dataBuffer.write(linearStep);
    dataBuffer.write(hybridTailRequired);
    dataBuffer.write(revision);
    dataBuffer.write(cachedBlockCount);
    dataBuffer.write(hashChecksum);
    dataBuffer.write(fullReadyBlockCount);
    dataBuffer.write(fullReadyHashChecksum);
    dataBuffer.write(tailReadyBlockCount);
    dataBuffer.write(tailReadyHashChecksum);
    dataBuffer.write(rtpCachedBlockCount);
}

void AgentCacheToMasterData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(cacheKeyType);
    dataBuffer.read(totalBlocks);
    dataBuffer.read(fullSnapshot);
    BlockHashUtil::read(dataBuffer, addedBlockHashes);
    BlockHashUtil::read(dataBuffer, removedBlockHashes);
    BlockHashUtil::read(dataBuffer, addedFullReadyBlockHashes);
    BlockHashUtil::read(dataBuffer, removedFullReadyBlockHashes);
    BlockHashUtil::read(dataBuffer, addedTailReadyBlockHashes);
    BlockHashUtil::read(dataBuffer, removedTailReadyBlockHashes);
    dataBuffer.read(blockSize);
    dataBuffer.read(linearStep);
    dataBuffer.read(hybridTailRequired);
    dataBuffer.read(revision);
    dataBuffer.read(cachedBlockCount);
    dataBuffer.read(hashChecksum);
    dataBuffer.read(fullReadyBlockCount);
    dataBuffer.read(fullReadyHashChecksum);
    dataBuffer.read(tailReadyBlockCount);
    dataBuffer.read(tailReadyHashChecksum);
    dataBuffer.read(rtpCachedBlockCount);
}

}
