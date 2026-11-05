#include "aios/navi_sched/data/AgentPendingCacheToMasterData.h"

namespace navi_sched {

void NaviSchedPendingCacheReservation::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(batchId);
    dataBuffer.write(scheduleBatchId);
    dataBuffer.write(fullReadyBlockHashes);
    dataBuffer.write(tailReadyBlockHashes);
}

void NaviSchedPendingCacheReservation::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(batchId);
    dataBuffer.read(scheduleBatchId);
    dataBuffer.read(fullReadyBlockHashes);
    dataBuffer.read(tailReadyBlockHashes);
}

void AgentPendingCacheToMasterData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(cacheKeyType);
    dataBuffer.write(fullSnapshot);
    dataBuffer.write(addedReservations);
    dataBuffer.write(removedBatchIds);
    dataBuffer.write(revision);
    dataBuffer.write(batchCount);
    dataBuffer.write(uniqueBlockCount);
    dataBuffer.write(uniqueTailReadyBlockCount);
    dataBuffer.write(checksum);
}

void AgentPendingCacheToMasterData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(cacheKeyType);
    dataBuffer.read(fullSnapshot);
    dataBuffer.read(addedReservations);
    dataBuffer.read(removedBatchIds);
    dataBuffer.read(revision);
    dataBuffer.read(batchCount);
    dataBuffer.read(uniqueBlockCount);
    dataBuffer.read(uniqueTailReadyBlockCount);
    dataBuffer.read(checksum);
}

}
