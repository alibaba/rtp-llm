#include "aios/navi_sched/data/AgentUpdateToMasterData.h"
#include "unittest/unittest.h"

using namespace navi_sched;
class CacheKeyTypeTest : public TESTBASE {};

TEST_F(CacheKeyTypeTest, testBusinessDefaultIsBlockPool)
{
    ASSERT_EQ(NaviSchedCacheKeyType::BLOCK_POOL, AgentCacheToMasterData().cacheKeyType);
    ASSERT_EQ(NaviSchedCacheKeyType::BLOCK_POOL,
              AgentPendingCacheToMasterData().cacheKeyType);
    ASSERT_EQ(24, AgentUpdateToMasterData().version);
}

TEST_F(CacheKeyTypeTest, testCacheRoundtripPreservesTypeAndOriginalState)
{
    for (auto type : {NaviSchedCacheKeyType::BLOCK_POOL, NaviSchedCacheKeyType::V6D}) {
        AgentUpdateToMasterData input;
        input.event = NaviSchedAgentEvent::CACHE_UPDATED;
        input.cache = std::make_shared<AgentCacheToMasterData>();
        auto &cache = *input.cache;
        cache.cacheKeyType = type;
        cache.fullSnapshot = true;
        cache.totalBlocks = 99;
        cache.blockSize = 256;
        cache.linearStep = 24;
        cache.hybridTailRequired = 1;
        cache.revision = 17;
        cache.cachedBlockCount = 2;
        cache.hashChecksum = 33;
        cache.fullReadyBlockCount = 2;
        cache.fullReadyHashChecksum = 34;
        cache.tailReadyBlockCount = 1;
        cache.tailReadyHashChecksum = 35;
        cache.rtpCachedBlockCount = 7;
        cache.addedBlockHashes =
            std::make_shared<NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
        cache.removedBlockHashes =
            std::make_shared<NaviSchedBlockHashes>(NaviSchedBlockHashes{44});
        ASSERT_TRUE(input.valid());
        autil::DataBuffer buffer;
        input.serialize(buffer);
        AgentUpdateToMasterData output;
        output.deserialize(buffer);
        ASSERT_TRUE(output.valid());
        ASSERT_EQ(type, output.cache->cacheKeyType);
        ASSERT_TRUE(output.cache->fullSnapshot);
        ASSERT_EQ(17u, output.cache->revision);
        ASSERT_EQ(33u, output.cache->hashChecksum);
        ASSERT_EQ(34u, output.cache->fullReadyHashChecksum);
        ASSERT_EQ(35u, output.cache->tailReadyHashChecksum);
        ASSERT_EQ(7, output.cache->rtpCachedBlockCount);
        ASSERT_EQ(*cache.addedBlockHashes, *output.cache->addedBlockHashes);
        ASSERT_EQ(*cache.removedBlockHashes, *output.cache->removedBlockHashes);
    }
}

TEST_F(CacheKeyTypeTest, testPendingRoundtripScopesSnapshotDeltaAndRetire)
{
    for (auto type : {NaviSchedCacheKeyType::BLOCK_POOL, NaviSchedCacheKeyType::V6D}) {
        AgentUpdateToMasterData input;
        input.event = NaviSchedAgentEvent::PENDING_CACHE_UPDATED;
        input.pendingCache = std::make_shared<AgentPendingCacheToMasterData>();
        auto &pending = *input.pendingCache;
        pending.cacheKeyType = type;
        pending.fullSnapshot = true;
        pending.revision = 5;
        pending.batchCount = 1;
        pending.uniqueBlockCount = 2;
        pending.uniqueTailReadyBlockCount = 1;
        pending.checksum = 81;
        NaviSchedPendingCacheReservation reservation;
        reservation.batchId = 9;
        reservation.scheduleBatchId = 19;
        reservation.fullReadyBlockHashes = {11, 22};
        reservation.tailReadyBlockHashes = {22};
        pending.addedReservations = {reservation};
        pending.removedBatchIds = {8};
        autil::DataBuffer buffer;
        input.serialize(buffer);
        AgentUpdateToMasterData output;
        output.deserialize(buffer);
        ASSERT_TRUE(output.valid());
        ASSERT_EQ(type, output.pendingCache->cacheKeyType);
        ASSERT_EQ(5u, output.pendingCache->revision);
        ASSERT_EQ(81u, output.pendingCache->checksum);
        ASSERT_EQ(pending.removedBatchIds, output.pendingCache->removedBatchIds);
        ASSERT_EQ(reservation.fullReadyBlockHashes,
                  output.pendingCache->addedReservations[0].fullReadyBlockHashes);
        ASSERT_EQ(reservation.tailReadyBlockHashes,
                  output.pendingCache->addedReservations[0].tailReadyBlockHashes);
    }
}

TEST_F(CacheKeyTypeTest, testRejectUnknownTypeAndNoncurrentVersion)
{
    AgentUpdateToMasterData input;
    input.event = NaviSchedAgentEvent::CACHE_UPDATED;
    input.cache = std::make_shared<AgentCacheToMasterData>();
    input.cache->cacheKeyType = static_cast<NaviSchedCacheKeyType>(2);
    ASSERT_FALSE(input.valid());
    input.cache->cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL;
    for (int version : {10, 11, 12, 13, 15}) {
        input.version = version;
        ASSERT_FALSE(input.valid());
        autil::DataBuffer buffer;
        buffer.write(version);
        AgentUpdateToMasterData output;
        output.deserialize(buffer);
        ASSERT_EQ(version, output.version);
        ASSERT_FALSE(output.valid());
        ASSERT_FALSE(output.cache);
    }
}
