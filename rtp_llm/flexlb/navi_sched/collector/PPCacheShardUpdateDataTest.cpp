#include "aios/navi_sched/collector/PPCacheShardUpdateData.h"
#include "unittest/unittest.h"
#include <stdexcept>

using namespace navi_sched;
class PPCacheShardUpdateDataTest : public TESTBASE {};

TEST_F(PPCacheShardUpdateDataTest, testAllTerminalOutcomesAndCacheChangesRoundtrip)
{
    PPCacheShardUpdateData input;
    auto &u = input.update;
    u.shardId = 3;
    u.streamGeneration = 42;
    u.sequence = 17;
    u.addedFull = {1, 2};
    u.removedFull = {3};
    u.addedTail = {2};
    u.removedTail = {4};
    u.terminals = {{{10, 1}, PPShardRequestStatus::SUCCEEDED},
                   {{11, 2}, PPShardRequestStatus::FAILED},
                   {{12, 3}, PPShardRequestStatus::CANCELLED}};
    ASSERT_TRUE(input.valid());
    autil::DataBuffer buffer;
    input.serialize(buffer);
    PPCacheShardUpdateData output;
    output.deserialize(buffer);
    ASSERT_TRUE(output.valid());
    ASSERT_EQ(3, output.update.shardId);
    ASSERT_EQ(42, output.update.streamGeneration);
    ASSERT_EQ(17, output.update.sequence);
    ASSERT_EQ(u.addedFull, output.update.addedFull);
    ASSERT_EQ(u.removedFull, output.update.removedFull);
    ASSERT_EQ(u.addedTail, output.update.addedTail);
    ASSERT_EQ(u.removedTail, output.update.removedTail);
    ASSERT_EQ(3, output.update.terminals.size());
    for (size_t i = 0; i < u.terminals.size(); ++i) {
        ASSERT_TRUE(u.terminals[i].request == output.update.terminals[i].request);
        ASSERT_EQ(u.terminals[i].status, output.update.terminals[i].status);
    }
}

TEST_F(PPCacheShardUpdateDataTest, testInvalidTerminalCannotBeSerialized)
{
    PPCacheShardUpdateData data;
    data.update.streamGeneration = 1;
    data.update.sequence = 1;
    for (auto status : {PPShardRequestStatus::PENDING, PPShardRequestStatus::DISCONNECTED,
                        static_cast<PPShardRequestStatus>(255)}) {
        data.update.terminals = {{{1, 1}, status}};
        ASSERT_FALSE(data.valid());
        autil::DataBuffer buffer;
        ASSERT_THROW(data.serialize(buffer), std::invalid_argument);
    }
}

TEST_F(PPCacheShardUpdateDataTest, testUnknownVersionCannotReuseOldPayload)
{
    PPCacheShardUpdateData data;
    data.update.streamGeneration = 1;
    data.update.sequence = 1;
    data.update.addedFull = {100};
    autil::DataBuffer buffer;
    buffer.write(uint32_t(PPCacheShardUpdateData::VERSION + 1));
    data.deserialize(buffer);
    ASSERT_FALSE(data.valid());
    ASSERT_TRUE(data.update.addedFull.empty());
    ASSERT_EQ(0, data.update.streamGeneration);
}

TEST_F(PPCacheShardUpdateDataTest, testSnapshotAndDeltaRoundtripUseTheSameMerge)
{
    PPCacheCollectorState collector(2);
    for (uint32_t shard = 0; shard < 2; ++shard) {
        collector.connect(shard, 1);
        PPCacheShardUpdateData data;
        data.update.shardId = shard;
        data.update.streamGeneration = 1;
        data.update.sequence = 1;
        data.update.fullSnapshot = true;
        data.update.addedFull = {7};
        autil::DataBuffer buffer;
        data.serialize(buffer);
        PPCacheShardUpdateData received;
        received.deserialize(buffer);
        collector.apply(received.update);
    }
    ASSERT_TRUE(collector.fullReady(7));
    PPCacheShardUpdateData delta;
    delta.update.shardId = 1;
    delta.update.streamGeneration = 1;
    delta.update.sequence = 2;
    delta.update.removedFull = {7};
    autil::DataBuffer buffer;
    delta.serialize(buffer);
    PPCacheShardUpdateData received;
    received.deserialize(buffer);
    auto result = collector.apply(received.update);
    ASSERT_EQ(std::vector<uint64_t>({7}), result.removedFull);
    ASSERT_FALSE(collector.fullReady(7));
}

TEST_F(PPCacheShardUpdateDataTest, testPeerReadinessRoundtripAndInvalidValue)
{
    for (int8_t available : {int8_t(-1), int8_t(0), int8_t(1)}) {
        PPCacheShardUpdateData data;
        data.update.streamGeneration = 1;
        data.update.sequence = 1;
        data.update.peerCacheReadAvailable = available;
        autil::DataBuffer buffer;
        data.serialize(buffer);
        PPCacheShardUpdateData copy;
        copy.deserialize(buffer);
        ASSERT_TRUE(copy.valid());
        EXPECT_EQ(available, copy.update.peerCacheReadAvailable);
    }
    for (int8_t invalid : {int8_t(-2), int8_t(2)}) {
        PPCacheShardUpdateData data;
        data.update.streamGeneration = 1;
        data.update.sequence = 1;
        data.update.peerCacheReadAvailable = invalid;
        EXPECT_FALSE(data.valid());
    }
}
