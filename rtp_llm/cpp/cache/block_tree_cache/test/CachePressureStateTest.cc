#include "rtp_llm/cpp/cache/block_tree_cache/CachePressureState.h"

#include <gtest/gtest.h>

namespace rtp_llm {

TEST(CachePressureStateTest, ReservationsAndDetachedCopiesAreNotPublished) {
    CachePressureState  state;
    PressureTransferKey key{0, "host-full", CacheTransferOperation::EVICT, Tier::DEVICE, Tier::HOST};
    state.recordCopy(key, 1, 4096, true);
    EXPECT_EQ(state.transfer(key).copied_blocks, 1u);
    EXPECT_EQ(state.transfer(key).published_blocks, 0u);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 0u);
}

TEST(CachePressureStateTest, DeduplicatesPhysicalBlockAndRetainsCumulativeTotals) {
    CachePressureState  state;
    PressureTransferKey key{0, "host-full", CacheTransferOperation::EVICT, Tier::DEVICE, Tier::HOST};
    state.setHostOwner({0, 101}, PressureBlock{"host-full", 7});
    state.setHostOwner({1, 202}, PressureBlock{"host-full", 7});
    state.recordPublication(key, 1, 4096);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 1u);
    state.setHostOwner({0, 101}, std::nullopt);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 1u);
    state.setHostOwner({1, 202}, std::nullopt);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 0u);
    EXPECT_EQ(state.transfer(key).published_blocks, 1u);
}

TEST(CachePressureStateTest, ReplacementAndRepeatedUpdatesPreserveDistinctResidency) {
    CachePressureState state;
    state.setHostOwner({0, 101}, PressureBlock{"host-full", 7});
    state.setHostOwner({0, 101}, PressureBlock{"host-full", 7});
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 1u);
    state.setHostOwner({0, 101}, PressureBlock{"other-host", 8});
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 0u);
    EXPECT_EQ(state.pool("other-host").published_host_blocks, 1u);
    state.setHostOwner({0, 101}, std::nullopt);
    state.setHostOwner({0, 101}, std::nullopt);
    EXPECT_EQ(state.pool("other-host").published_host_blocks, 0u);
    EXPECT_TRUE(state.owners_.empty());
    EXPECT_TRUE(state.owners_per_block_.empty());
}

TEST(CachePressureStateTest, PartialCopyFailureAndPublicationHaveSeparateTotals) {
    CachePressureState  state;
    PressureTransferKey key{0, "host-full", CacheTransferOperation::EVICT, Tier::DEVICE, Tier::HOST};
    state.recordCopy(key, 2, 8192, false);
    state.recordCopy(key, 1, 4096, true);
    EXPECT_EQ(state.transfer(key).copied_blocks, 3u);
    EXPECT_EQ(state.transfer(key).copied_bytes, 12288u);
    EXPECT_EQ(state.transfer(key).physical_failed_operations, 1u);
    EXPECT_EQ(state.transfer(key).failed_operations, 0u);
    state.recordWorkflowFailure(key);
    EXPECT_EQ(state.transfer(key).failed_operations, 1u);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 0u);
    state.recordPublication(key, 1, 4096);
    EXPECT_EQ(state.transfer(key).published_blocks, 1u);
    EXPECT_EQ(state.transfer(key).published_bytes, 4096u);
}

TEST(CachePressureStateTest, EmptyReadsDoNotCreateEntriesAndTriggersArePerGroup) {
    CachePressureState  state;
    PressureTransferKey key{0, "host-full", CacheTransferOperation::EVICT, Tier::DEVICE, Tier::HOST};
    EXPECT_EQ(state.transfer(key).copied_blocks, 0u);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 0u);
    EXPECT_EQ(state.watermarkTriggers(0), 0u);
    EXPECT_TRUE(state.transfers_.empty());
    EXPECT_TRUE(state.pools_.empty());
    EXPECT_TRUE(state.triggers_.empty());
    state.recordWatermarkTrigger(0);
    state.recordWatermarkTrigger(0);
    state.recordWatermarkTrigger(1);
    EXPECT_EQ(state.watermarkTriggers(0), 2u);
    EXPECT_EQ(state.watermarkTriggers(1), 1u);
}

}  // namespace rtp_llm
