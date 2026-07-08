#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include <cstdlib>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {

TEST(BlockTreeCacheFactoryTest, UsableBlockCountReservesBlockZeroWithinBudget) {
    // Budget fits 4 blocks: reserved block 0 counts within the budget, so usable = 3.
    EXPECT_EQ(computeHostUsableBlockCount(4 * 4096, 4096), 3u);
    // Budget fits exactly 1 block: only the reserved block, usable = 0.
    EXPECT_EQ(computeHostUsableBlockCount(4096, 4096), 0u);
    // Budget smaller than one block: usable = 0.
    EXPECT_EQ(computeHostUsableBlockCount(100, 4096), 0u);
    // Defensive: zero stride returns 0 instead of dividing by zero.
    EXPECT_EQ(computeHostUsableBlockCount(4096, 0), 0u);
}

TEST(BlockTreeCacheFactoryTest, ShouldPinHostBlockPoolHonorsEnv) {
    ::unsetenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
    EXPECT_TRUE(shouldPinHostBlockPool());  // default on when unset

    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "0", 1);
    EXPECT_FALSE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "off", 1);
    EXPECT_FALSE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "FALSE", 1);
    EXPECT_FALSE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "1", 1);
    EXPECT_TRUE(shouldPinHostBlockPool());

    ::unsetenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
}

TEST(BlockTreeCacheFactoryTest, ResolveDiskMountPathSelectsByLocalRank) {
    EXPECT_EQ(resolveDiskMountPath("/mnt/d0,/mnt/d1,/mnt/d2", 3, 0), "/mnt/d0");
    EXPECT_EQ(resolveDiskMountPath("/mnt/d0,/mnt/d1,/mnt/d2", 3, 2), "/mnt/d2");
    // split() trims surrounding whitespace.
    EXPECT_EQ(resolveDiskMountPath(" /mnt/d0 , /mnt/d1 ", 2, 1), "/mnt/d1");
}

TEST(BlockTreeCacheFactoryTest, ResolveDiskMountPathRejectsCountMismatch) {
    // RTP_LLM_CHECK aborts unless core-dump-on-exception is disabled; flip it so the
    // guard is observable as a throw in this test env.
    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(resolveDiskMountPath("/mnt/d0,/mnt/d1", 3, 0));
    EXPECT_ANY_THROW(resolveDiskMountPath("", 1, 0));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;
}

TEST(BlockTreeCacheFactoryTest, ResolveDiskMountPathRejectsOutOfRangeRank) {
    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(resolveDiskMountPath("/mnt/d0,/mnt/d1", 2, 2));
    EXPECT_ANY_THROW(resolveDiskMountPath("/mnt/d0,/mnt/d1", 2, -1));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;
}

}  // namespace rtp_llm
