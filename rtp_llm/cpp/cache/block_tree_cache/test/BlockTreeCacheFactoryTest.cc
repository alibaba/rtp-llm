#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include <cstdlib>

#include "gtest/gtest.h"

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

}  // namespace rtp_llm
