#include "gtest/gtest.h"

#include "rtp_llm/cpp/cuda_graph/grouped_cache_replay_policy.h"

namespace rtp_llm {
namespace {

TEST(GroupedCacheReplayPolicyTest, SupportsOnlyCudaBackend) {
    EXPECT_TRUE(groupedCacheReplaySupported(true));
    EXPECT_FALSE(groupedCacheReplaySupported(false));
}

}  // namespace
}  // namespace rtp_llm
