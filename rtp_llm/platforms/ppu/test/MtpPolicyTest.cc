#include <gtest/gtest.h>
#include "rtp_llm/platforms/ppu/MtpPolicy.h"

namespace rtp_llm::platforms {

TEST(MtpPlatformPolicyTest, OtherModelsRetainDefaultProtocol) {
    EXPECT_EQ(mtpPolicy("deepseek_v4", true, 4096, 43).select_hidden, nullptr);
    EXPECT_EQ(mtpPolicy("deepseek_v4", false, 7168, 61).select_hidden, nullptr);
    EXPECT_EQ(mtpPolicy("deepseek_v4", true).clone_cache_snapshot, nullptr);
    for (const auto& name : {"qwen3", "deepseek_v4_mtp", "dspark"}) {
        const auto policy = mtpPolicy(name);
        EXPECT_FALSE(policy.preserve_accepted_hidden);
        EXPECT_EQ(policy.select_hidden, nullptr);
        EXPECT_EQ(policy.clone_cache_snapshot, nullptr);
    }
#if !USE_PPU
    EXPECT_EQ(mtpPolicy("deepseek_v4").select_hidden, nullptr);
#endif
}

TEST(MtpPlatformPolicyTest, AcceptedHiddenStatesPreservePreHcBranches) {
    const auto policy = ppu::mtpPolicy("deepseek_v4", 4096, 43);
    ASSERT_TRUE(policy.select_hidden);
    EXPECT_TRUE(policy.preserve_accepted_hidden);
    auto indices = torch::tensor({1, 0, 3}, torch::kInt64);
    for (const auto& shape : std::vector<std::vector<int64_t>>{{12, 16}, {12, 4, 16}}) {
        int64_t count = 1;
        for (auto dim : shape) {
            count *= dim;
        }
        auto hidden = torch::arange(count, torch::kFloat32).reshape(shape);
        auto selected = policy.select_hidden(hidden, indices, 3, 4);
        auto expected = torch::stack({hidden[1], hidden[4], hidden[11]});
        EXPECT_EQ(selected.sizes(), expected.sizes());
        EXPECT_TRUE(torch::equal(selected, expected));
    }
    EXPECT_THROW(policy.select_hidden(torch::zeros({12, 4}), indices, 2, 4), c10::Error);
}

TEST(MtpPlatformPolicyTest, FixedCacheTablesDoNotUseLinearPermutation) {
    const auto policy = ppu::mtpPolicy("deepseek_v4", 4096, 43);
    ASSERT_TRUE(policy.clone_cache_snapshot);
    EXPECT_TRUE(policy.clone_cache_snapshot({CacheGroupType::SWA, CacheGroupType::FULL}));
    EXPECT_FALSE(policy.clone_cache_snapshot({CacheGroupType::SWA, CacheGroupType::LINEAR}));
}

}  // namespace rtp_llm::platforms
