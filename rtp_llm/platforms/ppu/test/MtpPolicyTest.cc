#include <gtest/gtest.h>
#include "rtp_llm/platforms/ppu/MtpPolicy.h"

namespace rtp_llm::platforms {

TEST(MtpPlatformPolicyTest, OtherModelsRetainDefaultProtocol) {
    EXPECT_FALSE(ppu::isDsv4Flash("deepseek_v4", 7168, 61));
    for (const auto& name : {"qwen3", "deepseek_v4_mtp", "dspark"}) {
        EXPECT_FALSE(ppu::isDsv4Flash(name, 4096, 43));
    }
#if USE_PPU
    EXPECT_TRUE(ppu::isDsv4Flash("deepseek_v4", 4096, 43));
#else
    EXPECT_FALSE(ppu::isDsv4Flash("deepseek_v4", 4096, 43));
#endif
}

TEST(MtpPlatformPolicyTest, AcceptedHiddenStatesPreservePreHcBranches) {
    auto indices = torch::tensor({1, 0, 3}, torch::kInt64);
    for (const auto& shape : std::vector<std::vector<int64_t>>{{12, 16}, {12, 4, 16}}) {
        int64_t count = 1;
        for (auto dim : shape) {
            count *= dim;
        }
        auto hidden = torch::arange(count, torch::kFloat32).reshape(shape);
        auto selected = ppu::selectAcceptedHiddenStates(hidden, indices, 3, 4);
        auto expected = torch::stack({hidden[1], hidden[4], hidden[11]});
        EXPECT_EQ(selected.sizes(), expected.sizes());
        EXPECT_TRUE(torch::equal(selected, expected));
    }
    EXPECT_THROW(ppu::selectAcceptedHiddenStates(torch::zeros({12, 4}), indices, 2, 4), c10::Error);
}

TEST(MtpPlatformPolicyTest, FixedCacheTablesDoNotUseLinearPermutation) {
    EXPECT_TRUE(ppu::cloneFixedCacheSnapshot({CacheGroupType::SWA, CacheGroupType::FULL}));
    EXPECT_FALSE(ppu::cloneFixedCacheSnapshot({CacheGroupType::SWA, CacheGroupType::LINEAR}));
}

}  // namespace rtp_llm::platforms
