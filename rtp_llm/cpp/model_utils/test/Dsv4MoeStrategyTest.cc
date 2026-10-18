#include "rtp_llm/cpp/model_utils/Dsv4MoeStrategy.h"
#include "gtest/gtest.h"

namespace rtp_llm {

TEST(Dsv4MoeStrategyTest, ExplicitStrategyTakesPrecedence) {
    EXPECT_EQ(resolveDsv4MoeStrategy(" grouped_fp4 ", "sm120_decode"), "grouped_fp4");
    EXPECT_EQ(resolveDsv4MoeStrategy("sm120_fused_moe", "sm120_decode"), "sm120_fused_moe");
}

TEST(Dsv4MoeStrategyTest, AutoAcceptsOnlySupportedAliases) {
    EXPECT_EQ(resolveDsv4MoeStrategy("auto", " sm120_decode\n"), "sm120_decode");
    EXPECT_EQ(resolveDsv4MoeStrategy(" \t", "sm120_fused_moe"), "sm120_fused_moe");
    EXPECT_EQ(resolveDsv4MoeStrategy("auto", nullptr), "auto");
    EXPECT_EQ(resolveDsv4MoeStrategy("auto", "other_model_strategy"), "auto");
}

TEST(Dsv4MoeStrategyTest, AliasIsScopedToDsv4Descriptors) {
    for (const char* model : {"deepseek_v4", "deepseek_v4_mtp", "deepseek_v4_dspark"}) {
        EXPECT_EQ(resolveModelMoeStrategy(model, "auto", "sm120_decode"), "sm120_decode");
    }
    EXPECT_EQ(resolveModelMoeStrategy("qwen_2", "auto", "sm120_decode"), "auto");
    EXPECT_EQ(resolveModelMoeStrategy("deepseek_v41", "auto", "sm120_decode"), "auto");
}

}  // namespace rtp_llm
