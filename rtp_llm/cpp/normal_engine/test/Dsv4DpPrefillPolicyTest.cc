#include "rtp_llm/cpp/normal_engine/Dsv4DpPrefillPolicy.h"

#include <string>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(Dsv4DpPrefillPolicyTest, ResolvedStrategyUsesExplicitConfigBeforeAlias) {
    EXPECT_EQ(resolveDsv4MoeStrategy("sm120_a2a_decode", nullptr), "sm120_a2a_decode");
    EXPECT_EQ(resolveDsv4MoeStrategy("auto", "sm120_a2a_decode"), "sm120_a2a_decode");
    EXPECT_EQ(resolveDsv4MoeStrategy("sm120_fused_moe", "sm120_a2a_decode"), "sm120_fused_moe");
    EXPECT_EQ(resolveDsv4MoeStrategy("auto", "unknown"), "auto");
    EXPECT_EQ(resolveDsv4MoeStrategy("auto", nullptr), "auto");
    EXPECT_EQ(resolveDsv4MoeStrategy(" auto\t", " sm120_a2a_decode\n"), "sm120_a2a_decode");
    EXPECT_EQ(resolveDsv4MoeStrategy("\tsm120_fused_moe ", "sm120_a2a_decode"), "sm120_fused_moe");
    EXPECT_EQ(resolveDsv4MoeStrategy(" \n", "sm120_a2a_decode"), "sm120_a2a_decode");
}

TEST(Dsv4DpPrefillPolicyTest, EnvironmentAliasIsScopedToDsv4Descriptors) {
    for (const auto* model : {"deepseek_v4", "deepseek_v4_mtp", "deepseek_v4_dspark"}) {
        EXPECT_EQ(resolveModelMoeStrategy(model, "auto", "sm120_a2a_decode"), "sm120_a2a_decode");
    }
    for (const auto* model : {"qwen_3", "deepseek_v3", "deepseek_v41", "deepseek_v41_dspark", ""}) {
        EXPECT_EQ(resolveModelMoeStrategy(model, "auto", "sm120_a2a_decode"), "auto");
        EXPECT_EQ(resolveModelMoeStrategy(model, "sm120_fused_moe", "sm120_a2a_decode"), "sm120_fused_moe");
        EXPECT_EQ(resolveModelMoeStrategy(model, "sm120_a2a_decode", nullptr), "sm120_a2a_decode");
    }
}

TEST(Dsv4DpPrefillPolicyTest, ProfileGating) {
    // The retired profile is now an unsupported combination: non-speculative
    // sm120_a2a_decode PDFUSION with PP1/TP1 and DP=EP=world>1 is rejected at
    // startup. Every other combination keeps its previous behavior.
    for (int world : {2, 4, 8}) {
        SCOPED_TRACE(testing::Message() << "world=" << world);
        EXPECT_TRUE(sm120DecodeNonMtpFusedProfileUnsupported(
            "deepseek_v4", "sm120_a2a_decode", RoleType::PDFUSION, 1, 1, world, world, world, false, false));
        for (RoleType role : {RoleType::PREFILL, RoleType::DECODE}) {
            SCOPED_TRACE(testing::Message() << "role=" << role);
            EXPECT_FALSE(sm120DecodeNonMtpFusedProfileUnsupported(
                "deepseek_v4", "sm120_a2a_decode", role, 1, 1, world, world, world, false, false));
        }
    }
    auto unsupported = [](const char* model,
                          const char* strategy,
                          int         pp,
                          int         tp,
                          int         dp,
                          int         ep,
                          int         world,
                          bool        speculative = false,
                          bool        cp          = false) {
        return sm120DecodeNonMtpFusedProfileUnsupported(
            model, strategy, RoleType::PDFUSION, pp, tp, dp, ep, world, speculative, cp);
    };
    EXPECT_FALSE(unsupported("other", "sm120_a2a_decode", 1, 1, 4, 4, 4)) << "other models unchanged";
    EXPECT_FALSE(unsupported("deepseek_v4", nullptr, 1, 1, 4, 4, 4)) << "unset strategy unchanged";
    EXPECT_FALSE(unsupported("deepseek_v4", "other", 1, 1, 4, 4, 4)) << "other strategies unchanged";
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 2, 2, 1, 2, 4)) << "CEP2PP2 unchanged";
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 2, 4, 1, 4, 8)) << "CEP4PP2 unchanged";
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 2, 2, 4, 4)) << "TP split unchanged";
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 1, 1, 1)) << "single rank unchanged";
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 4, 2, 4)) << "partial EP group unchanged";
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 4, 4, 8)) << "world mismatch rejected";
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 4, 4, 4, true))
        << "speculative fused profile stays supported";
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 4, 4, 4, false, true)) << "local CP unchanged";
}

TEST(Dsv4DpPrefillPolicyTest, Sm120DecodeCudaGraphRequiresEagerExecution) {
    // The canonical decode backend is eager-only: a resolved sm120_a2a_decode
    // profile rejects CUDA graphs at construction, before warmup.
    EXPECT_TRUE(sm120DecodeRequiresEagerExecution("sm120_a2a_decode", true));
    EXPECT_FALSE(sm120DecodeRequiresEagerExecution("sm120_a2a_decode", false));
    // Unrelated strategies and an unresolved "auto" keep their graph policy.
    EXPECT_FALSE(sm120DecodeRequiresEagerExecution("sm120_fused_moe", true));
    EXPECT_FALSE(sm120DecodeRequiresEagerExecution("auto", true));
    EXPECT_FALSE(sm120DecodeRequiresEagerExecution(nullptr, true));
}

TEST(Dsv4DpPrefillPolicyTest, Sm120FusedMoeDpDecodeGuardAcceptsSupportedProfiles) {
    // PREFILL-only roles never join decode-phase collectives.
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", RoleType::PREFILL, 4));
    // TP-only and single-DP engines keep every rank in the same phase.
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", RoleType::DECODE, 1));
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", RoleType::PDFUSION, 1));
    // Other strategies, including unresolved "auto", are unaffected here.
    // Python strategy selection decides whether auto may pick sm120_fused_moe.
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_a2a_decode", RoleType::DECODE, 4));
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("auto", RoleType::PDFUSION, 4));
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned(nullptr, RoleType::DECODE, 4));
}

TEST(Dsv4DpPrefillPolicyTest, Sm120FusedMoeDpDecodeGuardRejectsDecodeCapableTopologies) {
    for (RoleType role : {RoleType::DECODE, RoleType::PDFUSION}) {
        SCOPED_TRACE(testing::Message() << "role=" << role);
        // Decode-capable roles would issue WORLD collectives out of phase.
        EXPECT_TRUE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", role, 2));
        EXPECT_TRUE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", role, 8));
        EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("auto", role, 2));
        EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("auto", role, 8));
    }
}

}  // namespace
}  // namespace rtp_llm
