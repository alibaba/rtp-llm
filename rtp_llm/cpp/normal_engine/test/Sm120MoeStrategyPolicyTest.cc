#include "rtp_llm/cpp/normal_engine/Sm120MoeStrategyPolicy.h"

#include <initializer_list>
#include <string>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(Sm120MoeStrategyPolicyTest, NormalizeMoeStrategyTrimsAndDefaultsEmptyToAuto) {
    EXPECT_EQ(normalizeMoeStrategy("sm120_a2a_decode"), "sm120_a2a_decode");
    EXPECT_EQ(normalizeMoeStrategy("\tsm120_fused_moe "), "sm120_fused_moe");
    EXPECT_EQ(normalizeMoeStrategy(""), "auto");
    EXPECT_EQ(normalizeMoeStrategy(" \n\t"), "auto");
    EXPECT_EQ(normalizeMoeStrategy("auto"), "auto");
}

TEST(Sm120MoeStrategyPolicyTest, ProfileGating) {
    // Reject non-speculative sm120_a2a_decode PDFUSION with PP1/TP1 and DP=EP=world>1.
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
                          bool        pads_missing_phases = false,
                          bool        cp                  = false) {
        return sm120DecodeNonMtpFusedProfileUnsupported(
            model, strategy, RoleType::PDFUSION, pp, tp, dp, ep, world, pads_missing_phases, cp);
    };
    EXPECT_FALSE(unsupported("other", "sm120_a2a_decode", 1, 1, 4, 4, 4));
    EXPECT_FALSE(unsupported("deepseek_v4", nullptr, 1, 1, 4, 4, 4));
    EXPECT_FALSE(unsupported("deepseek_v4", "other", 1, 1, 4, 4, 4));
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 2, 2, 1, 2, 4));
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 2, 4, 1, 4, 8));
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 2, 2, 4, 4));
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 1, 1, 1));
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 4, 2, 4));
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 4, 4, 8));
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 4, 4, 4, true));
    EXPECT_FALSE(unsupported("deepseek_v4", "sm120_a2a_decode", 1, 1, 4, 4, 4, false, true));
}

TEST(Sm120MoeStrategyPolicyTest, Sm120DecodeCudaGraphRequiresEagerExecution) {
    // The canonical decode backend is eager-only: a resolved sm120_a2a_decode
    // profile rejects CUDA graphs at construction, before warmup.
    EXPECT_TRUE(sm120DecodeRequiresEagerExecution("sm120_a2a_decode", true));
    EXPECT_FALSE(sm120DecodeRequiresEagerExecution("sm120_a2a_decode", false));
    // Unrelated strategies and an unresolved "auto" keep their graph policy.
    EXPECT_FALSE(sm120DecodeRequiresEagerExecution("sm120_fused_moe", true));
    EXPECT_FALSE(sm120DecodeRequiresEagerExecution("auto", true));
    EXPECT_FALSE(sm120DecodeRequiresEagerExecution(nullptr, true));
}

TEST(Sm120MoeStrategyPolicyTest, Sm120FusedMoeDpDecodeGuardAcceptsSupportedProfiles) {
    // Single-DP engines keep every rank in the same phase on every role.
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", 1));
    // Other strategies, including unresolved "auto", are unaffected here.
    // Python strategy selection decides whether auto may pick sm120_fused_moe.
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("sm120_a2a_decode", 4));
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("auto", 4));
    EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned(nullptr, 4));
}

TEST(Sm120MoeStrategyPolicyTest, Sm120FusedMoeDpDecodeGuardRejectsDpAcrossBackendsAndRoles) {
    // Every role and model backend decodes: DP>1 with sm120_fused_moe would
    // issue WORLD collectives out of phase across ranks.
    for (int dp : {2, 4, 8}) {
        SCOPED_TRACE(testing::Message() << "dp=" << dp);
        EXPECT_TRUE(sm120FusedMoeDpDecodeMisaligned("sm120_fused_moe", dp));
        EXPECT_FALSE(sm120FusedMoeDpDecodeMisaligned("auto", dp));
    }
    // Explicitly pin backend-wide behavior: non-V4 callers hitting NormalEngine
    // with sm120_fused_moe and DP>1 are covered by this model-agnostic predicate.
}

TEST(Sm120MoeStrategyPolicyTest, Sm120CpPrefillCudaGraphRequiresEagerExecution) {
    // CP4 PREFILL with pure-CP sm120_fused_moe rejects CUDA graphs before
    // warmup; the same profile without graphs stays admitted.
    EXPECT_TRUE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::PREFILL, 1, 4, 4, true, true));
    EXPECT_FALSE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::PREFILL, 1, 4, 4, true, false));
    // Every pure-CP shape (CP == TP == EP > 1) on a role that keeps MoE CP
    // reaches the same eager-only executor path.
    for (int cp : {2, 8}) {
        SCOPED_TRACE(testing::Message() << "cp=" << cp);
        EXPECT_TRUE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::PREFILL, 1, cp, cp, true, true));
    }
    EXPECT_TRUE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::PDFUSION, 1, 4, 4, true, true));
    // Direct C++ callers leaving strategy="auto" on SM120 DeepSeek-V4 pure CP are also caught.
    EXPECT_TRUE(
        sm120CpPrefillRequiresEagerExecution("auto", "deepseek_v4", RoleType::PREFILL, 1, 4, 4, true, true, true));
    EXPECT_FALSE(
        sm120CpPrefillRequiresEagerExecution("auto", "deepseek_v4", RoleType::PREFILL, 1, 4, 4, true, true, false));
    EXPECT_FALSE(sm120CpPrefillRequiresEagerExecution("auto", "qwen3", RoleType::PREFILL, 1, 4, 4, true, true, true));
}

TEST(Sm120MoeStrategyPolicyTest, Sm120CpPrefillCudaGraphGuardKeepsOtherProfiles) {
    // DECODE roles never enable MoE CP.
    EXPECT_FALSE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::DECODE, 1, 4, 4, true, true));
    // Without prefill CP sm120_fused_moe uses its WORLD path, including
    // single-worker PDFUSION with MTP.
    EXPECT_FALSE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::PDFUSION, 1, 1, 1, false, true));
    EXPECT_FALSE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::PREFILL, 1, 4, 4, false, true));
    // CP that does not span the EP group, or a single rank, is not pure CP.
    EXPECT_FALSE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::PREFILL, 1, 4, 8, true, true));
    EXPECT_FALSE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::PREFILL, 1, 1, 1, true, true));
    // PP>1 engines never capture graphs, so CEP4PP2 keeps its behavior.
    EXPECT_FALSE(sm120CpPrefillRequiresEagerExecution("sm120_fused_moe", RoleType::PREFILL, 2, 4, 4, true, true));
    // Other strategies, including the non-SM120 defaults and unresolved
    // "auto", keep their graph policy.
    for (const char* strategy :
         std::initializer_list<const char*>{"sm120_a2a_decode", "auto", "fp8_per_block_pure_cp", nullptr}) {
        SCOPED_TRACE(testing::Message() << "strategy=" << (strategy ? strategy : "<null>"));
        EXPECT_FALSE(sm120CpPrefillRequiresEagerExecution(strategy, RoleType::PREFILL, 1, 4, 4, true, true));
    }
}

}  // namespace
}  // namespace rtp_llm
