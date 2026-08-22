#include "gtest/gtest.h"

#include "rtp_llm/cpp/multimodal_processor/MMProcessorConfig.h"

// Ingress-ownership decision table shared by processor construction sites. Kept out of
// MMRdmaTransportTest.cc because that binary links torch (and therefore has to run on a GPU
// node) while these assertions are pure logic over enums.
namespace rtp_llm {

TEST(MMProcessorConfigTest, resolvesOnlyValidIngressConfigurations) {
    EXPECT_EQ(resolveMMProcessorKind(false, VIT_SEPARATION_ROLE, false, PDFUSION, 0), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(false, VIT_SEPARATION_REMOTE, true, PREFILL, 0), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, PDFUSION, 0), MMProcessorKind::LOCAL);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_REMOTE, false, PREFILL, 0), MMProcessorKind::REMOTE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, PDFUSION, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_REMOTE, true, PDFUSION, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_ROLE, false, PDFUSION, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_ROLE, false, PREFILL, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, PDFUSION, 1), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, DECODE, 0), MMProcessorKind::NONE);
    // VIT and FRONTEND never own multimodal ingress, so they short-circuit to NONE regardless of
    // separation or a present local engine. These rows are production-unreachable for this gate
    // (those roles don't run the LLM ingress path) but pin the table's completeness.
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, VIT, 0), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, FRONTEND, 0), MMProcessorKind::NONE);
}

TEST(MMProcessorConfigTest, resolveAndLogReportsDecisionAndError) {
    const auto none_decision =
        resolveAndLogMMProcessorKind(true, VIT_SEPARATION_REMOTE, false, DECODE, 3, "qwen_vl", "probe");
    EXPECT_EQ(none_decision.kind, MMProcessorKind::NONE);
    EXPECT_TRUE(none_decision.ok());
    EXPECT_TRUE(none_decision.error.empty());

    const auto local_decision =
        resolveAndLogMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, PDFUSION, 0, "qwen_vl", "probe");
    EXPECT_EQ(local_decision.kind, MMProcessorKind::LOCAL);
    EXPECT_TRUE(local_decision.ok());

    const auto remote_decision =
        resolveAndLogMMProcessorKind(true, VIT_SEPARATION_REMOTE, false, PREFILL, 0, "qwen_vl", "probe");
    EXPECT_EQ(remote_decision.kind, MMProcessorKind::REMOTE);
    EXPECT_TRUE(remote_decision.ok());

    const auto invalid_decision =
        resolveAndLogMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, PDFUSION, 0, "qwen_vl", "probe");
    EXPECT_EQ(invalid_decision.kind, MMProcessorKind::INVALID);
    EXPECT_FALSE(invalid_decision.ok());
    EXPECT_NE(invalid_decision.error.find("vit_separation="), std::string::npos);
    EXPECT_NE(invalid_decision.error.find("has_local_engine="), std::string::npos);
    EXPECT_NE(invalid_decision.error.find("role_type="), std::string::npos);
    EXPECT_NE(invalid_decision.error.find("tp_rank="), std::string::npos);
    EXPECT_NE(invalid_decision.error.find("model_type=qwen_vl"), std::string::npos);
}

}  // namespace rtp_llm
