#include "gtest/gtest.h"

#include "rtp_llm/cpp/multimodal_processor/MMProcessorConfig.h"

// Ingress-ownership decision table shared by processor construction sites. Kept out of
// MMRdmaTransportTest.cc because that binary links torch (and therefore has to run on a GPU
// node) while these assertions are pure logic over enums.
namespace rtp_llm {

TEST(MMProcessorConfigTest, resolvesOnlyValidIngressConfigurations) {
    EXPECT_EQ(resolveMMProcessorKind(false, VIT_SEPARATION_ROLE, false, PDFUSION, 0, 0), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(false, VIT_SEPARATION_REMOTE, true, PREFILL, 0, 0), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, PDFUSION, 0, 0), MMProcessorKind::LOCAL);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_REMOTE, false, PREFILL, 0, 0), MMProcessorKind::REMOTE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, PDFUSION, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_REMOTE, true, PDFUSION, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_ROLE, false, PDFUSION, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_ROLE, false, PREFILL, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, PDFUSION, 1, 0), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, DECODE, 0, 0), MMProcessorKind::NONE);
    // VIT and FRONTEND never own multimodal ingress, so they short-circuit to NONE regardless of
    // separation or a present local engine. These rows are production-unreachable for this gate
    // (those roles don't run the LLM ingress path) but pin the table's completeness.
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, VIT, 0, 0), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, FRONTEND, 0, 0), MMProcessorKind::NONE);
}

TEST(MMProcessorConfigTest, downstreamPipelineStagesDoNotRequireMultimodalIngress) {
    for (auto role_type : {PDFUSION, PREFILL}) {
        for (int64_t pp_rank : {1, 2}) {
            for (auto separation : {VIT_SEPARATION_LOCAL, VIT_SEPARATION_REMOTE}) {
                const auto decision = resolveAndLogMMProcessorKind(
                    true, separation, false, role_type, 0, pp_rank, "qwen35_dense", "LocalRpcServer");
                EXPECT_TRUE(decision.ok()) << decision.error;
                EXPECT_EQ(decision.kind, MMProcessorKind::NONE);
            }
        }
    }
}

TEST(MMProcessorConfigTest, firstPipelineStageKeepsMultimodalValidation) {
    for (auto role_type : {PDFUSION, PREFILL}) {
        for (auto separation : {VIT_SEPARATION_LOCAL, VIT_SEPARATION_REMOTE}) {
            const bool requires_local_engine = separation == VIT_SEPARATION_LOCAL;
            const auto valid = resolveAndLogMMProcessorKind(
                true, separation, requires_local_engine, role_type, 0, 0, "qwen35_dense", "LocalRpcServer");
            EXPECT_TRUE(valid.ok()) << valid.error;
            EXPECT_EQ(valid.kind, requires_local_engine ? MMProcessorKind::LOCAL : MMProcessorKind::REMOTE);

            const auto invalid = resolveAndLogMMProcessorKind(
                true, separation, !requires_local_engine, role_type, 0, 0, "qwen35_dense", "LocalRpcServer");
            EXPECT_FALSE(invalid.ok());
            EXPECT_EQ(invalid.kind, MMProcessorKind::INVALID);
        }
    }
}

}  // namespace rtp_llm
