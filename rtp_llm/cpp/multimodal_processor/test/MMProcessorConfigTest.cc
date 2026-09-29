#include "gtest/gtest.h"

#include "rtp_llm/cpp/multimodal_processor/MMProcessorConfig.h"

// Ingress-ownership decision table shared by processor construction sites. Kept out of
// MMRdmaTransportTest.cc because that binary links torch (and therefore has to run on a GPU
// node) while these assertions are pure logic over enums.
namespace rtp_llm {

TEST(MMProcessorConfigTest, resolvesOnlyValidIngressConfigurations) {
    EXPECT_EQ(resolveMMProcessorKind(false, VIT_SEPARATION_ROLE, false, PDFUSION, 0, 0), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(false, VIT_SEPARATION_REMOTE, true, PREFILL, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, PDFUSION, 0, 0), MMProcessorKind::LOCAL);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_REMOTE, false, PREFILL, 0, 0), MMProcessorKind::REMOTE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, PDFUSION, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_REMOTE, true, PDFUSION, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_ROLE, false, PDFUSION, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_ROLE, false, PREFILL, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, PDFUSION, 1, 0), MMProcessorKind::NONE);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, PDFUSION, 1, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, PDFUSION, 0, 1), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, false, DECODE, 0, 0), MMProcessorKind::NONE);
    // VIT and FRONTEND never own LLM multimodal ingress; a local engine there is an assembly error.
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, VIT, 0, 0), MMProcessorKind::INVALID);
    EXPECT_EQ(resolveMMProcessorKind(true, VIT_SEPARATION_LOCAL, true, FRONTEND, 0, 0), MMProcessorKind::INVALID);
}

TEST(MMProcessorConfigTest, downstreamPipelineStagesDoNotRequireMultimodalIngress) {
    // ownsMultimodalIngress short-circuits on pp_rank, so role_type and vit_separation cannot change this outcome.
    for (int64_t pp_rank : {1, 2}) {
        SCOPED_TRACE(pp_rank);
        const auto decision = resolveAndLogMMProcessorKind(
            true, VIT_SEPARATION_LOCAL, false, PDFUSION, 0, pp_rank, "qwen35_dense", "LocalRpcServer");
        EXPECT_TRUE(decision.ok()) << decision.error;
        EXPECT_EQ(decision.kind, MMProcessorKind::NONE);
    }
}

TEST(MMProcessorConfigTest, firstPipelineStageKeepsMultimodalValidation) {
    for (auto role_type : {PDFUSION, PREFILL}) {
        for (auto separation : {VIT_SEPARATION_LOCAL, VIT_SEPARATION_REMOTE}) {
            const bool requires_local_engine = separation == VIT_SEPARATION_LOCAL;
            const auto valid                 = resolveAndLogMMProcessorKind(
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
