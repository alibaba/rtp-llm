#include "rtp_llm/cpp/models/PrefillForwardEvidence.h"
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

GptModelInputs prefillInputs() {
    GptModelInputs inputs;
    inputs.input_lengths    = torch::tensor({3, 5, 7}, torch::kInt32);
    inputs.prefix_lengths   = torch::tensor({0, 16, 32}, torch::kInt32);
    inputs.sequence_lengths = torch::empty({0}, torch::kInt32);
    inputs.request_id       = torch::tensor({31, 47, 59}, torch::kInt64);
    return inputs;
}

TEST(PrefillForwardEvidenceTest, UsesRealContextIdsWithoutInventingPaddingRequests) {
    const auto event = collectPrefillForwardEvidence(prefillInputs(), 20, 0, 2, 4, true);
    ASSERT_TRUE(event.has_value());
    EXPECT_TRUE(event->error.empty());
    EXPECT_EQ(event->request_ids, (std::vector<int64_t>{31, 47, 59}));
    EXPECT_EQ(event->physical_tokens, 20);
    EXPECT_EQ(event->local_tokens, 5);
    EXPECT_EQ(event->tp_rank, 2);
    EXPECT_EQ(prefillForwardEvidenceJson(*event, 123),
              "{\"event\":\"target_prefill_forward\",\"kind\":\"target_prefill_forward\","
              "\"request_ids\":[31,47,59],\"actual_batch\":3,\"physical_tokens\":20,\"local_tokens\":5,"
              "\"tp_rank\":2,\"boundary\":\"forward_return\",\"gpu_synchronized\":false,\"time_ns\":123}");
}

TEST(PrefillForwardEvidenceTest, MicroBatchUsesOnlyItsSlicedContextIds) {
    auto inputs           = prefillInputs();
    inputs.input_lengths  = inputs.input_lengths.narrow(0, 1, 2);
    inputs.prefix_lengths = inputs.prefix_lengths.narrow(0, 1, 2);
    inputs.request_id     = inputs.request_id.narrow(0, 1, 2);
    const auto event      = collectPrefillForwardEvidence(inputs, 12, 0, 0, 2, true);
    ASSERT_TRUE(event.has_value());
    EXPECT_TRUE(event->error.empty());
    EXPECT_EQ(event->request_ids, (std::vector<int64_t>{47, 59}));
    EXPECT_EQ(event->local_tokens, 6);
}

TEST(PrefillForwardEvidenceTest, NoncontiguousHostIdsRetainTheirOrder) {
    auto inputs       = prefillInputs();
    inputs.request_id = torch::tensor({31, -1, 47, -1, 59, -1}, torch::kInt64).slice(0, 0, 6, 2);
    const auto event  = collectPrefillForwardEvidence(inputs, 15, 0, 0, 4, false);
    ASSERT_TRUE(event.has_value());
    EXPECT_TRUE(event->error.empty());
    EXPECT_EQ(event->request_ids, (std::vector<int64_t>{31, 47, 59}));
    EXPECT_EQ(event->local_tokens, 15);  // Replicated TP input is not SP-sharded.
}

TEST(PrefillForwardEvidenceTest, ExcludesOtherModelRolesAndNonPrefillPhases) {
    EXPECT_FALSE(collectPrefillForwardEvidence(prefillInputs(), 20, 1, 0, 4, true).has_value());
    for (const auto flag : {&GptModelInputs::warmup,
                            &GptModelInputs::skip_run,
                            &GptModelInputs::is_fake_stream,
                            &GptModelInputs::is_target_verify,
                            &GptModelInputs::is_mtp_draft_update}) {
        auto inputs  = prefillInputs();
        inputs.*flag = true;
        EXPECT_FALSE(collectPrefillForwardEvidence(inputs, 20, 0, 0, 4, true).has_value());
    }
    auto inputs             = prefillInputs();
    inputs.sequence_lengths = torch::tensor({12}, torch::kInt32);  // Mixed decode/context.
    EXPECT_FALSE(collectPrefillForwardEvidence(inputs, 20, 0, 0, 4, true).has_value());
    inputs.input_lengths  = torch::tensor({3}, torch::kInt32);
    inputs.prefix_lengths = torch::empty({0}, torch::kInt32);  // Pure decode.
    EXPECT_FALSE(collectPrefillForwardEvidence(inputs, 20, 0, 0, 4, true).has_value());
    inputs.sequence_lengths = torch::empty({0}, torch::kInt32);
    inputs.input_lengths    = torch::empty({0}, torch::kInt32);  // Empty lane.
    EXPECT_FALSE(collectPrefillForwardEvidence(inputs, 20, 0, 0, 4, true).has_value());
}

TEST(PrefillForwardEvidenceTest, PrefillOnlyTargetRetainsItsRealContextIds) {
    auto inputs = prefillInputs();
    // User prefill_only requests skip lm_head and capture hidden states. They
    // still execute a real target prefill; these flags do not denote a draft.
    inputs.skip_lm_head          = true;
    inputs.capture_hidden_states = true;
    const auto event             = collectPrefillForwardEvidence(inputs, 20, 0, 2, 4, true);
    ASSERT_TRUE(event.has_value());
    EXPECT_TRUE(event->error.empty());
    EXPECT_EQ(event->request_ids, (std::vector<int64_t>{31, 47, 59}));
    EXPECT_EQ(event->local_tokens, 5);
    EXPECT_FALSE(collectPrefillForwardEvidence(inputs, 20, 1, 2, 4, true).has_value());
}

TEST(PrefillForwardEvidenceTest, InvalidRequestMetadataProducesExplicitEvidenceError) {
    for (const auto& ids : {torch::Tensor(),
                            torch::tensor({31, 47}, torch::kInt64),
                            torch::tensor({31, 47, 59}, torch::kInt32),
                            torch::tensor({{31, 47, 59}}, torch::kInt64)}) {
        auto inputs       = prefillInputs();
        inputs.request_id = ids;
        const auto event  = collectPrefillForwardEvidence(inputs, 20, 0, 0, 4, true);
        ASSERT_TRUE(event.has_value());
        EXPECT_EQ(event->error, "prefill_request_ids_require_cpu_int64_context_rows");
        EXPECT_TRUE(event->request_ids.empty());
        EXPECT_NE(prefillForwardEvidenceJson(*event, 123).find("\"event\":\"evidence_error\""), std::string::npos);
    }
}

TEST(PrefillForwardEvidenceTest, InvalidContextAndPhysicalGeometryCannotClaimCoverage) {
    auto inputs           = prefillInputs();
    inputs.prefix_lengths = inputs.prefix_lengths.narrow(0, 0, 2);
    EXPECT_EQ(collectPrefillForwardEvidence(inputs, 20, 0, 0, 4, true)->error, "prefill_context_count_mismatch");
    for (int64_t physical : {0, 19}) {
        EXPECT_EQ(collectPrefillForwardEvidence(prefillInputs(), physical, 0, 0, 4, true)->error,
                  "prefill_physical_token_geometry_mismatch");
    }
}

}  // namespace
}  // namespace rtp_llm
