#include "gtest/gtest.h"
#include "torch/all.h"

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

class PPSerializationTest: public DeviceTestBase {};

TEST_F(PPSerializationTest, EmptyAndShutdownPlansRemainDistinctOnRootAndPeerLanes) {
    for (bool shutdown : {false, true}) {
        for (bool metadata_only : {false, true}) {
            SCOPED_TRACE(testing::Message() << "shutdown=" << shutdown << ", metadata_only=" << metadata_only);
            PPExecutionPlan plan;
            plan.shutdown = shutdown;
            plan.model_input.skip_run = true;
            plan.model_input.kv_cache_group_tags = {"full", "linear"};
            plan.finished_request_ids = {101, 202};

            const auto received =
                pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, metadata_only));
            EXPECT_EQ(received.shutdown, shutdown);
            EXPECT_EQ(received.model_input.kv_cache_group_tags, plan.model_input.kv_cache_group_tags);
            if (!metadata_only) {
                EXPECT_TRUE(received.model_input.skip_run);
                EXPECT_FALSE(received.model_input.is_fake_stream);
                EXPECT_EQ(received.finished_request_ids, plan.finished_request_ids);
            }
        }
    }
}

TEST_F(PPSerializationTest, SamplingPlanRoundTripPreservesMixedAndHomogeneousProposalModes) {
    for (int mode = 0; mode < 3; ++mode) {
        PPExecutionPlan plan;
        auto& sampling = plan.sampling_plan;
        sampling.request_ids = torch::tensor({101, 202, 303}, torch::kInt64);
        if (mode != 0) {
            sampling.draft_all_probs = torch::full({3, 2, 4}, .25f);
        }
        if (mode == 2) {
            sampling.draft_point_mass_rows = torch::tensor({0, 2}, torch::kInt64);
            sampling.draft_all_probs[0].zero_();
            sampling.draft_all_probs[2].zero_();
        }
        const auto received = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
        EXPECT_TRUE(torch::equal(received.sampling_plan.request_ids, sampling.request_ids));
        EXPECT_EQ(received.sampling_plan.draft_all_probs.defined(), sampling.draft_all_probs.defined());
        EXPECT_EQ(received.sampling_plan.draft_point_mass_rows.defined(), sampling.draft_point_mass_rows.defined());
        if (sampling.draft_all_probs.defined()) {
            EXPECT_TRUE(torch::equal(received.sampling_plan.draft_all_probs, sampling.draft_all_probs));
        }
        if (sampling.draft_point_mass_rows.defined()) {
            EXPECT_TRUE(torch::equal(received.sampling_plan.draft_point_mass_rows, sampling.draft_point_mass_rows));
        }
    }
}

TEST_F(PPSerializationTest, MixedReturnSequencesPlanAndResultRoundTrip) {
    PPExecutionPlan plan;
    auto& sampling_plan = plan.sampling_plan;
    sampling_plan.num_return_sequences = {2, 1};
    sampling_plan.request_ids = torch::tensor({101, 202}, torch::kInt64);
    sampling_plan.token_ids = torch::tensor({{1, 2, 0, 0}, {1, 2, 0, 0}, {3, 4, 5, 0}}, torch::kInt32);
    sampling_plan.top_k = torch::tensor({7, 7, 3}, torch::kInt32);
    sampling_plan.draft_all_probs = torch::rand({2, 3, 8}, torch::TensorOptions().dtype(torch::kFloat32));
    sampling_plan.logits_processor_configs.resize(2);
    auto& processor_config = sampling_plan.logits_processor_configs[0];
    processor_config.combo_token_size = 2;
    processor_config.banned_combo_token_ids = {{20, 21}};
    processor_config.end_think_token_ids = {30, 31};
    processor_config.enable_cross_sequence_ban = true;
    processor_config.cross_seq_diverge_start_combo = 3;

    const auto round_trip_plan = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
    const auto& received_sampling = round_trip_plan.sampling_plan;
    EXPECT_EQ(received_sampling.num_return_sequences, sampling_plan.num_return_sequences);
    EXPECT_TRUE(torch::equal(received_sampling.request_ids, sampling_plan.request_ids));
    EXPECT_TRUE(torch::equal(received_sampling.token_ids, sampling_plan.token_ids));
    EXPECT_TRUE(torch::equal(received_sampling.top_k, sampling_plan.top_k));
    EXPECT_TRUE(torch::equal(received_sampling.draft_all_probs, sampling_plan.draft_all_probs));
    ASSERT_EQ(received_sampling.logits_processor_configs.size(), 2u);
    const auto& received_config = received_sampling.logits_processor_configs[0];
    EXPECT_EQ(received_config.combo_token_size, processor_config.combo_token_size);
    EXPECT_EQ(received_config.banned_combo_token_ids, processor_config.banned_combo_token_ids);
    EXPECT_EQ(received_config.end_think_token_ids, processor_config.end_think_token_ids);
    EXPECT_EQ(received_config.enable_cross_sequence_ban, processor_config.enable_cross_sequence_ban);
    EXPECT_EQ(received_config.cross_seq_diverge_start_combo, processor_config.cross_seq_diverge_start_combo);

    PPExecutionResult result;
    result.request_ids = sampling_plan.request_ids;
    result.new_token_ids     = torch::tensor({{10}, {11}, {12}}, torch::kInt32);
    result.new_token_lengths = torch::ones({3}, torch::kInt32);
    result.propose_token_ids = torch::tensor({{20, 21}, {22, 23}}, torch::kInt32);
    result.propose_all_probs = torch::rand({2, 2, 8}, torch::TensorOptions().dtype(torch::kFloat32));
    result.propose_hidden_states = torch::arange(12, torch::kFloat32).reshape({2, 6}).to(torch::kBFloat16);
    result.request_errors.resize(2);
    result.prompt_logits.resize(2);

    const auto round_trip_result =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    EXPECT_TRUE(torch::equal(round_trip_result.request_ids, result.request_ids));
    EXPECT_TRUE(torch::equal(round_trip_result.new_token_ids, result.new_token_ids));
    EXPECT_TRUE(torch::equal(round_trip_result.new_token_lengths, result.new_token_lengths));
    EXPECT_TRUE(torch::equal(round_trip_result.propose_token_ids, result.propose_token_ids));
    EXPECT_TRUE(torch::equal(round_trip_result.propose_all_probs, result.propose_all_probs));
    EXPECT_EQ(round_trip_result.propose_hidden_states.scalar_type(), torch::kBFloat16);
    EXPECT_EQ(round_trip_result.propose_hidden_states.sizes().vec(), (std::vector<int64_t>{2, 6}));
    EXPECT_TRUE(torch::equal(round_trip_result.propose_hidden_states, result.propose_hidden_states));
    result.propose_hidden_states = torch::Tensor();
    EXPECT_FALSE(pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result))
                     .propose_hidden_states.defined());
    ASSERT_EQ(round_trip_result.request_errors.size(), 2u);
    EXPECT_TRUE(round_trip_result.request_errors[0].ok());
    EXPECT_TRUE(round_trip_result.request_errors[1].ok());
}

TEST_F(PPSerializationTest, PlanRoundTripPreservesPromptLogitsAndHiddenStatesConfig) {
    PPExecutionPlan plan;
    plan.model_input.custom_output_indexes = torch::tensor({1, 3}, torch::kInt64);
    plan.model_input.kv_cache_group_tags    = {"full", "linear"};
    plan.model_input.need_all_logits        = true;
    plan.model_input.need_all_hidden_states = true;
    plan.output_config.return_hidden_states = true;
    plan.output_config.return_all_hidden_states = true;
    plan.output_config.prompt_logits_requests = {PPPromptLogitsRequest{true, 2, 1, 3, true}, PPPromptLogitsRequest{}};

    const auto round_trip = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
    EXPECT_TRUE(torch::equal(round_trip.model_input.custom_output_indexes, plan.model_input.custom_output_indexes));
    EXPECT_EQ(round_trip.model_input.kv_cache_group_tags, plan.model_input.kv_cache_group_tags);
    EXPECT_TRUE(round_trip.model_input.need_all_logits);
    EXPECT_TRUE(round_trip.model_input.need_all_hidden_states);
    EXPECT_TRUE(round_trip.output_config.return_hidden_states);
    EXPECT_TRUE(round_trip.output_config.return_all_hidden_states);
    ASSERT_EQ(round_trip.output_config.prompt_logits_requests.size(), 2u);
    const auto& request = round_trip.output_config.prompt_logits_requests[0];
    EXPECT_TRUE(request.enabled);
    EXPECT_EQ(request.top_k, 2);
    EXPECT_EQ(request.start, 1);
    EXPECT_EQ(request.end, 3);
    EXPECT_TRUE(request.return_target_logprob);
    EXPECT_FALSE(round_trip.output_config.prompt_logits_requests[1].enabled);
}

TEST_F(PPSerializationTest, ResultRoundTripPreservesPromptLogitsAndHiddenStates) {
    PPExecutionResult result;
    result.request_ids = torch::tensor({301, 302}, torch::kInt64);
    result.new_token_ids = torch::tensor({{10}, {11}}, torch::kInt32);
    result.new_token_lengths = torch::ones({2}, torch::kInt32);
    result.hidden_states = torch::tensor({{5.0f, 6.0f}, {9.0f, 10.0f}});
    result.all_hidden_states = torch::arange(10, torch::kFloat32).reshape({5, 2});
    result.request_errors.resize(2);
    result.prompt_logits.resize(2);
    PromptLogitsOutput prompt_logits;
    prompt_logits.topk_logprobs = torch::tensor({{-0.2f, -1.7f}, {-0.3f, -1.4f}});
    prompt_logits.topk_token_ids = torch::tensor({{2, 3}, {4, 5}}, torch::kInt32);
    prompt_logits.target_logprobs = torch::tensor({-0.3f});
    prompt_logits.start_pos = 1;
    prompt_logits.end_pos = 3;
    result.prompt_logits[0] = prompt_logits;

    const auto round_trip =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    EXPECT_TRUE(torch::equal(round_trip.request_ids, result.request_ids));
    EXPECT_TRUE(torch::equal(round_trip.new_token_ids, result.new_token_ids));
    EXPECT_TRUE(torch::equal(round_trip.hidden_states, result.hidden_states));
    EXPECT_TRUE(torch::equal(round_trip.all_hidden_states, result.all_hidden_states));
    ASSERT_EQ(round_trip.prompt_logits.size(), 2u);
    ASSERT_TRUE(round_trip.prompt_logits[0].has_value());
    EXPECT_FALSE(round_trip.prompt_logits[1].has_value());
    const auto& received = round_trip.prompt_logits[0].value();
    EXPECT_TRUE(torch::equal(received.topk_logprobs, prompt_logits.topk_logprobs));
    EXPECT_TRUE(torch::equal(received.topk_token_ids, prompt_logits.topk_token_ids));
    EXPECT_TRUE(torch::equal(received.target_logprobs, prompt_logits.target_logprobs));
    EXPECT_EQ(received.start_pos, prompt_logits.start_pos);
    EXPECT_EQ(received.end_pos, prompt_logits.end_pos);
}

}
