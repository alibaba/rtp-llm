#include "gtest/gtest.h"
#include "torch/all.h"

#include <cstring>

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/utils/Exception.h"

namespace rtp_llm {

namespace {

template<typename T>
void appendWireValue(std::vector<uint8_t>& bytes, T value) {
    const auto* data = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(T));
}

}

class PPSerializationTest: public DeviceTestBase {};

TEST_F(PPSerializationTest, EmptyAndShutdownPlansRemainDistinctOnRootAndPeerLanes) {
    for (bool shutdown : {false, true}) {
        for (bool metadata_only : {false, true}) {
            SCOPED_TRACE(testing::Message() << "shutdown=" << shutdown << ", metadata_only=" << metadata_only);
            PPExecutionPlan plan;
            plan.model_input.shutdown = shutdown;
            plan.model_input.skip_run = true;
            plan.model_input.kv_cache_group_tags = {"full", "linear"};
            plan.finished_request_ids = {101, 202};

            const auto received =
                pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, metadata_only));
            EXPECT_EQ(received.model_input.shutdown, shutdown);
            EXPECT_EQ(received.model_input.kv_cache_group_tags, plan.model_input.kv_cache_group_tags);
            if (!metadata_only) {
                EXPECT_TRUE(received.model_input.skip_run);
                EXPECT_FALSE(received.model_input.is_fake_stream);
                EXPECT_EQ(received.finished_request_ids, plan.finished_request_ids);
            }
        }
    }
}

TEST_F(PPSerializationTest, PlanDecodingCopiesTensorsToPinnedCpuWithoutAliasingPayload) {
    PPExecutionPlan plan;
    plan.model_input.combo_tokens = torch::tensor({3, 5}, torch::kInt32).cuda();
    plan.model_input.last_hidden_states = torch::arange(12, torch::kFloat32).reshape({3, 4}).t().cuda();
    plan.model_input.kv_cache_block_id = torch::tensor({1, 2, 3, 4}, torch::kInt32).reshape({1, 2, 2}).cuda();
    plan.model_input.multimodal_features =
        std::vector<torch::Tensor>{torch::arange(8, torch::kFloat32).reshape({2, 4}).to(torch::kFloat16).cuda()};
    plan.model_input.mm_extra_input = std::vector<torch::Tensor>{torch::arange(6, torch::kFloat32).cuda()};
    plan.sampling_plan.request_ids = torch::tensor({101, 202}, torch::kInt64);
    plan.sampling_plan.token_ids = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32).cuda();
    plan.sampling_plan.draft_all_probs = torch::arange(16, torch::kFloat32).reshape({2, 2, 4}).cuda();
    plan.sampling_plan.draft_point_mass_rows = torch::tensor({1}, torch::kInt64).cuda();
    plan.draft_next_position_ids = torch::tensor({5, 8}, torch::kInt32).cuda();

    const auto payload = pp_serialization::serializePlan(plan, false);
    const auto original_payload = payload.clone();
    auto       received = pp_serialization::deserializePlan(payload);
    const auto expect_host_copy = [](const torch::Tensor& actual, const torch::Tensor& source) {
        ASSERT_TRUE(actual.defined());
        EXPECT_TRUE(actual.device().is_cpu());
        EXPECT_TRUE(actual.is_pinned());
        EXPECT_EQ(actual.scalar_type(), source.scalar_type());
        EXPECT_EQ(actual.sizes().vec(), source.sizes().vec());
        EXPECT_TRUE(torch::equal(actual, source.cpu()));
    };
    expect_host_copy(received.model_input.combo_tokens, plan.model_input.combo_tokens);
    expect_host_copy(received.model_input.last_hidden_states, plan.model_input.last_hidden_states);
    expect_host_copy(received.model_input.kv_cache_block_id, plan.model_input.kv_cache_block_id);
    ASSERT_TRUE(received.model_input.multimodal_features.has_value());
    ASSERT_EQ(received.model_input.multimodal_features->size(), 1u);
    expect_host_copy(received.model_input.multimodal_features->front(), plan.model_input.multimodal_features->front());
    ASSERT_TRUE(received.model_input.mm_extra_input.has_value());
    ASSERT_EQ(received.model_input.mm_extra_input->size(), 1u);
    expect_host_copy(received.model_input.mm_extra_input->front(), plan.model_input.mm_extra_input->front());
    expect_host_copy(received.sampling_plan.request_ids, plan.sampling_plan.request_ids);
    expect_host_copy(received.sampling_plan.token_ids, plan.sampling_plan.token_ids);
    expect_host_copy(received.sampling_plan.draft_all_probs, plan.sampling_plan.draft_all_probs);
    expect_host_copy(received.sampling_plan.draft_point_mass_rows, plan.sampling_plan.draft_point_mass_rows);
    expect_host_copy(received.draft_next_position_ids, plan.draft_next_position_ids);

    /** Local execution can replace or modify decoded tensors while forwarding the original CPU payload. */
    received.model_input.combo_tokens.fill_(99);
    received.model_input.multimodal_features->front() = received.model_input.multimodal_features->front().cuda();
    received.model_input.multimodal_features->front().zero_();
    EXPECT_TRUE(torch::equal(payload, original_payload));
    const auto forwarded = pp_serialization::deserializePlan(payload);
    expect_host_copy(forwarded.model_input.combo_tokens, plan.model_input.combo_tokens);
    expect_host_copy(forwarded.model_input.multimodal_features->front(), plan.model_input.multimodal_features->front());
}

TEST_F(PPSerializationTest, ExecutionResultDecodingKeepsCudaPayloadsOnPinnedCpu) {
    PPExecutionResult result;
    result.request_ids = torch::tensor({101}, torch::kInt64);
    result.new_token_ids = torch::tensor({7}, torch::kInt32).reshape({1, 1});
    result.hidden_states = torch::arange(8, torch::kFloat32).reshape({2, 4}).cuda();
    PromptLogitsOutput prompt_logits;
    prompt_logits.topk_token_ids = torch::tensor({{2, 3}}, torch::kInt32).cuda();
    prompt_logits.topk_logprobs = torch::tensor({{-0.2f, -1.7f}}).cuda();
    result.prompt_logits = {prompt_logits};

    const auto received =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    EXPECT_TRUE(received.request_ids.device().is_cpu());
    EXPECT_TRUE(received.new_token_ids.device().is_cpu());
    EXPECT_TRUE(torch::equal(received.request_ids, result.request_ids));
    EXPECT_TRUE(torch::equal(received.new_token_ids, result.new_token_ids));
    EXPECT_TRUE(received.hidden_states.is_cpu());
    EXPECT_TRUE(received.hidden_states.is_pinned());
    EXPECT_EQ(received.hidden_states.scalar_type(), result.hidden_states.scalar_type());
    EXPECT_EQ(received.hidden_states.sizes().vec(), result.hidden_states.sizes().vec());
    EXPECT_TRUE(torch::equal(received.hidden_states, result.hidden_states.cpu()));
    ASSERT_EQ(received.prompt_logits.size(), 1u);
    ASSERT_TRUE(received.prompt_logits.front().has_value());
    EXPECT_TRUE(received.prompt_logits.front()->topk_token_ids.is_cpu());
    EXPECT_TRUE(received.prompt_logits.front()->topk_logprobs.is_cpu());
    EXPECT_TRUE(received.prompt_logits.front()->topk_token_ids.is_pinned());
    EXPECT_TRUE(received.prompt_logits.front()->topk_logprobs.is_pinned());
    EXPECT_TRUE(torch::equal(received.prompt_logits.front()->topk_token_ids, prompt_logits.topk_token_ids.cpu()));
    EXPECT_TRUE(torch::equal(received.prompt_logits.front()->topk_logprobs, prompt_logits.topk_logprobs.cpu()));
}

TEST_F(PPSerializationTest, ResultRoundTripPreservesMixedRequestErrors) {
    PPExecutionResult result;
    result.request_errors = {ErrorInfo::OkStatus(),
                             ErrorInfo(ErrorCode::UNKNOWN_ERROR, "sampling failed\nrequest_id=202"),
                             ErrorInfo::OkStatus(),
                             ErrorInfo(ErrorCode::INVALID_PARAMS, "")};

    const auto received =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    ASSERT_EQ(received.request_errors.size(), result.request_errors.size());
    for (size_t i = 0; i < result.request_errors.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(received.request_errors[i].code(), result.request_errors[i].code());
        EXPECT_EQ(received.request_errors[i].ToString(), result.request_errors[i].ToString());
        EXPECT_EQ(received.request_errors[i].ok(), result.request_errors[i].ok());
    }
}

TEST_F(PPSerializationTest, PrefillOnlyPlanAndPlaceholderTokensSurviveTransport) {
    for (bool prefill_only : {false, true}) {
        PPExecutionPlan plan;
        plan.model_input.skip_lm_head = prefill_only;
        const auto received = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
        EXPECT_EQ(received.model_input.skip_lm_head, prefill_only);
    }

    PPExecutionResult result;
    result.request_ids = torch::tensor({101, 202}, torch::kInt64);
    result.new_token_ids = torch::zeros({2, 1}, torch::kInt32);
    result.new_token_lengths = torch::ones({2}, torch::kInt32);
    result.request_errors.resize(2);
    result.prompt_logits.resize(2);
    const auto received =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    ASSERT_TRUE(received.new_token_ids.defined());
    EXPECT_EQ(received.new_token_ids.sizes().vec(), (std::vector<int64_t>{2, 1}));
    EXPECT_TRUE(torch::equal(received.new_token_ids, result.new_token_ids));
    EXPECT_EQ(received.new_token_ids.scalar_type(), torch::kInt32);
    EXPECT_TRUE(received.new_token_ids.is_cpu());
    EXPECT_TRUE(torch::equal(received.new_token_lengths, result.new_token_lengths));
    EXPECT_TRUE(torch::equal(received.request_ids, result.request_ids));
    EXPECT_EQ(received.request_errors.size(), 2u);
    EXPECT_EQ(received.prompt_logits.size(), 2u);
    EXPECT_FALSE(received.propose_token_ids.defined());
}

TEST_F(PPSerializationTest, ResultWireLayoutOmitsSourceDevice) {
    PPExecutionResult result;
    result.sample_input_us = 1234567890123;
    result.request_ids = torch::tensor({101, 202}, torch::kInt64);

    /** Fix the protocol layout independently of the shared read/write field traversal. */
    std::vector<uint8_t> expected;
    appendWireValue<uint32_t>(expected, 22);
    appendWireValue<int64_t>(expected, 1234567890123);
    appendWireValue<uint8_t>(expected, 1);
    appendWireValue<uint8_t>(expected, 1);
    appendWireValue<int64_t>(expected, 2);
    appendWireValue<int32_t>(expected, static_cast<int32_t>(torch::kInt64));
    appendWireValue<uint64_t>(expected, 16);
    appendWireValue<int64_t>(expected, 101);
    appendWireValue<int64_t>(expected, 202);
    /** The remaining 12 result tensors are undefined; both output lists are empty. */
    expected.insert(expected.end(), 12, uint8_t{0});
    appendWireValue<uint64_t>(expected, 0);
    appendWireValue<uint64_t>(expected, 0);
    const auto payload = torch::tensor(expected, torch::kUInt8);

    EXPECT_TRUE(torch::equal(pp_serialization::serializeExecutionResult(result), payload));
    const auto received = pp_serialization::deserializeExecutionResult(payload);
    EXPECT_EQ(received.sample_input_us, result.sample_input_us);
    EXPECT_TRUE(torch::equal(received.request_ids, result.request_ids));
    EXPECT_FALSE(received.new_token_ids.defined());
    EXPECT_TRUE(received.prompt_logits.empty());
    EXPECT_TRUE(received.request_errors.empty());

    result.request_ids = result.request_ids.cuda();
    EXPECT_TRUE(torch::equal(pp_serialization::serializeExecutionResult(result), payload));
}

TEST_F(PPSerializationTest, MetadataOnlyPlanWireLayoutKeepsShutdownInHeader) {
    PPExecutionPlan plan;
    plan.model_input.shutdown = true;
    plan.model_input.kv_cache_group_tags = {"z", "a"};
    plan.finished_request_ids = {101};

    std::vector<uint8_t> expected;
    appendWireValue<uint32_t>(expected, 22);
    appendWireValue<uint8_t>(expected, 1);
    appendWireValue<uint8_t>(expected, 1);
    appendWireValue<uint64_t>(expected, 2);
    appendWireValue<uint64_t>(expected, 1);
    appendWireValue<uint8_t>(expected, 'z');
    appendWireValue<uint64_t>(expected, 1);
    appendWireValue<uint8_t>(expected, 'a');
    const auto payload = torch::tensor(expected, torch::kUInt8);

    EXPECT_TRUE(torch::equal(pp_serialization::serializePlan(plan, true), payload));
    const auto received = pp_serialization::deserializePlan(payload);
    EXPECT_TRUE(received.model_input.shutdown);
    EXPECT_EQ(received.model_input.kv_cache_group_tags, plan.model_input.kv_cache_group_tags);
    EXPECT_TRUE(received.finished_request_ids.empty());
}

TEST_F(PPSerializationTest, PreviousProtocolVersionIsRejectedForEveryMessageType) {
    const uint32_t old_version = 21;
    for (bool metadata_only : {false, true}) {
        auto payload = pp_serialization::serializePlan(PPExecutionPlan{}, metadata_only);
        std::memcpy(payload.data_ptr(), &old_version, sizeof(old_version));
        EXPECT_THROW(pp_serialization::deserializePlan(payload), RTPException);
    }
    auto result_payload = pp_serialization::serializeExecutionResult(PPExecutionResult{});
    std::memcpy(result_payload.data_ptr(), &old_version, sizeof(old_version));
    EXPECT_THROW(pp_serialization::deserializeExecutionResult(result_payload), RTPException);

    auto metadata = pp_serialization::serializeTensorsMetadata(PPIntermediateTensors{});
    std::memcpy(metadata.data_ptr(), &old_version, sizeof(old_version));
    EXPECT_THROW(pp_serialization::deserializeTensorsMetadata(metadata), RTPException);
}

TEST_F(PPSerializationTest, PlanPreservesScalarEmptyAndOptionalTensorStates) {
    PPExecutionPlan plan;
    plan.model_input.combo_tokens = torch::tensor(7, torch::kInt32);
    plan.model_input.sequence_lengths = torch::empty({0}, torch::kInt32);
    plan.model_input.mm_extra_input = std::vector<torch::Tensor>{};
    plan.model_input.input_embeddings = std::vector<torch::Tensor>{
        torch::Tensor{}, torch::empty({0, 3}, torch::kFloat16), torch::tensor(1.25f)};

    const auto received = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
    EXPECT_EQ(received.model_input.combo_tokens.dim(), 0);
    EXPECT_EQ(received.model_input.combo_tokens.item<int32_t>(), 7);
    EXPECT_FALSE(received.model_input.input_lengths.defined());
    ASSERT_TRUE(received.model_input.sequence_lengths.defined());
    EXPECT_EQ(received.model_input.sequence_lengths.sizes().vec(), (std::vector<int64_t>{0}));
    EXPECT_FALSE(received.model_input.multimodal_features.has_value());
    ASSERT_TRUE(received.model_input.mm_extra_input.has_value());
    EXPECT_TRUE(received.model_input.mm_extra_input->empty());
    ASSERT_TRUE(received.model_input.input_embeddings.has_value());
    const auto& embeddings = *received.model_input.input_embeddings;
    ASSERT_EQ(embeddings.size(), 3u);
    EXPECT_FALSE(embeddings[0].defined());
    ASSERT_TRUE(embeddings[1].defined());
    EXPECT_EQ(embeddings[1].scalar_type(), torch::kFloat16);
    EXPECT_EQ(embeddings[1].sizes().vec(), (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(embeddings[2].dim(), 0);
    EXPECT_FLOAT_EQ(embeddings[2].item<float>(), 1.25f);
}

TEST_F(PPSerializationTest, IntermediateMetadataPreservesReceiveDevicesAndEmptySemantics) {
    PPIntermediateTensors tensors;
    tensors.tensors.emplace("host", torch::empty({2, 3}, torch::kFloat32));
    tensors.tensors.emplace(
        "device", torch::empty({4}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)));
    tensors.tensors.emplace("empty", torch::empty({0}, torch::kInt32));
    tensors.tensors.emplace("undefined", torch::Tensor{});

    const auto received =
        pp_serialization::deserializeTensorsMetadata(pp_serialization::serializeTensorsMetadata(tensors));
    ASSERT_EQ(received.tensors.size(), tensors.tensors.size());
    for (const auto& name : {"host", "device"}) {
        const auto& source = tensors.tensors.at(name);
        const auto& buffer = received.tensors.at(name);
        EXPECT_EQ(buffer.device(), source.device());
        EXPECT_EQ(buffer.scalar_type(), source.scalar_type());
        EXPECT_EQ(buffer.sizes().vec(), source.sizes().vec());
    }
    EXPECT_FALSE(received.tensors.at("empty").defined());
    EXPECT_FALSE(received.tensors.at("undefined").defined());
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
    sampling_plan.random_seeds = {std::nullopt, -7};
    sampling_plan.num_return_sequences = {2, 1};
    sampling_plan.request_ids = torch::tensor({101, 202}, torch::kInt64);
    sampling_plan.token_ids = torch::tensor({{1, 2, 0, 0}, {1, 2, 0, 0}, {3, 4, 5, 0}}, torch::kInt32);
    sampling_plan.top_k = torch::tensor({7, 7, 3}, torch::kInt32);
    sampling_plan.draft_all_probs = torch::rand({2, 3, 8}, torch::TensorOptions().dtype(torch::kFloat32));
    sampling_plan.logits_processor_configs.resize(2);
    auto& processor_config = sampling_plan.logits_processor_configs[0];
    processor_config.grammar_type = "regex";
    processor_config.grammar_value = "ab|cd";
    processor_config.combo_token_size = 2;
    processor_config.banned_combo_token_ids = {{20, 21}, {22, 23}};
    processor_config.end_think_token_ids = {30, 31};
    processor_config.enable_cross_sequence_ban = true;
    processor_config.cross_seq_diverge_start_combo = 3;

    const auto round_trip_plan = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan, false));
    const auto& received_sampling = round_trip_plan.sampling_plan;
    EXPECT_EQ(received_sampling.random_seeds, sampling_plan.random_seeds);
    EXPECT_EQ(received_sampling.num_return_sequences, sampling_plan.num_return_sequences);
    EXPECT_TRUE(torch::equal(received_sampling.request_ids, sampling_plan.request_ids));
    EXPECT_TRUE(torch::equal(received_sampling.token_ids, sampling_plan.token_ids));
    EXPECT_TRUE(torch::equal(received_sampling.top_k, sampling_plan.top_k));
    EXPECT_TRUE(torch::equal(received_sampling.draft_all_probs, sampling_plan.draft_all_probs));
    ASSERT_EQ(received_sampling.logits_processor_configs.size(), 2u);
    const auto& received_config = received_sampling.logits_processor_configs[0];
    EXPECT_EQ(received_config.grammar_type, processor_config.grammar_type);
    EXPECT_EQ(received_config.grammar_value, processor_config.grammar_value);
    EXPECT_EQ(received_config.combo_token_size, processor_config.combo_token_size);
    EXPECT_EQ(received_config.banned_combo_token_ids, processor_config.banned_combo_token_ids);
    EXPECT_EQ(received_config.end_think_token_ids, processor_config.end_think_token_ids);
    EXPECT_EQ(received_config.enable_cross_sequence_ban, processor_config.enable_cross_sequence_ban);
    EXPECT_EQ(received_config.cross_seq_diverge_start_combo, processor_config.cross_seq_diverge_start_combo);

    PPExecutionResult result;
    result.sample_input_us = 1234567890123;
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
    EXPECT_EQ(round_trip_result.sample_input_us, result.sample_input_us);
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
    EXPECT_EQ(round_trip.sample_input_us, 0);
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
