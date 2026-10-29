#include "gtest/gtest.h"
#include "torch/all.h"

#include <limits>

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/utils/Exception.h"

namespace rtp_llm {

namespace {

constexpr uint8_t  kWireBool        = 1;
constexpr uint8_t  kWireInt32       = 2;
constexpr uint8_t  kWireInt64       = 3;
constexpr uint8_t  kWireFloat16     = 4;
constexpr uint8_t  kWireBFloat16    = 5;
constexpr uint8_t  kWireFloat32     = 6;

template<typename T>
void appendWireValue(std::vector<uint8_t>& bytes, T value) {
    const auto* data = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(T));
}

void appendWireString(std::vector<uint8_t>& bytes, const std::string& value) {
    appendWireValue<uint64_t>(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}

torch::Tensor makeWireTensor(const std::vector<uint8_t>& bytes) {
    return torch::tensor(bytes, torch::kUInt8);
}

torch::Tensor makeIntermediateMetadata(const std::vector<int64_t>& sizes, uint8_t wire_dtype = kWireInt32) {
    std::vector<uint8_t> bytes;
    appendWireValue<uint64_t>(bytes, 1);
    appendWireString(bytes, "hidden_states");
    appendWireValue<uint8_t>(bytes, 1);
    appendWireValue<uint64_t>(bytes, sizes.size());
    for (const auto size : sizes) {
        appendWireValue<int64_t>(bytes, size);
    }
    appendWireValue<uint8_t>(bytes, wire_dtype);
    appendWireValue<uint8_t>(bytes, 0);
    return makeWireTensor(bytes);
}

}  // namespace

class PPSerializationTest: public DeviceTestBase {};

TEST_F(PPSerializationTest, EmptyAndShutdownPlansRemainDistinctWithAndWithoutSampling) {
    for (bool shutdown : {false, true}) {
        for (bool has_sampling : {false, true}) {
            SCOPED_TRACE(testing::Message() << "shutdown=" << shutdown << ", has_sampling=" << has_sampling);
            PPExecutionPlan plan;
            plan.model_input.shutdown            = shutdown;
            plan.model_input.skip_run            = true;
            plan.model_input.kv_cache_group_tags = {"full", "linear"};
            plan.finished_request_ids            = {101, 202};
            if (has_sampling) {
                plan.sampling_plan.request_ids = torch::tensor({101}, torch::kInt64);
            }

            const auto received = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan));
            EXPECT_EQ(received.model_input.shutdown, shutdown);
            EXPECT_EQ(received.model_input.kv_cache_group_tags, plan.model_input.kv_cache_group_tags);
            EXPECT_TRUE(received.model_input.skip_run);
            EXPECT_FALSE(received.model_input.is_fake_stream);
            EXPECT_EQ(received.finished_request_ids, plan.finished_request_ids);
            EXPECT_EQ(received.sampling_plan.request_ids.defined(), has_sampling);
            if (has_sampling) {
                EXPECT_EQ(received.sampling_plan.request_ids.item<int64_t>(), 101);
            }
        }
    }
}

TEST_F(PPSerializationTest, PlanDecodingCopiesTensorsToPinnedCpuWithoutAliasingPayload) {
    PPExecutionPlan plan;
    plan.model_input.combo_tokens = torch::tensor({3, 5}, torch::kInt32).cuda();
    plan.model_input.engram_token_windows =
        torch::tensor({-1, -1, 1, 3, -1, 1, 3, 5}, torch::kInt32).reshape({2, 4}).cuda();
    plan.model_input.mm_features_spans  = torch::tensor({0, 1, 2}, torch::kInt64).reshape({1, 3}).cuda();
    plan.model_input.last_hidden_states = torch::arange(12, torch::kFloat32).reshape({3, 4}).t().cuda();
    plan.model_input.kv_cache_block_id  = torch::tensor({1, 2, 3, 4}, torch::kInt32).reshape({1, 2, 2}).cuda();
    plan.model_input.cache_store_publish_start_tokens = torch::tensor({0, 128}, torch::kInt32);
    plan.model_input.multimodal_features =
        std::vector<torch::Tensor>{torch::arange(8, torch::kFloat32).reshape({2, 4}).to(torch::kFloat16).cuda()};
    plan.model_input.mm_extra_input  = std::vector<torch::Tensor>{torch::arange(6, torch::kFloat32).cuda()};
    plan.sampling_plan.request_ids   = torch::tensor({101, 202}, torch::kInt64);
    plan.sampling_plan.token_ids     = torch::tensor({1, 2, 3, 4}, torch::kInt32).cuda();
    plan.sampling_plan.token_offsets = torch::tensor({0, 2, 4}, torch::kInt64);
    plan.draft_next_position_ids     = torch::tensor({5, 8}, torch::kInt32).cuda();

    const auto payload          = pp_serialization::serializePlan(plan);
    const auto original_payload = payload.clone();
    auto       received         = pp_serialization::deserializePlan(payload);
    const auto expect_host_copy = [](const torch::Tensor& actual, const torch::Tensor& source) {
        ASSERT_TRUE(actual.defined());
        EXPECT_TRUE(actual.device().is_cpu());
        EXPECT_TRUE(actual.is_pinned());
        EXPECT_EQ(actual.scalar_type(), source.scalar_type());
        EXPECT_EQ(actual.sizes().vec(), source.sizes().vec());
        EXPECT_TRUE(torch::equal(actual, source.cpu()));
    };
    expect_host_copy(received.model_input.combo_tokens, plan.model_input.combo_tokens);
    expect_host_copy(received.model_input.engram_token_windows, plan.model_input.engram_token_windows);
    expect_host_copy(received.model_input.mm_features_spans, plan.model_input.mm_features_spans);
    expect_host_copy(received.model_input.last_hidden_states, plan.model_input.last_hidden_states);
    expect_host_copy(received.model_input.kv_cache_block_id, plan.model_input.kv_cache_block_id);
    expect_host_copy(received.model_input.cache_store_publish_start_tokens,
                     plan.model_input.cache_store_publish_start_tokens);
    ASSERT_TRUE(received.model_input.multimodal_features.has_value());
    ASSERT_EQ(received.model_input.multimodal_features->size(), 1u);
    expect_host_copy(received.model_input.multimodal_features->front(), plan.model_input.multimodal_features->front());
    ASSERT_TRUE(received.model_input.mm_extra_input.has_value());
    ASSERT_EQ(received.model_input.mm_extra_input->size(), 1u);
    expect_host_copy(received.model_input.mm_extra_input->front(), plan.model_input.mm_extra_input->front());
    expect_host_copy(received.sampling_plan.request_ids, plan.sampling_plan.request_ids);
    expect_host_copy(received.sampling_plan.token_ids, plan.sampling_plan.token_ids);
    expect_host_copy(received.sampling_plan.token_offsets, plan.sampling_plan.token_offsets);
    expect_host_copy(received.draft_next_position_ids, plan.draft_next_position_ids);

    /** Local execution can replace or modify decoded tensors while forwarding the original CPU payload. */
    received.model_input.combo_tokens.fill_(99);
    received.model_input.engram_token_windows.zero_();
    received.model_input.mm_features_spans.zero_();
    received.model_input.multimodal_features->front() = received.model_input.multimodal_features->front().cuda();
    received.model_input.multimodal_features->front().zero_();
    EXPECT_TRUE(torch::equal(payload, original_payload));
    const auto forwarded = pp_serialization::deserializePlan(payload);
    expect_host_copy(forwarded.model_input.combo_tokens, plan.model_input.combo_tokens);
    expect_host_copy(forwarded.model_input.engram_token_windows, plan.model_input.engram_token_windows);
    expect_host_copy(forwarded.model_input.mm_features_spans, plan.model_input.mm_features_spans);
    expect_host_copy(forwarded.model_input.multimodal_features->front(), plan.model_input.multimodal_features->front());
}

TEST_F(PPSerializationTest, ExecutionResultDecodingOwnsPinnedCpuRequestOutputs) {
    PPExecutionResult result;
    result.request_ids       = torch::tensor({101}, torch::kInt64);
    result.new_token_ids     = torch::tensor({7, 8}, torch::kInt32).reshape({2, 1});
    result.new_token_lengths = torch::ones({2}, torch::kInt32);
    result.request_errors.resize(1);
    result.request_outputs.resize(1);
    result.request_outputs[0].hidden_states = torch::arange(8, torch::kFloat32).reshape({2, 4});
    PromptLogitsOutput prompt_logits;
    prompt_logits.topk_token_ids            = torch::tensor({{2, 3}}, torch::kInt32);
    prompt_logits.topk_logprobs             = torch::tensor({{-0.2f, -1.7f}});
    result.request_outputs[0].prompt_logits = prompt_logits;

    const auto received =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    EXPECT_TRUE(received.request_ids.device().is_cpu());
    EXPECT_TRUE(received.new_token_ids.device().is_cpu());
    EXPECT_TRUE(torch::equal(received.request_ids, result.request_ids));
    EXPECT_TRUE(torch::equal(received.new_token_ids, result.new_token_ids));
    EXPECT_TRUE(received.request_outputs[0].hidden_states.is_cpu());
    EXPECT_TRUE(received.request_outputs[0].hidden_states.is_pinned());
    EXPECT_EQ(received.request_outputs[0].hidden_states.scalar_type(),
              result.request_outputs[0].hidden_states.scalar_type());
    EXPECT_EQ(received.request_outputs[0].hidden_states.sizes().vec(),
              result.request_outputs[0].hidden_states.sizes().vec());
    EXPECT_TRUE(torch::equal(received.request_outputs[0].hidden_states, result.request_outputs[0].hidden_states));
    EXPECT_NE(received.request_outputs[0].hidden_states.data_ptr(), result.request_outputs[0].hidden_states.data_ptr());
    ASSERT_EQ(received.request_outputs.size(), 1u);
    ASSERT_TRUE(received.request_outputs.front().prompt_logits.has_value());
    EXPECT_TRUE(received.request_outputs.front().prompt_logits->topk_token_ids.is_cpu());
    EXPECT_TRUE(received.request_outputs.front().prompt_logits->topk_logprobs.is_cpu());
    EXPECT_TRUE(received.request_outputs.front().prompt_logits->topk_token_ids.is_pinned());
    EXPECT_TRUE(received.request_outputs.front().prompt_logits->topk_logprobs.is_pinned());
    EXPECT_TRUE(
        torch::equal(received.request_outputs.front().prompt_logits->topk_token_ids, prompt_logits.topk_token_ids));
    EXPECT_TRUE(
        torch::equal(received.request_outputs.front().prompt_logits->topk_logprobs, prompt_logits.topk_logprobs));
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

TEST_F(PPSerializationTest, SkippedSamplingOmitsHistoryAndPreservesPlaceholderResults) {
    for (bool fake : {false, true}) {
        SCOPED_TRACE(fake);
        PPExecutionPlan plan;
        plan.model_input.skip_lm_head   = !fake;
        plan.model_input.is_fake_stream = fake;
        plan.sampling_plan.request_ids  = torch::tensor({101, 202}, torch::kInt64);
        const auto received = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan));
        EXPECT_EQ(received.model_input.skip_lm_head, !fake);
        EXPECT_EQ(received.model_input.is_fake_stream, fake);
        EXPECT_TRUE(torch::equal(received.sampling_plan.request_ids, plan.sampling_plan.request_ids));
        EXPECT_FALSE(received.sampling_plan.token_ids.defined());
        EXPECT_FALSE(received.sampling_plan.token_offsets.defined());
    }

    PPExecutionResult result;
    result.request_ids       = torch::tensor({101, 202}, torch::kInt64);
    result.new_token_ids     = torch::zeros({2, 1}, torch::kInt32);
    result.new_token_lengths = torch::ones({2}, torch::kInt32);
    result.request_errors.resize(2);
    result.request_outputs.resize(2);
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
    EXPECT_EQ(received.request_outputs.size(), 2u);
    EXPECT_FALSE(received.propose_token_ids.defined());
}

TEST_F(PPSerializationTest, ResultWireLayoutIsDeviceIndependent) {
    PPExecutionResult result;
    result.sample_input_us = 1234567890123;
    result.request_ids     = torch::tensor({101, 202}, torch::kInt64);
    result.request_outputs.resize(2);
    result.request_errors.resize(2);

    /** Round trips can hide symmetric codec errors; compare against independently specified wire bytes. */
    std::vector<uint8_t> expected;
    appendWireValue<int64_t>(expected, 1234567890123);
    appendWireValue<uint8_t>(expected, 1);
    appendWireValue<uint64_t>(expected, 1);
    appendWireValue<int64_t>(expected, 2);
    appendWireValue<uint8_t>(expected, kWireInt64);
    appendWireValue<uint64_t>(expected, 16);
    appendWireValue<int64_t>(expected, 101);
    appendWireValue<int64_t>(expected, 202);
    /** Empty output slots preserve request indexing. */
    expected.insert(expected.end(), 5, uint8_t{0});
    appendWireValue<uint64_t>(expected, 2);
    expected.insert(expected.end(), 16, uint8_t{0});
    appendWireValue<uint64_t>(expected, 2);
    appendWireValue<int32_t>(expected, static_cast<int32_t>(ErrorCode::NONE_ERROR));
    appendWireValue<int32_t>(expected, static_cast<int32_t>(ErrorCode::NONE_ERROR));
    const auto payload = torch::tensor(expected, torch::kUInt8);

    EXPECT_TRUE(torch::equal(pp_serialization::serializeExecutionResult(result), payload));
    const auto received = pp_serialization::deserializeExecutionResult(payload);
    EXPECT_EQ(received.sample_input_us, result.sample_input_us);
    EXPECT_TRUE(torch::equal(received.request_ids, result.request_ids));
    EXPECT_FALSE(received.new_token_ids.defined());
    EXPECT_EQ(received.request_outputs.size(), 2u);
    EXPECT_EQ(received.request_errors.size(), 2u);

    result.request_ids = result.request_ids.cuda();
    EXPECT_TRUE(torch::equal(pp_serialization::serializeExecutionResult(result), payload));
}

TEST_F(PPSerializationTest, PlanPreservesEveryModelInputTensorAndScalar) {
    const std::vector<std::pair<const char*, torch::Tensor GptModelInputs::*>> tensor_fields = {
        {"engram_token_windows", &GptModelInputs::engram_token_windows},
        {"mm_features_spans", &GptModelInputs::mm_features_spans},
        {"combo_tokens", &GptModelInputs::combo_tokens},
        {"input_lengths", &GptModelInputs::input_lengths},
        {"sequence_lengths", &GptModelInputs::sequence_lengths},
        {"lm_output_indexes", &GptModelInputs::lm_output_indexes},
        {"custom_output_indexes", &GptModelInputs::custom_output_indexes},
        {"lm_output_lengths", &GptModelInputs::lm_output_lengths},
        {"prefix_lengths", &GptModelInputs::prefix_lengths},
        {"cache_store_publish_start_tokens", &GptModelInputs::cache_store_publish_start_tokens},
        {"sequence_lengths_plus_1", &GptModelInputs::sequence_lengths_plus_1},
        {"combo_tokens_type_ids", &GptModelInputs::combo_tokens_type_ids},
        {"combo_position_ids", &GptModelInputs::combo_position_ids},
        {"last_hidden_states", &GptModelInputs::last_hidden_states},
        {"attention_mask", &GptModelInputs::attention_mask},
        {"kv_cache_block_id", &GptModelInputs::kv_cache_block_id},
        {"kv_cache_kernel_block_id", &GptModelInputs::kv_cache_kernel_block_id},
        {"kv_cache_group_types", &GptModelInputs::kv_cache_group_types},
        {"kv_cache_update_mapping", &GptModelInputs::kv_cache_update_mapping},
        {"text_tokens_mask", &GptModelInputs::text_tokens_mask},
        {"mm_features_locs", &GptModelInputs::mm_features_locs},
        {"input_embeddings_locs", &GptModelInputs::input_embeddings_locs},
        {"request_id", &GptModelInputs::request_id},
        {"request_pd_separation", &GptModelInputs::request_pd_separation},
        {"cache_keys", &GptModelInputs::cache_keys},
    };
    PPExecutionPlan plan;
    for (size_t i = 0; i < tensor_fields.size(); ++i) {
        const auto value = static_cast<int64_t>(i + 1);
        plan.model_input.*tensor_fields[i].second = torch::tensor({value, value + 100}, torch::kInt64);
    }
    plan.model_input.kv_cache_group_tags       = {"linear", "full"};
    plan.model_input.kv_block_stride_bytes     = 1024;
    plan.model_input.kv_scale_stride_bytes     = 2048;
    plan.model_input.seq_size_per_block        = 128;
    plan.model_input.kernel_seq_size_per_block = 16;
    plan.model_input.trace_ids                 = {"trace-a", "trace-b"};
    plan.model_input.pp_intermediates = {
        {"hidden", torch::tensor({1.5f, 2.5f})},
        {"empty", torch::empty({0, 4}, torch::kFloat16)},
        {"undefined", torch::Tensor()},
    };
    plan.model_input.multimodal_features = std::vector<torch::Tensor>{torch::tensor({3.5f})};
    plan.model_input.mm_extra_input      = std::vector<torch::Tensor>{torch::tensor({4.5f})};
    plan.model_input.input_embeddings    = std::vector<torch::Tensor>{torch::tensor({5.5f})};
    plan.is_decode                      = true;
    plan.draft_next_position_ids        = torch::tensor({71, 72}, torch::kInt32);
    plan.finished_request_ids           = {81, 82};

    const auto received = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan));
    for (size_t i = 0; i < tensor_fields.size(); ++i) {
        SCOPED_TRACE(tensor_fields[i].first);
        const auto& actual = received.model_input.*tensor_fields[i].second;
        const auto  value  = static_cast<int64_t>(i + 1);
        ASSERT_TRUE(actual.defined());
        EXPECT_EQ(actual.scalar_type(), torch::kInt64);
        EXPECT_EQ(actual.sizes().vec(), (std::vector<int64_t>{2}));
        EXPECT_TRUE(torch::equal(actual, torch::tensor({value, value + 100}, torch::kInt64)));
    }
    EXPECT_EQ(received.model_input.kv_cache_group_tags, (std::vector<std::string>{"linear", "full"}));
    EXPECT_EQ(received.model_input.kv_block_stride_bytes, 1024u);
    EXPECT_EQ(received.model_input.kv_scale_stride_bytes, 2048u);
    EXPECT_EQ(received.model_input.seq_size_per_block, 128u);
    EXPECT_EQ(received.model_input.kernel_seq_size_per_block, 16u);
    EXPECT_EQ(received.model_input.trace_ids, (std::vector<std::string>{"trace-a", "trace-b"}));
    const auto& intermediates = received.model_input.pp_intermediates;
    ASSERT_EQ(intermediates.size(), 3u);
    EXPECT_TRUE(torch::equal(intermediates.at("hidden"), torch::tensor({1.5f, 2.5f})));
    EXPECT_TRUE(intermediates.at("hidden").is_pinned());
    EXPECT_EQ(intermediates.at("empty").sizes().vec(), (std::vector<int64_t>{0, 4}));
    EXPECT_EQ(intermediates.at("empty").scalar_type(), torch::kFloat16);
    EXPECT_FALSE(intermediates.at("undefined").defined());
    ASSERT_TRUE(received.model_input.multimodal_features.has_value());
    ASSERT_EQ(received.model_input.multimodal_features->size(), 1u);
    EXPECT_FLOAT_EQ(received.model_input.multimodal_features->front().item<float>(), 3.5f);
    ASSERT_TRUE(received.model_input.mm_extra_input.has_value());
    ASSERT_EQ(received.model_input.mm_extra_input->size(), 1u);
    EXPECT_FLOAT_EQ(received.model_input.mm_extra_input->front().item<float>(), 4.5f);
    ASSERT_TRUE(received.model_input.input_embeddings.has_value());
    ASSERT_EQ(received.model_input.input_embeddings->size(), 1u);
    EXPECT_FLOAT_EQ(received.model_input.input_embeddings->front().item<float>(), 5.5f);
    EXPECT_TRUE(received.is_decode);
    EXPECT_TRUE(torch::equal(received.draft_next_position_ids, torch::tensor({71, 72}, torch::kInt32)));
    EXPECT_EQ(received.finished_request_ids, (std::vector<RequestId>{81, 82}));
}

TEST_F(PPSerializationTest, PlanExecutionFlagsAreIndependent) {
    const std::vector<std::pair<const char*, bool GptModelInputs::*>> flags = {
        {"pd_separation", &GptModelInputs::pd_separation},
        {"decode_entrance", &GptModelInputs::decode_entrance},
        {"use_opaque_kv_cache_store", &GptModelInputs::use_opaque_kv_cache_store},
        {"need_all_logits", &GptModelInputs::need_all_logits},
        {"need_all_hidden_states", &GptModelInputs::need_all_hidden_states},
        {"skip_lm_head", &GptModelInputs::skip_lm_head},
        {"capture_hidden_states", &GptModelInputs::capture_hidden_states},
        {"need_moe_gating", &GptModelInputs::need_moe_gating},
        {"warmup", &GptModelInputs::warmup},
        {"skip_run", &GptModelInputs::skip_run},
        {"shutdown", &GptModelInputs::shutdown},
        {"is_fake_stream", &GptModelInputs::is_fake_stream},
        {"is_target_verify", &GptModelInputs::is_target_verify},
    };
    for (const auto& [name, selected] : flags) {
        SCOPED_TRACE(name);
        PPExecutionPlan plan;
        plan.model_input.*selected = true;
        const auto received = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan));
        for (const auto& [field_name, field] : flags) {
            SCOPED_TRACE(field_name);
            EXPECT_EQ(received.model_input.*field, field == selected);
        }
    }
}

TEST_F(PPSerializationTest, MalformedContainerCountIsRejected) {
    std::vector<uint8_t> plan_bytes;
    /** The first sixteen model input tensors are undefined; the next field is the group-tag count. */
    plan_bytes.insert(plan_bytes.end(), 16, uint8_t{0});
    appendWireValue<uint64_t>(plan_bytes, std::numeric_limits<uint64_t>::max());
    EXPECT_THROW(pp_serialization::deserializePlan(makeWireTensor(plan_bytes)), RTPException);
}

TEST_F(PPSerializationTest, MalformedIntermediateMetadataIsRejectedBeforeAllocation) {
    PPIntermediateTensors backing_buffers;

    EXPECT_THROW(pp_serialization::deserializeTensorsMetadata(makeIntermediateMetadata({1}, /*wire_dtype=*/255),
                                                              backing_buffers),
                 RTPException);
    EXPECT_TRUE(backing_buffers.tensors.empty());

    EXPECT_THROW(pp_serialization::deserializeTensorsMetadata(
                     makeIntermediateMetadata({std::numeric_limits<int64_t>::max(), 3}), backing_buffers),
                 RTPException);
    EXPECT_TRUE(backing_buffers.tensors.empty());
}

TEST_F(PPSerializationTest, TensorDTypesHaveStableWireValues) {
    const std::vector<std::pair<torch::ScalarType, uint8_t>> cases = {{torch::kBool, kWireBool},
                                                                      {torch::kInt32, kWireInt32},
                                                                      {torch::kInt64, kWireInt64},
                                                                      {torch::kFloat16, kWireFloat16},
                                                                      {torch::kBFloat16, kWireBFloat16},
                                                                      {torch::kFloat32, kWireFloat32}};
    for (const auto& [dtype, wire_dtype] : cases) {
        SCOPED_TRACE(dtype);
        PPIntermediateTensors tensors;
        tensors.tensors.emplace("hidden_states", torch::empty({1}, torch::TensorOptions().dtype(dtype)));
        const auto metadata = pp_serialization::serializeTensorsMetadata(tensors);
        ASSERT_GE(metadata.numel(), 2);
        const auto* bytes = metadata.data_ptr<uint8_t>();
        EXPECT_EQ(bytes[metadata.numel() - 2], wire_dtype);

        PPIntermediateTensors backing_buffers;
        const auto            received = pp_serialization::deserializeTensorsMetadata(metadata, backing_buffers);
        EXPECT_EQ(received.tensors.at("hidden_states").scalar_type(), dtype);
    }
}

TEST_F(PPSerializationTest, WriterRejectsTensorDTypeOutsidePPProtocol) {
    PPIntermediateTensors tensors;
    tensors.tensors.emplace("hidden_states", torch::empty({1}, torch::kFloat64));
    EXPECT_THROW(pp_serialization::serializeTensorsMetadata(tensors), RTPException);
}

TEST_F(PPSerializationTest, PlanPreservesScalarEmptyAndOptionalTensorStates) {
    PPExecutionPlan plan;
    plan.model_input.combo_tokens         = torch::tensor(7, torch::kInt32);
    plan.model_input.sequence_lengths     = torch::empty({0}, torch::kInt32);
    plan.model_input.engram_token_windows = torch::empty({0, 4}, torch::kInt32);
    plan.model_input.mm_features_spans    = torch::empty({0, 3}, torch::kInt64);
    plan.model_input.mm_extra_input       = std::vector<torch::Tensor>{};
    plan.model_input.input_embeddings =
        std::vector<torch::Tensor>{torch::Tensor{}, torch::empty({0, 3}, torch::kFloat16), torch::tensor(1.25f)};

    const auto received = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan));
    EXPECT_EQ(received.model_input.combo_tokens.dim(), 0);
    EXPECT_EQ(received.model_input.combo_tokens.item<int32_t>(), 7);
    EXPECT_FALSE(received.model_input.input_lengths.defined());
    EXPECT_FALSE(received.model_input.combo_position_ids.defined());
    ASSERT_TRUE(received.model_input.engram_token_windows.defined());
    EXPECT_EQ(received.model_input.engram_token_windows.sizes().vec(), (std::vector<int64_t>{0, 4}));
    EXPECT_EQ(received.model_input.engram_token_windows.scalar_type(), torch::kInt32);
    ASSERT_TRUE(received.model_input.mm_features_spans.defined());
    EXPECT_EQ(received.model_input.mm_features_spans.sizes().vec(), (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(received.model_input.mm_features_spans.scalar_type(), torch::kInt64);
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
    tensors.tensors.emplace("device",
                            torch::empty({4}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)));
    tensors.tensors.emplace("scalar", torch::tensor(1.0f));
    tensors.tensors.emplace("empty", torch::empty({0}, torch::kInt32));
    tensors.tensors.emplace("undefined", torch::Tensor{});

    PPIntermediateTensors backing_buffers;
    const auto            received = pp_serialization::deserializeTensorsMetadata(
        pp_serialization::serializeTensorsMetadata(tensors), backing_buffers);
    ASSERT_EQ(received.tensors.size(), tensors.tensors.size());
    for (const auto& name : {"host", "device", "scalar"}) {
        const auto& source = tensors.tensors.at(name);
        const auto& buffer = received.tensors.at(name);
        EXPECT_EQ(buffer.device(), source.device());
        EXPECT_EQ(buffer.scalar_type(), source.scalar_type());
        EXPECT_EQ(buffer.sizes().vec(), source.sizes().vec());
        EXPECT_EQ(buffer.data_ptr(), backing_buffers.tensors.at(name).data_ptr());
    }
    EXPECT_FALSE(received.tensors.at("empty").defined());
    EXPECT_FALSE(received.tensors.at("undefined").defined());
    EXPECT_EQ(backing_buffers.tensors.count("empty"), 0u);
    EXPECT_EQ(backing_buffers.tensors.count("undefined"), 0u);
}

TEST_F(PPSerializationTest, IntermediateReceiveStorageGrowsOnlyWhenCapacityIsInsufficient) {
    for (const auto device : {torch::kCPU, torch::kCUDA}) {
        SCOPED_TRACE(device);
        const auto            options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        PPIntermediateTensors backing_buffers;
        const auto            decode = [&](int64_t rows) {
            PPIntermediateTensors tensors;
            tensors.tensors.emplace("hidden_states", torch::empty({rows, 4}, options));
            tensors.tensors.emplace("aux_hidden_states", torch::empty({rows, 8}, options));
            return pp_serialization::deserializeTensorsMetadata(pp_serialization::serializeTensorsMetadata(tensors),
                                                                backing_buffers);
        };

        auto initial = decode(4);
        for (auto& entry : initial.tensors) {
            entry.second.fill_(-1);
        }
        auto smaller = decode(2);
        for (const auto& name : {"hidden_states", "aux_hidden_states"}) {
            const auto& original = initial.tensors.at(name);
            const auto& view     = smaller.tensors.at(name);
            EXPECT_EQ(view.sizes().vec(), (std::vector<int64_t>{2, original.size(1)}));
            EXPECT_EQ(view.data_ptr(), original.data_ptr());
            EXPECT_EQ(view.data_ptr(), backing_buffers.tensors.at(name).data_ptr());
            EXPECT_EQ(backing_buffers.tensors.at(name).numel(), original.numel());
            view.fill_(7);
            EXPECT_TRUE(torch::equal(original.narrow(0, 0, 2), torch::full_like(view, 7)));
            EXPECT_TRUE(torch::equal(original.narrow(0, 2, 2), torch::full_like(view, -1)));
        }

        auto grown = decode(9);
        for (const auto& name : {"hidden_states", "aux_hidden_states"}) {
            const auto& view = grown.tensors.at(name);
            EXPECT_EQ(view.sizes().vec(), (std::vector<int64_t>{9, initial.tensors.at(name).size(1)}));
            EXPECT_NE(view.data_ptr(), initial.tensors.at(name).data_ptr());
            EXPECT_EQ(view.data_ptr(), backing_buffers.tensors.at(name).data_ptr());
            EXPECT_GE(backing_buffers.tensors.at(name).numel(), view.numel());
        }
        for (const int64_t rows : {3, 9}) {
            SCOPED_TRACE(rows);
            const auto received = decode(rows);
            for (const auto& name : {"hidden_states", "aux_hidden_states"}) {
                const auto& view = received.tensors.at(name);
                EXPECT_EQ(view.sizes().vec(), (std::vector<int64_t>{rows, grown.tensors.at(name).size(1)}));
                EXPECT_EQ(view.data_ptr(), grown.tensors.at(name).data_ptr());
                EXPECT_EQ(view.storage().nbytes(), grown.tensors.at(name).storage().nbytes());
            }
        }
    }
}

TEST_F(PPSerializationTest, HistoryDeltaPayloadSizeDoesNotGrowWithSequenceLength) {
    for (int delta_length : {0, 1, 4}) {
        SCOPED_TRACE(delta_length);
        PPExecutionPlan plan;
        auto&           sampling = plan.sampling_plan;
        sampling.request_ids     = torch::arange(64, torch::kInt64);
        sampling.num_return_sequences.assign(64, 1);
        sampling.token_ids        = torch::ones({64 * delta_length}, torch::kInt32);
        sampling.token_offsets    = torch::arange(65, torch::kInt64) * delta_length;
        sampling.sequence_lengths = torch::full({64}, 1024, torch::kInt32);
        const auto short_payload  = pp_serialization::serializePlan(plan);
        sampling.sequence_lengths.fill_(128 * 1024);
        const auto long_payload = pp_serialization::serializePlan(plan);
        EXPECT_EQ(short_payload.numel(), long_payload.numel());
        const auto decoded = pp_serialization::deserializePlan(long_payload);
        EXPECT_TRUE(decoded.sampling_plan.token_ids.defined());
        EXPECT_TRUE(torch::equal(decoded.sampling_plan.token_ids, sampling.token_ids));
        EXPECT_TRUE(torch::equal(decoded.sampling_plan.token_offsets, sampling.token_offsets));
        EXPECT_TRUE(torch::equal(decoded.sampling_plan.sequence_lengths, sampling.sequence_lengths));
    }
}

TEST_F(PPSerializationTest, MixedReturnSequencesPlanAndResultRoundTrip) {
    PPExecutionPlan plan;
    auto&           sampling_plan      = plan.sampling_plan;
    sampling_plan.random_seeds         = {std::nullopt, -7};
    sampling_plan.num_return_sequences = {2, 1};
    sampling_plan.request_ids          = torch::tensor({101, 202}, torch::kInt64);
    sampling_plan.token_ids            = torch::tensor({1, 2, 1, 2, 3, 4, 5}, torch::kInt32);
    sampling_plan.token_offsets        = torch::tensor({0, 2, 4, 7}, torch::kInt64);
    sampling_plan.top_k                = torch::tensor({7, 7, 3}, torch::kInt32);
    sampling_plan.logits_processor_configs.resize(2);
    auto& processor_config                         = sampling_plan.logits_processor_configs[0];
    processor_config.grammar_type                  = "regex";
    processor_config.grammar_value                 = "ab|cd";
    processor_config.combo_token_size              = 2;
    processor_config.banned_combo_token_ids        = {{20, 21}, {22, 23}};
    processor_config.end_think_token_ids           = {30, 31};
    processor_config.enable_cross_sequence_ban     = true;
    processor_config.cross_seq_diverge_start_combo = 3;

    const auto  round_trip_plan   = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan));
    const auto& received_sampling = round_trip_plan.sampling_plan;
    EXPECT_EQ(received_sampling.random_seeds, sampling_plan.random_seeds);
    EXPECT_EQ(received_sampling.num_return_sequences, sampling_plan.num_return_sequences);
    EXPECT_TRUE(torch::equal(received_sampling.request_ids, sampling_plan.request_ids));
    EXPECT_TRUE(torch::equal(received_sampling.token_ids, sampling_plan.token_ids));
    EXPECT_TRUE(torch::equal(received_sampling.token_offsets, sampling_plan.token_offsets));
    EXPECT_TRUE(torch::equal(received_sampling.top_k, sampling_plan.top_k));
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
    result.sample_input_us   = 1234567890123;
    result.request_ids       = sampling_plan.request_ids;
    result.new_token_ids     = torch::tensor({{10}, {11}, {12}}, torch::kInt32);
    result.new_token_lengths = torch::ones({3}, torch::kInt32);
    result.request_errors.resize(2);
    result.request_outputs.resize(2);

    const auto round_trip_result =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    EXPECT_EQ(round_trip_result.sample_input_us, result.sample_input_us);
    EXPECT_TRUE(torch::equal(round_trip_result.request_ids, result.request_ids));
    EXPECT_TRUE(torch::equal(round_trip_result.new_token_ids, result.new_token_ids));
    EXPECT_TRUE(torch::equal(round_trip_result.new_token_lengths, result.new_token_lengths));
    EXPECT_FALSE(round_trip_result.propose_token_ids.defined());
    EXPECT_FALSE(round_trip_result.propose_all_probs.defined());
    EXPECT_FALSE(round_trip_result.propose_hidden_states.defined());
    ASSERT_EQ(round_trip_result.request_errors.size(), 2u);
    EXPECT_TRUE(round_trip_result.request_errors[0].ok());
    EXPECT_TRUE(round_trip_result.request_errors[1].ok());
}

TEST_F(PPSerializationTest, NextVerificationCandidatesRoundTripWithOptionalMtpHandoff) {
    for (bool mtp_handoff : {false, true}) {
        SCOPED_TRACE(mtp_handoff);
        PPExecutionResult result;
        result.request_ids       = torch::tensor({101, 202}, torch::kInt64);
        result.new_token_ids     = torch::tensor({{10}, {11}}, torch::kInt32);
        result.new_token_lengths = torch::ones({2}, torch::kInt32);
        result.request_errors.resize(2);
        result.request_outputs.resize(2);
        if (mtp_handoff) {
            /** MTP/EAGLE PD prefill carries one candidate and the state needed by a non-PP D. */
            result.propose_token_ids          = torch::tensor({{20}, {22}}, torch::kInt32);
            result.propose_all_probs          = torch::zeros({2, 1, 8}, torch::kFloat32);
            result.propose_all_probs[0][0][2] = 1;
            result.propose_all_probs[1][0][3] = 1;
            result.propose_hidden_states      = torch::arange(12, torch::kFloat32).reshape({2, 6}).to(torch::kBFloat16);
        } else {
            result.propose_token_ids = torch::tensor({{20, 21}, {22, 23}}, torch::kInt32);
        }

        const auto received =
            pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
        EXPECT_TRUE(torch::equal(received.propose_token_ids, result.propose_token_ids));
        ASSERT_EQ(received.propose_all_probs.defined(), mtp_handoff);
        ASSERT_EQ(received.propose_hidden_states.defined(), mtp_handoff);
        if (mtp_handoff) {
            EXPECT_TRUE(torch::equal(received.propose_all_probs, result.propose_all_probs));
            EXPECT_EQ(received.propose_all_probs.scalar_type(), torch::kFloat32);
            EXPECT_EQ(received.propose_all_probs.sizes().vec(), (std::vector<int64_t>{2, 1, 8}));
            EXPECT_EQ(received.propose_hidden_states.scalar_type(), torch::kBFloat16);
            EXPECT_EQ(received.propose_hidden_states.sizes().vec(), (std::vector<int64_t>{2, 6}));
            EXPECT_TRUE(torch::equal(received.propose_hidden_states, result.propose_hidden_states));
        }
    }
}

TEST_F(PPSerializationTest, PlanRoundTripPreservesBatchAndRequestOutputRequirements) {
    PPExecutionPlan plan;
    plan.model_input.custom_output_indexes      = torch::tensor({1, 3}, torch::kInt64);
    plan.model_input.kv_cache_group_tags        = {"full", "linear"};
    plan.model_input.need_all_logits            = true;
    plan.model_input.need_all_hidden_states     = true;
    plan.output_config.return_logits            = true;
    plan.output_config.return_softmax_probs     = true;
    plan.output_config.return_cum_log_probs     = true;
    plan.output_config.return_all_probs         = ReturnAllProbsMode::ORIGINAL;
    plan.output_config.return_hidden_states     = true;
    plan.output_config.return_all_hidden_states = true;
    plan.output_config.requests.resize(2);
    plan.output_config.requests[0].return_logits            = true;
    plan.output_config.requests[0].return_softmax_probs     = true;
    plan.output_config.requests[0].return_cum_log_probs     = true;
    plan.output_config.requests[0].return_all_probs         = true;
    plan.output_config.requests[0].return_hidden_states     = true;
    plan.output_config.requests[0].return_all_hidden_states = true;
    plan.output_config.requests[0].prompt_logits            = {true, 2, 1, 3, true};

    const auto round_trip = pp_serialization::deserializePlan(pp_serialization::serializePlan(plan));
    EXPECT_TRUE(torch::equal(round_trip.model_input.custom_output_indexes, plan.model_input.custom_output_indexes));
    EXPECT_EQ(round_trip.model_input.kv_cache_group_tags, plan.model_input.kv_cache_group_tags);
    EXPECT_TRUE(round_trip.model_input.need_all_logits);
    EXPECT_TRUE(round_trip.model_input.need_all_hidden_states);
    EXPECT_TRUE(round_trip.output_config.return_logits);
    EXPECT_TRUE(round_trip.output_config.return_softmax_probs);
    EXPECT_TRUE(round_trip.output_config.return_cum_log_probs);
    EXPECT_EQ(round_trip.output_config.return_all_probs, ReturnAllProbsMode::ORIGINAL);
    EXPECT_FALSE(round_trip.output_config.calculate_loss);
    EXPECT_TRUE(round_trip.output_config.return_hidden_states);
    EXPECT_TRUE(round_trip.output_config.return_all_hidden_states);
    ASSERT_EQ(round_trip.output_config.requests.size(), 2u);
    for (auto flag : {&PPRequestOutputConfig::return_logits,
                      &PPRequestOutputConfig::return_softmax_probs,
                      &PPRequestOutputConfig::return_cum_log_probs,
                      &PPRequestOutputConfig::return_all_probs,
                      &PPRequestOutputConfig::return_hidden_states,
                      &PPRequestOutputConfig::return_all_hidden_states}) {
        EXPECT_TRUE(round_trip.output_config.requests[0].*flag);
        EXPECT_FALSE(round_trip.output_config.requests[1].*flag);
    }
    EXPECT_FALSE(round_trip.output_config.requests[0].calculate_loss);
    EXPECT_FALSE(round_trip.output_config.requests[1].calculate_loss);
    const auto& request = round_trip.output_config.requests[0].prompt_logits;
    EXPECT_TRUE(request.enabled);
    EXPECT_EQ(request.top_k, 2);
    EXPECT_EQ(request.start, 1);
    EXPECT_EQ(request.end, 3);
    EXPECT_TRUE(request.return_target_logprob);
    EXPECT_FALSE(round_trip.output_config.requests[1].prompt_logits.enabled);
}

TEST_F(PPSerializationTest, ResultRoundTripPreservesPromptLogitsAndHiddenStates) {
    PPExecutionResult result;
    result.request_ids       = torch::tensor({301, 302}, torch::kInt64);
    result.new_token_ids     = torch::tensor({{10}, {11}}, torch::kInt32);
    result.new_token_lengths = torch::ones({2}, torch::kInt32);
    result.request_outputs.resize(2);
    result.request_outputs[0].hidden_states     = torch::tensor({{5.0f, 6.0f}});
    result.request_outputs[0].all_hidden_states = torch::arange(6, torch::kFloat32).reshape({3, 2});
    result.request_outputs[1].hidden_states     = torch::tensor({{9.0f, 10.0f}});
    result.request_errors.resize(2);
    PromptLogitsOutput prompt_logits;
    prompt_logits.topk_logprobs             = torch::tensor({{-0.2f, -1.7f}, {-0.3f, -1.4f}});
    prompt_logits.topk_token_ids            = torch::tensor({{2, 3}, {4, 5}}, torch::kInt32);
    prompt_logits.target_logprobs           = torch::tensor({-0.3f});
    prompt_logits.start_pos                 = 1;
    prompt_logits.end_pos                   = 3;
    result.request_outputs[0].prompt_logits = prompt_logits;

    const auto round_trip =
        pp_serialization::deserializeExecutionResult(pp_serialization::serializeExecutionResult(result));
    EXPECT_EQ(round_trip.sample_input_us, 0);
    EXPECT_TRUE(torch::equal(round_trip.request_ids, result.request_ids));
    EXPECT_TRUE(torch::equal(round_trip.new_token_ids, result.new_token_ids));
    EXPECT_TRUE(torch::equal(round_trip.request_outputs[0].hidden_states, result.request_outputs[0].hidden_states));
    EXPECT_TRUE(
        torch::equal(round_trip.request_outputs[0].all_hidden_states, result.request_outputs[0].all_hidden_states));
    EXPECT_TRUE(torch::equal(round_trip.request_outputs[1].hidden_states, result.request_outputs[1].hidden_states));
    EXPECT_FALSE(round_trip.request_outputs[1].all_hidden_states.defined());
    ASSERT_EQ(round_trip.request_outputs.size(), 2u);
    ASSERT_TRUE(round_trip.request_outputs[0].prompt_logits.has_value());
    EXPECT_FALSE(round_trip.request_outputs[1].prompt_logits.has_value());
    const auto& received = round_trip.request_outputs[0].prompt_logits.value();
    EXPECT_TRUE(torch::equal(received.topk_logprobs, prompt_logits.topk_logprobs));
    EXPECT_TRUE(torch::equal(received.topk_token_ids, prompt_logits.topk_token_ids));
    EXPECT_TRUE(torch::equal(received.target_logprobs, prompt_logits.target_logprobs));
    EXPECT_EQ(received.start_pos, prompt_logits.start_pos);
    EXPECT_EQ(received.end_pos, prompt_logits.end_pos);
}

TEST_F(PPSerializationTest, RequestViewsSerializeOnlySelectedElementsAndKeepEmptySlots) {
    PPExecutionResult result;
    result.request_ids       = torch::tensor({101, 202, 303}, torch::kInt64);
    result.new_token_ids     = torch::tensor({10, 11, 12, 13}, torch::kInt32).reshape({4, 1});
    result.new_token_lengths = torch::ones({4}, torch::kInt32);
    result.request_errors.resize(3);
    result.request_outputs.resize(3);
    const auto batch_logits   = torch::arange(512, torch::kFloat32).reshape({4, 128});
    const auto batch_hidden   = torch::arange(64, torch::kFloat32).reshape({4, 16});
    auto&      output         = result.request_outputs[1];
    output.logits             = batch_logits.narrow(0, 1, 2);
    output.hidden_states      = batch_hidden.narrow(0, 1, 2);
    const auto sparse_payload = pp_serialization::serializeExecutionResult(result);
    const auto received       = pp_serialization::deserializeExecutionResult(sparse_payload);
    ASSERT_EQ(received.request_outputs.size(), 3u);
    EXPECT_TRUE(torch::equal(received.request_ids, result.request_ids));
    EXPECT_TRUE(torch::equal(received.request_outputs[1].logits, batch_logits.narrow(0, 1, 2)));
    EXPECT_TRUE(torch::equal(received.request_outputs[1].hidden_states, batch_hidden.narrow(0, 1, 2)));
    EXPECT_EQ(received.request_outputs[1].logits.storage().nbytes(), 2 * 128 * sizeof(float));
    EXPECT_EQ(received.request_outputs[1].hidden_states.storage().nbytes(), 2 * 16 * sizeof(float));
    for (auto index : {0, 2}) {
        EXPECT_FALSE(received.request_outputs[index].logits.defined());
        EXPECT_FALSE(received.request_outputs[index].hidden_states.defined());
    }

    output.logits              = output.logits.clone();
    output.hidden_states       = output.hidden_states.clone();
    const auto compact_payload = pp_serialization::serializeExecutionResult(result);
    EXPECT_TRUE(torch::equal(sparse_payload, compact_payload));
}
}  // namespace rtp_llm
