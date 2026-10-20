#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"

namespace rtp_llm {
namespace {

GenerateInputPB makeRequest() {
    GenerateInputPB request;
    request.set_request_id(42);
    request.mutable_generate_config()->set_timeout_ms(12345);
    request.add_token_ids(10);
    request.add_token_ids(1);
    request.add_token_ids(20);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);
    config->set_unique_key("pd-test");
    return request;
}

class TestMultimodalProcessor: public MultimodalProcessor {
public:
    TestMultimodalProcessor(): MultimodalProcessor(py::object{}, MMModelConfig{true, {{1}}, false}, 100) {}

    int  calls            = 0;
    int  embedding_length = 3;
    bool fail             = false;

private:
    ErrorResult<MultimodalOutput>
    MultimodalEmbedding(const std::vector<MultimodalInput>, std::string, int64_t, grpc::ServerContext*) override {
        ++calls;
        if (fail) {
            return {ErrorCode::MM_PROCESS_ERROR, "test MM failure"};
        }
        MultimodalOutput output;
        output.mm_features     = {torch::zeros({embedding_length, 2})};
        output.mm_position_ids = std::vector<torch::Tensor>{torch::arange(embedding_length, torch::kInt64)};
        output.mm_extra_input  = std::vector<torch::Tensor>{torch::tensor({2, 3}, torch::kInt32)};
        return output;
    }
};

GenerateInputPB makeMMRequest() {
    auto request = makeRequest();
    request.add_multimodal_inputs()->set_multimodal_url("test-image");
    return request;
}

}  // namespace

TEST(PDRequestUtilsTest, MasterEnqueuedHandoffKeyIsStablePerRequest) {
    EXPECT_EQ(masterEnqueuedHandoffUniqueKey(42), "master_enqueued_42");
    EXPECT_EQ(masterEnqueuedHandoffUniqueKey(42), masterEnqueuedHandoffUniqueKey(42));
    EXPECT_NE(masterEnqueuedHandoffUniqueKey(42), masterEnqueuedHandoffUniqueKey(43));
}

TEST(PDRequestUtilsTest, PromptLogitsFallsBackEvenWithLargerWireTokenLimit) {
    auto request = makeRequest();
    ASSERT_TRUE(checkPDSupport(request).supported);
    request.mutable_generate_config()->set_return_prompt_logits(true);
    EXPECT_FALSE(checkPDSupport(request).supported);
}

TEST(PDRequestUtilsTest, TextPreprocessingPreservesRequestAndAppliesSameSpPolicy) {
    const auto request = makeRequest();
    for (bool is_mtp_eagle : {false, true}) {
        auto input = QueryConverter::transQuery(&request);
        ASSERT_TRUE(preprocessForPD(input, nullptr, is_mtp_eagle, RoleType::PREFILL).ok());
        EXPECT_TRUE(input->generate_config->pd_separation);
        EXPECT_EQ(input->generate_config->force_disable_sp_run, !is_mtp_eagle);
        EXPECT_EQ(input->generate_config->unique_key, "pd-test");
        EXPECT_EQ(input->request_deadline_ms, 0);
        EXPECT_EQ(input->generate_config->timeout_ms, 12345);
        EXPECT_TRUE(torch::equal(input->input_ids, torch::tensor({10, 1, 20}, torch::kInt32)));
    }
    EXPECT_EQ(request.token_ids_size(), 3);
}

TEST(PDRequestUtilsTest, PrefillExpansionIsDeterministicAcrossWireRoundTrip) {
    const auto              request = makeMMRequest();
    auto                    decode  = QueryConverter::transQuery(&request);
    TestMultimodalProcessor decode_processor;
    TestMultimodalProcessor prefill_processor;
    ASSERT_TRUE(preprocessForPD(decode, &decode_processor, false, RoleType::PREFILL).ok());
    GenerateInputPB handoff;
    ASSERT_TRUE(handoff.ParseFromString(request.SerializeAsString()));
    auto prefill = QueryConverter::transQuery(&handoff);
    ASSERT_TRUE(preprocessForPD(prefill, &prefill_processor, false, RoleType::PREFILL).ok());
    EXPECT_EQ(decode_processor.calls, 1);
    EXPECT_EQ(prefill_processor.calls, 1);
    EXPECT_EQ(decode->input_ids.numel(), 5);
    EXPECT_EQ(request.token_ids_size(), 3);
    EXPECT_EQ(handoff.token_ids_size(), 3);

    // Keep expansion consistency assertions in tests instead of the request path.
    const auto expect_equal = [](const torch::Tensor& a, const torch::Tensor& b) {
        EXPECT_EQ(a.scalar_type(), b.scalar_type());
        EXPECT_EQ(a.sizes(), b.sizes());
        EXPECT_TRUE(torch::equal(a, b));
    };
    expect_equal(decode->input_ids, prefill->input_ids);
    ASSERT_TRUE(decode->text_tokens_mask && prefill->text_tokens_mask);
    expect_equal(*decode->text_tokens_mask, *prefill->text_tokens_mask);
    ASSERT_TRUE(decode->mm_locs && prefill->mm_locs);
    expect_equal(*decode->mm_locs, *prefill->mm_locs);
    ASSERT_TRUE(decode->mm_position_ids && prefill->mm_position_ids);
    ASSERT_TRUE(decode->mm_extra_input && prefill->mm_extra_input);
    ASSERT_TRUE(decode->multimodal_features && prefill->multimodal_features);
    const auto expect_tensors_equal = [&](const auto& a, const auto& b) {
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            expect_equal(a[i], b[i]);
        }
    };
    expect_tensors_equal(*decode->mm_position_ids, *prefill->mm_position_ids);
    expect_tensors_equal(*decode->mm_extra_input, *prefill->mm_extra_input);
    expect_tensors_equal(*decode->multimodal_features, *prefill->multimodal_features);
}

TEST(PDRequestUtilsTest, MissingProcessorOrProcessingFailureAreRejected) {
    auto request = makeMMRequest();
    auto input   = QueryConverter::transQuery(&request);
    EXPECT_EQ(preprocessForPD(input, nullptr, false, RoleType::PREFILL).code(), ErrorCode::MM_EMPTY_ENGINE_ERROR);
    TestMultimodalProcessor processor;
    processor.fail    = true;
    const auto result = preprocessForPD(input, &processor, false, RoleType::PREFILL);
    EXPECT_EQ(result.code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(result.ToString(), "test MM failure");
}

TEST(PDRequestUtilsTest, DecodeConsumesFrontendExpandedIdsWithoutProcessor) {
    auto request = makeMMRequest();
    request.clear_token_ids();
    for (int id : {10, -101, -102, -103, 20}) {
        request.add_token_ids(id);
    }
    auto* span = request.mutable_multimodal_token_layout()->add_spans();
    span->set_offset(1);
    span->set_length(3);
    auto        input        = QueryConverter::transQuery(&request);
    const auto* original_ids = input->input_ids.data_ptr<int32_t>();
    ASSERT_TRUE(preprocessForPD(input, nullptr, false, RoleType::DECODE).ok());
    EXPECT_EQ(input->input_ids.data_ptr<int32_t>(), original_ids);
    EXPECT_TRUE(torch::equal(input->input_ids, torch::tensor({10, -101, -102, -103, 20}, torch::kInt32)));
    ASSERT_TRUE(input->text_tokens_mask);
    EXPECT_TRUE(torch::equal(*input->text_tokens_mask, torch::tensor({1, 0, 0, 0, 1}, torch::kInt32)));
    ASSERT_TRUE(input->mm_locs);
    EXPECT_TRUE(torch::equal(*input->mm_locs, torch::tensor({1}, torch::kInt32)));
    EXPECT_FALSE(input->multimodal_features);
    EXPECT_TRUE(input->generate_config->pd_separation);
    EXPECT_TRUE(input->generate_config->force_disable_sp_run);
}

TEST(PDRequestUtilsTest, DecodeRejectsRawOrMalformedMultimodalIds) {
    auto request = makeMMRequest();
    auto input   = QueryConverter::transQuery(&request);
    EXPECT_EQ(preprocessForPD(input, nullptr, false, RoleType::DECODE).code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
    auto* span = request.mutable_multimodal_token_layout()->add_spans();
    span->set_offset(2);
    span->set_length(2);
    input = QueryConverter::transQuery(&request);
    EXPECT_EQ(preprocessForPD(input, nullptr, false, RoleType::DECODE).code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
    EXPECT_FALSE(input->text_tokens_mask);
}

}  // namespace rtp_llm
