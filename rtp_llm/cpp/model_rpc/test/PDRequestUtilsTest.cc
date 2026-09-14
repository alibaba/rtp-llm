#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"

namespace rtp_llm {
namespace {

GenerateInputPB makeRequest() {
    GenerateInputPB request;
    request.set_request_id(42);
    request.set_request_deadline_ms(12345);
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
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<MultimodalInput>, std::string) override {
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

TEST(PDRequestUtilsTest, TextPreprocessingPreservesRequestAndAppliesSameSpPolicy) {
    const auto request = makeRequest();
    for (bool is_mtp_eagle : {false, true}) {
        auto input = QueryConverter::transQuery(&request);
        ASSERT_TRUE(preprocessForPD(input, nullptr, is_mtp_eagle).ok());
        EXPECT_TRUE(input->generate_config->pd_separation);
        EXPECT_EQ(input->generate_config->force_disable_sp_run, !is_mtp_eagle);
        EXPECT_EQ(input->generate_config->unique_key, "pd-test");
        EXPECT_EQ(input->request_deadline_ms, 12345);
        EXPECT_TRUE(torch::equal(input->input_ids, torch::tensor({10, 1, 20}, torch::kInt32)));
        auto handoff                         = request;
        *handoff.mutable_pd_input_snapshot() = snapshotPDInput(*input);
        EXPECT_TRUE(validatePDInput(*input, handoff).ok());
    }
    EXPECT_EQ(request.token_ids_size(), 3);
}

TEST(PDRequestUtilsTest, BothSidesExpandOnceAndMatchAfterWireRoundTrip) {
    auto                    request = makeMMRequest();
    auto                    decode  = QueryConverter::transQuery(&request);
    auto                    prefill = QueryConverter::transQuery(&request);
    TestMultimodalProcessor decode_processor;
    TestMultimodalProcessor prefill_processor;
    ASSERT_TRUE(preprocessForPD(decode, &decode_processor, false).ok());
    *request.mutable_pd_input_snapshot() = snapshotPDInput(*decode);
    GenerateInputPB handoff;
    ASSERT_TRUE(handoff.ParseFromString(request.SerializeAsString()));
    ASSERT_TRUE(preprocessForPD(prefill, &prefill_processor, false).ok());
    EXPECT_TRUE(validatePDInput(*prefill, handoff).ok());
    EXPECT_EQ(decode_processor.calls, 1);
    EXPECT_EQ(prefill_processor.calls, 1);
    EXPECT_EQ(decode->input_ids.numel(), 5);
    EXPECT_EQ(request.token_ids_size(), 3);
}

TEST(PDRequestUtilsTest, DifferentExpansionLengthsFailBeforeHandoffCanBeUsed) {
    auto                    request = makeMMRequest();
    auto                    decode  = QueryConverter::transQuery(&request);
    auto                    prefill = QueryConverter::transQuery(&request);
    TestMultimodalProcessor processor;
    ASSERT_TRUE(preprocessForPD(decode, &processor, false).ok());
    *request.mutable_pd_input_snapshot() = snapshotPDInput(*decode);
    processor.embedding_length           = 4;
    ASSERT_TRUE(preprocessForPD(prefill, &processor, false).ok());
    const auto result = validatePDInput(*prefill, request);
    EXPECT_EQ(result.code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
    EXPECT_NE(result.ToString().find("input_ids"), std::string::npos);
    EXPECT_NE(result.ToString().find("request_id=42"), std::string::npos);
    EXPECT_NE(result.ToString().find("unique_key=pd-test"), std::string::npos);
}

TEST(PDRequestUtilsTest, SameLengthTokenAndMetadataChangesAreRejected) {
    auto                    request = makeMMRequest();
    TestMultimodalProcessor processor;
    const auto              check_mismatch = [&](auto mutate, const char* field) {
        auto input = QueryConverter::transQuery(&request);
        ASSERT_TRUE(preprocessForPD(input, &processor, false).ok());
        *request.mutable_pd_input_snapshot() = snapshotPDInput(*input);
        mutate(*input);
        auto result = validatePDInput(*input, request);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.ToString().find(field), std::string::npos);
    };
    check_mismatch([](GenerateInput& input) { input.input_ids[0] = 11; }, "input_ids");
    check_mismatch([](GenerateInput& input) { (*input.text_tokens_mask)[0] = 0; }, "text_tokens_mask");
    check_mismatch([](GenerateInput& input) { (*input.mm_locs)[0] = 2; }, "mm_locs");
    check_mismatch([](GenerateInput& input) { (*input.mm_position_ids)[0][0] = 9; }, "mm_position_ids");
    check_mismatch([](GenerateInput& input) { (*input.mm_extra_input)[0][0] = 9; }, "mm_extra_input");
    check_mismatch([](GenerateInput& input) { input.text_tokens_mask.reset(); }, "text_tokens_mask");
    check_mismatch([](GenerateInput& input) { input.mm_position_ids->clear(); }, "mm_position_ids");
    check_mismatch([](GenerateInput& input) { input.mm_locs = input.mm_locs->to(torch::kInt64); }, "mm_locs");
    check_mismatch([](GenerateInput& input) { (*input.multimodal_features)[0] = torch::zeros({3, 4}); },
                   "mm_feature_layouts");
}

TEST(PDRequestUtilsTest, TensorShapeAndPresenceMatterButStorageStridesDoNot) {
    auto request                         = makeRequest();
    auto input                           = QueryConverter::transQuery(&request);
    input->mm_extra_input                = std::vector<torch::Tensor>{torch::tensor({{1, 2}, {3, 4}}, torch::kInt64)};
    *request.mutable_pd_input_snapshot() = snapshotPDInput(*input);
    auto& tensor                         = (*input->mm_extra_input)[0];
    tensor                               = tensor.t().contiguous().t();
    EXPECT_TRUE(validatePDInput(*input, request).ok());
    tensor = tensor.reshape({4});
    EXPECT_FALSE(validatePDInput(*input, request).ok());
    input->mm_extra_input                = std::vector<torch::Tensor>{};
    *request.mutable_pd_input_snapshot() = snapshotPDInput(*input);
    input->mm_extra_input.reset();
    EXPECT_FALSE(validatePDInput(*input, request).ok());
}

TEST(PDRequestUtilsTest, MissingSnapshotAndProcessorOrProcessingFailureAreRejected) {
    auto request = makeMMRequest();
    auto input   = QueryConverter::transQuery(&request);
    EXPECT_FALSE(validatePDInput(*input, request).ok());
    EXPECT_EQ(preprocessForPD(input, nullptr, false).code(), ErrorCode::MM_EMPTY_ENGINE_ERROR);
    TestMultimodalProcessor processor;
    processor.fail    = true;
    const auto result = preprocessForPD(input, &processor, false);
    EXPECT_EQ(result.code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(result.ToString(), "test MM failure");
}

}  // namespace rtp_llm
