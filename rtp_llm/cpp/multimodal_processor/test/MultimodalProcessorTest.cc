#include <memory>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/multimodal_processor/test/FakeMultimodalProcessor.h"

using namespace std;

namespace rtp_llm {

class MultimodalProcessorTest: public DeviceTestBase {};

TEST_F(MultimodalProcessorTest, testSimple) {
    FakeMultimodalProcessor        processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    std::shared_ptr<GenerateInput> input     = std::make_shared<GenerateInput>();
    input->input_ids                         = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    auto mm_inputs                           = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), true);

    auto input_ids = input->input_ids.data_ptr<int32_t>();
    EXPECT_EQ(input->input_ids.numel(), 6);
    EXPECT_EQ(input_ids[0], 0);
    EXPECT_EQ(input_ids[4], 2);
    EXPECT_EQ(input_ids[5], 3);

    EXPECT_TRUE(input->text_tokens_mask);
    auto text_tokens_mask = input->text_tokens_mask.value().data_ptr<int32_t>();
    EXPECT_EQ(input->text_tokens_mask.value().numel(), 6);
    EXPECT_EQ(text_tokens_mask[0], 1);
    EXPECT_EQ(text_tokens_mask[1], 0);
    EXPECT_EQ(text_tokens_mask[2], 0);
    EXPECT_EQ(text_tokens_mask[3], 0);
    EXPECT_EQ(text_tokens_mask[4], 1);
    EXPECT_EQ(text_tokens_mask[5], 1);

    EXPECT_TRUE(input->mm_locs);
    auto locs = input->mm_locs.value().data_ptr<int32_t>();
    EXPECT_EQ(input->mm_locs.value().numel(), 1);
    EXPECT_EQ(locs[0], 1);

    EXPECT_TRUE(input->multimodal_features);
    EXPECT_EQ(input->multimodal_features.value().size(), 1);
}

TEST_F(MultimodalProcessorTest, testMultiInput) {
    FakeMultimodalProcessor processor =
        FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}, {2, 3}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");
    mm_inputs.emplace_back("2");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), true);

    EXPECT_EQ(input->input_ids.numel(), 8);

    EXPECT_TRUE(input->text_tokens_mask);
    auto text_tokens_mask = input->text_tokens_mask.value().data_ptr<int32_t>();
    EXPECT_EQ(input->text_tokens_mask.value().numel(), 8);
    EXPECT_EQ(text_tokens_mask[0], 1);
    EXPECT_EQ(text_tokens_mask[4], 1);
    EXPECT_EQ(text_tokens_mask[7], 1);

    EXPECT_TRUE(input->mm_locs);
    auto locs = input->mm_locs.value().data_ptr<int32_t>();
    EXPECT_EQ(input->mm_locs.value().numel(), 2);
    EXPECT_EQ(locs[0], 1);
    EXPECT_EQ(locs[1], 5);

    EXPECT_TRUE(input->multimodal_features);
    EXPECT_EQ(input->multimodal_features.value().size(), 2);
}

TEST_F(MultimodalProcessorTest, testWrongMMTag) {
    FakeMultimodalProcessor processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{2, 3, 4}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = torch::tensor({0, 1, 2, 3, 4}, torch::kInt32);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("2");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), false);
    EXPECT_EQ(res.ToString(), "more than 2 sep tokens or no sep tokens for multimodal model is not supported");
    EXPECT_EQ(res.code(), ErrorCode::MM_WRONG_FORMAT_ERROR);

    processor.sep_token_ids_ = {{3, 5}};
    res                      = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), false);
    EXPECT_EQ(res.ToString(), "unclosed multimodal tag pairs");
    EXPECT_EQ(res.code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
}

TEST_F(MultimodalProcessorTest, testTooLongInput) {
    FakeMultimodalProcessor processor    = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1, 2}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("10");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), false);
    EXPECT_EQ(res.ToString(), "input after multimodal process is 14 > max_seq_len(10)");
    EXPECT_EQ(res.code(), ErrorCode::MM_LONG_PROMPT_ERROR);
}

TEST_F(MultimodalProcessorTest, testGetMMFeatures) {
    FakeMultimodalProcessor processor    = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1, 2}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("2");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.getMultimodalFeatures(input->input_ids, mm_inputs).value();
    EXPECT_EQ(res.features.size(), 1);
    EXPECT_EQ(res.text_tokens_mask.numel(), 6);
    EXPECT_EQ(res.locs.numel(), 1);
    EXPECT_EQ(res.expanded_ids.numel(), 6);
}

TEST_F(MultimodalProcessorTest, testInterleavedVideoLayout) {
    FakeMultimodalProcessor processor =
        FakeMultimodalProcessor::createFakeMultimodalProcessor({{10, 11}, {20, 21}}, false, 64);
    auto                       token_ids = torch::tensor({0, 10, 99, 11, 5, 20, 98, 21, 6}, torch::kInt32);
    std::vector<torch::Tensor> features  = {
        torch::zeros({2, 1}),
        torch::ones({3, 1}),
        torch::full({1, 1}, 2),
    };
    std::vector<torch::Tensor> layouts = {
        // Image: one feature segment, no extra text around it.
        torch::tensor({-53530053, 1, 0, 0}, torch::kInt32),
        // Video frame 0: <boi>, three feature tokens, <eoi>, timestamp token.
        torch::tensor({-53530053, 1, 1, 2, 30, 31, 40}, torch::kInt32),
        // Video frame 1 belongs to the same source video.
        torch::tensor({-53530053, 0, 1, 2, 30, 31, 41}, torch::kInt32),
    };
    std::vector<MultimodalInput> inputs = {
        MultimodalInput("image", 1),
        MultimodalInput("video", 2),
    };

    auto result = processor.expandTokenIds(features, token_ids, inputs, {}, layouts);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    auto expanded = std::move(result.value());

    EXPECT_TRUE(expanded.consumed_mm_layout);
    EXPECT_EQ(expanded.expanded_ids.numel(), 19);
    EXPECT_EQ(expanded.locs.numel(), 3);
    EXPECT_EQ(expanded.locs[0].item<int32_t>(), 2);
    EXPECT_EQ(expanded.locs[1].item<int32_t>(), 8);
    EXPECT_EQ(expanded.locs[2].item<int32_t>(), 14);
    EXPECT_EQ(expanded.multimodal_inputs.size(), 3);
    EXPECT_EQ(expanded.multimodal_inputs[0].mm_type, 1);
    EXPECT_EQ(expanded.multimodal_inputs[1].mm_type, 2);
    EXPECT_EQ(expanded.multimodal_inputs[2].mm_type, 2);

    auto ids = expanded.expanded_ids.data_ptr<int32_t>();
    EXPECT_EQ(ids[0], 0);
    EXPECT_EQ(ids[1], 10);
    EXPECT_EQ(ids[4], 11);
    EXPECT_EQ(ids[5], 5);
    EXPECT_EQ(ids[6], 20);
    EXPECT_EQ(ids[7], 30);
    EXPECT_EQ(ids[11], 31);
    EXPECT_EQ(ids[12], 40);
    EXPECT_EQ(ids[13], 30);
    EXPECT_EQ(ids[15], 31);
    EXPECT_EQ(ids[16], 41);
    EXPECT_EQ(ids[17], 21);
    EXPECT_EQ(ids[18], 6);

    auto masks = expanded.text_tokens_mask.data_ptr<int32_t>();
    for (int index : {2, 3, 8, 9, 10, 14}) {
        EXPECT_EQ(masks[index], 0);
    }
    for (int index : {0, 1, 4, 5, 6, 7, 11, 12, 13, 15, 16, 17, 18}) {
        EXPECT_EQ(masks[index], 1);
    }
}

TEST_F(MultimodalProcessorTest, testInterleavedLayoutRejectsMissingGroupStart) {
    FakeMultimodalProcessor processor   = FakeMultimodalProcessor::createFakeMultimodalProcessor({{10, 11}}, false, 64);
    auto                    token_ids   = torch::tensor({10, 99, 11}, torch::kInt32);
    std::vector<torch::Tensor> features = {torch::zeros({1, 1})};
    std::vector<torch::Tensor> layouts  = {
        torch::tensor({-53530053, 0, 0, 0}, torch::kInt32),
    };
    std::vector<MultimodalInput> inputs = {MultimodalInput("video", 2)};

    auto result = processor.expandTokenIds(features, token_ids, inputs, {}, layouts);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
    EXPECT_EQ(result.status().ToString(), "interleaved multimodal layout must start a media group");
}

}  // namespace rtp_llm
