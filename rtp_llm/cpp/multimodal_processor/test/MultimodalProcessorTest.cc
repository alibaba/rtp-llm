#include <memory>
#include <optional>
#include <string>
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

TEST_F(MultimodalProcessorTest, testInputEmbeddingsLocsShiftAfterMultimodalExpansion) {
    FakeMultimodalProcessor        processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    std::shared_ptr<GenerateInput> input     = std::make_shared<GenerateInput>();
    input->input_ids                         = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    input->input_embeddings                  = {torch::ones({2, 1}, torch::kFloat32)};
    input->input_embeddings_locs             = {2};
    auto mm_inputs                           = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");
    input->multimodal_inputs = mm_inputs;

    auto res = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), true);

    EXPECT_EQ(input->input_ids.numel(), 6);
    ASSERT_TRUE(input->input_embeddings_locs.has_value());
    ASSERT_EQ(input->input_embeddings_locs->size(), 1);
    EXPECT_EQ(input->input_embeddings_locs->at(0), 4);
}

TEST_F(MultimodalProcessorTest, testInputEmbeddingsLocsShiftAcrossMultipleMultimodalTags) {
    FakeMultimodalProcessor processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}, {3}}, false, 20);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = torch::tensor({0, 1, 2, 3, 4, 5}, torch::kInt32);
    input->input_embeddings              = {torch::ones({1, 1}, torch::kFloat32), torch::ones({1, 1}, torch::kFloat32)};
    input->input_embeddings_locs         = {2, 4};
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");
    mm_inputs.emplace_back("2");
    input->multimodal_inputs = mm_inputs;

    auto res = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), true);

    ASSERT_TRUE(input->input_embeddings_locs.has_value());
    ASSERT_EQ(input->input_embeddings_locs->size(), 2);
    EXPECT_EQ(input->input_embeddings_locs->at(0), 4);
    EXPECT_EQ(input->input_embeddings_locs->at(1), 7);
}

TEST_F(MultimodalProcessorTest, testInputEmbeddingsRejectOverlapMultimodalTag) {
    FakeMultimodalProcessor        processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    std::shared_ptr<GenerateInput> input     = std::make_shared<GenerateInput>();
    input->input_ids                         = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    input->input_embeddings                  = {torch::ones({1, 1}, torch::kFloat32)};
    input->input_embeddings_locs             = {1};
    auto mm_inputs                           = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");
    input->multimodal_inputs = mm_inputs;

    auto res = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), false);
    EXPECT_NE(res.ToString().find("overlaps multimodal tag interval"), std::string::npos);
    EXPECT_EQ(res.code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
}

TEST_F(MultimodalProcessorTest, testEmbeddingEngineRejectInputEmbeddingsWithMultimodalInputs) {
    FakeMultimodalProcessor processor     = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    std::shared_ptr<EmbeddingInput> input = std::make_shared<EmbeddingInput>(std::vector<int32_t>{0, 1, 2, 3},
                                                                             std::vector<int32_t>{0, 0, 0, 0},
                                                                             std::vector<int32_t>{4},
                                                                             0,
                                                                             std::nullopt,
                                                                             torch::ones({4, 1}, torch::kFloat32));
    auto                            mm_inputs = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");

    auto res = processor.updateMultimodalFeatures(input, mm_inputs, "");
    EXPECT_EQ(res.ok(), false);
    EXPECT_NE(res.ToString().find("full-sequence tensor"), std::string::npos);
    EXPECT_EQ(res.code(), ErrorCode::MM_NOT_SUPPORTED_ERROR);
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

}  // namespace rtp_llm
