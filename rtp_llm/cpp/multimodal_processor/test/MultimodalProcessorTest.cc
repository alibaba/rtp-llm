#include <array>
#include <cstdlib>
#include <fstream>
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"
#include <memory>
#include <utility>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
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

TEST(MultimodalErrorTest, validatesMultimodalErrorCodes) {
    EXPECT_EQ(parseMultimodalErrorCode(static_cast<int>(ErrorCode::MM_WRONG_FORMAT_ERROR)),
              ErrorCode::MM_WRONG_FORMAT_ERROR);
    EXPECT_EQ(parseMultimodalErrorCode(static_cast<int>(ErrorCode::MM_DOWNLOAD_FAILED)), ErrorCode::MM_DOWNLOAD_FAILED);
    EXPECT_FALSE(parseMultimodalErrorCode(static_cast<int>(ErrorCode::EXECUTION_EXCEPTION)).has_value());
    EXPECT_FALSE(parseMultimodalErrorCode(999999).has_value());
}

TEST(MultimodalErrorTest, parsesKnownErrorCodes) {
    constexpr std::array<ErrorCode, 7> cases = {{
        ErrorCode::MM_LONG_PROMPT_ERROR,
        ErrorCode::MM_WRONG_FORMAT_ERROR,
        ErrorCode::MM_PROCESS_ERROR,
        ErrorCode::MM_EMPTY_ENGINE_ERROR,
        ErrorCode::MM_NOT_SUPPORTED_ERROR,
        ErrorCode::MM_DOWNLOAD_FAILED,
        ErrorCode::MM_REMOTE_RPC_FAILED,
    }};

    for (const auto code : cases) {
        const auto parsed = parseMultimodalErrorMessage("[" + ErrorCodeToString(code) + "] details");
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(parsed->code(), code);
        EXPECT_EQ(parsed->ToString(), "details");
    }
}

TEST(MultimodalErrorTest, classifiesRetryableErrorsCentrally) {
    EXPECT_FALSE(isRetryableMultimodalError(ErrorCode::MM_LONG_PROMPT_ERROR));
    EXPECT_FALSE(isRetryableMultimodalError(ErrorCode::MM_WRONG_FORMAT_ERROR));
    EXPECT_TRUE(isRetryableMultimodalError(ErrorCode::MM_PROCESS_ERROR));
    EXPECT_FALSE(isRetryableMultimodalError(ErrorCode::MM_NOT_SUPPORTED_ERROR));
    EXPECT_TRUE(isRetryableMultimodalError(ErrorCode::MM_EMPTY_ENGINE_ERROR));
    EXPECT_TRUE(isRetryableMultimodalError(ErrorCode::MM_DOWNLOAD_FAILED));
    EXPECT_TRUE(isRetryableMultimodalError(ErrorCode::MM_REMOTE_RPC_FAILED));
}

TEST(MultimodalErrorTest, rejectsMalformedOrUnknownMessages) {
    EXPECT_FALSE(parseMultimodalErrorMessage("plain grpc error").has_value());
    EXPECT_FALSE(parseMultimodalErrorMessage("prefix [MM_PROCESS_ERROR] details").has_value());
    EXPECT_FALSE(parseMultimodalErrorMessage("[MM_PROCESS_ERROR details").has_value());
    EXPECT_FALSE(parseMultimodalErrorMessage("[UNKNOWN_ERROR] details").has_value());
}

TEST(MultimodalErrorTest, allowsEmptyDetails) {
    const auto parsed = parseMultimodalErrorMessage("[MM_WRONG_FORMAT_ERROR]");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
    EXPECT_TRUE(parsed->ToString().empty());
}

TEST(MultimodalInputTest, timeoutDoesNotAffectCacheKey) {
    MMPreprocessConfig inherited_timeout;
    inherited_timeout.max_pixels        = 4096;
    inherited_timeout.mm_timeout_ms     = -1;
    MMPreprocessConfig explicit_timeout = inherited_timeout;
    explicit_timeout.mm_timeout_ms      = 120000;

    MultimodalInput inherited_input("http://image", 0, torch::empty({0}), inherited_timeout);
    MultimodalInput explicit_input("http://image", 0, torch::empty({0}), explicit_timeout);

    EXPECT_EQ(inherited_input.cache_key(), explicit_input.cache_key());
    EXPECT_NE(inherited_input.to_string(), explicit_input.to_string());
}

TEST_F(MultimodalProcessorTest, completeVideoEmbeddingPreservesOuterTagsAndMrope) {
    auto processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1, 2}}, false, 100);
    // The mixin returns timestamp / start / four visual rows / end for each frame.
    auto embedding          = torch::arange(14, torch::kFloat32).reshape({14, 1});
    auto relative_positions = torch::tensor({0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 4, 4, 4,
                                             5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 8, 7, 8, 7, 7, 8, 8, 9, 9, 9},
                                            torch::kInt32)
                                  .reshape({14, 3});
    MultimodalInput video_input("video");
    video_input.mm_type = 2;
    auto result = processor.expandTokenIds({embedding}, torch::tensor({9, 1, -1, 2, 8}, torch::kInt32), {video_input});
    ASSERT_TRUE(result.ok());
    const auto& expanded = result.value();
    EXPECT_EQ(expanded.expanded_ids.numel(), 18);
    EXPECT_TRUE(torch::equal(expanded.locs, torch::tensor({2}, torch::kInt32)));
    EXPECT_EQ(expanded.text_tokens_mask.sum().item<int>(), 4);
    EXPECT_EQ(expanded.expanded_ids[1].item<int>(), 1);
    EXPECT_EQ(expanded.expanded_ids[16].item<int>(), 2);
    GenerateInput input;
    input.multimodal_inputs   = std::vector<MultimodalInput>{video_input};
    input.multimodal_features = std::vector<torch::Tensor>{embedding};
    EXPECT_EQ(input.multimodalLengths().at(2), 14);
    auto positions = PositionIdsGenerator::generatePositionIds(
        18, MROPE, expanded.locs, std::vector<torch::Tensor>{relative_positions});
    auto expected =
        torch::tensor({0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,  4, 4,  4, 4, 4,  5,  4,  5,  4,  4,  5,  5,  6,  6,  6,
                       7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 10, 9, 10, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13},
                      torch::kInt32)
            .reshape({18, 3});
    EXPECT_TRUE(torch::equal(positions.reshape({18, 3}), expected));
}

}  // namespace rtp_llm
