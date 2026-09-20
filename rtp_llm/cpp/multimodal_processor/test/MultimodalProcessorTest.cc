#include <memory>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/multimodal_processor/test/FakeMultimodalProcessor.h"

using namespace std;

namespace rtp_llm {

class MultimodalProcessorTest: public DeviceTestBase {};

TEST_F(MultimodalProcessorTest, testFrontendExpansionMatchesLegacyAssembly) {
    auto processor            = FakeMultimodalProcessor::createFakeMultimodalProcessor({{90, 91}}, false, 100);
    auto legacy               = std::make_shared<GenerateInput>();
    legacy->input_ids         = torch::tensor({1, 90, 5, 91, 2, 90, 6, 91, 3}, torch::kInt32);
    legacy->multimodal_inputs = std::vector<MultimodalInput>{MultimodalInput("2"), MultimodalInput("3")};
    ASSERT_TRUE(processor.updateMultimodalFeatures(legacy).ok());
    auto expanded                     = std::make_shared<GenerateInput>();
    expanded->input_ids               = legacy->input_ids.clone();
    expanded->multimodal_inputs       = legacy->multimodal_inputs;
    expanded->multimodal_token_layout = MultimodalTokenLayout{{{2, 2}, {7, 3}}};
    const void* token_storage         = expanded->input_ids.data_ptr();
    ASSERT_TRUE(processor.updateMultimodalFeatures(expanded).ok());
    EXPECT_EQ(expanded->input_ids.data_ptr(), token_storage);
    EXPECT_TRUE(torch::equal(expanded->input_ids, legacy->input_ids));
    EXPECT_TRUE(torch::equal(*expanded->text_tokens_mask, *legacy->text_tokens_mask));
    EXPECT_TRUE(torch::equal(*expanded->mm_locs, *legacy->mm_locs));
    ASSERT_EQ(expanded->multimodal_features->size(), legacy->multimodal_features->size());
    for (size_t i = 0; i < expanded->multimodal_features->size(); ++i) {
        EXPECT_TRUE(torch::equal(expanded->multimodal_features->at(i), legacy->multimodal_features->at(i)));
    }
}

TEST_F(MultimodalProcessorTest, testExpandedLayoutDoesNotScanHashValuesAsTags) {
    auto          processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{99}}, false, 100);
    GenerateInput input;
    input.input_ids               = torch::tensor({1, 99, -2, 3, 99, 4}, torch::kInt32);
    input.multimodal_token_layout = MultimodalTokenLayout{{{1, 2}, {4, 1}}};
    std::vector<torch::Tensor> features{torch::zeros({2, 4}), torch::ones({1, 4})};
    std::vector<torch::Tensor> hashes{torch::tensor({99, -2}, torch::kInt32), torch::tensor({99}, torch::kInt32)};
    auto                       result = processor.useExpandedTokenLayout(input, features, hashes);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(torch::equal(result.value().text_tokens_mask, torch::tensor({1, 0, 0, 1, 0, 1}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(result.value().locs, torch::tensor({1, 4}, torch::kInt32)));

    for (auto spans : std::vector<std::vector<std::pair<int32_t, int32_t>>>{
             {}, {{1, 2}}, {{-1, 2}, {4, 1}}, {{1, 0}, {4, 1}}, {{1, 2}, {2, 1}}, {{1, 2}, {6, 1}}, {{1, 3}, {4, 1}}}) {
        input.multimodal_token_layout->spans = spans;
        EXPECT_FALSE(processor.useExpandedTokenLayout(input, features, hashes).ok());
    }
    input.multimodal_token_layout = MultimodalTokenLayout{{{1, 2}, {4, 1}}};
    hashes[0][0]                  = 42;
    auto refreshed                = processor.useExpandedTokenLayout(input, features, hashes);
    ASSERT_TRUE(refreshed.ok());
    EXPECT_TRUE(torch::equal(refreshed.value().expanded_ids, torch::tensor({1, 42, -2, 3, 99, 4}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(input.input_ids, torch::tensor({1, 99, -2, 3, 99, 4}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(refreshed.value().locs, result.value().locs));
    EXPECT_TRUE(torch::equal(refreshed.value().text_tokens_mask, result.value().text_tokens_mask));
    hashes[0] = torch::tensor({42}, torch::kInt32);
    EXPECT_FALSE(processor.useExpandedTokenLayout(input, features, hashes).ok());
}

TEST_F(MultimodalProcessorTest, testPrecomputedFeatureHashes) {
    auto                       processor  = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 100);
    auto                       tokens     = torch::tensor({0, 1, 2, 1, 3}, torch::kInt32);
    std::vector<torch::Tensor> embeddings = {torch::zeros({2, 4}), torch::ones({1, 4})};
    std::vector<torch::Tensor> hashes = {torch::tensor({-10, 11}, torch::kInt32), torch::tensor({12}, torch::kInt32)};
    auto                       result = processor.expandTokenIds(embeddings, tokens, {}, {}, hashes);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(torch::equal(result.value().expanded_ids, torch::tensor({0, -10, 11, 2, 12, 3}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(result.value().text_tokens_mask, torch::tensor({1, 0, 0, 1, 0, 1}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(result.value().locs, torch::tensor({1, 4}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(tokens, torch::tensor({0, 1, 2, 1, 3}, torch::kInt32)));
    hashes[0] = torch::tensor({-10}, torch::kInt32);
    EXPECT_FALSE(processor.expandTokenIds(embeddings, tokens, {}, {}, hashes).ok());
    hashes.pop_back();
    EXPECT_FALSE(processor.expandTokenIds(embeddings, tokens, {}, {}, hashes).ok());
}

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

TEST_F(MultimodalProcessorTest, testFeatureHashCpuGpuConsistency) {
    FakeMultimodalProcessor processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    auto cpu_embedding = torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f}, {1.0f, 2.0f, 3.0f, 4.0f}, {1.0f, 2.0f, 3.0f, 5.0f}});
    auto gpu_embedding = cpu_embedding.to(torch::kCUDA);

    std::vector<int32_t> cpu_hashes(cpu_embedding.size(0));
    std::vector<int32_t> gpu_hashes(gpu_embedding.size(0));
    EXPECT_TRUE(processor.getFeatureHash(cpu_hashes.data(), cpu_embedding).ok());
    EXPECT_TRUE(processor.getFeatureHash(gpu_hashes.data(), gpu_embedding).ok());

    EXPECT_EQ(cpu_hashes, gpu_hashes);
    EXPECT_EQ(cpu_hashes[0], cpu_hashes[1]);
    EXPECT_NE(cpu_hashes[0], cpu_hashes[2]);
}

TEST_F(MultimodalProcessorTest, testFeatureHashNonContiguousTensor) {
    FakeMultimodalProcessor processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    auto                    cpu_embedding =
        torch::arange(24, torch::TensorOptions().dtype(torch::kFloat32)).reshape({4, 6}).transpose(0, 1);
    ASSERT_FALSE(cpu_embedding.is_contiguous());
    auto gpu_embedding = cpu_embedding.to(torch::kCUDA);

    std::vector<int32_t> cpu_hashes(cpu_embedding.size(0));
    std::vector<int32_t> gpu_hashes(gpu_embedding.size(0));
    EXPECT_TRUE(processor.getFeatureHash(cpu_hashes.data(), cpu_embedding).ok());
    EXPECT_TRUE(processor.getFeatureHash(gpu_hashes.data(), gpu_embedding).ok());
    EXPECT_EQ(cpu_hashes, gpu_hashes);
}

TEST_F(MultimodalProcessorTest, testFeatureHashTailBytes) {
    FakeMultimodalProcessor processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    auto cpu_embedding                = torch::arange(39, torch::TensorOptions().dtype(torch::kUInt8)).reshape({3, 13});
    auto gpu_embedding                = cpu_embedding.to(torch::kCUDA);

    std::vector<int32_t> cpu_hashes(cpu_embedding.size(0));
    std::vector<int32_t> gpu_hashes(gpu_embedding.size(0));
    EXPECT_TRUE(processor.getFeatureHash(cpu_hashes.data(), cpu_embedding).ok());
    EXPECT_TRUE(processor.getFeatureHash(gpu_hashes.data(), gpu_embedding).ok());
    EXPECT_EQ(cpu_hashes, gpu_hashes);
}

TEST_F(MultimodalProcessorTest, testFeatureHashRealisticShape) {
    FakeMultimodalProcessor processor     = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    auto                    cpu_embedding = torch::randn({553, 4096}, torch::TensorOptions().dtype(torch::kBFloat16));
    auto                    gpu_embedding = cpu_embedding.to(torch::kCUDA);

    std::vector<int32_t> cpu_hashes(cpu_embedding.size(0));
    std::vector<int32_t> gpu_hashes(gpu_embedding.size(0));
    EXPECT_TRUE(processor.getFeatureHash(cpu_hashes.data(), cpu_embedding).ok());
    EXPECT_TRUE(processor.getFeatureHash(gpu_hashes.data(), gpu_embedding).ok());
    EXPECT_EQ(cpu_hashes, gpu_hashes);
}

}  // namespace rtp_llm
