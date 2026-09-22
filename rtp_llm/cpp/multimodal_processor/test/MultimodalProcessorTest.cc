#include <array>
#include <chrono>
#include <memory>
#include <utility>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/multimodal_processor/test/FakeMultimodalProcessor.h"

using namespace std;

namespace rtp_llm {

class MultimodalProcessorTest: public DeviceTestBase {};

namespace {
class PreparedV41Processor: public FakeMultimodalProcessor {
public:
    PreparedV41Processor(): FakeMultimodalProcessor(py::none(), {}, false, 32) {}
    int     calls           = 0;
    int64_t last_timeout_ms = -1;

protected:
    ErrorResult<MultimodalOutput> V41MultimodalEmbedding(const V41RequestInputs& inputs,
                                                         const std::string&,
                                                         int64_t timeout_ms,
                                                         int64_t,
                                                         grpc::ServerContext*) override {
        last_timeout_ms = timeout_ms;
        ++calls;
        MultimodalOutput output;
        for (const auto& image : inputs.images) {
            output.mm_features.push_back(torch::ones({image.types.numel(), 4}));
        }
        return output;
    }
};

std::shared_ptr<GenerateInput> preparedV41Input() {
    auto input           = std::make_shared<GenerateInput>();
    input->input_ids     = torch::zeros({11}, torch::kInt32);
    auto prepared_ptr    = std::make_shared<V41RequestInputs>();
    input->v41_inputs    = prepared_ptr;
    auto& prepared       = *prepared_ptr;
    prepared.token_types = torch::tensor({-1, 0, 1, 2, 3, -1, 0, 1, 2, 3, -1}, torch::kInt32);
    prepared.image_mask  = prepared.token_types.ne(-1);
    for (int32_t start : {1, 6}) {
        V41ImageInput image;
        image.start = start;
        image.types = torch::tensor({0, 1, 2, 3}, torch::kInt32);
        prepared.images.push_back(std::move(image));
    }
    return input;
}
}  // namespace

TEST_F(MultimodalProcessorTest, V41PreparedMetadataPreservesImagesAndTextOnlyRequests) {
    PreparedV41Processor processor;
    auto                 input = preparedV41Input();
    ASSERT_TRUE(processor.updateMultimodalFeatures(input).ok());
    EXPECT_EQ(processor.calls, 1);
    EXPECT_TRUE(torch::equal(input->mm_locs.value(), torch::tensor({1, 6}, torch::kInt32)));
    EXPECT_TRUE(
        torch::equal(input->text_tokens_mask.value(), input->v41_inputs->image_mask.logical_not().to(torch::kInt32)));
    EXPECT_EQ(input->multimodal_features->size(), 2);
    input     = preparedV41Input();
    auto text = std::make_shared<V41RequestInputs>(*input->v41_inputs);
    text->images.clear();
    text->token_types.fill_(-1);
    text->image_mask.fill_(false);
    input->v41_inputs = text;
    ASSERT_TRUE(processor.updateMultimodalFeatures(input).ok());
    EXPECT_EQ(processor.calls, 1);
    EXPECT_FALSE(input->multimodal_features.has_value());
    EXPECT_FALSE(input->mm_locs.has_value());
    EXPECT_FALSE(input->text_tokens_mask.has_value());
}

TEST_F(MultimodalProcessorTest, V41MalformedMetadataNeverReachesEmbedding) {
    for (const std::string invalid : {"token_shape",
                                      "token_dtype",
                                      "mask_length",
                                      "mask_dtype",
                                      "mask_value",
                                      "invalid_kind",
                                      "negative_start",
                                      "overlap",
                                      "past_end",
                                      "empty_types",
                                      "types_dtype",
                                      "types_mismatch",
                                      "missing_image",
                                      "uncovered_gap"}) {
        SCOPED_TRACE(invalid);
        PreparedV41Processor processor;
        auto                 input        = preparedV41Input();
        auto                 prepared_ptr = std::make_shared<V41RequestInputs>(*input->v41_inputs);
        input->v41_inputs                 = prepared_ptr;
        auto& prepared                    = *prepared_ptr;
        if (invalid == "token_shape")
            prepared.token_types = prepared.token_types.unsqueeze(0);
        if (invalid == "token_dtype")
            prepared.token_types = prepared.token_types.to(torch::kFloat32);
        if (invalid == "mask_length")
            prepared.image_mask = prepared.image_mask.slice(0, 0, 10);
        if (invalid == "mask_dtype")
            prepared.image_mask = prepared.image_mask.to(torch::kInt32);
        if (invalid == "mask_value")
            prepared.image_mask.index_put_({1}, false);
        if (invalid == "invalid_kind")
            prepared.token_types.index_put_({1}, 4);
        if (invalid == "negative_start")
            prepared.images[0].start = -1;
        if (invalid == "overlap")
            prepared.images[1].start = 3;
        if (invalid == "past_end")
            prepared.images[1].start = 10;
        if (invalid == "empty_types")
            prepared.images[1].types = torch::empty({0}, torch::kInt32);
        if (invalid == "types_dtype")
            prepared.images[1].types = prepared.images[1].types.to(torch::kFloat32);
        if (invalid == "types_mismatch")
            prepared.images[1].types.index_put_({1}, 2);
        if (invalid == "missing_image")
            prepared.images.pop_back();
        if (invalid == "uncovered_gap") {
            prepared.token_types.index_put_({5}, 1);
            prepared.image_mask.index_put_({5}, true);
        }
        EXPECT_FALSE(processor.updateMultimodalFeatures(input).ok());
        EXPECT_EQ(processor.calls, 0);
        EXPECT_FALSE(input->multimodal_features.has_value());
    }
}

TEST_F(MultimodalProcessorTest, SameUrlDifferentImageFeaturesHaveDifferentCacheTokens) {
    auto processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 32);
    auto expand    = [&processor]() {
        auto input               = std::make_shared<GenerateInput>();
        input->input_ids         = torch::tensor({0, 1, 2}, torch::kInt32);
        input->multimodal_inputs = std::vector<MultimodalInput>{MultimodalInput("8")};
        auto status              = processor.updateMultimodalFeatures(input);
        EXPECT_TRUE(status.ok());
        return input->input_ids;
    };
    auto first = expand();
    EXPECT_TRUE(torch::equal(first, expand()));
    processor.feature_value = 1.0f;
    auto second             = expand();
    EXPECT_FALSE(torch::equal(first, second));
    EXPECT_EQ(first[0].item<int32_t>(), second[0].item<int32_t>());
    EXPECT_EQ(first[-1].item<int32_t>(), second[-1].item<int32_t>());
}

TEST_F(MultimodalProcessorTest, V41EmbeddingReceivesOnlyRemainingRequestBudget) {
    for (int64_t elapsed_ms : {0, 2000}) {
        PreparedV41Processor processor;
        auto                 input         = preparedV41Input();
        input->generate_config             = std::make_shared<GenerateConfig>();
        input->generate_config->timeout_ms = 10000;
        input->begin_time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count()
            - elapsed_ms * 1000;
        ASSERT_TRUE(processor.updateMultimodalFeatures(input).ok());
        EXPECT_EQ(processor.calls, 1);
        EXPECT_GT(processor.last_timeout_ms, 0);
        EXPECT_LE(processor.last_timeout_ms, 10000 - elapsed_ms);
    }
    PreparedV41Processor processor;
    auto                 input         = preparedV41Input();
    input->generate_config             = std::make_shared<GenerateConfig>();
    input->generate_config->timeout_ms = 1000;
    input->begin_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count()
        - 2000000;
    auto result = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(result.code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(processor.calls, 0);
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

TEST(MultimodalErrorTest, validatesMultimodalErrorCodes) {
    EXPECT_EQ(parseMultimodalErrorCode(static_cast<int>(ErrorCode::MM_WRONG_FORMAT_ERROR)),
              ErrorCode::MM_WRONG_FORMAT_ERROR);
    EXPECT_EQ(parseMultimodalErrorCode(static_cast<int>(ErrorCode::MM_DOWNLOAD_FAILED)), ErrorCode::MM_DOWNLOAD_FAILED);
    EXPECT_FALSE(parseMultimodalErrorCode(static_cast<int>(ErrorCode::EXECUTION_EXCEPTION)).has_value());
    EXPECT_EQ(parseMultimodalErrorCode(499), ErrorCode::CANCELLED);
    EXPECT_EQ(parseMultimodalErrorCode(static_cast<int>(ErrorCode::CANCELLED)), ErrorCode::CANCELLED);
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
