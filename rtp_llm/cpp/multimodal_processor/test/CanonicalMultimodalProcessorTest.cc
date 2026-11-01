#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {
namespace {

constexpr int32_t kImageToken = 129264;

std::vector<int32_t> tensorValues(const torch::Tensor& tensor) {
    auto value = tensor.to(torch::kCPU).contiguous();
    return {value.data_ptr<int32_t>(), value.data_ptr<int32_t>() + value.numel()};
}

MMModelConfig modelConfig(int64_t semantic_mm_token_id) {
    MMModelConfig config;
    config.is_multimodal        = true;
    config.mm_sep_tokens        = {{99}};
    config.include_sep_tokens   = false;
    config.semantic_mm_token_id = semantic_mm_token_id;
    return config;
}

class FeatureHashProcessor final: public MultimodalProcessor {
public:
    explicit FeatureHashProcessor(int64_t semantic_mm_token_id):
        MultimodalProcessor(py::none(), modelConfig(semantic_mm_token_id), 64) {}

    std::vector<int32_t> row_hashes{101, 102, 103};

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<MultimodalInput>,
                                                      std::string          = "",
                                                      int64_t              = 0,
                                                      grpc::ServerContext* = nullptr) override {
        MultimodalOutput output;
        output.mm_features.push_back(torch::ones({3, 4}, torch::kFloat32));
        output.mm_feature_hashes =
            std::vector<torch::Tensor>{torch::tensor(row_hashes, torch::TensorOptions().dtype(torch::kInt32))};
        return output;
    }
};

std::shared_ptr<GenerateInput> makeInput() {
    auto input               = std::make_shared<GenerateInput>();
    input->input_ids         = torch::tensor({7, 99, 8}, torch::kInt32);
    input->multimodal_inputs = std::vector<MultimodalInput>{MultimodalInput("unused")};
    input->generate_config   = std::make_shared<GenerateConfig>();
    return input;
}

class CanonicalMultimodalProcessorTest: public DeviceTestBase {};

#if USING_CUDA
class CudaOutputProcessor final: public MultimodalProcessor {
public:
    CudaOutputProcessor(): MultimodalProcessor(py::none(), modelConfig(kImageToken), 64) {}

    torch::Tensor feature;
    torch::Tensor extra_input;

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<MultimodalInput>,
                                                      std::string          = "",
                                                      int64_t              = 0,
                                                      grpc::ServerContext* = nullptr) override {
        MultimodalOutput output;
        output.mm_features       = {feature};
        output.mm_extra_input    = std::vector<torch::Tensor>{extra_input};
        output.mm_feature_hashes = std::vector<torch::Tensor>{torch::tensor({101, 102, 103}, torch::kInt32)};
        return output;
    }
};

TEST_F(CanonicalMultimodalProcessorTest, CudaFeaturesAndExtraInputsArePinnedAndCpuReady) {
    CudaOutputProcessor processor;
    for (const auto dtype : {torch::kFloat32, torch::kBFloat16}) {
        for (const bool transposed : {false, true}) {
            auto expected = torch::arange(12, torch::TensorOptions().dtype(dtype));
            expected      = transposed ? expected.reshape({4, 3}).transpose(0, 1) : expected.reshape({3, 4});
            const auto expected_extra = torch::arange(6, torch::TensorOptions().dtype(torch::kInt32)).reshape({3, 2});
            processor.feature         = expected.to(torch::kCUDA);
            processor.extra_input     = expected_extra.to(torch::kCUDA);
            auto input                = makeInput();

            ASSERT_TRUE(processor.updateMultimodalFeatures(input).ok());
            ASSERT_TRUE(input->multimodal_features.has_value());
            ASSERT_EQ(input->multimodal_features->size(), 1u);
            const auto& feature = input->multimodal_features->front();
            EXPECT_TRUE(feature.device().is_cpu());
            EXPECT_TRUE(feature.is_pinned());
            EXPECT_TRUE(torch::equal(feature, expected));
            ASSERT_TRUE(input->mm_extra_input.has_value());
            ASSERT_EQ(input->mm_extra_input->size(), 1u);
            const auto& extra = input->mm_extra_input->front();
            EXPECT_TRUE(extra.device().is_cpu());
            EXPECT_TRUE(extra.is_pinned());
            EXPECT_TRUE(torch::equal(extra, expected_extra));
        }
    }
}
#endif

TEST_F(CanonicalMultimodalProcessorTest, SemanticModeStoresCanonicalIdsAndHashSidecar) {
    FeatureHashProcessor processor(kImageToken);
    auto                 input = makeInput();

    ASSERT_TRUE(processor.updateMultimodalFeatures(input).ok());
    EXPECT_EQ(tensorValues(input->input_ids), (std::vector<int32_t>{7, kImageToken, kImageToken, kImageToken, 8}));
    EXPECT_EQ(tensorValues(*input->text_tokens_mask), (std::vector<int32_t>{1, 0, 0, 0, 1}));
    EXPECT_EQ(tensorValues(*input->mm_locs), (std::vector<int32_t>{1}));
    ASSERT_TRUE(input->multimodal_cache_spans);
    ASSERT_EQ(input->multimodal_cache_spans->size(), 1u);
    EXPECT_EQ(input->multimodal_cache_spans->front().offset, 1);
    EXPECT_EQ(input->multimodal_cache_spans->front().row_hashes, (std::vector<int32_t>{101, 102, 103}));
}

TEST_F(CanonicalMultimodalProcessorTest, DifferentFeaturesKeepCanonicalIdsButChangeSidecar) {
    FeatureHashProcessor processor(kImageToken);
    auto                 first = makeInput();
    ASSERT_TRUE(processor.updateMultimodalFeatures(first).ok());

    processor.row_hashes = {201, 202, 203};
    auto second          = makeInput();
    ASSERT_TRUE(processor.updateMultimodalFeatures(second).ok());

    EXPECT_TRUE(torch::equal(first->input_ids, second->input_ids));
    ASSERT_TRUE(first->multimodal_cache_spans);
    ASSERT_TRUE(second->multimodal_cache_spans);
    EXPECT_NE(first->multimodal_cache_spans->front().row_hashes, second->multimodal_cache_spans->front().row_hashes);
}

TEST_F(CanonicalMultimodalProcessorTest, DefaultModePreservesLegacyHashAsTokenBehavior) {
    FeatureHashProcessor processor(-1);
    auto                 input = makeInput();

    ASSERT_TRUE(processor.updateMultimodalFeatures(input).ok());
    EXPECT_EQ(tensorValues(input->input_ids), (std::vector<int32_t>{7, 101, 102, 103, 8}));
    EXPECT_TRUE(!input->multimodal_cache_spans || input->multimodal_cache_spans->empty());
}

}  // namespace
}  // namespace rtp_llm
