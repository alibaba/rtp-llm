#include "torch/all.h"
#include "gtest/gtest.h"
#include <vector>
#include <pybind11/pybind11.h>
#include <memory>

#define private public
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class FakeMultimodalProcessor: public MultimodalProcessor {
public:
    using MultimodalProcessor::MultimodalProcessor;

    static FakeMultimodalProcessor createFakeMultimodalProcessor(std::vector<std::vector<int64_t>> sep_token_ids,
                                                                 bool                              include_sep_tokens,
                                                                 int                               max_seq_len) {
        MMModelConfig mm_model_config;
        mm_model_config.mm_sep_tokens      = sep_token_ids;
        mm_model_config.include_sep_tokens = include_sep_tokens;
        return FakeMultimodalProcessor(py::none(), mm_model_config, max_seq_len);
    }

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "") {
        MultimodalOutput res = MultimodalOutput();
        for (auto& mm_input : mm_inputs) {
            res.mm_features.push_back(torch::randn({std::stoi(mm_input.url), 2}));
        }
        return res;
    }
};

class MultimodalProcessorTest: public DeviceTestBase {};

TEST_F(MultimodalProcessorTest, testSimple) {
    FakeMultimodalProcessor        processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    std::shared_ptr<GenerateInput> input     = std::make_shared<GenerateInput>();
    input->input_ids                         = createBuffer<int32_t>({4}, {0, 1, 2, 3}, AllocationType::HOST);
    auto mm_inputs                           = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), true);

    auto input_ids = input->input_ids->data<int32_t>();
    EXPECT_EQ(input->input_ids->size(), 6);
    EXPECT_EQ(input_ids[0], 0);
    EXPECT_EQ(input_ids[4], 2);
    EXPECT_EQ(input_ids[5], 3);

    EXPECT_TRUE(input->text_tokens_mask);
    auto text_tokens_mask = input->text_tokens_mask.value()->data<int32_t>();
    EXPECT_EQ(input->text_tokens_mask.value()->size(), 6);
    EXPECT_EQ(text_tokens_mask[0], 1);
    EXPECT_EQ(text_tokens_mask[1], 0);
    EXPECT_EQ(text_tokens_mask[2], 0);
    EXPECT_EQ(text_tokens_mask[3], 0);
    EXPECT_EQ(text_tokens_mask[4], 1);
    EXPECT_EQ(text_tokens_mask[5], 1);

    EXPECT_TRUE(input->mm_locs);
    auto locs = input->mm_locs.value()->data<int32_t>();
    EXPECT_EQ(input->mm_locs.value()->size(), 1);
    EXPECT_EQ(locs[0], 1);

    EXPECT_TRUE(input->multimodal_features);
    EXPECT_EQ(input->multimodal_features.value().size(), 1);
}

TEST_F(MultimodalProcessorTest, testMultiInput) {
    FakeMultimodalProcessor processor =
        FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}, {2, 3}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = createBuffer<int32_t>({4}, {0, 1, 2, 3}, AllocationType::HOST);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");
    mm_inputs.emplace_back("2");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), true);

    EXPECT_EQ(input->input_ids->size(), 8);

    EXPECT_TRUE(input->text_tokens_mask);
    auto text_tokens_mask = input->text_tokens_mask.value()->data<int32_t>();
    EXPECT_EQ(input->text_tokens_mask.value()->size(), 8);
    EXPECT_EQ(text_tokens_mask[0], 1);
    EXPECT_EQ(text_tokens_mask[4], 1);
    EXPECT_EQ(text_tokens_mask[7], 1);

    EXPECT_TRUE(input->mm_locs);
    auto locs = input->mm_locs.value()->data<int32_t>();
    EXPECT_EQ(input->mm_locs.value()->size(), 2);
    EXPECT_EQ(locs[0], 1);
    EXPECT_EQ(locs[1], 5);

    EXPECT_TRUE(input->multimodal_features);
    EXPECT_EQ(input->multimodal_features.value().size(), 2);
}

TEST_F(MultimodalProcessorTest, testWrongMMTag) {
    FakeMultimodalProcessor processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{2, 3, 4}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = createBuffer<int32_t>({5}, {0, 1, 2, 3, 4}, AllocationType::HOST);
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
    input->input_ids                     = createBuffer<int32_t>({4}, {0, 1, 2, 3}, AllocationType::HOST);
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
    input->input_ids                     = createBuffer<int32_t>({4}, {0, 1, 2, 3}, AllocationType::HOST);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("2");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.getMultimodalFeatures(input->input_ids, mm_inputs).value();
    EXPECT_EQ(res.features.size(), 1);
    EXPECT_EQ(res.text_tokens_mask->size(), 6);
    EXPECT_EQ(res.locs->size(), 1);
    EXPECT_EQ(res.expanded_ids->size(), 6);
}

}  // namespace rtp_llm
