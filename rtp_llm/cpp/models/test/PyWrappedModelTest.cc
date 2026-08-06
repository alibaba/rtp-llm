#include "gtest/gtest.h"

#include "rtp_llm/cpp/models/PyWrappedModel.h"

namespace rtp_llm {

TEST(PyWrappedModelTest, HasMultimodalInputsDetectsFeatures) {
    GptModelInputs inputs;
    inputs.multimodal_features = std::vector<torch::Tensor>{torch::ones({1, 4})};

    EXPECT_TRUE(PyWrappedModel::hasMultimodalInputs(inputs));
}

TEST(PyWrappedModelTest, HasMultimodalInputsDetectsExtraInput) {
    GptModelInputs inputs;
    inputs.mm_extra_input = std::vector<torch::Tensor>{torch::ones({1, 4})};

    EXPECT_TRUE(PyWrappedModel::hasMultimodalInputs(inputs));
}

TEST(PyWrappedModelTest, HasMultimodalInputsRejectsEmptyInputs) {
    GptModelInputs inputs;

    EXPECT_FALSE(PyWrappedModel::hasMultimodalInputs(inputs));
}

TEST(PyWrappedModelTest, SplitRejectsEnabledMicroBatchPlanForMultimodalInputs) {
    GptModelInputs inputs;
    inputs.multimodal_features = std::vector<torch::Tensor>{torch::ones({1, 4})};
    MicroBatchPlan enabled_plan{true, {}};

    EXPECT_THROW(PyWrappedModel::splitInputsIntoMicroBatches(inputs, enabled_plan), std::exception);
}

TEST(PyWrappedModelTest, DisabledMicroBatchPlanKeepsMultimodalDataOnlyOnRealBatch) {
    GptModelInputs inputs;
    inputs.combo_tokens        = torch::tensor({1, 2}, torch::kInt32);
    inputs.multimodal_features = std::vector<torch::Tensor>{torch::ones({1, 4})};
    inputs.mm_features_locs    = torch::tensor({0}, torch::kInt32);

    const auto [micro_batches, token_slices] =
        PyWrappedModel::splitInputsIntoMicroBatches(inputs, MicroBatchPlan{false, {}});

    ASSERT_EQ(micro_batches.size(), 2);
    ASSERT_TRUE(micro_batches[0].multimodal_features.has_value());
    EXPECT_FALSE(micro_batches[0].multimodal_features->empty());
    EXPECT_TRUE(micro_batches[0].mm_features_locs.defined());
    EXPECT_FALSE(micro_batches[1].multimodal_features.has_value());
    EXPECT_FALSE(micro_batches[1].mm_features_locs.defined());
    EXPECT_TRUE(token_slices.empty());
}

}  // namespace rtp_llm
