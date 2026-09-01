
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

#include <limits>

using namespace std;

namespace rtp_llm {

class SamplerDataBuilder {
public:
    SamplerDataBuilder() = default;

    struct Config {
        size_t            batch_size;
        size_t            vocab_size;
        size_t            max_length;
        rtp_llm::DataType logits_type = rtp_llm::DataType::TYPE_FP32;
    };

    SamplerInputs allocate(Config config) {
        SamplerInputs sampler_inputs;
        sampler_inputs.step           = config.max_length;
        sampler_inputs.batch_size     = config.batch_size;
        sampler_inputs.batch_size_out = config.batch_size;
        auto bs                       = (int64_t)config.batch_size;
        sampler_inputs.logits         = torch::empty(
            {bs, (int64_t)config.vocab_size},
            torch::TensorOptions().dtype(rtp_llm::dataTypeToTorchType(config.logits_type)).device(torch::kCUDA));
        sampler_inputs.sequence_lengths   = torch::empty({bs}, torch::kInt32);
        sampler_inputs.input_lengths      = torch::empty({bs}, torch::kInt32);
        sampler_inputs.num_beams_in       = torch::empty({bs}, torch::kLong);
        sampler_inputs.num_beams_out      = torch::empty({bs}, torch::kLong);
        sampler_inputs.top_k              = torch::empty({bs}, torch::kInt32);
        sampler_inputs.top_p              = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.temperature        = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.repetition_penalty = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.cum_log_probs      = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.token_ids          = torch::empty({bs, (int64_t)(sampler_inputs.step + 1)}, torch::kInt32);
        return sampler_inputs;
    };

    void setSequenceLengths(SamplerInputs& sampler_inputs, std::vector<int>& sequence_lengths) {
        RTP_LLM_CHECK(sequence_lengths.size() == sampler_inputs.batch_size);
        sampler_inputs.sequence_lengths = torch::tensor(sequence_lengths, torch::kInt32);
    };
};

class ModelDataTest: public DeviceTestBase {};

TEST_F(ModelDataTest, testConstruct) {
    SamplerDataBuilder builder;
    SamplerInputs      sampler_inputs   = builder.allocate({4, 1024, 1024});
    std::vector<int>   sequence_lengths = {1, 2, 3, 4};
    builder.setSequenceLengths(sampler_inputs, sequence_lengths);
    auto sl = sampler_inputs.sequence_lengths;
    EXPECT_EQ(std::vector<int>(sl.data_ptr<int>(), sl.data_ptr<int>() + sl.numel()), std::vector<int>({1, 2, 3, 4}));
}

TEST(PyWrappedModelTest, ContextParallelRequiresPurePrefill) {
    GptModelInputs inputs;
    inputs.input_lengths    = torch::empty({1}, torch::kInt32);
    inputs.sequence_lengths = torch::empty({0}, torch::kInt32);

    EXPECT_TRUE(PyWrappedModel::shouldUseContextParallel(inputs, true));
    EXPECT_FALSE(PyWrappedModel::shouldUseContextParallel(inputs, false));

    inputs.is_target_verify = true;
    EXPECT_FALSE(PyWrappedModel::shouldUseContextParallel(inputs, true));
    inputs.is_target_verify = false;

    inputs.last_hidden_states = torch::empty({1, 1});
    EXPECT_FALSE(PyWrappedModel::shouldUseContextParallel(inputs, true));
}

TEST(PyWrappedModelTest, ContextParallelRejectsDecodeAndMixedBatches) {
    GptModelInputs inputs;
    inputs.input_lengths    = torch::empty({0}, torch::kInt32);
    inputs.sequence_lengths = torch::empty({0}, torch::kInt32);
    EXPECT_FALSE(PyWrappedModel::shouldUseContextParallel(inputs, true));

    inputs.input_lengths    = torch::empty({1}, torch::kInt32);
    inputs.sequence_lengths = torch::empty({1}, torch::kInt32);
    EXPECT_FALSE(PyWrappedModel::shouldUseContextParallel(inputs, true));

    inputs.input_lengths = torch::empty({2}, torch::kInt32);
    EXPECT_FALSE(PyWrappedModel::shouldUseContextParallel(inputs, true));
}

TEST_F(ModelDataTest, ContextParallelHandleInputsUsesProductionMetadata) {
    const auto original_tokens = torch::arange(100, 358, torch::TensorOptions().dtype(torch::kInt32));
    const auto original_lengths = torch::tensor({1, 257}, torch::kInt32);
    const auto prefix_lengths   = torch::tensor({64, 128}, torch::kInt32);

    for (int rank: {0, 3}) {
        ParallelismConfig config;
        config.tp_size                                  = 4;
        config.tp_rank                                  = rank;
        config.prefill_cp_config.segment_size_alignment = 64;
        ZigZagProcessor processor(config);

        GptModelInputs inputs;
        inputs.combo_tokens     = original_tokens;
        inputs.input_lengths    = original_lengths;
        inputs.sequence_lengths = torch::empty({0}, torch::kInt32);
        inputs.prefix_lengths   = prefix_lengths;
        torch_ext::PyContextParallelParams cp_params;

        processor.handleInputs(inputs, cp_params);

        EXPECT_TRUE(torch::equal(original_tokens, torch::arange(100, 358, torch::kInt32)));
        EXPECT_TRUE(torch::equal(original_lengths, torch::tensor({1, 257}, torch::kInt32)));
        EXPECT_TRUE(torch::equal(inputs.input_lengths, torch::tensor({128, 128}, torch::kInt32)));
        EXPECT_TRUE(torch::equal(inputs.prefix_lengths, prefix_lengths));
        EXPECT_TRUE(torch::equal(cp_params.prefill_actual_input_lengths_cpu, original_lengths));
        EXPECT_TRUE(
            torch::equal(cp_params.prefill_cp_padding_lengths.cpu(), torch::tensor({511, 255}, torch::kInt32)));
        EXPECT_TRUE(torch::equal(cp_params.prefill_cp_chunk_lengths.cpu(), torch::tensor({128, 128}, torch::kInt32)));
        EXPECT_EQ(cp_params.prefill_qkv_padding_mask.sum().item<int>(), 258);

        const auto shuffle = cp_params.prefill_shuffle_indices.cpu();
        ASSERT_EQ(shuffle.numel(), 256);
        ASSERT_EQ(inputs.combo_tokens.numel(), 256);
        const int actual_lengths[] = {1, 257};
        const int source_offsets[] = {0, 1};
        int       valid_tokens     = 0;
        for (int batch = 0; batch < 2; ++batch) {
            for (int local = 0; local < 128; ++local) {
                const int index    = batch * 128 + local;
                const int position = shuffle[index].item<int>();
                if (position < actual_lengths[batch]) {
                    ++valid_tokens;
                    EXPECT_EQ(inputs.combo_tokens[index].item<int>(),
                              original_tokens[source_offsets[batch] + position].item<int>());
                } else {
                    EXPECT_EQ(inputs.combo_tokens[index].item<int>(), 0);
                }
            }
        }
        EXPECT_EQ(valid_tokens, 65);
    }
}

TEST_F(ModelDataTest, ContextParallelHandleInputsRejectsAlignmentOverflow) {
    ParallelismConfig config;
    config.tp_size                                  = 4;
    config.tp_rank                                  = 0;
    config.prefill_cp_config.segment_size_alignment = std::numeric_limits<size_t>::max();
    ZigZagProcessor processor(config);

    GptModelInputs inputs;
    inputs.combo_tokens     = torch::tensor({1}, torch::kInt32);
    inputs.input_lengths    = torch::tensor({1}, torch::kInt32);
    inputs.sequence_lengths = torch::empty({0}, torch::kInt32);
    torch_ext::PyContextParallelParams cp_params;

    EXPECT_THROW(processor.handleInputs(inputs, cp_params), std::overflow_error);
}

}  // namespace rtp_llm
