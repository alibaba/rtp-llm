#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "torch/all.h"

#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"

namespace rtp_llm {
namespace {

class MtpTargetLogitsProcessorTest: public ::testing::TestWithParam<int> {
protected:
    GenerateStreamPtr makeThinkStream(const std::vector<int>& end_think_token_ids) {
        auto config                 = std::make_shared<GenerateConfig>();
        config->in_think_mode       = true;
        config->max_thinking_tokens = 1;
        config->end_think_token_ids = end_think_token_ids;

        auto input             = std::make_shared<GenerateInput>();
        input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
        input->generate_config = std::move(config);

        ModelConfig model_config;
        model_config.max_seq_len                  = 32;
        model_config.vocab_size                   = 10;
        model_config.num_layers                   = 1;
        model_config.attn_config.tokens_per_block = 2;
        model_config.special_tokens.eos_token_id  = 9;

        auto stream = std::make_shared<NormalGenerateStream>(
            input, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
        stream->setScoreLen(GetParam() + 1);
        stream->setSPOutputBuffer(std::make_shared<SpeculativeExecutorStreamOutput>());
        return stream;
    }

    GenerateStreamPtr makePlainStream() {
        auto config = std::make_shared<GenerateConfig>();

        auto input             = std::make_shared<GenerateInput>();
        input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
        input->generate_config = std::move(config);

        ModelConfig model_config;
        model_config.max_seq_len                  = 32;
        model_config.vocab_size                   = 10;
        model_config.num_layers                   = 1;
        model_config.attn_config.tokens_per_block = 2;
        model_config.special_tokens.eos_token_id  = 9;

        auto stream = std::make_shared<NormalGenerateStream>(
            input, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
        stream->logits_processor_list_.clear();
        stream->setScoreLen(GetParam() + 1);
        stream->setSPOutputBuffer(std::make_shared<SpeculativeExecutorStreamOutput>());
        return stream;
    }

    static void expectOnlyTokenAllowed(const torch::Tensor& logits, int token_id) {
        ASSERT_EQ(logits.dim(), 1);
        for (int64_t i = 0; i < logits.numel(); ++i) {
            if (i == token_id) {
                EXPECT_FLOAT_EQ(logits[i].item<float>(), 1.0f);
            } else {
                EXPECT_FLOAT_EQ(logits[i].item<float>(), BaseLogitsProcessor::neg_inf);
            }
        }
    }
};

TEST_P(MtpTargetLogitsProcessorTest, AppliesIndependentPerRowThinkStateWithoutMutatingCanonicalState) {
    const int propose_step = GetParam();
    const int score_len    = propose_step + 1;

    auto first_stream  = makeThinkStream({4, 5});
    auto second_stream = makeThinkStream({7, 8});
    auto first_processor =
        std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(first_stream->getAllLogitsProcessorPtr().front());
    auto second_processor =
        std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(second_stream->getAllLogitsProcessorPtr().front());
    ASSERT_NE(first_processor, nullptr);
    ASSERT_NE(second_processor, nullptr);
    ASSERT_EQ(first_processor->thinkEndTokensStatus(), std::vector<size_t>({0}));
    ASSERT_EQ(second_processor->thinkEndTokensStatus(), std::vector<size_t>({0}));

    std::vector<int32_t> combo_tokens;
    combo_tokens.reserve(2 * score_len);
    combo_tokens.insert(combo_tokens.end(), {3, 4});
    if (propose_step > 1) {
        combo_tokens.push_back(5);
    }
    combo_tokens.resize(score_len, 6);
    combo_tokens.insert(combo_tokens.end(), {3, 7});
    if (propose_step > 1) {
        combo_tokens.push_back(8);
    }
    combo_tokens.resize(2 * score_len, 6);

    GptModelInputs model_inputs;
    model_inputs.combo_tokens = torch::tensor(combo_tokens, torch::kInt32);

    GptModelOutputs model_output;
    model_output.logits = torch::zeros({2 * score_len, 10}, torch::kFloat32);

    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = propose_step;
    MtpBatchStreamProcessor processor(ModelConfig{},
                                      PDSepConfig{},
                                      ProfilingDebugLoggingConfig{},
                                      CacheConfig{},
                                      sp_config,
                                      false);

    StreamGroups stream_groups({first_stream, second_stream});
    auto         inputs_or = processor.gatherSpecSamplerInput(stream_groups, model_inputs, model_output);
    ASSERT_TRUE(inputs_or.ok()) << inputs_or.status();
    auto inputs = std::move(inputs_or.value());

    EXPECT_NE(inputs.logits_processor_states_ptr, nullptr);
    if (inputs.logits_processor_states_ptr) {
        inputs.logits_processor_states_ptr->batchProcess(inputs);
    }

    EXPECT_EQ(std::vector<int32_t>(inputs.sequence_lengths.data_ptr<int32_t>(),
                                   inputs.sequence_lengths.data_ptr<int32_t>()
                                       + inputs.sequence_lengths.numel()),
              ([score_len] {
                  std::vector<int32_t> expected;
                  for (int stream_idx = 0; stream_idx < 2; ++stream_idx) {
                      for (int row = 0; row < score_len; ++row) {
                          expected.push_back(3 + row);
                      }
                  }
                  return expected;
              })());

    expectOnlyTokenAllowed(inputs.logits[0], 4);
    expectOnlyTokenAllowed(inputs.logits[1], 5);
    expectOnlyTokenAllowed(inputs.logits[score_len], 7);
    expectOnlyTokenAllowed(inputs.logits[score_len + 1], 8);
    for (int row = 2; row < score_len; ++row) {
        EXPECT_TRUE(torch::equal(inputs.logits[row], torch::zeros({10}, torch::kFloat32)));
        EXPECT_TRUE(torch::equal(inputs.logits[score_len + row], torch::zeros({10}, torch::kFloat32)));
    }

    EXPECT_EQ(first_processor->thinkEndTokensStatus(), std::vector<size_t>({0}));
    EXPECT_EQ(second_processor->thinkEndTokensStatus(), std::vector<size_t>({0}));
}

TEST_P(MtpTargetLogitsProcessorTest, SkipsComboMaterializationWithoutLogitsProcessors) {
    const int score_len = GetParam() + 1;
    auto      stream    = makePlainStream();

    GptModelInputs model_inputs;
    EXPECT_FALSE(model_inputs.combo_tokens.defined());

    GptModelOutputs model_output;
    model_output.logits = torch::zeros({score_len, 10}, torch::kFloat32);

    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = GetParam();
    MtpBatchStreamProcessor processor(ModelConfig{},
                                      PDSepConfig{},
                                      ProfilingDebugLoggingConfig{},
                                      CacheConfig{},
                                      sp_config,
                                      false);

    StreamGroups stream_groups({stream});
    auto         inputs_or = processor.gatherSpecSamplerInput(stream_groups, model_inputs, model_output);
    ASSERT_TRUE(inputs_or.ok()) << inputs_or.status();
    auto inputs = std::move(inputs_or.value());

    EXPECT_EQ(inputs.logits_processor_states_ptr, nullptr);
    EXPECT_EQ(std::vector<int32_t>(inputs.sequence_lengths.data_ptr<int32_t>(),
                                   inputs.sequence_lengths.data_ptr<int32_t>()
                                       + inputs.sequence_lengths.numel()),
              ([score_len] {
                  std::vector<int32_t> expected;
                  for (int row = 0; row < score_len; ++row) {
                      expected.push_back(3 + row);
                  }
                  return expected;
              })());
}

TEST_P(MtpTargetLogitsProcessorTest, PreservesMixedStreamOffsetsAndMaterializesDeviceCombo) {
    const int propose_step = GetParam();
    const int score_len    = propose_step + 1;

    auto processor_stream = makeThinkStream({4, 5});
    auto plain_stream     = makePlainStream();
    auto canonical_processor =
        std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor_stream->getAllLogitsProcessorPtr().front());
    ASSERT_NE(canonical_processor, nullptr);

    std::vector<int32_t> combo_tokens;
    combo_tokens.reserve(2 * score_len);
    combo_tokens.insert(combo_tokens.end(), {3, 4});
    if (propose_step > 1) {
        combo_tokens.push_back(5);
    }
    combo_tokens.resize(score_len, 6);
    combo_tokens.insert(combo_tokens.end(), {3, 7});
    if (propose_step > 1) {
        combo_tokens.push_back(8);
    }
    combo_tokens.resize(2 * score_len, 6);

    GptModelInputs model_inputs;
    model_inputs.combo_tokens = torch::tensor(combo_tokens, torch::kInt32);
    if (propose_step == 3) {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "accelerator is required for the MTP3 device materialization case";
        }
        model_inputs.combo_tokens = model_inputs.combo_tokens.to(torch::kCUDA);
        ASSERT_TRUE(model_inputs.combo_tokens.is_cuda());
    }

    GptModelOutputs model_output;
    model_output.logits = torch::zeros({2 * score_len, 10}, torch::kFloat32);

    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = propose_step;
    MtpBatchStreamProcessor processor(ModelConfig{},
                                      PDSepConfig{},
                                      ProfilingDebugLoggingConfig{},
                                      CacheConfig{},
                                      sp_config,
                                      false);

    StreamGroups stream_groups({processor_stream, plain_stream});
    auto         inputs_or = processor.gatherSpecSamplerInput(stream_groups, model_inputs, model_output);
    ASSERT_TRUE(inputs_or.ok()) << inputs_or.status();
    auto inputs = std::move(inputs_or.value());
    ASSERT_NE(inputs.logits_processor_states_ptr, nullptr);
    inputs.logits_processor_states_ptr->batchProcess(inputs);

    expectOnlyTokenAllowed(inputs.logits[0], 4);
    expectOnlyTokenAllowed(inputs.logits[1], 5);
    for (int row = 0; row < score_len; ++row) {
        EXPECT_TRUE(torch::equal(inputs.logits[score_len + row], torch::zeros({10}, torch::kFloat32)));
    }
    EXPECT_EQ(canonical_processor->thinkEndTokensStatus(), std::vector<size_t>({0}));
}

INSTANTIATE_TEST_SUITE_P(MtpOneAndThreeDraftTokens,
                         MtpTargetLogitsProcessorTest,
                         ::testing::Values(1, 3));

}  // namespace
}  // namespace rtp_llm
