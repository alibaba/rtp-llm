#include <ATen/CPUGeneratorImpl.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {
GreedyOutput sampleGreedy(const GreedyParams& params);
}

namespace {

using namespace rtp_llm;

std::vector<at::Generator> makeGenerators(const std::vector<uint64_t>& seeds) {
    std::vector<at::Generator> generators;
    generators.reserve(seeds.size());
    for (auto seed : seeds) {
        auto generator = torch::make_generator<at::CPUGeneratorImpl>();
        generator.set_current_seed(seed);
        generators.push_back(generator);
    }
    return generators;
}

GreedyOutput runSample(torch::Tensor                logits,
                       torch::Tensor                token_ids,
                       torch::Tensor                input_lengths,
                       torch::Tensor                sequence_lengths,
                       torch::Tensor                top_k,
                       torch::Tensor                top_p,
                       torch::Tensor                temperature,
                       std::optional<torch::Tensor> no_repeat_ngram_size      = std::nullopt,
                       std::optional<torch::Tensor> cum_log_probs             = std::nullopt,
                       std::optional<torch::Tensor> output_all_probs          = std::nullopt,
                       std::optional<torch::Tensor> do_sample                 = std::nullopt,
                       std::vector<at::Generator>   generators                = {},
                       bool                         return_original_all_probs = false) {
    return sampleGreedy({logits,
                         input_lengths,
                         sequence_lengths,
                         token_ids,
                         static_cast<size_t>(token_ids.size(1) - 1),
                         top_k,
                         top_p,
                         temperature,
                         std::nullopt,
                         no_repeat_ngram_size,
                         cum_log_probs,
                         std::nullopt,
                         return_original_all_probs,
                         output_all_probs,
                         std::nullopt,
                         std::nullopt,
                         do_sample,
                         std::move(generators)});
}

TEST(GenericSamplerBatchFallbackTest, MixedTopKAndTopPReturnsSampledDistribution) {
    constexpr int64_t batch_size = 4;
    constexpr int64_t vocab_size = 8;

    auto logits      = torch::arange(vocab_size, torch::kFloat32).repeat({batch_size, 1});
    auto token_ids   = torch::zeros({batch_size, 1}, torch::kInt32);
    auto lengths     = torch::zeros({batch_size}, torch::kInt32);
    auto top_k       = torch::tensor({1, 2, 0, 3}, torch::kInt32);
    auto top_p       = torch::tensor({1.0f, 1.0f, 0.5f, 0.8f}, torch::kFloat32);
    auto temperature = torch::ones({batch_size}, torch::kFloat32);
    auto all_probs   = torch::zeros({batch_size, vocab_size}, torch::kFloat32);

    runSample(logits,
              token_ids,
              lengths,
              torch::empty({0}, torch::kInt32),
              top_k,
              top_p,
              temperature,
              std::nullopt,
              std::nullopt,
              all_probs,
              std::nullopt,
              makeGenerators({11, 12, 13, 14}));

    auto samples = token_ids.flatten().data_ptr<int32_t>();
    for (int64_t row = 0; row < batch_size; ++row) {
        EXPECT_GT(all_probs[row][samples[row]].item<float>(), 0.0f);
        EXPECT_NEAR(all_probs[row].sum().item<float>(), 1.0f, 1e-6f);
    }
    EXPECT_EQ(samples[0], 7);
    EXPECT_GE(samples[1], 6);
    EXPECT_EQ(samples[2], 7);
    EXPECT_GE(samples[3], 6);
}

TEST(GenericSamplerBatchFallbackTest, TopKOneUpdatesCumulativeLogProbWithoutProbabilityOutput) {
    auto logits              = torch::tensor({{0.0f, 1.0f, 2.0f}}, torch::kFloat32);
    auto token_ids           = torch::zeros({1, 1}, torch::kInt32);
    auto lengths             = torch::zeros({1}, torch::kInt32);
    auto cumulative_log_prob = torch::tensor({-3.0f}, torch::kFloat32);

    runSample(logits,
              token_ids,
              lengths,
              torch::empty({0}, torch::kInt32),
              torch::ones({1}, torch::kInt32),
              torch::ones({1}, torch::kFloat32),
              torch::ones({1}, torch::kFloat32),
              std::nullopt,
              cumulative_log_prob);

    EXPECT_EQ(token_ids.item<int32_t>(), 2);
    EXPECT_FLOAT_EQ(cumulative_log_prob.item<float>(), -3.0f);
}

TEST(GenericSamplerBatchFallbackTest, HonorsDoSampleWhenApplyingTemperature) {
    auto logits    = torch::tensor({{0.0f, 1.0f, 2.0f}, {0.0f, 1.0f, 2.0f}}, torch::kFloat32);
    auto original  = logits.clone();
    auto token_ids = torch::zeros({2, 1}, torch::kInt32);
    auto lengths   = torch::zeros({2}, torch::kInt32);
    auto all_probs = torch::zeros_like(logits);
    auto do_sample = torch::tensor({false, true}, torch::kBool);

    runSample(logits,
              token_ids,
              lengths,
              torch::empty({0}, torch::kInt32),
              torch::ones({2}, torch::kInt32),
              torch::ones({2}, torch::kFloat32),
              torch::full({2}, 2.0f, torch::kFloat32),
              std::nullopt,
              std::nullopt,
              all_probs,
              do_sample);

    auto expected_original = torch::softmax(original[0], -1);
    auto expected_scaled   = torch::softmax(original[1] / 2.0f, -1);
    EXPECT_TRUE(torch::allclose(all_probs[0], torch::one_hot(torch::tensor(2), 3).to(torch::kFloat32)));
    EXPECT_TRUE(torch::allclose(logits[0], expected_original));
    EXPECT_TRUE(torch::allclose(logits[1], expected_scaled));
}

TEST(GenericSamplerBatchFallbackTest, AppliesNoRepeatBigramBeforeTopKOneShortcut) {
    auto logits           = torch::tensor({{0.0f, 1.0f, 4.0f, 3.0f}}, torch::kFloat32);
    auto token_ids        = torch::tensor({{1, 2, 1, 0}}, torch::kInt32);
    auto input_lengths    = torch::zeros({1}, torch::kInt32);
    auto sequence_lengths = torch::tensor({2}, torch::kInt32);

    runSample(logits,
              token_ids,
              input_lengths,
              sequence_lengths,
              torch::ones({1}, torch::kInt32),
              torch::ones({1}, torch::kFloat32),
              torch::ones({1}, torch::kFloat32),
              torch::tensor({2}, torch::kInt32));

    EXPECT_EQ(token_ids.select(0, 0).select(0, token_ids.size(1) - 1).item<int32_t>(), 3);
}

TEST(GenericSamplerBatchFallbackTest, PerRequestGeneratorIsIndependentOfBatchRow) {
    auto sample_request = [](int64_t row, int64_t batch_size) {
        auto logits    = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f}}, torch::kFloat32).repeat({batch_size, 1});
        auto token_ids = torch::zeros({batch_size, 1}, torch::kInt32);
        auto lengths   = torch::zeros({batch_size}, torch::kInt32);
        std::vector<uint64_t> seeds(batch_size);
        for (int64_t index = 0; index < batch_size; ++index) {
            seeds[index] = static_cast<uint64_t>(100 + index);
        }
        seeds[row] = 2026;
        runSample(logits,
                  token_ids,
                  lengths,
                  torch::empty({0}, torch::kInt32),
                  torch::zeros({batch_size}, torch::kInt32),
                  torch::ones({batch_size}, torch::kFloat32),
                  torch::ones({batch_size}, torch::kFloat32),
                  std::nullopt,
                  std::nullopt,
                  std::nullopt,
                  std::nullopt,
                  makeGenerators(seeds));
        return token_ids[row].item<int32_t>();
    };

    EXPECT_EQ(sample_request(/*row=*/0, /*batch_size=*/1), sample_request(/*row=*/2, /*batch_size=*/4));
}

TEST(GenericSamplerBatchFallbackTest, PureTopPExcludesOutsideNucleus) {
    auto logits    = torch::tensor({{4.0f, 3.0f, 2.0f, 1.0f}}, torch::kFloat32).repeat({32, 1});
    auto token_ids = torch::zeros({32, 1}, torch::kInt32);
    auto lengths   = torch::zeros({32}, torch::kInt32);

    runSample(logits,
              token_ids,
              lengths,
              torch::empty({0}, torch::kInt32),
              torch::zeros({32}, torch::kInt32),
              torch::full({32}, 0.8f, torch::kFloat32),
              torch::ones({32}, torch::kFloat32),
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              makeGenerators(std::vector<uint64_t>(32, 17)));

    EXPECT_TRUE(token_ids.eq(0).logical_or(token_ids.eq(1)).all().item<bool>());
}

}  // namespace
