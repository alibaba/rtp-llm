#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(DSparkPrepareLogitsTest, MatchesEagerBitwiseForStridedRows) {
    constexpr int64_t batch = 3;
    constexpr int64_t gamma = 2;
    constexpr int64_t vocab = 7;
    auto base_storage =
        torch::linspace(-2.0F, 2.0F, batch * gamma * vocab, torch::kCUDA).reshape({batch, gamma, vocab});
    auto base_logits = base_storage.select(1, 1);
    ASSERT_FALSE(base_logits.is_contiguous());
    auto markov_bias =
        torch::linspace(-0.5F, 0.5F, batch * vocab, torch::kCUDA).reshape({batch, vocab}).to(torch::kBFloat16);
    auto temperature = torch::tensor({1.0e-6F, 0.5F, 1.0F}, torch::TensorOptions().device(torch::kCUDA));

    auto expected = base_logits + markov_bias.to(torch::kFloat32);
    expected.div_(temperature.unsqueeze(1));
    auto actual = torch::empty_like(expected);
    prepare_dspark_logits(base_logits, markov_bias, temperature, actual);
    at::cuda::getCurrentCUDAStream().synchronize();

    EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(DSparkPrepareLogitsTest, RejectsInvalidInputs) {
    auto base_logits = torch::zeros({2, 4}, torch::kCUDA);
    auto markov_bias = torch::zeros({2, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
    auto temperature = torch::ones({2}, torch::kCUDA);
    auto output = torch::empty_like(base_logits);

    EXPECT_THROW(prepare_dspark_logits(base_logits.cpu(), markov_bias, temperature, output), c10::Error);
    EXPECT_THROW(
        prepare_dspark_logits(base_logits, markov_bias.narrow(1, 0, 3), temperature, output), c10::Error);
    EXPECT_THROW(
        prepare_dspark_logits(base_logits, markov_bias, temperature.narrow(0, 0, 1), output), c10::Error);
}

TEST(DSparkPrepareLogitsTest, GraphReplayUsesLiveInputs) {
    constexpr int64_t batch = 2;
    constexpr int64_t vocab = 17;
    auto base_logits = torch::zeros({batch, vocab}, torch::kCUDA);
    auto markov_bias =
        torch::zeros({batch, vocab}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
    auto temperature = torch::ones({batch}, torch::kCUDA);
    auto graph_output = torch::empty_like(base_logits);
    auto eager_output = torch::empty_like(base_logits);

    auto stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    prepare_dspark_logits(base_logits, markov_bias, temperature, graph_output);
    stream.synchronize();

    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    prepare_dspark_logits(base_logits, markov_bias, temperature, graph_output);
    graph.capture_end();

    base_logits.copy_(torch::linspace(-3.0F, 3.0F, batch * vocab, torch::kCUDA).reshape({batch, vocab}));
    markov_bias.copy_(torch::linspace(0.5F, -0.5F, batch * vocab, torch::kCUDA)
                          .reshape({batch, vocab})
                          .to(torch::kBFloat16));
    temperature.copy_(torch::tensor({0.25F, 1.5F}, torch::TensorOptions().device(torch::kCUDA)));
    graph.replay();
    auto replay_output = graph_output.clone();
    prepare_dspark_logits(base_logits, markov_bias, temperature, eager_output);
    auto eager_reference = base_logits + markov_bias.to(torch::kFloat32);
    eager_reference.div_(temperature.unsqueeze(1));
    stream.synchronize();

    EXPECT_TRUE(torch::equal(eager_output, eager_reference));
    EXPECT_TRUE(torch::equal(replay_output, eager_output));
}

}  // namespace
}  // namespace rtp_llm
