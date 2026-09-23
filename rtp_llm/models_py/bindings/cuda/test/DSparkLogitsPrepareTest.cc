#include <gtest/gtest.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/dspark_logits.h"

namespace rtp_llm {
namespace {

constexpr const char* kFusionEnv = "DSV41_FUSED_DSPARK_LOGITS";

class DSparkLogitsPrepareTest: public ::testing::Test {
protected:
    void SetUp() override {
        if (const auto* value = std::getenv(kFusionEnv)) {
            previous_env_ = value;
        }
        ASSERT_EQ(setenv(kFusionEnv, "1", 1), 0);
    }

    void TearDown() override {
        if (previous_env_) {
            setenv(kFusionEnv, previous_env_->c_str(), 1);
        } else {
            unsetenv(kFusionEnv);
        }
    }

    static torch::Tensor
    reference(const torch::Tensor& base, const torch::Tensor& bias, const torch::Tensor& temperature) {
        auto output = base + bias.to(torch::kFloat32);
        return output.div_(temperature.unsqueeze(1));
    }

    static void expectBitwiseEqual(const torch::Tensor& actual, const torch::Tensor& expected) {
        ASSERT_EQ(actual.sizes(), expected.sizes());
        ASSERT_EQ(actual.scalar_type(), torch::kFloat32);
        EXPECT_TRUE(actual.is_contiguous());
        EXPECT_TRUE(torch::equal(actual.view(torch::kInt32), expected.contiguous().view(torch::kInt32)));
    }

    static torch::TensorOptions floatCuda() {
        return torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    }

private:
    std::optional<std::string> previous_env_;
};

TEST_F(DSparkLogitsPrepareTest, ExactBF16AndFP32BiasWithStepStrideAndPaddedVocabulary) {
    constexpr int64_t gamma = 5;
    for (auto dtype : {torch::kBFloat16, torch::kFloat32}) {
        for (int64_t batch : {1, 4, 8}) {
            for (int64_t vocab : {1, 31, 1025, 129280}) {
                SCOPED_TRACE(::testing::Message() << "batch=" << batch << " vocab=" << vocab << " dtype=" << dtype);
                auto padded = torch::randn({batch, gamma, vocab + 7}, floatCuda());
                padded.narrow(2, vocab, 7).fill_(std::numeric_limits<float>::quiet_NaN());
                auto bias         = torch::randn({batch, vocab + 3}, floatCuda()).to(dtype).narrow(1, 0, vocab);
                auto temperatures = torch::tensor({1.0e-6f, 0.17f, 0.5f, 0.7f, 1.0f, 1.3f, 2.0f, 100.0f})
                                        .narrow(0, 0, batch)
                                        .to(torch::kCUDA);
                for (int64_t step : {0, 4}) {
                    auto base     = padded.select(1, step).narrow(1, 0, vocab);
                    auto expected = reference(base, bias, temperatures);
                    auto actual   = tryPrepareDSparkLogits(base, bias, temperatures);
                    ASSERT_TRUE(actual.defined());
                    expectBitwiseEqual(actual, expected);
                    expectBitwiseEqual(torch::softmax(actual, -1), torch::softmax(expected, -1));
                }
            }
        }
    }
}

TEST_F(DSparkLogitsPrepareTest, PreservesFP32RoundingAndSubnormals) {
    const float tiny = std::numeric_limits<float>::denorm_min();
    auto        base = torch::tensor({tiny, -tiny, 1.0e-38f, -1.0e-38f, 1.0e38f, -1.0e38f, 1.0000001f, -0.0f})
                    .reshape({1, 8})
                    .repeat({8, 1})
                    .to(torch::kCUDA);
    auto bias = torch::tensor({tiny, tiny, 0.0f, -0.0f, -1.0e38f, 1.0e38f, -1.0f, -0.0f})
                    .reshape({1, 8})
                    .repeat({8, 1})
                    .to(torch::kCUDA);
    auto temperatures = torch::tensor({1.0e-6f, 0.17f, 0.5f, 0.7f, 1.0f, 1.3f, 2.0f, 100.0f}).to(torch::kCUDA);
    expectBitwiseEqual(tryPrepareDSparkLogits(base, bias, temperatures), reference(base, bias, temperatures));
}

TEST_F(DSparkLogitsPrepareTest, DisabledAndUnsupportedInputsUseReference) {
    auto base         = torch::randn({4, 65}, floatCuda());
    auto bias         = torch::randn_like(base).to(torch::kBFloat16);
    auto temperatures = torch::full({4}, 0.7f, floatCuda());
    auto expected     = reference(base, bias, temperatures);
    ASSERT_EQ(setenv(kFusionEnv, "0", 1), 0);
    expectBitwiseEqual(execPrepareDSparkLogits(base, bias, temperatures), expected);
    ASSERT_EQ(setenv(kFusionEnv, "1", 1), 0);

    auto half_bias = bias.to(torch::kFloat16);
    EXPECT_FALSE(tryPrepareDSparkLogits(base, half_bias, temperatures).defined());
    expectBitwiseEqual(execPrepareDSparkLogits(base, half_bias, temperatures),
                       reference(base, half_bias, temperatures));

    auto strided_base = base.slice(1, 0, 65, 2);
    auto strided_bias = bias.slice(1, 0, 65, 2);
    EXPECT_FALSE(tryPrepareDSparkLogits(strided_base, strided_bias, temperatures).defined());
    expectBitwiseEqual(execPrepareDSparkLogits(strided_base, strided_bias, temperatures),
                       reference(strided_base, strided_bias, temperatures));

    auto cpu_base        = base.cpu();
    auto cpu_bias        = bias.cpu();
    auto cpu_temperature = temperatures.cpu();
    EXPECT_FALSE(tryPrepareDSparkLogits(cpu_base, cpu_bias, cpu_temperature).defined());
    expectBitwiseEqual(execPrepareDSparkLogits(cpu_base, cpu_bias, cpu_temperature),
                       reference(cpu_base, cpu_bias, cpu_temperature));
}

TEST_F(DSparkLogitsPrepareTest, EmptyBatchDoesNotLaunch) {
    auto base        = torch::empty({0, 129280}, floatCuda());
    auto bias        = torch::empty({0, 129280}, floatCuda().dtype(torch::kBFloat16));
    auto temperature = torch::empty({0}, floatCuda());
    auto actual      = tryPrepareDSparkLogits(base, bias, temperature);
    ASSERT_TRUE(actual.defined());
    EXPECT_EQ(actual.sizes(), base.sizes());
    EXPECT_EQ(actual.numel(), 0);
}

TEST_F(DSparkLogitsPrepareTest, NonDefaultStreamAndGraphReplayReadUpdatedInputs) {
    const auto                 stream = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto                       padded       = torch::randn({8, 5, 1032}, floatCuda());
    auto                       base         = padded.select(1, 3).narrow(1, 0, 1025);
    auto                       bias         = torch::randn({8, 1025}, floatCuda()).to(torch::kBFloat16);
    auto                       temperatures = torch::full({8}, 0.7f, floatCuda());
    auto                       warmup       = tryPrepareDSparkLogits(base, bias, temperatures);
    stream.synchronize();
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    auto actual = tryPrepareDSparkLogits(base, bias, temperatures);
    graph.capture_end();
    base.mul_(3.0f);
    bias.add_(0.5f);
    temperatures.fill_(1.3f);
    graph.replay();
    expectBitwiseEqual(actual, reference(base, bias, temperatures));
}

TEST_F(DSparkLogitsPrepareTest, SequentialSamplingPreservesDenseQTokensAndRngState) {
    constexpr int64_t batch = 4, gamma = 5, vocab = 257, rank = 256;
    auto              base           = torch::randn({batch, gamma, vocab + 7}, floatCuda()).narrow(2, 0, vocab);
    auto              w1             = torch::randn({vocab, rank}, floatCuda()).to(torch::kBFloat16);
    auto              w2             = torch::randn({vocab, rank}, floatCuda()).to(torch::kBFloat16);
    auto              temperature    = torch::tensor({0.5f, 0.7f, 1.0f, 1.3f}).to(torch::kCUDA);
    auto              generator      = at::cuda::detail::getDefaultCUDAGenerator();
    auto              original_state = generator.get_state();
    auto              run            = [&](bool fused) {
        auto previous  = torch::arange(batch, floatCuda().dtype(torch::kLong));
        auto all_probs = torch::empty({batch, gamma, vocab}, floatCuda());
        auto token_ids = torch::empty({batch, gamma}, floatCuda().dtype(torch::kInt32));
        for (int64_t step = 0; step < gamma; ++step) {
            auto bias   = torch::mm(w1.index_select(0, previous), w2.transpose(0, 1));
            auto logits = fused ? execPrepareDSparkLogits(base.select(1, step), bias, temperature) :
                                                          reference(base.select(1, step), bias, temperature);
            auto probs  = torch::softmax(logits, -1);
            auto tokens = execSampleFromProbs(probs);
            all_probs.select(1, step).copy_(probs);
            token_ids.select(1, step).copy_(tokens);
            previous = tokens.to(torch::kLong);
        }
        return std::make_pair(all_probs, token_ids);
    };
    auto expected     = run(false);
    auto expected_rng = generator.get_state();
    generator.set_state(original_state);
    auto actual = run(true);
    expectBitwiseEqual(actual.first, expected.first);
    EXPECT_TRUE(torch::equal(actual.second, expected.second));
    EXPECT_TRUE(torch::equal(generator.get_state(), expected_rng));
}

TEST_F(DSparkLogitsPrepareTest, DISABLED_BenchmarkGraphReplay) {
    const auto                 stream = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto                       measure = [&](const std::function<torch::Tensor()>& operation) {
        auto warmup = operation();
        stream.synchronize();
        at::cuda::CUDAGraph graph;
        graph.capture_begin();
        auto output = operation();
        graph.capture_end();
        for (int i = 0; i < 20; ++i) {
            graph.replay();
        }
        cudaEvent_t start, end;
        TORCH_CHECK(cudaEventCreate(&start) == cudaSuccess);
        TORCH_CHECK(cudaEventCreate(&end) == cudaSuccess);
        TORCH_CHECK(cudaEventRecord(start, stream.stream()) == cudaSuccess);
        for (int i = 0; i < 1000; ++i) {
            graph.replay();
        }
        TORCH_CHECK(cudaEventRecord(end, stream.stream()) == cudaSuccess);
        TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);
        float milliseconds = 0;
        TORCH_CHECK(cudaEventElapsedTime(&milliseconds, start, end) == cudaSuccess);
        TORCH_CHECK(cudaEventDestroy(start) == cudaSuccess);
        TORCH_CHECK(cudaEventDestroy(end) == cudaSuccess);
        return milliseconds;  // 1000 replays: milliseconds total == microseconds/replay.
    };
    for (int64_t batch : {4, 8}) {
        auto base         = torch::randn({batch, 5, 129280}, floatCuda()).select(1, 2);
        auto bias         = torch::randn({batch, 129280}, floatCuda()).to(torch::kBFloat16);
        auto temperatures = torch::full({batch}, 0.7f, floatCuda());
        for (bool with_softmax : {false, true}) {
            auto run = [&](bool fused) {
                auto logits =
                    fused ? tryPrepareDSparkLogits(base, bias, temperatures) : reference(base, bias, temperatures);
                return with_softmax ? torch::softmax(logits, -1) : logits;
            };
            const auto reference_us = measure([&]() { return run(false); });
            const auto fused_us     = measure([&]() { return run(true); });
            std::cout << "DSPARK_LOGITS_BENCHMARK {\"batch\":" << batch
                      << ",\"vocab\":129280,\"with_softmax\":" << (with_softmax ? "true" : "false")
                      << ",\"reference_us\":" << reference_us << ",\"fused_us\":" << fused_us << "}" << std::endl;
        }
    }
}

}  // namespace
}  // namespace rtp_llm
