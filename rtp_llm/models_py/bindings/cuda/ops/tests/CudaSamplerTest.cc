#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/SamplingParams.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/greedy_top1_renorm.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <limits>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
using namespace std;
using namespace rtp_llm;

namespace {

auto execPreparedGreedyForTest(const GreedyParams& params) {
    auto prepared = prepareSamplingParams(params, params.logits.size(0));
    auto result   = execSampleGreedy(prepared);
    if (params.token_ids.is_cpu()) {
        // The fixture owns host staging; production sampler input is device storage.
        params.token_ids.copy_(prepared.token_ids, /*non_blocking=*/false);
    }
    return result;
}

GreedyOutput referenceTopPOneSample(const GreedyParams& params) {
    auto probabilities = torch::softmax(params.logits, -1);
    params.logits.copy_(probabilities);
    params.output_all_probs.value().copy_(probabilities);
    const auto batch   = probabilities.size(0);
    auto       seeds   = torch::empty({batch}, torch::TensorOptions().dtype(torch::kInt64));
    auto       offsets = torch::empty_like(seeds);
    for (int64_t i = 0; i < batch; ++i) {
        auto [seed, offset]            = get_seed_and_offset(32, params.generator[i]);
        seeds.data_ptr<int64_t>()[i]   = seed;
        offsets.data_ptr<int64_t>()[i] = offset;
    }
    auto tokens = torch::empty({batch}, probabilities.options().dtype(torch::kInt32));
    auto valid  = torch::empty({batch}, probabilities.options().dtype(torch::kBool));
    top_p_sampling_from_probs(probabilities,
                              tokens,
                              valid,
                              std::nullopt,
                              params.top_p,
                              1.0,
                              true,
                              seeds.to(torch::kCUDA),
                              0,
                              offsets.to(torch::kCUDA),
                              0,
                              reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream()));
    params.token_ids.select(1, params.token_ids.size(1) - 1).copy_(tokens);
    params.cum_log_probs.value().add_(probabilities.gather(1, tokens.to(torch::kLong).unsqueeze(1)).squeeze(1).log());
    return {valid};
}

void assertAllValid(const torch::Tensor& valid) {
    ASSERT_TRUE(valid.defined());
    ASSERT_EQ(valid.scalar_type(), torch::kBool);
    ASSERT_TRUE(valid.cpu().all().item<bool>());
}

void assertTokenIn(const std::vector<int32_t>& token_ids, size_t index, std::initializer_list<int32_t> allowed) {
    ASSERT_NE(std::find(allowed.begin(), allowed.end(), token_ids[index]), allowed.end())
        << "token_ids[" << index << "]=" << token_ids[index];
}

void assertCumLogProbMatches(const std::vector<float>&   cum_log_probs,
                             const std::vector<float>&   output_all_probs,
                             const std::vector<int32_t>& token_ids,
                             const std::vector<float>&   initial_cum_log_probs,
                             int64_t                     batch_size,
                             int64_t                     step,
                             int64_t                     vocab_size) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const auto token = token_ids[batch * (step + 1) + step];
        ASSERT_GE(token, 0);
        ASSERT_LT(token, vocab_size);
        const auto prob = output_all_probs[batch * vocab_size + token];
        ASSERT_GT(prob, 0.0f);
        ASSERT_NEAR(cum_log_probs[batch], initial_cum_log_probs[batch] + std::log(prob), 1e-3);
    }
}


}  // namespace

class CudaSamplerTest: public DeviceTestBase {
public:
protected:
    // Helper: create a CUDA tensor from float data
    torch::Tensor cudaTensor(std::vector<float> data, std::vector<int64_t> shape) {
        return torch::tensor(data, torch::kFloat32).reshape(shape).to(torch::kCUDA);
    }

    // Helper: create a CUDA tensor from int32 data
    torch::Tensor cudaIntTensor(std::vector<int32_t> data, std::vector<int64_t> shape) {
        return torch::tensor(data, torch::kInt32).reshape(shape).to(torch::kCUDA);
    }

    // Helper: create a pinned CPU tensor from int32 data (for HOST buffers)
    torch::Tensor pinnedIntTensor(std::vector<int32_t> data) {
        return torch::tensor(data, torch::kInt32).pin_memory();
    }

    // Helper: create a pinned CPU tensor from float data
    torch::Tensor pinnedFloatTensor(std::vector<float> data) {
        return torch::tensor(data, torch::kFloat32).pin_memory();
    }

    // Helper: read GPU int32 tensor to host vector
    std::vector<int32_t> toHostInt(const torch::Tensor& t) {
        auto cpu = t.cpu().contiguous();
        return std::vector<int32_t>(cpu.data_ptr<int32_t>(), cpu.data_ptr<int32_t>() + cpu.numel());
    }

    // Helper: read GPU float tensor to host vector
    std::vector<float> toHostFloat(const torch::Tensor& t) {
        auto cpu = t.cpu().contiguous();
        return std::vector<float>(cpu.data_ptr<float>(), cpu.data_ptr<float>() + cpu.numel());
    }
};

TEST_F(CudaSamplerTest, testMarkovSoftmaxRejectsBadOutputRank) {
    const auto              options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    MarkovSoftmaxStepParams params{torch::zeros({1, 4}, options),
                                   torch::zeros({1, 4}, options),
                                   torch::ones({1}, options),
                                   torch::empty({1, 4}, options),
                                   torch::empty({1, 4, 2}, options)};
    EXPECT_ANY_THROW(execMarkovSoftmaxStep(params));
    params.q_contig = torch::empty({1, 4}, options);
    execMarkovSoftmaxStep(params);
    EXPECT_TRUE(torch::equal(params.q_contig, torch::full({1, 4}, 0.25f, options)));
    EXPECT_TRUE(torch::equal(params.q_all_probs, params.q_contig));
}

TEST_F(CudaSamplerTest, testMarkovWrappersRejectDifferentDevices) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "Requires two visible CUDA devices";
    }
    const auto options      = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto       logits       = torch::empty({0, 4}, options);
    const auto other_device = torch::Device(torch::kCUDA, (logits.get_device() + 1) % device_count);
    // Empty batches keep the rejection check safe even if validation regresses.
    for (int argument = 0; argument < 4; ++argument) {
        SCOPED_TRACE(argument);
        MarkovSoftmaxStepParams params{logits,
                                       torch::empty_like(logits),
                                       torch::empty({0}, options),
                                       torch::empty_like(logits),
                                       torch::empty_like(logits)};
        torch::Tensor* inputs[] = {&params.markov_bias, &params.temperature, &params.q_all_probs, &params.q_contig};
        *inputs[argument]       = inputs[argument]->to(other_device);
        EXPECT_ANY_THROW(execMarkovSoftmaxStep(params));
        if (argument < 2) {
            MarkovArgmaxStepParams argmax_params{params.logits, params.markov_bias, params.temperature};
            EXPECT_ANY_THROW(execMarkovArgmaxStep(argmax_params));
        }
    }
}

TEST_F(CudaSamplerTest, testCudaSamplingKernelTopK1) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({1, 1, 1, 1});
    auto top_p_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();
    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 7);
    ASSERT_EQ(output_token_ids_host[23], 7);
}

TEST_F(CudaSamplerTest, testLastColumnWritePreservesCpuAndStridedTokenStorage) {
    constexpr int64_t batch = 3, history = 4, vocab = 7;
    auto              options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto              lengths = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    auto              top_p   = torch::ones({batch}, lengths.options().dtype(torch::kFloat32));
    for (bool cpu_tokens : {false, true}) {
        for (bool strided : {false, true}) {
            for (bool greedy : {false, true}) {
                auto storage =
                    torch::full({batch, history + 1 + (strided ? 2 : 0)},
                                17,
                                lengths.options().pinned_memory(false).device(cpu_tokens ? torch::kCPU : torch::kCUDA));
                auto tokens   = storage.slice(1, 0, history + 1);
                auto expected = storage.clone();
                expected.select(1, history).fill_(2);
                auto logits = torch::full({batch, vocab}, -std::numeric_limits<float>::infinity(), options);
                logits.select(1, 2).zero_();
                auto         top_k = torch::full_like(lengths, greedy ? 1 : 0);
                GreedyParams params{logits,
                                    lengths,
                                    lengths,
                                    tokens,
                                    history,
                                    top_k,
                                    top_p,
                                    top_p,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    false,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::vector<at::Generator>(batch)};
                auto         result = execPreparedGreedyForTest(params);
                at::cuda::getCurrentCUDAStream().synchronize();
                ASSERT_TRUE(!result.success.defined() || result.success.all().item<bool>());
                ASSERT_TRUE(torch::equal(storage.cpu(), expected.cpu()));
            }
        }
    }
}

TEST_F(CudaSamplerTest, testCudaSamplingSeedOffsetSameAcrossBatchRows) {
    constexpr int64_t batch_size = 4;
    constexpr int64_t vocab_size = 6;
    auto              probs      = cudaTensor(
        {
            0.04, 0.08, 0.12, 0.18, 0.24, 0.34, 0.04, 0.08, 0.12, 0.18, 0.24, 0.34,
            0.04, 0.08, 0.12, 0.18, 0.24, 0.34, 0.04, 0.08, 0.12, 0.18, 0.24, 0.34,
        },
        {batch_size, vocab_size});
    auto int_options  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    auto seed_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto output       = torch::empty({batch_size}, int_options);
    auto valid        = torch::empty({batch_size}, bool_options);
    auto seed         = torch::full({batch_size}, 20260706, seed_options);
    auto offset       = torch::full({batch_size}, 128, seed_options);
    auto stream       = at::cuda::getCurrentCUDAStream().stream();

    rtp_llm::top_k_sampling_from_probs(
        probs, output, valid, std::nullopt, std::nullopt, 0, true, seed, 0, offset, 0, (int64_t)stream);
    cudaDeviceSynchronize();

    assertAllValid(valid);
    auto output_host = toHostInt(output);
    for (int64_t i = 1; i < batch_size; ++i) {
        ASSERT_EQ(output_host[0], output_host[i]);
    }
}

// Seed/offset materialization gating: an all-greedy batch consumes no
// entropy (the fused top-1 point-mass kernel and the argmax fallback both
// ignore the Philox tensors), so execSampleGreedy must neither upload them
// nor advance any generator state. A single stochastic row restores the
// historical materialization and advances every row's generator as before.
TEST_F(CudaSamplerTest, testCudaSamplingGreedyBatchSkipsSeedOffsetMaterialization) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});
    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});
    auto top_p_t            = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperature_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});

    std::vector<at::Generator> generators;
    for (size_t i = 0; i < batch_size; ++i) {
        auto gen = torch::make_generator<at::CUDAGeneratorImpl>();
        gen.set_current_seed(static_cast<int64_t>(20261018 + i));
        generators.push_back(gen);
    }
    std::vector<torch::Tensor> states_before;
    for (const auto& gen : generators) {
        states_before.push_back(torch::Tensor(gen.get_state()));
    }

    // Round 1: all-greedy with the production verify shape (private logits,
    // full-width point-mass all_probs) -> fused token-only kernel. Tokens
    // are the row argmaxes; no generator may be touched.
    auto all_probs_t =
        torch::empty({(int64_t)batch_size, 10}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto         top_k_greedy = pinnedIntTensor({1, 1, 1, 1});
    GreedyParams greedy_params({logits_t,
                                input_lengths_t,
                                sequence_lengths_t,
                                output_token_ids_t,
                                step,
                                top_k_greedy,
                                top_p_t,
                                temperature_t,
                                nullopt,
                                nullopt,
                                nullopt,
                                nullopt,
                                false,
                                all_probs_t,
                                nullopt,
                                nullopt,
                                nullopt,
                                generators,
                                /*skip_logits_probs_restore=*/true,
                                /*all_probs_point_mass=*/true});
    auto         greedy_output = execSampleGreedy(prepareSamplingParams(greedy_params, greedy_params.logits.size(0)));
    check_cuda_error();
    ASSERT_TRUE(greedy_output.all_probs_point_mass);
    auto greedy_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(greedy_host[5], 5);
    ASSERT_EQ(greedy_host[11], 2);
    ASSERT_EQ(greedy_host[17], 7);
    ASSERT_EQ(greedy_host[23], 7);
    for (size_t i = 0; i < batch_size; ++i) {
        EXPECT_TRUE(torch::equal(torch::Tensor(generators[i].get_state()), states_before[i]))
            << "all-greedy round advanced generator " << i;
    }

    // Round 2: one stochastic row appears -> the historical seed/offset
    // materialization runs and every row's generator advances.
    auto         top_k_mixed = pinnedIntTensor({1, 1, 3, 2});
    GreedyParams mixed_params({logits_t,
                               input_lengths_t,
                               sequence_lengths_t,
                               output_token_ids_t,
                               step,
                               top_k_mixed,
                               top_p_t,
                               temperature_t,
                               nullopt,
                               nullopt,
                               nullopt,
                               nullopt,
                               false,
                               nullopt,
                               nullopt,
                               nullopt,
                               nullopt,
                               generators});
    execSampleGreedy(prepareSamplingParams(mixed_params, mixed_params.logits.size(0)));
    check_cuda_error();
    for (size_t i = 0; i < batch_size; ++i) {
        EXPECT_FALSE(torch::equal(torch::Tensor(generators[i].get_state()), states_before[i]))
            << "stochastic round did not advance generator " << i;
    }
}

TEST_F(CudaSamplerTest, testCudaSamplingKernelTopK) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({1, 1, 3, 2});
    auto top_p_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();
    ASSERT_TRUE(greedy_output.success.defined());
    ASSERT_EQ(greedy_output.success.numel(), 4);
    auto         success_cpu = greedy_output.success.cpu();
    auto         success     = success_cpu.data_ptr<bool>();
    vector<bool> expect_success{true, true, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], (bool)expect_success[i]);
    }
    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    assertTokenIn(output_token_ids_host, 17, {5, 7, 8});
    assertTokenIn(output_token_ids_host, 23, {7, 8});
}

TEST_F(CudaSamplerTest, testCudaSamplingKernelTopP) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({0, 0, 0, 0});
    auto top_p_t      = pinnedFloatTensor({0.1, 0.1, 0.6, 0.8});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();
    ASSERT_TRUE(greedy_output.success.defined());
    ASSERT_EQ(greedy_output.success.numel(), 4);

    auto         success_cpu = greedy_output.success.cpu();
    auto         success     = success_cpu.data_ptr<bool>();
    vector<bool> expect_success{true, true, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], expect_success[i]);
    }

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    assertTokenIn(output_token_ids_host, 17, {7, 8, 5, 0, 4, 3});
    assertTokenIn(output_token_ids_host, 23, {7, 8, 5, 0, 4, 3, 9, 1});
}

TEST_F(CudaSamplerTest, testCudaSamplingKernelTopKTopP) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({1, 0, 0, 2});
    auto top_p_t      = pinnedFloatTensor({0.2, 0.2, 0.6, 0.6});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();
    ASSERT_TRUE(greedy_output.success.defined());
    ASSERT_EQ(greedy_output.success.numel(), 4);

    auto         success_cpu = greedy_output.success.cpu();
    auto         success     = success_cpu.data_ptr<bool>();
    vector<bool> expect_success{true, true, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], expect_success[i]);
    }
    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 5);
    assertTokenIn(output_token_ids_host, 11, {2, 8});
    assertTokenIn(output_token_ids_host, 17, {7, 8, 5, 0, 4, 3});
    assertTokenIn(output_token_ids_host, 23, {7, 8});
}

TEST_F(CudaSamplerTest, testTopKTopPAllProbsMatchesJointSamplingSupport) {
    auto              logits_t = cudaTensor({std::log(0.5f), std::log(0.3f), std::log(0.19f), std::log(0.01f)}, {1, 4});
    constexpr int64_t step     = 5;
    auto              output_token_ids_t = cudaIntTensor({0, 0, 0, 0, 0, 0}, {1, step + 1});
    auto              sequence_lengths_t = cudaIntTensor({5}, {1});
    auto              input_lengths_t    = cudaIntTensor({-1}, {1});
    auto              top_k_t            = pinnedIntTensor({2});
    auto              top_p_t            = pinnedFloatTensor({0.55f});
    auto              temperature_t      = pinnedFloatTensor({1.0f});
    auto              output_all_probs_t = cudaTensor({0, 0, 0, 0}, {1, 4});
    std::vector<at::Generator> generators{torch::make_generator<at::CUDAGeneratorImpl>()};
    generators[0].set_current_seed(20261022);

    GreedyParams params({logits_t,
                         input_lengths_t,
                         sequence_lengths_t,
                         output_token_ids_t,
                         step,
                         top_k_t,
                         top_p_t,
                         temperature_t,
                         nullopt,
                         nullopt,
                         nullopt,
                         nullopt,
                         false,
                         output_all_probs_t,
                         nullopt,
                         nullopt,
                         nullopt,
                         generators});
    execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();

    ASSERT_VECTOR_NEAR(toHostFloat(output_all_probs_t), std::vector<float>({0.625f, 0.375f, 0, 0}), 1e-5);
    assertTokenIn(toHostInt(output_token_ids_t), step, {0, 1});
}

TEST_F(CudaSamplerTest, testLogprobDistributionIncludesTemperaturePenaltyAndJointFilters) {
    // Temperature yields weights [.5, .3, .19, .01]. Penalizing history token 0
    // halves its weight; joint top-k/top-p retains tokens 1 and 0, with odds 6:5.
    auto raw      = cudaTensor({2 * std::log(.5f),
                                2 * std::log(.3f),
                                2 * std::log(.19f),
                                2 * std::log(.01f),
                                100,
                                100,
                                2 * std::log(.5f),
                                2 * std::log(.3f),
                                2 * std::log(.19f),
                                2 * std::log(.01f),
                                100,
                                100},
                               {2, 6});
    auto expected = cudaTensor({5.f / 11, 6.f / 11, 0, 0, 5.f / 11, 6.f / 11, 0, 0}, {2, 4});
    for (int mode : {0, 1, 2}) {  // no public output, normalized output, ORIGINAL output
        SCOPED_TRACE(mode);
        torch::Tensor reference_tokens, reference_rng;
        for (bool return_scores : {false, true}) {
            GreedyParams params;
            auto         backing             = raw.clone();
            params.logits                    = backing.narrow(1, 0, 4);
            params.input_lengths             = cudaIntTensor({1, 1}, {2});
            params.sequence_lengths          = cudaIntTensor({1, 1}, {2});
            params.token_ids                 = cudaIntTensor({0, 0, 0, 0}, {2, 2});
            params.step                      = 1;
            params.top_k                     = pinnedIntTensor({2, 2});
            params.top_p                     = pinnedFloatTensor({.55f, .55f});
            params.temperature               = pinnedFloatTensor({2, 2});
            params.repetition_penalty        = pinnedFloatTensor({1, 1});
            params.presence_penalty          = pinnedFloatTensor({std::log(2.f), std::log(2.f)});
            params.frequency_penalty         = pinnedFloatTensor({0, 0});
            params.return_original_all_probs = mode == 2;
            if (mode != 0) {
                params.output_all_probs = torch::empty({2, 4}, raw.options());
                params.cum_log_probs    = torch::zeros({2}, raw.options());
            }
            params.return_sampling_probs = return_scores;
            params.generator             = {torch::make_generator<at::CUDAGeneratorImpl>(),
                                            torch::make_generator<at::CUDAGeneratorImpl>()};
            for (auto& generator : params.generator) {
                generator.set_current_seed(20261105);
            }
            auto output = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
            assertAllValid(output.success);
            auto tokens = params.token_ids.select(1, 1).to(torch::kLong);
            if (!return_scores) {
                reference_tokens = tokens.clone();
                reference_rng    = params.generator[0].get_state();
                EXPECT_FALSE(output.sampling_probs.defined());
            } else {
                EXPECT_TRUE(torch::equal(tokens, reference_tokens));
                EXPECT_TRUE(torch::equal(params.generator[0].get_state(), reference_rng));
                EXPECT_TRUE(torch::allclose(output.sampling_probs, expected, 1e-5, 1e-6));
                if (mode == 1) {
                    EXPECT_EQ(output.sampling_probs.data_ptr(), params.output_all_probs->data_ptr());
                }
            }
            if (mode != 0) {
                auto public_probs = expected;
                if (mode == 2) {
                    auto processed = raw.narrow(1, 0, 4) / (2.f + 1e-6f);
                    processed.select(1, 0).sub_(std::log(2.f));
                    public_probs = torch::softmax(processed, -1);
                }
                EXPECT_TRUE(torch::allclose(*params.output_all_probs, public_probs, 1e-5, 1e-6));
                EXPECT_TRUE(torch::allclose(
                    *params.cum_log_probs, public_probs.gather(1, tokens.unsqueeze(1)).squeeze(1).log(), 1e-5, 1e-6));
            }
            EXPECT_TRUE(backing.narrow(1, 4, 2).eq(100).all().item<bool>());
        }
    }
}

TEST_F(CudaSamplerTest, testCompactAndDenseLogprobDistributionsFromFusedTargetSampling) {
    for (int64_t width : {512, 1024, 4096}) {
        SCOPED_TRACE(width);
        for (bool compact : {false, true}) {
            SCOPED_TRACE(compact);
            GreedyParams params;
            params.logits = torch::full(
                {2, 8192}, -std::numeric_limits<float>::infinity(), torch::TensorOptions().device(torch::kCUDA));
            params.logits.select(1, 7).fill_(.5f * std::log(.6f));
            params.logits.select(1, 91).fill_(.5f * std::log(.3f));
            params.logits.select(1, 109).fill_(.5f * std::log(.1f));
            params.input_lengths    = cudaIntTensor({1, 1}, {2});
            params.sequence_lengths = cudaIntTensor({1, 1}, {2});
            params.token_ids        = cudaIntTensor({0, 0, 0, 0}, {2, 2});
            params.step             = 1;
            params.top_k            = pinnedIntTensor({0, 0});
            params.top_p            = pinnedFloatTensor({.75f, .75f});
            params.temperature      = pinnedFloatTensor({.5f, .5f});
            params.output_all_probs = torch::empty({2, compact ? width : 8192}, params.logits.options());
            if (compact) {
                params.output_all_probs_indices =
                    torch::empty({2, width}, params.logits.options().dtype(torch::kInt32));
            }
            params.skip_logits_probs_restore = true;
            params.force_topk                = width;
            params.return_sampling_probs     = true;
            params.generator                 = {torch::make_generator<at::CUDAGeneratorImpl>(),
                                                torch::make_generator<at::CUDAGeneratorImpl>()};
            auto output                      = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
            assertAllValid(output.success);
            EXPECT_EQ(output.sampling_probs.data_ptr(), params.output_all_probs->data_ptr());
            auto actual = output.sampling_probs;
            if (compact) {
                actual = torch::zeros({2, 8192}, params.logits.options());
                actual.scatter_(1, params.output_all_probs_indices->to(torch::kLong), output.sampling_probs);
            }
            auto expected = torch::zeros_like(actual);
            expected.select(1, 7).fill_(2.f / 3);
            expected.select(1, 91).fill_(1.f / 3);
            EXPECT_TRUE(torch::allclose(actual, expected, 1e-5, 1e-6));
            EXPECT_TRUE(
                actual.gather(1, params.token_ids.select(1, 1).to(torch::kLong).unsqueeze(1)).gt(0).all().item<bool>());
        }
    }
}

TEST_F(CudaSamplerTest, testTemperatureFoldAtZeroAndOneMatchesFullLogits) {
    // Mixed zero/unit rows used to send the entire batch through the full-vocabulary temperature kernel.
    for (const std::vector<float>& temperatures :
         {std::vector<float>{0.f, 1.f, .5f}, std::vector<float>{1.f, 1.f, 1.f}}) {
        for (int64_t width : {512, 1024, 4096}) {
            SCOPED_TRACE(width);
            auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
            auto logits  = torch::full({3, 8192}, -std::numeric_limits<float>::infinity(), options);
            logits.select(1, 7).fill_(std::log(.6f));
            logits.select(1, 91).fill_(std::log(.3f));
            logits.select(1, 109).fill_(std::log(.1f));
            torch::Tensor reference;
            for (bool compact : {false, true}) {
                GreedyParams params;
                params.logits           = logits.clone();
                params.input_lengths    = cudaIntTensor({1, 1, 1}, {3});
                params.sequence_lengths = cudaIntTensor({1, 1, 1}, {3});
                params.token_ids        = torch::zeros({3, 2}, options.dtype(torch::kInt32));
                params.step             = 1;
                params.top_k            = pinnedIntTensor({0, 0, 0});
                params.top_p            = pinnedFloatTensor({.95f, .95f, .95f});
                params.temperature      = pinnedFloatTensor(temperatures);
                params.output_all_probs = torch::empty({3, compact ? width : 8192}, options);
                if (compact) {
                    params.output_all_probs_indices = torch::empty({3, width}, options.dtype(torch::kInt32));
                }
                params.skip_logits_probs_restore = true;
                params.force_topk                = compact ? width : 0;
                params.return_sampling_probs     = true;
                for (int i = 0; i < 3; ++i) {
                    params.generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
                }
                auto output = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
                assertAllValid(output.success);
                if (!compact) {
                    reference = output.sampling_probs.clone();
                    continue;
                }
                auto actual = torch::zeros_like(logits);
                actual.scatter_(1, params.output_all_probs_indices->to(torch::kLong), output.sampling_probs);
                EXPECT_TRUE(torch::allclose(actual, reference, 1e-5, 1e-6));
                // Proves that selection scaled only the candidates, not the input logits.
                EXPECT_TRUE(torch::equal(params.logits, logits));
                EXPECT_TRUE(actual.gather(1, params.token_ids.select(1, 1).to(torch::kLong).unsqueeze(1))
                                .gt(0)
                                .all()
                                .item<bool>());
            }
        }
    }
}

TEST_F(CudaSamplerTest, testUnrestrictedTopPReturnsOriginalDistributionForRejection) {
    constexpr int64_t  vocab_size = 4096;
    std::vector<float> logits(vocab_size, -20.0f);
    logits[0]                                     = 2.0f;
    logits[1]                                     = 1.0f;
    logits[2]                                     = 0.0f;
    auto                       logits_t           = cudaTensor(logits, {1, vocab_size});
    auto                       expected           = torch::softmax(logits_t, -1);
    auto                       output_all_probs_t = torch::empty_like(logits_t);
    auto                       output_token_ids_t = cudaIntTensor({0, 0}, {1, 2});
    std::vector<at::Generator> generators{torch::make_generator<at::CUDAGeneratorImpl>()};
    generators[0].set_current_seed(20261023);

    GreedyParams params({logits_t,
                         cudaIntTensor({-1}, {1}),
                         cudaIntTensor({1}, {1}),
                         output_token_ids_t,
                         1,
                         pinnedIntTensor({0}),
                         pinnedFloatTensor({1.0f}),
                         pinnedFloatTensor({1.0f}),
                         nullopt,
                         nullopt,
                         nullopt,
                         nullopt,
                         false,
                         output_all_probs_t,
                         nullopt,
                         nullopt,
                         nullopt,
                         generators});
    auto         output = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();

    ASSERT_TRUE(output.success.cpu().all().item<bool>());
    EXPECT_TRUE(torch::allclose(output_all_probs_t, expected, 1e-6, 1e-7));
    assertTokenIn(toHostInt(output_token_ids_t), 1, {0, 1, 2});
}

TEST_F(CudaSamplerTest, testPureTopPSamplesFromReturnedDistribution) {
    constexpr int64_t batch = 2, vocab = 8192;
    auto              dev    = torch::TensorOptions().device(torch::kCUDA);
    auto              logits = torch::randn({batch, vocab}, dev.dtype(torch::kFloat32));
    logits.select(0, 0).fill_(-std::numeric_limits<float>::infinity());
    logits.index_put_({0, 17}, 4.0f);

    for (int force_topk : {0, 4096}) {
        auto tokens = torch::zeros({batch, 2}, dev.dtype(torch::kInt32));
        auto q      = torch::empty({batch, force_topk > 0 ? force_topk : vocab}, dev.dtype(torch::kFloat32));
        auto candidate_ids =
            force_topk > 0 ? torch::empty({batch, force_topk}, dev.dtype(torch::kInt32)) : torch::Tensor();
        GreedyParams params({logits.clone(),
                             pinnedIntTensor({-1, -1}),
                             pinnedIntTensor({1, 1}),
                             tokens,
                             1,
                             pinnedIntTensor({0, 0}),
                             pinnedFloatTensor({0.5f, 0.95f}),
                             pinnedFloatTensor({1.0f, 1.0f})});
        params.output_all_probs = q;
        if (candidate_ids.defined()) {
            params.output_all_probs_indices = candidate_ids;
        }
        params.skip_logits_probs_restore = true;
        params.force_topk                = force_topk;
        auto generator                   = torch::make_generator<at::CUDAGeneratorImpl>();
        generator.set_current_seed(20261030);
        params.generator = {generator, generator};

        execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
        check_cuda_error();
        EXPECT_TRUE(q.sum(-1).sub(1.0f).abs().lt(1e-4).all().item<bool>());
        if (candidate_ids.defined()) {
            EXPECT_TRUE(candidate_ids.ge(0).logical_and(candidate_ids.lt(vocab)).all().item<bool>());
            auto full_q = torch::zeros({batch, vocab}, q.options());
            full_q.scatter_(1, candidate_ids.to(torch::kLong), q);
            EXPECT_TRUE(full_q.gather(1, tokens.select(1, 1).to(torch::kLong).unsqueeze(1)).gt(0).all().item<bool>());
        } else {
            EXPECT_TRUE(q.gather(1, tokens.select(1, 1).to(torch::kLong).unsqueeze(1)).gt(0).all().item<bool>());
        }
        EXPECT_EQ(tokens.index({0, 1}).item<int32_t>(), 17);
    }
}

TEST_F(CudaSamplerTest, testAirTopPRenormHandlesFlatRowsInMixedBatch) {
    // Regression: the DSpARK packed target-verify path scatters
    // [packed_rows, vocab] logits into a zero-filled dense buffer, so the
    // unverified tail rows reach the sampler as all-zero logits. softmax(0) is
    // exactly uniform, which puts every element of such a row into a single
    // radix bucket. The histogram update must fold a converged warp that maps
    // to one bucket into a single atomic; otherwise the per-element shared
    // atomics serialize and the three radix sweeps cost milliseconds instead
    // of microseconds. This also guards the warp-aggregation fast path against
    // deadlocking on the divergent tail of the vectorized sweep.
    //
    // 8192 is above NUM_BUCKETS and is not the 4096 special case, so this
    // takes the radix path.
    constexpr int64_t  batch = 4, width = 8192;
    const float        flat = 1.0f / static_cast<float>(width);
    std::vector<float> values(batch * width, flat);
    // Rows 0 and 2 are peaked; rows 1 and 3 stay exactly flat.
    for (int64_t row : {int64_t{0}, int64_t{2}}) {
        for (int64_t col = 0; col < width; ++col)
            values[row * width + col] = 0.0f;
        values[row * width]     = 0.5f;
        values[row * width + 1] = 0.3f;
        values[row * width + 2] = 0.2f;
    }
    auto probs  = cudaTensor(values, {batch, width});
    auto output = torch::empty_like(probs);
    auto top_p  = pinnedFloatTensor({0.95f, 0.95f, 0.95f, 0.95f}).to(torch::kCUDA);
    top_p_renorm_probs(probs, output, top_p, 1.0);
    check_cuda_error();

    // A flat row is already normalized inside the nucleus: every element is
    // >= the threshold, so renormalization is the identity.
    for (int64_t row : {int64_t{1}, int64_t{3}}) {
        auto out_row = toHostFloat(output.select(0, row));
        for (int64_t col = 0; col < width; col += 64)
            EXPECT_NEAR(out_row[col], flat, 1e-7f) << "row " << row << " col " << col;
        EXPECT_NEAR(output.select(0, row).sum().item<float>(), 1.0f, 1e-4f);
    }
    // The peaked rows keep the ordinary top-p semantics. 0.95 keeps the whole
    // {0.5, 0.3, 0.2} nucleus, so the renormalized row is unchanged.
    auto expected = cudaTensor({0.5f, 0.3f, 0.2f}, {3});
    for (int64_t row : {int64_t{0}, int64_t{2}}) {
        EXPECT_TRUE(torch::allclose(output.select(0, row).slice(0, 0, 3), expected, 1e-5, 1e-6));
        EXPECT_NEAR(output.select(0, row).sum().item<float>(), 1.0f, 1e-5f);
    }
}

TEST_F(CudaSamplerTest, testAirTopPRenormSkipsUnrestrictedRowsInMixedBatch) {
    constexpr int64_t  vocab_size = 4096;
    std::vector<float> values(2 * vocab_size, 0.0f);
    for (int64_t row = 0; row < 2; ++row) {
        values[row * vocab_size]     = 0.5f;
        values[row * vocab_size + 1] = 0.3f;
        values[row * vocab_size + 2] = 0.2f;
    }
    auto probs  = cudaTensor(values, {2, vocab_size});
    auto output = torch::empty_like(probs);
    auto top_p  = pinnedFloatTensor({1.0f, 0.55f}).to(torch::kCUDA);
    top_p_renorm_probs(probs, output, top_p, 1.0);
    check_cuda_error();

    EXPECT_TRUE(torch::equal(output.select(0, 0), probs.select(0, 0)));
    auto expected = cudaTensor({0.625f, 0.375f, 0.0f}, {3});
    EXPECT_TRUE(torch::allclose(output.select(0, 1).slice(0, 0, 3), expected, 1e-5, 1e-6));
    EXPECT_NEAR(output.select(0, 1).sum().item<float>(), 1.0f, 1e-5f);
}

TEST_F(CudaSamplerTest, testCompactTopPRenorm4096MatchesSortedReference) {
    constexpr int64_t batch = 4, width = 4096;
    torch::manual_seed(20261024);
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto probs   = torch::rand({batch, width}, options);
    probs.div_(probs.sum(-1, true));
    auto top_p  = torch::tensor({0.25f, 0.55f, 0.95f, 1.0f}, options);
    auto output = torch::empty_like(probs);
    top_p_renorm_probs(probs, output, top_p, 1.0);
    check_cuda_error();

    auto        host_probs   = toHostFloat(probs);
    auto        host_output  = toHostFloat(output);
    const float thresholds[] = {0.25f, 0.55f, 0.95f, 1.0f};
    for (int64_t row = 0; row < batch; ++row) {
        if (thresholds[row] == 1.0f) {
            for (int64_t col = 0; col < width; ++col) {
                EXPECT_EQ(host_output[row * width + col], host_probs[row * width + col]);
            }
            continue;
        }
        std::vector<float> sorted(host_probs.begin() + row * width, host_probs.begin() + (row + 1) * width);
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        float mass   = 0.0f;
        float cutoff = 0.0f;
        for (float value : sorted) {
            mass += value;
            cutoff = value;
            if (mass >= thresholds[row])
                break;
        }
        float kept_mass = 0.0f;
        for (int64_t col = 0; col < width; ++col) {
            float value = host_probs[row * width + col];
            if (value >= cutoff)
                kept_mass += value;
        }
        for (int64_t col = 0; col < width; ++col) {
            float value    = host_probs[row * width + col];
            float expected = value >= cutoff ? value / kept_mass : 0.0f;
            EXPECT_NEAR(host_output[row * width + col], expected, 1e-5f);
        }
    }
}

TEST_F(CudaSamplerTest, testCudaSamplingKernelFailed) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({1, 2, 2, 2});
    auto top_p_t      = pinnedFloatTensor({-1.0, -1.0, 0.6, 0.2});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();

    ASSERT_TRUE(greedy_output.success.defined());
    ASSERT_EQ(greedy_output.success.numel(), 4);

    auto         success_cpu = greedy_output.success.cpu();
    auto         success     = success_cpu.data_ptr<bool>();
    vector<bool> expect_success{false, false, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], expect_success[i]);
    }
    auto output_token_ids_host = toHostInt(output_token_ids_t);
    assertTokenIn(output_token_ids_host, 17, {7, 8});
    assertTokenIn(output_token_ids_host, 23, {7, 8});
}

TEST_F(CudaSamplerTest, testBanRepeatNGram) {
    const auto vocab_size = 10;
    const auto batch_size = 4;
    const auto beam_width = 1;

    auto logits_t = cudaTensor(
        {
            0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1,
        },
        {(int64_t)batch_size, (int64_t)vocab_size});

    size_t step               = 8;
    auto   output_token_ids_t = cudaIntTensor(
        {
            0, 2, 3, 4, 5, 0, 0, 2, 0, 1, 2, 3, 3, 3, 1, 2, 3, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 9, 8, 6, 9, 8, 0, 0, 0, 0,
        },
        {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t     = cudaIntTensor({7, 7, 7, 4}, {4});
    auto no_repeat_ngram_size_t = cudaIntTensor({3, 4, 5, 2}, {4});

    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    check_cuda_error();

    std::vector<uint64_t> output_ids_ptrs(batch_size);
    for (int i = 0; i < batch_size; i++) {
        output_ids_ptrs[i] = (uint64_t)(output_token_ids_t.data_ptr<int32_t>() + i * (step + 1));
    }
    auto output_ids_ptrs_t =
        torch::tensor(std::vector<int64_t>(output_ids_ptrs.begin(), output_ids_ptrs.end()), torch::kLong)
            .to(torch::kCUDA);

    tensorrt_llm::kernels::invokeBanRepeatNgram(logits_t.data_ptr<float>(),
                                                (int32_t const**)(output_ids_ptrs_t.data_ptr()),
                                                nullptr,  // finished_buf
                                                nullptr,  // parent_ids_buf
                                                nullptr,  // batch_slot
                                                sequence_lengths_t.data_ptr<int32_t>(),
                                                batch_size,
                                                beam_width,
                                                step,
                                                no_repeat_ngram_size_t.data_ptr<int32_t>(),
                                                vocab_size,
                                                step,
                                                stream);
    check_cuda_error();

    std::vector<int32_t> expcted_ban_token_ids = {3, 3, 1, 6};
    const auto           logits_cpu            = logits_t.cpu();
    for (int i = 0; i < batch_size; i++) {
        auto ban_id = expcted_ban_token_ids[i];
        for (int j = 0; j < vocab_size; j++) {
            if (j == ban_id) {
                EXPECT_EQ(logits_cpu[i][j].item<float>(), -INFINITY);
            } else {
                EXPECT_GT(logits_cpu[i][j].item<float>(), 0.0f);
            }
        }
    }
}

TEST_F(CudaSamplerTest, testPenalty) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1, 0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        },
        {(int64_t)batch_size, 10});
    size_t step               = 5;
    auto   output_token_ids_t = cudaIntTensor({2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0},
                                              {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});
    auto cum_log_probs_t    = cudaTensor({-1.0, -2.0, -3.0, -3.0}, {4});

    auto top_k_t              = pinnedIntTensor({0, 0, 0, 0});
    auto top_p_t              = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperture_t         = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto repetition_penalty_t = pinnedFloatTensor({2.4, 1.0, 1.0, 1.2});
    auto presence_penalty_t   = pinnedFloatTensor({0, 0.6, 0, 0.3});
    auto frequency_penalty_t  = pinnedFloatTensor({0, 0, 0.2, 0.1});

    auto output_all_probs_t = torch::zeros({4, 10}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    std::vector<at::Generator> generator;
    for (int i = 0; i < batch_size; i++) {
        generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
        generator[i].set_current_seed(i + 1);
    }

    GreedyParams params({logits_t,
                         input_lengths_t,
                         sequence_lengths_t,
                         output_token_ids_t,
                         step,
                         top_k_t,
                         top_p_t,
                         temperture_t,
                         repetition_penalty_t,
                         nullopt,
                         cum_log_probs_t,
                         nullopt,
                         false,
                         output_all_probs_t,
                         presence_penalty_t,
                         frequency_penalty_t,
                         nullopt,
                         generator});
    execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 9);
    ASSERT_EQ(output_token_ids_host[11], 5);
    ASSERT_EQ(output_token_ids_host[17], 7);
    ASSERT_EQ(output_token_ids_host[23], 1);

    auto output_all_probs_host = toHostFloat(output_all_probs_t);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0.0693098, 0.0990131, 0.100677,  0.075837,  0.0838128, 0.0926275, 0.102369,  0.113135,
                            0.125034,  0.138184,  0.0703223, 0.0921197, 0.0958792, 0.0769448, 0.0850372, 0.0939806,
                            0.103865,  0.114788,  0.126861,  0.140203,  0.080888,  0.12942,   0.110285,  0.0885056,
                            0.0978138, 0.108101,  0.11947,   0.0885056, 0.0885056, 0.0885056, 0.0715989, 0.0895156,
                            0.0837425, 0.0783417, 0.0865809, 0.0956867, 0.10575,   0.116872,  0.129164,  0.142748}),
        1e-3);
    auto cum_log_probs_host = toHostFloat(cum_log_probs_t);
    ASSERT_NEAR(cum_log_probs_host[0], -2.97917, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[1], -4.36467, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[2], -5.42469, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -5.41334, 1e-3);
    assertCumLogProbMatches(cum_log_probs_host,
                            output_all_probs_host,
                            output_token_ids_host,
                            {-1.0, -2.0, -3.0, -3.0},
                            batch_size,
                            step,
                            10);
}

TEST_F(CudaSamplerTest, testDoSample) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        },
        {(int64_t)batch_size, 10});
    size_t step               = 5;
    auto   output_token_ids_t = cudaIntTensor({2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0},
                                              {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});
    auto cum_log_probs_t    = cudaTensor({-1.0, -2.0, -3.0, -3.0}, {4});

    auto top_k_t      = pinnedIntTensor({2, 2, 2, 2});
    auto top_p_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperture_t = pinnedFloatTensor({2.0, 2.0, 4.0, 4.0});
    // do_sample: bool pinned tensor
    auto do_sample_t = torch::zeros({4}, torch::kBool).pin_memory();
    do_sample_t[0]   = false;
    do_sample_t[1]   = true;
    do_sample_t[2]   = false;
    do_sample_t[3]   = true;

    auto output_all_probs_t = torch::zeros({4, 10}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    std::vector<at::Generator> generator;
    for (int i = 0; i < batch_size; i++) {
        generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
        generator[i].set_current_seed(i + 1);
    }

    GreedyParams params({logits_t,
                         input_lengths_t,
                         sequence_lengths_t,
                         output_token_ids_t,
                         step,
                         top_k_t,
                         top_p_t,
                         temperture_t,
                         nullopt,
                         nullopt,
                         cum_log_probs_t,
                         nullopt,
                         false,
                         output_all_probs_t,
                         nullopt,
                         nullopt,
                         do_sample_t,
                         generator});
    execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 2);
    ASSERT_EQ(output_token_ids_host[11], 1);
    ASSERT_EQ(output_token_ids_host[17], 1);
    ASSERT_EQ(output_token_ids_host[23], 1);
    for (size_t row = 0; row < batch_size; ++row) {
        ASSERT_EQ(std::vector<int32_t>(output_token_ids_host.begin() + row * (step + 1),
                                       output_token_ids_host.begin() + row * (step + 1) + step),
                  std::vector<int32_t>({2, 2, 2, 1, 1}));
    }

    auto output_all_probs_host = toHostFloat(output_all_probs_t);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.477515, 0.522485, 0, 0, 0, 0, 0, 0, 0,
                            0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.488752, 0.511248, 0, 0, 0, 0, 0, 0, 0}),
        1e-3);
    auto cum_log_probs_host = toHostFloat(cum_log_probs_t);
    ASSERT_NEAR(cum_log_probs_host[0], -1.60719, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[1], -2.73916, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[2], -3.78719, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -3.71590, 1e-3);
    assertCumLogProbMatches(cum_log_probs_host,
                            output_all_probs_host,
                            output_token_ids_host,
                            {-1.0, -2.0, -3.0, -3.0},
                            batch_size,
                            step,
                            10);
}

TEST_F(CudaSamplerTest, testTopPOneRenormPreservesUnfilteredDistribution) {
    torch::manual_seed(76103);
    for (int64_t batch : {1, 4, 17, 32}) {
        for (int64_t vocab : {257, 248320}) {
            for (bool mask_padding : {false, true}) {
                auto logits =
                    torch::randn({batch, vocab}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                if (mask_padding)
                    logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto input_lengths =
                    torch::full({batch}, -1, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
                auto seq_lengths = torch::zeros({batch}, input_lengths.options());
                auto top_k       = torch::zeros({batch}, input_lengths.options());
                auto top_p = torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                auto temperature = torch::ones_like(top_p);
                std::vector<torch::Tensor> distributions, samples;
                auto                       cum_log_probs = torch::zeros({batch}, logits.options());
                for (int repetition = 0; repetition < 2; ++repetition) {
                    auto output_probs = torch::empty_like(logits);
                    auto tokens       = torch::zeros({batch, 1}, logits.options().dtype(torch::kInt32));
                    cum_log_probs.zero_();
                    std::vector<at::Generator> generators;
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(711 + i);
                        generators.push_back(generator);
                    }
                    GreedyParams params({logits.clone(),
                                         input_lengths,
                                         seq_lengths,
                                         tokens,
                                         0,
                                         top_k,
                                         top_p,
                                         temperature,
                                         std::nullopt,
                                         std::nullopt,
                                         cum_log_probs,
                                         std::nullopt,
                                         false,
                                         output_probs,
                                         std::nullopt,
                                         std::nullopt,
                                         std::nullopt,
                                         generators});
                    auto         result = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
                    check_cuda_error();
                    ASSERT_TRUE(!result.success.defined() || result.success.all().item<bool>());
                    for (auto& generator : generators) {
                        ASSERT_EQ(generator.get_offset(), 32);
                    }
                    distributions.push_back(output_probs);
                    samples.push_back(tokens);
                }
                ASSERT_TRUE(torch::equal(samples[0], samples[1]));
                ASSERT_TRUE(torch::equal(distributions[0], distributions[1]));
                // Independent FP64 reference: top_p=1 keeps every nonzero probability.
                auto expected = torch::softmax(logits.to(torch::kFloat64), -1).to(torch::kFloat32);
                ASSERT_TRUE(torch::allclose(distributions[1], expected, 1e-6, 1e-8));
                ASSERT_TRUE(torch::equal(distributions[1] == 0, expected == 0));
                ASSERT_TRUE(
                    torch::allclose(distributions[1].sum(-1), torch::ones({batch}, logits.options()), 2e-6, 2e-6));
                auto selected_probs = distributions[1].gather(1, samples[1].to(torch::kLong)).squeeze(1);
                ASSERT_TRUE(torch::allclose(cum_log_probs, selected_probs.log(), 1e-6, 1e-8));
            }
        }
    }
}

TEST_F(CudaSamplerTest, testTopPOneCategoricalPreservesPublicOutputsAndRngReservation) {
    torch::manual_seed(20261019);
    for (int64_t batch : {1, 4, 17, 32, 33}) {
        for (int64_t vocab : {257, 4096, 248320}) {
            for (int distribution : {0, 1, 2}) {
                auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
                auto logits  = torch::randn({batch, vocab}, options);
                if (distribution == 1)
                    logits.zero_();
                if (distribution == 2)
                    logits.mul_(8);
                logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto lengths = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
                auto top_p   = torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                auto top_k   = torch::zeros_like(lengths);
                auto temperature = torch::ones_like(top_p);
                std::vector<torch::Tensor> samples, probabilities, log_probabilities, states;
                for (bool reference : {true, false}) {
                    std::vector<at::Generator> generators;
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(190000 + i);
                        generators.push_back(generator);
                    }
                    for (int generation = 0; generation < 4; ++generation) {
                        auto tokens = torch::zeros({batch, 3}, options.dtype(torch::kInt32));
                        tokens.select(1, 0).fill_(3);
                        tokens.select(1, 1).fill_(5);
                        auto probs = torch::empty_like(logits);
                        auto cum   = torch::full({batch}, -2.0, options);
                        // Return-original avoids conflating AIR renormalization
                        // with this dispatch-only numerical candidate.
                        GreedyParams params({logits.clone(),
                                             lengths,
                                             lengths,
                                             tokens,
                                             2,
                                             top_k,
                                             top_p,
                                             temperature,
                                             std::nullopt,
                                             std::nullopt,
                                             cum,
                                             std::nullopt,
                                             true,
                                             probs,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             generators});
                        auto         result = reference ?
                                                  referenceTopPOneSample(params) :
                                                  execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
                        check_cuda_error();
                        ASSERT_TRUE(result.success.all().item<bool>());
                        ASSERT_TRUE(
                            torch::equal(tokens.slice(1, 0, 2),
                                         torch::tensor({3, 5}, options.dtype(torch::kInt32)).repeat({batch, 1})));
                        ASSERT_TRUE(torch::equal(probs, torch::softmax(logits, -1)));
                        auto selected = probs.gather(1, tokens.slice(1, 2).to(torch::kLong)).squeeze(1);
                        ASSERT_TRUE(torch::allclose(cum, selected.log() - 2, 1e-6, 1e-8));
                        if (reference) {
                            samples.push_back(tokens.clone());
                            probabilities.push_back(probs.clone());
                            log_probabilities.push_back(cum.clone());
                            for (auto& generator : generators)
                                states.push_back(generator.get_state());
                        } else {
                            ASSERT_TRUE(torch::equal(tokens, samples[generation]));
                            ASSERT_TRUE(torch::equal(probs, probabilities[generation]));
                            ASSERT_TRUE(torch::equal(cum, log_probabilities[generation]));
                            for (int64_t i = 0; i < batch; ++i)
                                ASSERT_TRUE(torch::equal(generators[i].get_state(), states[generation * batch + i]));
                        }
                    }
                }
            }
        }
    }
}




TEST_F(CudaSamplerTest, testCompactGreedyOutputAcrossVocabulariesAndBatchSizes) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    for (int64_t batch : {1, 2, 32, 33, 64}) {
        for (int64_t vocab : {7, 257, 4096, 248320}) {
            auto expected_ids = torch::arange(batch, options.dtype(torch::kInt64)).remainder(vocab);
            auto logits       = torch::zeros({batch, vocab}, options);
            logits.scatter_(1, expected_ids.unsqueeze(1), 1.0);
            auto top_k = torch::ones({batch}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
            auto ones  = torch::ones({batch}, top_k.options().dtype(torch::kFloat32));
            for (int64_t history : {0, 1, 128, 4096}) {
                for (bool cpu_tokens : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << "batch=" << batch << " vocab=" << vocab
                                                      << " history=" << history << " cpu_tokens=" << cpu_tokens);
                    auto tokens =
                        torch::full({batch, history + 1},
                                    3,
                                    options.dtype(torch::kInt32).device(cpu_tokens ? torch::kCPU : torch::kCUDA));
                    auto expected = tokens.clone();
                    expected.select(1, history).copy_(expected_ids.to(expected.options()));
                    auto         lengths = torch::full_like(top_k, history);
                    GreedyParams params{logits.clone(),
                                        lengths,
                                        lengths,
                                        tokens,
                                        history,
                                        top_k,
                                        ones,
                                        ones,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        false,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::vector<at::Generator>(batch)};
                    auto         result = execPreparedGreedyForTest(params);
                    at::cuda::getCurrentCUDAStream().synchronize();
                    ASSERT_TRUE(!result.success.defined());
                    ASSERT_TRUE(torch::equal(tokens, expected));
                }
            }
        }
    }
}

TEST_F(CudaSamplerTest, testCompactOutputPreservesTokensProbabilitiesRngAndPenaltyFallback) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261019);
    for (int64_t batch : {1, 2, 4, 17, 32, 33, 64}) {
        for (int64_t vocab : {257, 4096, 248320}) {
            for (int penalty_case = 0; penalty_case < 8; ++penalty_case) {
                SCOPED_TRACE(::testing::Message()
                             << "batch=" << batch << " vocab=" << vocab << " penalty_case=" << penalty_case);
                const bool  probability_outputs = penalty_case != 7;
                const float gain                = penalty_case == 0 ? 0.0f : 1.0f;
                auto        logits              = torch::randn({batch, vocab}, options) * gain;
                logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto lengths =
                    torch::full({batch}, 128, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
                auto top_p = torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                auto top_k = torch::zeros_like(lengths);
                // Exercise argmax both with probability outputs and its fast path.
                if (penalty_case >= 6)
                    top_k.fill_(1);
                auto temperature = torch::full_like(top_p, 0.7);
                auto repetition  = torch::ones_like(top_p);
                auto presence    = torch::zeros_like(top_p);
                auto frequency   = torch::zeros_like(top_p);
                auto ngram       = torch::zeros_like(lengths);
                if (penalty_case == 2)
                    repetition.fill_(1.2);
                if (penalty_case == 3)
                    presence.fill_(0.2);
                if (penalty_case == 4)
                    frequency.fill_(0.3);
                if (penalty_case == 5)
                    ngram.fill_(2);
                std::vector<torch::Tensor> samples, probabilities, log_probabilities, states, in_place;
                // CPU tokens exercise full-history staging; contiguous CUDA tokens
                // exercise compact output on every CUDA architecture and vocabulary.
                for (bool reference : {true, false}) {
                    std::vector<at::Generator> generators;
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(210000 + i);
                        generators.push_back(generator);
                    }
                    for (int generation = 0; generation < 4; ++generation) {
                        auto tokens = torch::zeros(
                            {batch, 129}, options.dtype(torch::kInt32).device(reference ? torch::kCPU : torch::kCUDA));
                        tokens.select(1, 0).fill_(3);
                        tokens.select(1, 1).fill_(5);
                        tokens.select(1, 128).fill_(17);
                        auto         prefix  = tokens.slice(1, 0, 128).clone();
                        auto         probs   = torch::empty_like(logits);
                        auto         cum     = torch::full({batch}, -2.0, options);
                        auto         mutated = logits.clone();
                        GreedyParams params({mutated,
                                             lengths,
                                             lengths,
                                             tokens,
                                             128,
                                             top_k,
                                             top_p,
                                             temperature,
                                             penalty_case == 0 ? std::nullopt : std::make_optional(repetition),
                                             ngram,
                                             probability_outputs ? std::make_optional(cum) : std::nullopt,
                                             std::nullopt,
                                             false,
                                             probability_outputs ? std::make_optional(probs) : std::nullopt,
                                             presence,
                                             frequency,
                                             std::nullopt,
                                             generators});
                        auto         result = execPreparedGreedyForTest(params);
                        check_cuda_error();
                        ASSERT_TRUE(!result.success.defined() || result.success.all().item<bool>());
                        ASSERT_TRUE(torch::equal(tokens.slice(1, 0, 128), prefix));
                        if (reference) {
                            samples.push_back(tokens.cpu());
                            probabilities.push_back(probs.clone());
                            log_probabilities.push_back(cum.clone());
                            in_place.push_back(mutated.clone());
                            for (auto& generator : generators)
                                states.push_back(generator.get_state());
                        } else {
                            ASSERT_TRUE(torch::equal(tokens.cpu(), samples[generation]));
                            if (probability_outputs) {
                                ASSERT_TRUE(torch::equal(probs, probabilities[generation]));
                            }
                            ASSERT_TRUE(torch::equal(mutated, in_place[generation]));
                            ASSERT_TRUE(torch::equal(cum, log_probabilities[generation]));
                            for (int64_t i = 0; i < batch; ++i)
                                ASSERT_TRUE(torch::equal(generators[i].get_state(), states[generation * batch + i]));
                        }
                    }
                }
            }
        }
    }
}

// The fused greedy top-1 path (skip_logits_probs_restore + all top_k==1 +
// full-width all_probs) must produce bit-identical tokens and renormalized
// all_probs as the legacy softmax -> argmax -> top_k_renorm_probs chain,
// while leaving the private logits buffer raw (no probs writeback).
TEST_F(CudaSamplerTest, testGreedyTopKFusedSkipRestoreMatchesLegacy) {
    const int64_t batch_size = 4, vocab_size = 64, step = 5;
    torch::manual_seed(20261017);
    auto base_logits =
        (torch::randn({batch_size, vocab_size}, torch::TensorOptions().dtype(torch::kFloat32)) * 3.0f).to(torch::kCUDA);
    auto expected_tokens = base_logits.argmax(-1);  // tie-free random rows

    const std::vector<int32_t> initial_tokens(batch_size * (step + 1), 7);

    auto run_case = [&](bool skip_restore, torch::Tensor& logits_out_state) {
        auto logits_t           = base_logits.clone();
        auto output_token_ids_t = cudaIntTensor(initial_tokens, {batch_size, step + 1});
        auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
        auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});
        auto top_k_t            = pinnedIntTensor({1, 1, 1, 1});
        auto top_p_t            = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
        auto temperature_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
        auto all_probs_t =
            torch::zeros({batch_size, vocab_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        std::vector<at::Generator> generator;
        generator.resize(batch_size);

        GreedyParams params({logits_t,
                             input_lengths_t,
                             sequence_lengths_t,
                             output_token_ids_t,
                             (size_t)step,
                             top_k_t,
                             top_p_t,
                             temperature_t,
                             nullopt,
                             nullopt,
                             nullopt,
                             nullopt,
                             false,
                             all_probs_t,
                             nullopt,
                             nullopt,
                             nullopt,
                             generator});
        params.skip_logits_probs_restore = skip_restore;
        auto greedy_output               = execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
        check_cuda_error();
        EXPECT_FALSE(greedy_output.success.defined());  // all_top_k_one marks all succeeded

        logits_out_state = logits_t;
        return std::make_pair(toHostInt(output_token_ids_t), toHostFloat(all_probs_t));
    };

    torch::Tensor logits_legacy, logits_fused;
    auto [tokens_legacy, all_probs_legacy] = run_case(false, logits_legacy);
    auto [tokens_fused, all_probs_fused]   = run_case(true, logits_fused);

    // Tokens land in the step column and equal the row argmax in both runs.
    for (int64_t b = 0; b < batch_size; ++b) {
        ASSERT_EQ(tokens_legacy[b * (step + 1) + step], expected_tokens[b].item<int64_t>());
        ASSERT_EQ(tokens_fused[b * (step + 1) + step], tokens_legacy[b * (step + 1) + step]);
    }
    // Renormalized all_probs: bit-identical one-hot(argmax) in both runs.
    ASSERT_EQ(all_probs_fused, all_probs_legacy);
    for (int64_t b = 0; b < batch_size; ++b) {
        float row_sum = 0.0f;
        for (int64_t v = 0; v < vocab_size; ++v) {
            row_sum += all_probs_fused[b * vocab_size + v];
        }
        ASSERT_FLOAT_EQ(row_sum, 1.0f);
        ASSERT_FLOAT_EQ(all_probs_fused[b * vocab_size + expected_tokens[b].item<int64_t>()], 1.0f);
    }
    // Legacy restores softmax probs into logits; the fused path leaves it raw.
    auto legacy_row_sum = logits_legacy.sum(-1).cpu();
    for (int64_t b = 0; b < batch_size; ++b) {
        ASSERT_NEAR(legacy_row_sum[b].item<float>(), 1.0f, 1e-4);
    }
    ASSERT_TRUE(torch::equal(logits_fused.cpu(), base_logits.cpu()));
}

// The fused path is gated to the exact all_top_k_one renorm contract: with a
// mixed top_k batch the flagged caller must still get the legacy chain
// (including the probs writeback into logits).
TEST_F(CudaSamplerTest, testGreedyTopKFusedGateFallsBackForMixedTopK) {
    const int64_t batch_size = 4, vocab_size = 64, step = 5;
    torch::manual_seed(20261017);
    auto logits_t =
        (torch::randn({batch_size, vocab_size}, torch::TensorOptions().dtype(torch::kFloat32)) * 3.0f).to(torch::kCUDA);
    auto raw_logits         = logits_t.clone();
    auto output_token_ids_t = cudaIntTensor(std::vector<int32_t>(batch_size * (step + 1), 7), {batch_size, step + 1});
    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});
    auto top_k_t            = pinnedIntTensor({1, 3, 1, 1});
    auto top_p_t            = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperature_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto all_probs_t =
        torch::zeros({batch_size, vocab_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({logits_t,
                         input_lengths_t,
                         sequence_lengths_t,
                         output_token_ids_t,
                         (size_t)step,
                         top_k_t,
                         top_p_t,
                         temperature_t,
                         nullopt,
                         nullopt,
                         nullopt,
                         nullopt,
                         false,
                         all_probs_t,
                         nullopt,
                         nullopt,
                         nullopt,
                         generator});
    params.skip_logits_probs_restore = true;  // flag set, but gate must reject (top_k != 1 row)
    execSampleGreedy(prepareSamplingParams(params, params.logits.size(0)));
    check_cuda_error();

    // Legacy chain ran: logits now holds softmax probs (writeback happened).
    auto row_sum = logits_t.sum(-1).cpu();
    for (int64_t b = 0; b < batch_size; ++b) {
        ASSERT_NEAR(row_sum[b].item<float>(), 1.0f, 1e-4);
    }
    ASSERT_FALSE(torch::equal(logits_t.cpu(), raw_logits.cpu()));
}

// Direct token scatter path (CUDA-resident history, no active penalties): the
// last-column scatter must leave the whole [batch, step+1] matrix bit-identical
// to the legacy transpose-out/copy-back round trip (executed by the CPU-history
// run), with the fused greedy top-1 kernel as the sampler — the exact shape of
// the MTP target-verify production path.
TEST_F(CudaSamplerTest, testTokenHistoryDirectScatterMatchesHostStaging) {
    const int64_t batch_size = 4, vocab_size = 64, step = 5;
    torch::manual_seed(20261018);
    auto base_logits =
        (torch::randn({batch_size, vocab_size}, torch::TensorOptions().dtype(torch::kFloat32)) * 3.0f).to(torch::kCUDA);
    // Distinct values everywhere so a write to the wrong column is caught.
    std::vector<int32_t> history(batch_size * (step + 1));
    for (size_t i = 0; i < history.size(); ++i) {
        history[i] = static_cast<int32_t>(1000 + i * 7);
    }

    auto run_case = [&](bool cuda_history) {
        auto logits_t           = base_logits.clone();
        auto token_ids_t        = torch::tensor(history, torch::kInt32).reshape({batch_size, step + 1});
        token_ids_t             = cuda_history ? token_ids_t.to(torch::kCUDA) : token_ids_t.clone();
        auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
        auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});
        auto top_k_t            = pinnedIntTensor({1, 1, 1, 1});
        auto top_p_t            = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
        auto temperature_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
        auto all_probs_t =
            torch::zeros({batch_size, vocab_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        std::vector<at::Generator> generator;
        generator.resize(batch_size);

        GreedyParams params({logits_t,
                             input_lengths_t,
                             sequence_lengths_t,
                             token_ids_t,
                             (size_t)step,
                             top_k_t,
                             top_p_t,
                             temperature_t,
                             nullopt,
                             nullopt,
                             nullopt,
                             nullopt,
                             false,
                             all_probs_t,
                             nullopt,
                             nullopt,
                             nullopt,
                             generator});
        params.skip_logits_probs_restore = true;  // fused deterministic top-1
        execPreparedGreedyForTest(params);
        check_cuda_error();
        cudaDeviceSynchronize();
        return std::make_pair(toHostInt(token_ids_t), toHostFloat(all_probs_t));
    };

    auto [tokens_direct, probs_direct] = run_case(true);   // direct scatter
    auto [tokens_staged, probs_staged] = run_case(false);  // host staging
    ASSERT_EQ(tokens_direct, tokens_staged);
    ASSERT_EQ(probs_direct, probs_staged);

    auto expected_tokens = base_logits.argmax(-1).cpu();
    for (int64_t b = 0; b < batch_size; ++b) {
        ASSERT_EQ(tokens_direct[b * (step + 1) + step], expected_tokens[b].item<int64_t>());
        for (int64_t s = 0; s < step; ++s) {
            ASSERT_EQ(tokens_direct[b * (step + 1) + s], history[b * (step + 1) + s]);
        }
    }
}

// Inactive penalty tensors (repetition==1.0, presence/frequency==0.0) must not
// disable the direct scatter path, and the stochastic top-k branch must still
// produce bit-identical tokens and renormalized all_probs versus the legacy
// round trip (same per-row generator seeds on both runs).
TEST_F(CudaSamplerTest, testTokenHistoryDirectScatterInactivePenaltiesParity) {
    const int64_t batch_size = 4, vocab_size = 32, step = 5;
    torch::manual_seed(20261019);
    auto base_logits =
        (torch::randn({batch_size, vocab_size}, torch::TensorOptions().dtype(torch::kFloat32)) * 2.0f).to(torch::kCUDA);
    std::vector<int32_t> history(batch_size * (step + 1));
    for (size_t i = 0; i < history.size(); ++i) {
        history[i] = static_cast<int32_t>(100 + i * 3);
    }

    auto run_case = [&](bool cuda_history) {
        auto logits_t             = base_logits.clone();
        auto token_ids_t          = torch::tensor(history, torch::kInt32).reshape({batch_size, step + 1});
        token_ids_t               = cuda_history ? token_ids_t.to(torch::kCUDA) : token_ids_t.clone();
        auto sequence_lengths_t   = cudaIntTensor({5, 5, 5, 5}, {4});
        auto input_lengths_t      = cudaIntTensor({-1, -1, -1, -1}, {4});
        auto top_k_t              = pinnedIntTensor({2, 2, 2, 2});
        auto top_p_t              = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
        auto temperature_t        = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
        auto repetition_penalty_t = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
        auto presence_penalty_t   = pinnedFloatTensor({0.0, 0.0, 0.0, 0.0});
        auto frequency_penalty_t  = pinnedFloatTensor({0.0, 0.0, 0.0, 0.0});
        auto all_probs_t =
            torch::zeros({batch_size, vocab_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        std::vector<at::Generator> generator;
        for (int i = 0; i < batch_size; i++) {
            generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
            generator[i].set_current_seed(i + 1);
        }

        GreedyParams params({logits_t,
                             input_lengths_t,
                             sequence_lengths_t,
                             token_ids_t,
                             (size_t)step,
                             top_k_t,
                             top_p_t,
                             temperature_t,
                             repetition_penalty_t,
                             nullopt,
                             nullopt,
                             nullopt,
                             false,
                             all_probs_t,
                             presence_penalty_t,
                             frequency_penalty_t,
                             nullopt,
                             generator});
        execPreparedGreedyForTest(params);
        check_cuda_error();
        cudaDeviceSynchronize();
        return std::make_pair(toHostInt(token_ids_t), toHostFloat(all_probs_t));
    };

    auto [tokens_direct, probs_direct] = run_case(true);
    auto [tokens_staged, probs_staged] = run_case(false);
    ASSERT_EQ(tokens_direct, tokens_staged);
    ASSERT_EQ(probs_direct, probs_staged);
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < step; ++s) {
            ASSERT_EQ(tokens_direct[b * (step + 1) + s], history[b * (step + 1) + s]);
        }
        ASSERT_GE(tokens_direct[b * (step + 1) + step], 0);
        ASSERT_LT(tokens_direct[b * (step + 1) + step], vocab_size);
    }
}

// Ban-repeat-ngram reads the row-major device history, so it must keep working
// on the direct scatter path (no transposed clone is built): the repeated
// bigram (3, 4) bans token 4 and the argmax falls back to token 9, identical
// to the legacy round-trip run.
TEST_F(CudaSamplerTest, testTokenHistoryDirectScatterNgramStillBans) {
    const int64_t batch_size = 4, vocab_size = 16, step = 5;
    // Per-row history [1, 3, 4, 2, 3, x]: the trailing 3 repeats the 3 at
    // index 1, so with n=2 the successor token 4 is banned.
    std::vector<int32_t> history;
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int32_t t : {1, 3, 4, 2, 3, 0}) {
            history.push_back(t);
        }
    }
    std::vector<float> logits_flat(batch_size * vocab_size, 0.0f);
    for (int64_t b = 0; b < batch_size; ++b) {
        logits_flat[b * vocab_size + 4] = 10.0f;  // argmax without the ban
        logits_flat[b * vocab_size + 9] = 5.0f;   // expected after the ban
        logits_flat[b * vocab_size + 1] = 1.0f;
    }
    auto base_logits = torch::tensor(logits_flat, torch::kFloat32).reshape({batch_size, vocab_size}).to(torch::kCUDA);

    auto run_case = [&](bool cuda_history) {
        auto logits_t           = base_logits.clone();
        auto token_ids_t        = torch::tensor(history, torch::kInt32).reshape({batch_size, step + 1});
        token_ids_t             = cuda_history ? token_ids_t.to(torch::kCUDA) : token_ids_t.clone();
        auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
        auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});
        auto top_k_t            = pinnedIntTensor({1, 1, 1, 1});
        auto top_p_t            = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
        auto temperature_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
        auto ngram_t            = pinnedIntTensor({2, 2, 2, 2});
        auto all_probs_t =
            torch::zeros({batch_size, vocab_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        std::vector<at::Generator> generator;
        generator.resize(batch_size);

        GreedyParams params({logits_t,
                             input_lengths_t,
                             sequence_lengths_t,
                             token_ids_t,
                             (size_t)step,
                             top_k_t,
                             top_p_t,
                             temperature_t,
                             nullopt,
                             ngram_t,
                             nullopt,
                             nullopt,
                             false,
                             all_probs_t,
                             nullopt,
                             nullopt,
                             nullopt,
                             generator});
        execPreparedGreedyForTest(params);
        check_cuda_error();
        cudaDeviceSynchronize();
        return std::make_pair(toHostInt(token_ids_t), toHostFloat(all_probs_t));
    };

    auto [tokens_direct, probs_direct] = run_case(true);   // direct scatter + ngram
    auto [tokens_staged, probs_staged] = run_case(false);  // host staging + ngram
    ASSERT_EQ(tokens_direct, tokens_staged);
    ASSERT_EQ(probs_direct, probs_staged);
    for (int64_t b = 0; b < batch_size; ++b) {
        ASSERT_EQ(tokens_direct[b * (step + 1) + step], 9);  // 4 banned, fell back
        for (int64_t s = 0; s < step; ++s) {
            ASSERT_EQ(tokens_direct[b * (step + 1) + s], history[b * (step + 1) + s]);
        }
    }
}
