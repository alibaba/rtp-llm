#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using namespace rtp_llm;

namespace {

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
    auto         greedy_output = execSampleGreedy(params);
    check_cuda_error();
    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 7);
    ASSERT_EQ(output_token_ids_host[23], 7);
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
                                nullptr,
                                /*skip_logits_probs_restore=*/true,
                                /*all_probs_point_mass=*/true});
    auto         greedy_output = execSampleGreedy(greedy_params);
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
    execSampleGreedy(mixed_params);
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
    auto         greedy_output = execSampleGreedy(params);
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
    auto         greedy_output = execSampleGreedy(params);
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
    auto         greedy_output = execSampleGreedy(params);
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
    auto         greedy_output = execSampleGreedy(params);
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
    execSampleGreedy(params);
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
    execSampleGreedy(params);
    check_cuda_error();

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 2);
    ASSERT_EQ(output_token_ids_host[11], 1);
    ASSERT_EQ(output_token_ids_host[17], 1);
    ASSERT_EQ(output_token_ids_host[23], 1);

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
        auto greedy_output               = execSampleGreedy(params);
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
    execSampleGreedy(params);
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
TEST_F(CudaSamplerTest, testTokenHistoryDirectScatterMatchesLegacyRoundTrip) {
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
        execSampleGreedy(params);
        check_cuda_error();
        cudaDeviceSynchronize();
        return std::make_pair(toHostInt(token_ids_t), toHostFloat(all_probs_t));
    };

    auto [tokens_direct, probs_direct] = run_case(true);   // direct scatter
    auto [tokens_legacy, probs_legacy] = run_case(false);  // legacy round trip
    ASSERT_EQ(tokens_direct, tokens_legacy);
    ASSERT_EQ(probs_direct, probs_legacy);

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
        execSampleGreedy(params);
        check_cuda_error();
        cudaDeviceSynchronize();
        return std::make_pair(toHostInt(token_ids_t), toHostFloat(all_probs_t));
    };

    auto [tokens_direct, probs_direct] = run_case(true);
    auto [tokens_legacy, probs_legacy] = run_case(false);
    ASSERT_EQ(tokens_direct, tokens_legacy);
    ASSERT_EQ(probs_direct, probs_legacy);
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
        execSampleGreedy(params);
        check_cuda_error();
        cudaDeviceSynchronize();
        return std::make_pair(toHostInt(token_ids_t), toHostFloat(all_probs_t));
    };

    auto [tokens_direct, probs_direct] = run_case(true);   // direct scatter + ngram
    auto [tokens_legacy, probs_legacy] = run_case(false);  // legacy round trip + ngram
    ASSERT_EQ(tokens_direct, tokens_legacy);
    ASSERT_EQ(probs_direct, probs_legacy);
    for (int64_t b = 0; b < batch_size; ++b) {
        ASSERT_EQ(tokens_direct[b * (step + 1) + step], 9);  // 4 banned, fell back
        for (int64_t s = 0; s < step; ++s) {
            ASSERT_EQ(tokens_direct[b * (step + 1) + s], history[b * (step + 1) + s]);
        }
    }
}
