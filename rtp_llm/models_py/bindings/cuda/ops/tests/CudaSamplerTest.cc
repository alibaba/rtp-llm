#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <limits>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using namespace rtp_llm;

namespace {

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
                auto         result = execSampleGreedy(params);
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
    execSampleGreedy(params);
    check_cuda_error();

    ASSERT_VECTOR_NEAR(toHostFloat(output_all_probs_t), std::vector<float>({0.625f, 0.375f, 0, 0}), 1e-5);
    assertTokenIn(toHostInt(output_token_ids_t), step, {0, 1});
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
                    auto         result = execSampleGreedy(params);
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
                        auto         result = reference ? referenceTopPOneSample(params) : execSampleGreedy(params);
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
                    auto         result = execSampleGreedy(params);
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
                        auto         result = execSampleGreedy(params);
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
