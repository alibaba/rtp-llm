#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/split_softmax.h"
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
    const char* previous = std::getenv("RTP_LLM_TOP_P_ONE_CATEGORICAL");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_TOP_P_ONE_CATEGORICAL", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_TOP_P_ONE_CATEGORICAL");
        }
    } restore{saved};
    torch::manual_seed(20261019);
    for (int64_t batch : {1, 4, 17, 32}) {
        for (int64_t vocab : {257, 248320}) {
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
                for (const char* flag : {"0", "1"}) {
                    setenv("RTP_LLM_TOP_P_ONE_CATEGORICAL", flag, 1);
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
                        auto         result = execSampleGreedy(params);
                        check_cuda_error();
                        ASSERT_TRUE(result.success.all().item<bool>());
                        ASSERT_TRUE(
                            torch::equal(tokens.slice(1, 0, 2),
                                         torch::tensor({3, 5}, options.dtype(torch::kInt32)).repeat({batch, 1})));
                        ASSERT_TRUE(torch::equal(probs, torch::softmax(logits, -1)));
                        auto selected = probs.gather(1, tokens.slice(1, 2).to(torch::kLong)).squeeze(1);
                        ASSERT_TRUE(torch::allclose(cum, selected.log() - 2, 1e-6, 1e-8));
                        if (flag[0] == '0') {
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


TEST_F(CudaSamplerTest, testSplitSoftmaxMatchesFloat64ReferenceAndNonfiniteContract) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261019);
    for (int64_t rows : {1, 3, 4, 8, 17, 32}) {
        for (int64_t vocab : {257, 4093, 248320}) {
            for (float gain : {0.0f, 1.0f, 20.0f, 100.0f}) {
                auto logits = torch::randn({rows, vocab}, options) * gain;
                logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto expected      = torch::softmax(logits.to(torch::kFloat64), -1).to(torch::kFloat32);
                auto original_fp32 = torch::softmax(logits, -1);
                rtp_llm::splitSoftmaxInplace(logits);
                if (vocab == 248320) {
                    ASSERT_TRUE(torch::equal(logits, original_fp32));
                }
                ASSERT_TRUE(torch::allclose(logits, expected, 3e-6, 1e-8));
                ASSERT_TRUE(torch::allclose(logits.sum(-1), torch::ones({rows}, options), 3e-6, 1e-8));
            }
        }
    }
    // Contiguous does not imply a 16-byte aligned storage offset.
    auto storage         = torch::randn({3 * 248320 + 1}, options);
    auto sentinel        = storage.slice(0, 0, 1).clone();
    auto unaligned       = storage.slice(0, 1).reshape({3, 248320});
    auto exact_unaligned = torch::softmax(unaligned, -1);
    rtp_llm::splitSoftmaxInplace(unaligned);
    ASSERT_TRUE(torch::equal(unaligned, exact_unaligned));
    ASSERT_TRUE(torch::equal(storage.slice(0, 0, 1), sentinel));
    for (float value : {-std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        auto logits = torch::zeros({3, 4093}, options);
        if (value < 0)
            logits.fill_(value);
        else
            logits.select(1, 1040).fill_(value);
        auto expected = torch::softmax(logits, -1);
        rtp_llm::splitSoftmaxInplace(logits);
        ASSERT_TRUE(torch::allclose(logits, expected, 3e-6, 1e-8, true));
    }
}

TEST_F(CudaSamplerTest, testSplitSoftmaxPreservesSeededTokensAndRngWithCloseProbabilityOutputs) {
    const char* previous = std::getenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
        }
    } restore{saved};
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261019);
    for (int64_t batch : {1, 4, 17, 32}) {
        for (int64_t vocab : {257, 248320}) {
            for (float gain : {0.0f, 1.0f, 8.0f}) {
                auto logits = torch::randn({batch, vocab}, options) * gain;
                logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto lengths = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
                auto top_p   = torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                auto top_k   = torch::zeros_like(lengths);
                auto temperature = torch::ones_like(top_p);
                std::vector<torch::Tensor> samples, probabilities, log_probabilities, states, in_place;
                for (const char* flag : {"0", "1"}) {
                    setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", flag, 1);
                    std::vector<at::Generator> generators;
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(210000 + i);
                        generators.push_back(generator);
                    }
                    for (int generation = 0; generation < 4; ++generation) {
                        auto tokens = torch::zeros({batch, 3}, options.dtype(torch::kInt32));
                        tokens.select(1, 0).fill_(3);
                        tokens.select(1, 1).fill_(5);
                        auto         probs   = torch::empty_like(logits);
                        auto         cum     = torch::full({batch}, -2.0, options);
                        auto         mutated = logits.clone();
                        GreedyParams params({mutated,
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
                                             false,
                                             probs,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             generators});
                        auto         result = execSampleGreedy(params);
                        check_cuda_error();
                        ASSERT_TRUE(result.success.all().item<bool>());
                        if (flag[0] == '0') {
                            samples.push_back(tokens.clone());
                            probabilities.push_back(probs.clone());
                            log_probabilities.push_back(cum.clone());
                            in_place.push_back(mutated.clone());
                            for (auto& generator : generators)
                                states.push_back(generator.get_state());
                        } else {
                            ASSERT_TRUE(torch::equal(tokens, samples[generation]));
                            ASSERT_TRUE(torch::allclose(probs, probabilities[generation], 3e-6, 1e-8));
                            ASSERT_TRUE(torch::allclose(mutated, in_place[generation], 3e-6, 1e-8));
                            ASSERT_TRUE(torch::allclose(cum, log_probabilities[generation], 3e-6, 1e-6));
                            for (int64_t i = 0; i < batch; ++i)
                                ASSERT_TRUE(torch::equal(generators[i].get_state(), states[generation * batch + i]));
                        }
                    }
                }
            }
        }
    }
}


TEST_F(CudaSamplerTest, testSplitPointMassRejectionMatchesOriginalForAllAcceptancePaths) {
    const char* previous = std::getenv("RTP_LLM_SPLIT_REJECTION");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SPLIT_REJECTION", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SPLIT_REJECTION");
        }
    } restore{saved};
    torch::manual_seed(20261019);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    for (int64_t batch : {1, 3, 8}) {
        for (int64_t stride : {1, 5}) {
            auto probs        = torch::softmax(torch::randn({batch, 4, 248320}, options), -1);
            auto draft        = torch::randint(0, 248320, {batch, 3}, options.dtype(torch::kInt32));
            auto target       = torch::zeros({batch * 4, stride}, options.dtype(torch::kInt32));
            auto expected     = torch::empty({batch, 4}, options.dtype(torch::kInt32));
            auto actual       = torch::empty_like(expected);
            auto expected_len = torch::empty({batch}, options.dtype(torch::kInt32));
            auto actual_len   = torch::empty_like(expected_len);
            for (int generation = 0; generation < 128; ++generation) {
                const int pos     = generation % 4;
                auto      uniform = torch::rand({batch, 4}, options);
                uniform.slice(1, 0, pos).zero_();
                if (pos < 3)
                    uniform.select(1, pos).fill_(1.0f);
                if (generation % 8 == 0)
                    uniform.select(1, std::min(pos + 1, 3)).zero_();
                if (generation % 8 == 1)
                    uniform.select(1, std::min(pos + 1, 3)).fill_(1.0f);
                auto do_sample = torch::ones({batch}, options.dtype(torch::kBool));
                if (generation % 3 == 0)
                    do_sample.select(0, 0).fill_(false);
                target.select(1, stride - 1).copy_(torch::argmax(probs, -1).flatten().to(torch::kInt32));
                for (int i = 0; i < pos; ++i)
                    target.select(1, stride - 1).reshape({batch, 4}).select(1, i).copy_(draft.select(1, i));
                for (int variant = 0; variant < 2; ++variant) {
                    setenv("RTP_LLM_SPLIT_REJECTION", variant ? "1" : "0", 1);
                    auto output  = variant ? actual : expected;
                    auto lengths = variant ? actual_len : expected_len;
                    output.fill_(-999);
                    lengths.fill_(-999);
                    execRejectionSampling(
                        {torch::Tensor(), draft, uniform, probs, target, output, lengths, do_sample, true});
                }
                ASSERT_TRUE(torch::equal(actual, expected))
                    << "batch=" << batch << " stride=" << stride << " generation=" << generation;
                ASSERT_TRUE(torch::equal(actual_len, expected_len));
            }
            // Zero residual mass follows the original last-vocabulary fallback.
            probs.zero_();
            auto uniform    = torch::ones({batch, 4}, options);
            auto stochastic = torch::ones({batch}, options.dtype(torch::kBool));
            setenv("RTP_LLM_SPLIT_REJECTION", "0", 1);
            execRejectionSampling(
                {torch::Tensor(), draft, uniform, probs, target, expected, expected_len, stochastic, true});
            setenv("RTP_LLM_SPLIT_REJECTION", "1", 1);
            execRejectionSampling(
                {torch::Tensor(), draft, uniform, probs, target, actual, actual_len, stochastic, true});
            ASSERT_TRUE(torch::equal(actual, expected));
            ASSERT_TRUE(torch::equal(actual_len, expected_len));
        }
    }
}


TEST_F(CudaSamplerTest, testSplitDenseRejectionMatchesOriginalForAllAcceptancePaths) {
    const char* previous = std::getenv("RTP_LLM_SPLIT_REJECTION");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SPLIT_REJECTION", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SPLIT_REJECTION");
        }
    } restore{saved};
    torch::manual_seed(20261019);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    for (int64_t batch : {1, 3, 8}) {
        for (int64_t stride : {1, 5}) {
            auto probs       = torch::softmax(torch::randn({batch, 4, 248320}, options), -1);
            auto draft_probs = torch::softmax(torch::randn({batch, 3, 248320}, options), -1);
            auto draft       = torch::argmax(draft_probs, -1).to(torch::kInt32);
            // Dense q differs from p throughout the vocabulary. Set a known
            // rejection ratio at each proposed token while retaining normalized rows.
            auto draft_index = draft.to(torch::kLong).unsqueeze(-1);
            probs.slice(1, 0, 3).scatter_(2, draft_index, draft_probs.gather(2, draft_index) * 0.125);
            probs.div_(probs.sum(-1, true));
            auto target       = torch::zeros({batch * 4, stride}, options.dtype(torch::kInt32));
            auto expected     = torch::empty({batch, 4}, options.dtype(torch::kInt32));
            auto actual       = torch::empty_like(expected);
            auto expected_len = torch::empty({batch}, options.dtype(torch::kInt32));
            auto actual_len   = torch::empty_like(expected_len);
            for (int generation = 0; generation < 128; ++generation) {
                const int pos     = generation % 4;
                auto      uniform = torch::rand({batch, 4}, options);
                uniform.slice(1, 0, pos).zero_();
                if (pos < 3)
                    uniform.select(1, pos).fill_(1.0f);
                if (generation % 8 == 0)
                    uniform.select(1, std::min(pos + 1, 3)).zero_();
                if (generation % 8 == 1)
                    uniform.select(1, std::min(pos + 1, 3)).fill_(1.0f);
                auto do_sample = torch::ones({batch}, options.dtype(torch::kBool));
                if (generation % 3 == 0)
                    do_sample.select(0, 0).fill_(false);
                target.select(1, stride - 1).copy_(torch::argmax(probs, -1).flatten().to(torch::kInt32));
                for (int i = 0; i < pos; ++i)
                    target.select(1, stride - 1).reshape({batch, 4}).select(1, i).copy_(draft.select(1, i));
                for (int variant = 0; variant < 2; ++variant) {
                    setenv("RTP_LLM_SPLIT_REJECTION", variant ? "1" : "0", 1);
                    auto output  = variant ? actual : expected;
                    auto lengths = variant ? actual_len : expected_len;
                    output.fill_(-999);
                    lengths.fill_(-999);
                    execRejectionSampling(
                        {draft_probs, draft, uniform, probs, target, output, lengths, do_sample, false});
                }
                ASSERT_TRUE(torch::equal(actual, expected))
                    << "batch=" << batch << " stride=" << stride << " generation=" << generation;
                ASSERT_TRUE(torch::equal(actual_len, expected_len));
            }
            // Zero residual mass follows the original last-vocabulary fallback.
            probs.zero_();
            auto uniform    = torch::ones({batch, 4}, options);
            auto stochastic = torch::ones({batch}, options.dtype(torch::kBool));
            setenv("RTP_LLM_SPLIT_REJECTION", "0", 1);
            execRejectionSampling(
                {draft_probs, draft, uniform, probs, target, expected, expected_len, stochastic, false});
            setenv("RTP_LLM_SPLIT_REJECTION", "1", 1);
            execRejectionSampling({draft_probs, draft, uniform, probs, target, actual, actual_len, stochastic, false});
            ASSERT_TRUE(torch::equal(actual, expected));
            ASSERT_TRUE(torch::equal(actual_len, expected_len));
        }
    }
}


TEST_F(CudaSamplerTest, testTiledTemperatureMatchesNativeBitsAndUnalignedViews) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto stream  = at::cuda::getCurrentCUDAStream().stream();
    torch::manual_seed(20261020);
    for (int batch : {1, 3, 4, 17, 32}) {
        for (int offset : {0, 1}) {
            auto storage = torch::empty({batch * 248320 + offset}, options);
            auto actual  = storage.slice(0, offset).reshape({batch, 248320});
            for (float temp : {0.0f, 0.01f, 0.7f, 1.0f, 2.0f, 1.e20f}) {
                actual.normal_();
                actual.slice(1, 0, 1).fill_(std::numeric_limits<float>::denorm_min());
                actual.slice(1, 1, 2).fill_(-std::numeric_limits<float>::denorm_min());
                actual.slice(1, 2, 3).fill_(std::numeric_limits<float>::infinity());
                actual.slice(1, 3, 4).fill_(-std::numeric_limits<float>::infinity());
                actual.slice(1, 4, 5).fill_(std::numeric_limits<float>::quiet_NaN());
                actual.slice(1, 5, 6).fill_(-0.0f);
                auto expected     = actual.clone();
                auto temperatures = torch::full({batch}, temp, options);
                if (batch > 1)
                    temperatures.slice(0, 1, 2).fill_(0.31f);
                invokeBatchApplyTemperaturePenalty<float>(
                    expected.data_ptr<float>(), nullptr, temperatures.data_ptr<float>(), batch, 248320, 248320, stream);
                invokeTiledTemperaturePenalty(
                    actual.data_ptr<float>(), temperatures.data_ptr<float>(), batch, 248320, stream);
                check_cuda_error();
                ASSERT_TRUE(torch::equal(actual.view(torch::kInt32), expected.view(torch::kInt32)));
            }
        }
    }
}

TEST_F(CudaSamplerTest, testTiledTemperaturePreservesSeededTokensProbabilitiesAndRng) {
    const char* previous = std::getenv("RTP_LLM_SAMPLER_TILED_TEMPERATURE");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SAMPLER_TILED_TEMPERATURE", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_TILED_TEMPERATURE");
        }
    } restore{saved};
    const char* softmax_flag = std::getenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
    struct RestoreSoftmax {
        std::optional<std::string> flag;
        ~RestoreSoftmax() {
            if (flag)
                setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", flag->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
        }
    } restore_softmax{softmax_flag == nullptr ? std::optional<std::string>() :
                                                std::make_optional(std::string(softmax_flag))};
    setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", "1", 1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261019);
    for (int64_t batch : {1, 4, 17, 32}) {
        for (int64_t vocab : {257, 248320}) {
            for (float gain : {0.0f, 1.0f, 8.0f}) {
                auto logits = torch::randn({batch, vocab}, options) * gain;
                logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto lengths = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
                auto top_p   = torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                auto top_k   = torch::zeros_like(lengths);
                auto temperature = torch::full_like(top_p, 0.7);
                std::vector<torch::Tensor> samples, probabilities, log_probabilities, states, in_place;
                for (const char* flag : {"0", "1"}) {
                    setenv("RTP_LLM_SAMPLER_TILED_TEMPERATURE", flag, 1);
                    std::vector<at::Generator> generators;
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(210000 + i);
                        generators.push_back(generator);
                    }
                    for (int generation = 0; generation < 4; ++generation) {
                        auto tokens = torch::zeros({batch, 3}, options.dtype(torch::kInt32));
                        tokens.select(1, 0).fill_(3);
                        tokens.select(1, 1).fill_(5);
                        auto         probs   = torch::empty_like(logits);
                        auto         cum     = torch::full({batch}, -2.0, options);
                        auto         mutated = logits.clone();
                        GreedyParams params({mutated,
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
                                             false,
                                             probs,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             generators});
                        auto         result = execSampleGreedy(params);
                        check_cuda_error();
                        ASSERT_TRUE(result.success.all().item<bool>());
                        if (flag[0] == '0') {
                            samples.push_back(tokens.clone());
                            probabilities.push_back(probs.clone());
                            log_probabilities.push_back(cum.clone());
                            in_place.push_back(mutated.clone());
                            for (auto& generator : generators)
                                states.push_back(generator.get_state());
                        } else {
                            ASSERT_TRUE(torch::equal(tokens, samples[generation]));
                            ASSERT_TRUE(torch::equal(probs, probabilities[generation]));
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


TEST_F(CudaSamplerTest, testCompactOutputPreservesTokensProbabilitiesRngAndPenaltyFallback) {
    const char* previous = std::getenv("RTP_LLM_SAMPLER_COMPACT_OUTPUT");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SAMPLER_COMPACT_OUTPUT", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_COMPACT_OUTPUT");
        }
    } restore{saved};
    const char* softmax_flag = std::getenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
    struct RestoreSoftmax {
        std::optional<std::string> flag;
        ~RestoreSoftmax() {
            if (flag)
                setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", flag->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
        }
    } restore_softmax{softmax_flag == nullptr ? std::optional<std::string>() :
                                                std::make_optional(std::string(softmax_flag))};
    setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", "1", 1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261019);
    for (int64_t batch : {1, 4, 17, 32}) {
        for (int64_t vocab : {257, 248320}) {
            for (int penalty_case = 0; penalty_case < 6; ++penalty_case) {
                const float gain   = penalty_case == 0 ? 0.0f : 1.0f;
                auto        logits = torch::randn({batch, vocab}, options) * gain;
                logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto lengths =
                    torch::full({batch}, 128, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
                auto top_p = torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                auto top_k = torch::zeros_like(lengths);
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
                for (const char* flag : {"0", "1"}) {
                    setenv("RTP_LLM_SAMPLER_COMPACT_OUTPUT", flag, 1);
                    std::vector<at::Generator> generators;
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(210000 + i);
                        generators.push_back(generator);
                    }
                    for (int generation = 0; generation < 4; ++generation) {
                        auto tokens = torch::zeros({batch, 129}, options.dtype(torch::kInt32));
                        tokens.select(1, 0).fill_(3);
                        tokens.select(1, 1).fill_(5);
                        tokens.select(1, 128).fill_(17);
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
                                             cum,
                                             std::nullopt,
                                             false,
                                             probs,
                                             presence,
                                             frequency,
                                             std::nullopt,
                                             generators});
                        auto         result = execSampleGreedy(params);
                        check_cuda_error();
                        ASSERT_TRUE(result.success.all().item<bool>());
                        if (flag[0] == '0') {
                            samples.push_back(tokens.clone());
                            probabilities.push_back(probs.clone());
                            log_probabilities.push_back(cum.clone());
                            in_place.push_back(mutated.clone());
                            for (auto& generator : generators)
                                states.push_back(generator.get_state());
                        } else {
                            ASSERT_TRUE(torch::equal(tokens, samples[generation]));
                            ASSERT_TRUE(torch::equal(probs, probabilities[generation]));
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
