#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
using namespace std;
using namespace rtp_llm;

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

TEST_F(CudaSamplerTest, testFlashinferKernelTopK1) {
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

TEST_F(CudaSamplerTest, testFlashinferKernelTopK) {
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

    auto top_k_t      = pinnedIntTensor({1, 1, 0, 2});
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
    ASSERT_EQ(output_token_ids_host[17], 0);
    ASSERT_EQ(output_token_ids_host[23], 8);
}

TEST_F(CudaSamplerTest, testFlashinferKernelTopP) {
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
    ASSERT_EQ(output_token_ids_host[17], 0);
    ASSERT_EQ(output_token_ids_host[23], 1);
}

TEST_F(CudaSamplerTest, testFlashinferKernelTopKTopP) {
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
    ASSERT_EQ(output_token_ids_host[17], 0);
    ASSERT_EQ(output_token_ids_host[23], 8);
}

TEST_F(CudaSamplerTest, testFlashinferKernelFailed) {
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
    ASSERT_EQ(output_token_ids_host[17], 7);
    ASSERT_EQ(output_token_ids_host[23], 8);
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
                         output_all_probs_t,
                         presence_penalty_t,
                         frequency_penalty_t,
                         nullopt,
                         generator});
    execSampleGreedy(params);
    check_cuda_error();

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    auto cum_log_probs_host    = toHostFloat(cum_log_probs_t);
    ASSERT_EQ(output_token_ids_host[5], 9);
    ASSERT_EQ(output_token_ids_host[11], 5);
    ASSERT_EQ(output_token_ids_host[17], 7);
    ASSERT_EQ(output_token_ids_host[23], 1);
    ASSERT_NEAR(cum_log_probs_host[0], -2.97917, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[1], -4.36467, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[2], -5.42469, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -5.41334, 1e-3);

    auto output_all_probs_host = toHostFloat(output_all_probs_t);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0.0693098, 0.0990131, 0.100677,  0.075837,  0.0838128, 0.0926275, 0.102369,  0.113135,
                            0.125034,  0.138184,  0.0703223, 0.0921197, 0.0958792, 0.0769448, 0.0850372, 0.0939806,
                            0.103865,  0.114788,  0.126861,  0.140203,  0.080888,  0.12942,   0.110285,  0.0885056,
                            0.0978138, 0.108101,  0.11947,   0.0885056, 0.0885056, 0.0885056, 0.0715989, 0.0895156,
                            0.0837425, 0.0783417, 0.0865809, 0.0956867, 0.10575,   0.116872,  0.129164,  0.142748}),
        1e-3);
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
                         output_all_probs_t,
                         nullopt,
                         nullopt,
                         do_sample_t,
                         generator});
    execSampleGreedy(params);
    check_cuda_error();

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    auto cum_log_probs_host    = toHostFloat(cum_log_probs_t);
    ASSERT_EQ(output_token_ids_host[5], 1);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 1);
    ASSERT_EQ(output_token_ids_host[23], 1);
    ASSERT_NEAR(cum_log_probs_host[0], -1.78719, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[1], -2.64916, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[2], -3.78719, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -3.7159, 1e-3);

    auto output_all_probs_host = toHostFloat(output_all_probs_t);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.477515, 0.522485, 0, 0, 0, 0, 0, 0, 0,
                            0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.488752, 0.511248, 0, 0, 0, 0, 0, 0, 0}),
        1e-3);
}
