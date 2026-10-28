#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/greedy_top1_renorm.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
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

void assertSparseTopKPreservesDistribution(const torch::Tensor&                            logits,
                                           const std::tuple<torch::Tensor, torch::Tensor>& selected) {
    const auto& values  = std::get<0>(selected);
    const auto& indices = std::get<1>(selected);
    ASSERT_EQ(indices.scalar_type(), torch::kInt64);
    ASSERT_EQ(indices.dim(), 2);
    ASSERT_EQ(indices.size(0), logits.size(0));
    ASSERT_EQ(indices.size(1), 4096);
    ASSERT_EQ(values.sizes(), indices.sizes());

    // Check on the host before gather/scatter so a broken selector fails without poisoning CUDA.
    auto        indices_cpu = indices.cpu().contiguous();
    const auto* ids         = indices_cpu.data_ptr<int64_t>();
    for (int64_t row = 0; row < indices.size(0); ++row) {
        SCOPED_TRACE(row);
        std::vector<int64_t> sorted(ids + row * 4096, ids + (row + 1) * 4096);
        std::sort(sorted.begin(), sorted.end());
        ASSERT_GE(sorted.front(), 0);
        ASSERT_LT(sorted.back(), logits.size(1));
        ASSERT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end());
    }

    ASSERT_TRUE(torch::equal(values, logits.gather(1, indices)));
    auto probabilities = torch::softmax(values, -1);
    auto restored      = torch::zeros_like(logits);
    restored.scatter_(1, indices, probabilities);
    EXPECT_TRUE(torch::equal(restored, torch::softmax(logits, -1)));
    EXPECT_TRUE(restored.select(1, logits.size(1) - 1).eq(1.0f).all().item<bool>());
    EXPECT_TRUE(restored.sum(-1).eq(1.0f).all().item<bool>());
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

TEST_F(CudaSamplerTest, testDeepSelect4096ReturnsOriginalValues) {
    const auto* properties = at::cuda::getDeviceProperties(0);
    if (!torch_ext::deepselect_fp32_available() || properties->major != 10
        || (properties->minor != 0 && properties->minor != 3)) {
        GTEST_SKIP() << "DeepSelect FP32 requires SM100 or SM103";
    }
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int64_t vocab_size : {4096, 129280}) {
        auto logits = torch::randn({2, vocab_size}, options);
        logits.select(1, 0).fill_(std::numeric_limits<float>::quiet_NaN());
        logits.select(1, 1).fill_(std::numeric_limits<float>::infinity());
        logits.select(1, vocab_size - 1).fill_(-std::numeric_limits<float>::infinity());
        auto selected = torch_ext::deepselect_fp32_topk(logits, 4096);
        auto expected = logits.gather(1, std::get<1>(selected));
        EXPECT_TRUE(torch::equal(std::get<0>(selected).view(torch::kInt32), expected.view(torch::kInt32)))
            << "vocab_size=" << vocab_size;
    }
}

TEST_F(CudaSamplerTest, testDeepSelect4096SparseLogitsKeepValidUniqueIndices) {
    const auto* properties = at::cuda::getDeviceProperties(0);
    if (!torch_ext::deepselect_fp32_available() || properties->major != 10
        || (properties->minor != 0 && properties->minor != 3)) {
        GTEST_SKIP() << "DeepSelect FP32 requires SM100 or SM103";
    }
    constexpr int64_t vocab_size = 129280;
    const auto        options    = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int64_t batch_size : {6, 12, 16}) {
        SCOPED_TRACE(batch_size);
        for (float masked_logit : {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::max()}) {
            SCOPED_TRACE(masked_logit);
            auto logits = torch::full({batch_size, vocab_size}, masked_logit, options);
            logits.select(1, vocab_size - 1).zero_();
            auto temperature = torch::full({batch_size}, 0.5f, options);
            auto scaled      = logits / temperature.unsqueeze(1);
            ASSERT_TRUE(scaled.select(1, 0).eq(-std::numeric_limits<float>::infinity()).all().item<bool>());

            {
                SCOPED_TRACE("ordinary");
                assertSparseTopKPreservesDistribution(scaled, torch_ext::deepselect_fp32_topk(scaled, 4096));
            }
            {
                SCOPED_TRACE("prefilter");
                assertSparseTopKPreservesDistribution(scaled, torch_ext::deepselect_fp32_topk_prefilter(scaled));
            }
            {
                SCOPED_TRACE("markov_prefilter");
                auto bias = torch::zeros_like(logits, options.dtype(torch::kBFloat16));
                assertSparseTopKPreservesDistribution(
                    scaled, torch_ext::deepselect_fp32_markov_topk_prefilter(logits, bias, temperature));
            }
        }
    }
}

TEST_F(CudaSamplerTest, testDeepSelect4096SparseAndBoundaryWidths) {
    const auto* properties = at::cuda::getDeviceProperties(0);
    if (!torch_ext::deepselect_fp32_available() || properties->major != 10
        || (properties->minor != 0 && properties->minor != 3)) {
        GTEST_SKIP() << "DeepSelect FP32 requires SM100 or SM103";
    }
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int64_t vocab : {8192, 128256, 129280, 130048, 131072}) {
        for (int64_t finite_count : {1, 4095, 4096, 4097}) {
            SCOPED_TRACE(vocab);
            SCOPED_TRACE(finite_count);
            std::vector<float> host(vocab, -std::numeric_limits<float>::infinity());
            host[vocab - 1] = 5000.0f;
            for (int64_t i = 1; i < finite_count; ++i) {
                host[i - 1] = static_cast<float>(5000 - i);
            }
            auto logits   = cudaTensor(host, {1, vocab});
            auto selected = torch_ext::deepselect_fp32_topk(logits, 4096);
            if (finite_count == 1) {
                assertSparseTopKPreservesDistribution(logits, torch_ext::deepselect_fp32_topk_prefilter(logits));
                auto bias        = torch::zeros_like(logits, options.dtype(torch::kBFloat16));
                auto temperature = torch::ones({1}, options);
                assertSparseTopKPreservesDistribution(
                    logits, torch_ext::deepselect_fp32_markov_topk_prefilter(logits, bias, temperature));
            }
            auto                 indices = std::get<1>(selected).cpu().contiguous();
            std::vector<int64_t> ids(indices.data_ptr<int64_t>(), indices.data_ptr<int64_t>() + 4096);
            std::sort(ids.begin(), ids.end());
            ASSERT_GE(ids.front(), 0);
            ASSERT_LT(ids.back(), vocab);
            ASSERT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
            EXPECT_TRUE(torch::equal(std::get<0>(selected), logits.gather(1, std::get<1>(selected))));
            EXPECT_EQ(torch::isfinite(std::get<0>(selected)).sum().item<int64_t>(), std::min(finite_count, 4096L));
            for (int64_t i = 0; i < std::min(finite_count, 4096L); ++i) {
                EXPECT_TRUE(std::binary_search(ids.begin(), ids.end(), i == 0 ? vocab - 1 : i - 1));
            }
        }
    }

    // Exercise an unaligned logical width and different valid ends per row
    // while the backing row stride still satisfies the TMA alignment contract.
    auto backing = torch::full({2, 130304}, -std::numeric_limits<float>::infinity(), options);
    auto logits  = backing.narrow(1, 0, 130049);
    logits.select(1, 7).fill_(5.0f);
    auto ends    = cudaIntTensor({130049, 128257}, {2});
    auto indices = torch::empty({2, 4096}, options.dtype(torch::kInt32));
    auto values  = torch::empty({2, 4096}, options);
    torch_ext::deepselect_fp32(logits, ends, indices, false, &values);
    auto ids_cpu = indices.cpu().contiguous();
    for (int64_t row = 0; row < 2; ++row) {
        std::vector<int32_t> ids(ids_cpu.data_ptr<int32_t>() + row * 4096,
                                 ids_cpu.data_ptr<int32_t>() + (row + 1) * 4096);
        std::sort(ids.begin(), ids.end());
        ASSERT_GE(ids.front(), 0);
        ASSERT_LT(ids.back(), ends[row].item<int32_t>());
        ASSERT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
    }
    EXPECT_TRUE(torch::equal(values, logits.gather(1, indices.to(torch::kLong))));
}

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
    params.q_contig = torch::Tensor();
    params.q_all_probs.zero_();
    execMarkovSoftmaxStep(params);
    EXPECT_TRUE(torch::equal(params.q_all_probs, torch::full({1, 4}, 0.25f, options)));
}

TEST_F(CudaSamplerTest, testMarkovScaledLogitsMatchesEagerForStridedProposal) {
    const auto options     = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto       proposal    = torch::randn({3, 5, 513}, options);
    auto       logits      = proposal.select(1, 2);
    auto       temperature = cudaTensor({0.7f, 1.0f, 1.3f}, {3});
    ASSERT_FALSE(logits.is_contiguous());
    for (auto dtype : {torch::kFloat32, torch::kBFloat16, torch::kFloat16}) {
        auto bias     = torch::randn({3, 513}, options).to(dtype);
        auto expected = (logits + bias.to(torch::kFloat32)).div(temperature.unsqueeze(1));
        auto actual   = torch_ext::markov_scaled_logits(logits, bias, temperature);
        EXPECT_TRUE(actual.is_contiguous());
        EXPECT_LE((actual - expected).abs().max().item<float>(), 1e-6f) << "bias dtype " << dtype;
    }
}

TEST_F(CudaSamplerTest, testSamplingFromStridedDraftProbabilitiesMatchesContiguous) {
    constexpr int64_t  vocab_size = 4096;
    std::vector<float> values(2 * 3 * vocab_size, 0.0f);
    values[vocab_size]         = 0.5f;
    values[vocab_size + 1]     = 0.5f;
    values[4 * vocab_size + 2] = 0.5f;
    values[4 * vocab_size + 3] = 0.5f;
    auto q                     = cudaTensor(values, {2, 3, vocab_size});
    auto strided               = q.select(1, 1);
    ASSERT_FALSE(strided.is_contiguous());
    auto options            = torch::TensorOptions().device(torch::kCUDA);
    auto sampled_strided    = torch::empty({2}, options.dtype(torch::kInt32));
    auto sampled_contiguous = torch::empty_like(sampled_strided);
    auto sampled_unaligned  = torch::empty_like(sampled_strided);
    auto valid_strided      = torch::empty({2}, options.dtype(torch::kBool));
    auto valid_contiguous   = torch::empty_like(valid_strided);
    auto valid_unaligned    = torch::empty_like(valid_strided);
    auto padded             = torch::zeros({2, 3, vocab_size + 1}, options);
    auto unaligned          = padded.select(1, 1).slice(1, 0, vocab_size);
    unaligned.copy_(strided);

    sampling_from_probs(
        strided, sampled_strided, valid_strided, std::nullopt, true, std::nullopt, 123, std::nullopt, 0);
    sampling_from_probs(strided.contiguous(),
                        sampled_contiguous,
                        valid_contiguous,
                        std::nullopt,
                        true,
                        std::nullopt,
                        123,
                        std::nullopt,
                        0);
    sampling_from_probs(
        unaligned, sampled_unaligned, valid_unaligned, std::nullopt, true, std::nullopt, 123, std::nullopt, 0);
    check_cuda_error();

    EXPECT_TRUE(torch::equal(sampled_strided, sampled_contiguous));
    EXPECT_TRUE(torch::equal(sampled_unaligned, sampled_contiguous));
    EXPECT_TRUE(valid_strided.all().item<bool>());
    EXPECT_TRUE(valid_contiguous.all().item<bool>());
    EXPECT_TRUE(valid_unaligned.all().item<bool>());
}

TEST_F(CudaSamplerTest, testUnrestrictedTopPSamplingMatchesUnfilteredSampling) {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto probs   = torch::rand({32, 4096}, options);
    probs.div_(probs.sum(-1, true));
    auto sampled_top_p      = torch::empty({32}, options.dtype(torch::kInt32));
    auto sampled_unfiltered = torch::empty_like(sampled_top_p);
    auto valid_top_p        = torch::empty({32}, options.dtype(torch::kBool));
    auto valid_unfiltered   = torch::empty_like(valid_top_p);

    top_p_sampling_from_probs(
        probs, sampled_top_p, valid_top_p, std::nullopt, std::nullopt, 1.0, true, std::nullopt, 123, std::nullopt, 0);
    sampling_from_probs(
        probs, sampled_unfiltered, valid_unfiltered, std::nullopt, true, std::nullopt, 123, std::nullopt, 0);
    check_cuda_error();

    EXPECT_TRUE(torch::equal(sampled_top_p, sampled_unfiltered));
    EXPECT_TRUE(torch::equal(valid_top_p, valid_unfiltered));
    EXPECT_TRUE(valid_unfiltered.all().item<bool>());
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
    auto         output = execSampleGreedy(params);
    check_cuda_error();

    ASSERT_TRUE(output.success.cpu().all().item<bool>());
    EXPECT_TRUE(torch::allclose(output_all_probs_t, expected, 1e-6, 1e-7));
    assertTokenIn(toHostInt(output_token_ids_t), 1, {0, 1, 2});
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

TEST_F(CudaSamplerTest, testForceTopKReturnsVocabularyProbabilities) {
    constexpr int64_t  vocab_size = 16;
    std::vector<float> logits(vocab_size, -10.0f);
    logits[1] = 1.0f;
    logits[3] = 3.0f;
    logits[5] = 5.0f;
    logits[7] = 4.0f;
    auto run  = [&](int force_topk, int request_topk, float top_p) {
        auto         scores = cudaTensor(logits, {1, vocab_size});
        auto         tokens = cudaIntTensor({0, 0}, {1, 2});
        auto         probs  = torch::zeros_like(scores);
        GreedyParams p({scores,
                        cudaIntTensor({-1}, {1}),
                        cudaIntTensor({1}, {1}),
                        tokens,
                        1,
                        pinnedIntTensor({request_topk}),
                        pinnedFloatTensor({top_p}),
                        pinnedFloatTensor({1.0f})});
        p.output_all_probs          = probs;
        p.skip_logits_probs_restore = true;
        p.force_topk                = force_topk;
        auto generator              = torch::make_generator<at::CUDAGeneratorImpl>();
        generator.set_current_seed(20261023);
        p.generator = {generator};
        execSampleGreedy(p);
        check_cuda_error();
        return std::make_pair(toHostInt(tokens)[1], toHostFloat(probs));
    };

    auto [token, probs] = run(4, 0, 1.0f);
    auto expected       = torch::softmax(torch::tensor(logits), -1);
    ASSERT_GE(token, 0);
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(probs[i], expected[i].item<float>(), 1e-5f);
    }
    auto [limited_token, limited_probs] = run(4, 2, 1.0f);
    ASSERT_TRUE(limited_token == 5 || limited_token == 7);
    EXPECT_EQ(limited_probs[1], 0.0f);
    EXPECT_EQ(limited_probs[3], 0.0f);
    auto [nucleus_token, nucleus_probs] = run(4, 0, 0.8f);
    ASSERT_TRUE(nucleus_token == 5 || nucleus_token == 7);
    EXPECT_EQ(nucleus_probs[1], 0.0f);
    EXPECT_EQ(nucleus_probs[3], 0.0f);
    auto [joint_token, joint_probs] = run(4, 2, 0.8f);
    ASSERT_TRUE(joint_token == 5 || joint_token == 7);
    auto expected_joint = torch::softmax(torch::tensor({5.0f, 4.0f}), -1);
    EXPECT_NEAR(joint_probs[5], expected_joint[0].item<float>(), 1e-5f);
    EXPECT_NEAR(joint_probs[7], expected_joint[1].item<float>(), 1e-5f);
    EXPECT_EQ(joint_probs[1], 0.0f);
    EXPECT_EQ(joint_probs[3], 0.0f);
    // Explicit joint filtering retains the full-vocabulary top-p boundary.
    auto [tight_token, tight_probs]       = run(4, 2, 0.7f);
    auto [baseline_token, baseline_probs] = run(0, 2, 0.7f);
    ASSERT_TRUE(tight_token == 5 || tight_token == 7);
    ASSERT_TRUE(baseline_token == 5 || baseline_token == 7);
    EXPECT_EQ(tight_probs, baseline_probs);
    EXPECT_EQ(tight_probs[1], 0.0f);
    EXPECT_EQ(tight_probs[3], 0.0f);
    auto [full_token, full_probs] = run(0, 0, 1.0f);
    EXPECT_GT(full_probs[0], 0.0f);
}

TEST_F(CudaSamplerTest, testForceTopK4096SparseMaskPreservesProbabilityMass) {
    const auto* properties = at::cuda::getDeviceProperties(0);
    if (!torch_ext::deepselect_fp32_available() || properties->major != 10
        || (properties->minor != 0 && properties->minor != 3)) {
        GTEST_SKIP() << "DeepSelect FP32 requires SM100 or SM103";
    }
    constexpr int64_t vocab_size = 129280;
    const auto        options    = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int64_t batch_size : {6, 12, 16}) {
        SCOPED_TRACE(batch_size);
        // A packed FP32 mask uses -FLT_MAX; temperature below one turns it into -INF.
        auto scores = torch::full({batch_size, vocab_size}, -std::numeric_limits<float>::max(), options);
        scores.select(1, vocab_size - 1).zero_();
        auto         tokens = torch::zeros({batch_size, 2}, options.dtype(torch::kInt32));
        auto         probs  = torch::zeros_like(scores);
        GreedyParams params({scores,
                             cudaIntTensor(std::vector<int32_t>(batch_size, -1), {batch_size}),
                             cudaIntTensor(std::vector<int32_t>(batch_size, 1), {batch_size}),
                             tokens,
                             1,
                             pinnedIntTensor(std::vector<int32_t>(batch_size, 0)),
                             pinnedFloatTensor(std::vector<float>(batch_size, 0.8f)),
                             pinnedFloatTensor(std::vector<float>(batch_size, 0.5f))});
        params.output_all_probs          = probs;
        params.skip_logits_probs_restore = true;
        params.force_topk                = 4096;
        for (int64_t row = 0; row < batch_size; ++row) {
            auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
            generator.set_current_seed(20261027 + row);
            params.generator.push_back(generator);
        }

        auto output = execSampleGreedy(params);
        check_cuda_error();
        assertAllValid(output.success);
        EXPECT_TRUE(tokens.select(1, 0).eq(0).all().item<bool>());
        EXPECT_TRUE(tokens.select(1, 1).eq(vocab_size - 1).all().item<bool>());
        EXPECT_TRUE(probs.select(1, vocab_size - 1).eq(1.0f).all().item<bool>());
        EXPECT_TRUE(probs.slice(1, 0, vocab_size - 1).eq(0.0f).all().item<bool>());
    }
}

TEST_F(CudaSamplerTest, testExactJointSmallTopKUsesFullVocabularyTopPAndRetainsTies) {
    constexpr int64_t  batch = 4, vocab = 512;
    std::vector<float> logits(batch * vocab, -100.0f);
    logits[0]         = std::log(0.5f);
    logits[1]         = std::log(0.3f);
    logits[2]         = std::log(0.19f);
    logits[3]         = std::log(0.01f);
    logits[vocab]     = 3.0f;
    logits[vocab + 1] = 2.0f;
    logits[vocab + 2] = 1.0f;
    for (int i = 0; i < 100; ++i)
        logits[2 * vocab + i] = 5.0f;
    logits[3 * vocab]     = std::log(0.6f);
    logits[3 * vocab + 1] = std::log(0.4f);
    logits[3 * vocab + 2] = std::log(1.0e-8f);
    auto         scores   = cudaTensor(logits, {batch, vocab});
    auto         tokens   = cudaIntTensor(std::vector<int32_t>(batch * 2, 0), {batch, 2});
    auto         probs    = torch::zeros_like(scores);
    GreedyParams params({scores,
                         cudaIntTensor(std::vector<int32_t>(batch, -1), {batch}),
                         cudaIntTensor(std::vector<int32_t>(batch, 1), {batch}),
                         tokens,
                         1,
                         pinnedIntTensor({2, 40, 40, 3}),
                         pinnedFloatTensor({0.55f, 0.2f, 0.95f, 1.0f}),
                         pinnedFloatTensor({1.0f, 1.0f, 1.0f, 1.0f})});
    params.output_all_probs          = probs;
    params.skip_logits_probs_restore = true;
    auto generator                   = torch::make_generator<at::CUDAGeneratorImpl>();
    generator.set_current_seed(1729);
    params.generator.assign(batch, generator);
    execSampleGreedy(params);
    check_cuda_error();
    auto q = toHostFloat(probs);
    EXPECT_NEAR(q[0], 0.625f, 1e-5f);
    EXPECT_NEAR(q[1], 0.375f, 1e-5f);
    for (int i = 2; i < vocab; ++i)
        EXPECT_EQ(q[i], 0.0f);
    EXPECT_NEAR(q[vocab], 1.0f, 1e-5f);
    for (int i = 1; i < vocab; ++i)
        EXPECT_EQ(q[vocab + i], 0.0f);
    for (int i = 0; i < 100; ++i)
        EXPECT_NEAR(q[2 * vocab + i], 0.01f, 1e-5f);
    for (int i = 100; i < vocab; ++i)
        EXPECT_EQ(q[2 * vocab + i], 0.0f);
    EXPECT_GT(q[3 * vocab + 2], 0.0f);
    for (int i = 3; i < vocab; ++i)
        EXPECT_EQ(q[3 * vocab + i], 0.0f);
    auto sampled = toHostInt(tokens);
    EXPECT_LT(sampled[1], 2);
    EXPECT_EQ(sampled[3], 0);
    EXPECT_LT(sampled[5], 100);
    EXPECT_LT(sampled[7], 3);
}

TEST_F(CudaSamplerTest, AttentionMetadataMatchesSerialForRaggedAndMultiTileBatches) {
    for (int batch : {0, 1, 7, 64, 257, 513}) {
        for (bool with_prefix : {false, true}) {
            std::vector<int32_t> lengths(batch), prefixes(batch), q{0}, kv{0}, offsets;
            int                  max_len = 0;
            for (int b = 0; b < batch; ++b) {
                lengths[b]  = (b * 7 + 3) % 19;
                prefixes[b] = with_prefix ? 32768 + b * 31 : 0;
                max_len     = std::max(max_len, lengths[b]);
                q.push_back(q.back() + lengths[b]);
                kv.push_back(kv.back() + lengths[b] + prefixes[b]);
            }
            for (int b = 0; b < batch; ++b)
                for (int j = 0; j < lengths[b]; ++j)
                    offsets.push_back(b * max_len - q[b]);
            auto len     = cudaIntTensor(lengths, {batch});
            auto pre     = with_prefix ? cudaIntTensor(prefixes, {batch}) : torch::Tensor();
            auto q_out   = torch::empty({batch + 1}, len.options());
            auto kv_out  = torch::empty_like(q_out);
            auto pad_out = torch::empty({q.back()}, len.options());
            invokeBuildAttentionInputMetadata(len, pre, q_out, kv_out, pad_out, at::cuda::getCurrentCUDAStream());
            EXPECT_EQ(toHostInt(q_out), q);
            EXPECT_EQ(toHostInt(kv_out), kv);
            EXPECT_EQ(toHostInt(pad_out), offsets);
        }
    }
}

TEST_F(CudaSamplerTest, GreedyTop1TokenStridePreservesHistoryAndTieRules) {
    for (int batch : {1, 7, 64, 129, 384}) {
        for (int vocab : {17, 64, 16385, 129280}) {
            auto logits = torch::randn({batch, vocab}, torch::TensorOptions().device(torch::kCUDA));
            logits.select(0, 0).fill_(-INFINITY);
            if (batch > 1) {
                logits.select(0, 1).fill_(0);
                logits[1][3] = NAN;
                logits[1][7] = NAN;
            }
            auto      expected = torch::argmax(logits, -1).to(torch::kInt32);
            const int splits   = greedyTop1RenormSplits(batch, vocab);
            auto      partials = torch::empty({2, batch, splits}, logits.options());
            for (bool onehot : {false, true}) {
                auto output = torch::full({batch, 11}, -7, expected.options());
                auto probs  = torch::empty_like(logits);
                ASSERT_EQ(invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                 probs.data_ptr<float>(),
                                                 output.data_ptr<int32_t>() + 10,
                                                 batch,
                                                 vocab,
                                                 partials.data_ptr<float>(),
                                                 at::cuda::getCurrentCUDAStream(),
                                                 onehot,
                                                 11),
                          cudaSuccess);
                EXPECT_TRUE(torch::equal(output.select(1, 10), expected));
                EXPECT_TRUE(output.narrow(1, 0, 10).eq(-7).all().item<bool>());
                if (onehot) {
                    auto reference = torch::zeros_like(probs);
                    reference.scatter_(1, expected.to(torch::kLong).unsqueeze(1), 1);
                    EXPECT_TRUE(torch::equal(probs, reference));
                }
            }
        }
    }
}
