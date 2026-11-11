#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "pybind11/embed.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <limits>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
namespace rtp_llm {
template<typename T>
void invokeBatchApplyTemperaturePenalty(T* logits,
                                        const T* bias,
                                        const float* temperatures,
                                        int batch_size,
                                        int vocab_size,
                                        int vocab_size_padd,
                                        cudaStream_t stream);
}

using namespace std;
using namespace rtp_llm;

namespace {

constexpr int64_t kL2FlushBytes = 512LL * 1024 * 1024;

void initializeSamplerPython() {
    static pybind11::scoped_interpreter interpreter{};
    pybind11::exec(R"(
import os
import sys
from pathlib import Path

runfiles = Path(os.environ["TEST_SRCDIR"])
sys.path[:0] = [str(runfiles), str(runfiles / os.environ["TEST_WORKSPACE"])] + [
    str(path) for path in sorted(runfiles.glob("pip*/site-packages"))
]
import torch
)");
}

void flushL2Cache(const torch::Tensor& flush_buffer) {
    flush_buffer.add_(1.0f);
}

double benchmarkCudaEventUs(const std::function<void()>& fn,
                            const torch::Tensor&         l2_flush_buffer,
                            int                          warmup,
                            int                          iterations) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    for (int i = 0; i < warmup; ++i) {
        flushL2Cache(l2_flush_buffer);
        fn();
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double elapsed_us = 0.0;
    for (int i = 0; i < iterations; ++i) {
        flushL2Cache(l2_flush_buffer);
        cudaEventRecord(start, stream);
        fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        elapsed_us += static_cast<double>(elapsed_ms) * 1000.0;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_us / iterations;
}

torch::Tensor normalizedRandomProbs(int64_t batch_size, int64_t vocab_size) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto probs   = torch::rand({batch_size, vocab_size}, options);
    return probs.div(probs.sum(-1, /*keepdim=*/true));
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

struct SamplingAccuracyStats {
    double l1                  = 0.0;
    double max_abs             = 0.0;
    double kl                  = 0.0;
    int64_t valid_count        = 0;
    int64_t invalid_count      = 0;
    int64_t support_violations = 0;
};

SamplingAccuracyStats compareEmpiricalToReference(const std::vector<int32_t>& token_ids,
                                                  const std::vector<bool>&    valid,
                                                  const std::vector<float>&   reference) {
    SamplingAccuracyStats stats;
    std::vector<int64_t>  counts(reference.size(), 0);

    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (!valid[i]) {
            ++stats.invalid_count;
            continue;
        }
        ++stats.valid_count;
        const auto token = token_ids[i];
        if (token < 0 || static_cast<size_t>(token) >= reference.size()) {
            ++stats.support_violations;
            continue;
        }
        if (reference[token] <= 1e-6f) {
            ++stats.support_violations;
        }
        ++counts[token];
    }

    const double denom = std::max<int64_t>(stats.valid_count, 1);
    for (size_t i = 0; i < reference.size(); ++i) {
        const double empirical = static_cast<double>(counts[i]) / denom;
        const double expected  = reference[i];
        const double abs_diff  = std::abs(empirical - expected);
        stats.l1 += abs_diff;
        stats.max_abs = std::max(stats.max_abs, abs_diff);
        if (empirical > 0.0 && expected > 0.0) {
            stats.kl += empirical * std::log(empirical / expected);
        }
    }
    return stats;
}

void assertSamplingStatsClose(const std::string& name, const SamplingAccuracyStats& stats) {
    ASSERT_EQ(stats.invalid_count, 0) << name;
    ASSERT_EQ(stats.support_violations, 0) << name;
    ASSERT_LT(stats.l1, 0.08) << name;
    ASSERT_LT(stats.max_abs, 0.04) << name;
    ASSERT_LT(stats.kl, 0.02) << name;
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

TEST_F(CudaSamplerTest, testSamplingPreflightConcurrency) {
    if (std::getenv("RTP_LLM_TEST_PREFLIGHT_CONCURRENCY") == nullptr) {
        GTEST_SKIP() << "Run in a separate process to test concurrent startup";
    }
    initializeSamplerPython();
    pybind11::exec(R"(
import threading
from rtp_llm.models_py.sampling import runtime
calls = []
entered = threading.Event()
proceed = threading.Event()
def fake_warmup(vocab, rejection):
    calls.append((vocab, rejection))
    if vocab == 32769 and not rejection:
        entered.set()
        if not proceed.wait(30):
            raise RuntimeError("concurrent preflight did not resume")
    return ""
runtime.warmup = fake_warmup
)");
    EXPECT_TRUE(supportsDeferredRejectionSampling());
    const int          device = at::cuda::current_device();
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    std::promise<void> second_entered;
    auto               second_started = second_entered.get_future();
    std::thread        first([&]() {
        try {
            at::cuda::set_device(device);
            initializeSamplingRuntime(32769);
        } catch (...) {
            first_error = std::current_exception();
        }
    });
    auto               globals = pybind11::globals();
    EXPECT_TRUE(globals["entered"].attr("wait")(10).cast<bool>());
    std::thread second([&]() {
        try {
            at::cuda::set_device(device);
            pybind11::gil_scoped_acquire gil;
            second_entered.set_value();
            initializeSamplingRuntime(32769);
        } catch (...) {
            second_error = std::current_exception();
            try {
                second_entered.set_exception(second_error);
            } catch (const std::future_error&) {}
        }
    });
    {
        pybind11::gil_scoped_release release;
        second_started.wait();
    }
    globals["proceed"].attr("set")();
    {
        pybind11::gil_scoped_release release;
        first.join();
        second.join();
    }
    EXPECT_FALSE(first_error);
    EXPECT_FALSE(second_error);
    pybind11::exec(R"(
assert calls.count((8193, True)) == 1, calls
assert calls.count((32769, False)) == 1, calls
assert calls.count((32769, True)) == 1, calls
assert len(calls) == 3, calls
)");
}

TEST_F(CudaSamplerTest, testSamplingPreflightFailure) {
    if (std::getenv("RTP_LLM_TEST_PREFLIGHT_FAILURE") == nullptr) {
        GTEST_SKIP() << "Run in a separate process to test startup route selection";
    }
    initializeSamplerPython();
    pybind11::exec(R"(
from rtp_llm.models_py.sampling import runtime
runtime.warmup = lambda vocab, rejection: "injected dependency/compiler failure"
)");
    initializeSamplingRuntime(8193);
    EXPECT_FALSE(supportsDeferredRejectionSampling());
    auto         logits = cudaTensor({std::log(6.0f), std::log(3.0f), 0.0f}, {1, 3});
    auto         tokens = torch::full({1, 2}, -1, logits.options().dtype(torch::kInt32));
    GreedyParams params({logits,
                         cudaIntTensor({1}, {1}),
                         cudaIntTensor({1}, {1}),
                         tokens,
                         1,
                         pinnedIntTensor({0}),
                         pinnedFloatTensor({1.0f}),
                         pinnedFloatTensor({1.0f}),
                         nullopt,
                         nullopt,
                         nullopt,
                         nullopt,
                         nullopt,
                         nullopt,
                         nullopt,
                         nullopt,
                         {}});
    execSampleGreedy(params);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_TRUE(torch::allclose(logits, cudaTensor({0.6f, 0.3f, 0.1f}, {1, 3})));
    assertTokenIn(toHostInt(tokens), 1, {0, 1, 2});
}

TEST_F(CudaSamplerTest, testVllmGumbelBridge) {
    initializeSamplerPython();
    auto sample = pybind11::module_::import("rtp_llm.models_py.sampling.adapter").attr("sample");
    constexpr int64_t batch_size = 128;
    auto original = cudaTensor({std::log(6.0f), std::log(3.0f), 0.0f}, {1, 3}).repeat({batch_size, 1});
    auto logits = original.clone();
    auto tokens = torch::full({batch_size, 2}, -1, logits.options().dtype(torch::kInt32));
    tokens.select(1, 0).fill_(2);
    auto seeds = torch::arange(batch_size, logits.options().dtype(torch::kInt64)) + 12345;
    std::vector<at::Generator> generators;
    for (int64_t row = 0; row < batch_size; ++row) {
        generators.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
        generators.back().set_current_seed(12345 + row);
        generators.back().set_offset(0);
    }
    GreedyParams params({logits,
                         cudaIntTensor(std::vector<int32_t>(batch_size, 1), {batch_size}),
                         cudaIntTensor(std::vector<int32_t>(batch_size, 1), {batch_size}),
                         tokens,
                         1,
                         pinnedIntTensor(std::vector<int32_t>(batch_size, 0)),
                         pinnedFloatTensor(std::vector<float>(batch_size, 1.0f)),
                         pinnedFloatTensor(std::vector<float>(batch_size, 1.0f)),
                         nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt,
                         generators});
    for (int64_t offset : {0, 32}) {
        SCOPED_TRACE("offset=" + std::to_string(offset));
        auto offsets = torch::full_like(seeds, offset);
        auto expected = sample(original, seeds, offsets).cast<torch::Tensor>().to(torch::kInt32);
        execSampleGreedy(params);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(toHostInt(tokens.select(1, 1)), toHostInt(expected));
        EXPECT_EQ(toHostInt(tokens.select(1, 0)), std::vector<int32_t>(batch_size, 2));
        EXPECT_TRUE(torch::equal(logits, original));
        for (const auto& generator : generators) {
            EXPECT_EQ(generator.get_offset(), static_cast<uint64_t>(offset + 32));
        }
    }
    auto all_probs = torch::empty_like(logits);
    params.output_all_probs = all_probs;
    auto result = execSampleGreedy(params);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    assertAllValid(result.success);
    EXPECT_TRUE(torch::allclose(all_probs, torch::softmax(original, -1), 1e-5, 1e-6));
    params.defer_token_sampling = true;
    logits.copy_(original);
    all_probs.fill_(-7);
    tokens.select(1, 1).fill_(-1);
    const auto offset_before = generators.front().get_offset();
    auto       prepared      = execSampleGreedy(params);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_TRUE(prepared.deferred_token_sampling);
    EXPECT_TRUE(torch::equal(logits, original));
    EXPECT_TRUE(torch::all(all_probs == -7).item<bool>());
    EXPECT_TRUE(torch::all(tokens.select(1, 1) == -1).item<bool>());
    EXPECT_EQ(generators.front().get_offset(), offset_before);
    for (auto [k, p] : std::vector<std::pair<int, float>>{{0, 1.0f}, {0, 0.8f}, {2, 1.0f}, {2, 0.8f}}) {
        params.top_k.fill_(k);
        params.top_p.fill_(p);
        params.defer_token_sampling = false;
        logits.copy_(original);
        execSampleGreedy(params);
        auto reference_probs = all_probs.clone();
        all_probs.fill_(-7);
        params.defer_token_sampling = true;
        logits.copy_(original);
        tokens.select(1, 1).fill_(-1);
        auto filtered = execSampleGreedy(params);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(filtered.deferred_token_sampling);
        EXPECT_TRUE(torch::allclose(torch::softmax(logits, -1), reference_probs, 1e-5, 1e-6));
        EXPECT_TRUE(torch::all(all_probs == -7).item<bool>());
        EXPECT_TRUE(torch::all(tokens.select(1, 1) == -1).item<bool>());
    }
}

TEST_F(CudaSamplerTest, testDeferredPreparationPreservesHistoryAndPenalties) {
    initializeSamplerPython();
    constexpr int64_t batch = 4;
    constexpr int64_t vocab = 10;
    auto              original =
        cudaTensor({0.1f, 0.8f, 0.9f, -0.2f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f}, {1, vocab}).repeat({batch, 1});
    auto history =
        pinnedIntTensor({2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0}).reshape({batch, 6});
    for (int mode = 0; mode < 6; ++mode) {
        SCOPED_TRACE("penalty mode=" + std::to_string(mode));
        GreedyParams params;
        params.logits           = original.clone();
        params.input_lengths    = cudaIntTensor({5, 5, 5, 5}, {batch});
        params.sequence_lengths = cudaIntTensor({5, 5, 5, 5}, {batch});
        params.token_ids        = history.clone();
        params.step             = 5;
        params.top_k            = pinnedIntTensor({0, 0, 0, 0});
        params.top_p            = pinnedFloatTensor({1, 1, 1, 1});
        params.temperature      = pinnedFloatTensor({0.8f, 1, 1.2f, 1});
        params.output_all_probs = torch::empty_like(original);
        if (mode > 0) {
            params.repetition_penalty = pinnedFloatTensor({1, 1, 1, 1});
            params.presence_penalty   = pinnedFloatTensor({0, 0, 0, 0});
            params.frequency_penalty  = pinnedFloatTensor({0, 0, 0, 0});
        }
        if (mode == 2) {
            params.repetition_penalty->fill_(1.7f);
        } else if (mode == 3) {
            params.presence_penalty->fill_(0.6f);
        } else if (mode == 4) {
            params.frequency_penalty->fill_(0.2f);
        } else if (mode == 5) {
            params.no_repeat_ngram_size = pinnedIntTensor({1, 0, 1, 0});
        }
        params.do_sample = pinnedIntTensor({1, 0, 1, 1}).to(torch::kBool);
        execSampleGreedy(params);
        auto reference = params.output_all_probs->clone();
        params.logits.copy_(original);
        params.token_ids.copy_(history);
        params.output_all_probs->fill_(-7);
        params.defer_token_sampling = true;
        EXPECT_TRUE(execSampleGreedy(params).deferred_token_sampling);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(torch::allclose(torch::softmax(params.logits, -1), reference, 1e-5, 1e-6));
        EXPECT_TRUE(torch::equal(params.token_ids, history));
        EXPECT_TRUE(torch::all(params.output_all_probs.value() == -7).item<bool>());
    }

    GreedyParams unfiltered;
    unfiltered.logits               = original.clone();
    unfiltered.input_lengths        = cudaIntTensor({5, 5, 5, 5}, {batch});
    unfiltered.sequence_lengths     = cudaIntTensor({5, 5, 5, 5}, {batch});
    unfiltered.token_ids            = history.clone();
    unfiltered.step                 = 5;
    unfiltered.top_k                = pinnedIntTensor({-1, 1, vocab, 0});
    unfiltered.top_p                = pinnedFloatTensor({0, std::numeric_limits<float>::quiet_NaN(), 1, -1e-8f});
    unfiltered.temperature          = pinnedFloatTensor({1, 1, 1, 1});
    unfiltered.defer_token_sampling = true;
    EXPECT_TRUE(execSampleGreedy(unfiltered).deferred_token_sampling);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_TRUE(torch::equal(unfiltered.logits, original));
    EXPECT_TRUE(torch::equal(unfiltered.token_ids, history));
}

TEST_F(CudaSamplerTest, testDeferredLogitTopKTopPMatchesDense) {
    initializeSamplerPython();
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    if (properties->major != 10 || (properties->minor != 0 && properties->minor != 3))
        GTEST_SKIP() << "DeepSelect requires SM100 or SM103";
    constexpr int batch = 8, rows = 6, vocab = 129280, count = batch * rows;
    auto          options  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto          top_k    = pinnedIntTensor(std::vector<int32_t>(count, 1024));
    const auto    stream   = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());
    auto          original = torch::randn({count, vocab}, options);
    // Exact boundary ties, including more than K identical winners.
    original.select(0, 1).zero_();
    original.select(0, 2).fill_(-8);
    original.select(0, 2).slice(0, 0, 1500).fill_(1);
    original.select(0, 3).fill_(-8);
    original.select(0, 3).slice(0, 0, 32).fill_(2);
    original.select(0, 3).slice(0, 32, 1600).fill_(1);
    // Nearly equal FP32 logits test probability-rounding boundaries explicitly.
    original.select(0, 4).zero_();
    original.select(0, 4).slice(0, 0, 1024).fill_(std::nextafter(1.0f, 2.0f));
    original.select(0, 4).slice(0, 1024, 2048).fill_(1.0f);
    for (float p : {1.0f, 0.9f, 0.5f}) {
        SCOPED_TRACE("top_p=" + std::to_string(p));
        auto probabilities = torch::softmax(original, -1);
        auto temporary     = torch::empty_like(probabilities);
        auto expected      = torch::empty_like(probabilities);
        auto top_p         = pinnedFloatTensor(std::vector<float>(count, p));
        top_k_renorm_probs(probabilities, temporary, top_k, 0, stream);
        if (p == 1.0f)
            expected.copy_(temporary);
        else
            top_p_renorm_probs(temporary, expected, top_p, 1.0, stream);
        auto filtered = original.clone();
        ASSERT_TRUE(torch_ext::deferred_topk_topp_filter(filtered, 1024, p));
        auto actual = torch::softmax(filtered, -1);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(torch::equal(actual > 0, expected > 0))
            << "actual counts=" << (actual > 0).sum(-1).cpu() << " expected counts=" << (expected > 0).sum(-1).cpu();
        EXPECT_TRUE(torch::allclose(actual, expected, 3e-5, 2e-7));

        RejectionSamplingParams params;
        params.draft_token_ids_d           = torch::zeros({batch, rows - 1}, options.dtype(torch::kInt32));
        params.do_sample_d                 = torch::ones({batch}, options.dtype(torch::kBool));
        params.output_token_ids_d          = torch::empty({batch, rows}, options.dtype(torch::kInt32));
        params.output_accepted_token_num_d = torch::empty({batch}, options.dtype(torch::kInt32));
        params.deferred_token_sampling     = true;
        params.draft_point_mass            = true;
        for (int i = 0; i < batch; ++i) {
            params.generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
            params.generator.back().set_current_seed(12345 + i);
            params.generator.back().set_offset(0);
        }
        params.target_logits_d = expected.log().reshape({batch, rows, vocab});
        execRejectionSampling(params);
        auto expected_ids     = params.output_token_ids_d.clone();
        auto expected_lengths = params.output_accepted_token_num_d.clone();
        for (auto& generator : params.generator) {
            EXPECT_EQ(generator.get_offset(), 32u);
            generator.set_offset(0);
        }
        params.target_logits_d = filtered.reshape({batch, rows, vocab});
        execRejectionSampling(params);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(torch::equal(params.output_token_ids_d, expected_ids));
        EXPECT_TRUE(torch::equal(params.output_accepted_token_num_d, expected_lengths));
        for (const auto& generator : params.generator)
            EXPECT_EQ(generator.get_offset(), 32u);
    }
    auto unsupported = original.clone();
    EXPECT_FALSE(torch_ext::deferred_topk_topp_filter(unsupported, 0, 0.9f));
    EXPECT_TRUE(torch::equal(unsupported, original));
}

TEST_F(CudaSamplerTest, testCompactDeferredGumbelMatchesDense) {
    initializeSamplerPython();
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    if (properties->major != 10 || (properties->minor != 0 && properties->minor != 3))
        GTEST_SKIP() << "DeepSelect requires SM100 or SM103";
    auto mock        = pybind11::module_::import("unittest.mock");
    auto environment = pybind11::module_::import("os").attr("environ");
    auto patch =
        mock.attr("patch").attr("dict")(environment,
                                        pybind11::dict(pybind11::arg("RTP_LLM_GUMBEL_REJECTION_GRAPH")  = "0",
                                                       pybind11::arg("RTP_LLM_GUMBEL_REJECTION_NATIVE") = "1"));
    patch.attr("__enter__")();
    struct RestorePatch {
        pybind11::object patch;
        ~RestorePatch() {
            patch.attr("__exit__")(pybind11::none(), pybind11::none(), pybind11::none());
        }
    } restore_outer{patch};
    auto eager = pybind11::module_::import("rtp_llm.models_py.sampling.adapter").attr("_rejection_sample_logits_eager");
    auto materialize = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_compact").attr("materialize");
    auto native      = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_native");
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    native.attr("_plans").attr("clear")();
    constexpr int batch = 8, rows = 6, vocab = 129280;
    auto          options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int variant = 0; variant < 4; ++variant) {
        SCOPED_TRACE("variant=" + std::to_string(variant));
        auto original = torch::randn({batch * rows, vocab}, options) * 0.2;
        original.select(1, vocab - 1).fill_(7.7);
        if (variant == 1) {
            original.select(0, 0).zero_();  // >K boundary ties require dense fallback.
            original.select(0, 1).fill_(-INFINITY);
            original.select(0, 1).index_put_({vocab - 1}, 1.0);  // Sparse selection.
        }
        auto dense    = original.clone();
        auto selected = torch_ext::deferred_compact_topk_topp(dense, 1024, 0.9f);
        if (variant == 0) {
            GreedyParams producer;
            producer.logits               = original.clone();
            producer.input_lengths        = cudaIntTensor(std::vector<int32_t>(batch * rows, 1), {batch * rows});
            producer.sequence_lengths     = producer.input_lengths;
            producer.token_ids            = torch::zeros({batch * rows, 2}, options.dtype(torch::kInt32));
            producer.step                 = 1;
            producer.top_k                = pinnedIntTensor(std::vector<int32_t>(batch * rows, 1024));
            producer.top_p                = pinnedFloatTensor(std::vector<float>(batch * rows, 0.9f));
            producer.temperature          = pinnedFloatTensor(std::vector<float>(batch * rows, 1.0f));
            producer.defer_token_sampling = true;
            producer.allow_compact_deferred_output = true;
            auto prepared                          = execSampleGreedy(producer);
            ASSERT_TRUE(prepared.deferred_logits.defined());
            EXPECT_TRUE(prepared.deferred_token_sampling);
            EXPECT_TRUE(torch::equal(prepared.deferred_dense_fallback, original));
            selected = {prepared.deferred_logits, prepared.deferred_token_ids, prepared.deferred_fallback_rows};
            dense    = prepared.deferred_dense_fallback;
        }
        ASSERT_TRUE(std::get<0>(selected).defined());
        if (variant == 0) {
            EXPECT_EQ(std::get<2>(selected).sum().item<int64_t>(), 0);
        }
        if (variant == 1) {
            EXPECT_GT(std::get<2>(selected).sum().item<int64_t>(), 0);
        }
        auto logits          = std::get<0>(selected).reshape({batch, rows, 1024});
        auto ids             = std::get<1>(selected).reshape({batch, rows, 1024});
        auto flags           = std::get<2>(selected);
        dense                = dense.reshape({batch, rows, vocab});
        auto expanded        = materialize(logits, ids, dense, flags).cast<torch::Tensor>();
        auto expected_logits = original.clone();
        ASSERT_TRUE(torch_ext::deferred_topk_topp_filter(expected_logits, 1024, 0.9f));
        EXPECT_TRUE(torch::equal(expanded, expected_logits.reshape_as(expanded)));
        RejectionSamplingParams params;
        params.target_logits_d         = logits;
        params.target_candidate_ids_d  = ids;
        params.target_dense_fallback_d = dense;
        params.target_fallback_rows_d  = flags;
        params.draft_token_ids_d       = torch::full({batch, rows - 1}, vocab - 1, options.dtype(torch::kInt32));
        params.draft_token_ids_d.select(0, 3).fill_(0);  // Candidate absent from random rows.
        params.do_sample_d                 = torch::arange(batch, options.dtype(torch::kInt32)).remainder(2) == 0;
        params.output_token_ids_d          = torch::empty({batch, rows}, options.dtype(torch::kInt32));
        params.output_accepted_token_num_d = torch::empty({batch}, options.dtype(torch::kInt32));
        params.deferred_token_sampling     = true;
        params.draft_point_mass            = true;
        if (variant >= 2)
            params.acceptance_cap = torch::arange(batch, options.dtype(torch::kInt32)).remainder(rows);
        if (variant == 3)
            params.force_accept.assign(batch, true);
        for (int i = 0; i < batch; ++i) {
            auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
            generator.set_current_seed(12345 + i);
            generator.set_offset(32 * variant);
            params.generator.push_back(generator);
        }
        auto seeds     = torch::arange(batch, options.dtype(torch::kInt64)) + 12345;
        auto positions = torch::full({batch}, variant * 32, options.dtype(torch::kInt64));
        auto expected =
            eager(expanded,
                  pybind11::none(),
                  params.draft_token_ids_d,
                  seeds,
                  positions,
                  params.do_sample_d,
                  variant == 3 ? pybind11::cast(params.force_accept) : pybind11::none(),
                  params.acceptance_cap.defined() ? pybind11::cast(params.acceptance_cap) : pybind11::none(),
                  pybind11::arg("clean_output") = true)
                .cast<std::pair<torch::Tensor, torch::Tensor>>();
        execRejectionSampling(params);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(torch::equal(params.output_token_ids_d, expected.first));
        EXPECT_TRUE(torch::equal(params.output_accepted_token_num_d, expected.second));
        EXPECT_GT(pybind11::len(native.attr("_plans")), 0);
        for (auto& generator : params.generator) {
            EXPECT_EQ(generator.get_offset(), static_cast<uint64_t>(32 * (variant + 1)));
            generator.set_offset(32 * variant);
        }
        // Same keys, shuffled storage: Gumbel noise and global-ID tie order must stay fixed.
        params.target_logits_d        = logits.flip({2});
        params.target_candidate_ids_d = ids.flip({2});
        execRejectionSampling(params);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(torch::equal(params.output_token_ids_d, expected.first));
        EXPECT_TRUE(torch::equal(params.output_accepted_token_num_d, expected.second));
        if (variant == 0) {
            // Adjacent requests may share one generator: each consumes its own
            // consecutive 32-counter interval, on native and materialized paths.
            auto shared = torch::make_generator<at::CUDAGeneratorImpl>();
            shared.set_current_seed(777);
            shared.set_offset(0);
            params.generator.assign(batch, shared);
            auto shared_seeds     = torch::full({batch}, 777, options.dtype(torch::kInt64));
            auto shared_positions = torch::arange(batch, options.dtype(torch::kInt64)) * 32;
            auto shared_expected  = eager(expanded,
                                         pybind11::none(),
                                         params.draft_token_ids_d,
                                         shared_seeds,
                                         shared_positions,
                                         params.do_sample_d,
                                         pybind11::arg("clean_output") = true)
                                       .cast<std::pair<torch::Tensor, torch::Tensor>>();
            execRejectionSampling(params);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            EXPECT_EQ(shared.get_offset(), static_cast<uint64_t>(32 * batch));
            EXPECT_TRUE(torch::equal(params.output_token_ids_d, shared_expected.first));
            EXPECT_TRUE(torch::equal(params.output_accepted_token_num_d, shared_expected.second));
            shared.set_offset(0);
            auto fallback_patch = mock.attr("patch").attr("dict")(
                environment, pybind11::dict(pybind11::arg("RTP_LLM_GUMBEL_REJECTION_NATIVE") = "0"));
            fallback_patch.attr("__enter__")();
            RestorePatch restore{fallback_patch};
            execRejectionSampling(params);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            EXPECT_EQ(shared.get_offset(), static_cast<uint64_t>(32 * batch));
            EXPECT_TRUE(torch::equal(params.output_token_ids_d, shared_expected.first));
            EXPECT_TRUE(torch::equal(params.output_accepted_token_num_d, shared_expected.second));
        }
    }
}

TEST_F(CudaSamplerTest, testCompactTemperatureFusionMatchesLegacy) {
    initializeSamplerPython();
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    if (properties->major != 10 || (properties->minor != 0 && properties->minor != 3))
        GTEST_SKIP() << "DeepSelect requires SM100 or SM103";
    constexpr int batch = 8, rows = 6, count = batch * rows, vocab = 129280;
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto materialize = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_compact").attr("materialize");
    auto expand = [&](const GreedyOutput& output) {
        return materialize(output.deferred_logits.reshape({batch, rows, 1024}),
                           output.deferred_token_ids.reshape({batch, rows, 1024}),
                           output.deferred_dense_fallback.reshape({batch, rows, vocab}),
                           output.deferred_fallback_rows)
            .cast<torch::Tensor>();
    };
    for (int variant = 0; variant < 10; ++variant) {
        SCOPED_TRACE("temperature variant=" + std::to_string(variant));
        const float temperature = variant == 0 ? 0.7f : variant == 2 ? 1.0f : variant == 9 ? 1e10f : 1.3f;
        std::vector<float> temperatures(count, temperature);
        if (variant == 3)
            for (int row = 0; row < count; ++row)
                temperatures[row] = row % 3 == 0 ? 0.7f : row % 3 == 1 ? 1.0f : 1.3f;
        std::vector<float> values(vocab);
        for (int i = 0; i < vocab; ++i)
            values[i] = -static_cast<float>(i) / vocab;
        values[0] = 6.0f;
        auto original = torch::tensor(values).to(options).unsqueeze(0).repeat({count, 1});
        if (variant == 4) {
            original.select(0, 0).zero_();
        } else if (variant == 5) {
            const float inverse = 1.0f / (temperature + 1e-6f);
            float lower = 1.5f, upper = lower;
            for (int attempt = 0; attempt < 128; ++attempt) {
                upper = std::nextafter(lower, INFINITY);
                if (lower * inverse == upper * inverse)
                    break;
                lower = upper;
            }
            ASSERT_LT(lower, upper);
            ASSERT_EQ(lower * inverse, upper * inverse);
            auto row = original.select(0, 0);
            row.fill_(-4.0f);
            row.narrow(0, 0, 1023).fill_(4.0f);
            row.index_put_({1023}, upper);
            row.index_put_({1024}, lower);
        } else if (variant == 6) {
            original.select(0, 0).fill_(-INFINITY);
            original.select(0, 0).index_put_({0}, 1.0f);
            original.select(0, 1).index_put_({0}, INFINITY);
            original.select(0, 2).index_put_({0}, std::numeric_limits<float>::quiet_NaN());
            original.select(0, 3).index_put_({0}, -std::numeric_limits<float>::quiet_NaN());
        } else if (variant == 8) {
            for (int i = 0; i < vocab; ++i)
                values[i] = (i % 2 ? -1.0f : 1.0f) * std::numeric_limits<float>::denorm_min() * (i % 127 + 1);
            original = torch::tensor(values).to(options).unsqueeze(0).repeat({count, 1});
        } else if (variant == 9) {
            original.mul_(1e-30f);
        }
        auto sample = torch::ones({count}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
        if (variant == 7)
            sample.narrow(0, 0, rows).fill_(false);
        auto scaled = original.clone();
        if (variant != 2) {
            auto temperature_device = pinnedFloatTensor(temperatures).to(torch::kCUDA);
            invokeBatchApplyTemperaturePenalty<float>(scaled.data_ptr<float>(),
                                                       nullptr,
                                                       temperature_device.data_ptr<float>(),
                                                       count,
                                                       vocab,
                                                       vocab,
                                                       at::cuda::getCurrentCUDAStream().stream());
            if (variant == 7)
                scaled.narrow(0, 0, rows).copy_(original.narrow(0, 0, rows));
        }
        if (variant == 5) {
            ASSERT_EQ(scaled.index({0, 1023}).item<float>(), scaled.index({0, 1024}).item<float>());
        }
        if (variant == 8 || variant == 9) {
            ASSERT_TRUE(torch::equal(scaled, torch::zeros_like(scaled)));
        }
        for (bool shared : {false, true}) {
            SCOPED_TRACE("shared generator=" + std::to_string(shared));
            std::vector<at::Generator> generators;
            for (int i = 0; i < batch; ++i) {
                auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                generator.set_current_seed(23456 + i);
                generators.push_back(generator);
            }
            if (shared)
                generators.assign(batch, generators.front());
            auto reset = [&] {
                for (auto& generator : generators)
                    generator.set_offset(0);
            };
            auto produce = [&](bool fused) {
                GreedyParams params;
                params.logits = fused ? original.clone() : scaled.clone();
                params.input_lengths = cudaIntTensor(std::vector<int32_t>(count, 1), {count});
                params.sequence_lengths = params.input_lengths;
                params.token_ids = torch::zeros({count, 2}, options.dtype(torch::kInt32));
                params.step = 1;
                params.top_k = pinnedIntTensor(std::vector<int32_t>(count, 1024));
                params.top_p = pinnedFloatTensor(std::vector<float>(count, variant == 5 ? 1.0f : 0.9f));
                params.temperature = pinnedFloatTensor(fused ? temperatures : std::vector<float>(count, 1.0f));
                params.do_sample = sample;
                params.defer_token_sampling = true;
                params.allow_compact_deferred_output = true;
                params.deferred_rejection_rows = rows;
                params.deferred_rejection_generators = generators;
                return execSampleGreedy(params);
            };
            auto reject = [&](const GreedyOutput& output) {
                RejectionSamplingParams params;
                params.target_logits_d = output.deferred_logits.reshape({batch, rows, 1024});
                params.target_candidate_ids_d = output.deferred_token_ids.reshape({batch, rows, 1024});
                params.target_dense_fallback_d = output.deferred_dense_fallback.reshape({batch, rows, vocab});
                params.target_fallback_rows_d = output.deferred_fallback_rows;
                params.draft_token_ids_d = torch::zeros({batch, rows - 1}, options.dtype(torch::kInt32));
                params.do_sample_d = torch::ones({batch}, options.dtype(torch::kBool));
                if (variant == 7)
                    params.do_sample_d.select(0, 0).fill_(false);
                params.output_token_ids_d = torch::empty({batch, rows}, options.dtype(torch::kInt32));
                params.output_accepted_token_num_d = torch::empty({batch}, options.dtype(torch::kInt32));
                params.generator = generators;
                params.deferred_token_sampling = true;
                params.draft_point_mass = true;
                params.prepared_rejection_rng = output.prepared_rejection_rng;
                execRejectionSampling(params);
                for (const auto& generator : generators)
                    EXPECT_EQ(generator.get_offset(), uint64_t(32 * (shared ? batch : 1)));
                return std::make_pair(params.output_token_ids_d, params.output_accepted_token_num_d);
            };
            reset();
            auto expected = produce(false);
            ASSERT_TRUE(expected.deferred_logits.defined());
            auto expected_dense = expand(expected);
            reset();
            auto actual = produce(true);
            ASSERT_TRUE(actual.deferred_logits.defined());
            auto actual_dense = expand(actual);
            EXPECT_TRUE(torch::equal(torch::isnan(actual_dense), torch::isnan(expected_dense)));
            EXPECT_TRUE(torch::equal(torch::where(torch::isnan(actual_dense), 0, actual_dense),
                                     torch::where(torch::isnan(expected_dense), 0, expected_dense)));
            EXPECT_TRUE(torch::equal(actual.deferred_fallback_rows, expected.deferred_fallback_rows));
            if (variant == 0 || variant == 1) {
                EXPECT_TRUE(torch::equal(actual.deferred_dense_fallback, original));
            }
            if (variant == 3 || variant == 7) {
                EXPECT_TRUE(torch::equal(actual.deferred_dense_fallback, scaled));
            }
            if (variant == 4 || variant == 5 || variant == 8 || variant == 9) {
                EXPECT_EQ(actual.deferred_fallback_rows[0].item<int32_t>(), 1);
            }
            // Non-finite rows validate filtering/fallback equivalence, not a categorical distribution.
            if (variant == 6)
                continue;
            auto expected_output = reject(expected);
            auto actual_output = reject(actual);
            EXPECT_TRUE(torch::equal(actual_output.first, expected_output.first));
            EXPECT_TRUE(torch::equal(actual_output.second, expected_output.second));
        }
    }
}

TEST_F(CudaSamplerTest, testCompactTemperatureFusionDeclinesGraphCapture) {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    if (properties->major != 10 || (properties->minor != 0 && properties->minor != 3)) {
        GTEST_SKIP() << "DeepSelect requires SM100 or SM103";
    }
    auto                       stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto input = torch::arange(129280, options).reshape({1, 129280});
    auto original = input.clone();
    auto dummy = torch::zeros({1}, options);
    dummy.add_(1);
    dummy.zero_();
    ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    auto captured = torch_ext::deferred_compact_topk_topp(input, 1024, 0.9f, 0.7f);
    dummy.add_(1);
    graph.capture_end();
    EXPECT_FALSE(std::get<0>(captured).defined());
    EXPECT_FALSE(std::get<1>(captured).defined());
    EXPECT_FALSE(std::get<2>(captured).defined());
    graph.replay();
    ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
    EXPECT_TRUE(torch::equal(input, original));
    EXPECT_EQ(dummy.item<float>(), 1.0f);
}

TEST_F(CudaSamplerTest, testCompactPreparedRejectionRng) {
    initializeSamplerPython();
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    if (properties->major != 10 || (properties->minor != 0 && properties->minor != 3))
        GTEST_SKIP() << "DeepSelect requires SM100 or SM103";
    constexpr int batch = 4, rows = 6, vocab = 129280;
    auto          options  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto          original = torch::randn({batch * rows, vocab}, options) * 0.2;
    original.select(1, vocab - 1).fill_(7.7);
    auto default_generator = at::cuda::detail::getDefaultCUDAGenerator();
    struct RestoreGenerator {
        at::Generator generator;
        torch::Tensor state;
        ~RestoreGenerator() {
            generator.set_state(state);
        }
    } restore{default_generator, default_generator.get_state()};
    for (int mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE(mode);
        std::vector<at::Generator> generators;
        for (int i = 0; i < batch; ++i) {
            auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
            generator.set_current_seed(4567 + i);
            generators.push_back(generator);
        }
        if (mode == 1)
            generators.assign(batch, generators.front());
        if (mode == 2)
            generators.clear();  // The production helper supplies the default for missing entries.
        auto reset = [&] {
            default_generator.set_current_seed(9012);
            default_generator.set_offset(64);
            for (auto& generator : generators)
                generator.set_offset(64);
        };
        auto offsets = [&] {
            std::vector<uint64_t> result{default_generator.get_offset()};
            for (auto& generator : generators)
                result.push_back(generator.get_offset());
            return result;
        };
        auto producer = [&](bool prepare) {
            GreedyParams params;
            params.logits                        = original.clone();
            params.input_lengths                 = cudaIntTensor(std::vector<int32_t>(batch * rows, 1), {batch * rows});
            params.sequence_lengths              = params.input_lengths;
            params.token_ids                     = torch::zeros({batch * rows, 2}, options.dtype(torch::kInt32));
            params.step                          = 1;
            params.top_k                         = pinnedIntTensor(std::vector<int32_t>(batch * rows, 1024));
            params.top_p                         = pinnedFloatTensor(std::vector<float>(batch * rows, 0.9f));
            params.temperature                   = pinnedFloatTensor(std::vector<float>(batch * rows, 1.0f));
            params.defer_token_sampling          = true;
            params.allow_compact_deferred_output = true;
            params.deferred_rejection_generators = generators;
            params.deferred_rejection_rows       = prepare ? rows : 0;
            return execSampleGreedy(params);
        };
        auto rejection = [&](const GreedyOutput& prepared) {
            RejectionSamplingParams params;
            params.target_logits_d         = prepared.deferred_logits.reshape({batch, rows, 1024});
            params.target_candidate_ids_d  = prepared.deferred_token_ids.reshape({batch, rows, 1024});
            params.target_dense_fallback_d = prepared.deferred_dense_fallback.reshape({batch, rows, vocab});
            params.target_fallback_rows_d  = prepared.deferred_fallback_rows;
            params.draft_token_ids_d       = torch::full({batch, rows - 1}, vocab - 1, options.dtype(torch::kInt32));
            params.do_sample_d             = torch::ones({batch}, options.dtype(torch::kBool));
            params.output_token_ids_d      = torch::empty({batch, rows}, options.dtype(torch::kInt32));
            params.output_accepted_token_num_d = torch::empty({batch}, options.dtype(torch::kInt32));
            params.generator                   = generators;
            params.deferred_token_sampling     = true;
            params.draft_point_mass            = true;
            params.prepared_rejection_rng      = prepared.prepared_rejection_rng;
            return params;
        };
        reset();
        auto ordinary_output = producer(false);
        ASSERT_FALSE(ordinary_output.prepared_rejection_rng.defined());
        auto ordinary = rejection(ordinary_output);
        execRejectionSampling(ordinary);
        auto expected_offsets = offsets();
        reset();
        auto prepared_output = producer(true);
        ASSERT_TRUE(prepared_output.prepared_rejection_rng.defined());
        EXPECT_EQ(offsets(), expected_offsets);
        auto prepared = rejection(prepared_output);
        execRejectionSampling(prepared);
        EXPECT_EQ(offsets(), expected_offsets);  // No second reservation in the consumer.
        EXPECT_TRUE(torch::equal(prepared.output_token_ids_d, ordinary.output_token_ids_d));
        EXPECT_TRUE(torch::equal(prepared.output_accepted_token_num_d, ordinary.output_accepted_token_num_d));
        prepared.prepared_rejection_rng = prepared.prepared_rejection_rng.narrow(1, 0, batch - 1);
        EXPECT_THROW(execRejectionSampling(prepared), c10::Error);
        EXPECT_EQ(offsets(), expected_offsets);
    }
}

TEST_F(CudaSamplerTest, testCompactProducerScratchLifetime) {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    if (properties->major != 10 || (properties->minor != 0 && properties->minor != 3))
        GTEST_SKIP() << "DeepSelect requires SM100 or SM103";
    constexpr int vocab       = 129280;
    auto          options     = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto          materialize = [](const auto& result, const torch::Tensor& dense) {
        auto expanded = torch::full_like(dense, -INFINITY);
        expanded.scatter_(1, std::get<1>(result).to(torch::kInt64), std::get<0>(result));
        return torch::where(std::get<2>(result).to(torch::kBool).unsqueeze(1), dense, expanded);
    };
    for (int stream_index = 0; stream_index < 2; ++stream_index) {
        auto                       stream = at::cuda::getStreamFromPool();
        c10::cuda::CUDAStreamGuard guard(stream);
        for (int batch : {64, 8, 128}) {
            auto a = torch::randn({batch, vocab}, options);
            auto b = torch::randn({batch, vocab}, options);
            b.select(0, 0).zero_();
            auto expected_a = a.clone();
            auto expected_b = b.clone();
            // Queue both producers before reading either result; scratch reuse
            // must not alias the compact logits, IDs or fallback flags.
            auto output_a = torch_ext::deferred_compact_topk_topp(a, 1024, 0.9f);
            auto output_b = torch_ext::deferred_compact_topk_topp(b, 1024, 0.9f);
            ASSERT_TRUE(torch_ext::deferred_topk_topp_filter(expected_a, 1024, 0.9f));
            ASSERT_TRUE(torch_ext::deferred_topk_topp_filter(expected_b, 1024, 0.9f));
            EXPECT_TRUE(torch::equal(materialize(output_a, a), expected_a));
            EXPECT_TRUE(torch::equal(materialize(output_b, b), expected_b));
        }
        auto input = torch::randn({8, vocab}, options);
        torch_ext::deferred_compact_topk_topp(input, 1024, 0.9f);
        ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
        at::cuda::CUDAGraph graph;
        graph.capture_begin();
        auto captured = torch_ext::deferred_compact_topk_topp(input, 1024, 0.9f);
        graph.capture_end();
        for (int replay = 0; replay < 3; ++replay) {
            input.normal_();
            if (replay == 1)
                input.select(0, 0).zero_();
            auto expected = input.clone();
            graph.replay();
            ASSERT_TRUE(torch_ext::deferred_topk_topp_filter(expected, 1024, 0.9f));
            EXPECT_TRUE(torch::equal(materialize(captured, input), expected));
        }
    }
}

TEST_F(CudaSamplerTest, testDeferredRejectionBridge) {
    initializeSamplerPython();
    auto options = torch::TensorOptions().device(torch::kCUDA);
    auto target  = torch::zeros({4, 4, 4}, options);
    for (int64_t request = 0; request < 4; ++request) {
        for (int64_t position = 0; position < 3; ++position) {
            target.index_put_({request, position, position < request ? 0 : 2}, 1.0f);
        }
    }
    target.select(1, 3).select(1, 3).fill_(1.0f);
    auto                    draft   = torch::zeros({4, 3}, options.dtype(torch::kInt32));
    auto                    uniform = torch::full({4, 4}, 0.5f, options);
    auto                    output  = torch::empty({4, 4}, options.dtype(torch::kInt32));
    auto                    lengths = torch::empty({4}, options.dtype(torch::kInt32));
    auto                    random  = torch::ones({4}, options.dtype(torch::kBool));
    RejectionSamplingParams params;
    params.draft_token_ids_d           = draft;
    params.uniform_samples_d           = uniform;
    params.target_logits_d             = target.log();
    params.output_token_ids_d          = output;
    params.output_accepted_token_num_d = lengths;
    params.do_sample_d                 = random;
    params.deferred_token_sampling     = true;
    params.draft_point_mass            = true;
    execRejectionSampling(params);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(toHostInt(output), (std::vector<int32_t>{2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3}));
    EXPECT_EQ(toHostInt(lengths), (std::vector<int32_t>{1, 2, 3, 4}));
    pybind11::exec(R"PY(
from unittest.mock import patch
from rtp_llm.models_py.sampling import adapter, rejection_graph

assert callable(rejection_graph._native_replay)
torch.cuda.synchronize()
rejection_graph._entries.clear()
native_replay = rejection_graph._native_replay
native_calls = []
def counted_native_replay(*args):
    native_calls.append(1)
    return native_replay(*args)

producer = torch.cuda.Stream()
consumer = torch.cuda.current_stream()
saved_outputs = []
with patch.object(rejection_graph, "_native_replay", counted_native_replay), patch.dict(
    os.environ, {"RTP_LLM_GUMBEL_REJECTION_GRAPH": "1"}
):
    for iteration in range(6):
        with torch.cuda.stream(producer):
            logits = torch.randn(16, 6, 513, device="cuda")
            ids = torch.randint(513, (16, 5), device="cuda", dtype=torch.int32)
            seeds = torch.arange(16, device="cuda", dtype=torch.int64) + iteration * 13
            positions = seeds + iteration * 17
            sample = seeds % 2 == 0
            caps = (seeds % 6).int()
            expected, expected_lengths = adapter._rejection_sample_logits_eager(
                logits, None, ids, seeds, positions, sample, None, caps, clean_output=True
            )
        consumer.wait_stream(producer)
        output = torch.full((16, 6), -777, device="cuda", dtype=torch.int32)
        lengths = torch.full((16,), -777, device="cuda", dtype=torch.int32)
        assert rejection_graph.try_replay(
            adapter._rejection_sample_logits_eager,
            logits, ids, seeds, positions, sample, None, caps, output, lengths,
        )
        saved_outputs.append((output, lengths, expected, expected_lengths))
        del logits, ids, seeds, positions, sample, caps

torch.cuda.synchronize()
assert len(native_calls) == 6
for output, lengths, expected, expected_lengths in saved_outputs:
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(lengths, expected_lengths, rtol=0, atol=0)
rejection_graph._entries.clear()
)PY");
}

TEST_F(CudaSamplerTest, testDeferredNativePlanBypassesPythonAndExpires) {
    initializeSamplerPython();
    auto native_env_patch = pybind11::module_::import("unittest.mock")
                                .attr("patch")
                                .attr("dict")(pybind11::module_::import("os").attr("environ"),
                                              pybind11::dict(pybind11::arg("RTP_LLM_GUMBEL_REJECTION_NATIVE") = "0"));
    native_env_patch.attr("__enter__")();
    struct RestoreGraphEnv {
        std::optional<std::string> value;
        ~RestoreGraphEnv() {
            if (value) {
                setenv("RTP_LLM_GUMBEL_REJECTION_GRAPH", value->c_str(), 1);
            } else {
                unsetenv("RTP_LLM_GUMBEL_REJECTION_GRAPH");
            }
        }
    } restore;
    if (const char* value = std::getenv("RTP_LLM_GUMBEL_REJECTION_GRAPH")) {
        restore.value = value;
    }
    setenv("RTP_LLM_GUMBEL_REJECTION_GRAPH", "1", 1);
    auto                    options = torch::TensorOptions().device(torch::kCUDA);
    RejectionSamplingParams params;
    params.target_logits_d = torch::full({16, 6, 513}, -std::numeric_limits<float>::infinity(), options);
    params.target_logits_d.select(2, 0).fill_(0);
    params.draft_token_ids_d           = torch::zeros({16, 5}, options.dtype(torch::kInt32));
    params.output_token_ids_d          = torch::empty({16, 6}, options.dtype(torch::kInt32));
    params.output_accepted_token_num_d = torch::empty({16}, options.dtype(torch::kInt32));
    params.do_sample_d                 = torch::ones({16}, options.dtype(torch::kBool));
    params.deferred_token_sampling     = true;
    params.draft_point_mass            = true;
    execRejectionSampling(params);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto graph = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_graph");
    auto reject_dispatch =
        pybind11::cpp_function([](pybind11::args) { throw std::runtime_error("unexpected Python Graph dispatch"); });
    auto patcher =
        pybind11::module_::import("unittest.mock").attr("patch").attr("object")(graph, "try_replay", reject_dispatch);
    patcher.attr("__enter__")();
    auto previous_output = params.output_token_ids_d;
    params.target_logits_d.fill_(-std::numeric_limits<float>::infinity());
    params.target_logits_d.select(2, 3).fill_(0);
    params.draft_token_ids_d.fill_(3);
    params.output_token_ids_d          = torch::full_like(previous_output, -777);
    params.output_accepted_token_num_d = torch::full_like(params.output_accepted_token_num_d, -777);
    EXPECT_NO_THROW(execRejectionSampling(params));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_TRUE(torch::all(params.output_token_ids_d == 3).item<bool>());
    EXPECT_TRUE(torch::all(previous_output == 0).item<bool>());
    EXPECT_TRUE(torch::all(params.output_accepted_token_num_d == 6).item<bool>());

    // A competing eviction guard must force a nonblocking fallback. Hold it
    // on another thread: relocking std::mutex on its owner thread is undefined.
    pybind11::exec(R"PY(
import threading
from rtp_llm.models_py.sampling import rejection_graph
native_guard_ready = threading.Event()
native_guard_release = threading.Event()
native_guard_acquired = []
def hold_native_guard():
    acquired = rejection_graph._try_native_guard()
    native_guard_acquired.append(acquired)
    native_guard_ready.set()
    if acquired:
        try:
            native_guard_release.wait(10)
        finally:
            rejection_graph._release_native_guard()
native_guard_thread = threading.Thread(target=hold_native_guard)
native_guard_thread.start()
assert native_guard_ready.wait(10) and native_guard_acquired == [True]
)PY");
    EXPECT_THROW(execRejectionSampling(params), pybind11::error_already_set);
    pybind11::exec(R"PY(
native_guard_release.set()
native_guard_thread.join(timeout=10)
assert not native_guard_thread.is_alive()
)PY");
    EXPECT_NO_THROW(execRejectionSampling(params));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Disabling Graph must bypass the native plan as well as the Python cache.
    setenv("RTP_LLM_GUMBEL_REJECTION_GRAPH", "0", 1);
    EXPECT_THROW(execRejectionSampling(params), pybind11::error_already_set);
    setenv("RTP_LLM_GUMBEL_REJECTION_GRAPH", "1", 1);
    // An expired Python owner invalidates the native weak registry entry.
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    graph.attr("_entries").attr("clear")();
    EXPECT_THROW(execRejectionSampling(params), pybind11::error_already_set);
    patcher.attr("__exit__")(pybind11::none(), pybind11::none(), pybind11::none());

    execRejectionSampling(params);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_TRUE(torch::all(params.output_accepted_token_num_d == 6).item<bool>());
    native_env_patch.attr("__exit__")(pybind11::none(), pybind11::none(), pybind11::none());
}

TEST_F(CudaSamplerTest, testDeferredPackedRngMatchesEagerAcrossCacheMisses) {
    initializeSamplerPython();
    auto native_env_patch = pybind11::module_::import("unittest.mock")
                                .attr("patch")
                                .attr("dict")(pybind11::module_::import("os").attr("environ"),
                                              pybind11::dict(pybind11::arg("RTP_LLM_GUMBEL_REJECTION_NATIVE") = "0"));
    native_env_patch.attr("__enter__")();
    const char*             old_env  = std::getenv("RTP_LLM_GUMBEL_REJECTION_GRAPH");
    const auto              previous = old_env ? std::make_optional(std::string(old_env)) : std::nullopt;
    auto                    options  = torch::TensorOptions().device(torch::kCUDA);
    RejectionSamplingParams params;
    params.deferred_token_sampling = true;
    params.draft_point_mass        = true;
    std::vector<at::Generator> eager_generators;
    for (int i = 0; i < 16; ++i) {
        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
        generator.set_current_seed(8147 + i);
        params.generator.push_back(generator);
        auto reference = torch::make_generator<at::CUDAGeneratorImpl>();
        reference.set_current_seed(8147 + i);
        eager_generators.push_back(reference);
    }
    auto graph = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_graph");
    // Repeated states exercise cold capture, native hits, cap-key changes, and
    // eviction. Compare both tokens and generator progression to eager.
    for (int iteration = 0; iteration < 10; ++iteration) {
        const int batch        = iteration == 8 ? 8 : 16;
        params.target_logits_d = torch::randn({batch, 6, 513}, options);
        params.target_logits_d.select(2, 0).add_(6);
        params.draft_token_ids_d = torch::zeros({batch, 5}, options.dtype(torch::kInt32));
        params.do_sample_d       = (torch::arange(batch, options.dtype(torch::kInt32)) % 3 != 0);
        params.acceptance_cap =
            iteration >= 4 && iteration <= 6 ? torch::arange(batch, options.dtype(torch::kInt32)) % 6 : torch::Tensor();
        params.output_token_ids_d          = torch::full({batch, 6}, -777, options.dtype(torch::kInt32));
        params.output_accepted_token_num_d = torch::empty({batch}, options.dtype(torch::kInt32));
        if (iteration == 7) {
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            graph.attr("_entries").attr("clear")();
        }
        setenv("RTP_LLM_GUMBEL_REJECTION_GRAPH", "1", 1);
        execRejectionSampling(params);
        auto expected                        = params;
        expected.generator                   = eager_generators;
        expected.output_token_ids_d          = torch::full_like(params.output_token_ids_d, -999);
        expected.output_accepted_token_num_d = torch::empty_like(params.output_accepted_token_num_d);
        setenv("RTP_LLM_GUMBEL_REJECTION_GRAPH", "0", 1);
        execRejectionSampling(expected);
        EXPECT_TRUE(torch::equal(params.output_token_ids_d, expected.output_token_ids_d));
        EXPECT_TRUE(torch::equal(params.output_accepted_token_num_d, expected.output_accepted_token_num_d));
        for (int i = 0; i < 16; ++i) {
            EXPECT_TRUE(torch::equal(params.generator[i].get_state(), eager_generators[i].get_state()));
        }
    }
    if (previous) {
        setenv("RTP_LLM_GUMBEL_REJECTION_GRAPH", previous->c_str(), 1);
    } else {
        unsetenv("RTP_LLM_GUMBEL_REJECTION_GRAPH");
    }
    native_env_patch.attr("__exit__")(pybind11::none(), pybind11::none(), pybind11::none());
}

TEST_F(CudaSamplerTest, testNativeEagerBucketsWithoutGraphs) {
    initializeSamplerPython();
    struct PatchScope {
        pybind11::object patch;
        explicit PatchScope(pybind11::object value): patch(std::move(value)) {
            patch.attr("__enter__")();
        }
        ~PatchScope() {
            patch.attr("__exit__")(pybind11::none(), pybind11::none(), pybind11::none());
        }
    };
    auto       mock         = pybind11::module_::import("unittest.mock");
    auto       environment  = pybind11::module_::import("os").attr("environ");
    auto       patch_dict   = mock.attr("patch").attr("dict");
    auto       patch_object = mock.attr("patch").attr("object");
    PatchScope env(patch_dict(environment,
                              pybind11::dict(pybind11::arg("RTP_LLM_GUMBEL_REJECTION_GRAPH")  = "0",
                                             pybind11::arg("RTP_LLM_GUMBEL_REJECTION_NATIVE") = "1")));
    auto       forbidden = pybind11::cpp_function([](pybind11::args, pybind11::kwargs) {
        throw std::runtime_error("unexpected cold builder or CUDA Graph construction");
    });
    PatchScope no_graph(patch_object(pybind11::module_::import("torch").attr("cuda"), "CUDAGraph", forbidden));
    auto       native = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_native");
    auto eager = pybind11::module_::import("rtp_llm.models_py.sampling.adapter").attr("_rejection_sample_logits_eager");
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    native.attr("_plans").attr("clear")();
    auto                       options = torch::TensorOptions().device(torch::kCUDA);
    std::vector<at::Generator> generators;
    for (int64_t i = 0; i < 64; ++i) {
        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
        generator.set_current_seed(12345 + i);
        generators.push_back(generator);
    }
    std::unique_ptr<PatchScope>               no_builder;
    std::vector<std::array<torch::Tensor, 4>> outputs;
    RejectionSamplingParams                   last;
    int                                       iteration = 0;
    for (int64_t batch : {64, 62, 63, 64, 62, 63}) {
        RejectionSamplingParams params;
        params.target_logits_d = torch::zeros({batch, 6, 129280}, options);
        params.target_logits_d.select(2, iteration % 3).fill_(std::log(129279.0f * 0.7f / 0.3f));
        params.draft_token_ids_d           = torch::full({batch, 5}, iteration % 3, options.dtype(torch::kInt32));
        params.output_token_ids_d          = torch::full({batch, 6}, -777, options.dtype(torch::kInt32));
        params.output_accepted_token_num_d = torch::full({batch}, -777, options.dtype(torch::kInt32));
        auto indices                       = torch::arange(batch, options.dtype(torch::kInt64));
        params.do_sample_d                 = (indices + iteration).remainder(2) == 0;
        params.acceptance_cap              = (indices + iteration).remainder(6).to(torch::kInt32);
        params.deferred_token_sampling     = true;
        params.draft_point_mass            = true;
        params.generator.assign(generators.begin(), generators.begin() + batch);
        for (auto& generator : params.generator)
            generator.set_offset(iteration * 32);
        auto seeds     = indices + 12345;
        auto positions = torch::full({batch}, iteration * 32, options.dtype(torch::kInt64));
        std::pair<torch::Tensor, torch::Tensor> expected;
        {
            PatchScope oracle(
                patch_dict(environment, pybind11::dict(pybind11::arg("RTP_LLM_GUMBEL_REJECTION_NATIVE") = "0")));
            expected = eager(params.target_logits_d,
                             pybind11::none(),
                             params.draft_token_ids_d,
                             seeds,
                             positions,
                             params.do_sample_d,
                             pybind11::none(),
                             params.acceptance_cap,
                             pybind11::arg("clean_output") = true)
                           .cast<std::pair<torch::Tensor, torch::Tensor>>();
        }
        execRejectionSampling(params);
        if (iteration == 0) {
            ASSERT_EQ(pybind11::len(native.attr("_plans")), 1);
            no_builder = std::make_unique<PatchScope>(patch_object(native, "build_plan", forbidden));
        }
        for (const auto& generator : params.generator) {
            EXPECT_EQ(generator.get_offset(), static_cast<uint64_t>((iteration + 1) * 32));
        }
        outputs.push_back(
            {params.output_token_ids_d, params.output_accepted_token_num_d, expected.first, expected.second});
        last = params;
        ++iteration;
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(pybind11::len(native.attr("_plans")), 1);
    for (const auto& output : outputs) {
        EXPECT_TRUE(torch::equal(output[0], output[2]));
        EXPECT_TRUE(torch::equal(output[1], output[3]));
    }
    // Cumulative probability consumers are outside the native point-mass ABI.
    // They must retain the original eager route, even with a matching bucket.
    last.cum_log_probs = torch::zeros({63 * 6}, options);
    EXPECT_NO_THROW(execRejectionSampling(last));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_TRUE(torch::isfinite(last.cum_log_probs).all().item<bool>());
}

TEST_F(CudaSamplerTest, DISABLED_benchmarkRejectionMemory) {
    initializeSamplerPython();
    const bool optimized =
        std::string(std::getenv("SAMPLING_MEMORY_MODE") ? std::getenv("SAMPLING_MEMORY_MODE") : "legacy")
        == "optimized";
    constexpr int64_t vocab = 154880, steps = 3, rows = steps + 1;
    initializeSamplingRuntime(vocab);
    if (optimized) {
        ASSERT_TRUE(supportsDeferredRejectionSampling());
    }
    auto cuda    = pybind11::module_::import("torch").attr("cuda");
    auto bytes   = [&](const char* name) { return cuda.attr(name)().cast<int64_t>(); };
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto stream  = at::cuda::getCurrentCUDAStream().stream();
    for (int64_t batch : {16, 20, 24, 32, 64, 128}) {
        for (bool compact : {false, true}) {
            for (bool random_sampling : {false, true}) {
                for (bool accept_all : {false, true}) {
                    auto run = [&](bool report, int repeat) {
                        cuda.attr("synchronize")();
                        cuda.attr("empty_cache")();
                        const auto baseline = bytes("memory_allocated");
                        cuda.attr("reset_peak_memory_stats")();
                        auto logits = torch::zeros({batch * rows, vocab}, options);
                        logits.select(1, 0).fill_(accept_all ? 80.0f : -80.0f);
                        auto probabilities = optimized ? torch::Tensor() : torch::empty_like(logits);
                        auto tokens        = torch::zeros({batch * rows, 2}, options.dtype(torch::kInt32));
                        auto target_ids    = torch::empty({batch, rows}, options.dtype(torch::kInt32));
                        auto draft_ids     = torch::zeros({batch, steps}, options.dtype(torch::kInt32));
                        auto draft_probs   = compact ? torch::Tensor() : torch::zeros({batch, steps, vocab}, options);
                        if (!compact) {
                            draft_probs.select(2, 0).fill_(1);
                        }
                        auto uniforms = torch::rand({batch, rows}, options);
                        auto output   = torch::empty({batch, rows}, options.dtype(torch::kInt32));
                        auto lengths  = torch::empty({batch}, options.dtype(torch::kInt32));
                        auto random   = torch::full({batch}, random_sampling, options.dtype(torch::kBool));
                        std::vector<at::Generator> target_generators, verify_generators;
                        for (int64_t i = 0; i < batch * rows; ++i) {
                            auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                            generator.set_current_seed(20261109 + i);
                            target_generators.push_back(generator);
                        }
                        for (int64_t i = 0; i < batch; ++i) {
                            auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                            generator.set_current_seed(20261109 + i);
                            verify_generators.push_back(generator);
                        }
                        GreedyParams target(
                            {logits,
                             cudaIntTensor(std::vector<int32_t>(batch * rows, 1), {batch * rows}),
                             cudaIntTensor(std::vector<int32_t>(batch * rows, 1), {batch * rows}),
                             tokens,
                             1,
                             pinnedIntTensor(std::vector<int32_t>(batch * rows, random_sampling ? 0 : 1)),
                             pinnedFloatTensor(std::vector<float>(batch * rows, 1)),
                             pinnedFloatTensor(std::vector<float>(batch * rows, 1)),
                             nullopt,
                             nullopt,
                             nullopt,
                             nullopt,
                             std::nullopt,
                             nullopt,
                             nullopt,
                             nullopt,
                             target_generators});
                        target.defer_token_sampling = optimized;
                        if (!optimized) {
                            target.output_all_probs = probabilities;
                        }
                        RejectionSamplingParams deferred;
                        deferred.target_logits_d             = logits.reshape({batch, rows, vocab});
                        deferred.draft_token_ids_d           = draft_ids;
                        deferred.output_token_ids_d          = output;
                        deferred.output_accepted_token_num_d = lengths;
                        deferred.do_sample_d                 = random;
                        deferred.deferred_token_sampling     = true;
                        deferred.draft_point_mass            = compact;
                        deferred.draft_probs_d               = draft_probs;
                        deferred.generator                   = verify_generators;
                        cuda.attr("synchronize")();
                        const auto input_bytes = bytes("memory_allocated") - baseline;
                        cuda.attr("reset_peak_memory_stats")();
                        const auto sampled = execSampleGreedy(target);
                        if (optimized) {
                            ASSERT_TRUE(sampled.deferred_token_sampling);
                            execRejectionSampling(deferred);
                        } else {
                            target_ids.copy_(tokens.select(1, 1).reshape({batch, rows}), true);
                            auto status = compact ?
                                              invokePointMassRejectionSampling(draft_ids.data_ptr<int32_t>(),
                                                                               uniforms.data_ptr<float>(),
                                                                               probabilities.data_ptr<float>(),
                                                                               target_ids.data_ptr<int32_t>(),
                                                                               1,
                                                                               output.data_ptr<int32_t>(),
                                                                               lengths.data_ptr<int32_t>(),
                                                                               random.data_ptr<bool>(),
                                                                               batch,
                                                                               steps,
                                                                               vocab,
                                                                               stream) :
                                              invokeRejectionSampling<float, int32_t>(draft_probs.data_ptr<float>(),
                                                                                      draft_ids.data_ptr<int32_t>(),
                                                                                      uniforms.data_ptr<float>(),
                                                                                      probabilities.data_ptr<float>(),
                                                                                      target_ids.data_ptr<int32_t>(),
                                                                                      1,
                                                                                      output.data_ptr<int32_t>(),
                                                                                      lengths.data_ptr<int32_t>(),
                                                                                      random.data_ptr<bool>(),
                                                                                      batch,
                                                                                      steps,
                                                                                      vocab,
                                                                                      stream);
                            ASSERT_EQ(status, cudaSuccess);
                        }
                        cuda.attr("synchronize")();
                        const auto peak     = bytes("max_memory_allocated") - baseline;
                        const auto retained = bytes("memory_allocated") - baseline;
                        const auto reserved = bytes("max_memory_reserved");
                        ASSERT_TRUE(torch::all(lengths == (accept_all ? rows : 1)).item<bool>());
                        ASSERT_TRUE(torch::all(output.select(1, 0) >= 0).item<bool>());
                        ASSERT_TRUE(torch::all(output.select(1, 0) < vocab).item<bool>());
                        ASSERT_EQ(torch::all(output.select(1, 0) == 0).item<bool>(), accept_all);
                        if (report) {
                            std::cout << "[rejection-memory] optimized=" << optimized << " batch=" << batch
                                      << " steps=" << steps << " vocab=" << vocab << " compact=" << compact
                                      << " random=" << random_sampling << " accept_all=" << accept_all
                                      << " repeat=" << repeat << " baseline_bytes=" << baseline
                                      << " inputs_bytes=" << input_bytes << " peak_bytes=" << peak
                                      << " transient_bytes=" << peak - input_bytes << " retained_bytes=" << retained
                                      << " reserved_bytes=" << reserved << std::endl;
                            if (optimized && !compact && std::getenv("SAMPLING_MEMORY_TIME_LOG")) {
                                auto flush =
                                    torch::zeros({kL2FlushBytes / static_cast<int64_t>(sizeof(float))}, options);
                                const auto gpu_us =
                                    benchmarkCudaEventUs([&]() { execRejectionSampling(deferred); }, flush, 10, 50);
                                std::cout << "[rejection-log-timing] batch=" << batch << " steps=" << steps
                                          << " random=" << random_sampling << " accept_all=" << accept_all
                                          << " repeat=" << repeat << " cuda_span_us=" << gpu_us << std::endl;
                            }
                        }
                    };
                    run(false, -1);
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        run(true, repeat);
                    }
                }
            }
        }
    }
}

TEST_F(CudaSamplerTest, DISABLED_benchmarkRejectionStages) {
    initializeSamplerPython();
    struct RestoreFlags {
        std::vector<std::pair<std::string, std::optional<std::string>>> previous;
        ~RestoreFlags() {
            for (const auto& [name, value] : previous) {
                if (value) {
                    setenv(name.c_str(), value->c_str(), 1);
                } else {
                    unsetenv(name.c_str());
                }
            }
        }
    } restore;
    for (const char* name : {"RTP_LLM_USE_GUMBEL_SAMPLING", "RTP_LLM_USE_VLLM_REJECTION_SAMPLING"}) {
        const char* value = std::getenv(name);
        restore.previous.emplace_back(name, value ? std::make_optional(std::string(value)) : std::nullopt);
    }
    setenv("RTP_LLM_USE_GUMBEL_SAMPLING", "0", 1);
    setenv("RTP_LLM_USE_VLLM_REJECTION_SAMPLING", "1", 1);
    constexpr int64_t vocab      = 154880;
    constexpr int     warmup     = 10;
    constexpr int     iterations = 50;
    initializeSamplingRuntime(vocab);
    ASSERT_TRUE(supportsDeferredRejectionSampling());
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto flush   = torch::zeros({kL2FlushBytes / static_cast<int64_t>(sizeof(float))}, options);
    auto stream  = at::cuda::getCurrentCUDAStream().stream();
    struct Timing {
        double gpu_us   = 0;
        double wall_us  = 0;
        double accepted = 0;
    };
    auto measure = [&](const std::function<void()>& run, const torch::Tensor& lengths) {
        for (int i = 0; i < warmup; ++i) {
            flushL2Cache(flush);
            run();
        }
        cudaStreamSynchronize(stream);
        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        Timing result;
        for (int i = 0; i < iterations; ++i) {
            flushL2Cache(flush);
            cudaStreamSynchronize(stream);
            auto started = std::chrono::steady_clock::now();
            cudaEventRecord(begin, stream);
            run();
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);
            auto  stopped      = std::chrono::steady_clock::now();
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, begin, end);
            result.gpu_us += milliseconds * 1000;
            result.wall_us += std::chrono::duration<double, std::micro>(stopped - started).count();
            result.accepted += lengths.sum().item<int64_t>() / static_cast<double>(lengths.numel());
        }
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        result.gpu_us /= iterations;
        result.wall_us /= iterations;
        result.accepted /= iterations;
        return result;
    };
    std::cout << "[rejection-stage-config] vocab=" << vocab << " warmup=" << warmup << " iterations=" << iterations
              << " repeats=3 l2_flush_bytes=" << kL2FlushBytes
              << " cache_flush_excluded=1 gpu_metric=cuda_event_enclosed_span wall_metric=synchronized_call"
              << " draft=dense_onehot_or_compact target_top_k=0 target_top_p=1 temperature=1" << std::endl;
    for (int64_t steps : {3}) {
        const int64_t rows = steps + 1;
        for (int64_t batch : {8, 16, 20, 24}) {
            for (bool compact : {false, true}) {
                for (const std::string distribution : {"reject_first", "mixed", "accept_all"}) {
                    auto  original = torch::zeros({batch * rows, vocab}, options);
                    float boost    = distribution == "reject_first" ? -80.0f :
                                     distribution == "accept_all"   ? 80.0f :
                                                                    static_cast<float>(std::log((vocab - 1) * 0.6 / 0.4));
                    original.select(1, 0).fill_(boost);
                    auto logits        = original.clone();
                    auto probabilities = torch::empty_like(logits);
                    auto tokens        = torch::zeros({batch * rows, 2}, options.dtype(torch::kInt32));
                    auto target_ids    = torch::empty({batch, rows}, options.dtype(torch::kInt32));
                    auto draft_ids     = torch::zeros({batch, steps}, options.dtype(torch::kInt32));
                    auto draft_probs   = compact ? torch::Tensor() : torch::zeros({batch, steps, vocab}, options);
                    if (!compact) {
                        draft_probs.select(2, 0).fill_(1);
                    }
                    auto                       uniforms = torch::rand({batch, rows}, options);
                    auto                       output   = torch::empty({batch, rows}, options.dtype(torch::kInt32));
                    auto                       lengths  = torch::empty({batch}, options.dtype(torch::kInt32));
                    auto                       random   = torch::ones({batch}, options.dtype(torch::kBool));
                    std::vector<at::Generator> target_generators, verify_generators;
                    for (int64_t i = 0; i < batch * rows; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(20261107 + i);
                        target_generators.push_back(generator);
                    }
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(20261107 + i);
                        verify_generators.push_back(generator);
                    }
                    GreedyParams            target({logits,
                                                    cudaIntTensor(std::vector<int32_t>(batch * rows, 1), {batch * rows}),
                                                    cudaIntTensor(std::vector<int32_t>(batch * rows, 1), {batch * rows}),
                                                    tokens,
                                                    1,
                                                    pinnedIntTensor(std::vector<int32_t>(batch * rows, 0)),
                                                    pinnedFloatTensor(std::vector<float>(batch * rows, 1)),
                                                    pinnedFloatTensor(std::vector<float>(batch * rows, 1)),
                                                    nullopt,
                                                    nullopt,
                                                    nullopt,
                                                    nullopt,
                                                    probabilities,
                                                    nullopt,
                                                    nullopt,
                                                    nullopt,
                                                    target_generators});
                    RejectionSamplingParams deferred;
                    deferred.target_logits_d             = logits.reshape({batch, rows, vocab});
                    deferred.draft_token_ids_d           = draft_ids;
                    deferred.output_token_ids_d          = output;
                    deferred.output_accepted_token_num_d = lengths;
                    deferred.do_sample_d                 = random;
                    deferred.deferred_token_sampling     = true;
                    deferred.draft_point_mass            = compact;
                    deferred.draft_probs_d               = draft_probs;
                    deferred.generator                   = verify_generators;
                    auto legacy_verify                   = [&]() {
                        uniforms.uniform_(0, 1);
                        auto status = compact ? invokePointMassRejectionSampling(draft_ids.data_ptr<int32_t>(),
                                                                                 uniforms.data_ptr<float>(),
                                                                                 probabilities.data_ptr<float>(),
                                                                                 target_ids.data_ptr<int32_t>(),
                                                                                 1,
                                                                                 output.data_ptr<int32_t>(),
                                                                                 lengths.data_ptr<int32_t>(),
                                                                                 random.data_ptr<bool>(),
                                                                                 batch,
                                                                                 steps,
                                                                                 vocab,
                                                                                 stream) :
                                                                  invokeRejectionSampling<float, int32_t>(draft_probs.data_ptr<float>(),
                                                                                        draft_ids.data_ptr<int32_t>(),
                                                                                        uniforms.data_ptr<float>(),
                                                                                        probabilities.data_ptr<float>(),
                                                                                        target_ids.data_ptr<int32_t>(),
                                                                                        1,
                                                                                        output.data_ptr<int32_t>(),
                                                                                        lengths.data_ptr<int32_t>(),
                                                                                        random.data_ptr<bool>(),
                                                                                        batch,
                                                                                        steps,
                                                                                        vocab,
                                                                                        stream);
                        TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
                    };
                    auto legacy_pipeline = [&]() {
                        logits.copy_(original, true);
                        target.defer_token_sampling = false;
                        target.output_all_probs     = probabilities;
                        execSampleGreedy(target);
                        target_ids.copy_(tokens.select(1, 1).reshape({batch, rows}), true);
                        legacy_verify();
                    };
                    auto vllm_verify   = [&]() { execRejectionSampling(deferred); };
                    auto vllm_pipeline = [&]() {
                        logits.copy_(original, true);
                        target.defer_token_sampling = true;
                        target.output_all_probs     = std::nullopt;
                        TORCH_CHECK(execSampleGreedy(target).deferred_token_sampling);
                        vllm_verify();
                    };
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        for (int i = 0; i < 4; ++i) {
                            const int stage = (i + repeat) % 4;
                            Timing    timing;
                            if (stage == 0) {
                                legacy_pipeline();
                                timing = measure(legacy_verify, lengths);
                            } else if (stage == 1) {
                                logits.copy_(original, true);
                                timing = measure(vllm_verify, lengths);
                            } else if (stage == 2) {
                                timing = measure(legacy_pipeline, lengths);
                            } else {
                                timing = measure(vllm_pipeline, lengths);
                            }
                            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                            ASSERT_GE(lengths.min().item<int>(), 1);
                            ASSERT_LE(lengths.max().item<int>(), rows);
                            if (distribution == "reject_first") {
                                ASSERT_TRUE(torch::all(lengths == 1).item<bool>());
                            } else if (distribution == "accept_all") {
                                ASSERT_TRUE(torch::all(lengths == rows).item<bool>());
                            }
                            std::cout << "[rejection-stage-result] steps=" << steps << " batch=" << batch
                                      << " compact=" << compact << " distribution=" << distribution
                                      << " repeat=" << repeat << " stage=" << stage << " cuda_span_us=" << timing.gpu_us
                                      << " wall_us=" << timing.wall_us << " accepted=" << timing.accepted << std::endl;
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CudaSamplerTest, DISABLED_benchmarkVllmGumbelBridgeVsFlashInfer) {
    initializeSamplerPython();
    struct RestoreEnvironment {
        std::optional<std::string> value;
        ~RestoreEnvironment() {
            if (value.has_value()) {
                setenv("RTP_LLM_USE_GUMBEL_SAMPLING", value->c_str(), 1);
            } else {
                unsetenv("RTP_LLM_USE_GUMBEL_SAMPLING");
            }
        }
    };
    const auto previous = std::getenv("RTP_LLM_USE_GUMBEL_SAMPLING");
    RestoreEnvironment restore{previous ? std::make_optional(std::string(previous)) : std::nullopt};
    constexpr int64_t vocab_size = 154880;
    constexpr int warmup = 10;
    constexpr int iterations = 50;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto flush = torch::zeros({kL2FlushBytes / static_cast<int64_t>(sizeof(float))}, options);
    std::cout << "[rtp-gumbel-benchmark] timing=cuda_event warmup=" << warmup << " iterations=" << iterations
              << " l2_flush_bytes=" << kL2FlushBytes
              << " l2_flush_timing=excluded measured=logits_reset_and_execSampleGreedy"
              << " includes=token_copies_seed_offset_metadata_sampling_and_python_bridge" << std::endl;
    for (int64_t batch_size : {1, 32, 64, 128}) {
        auto original = torch::randn({batch_size, vocab_size}, options);
        auto logits = original.clone();
        auto tokens = torch::zeros({batch_size, 2}, options.dtype(torch::kInt32));
        std::vector<at::Generator> generators;
        for (int64_t row = 0; row < batch_size; ++row) {
            generators.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
        }
        GreedyParams params({logits,
                             cudaIntTensor(std::vector<int32_t>(batch_size, 1), {batch_size}),
                             cudaIntTensor(std::vector<int32_t>(batch_size, 1), {batch_size}),
                             tokens,
                             1,
                             pinnedIntTensor(std::vector<int32_t>(batch_size, 0)),
                             pinnedFloatTensor(std::vector<float>(batch_size, 1.0f)),
                             pinnedFloatTensor(std::vector<float>(batch_size, 1.0f)),
                             nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt,
                             generators});
        std::vector<double> timings;
        for (const char* enabled : {"0", "1"}) {
            ASSERT_EQ(setenv("RTP_LLM_USE_GUMBEL_SAMPLING", enabled, 1), 0);
            for (int64_t row = 0; row < batch_size; ++row) {
                generators[row].set_current_seed(20261106 + row);
                generators[row].set_offset(0);
            }
            auto run = [&]() {
                logits.copy_(original, true);
                execSampleGreedy(params);
            };
            timings.push_back(benchmarkCudaEventUs(run, flush, warmup, iterations));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            for (auto token : toHostInt(tokens.select(1, 1))) {
                ASSERT_GE(token, 0);
                ASSERT_LT(token, vocab_size);
            }
            ASSERT_GT(timings.back(), 0.0);
        }
        std::cout << "[rtp-gumbel-benchmark] batch=" << batch_size << " vocab=" << vocab_size
                  << " flashinfer_us=" << timings[0] << " gumbel_bridge_us=" << timings[1]
                  << " speedup=" << timings[0] / timings[1] << std::endl;
    }
}

TEST_F(CudaSamplerTest, DISABLED_benchmarkLatestFlashinferSamplingVsCurrentRtp) {
    enum class Kind { TopK, TopP, TopKTopP };
    struct Case {
        std::string name;
        Kind        kind;
        int64_t     batch_size;
        int64_t     vocab_size;
        int32_t     top_k;
        float       top_p;
    };

    constexpr int64_t      vocab_size  = 129280;
    const std::vector<int64_t> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 96, 128};
    std::vector<Case>      cases;
    cases.reserve(batch_sizes.size() * 3);
    for (auto batch_size : batch_sizes) {
        const auto suffix = "_b" + std::to_string(batch_size) + "_v" + std::to_string(vocab_size);
        cases.push_back({"top_k" + suffix, Kind::TopK, batch_size, vocab_size, 50, 1.0f});
        cases.push_back({"top_p" + suffix, Kind::TopP, batch_size, vocab_size, 0, 0.85f});
        cases.push_back({"top_k_top_p" + suffix, Kind::TopKTopP, batch_size, vocab_size, 100, 0.9f});
    }

    constexpr bool deterministic = true;
    constexpr int  warmup        = 20;
    constexpr int  iterations    = 100;
    auto           stream        = at::cuda::getCurrentCUDAStream().stream();
    auto           l2_flush_buffer =
        torch::empty({kL2FlushBytes / static_cast<int64_t>(sizeof(float))},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    l2_flush_buffer.zero_();
    cudaDeviceSynchronize();

    std::cout << "[sampling-benchmark] baseline=current_RTP_old_uniform_api latest=../flashinfer_seed_offset_api "
              << "timing=cuda_event warmup=" << warmup << " iterations=" << iterations
              << " l2_flush_bytes=" << kL2FlushBytes
              << " l2_flush_timing=excluded old_with_uniform_rng=sum_of_individually_flushed_rng_and_sampling"
              << std::endl;

    for (const auto& c : cases) {
        auto probs        = normalizedRandomProbs(c.batch_size, c.vocab_size).contiguous();
        auto old_uniform  = torch::rand({32, c.batch_size}, probs.options());
        auto rng_uniform  = torch::empty({32, c.batch_size}, probs.options());
        auto int_options  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
        auto old_samples  = torch::empty({1, c.batch_size}, int_options);
        auto old_valid    = torch::empty({c.batch_size}, bool_options);
        auto new_samples  = torch::empty({c.batch_size}, int_options);
        auto new_valid    = torch::empty({c.batch_size}, bool_options);
        auto top_k_h = torch::full({c.batch_size}, c.top_k, torch::TensorOptions().dtype(torch::kInt32)).pin_memory();
        auto top_p_h =
            torch::full({c.batch_size}, c.top_p, torch::TensorOptions().dtype(torch::kFloat32)).pin_memory();
        auto top_k_d     = top_k_h.to(torch::kCUDA, /*non_blocking=*/true).contiguous();
        auto top_p_d     = top_p_h.to(torch::kCUDA, /*non_blocking=*/true).contiguous();
        auto seed_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        auto seed_d      = torch::full({c.batch_size}, 20260525, seed_options);
        auto offset_d =
            torch::arange(0, c.batch_size * 32, 32, seed_options);

        auto old_kernel = [&](const torch::Tensor& uniform) {
            switch (c.kind) {
                case Kind::TopK:
                    ::top_k_sampling_from_probs(
                        probs, uniform, old_samples, old_valid, top_k_h, 0, deterministic, (int64_t)stream);
                    break;
                case Kind::TopP:
                    ::top_p_sampling_from_probs(
                        probs, uniform, old_samples, old_valid, top_p_h, 1.0, deterministic, (int64_t)stream);
                    break;
                case Kind::TopKTopP:
                    ::top_k_top_p_sampling_from_probs(probs,
                                                       uniform,
                                                       old_samples,
                                                       old_valid,
                                                       top_k_h,
                                                       0,
                                                       top_p_h,
                                                       1.0,
                                                       deterministic,
                                                       (int64_t)stream);
                    break;
            }
        };

        auto latest_kernel = [&]() {
            switch (c.kind) {
                case Kind::TopK:
                    rtp_llm::top_k_sampling_from_probs(probs,
                                                       new_samples,
                                                       new_valid,
                                                       std::nullopt,
                                                       top_k_d,
                                                       0,
                                                       deterministic,
                                                       seed_d,
                                                       0,
                                                       offset_d,
                                                       0,
                                                       (int64_t)stream);
                    break;
                case Kind::TopP:
                    rtp_llm::top_p_sampling_from_probs(probs,
                                                       new_samples,
                                                       new_valid,
                                                       std::nullopt,
                                                       top_p_d,
                                                       1.0,
                                                       deterministic,
                                                       seed_d,
                                                       0,
                                                       offset_d,
                                                       0,
                                                       (int64_t)stream);
                    break;
                case Kind::TopKTopP:
                    rtp_llm::top_k_top_p_sampling_from_probs(probs,
                                                             new_samples,
                                                             new_valid,
                                                             std::nullopt,
                                                             top_k_d,
                                                             0,
                                                             top_p_d,
                                                             1.0,
                                                             deterministic,
                                                             seed_d,
                                                             0,
                                                             offset_d,
                                                             0,
                                                             (int64_t)stream);
                    break;
            }
        };

        old_kernel(old_uniform);
        latest_kernel();
        cudaDeviceSynchronize();
        assertAllValid(old_valid);
        assertAllValid(new_valid);

        double old_kernel_us =
            benchmarkCudaEventUs([&]() { old_kernel(old_uniform); }, l2_flush_buffer, warmup, iterations);
        double old_uniform_rng_us =
            benchmarkCudaEventUs([&]() { rng_uniform.uniform_(0.0, 1.0); }, l2_flush_buffer, warmup, iterations);
        double old_full_us = old_kernel_us + old_uniform_rng_us;
        double latest_us = benchmarkCudaEventUs(latest_kernel, l2_flush_buffer, warmup, iterations);

        std::cout << "[sampling-benchmark] case=" << c.name << " batch=" << c.batch_size
                  << " vocab=" << c.vocab_size << " old_kernel_us=" << old_kernel_us
                  << " old_uniform_rng_us=" << old_uniform_rng_us << " old_with_uniform_rng_us=" << old_full_us
                  << " latest_us=" << latest_us
                  << " speedup_vs_old_kernel=" << (old_kernel_us / latest_us)
                  << " speedup_vs_old_with_rng=" << (old_full_us / latest_us) << std::endl;
    }
}

TEST_F(CudaSamplerTest, DISABLED_compareLatestFlashinferSamplingAccuracyVsCurrentRtp) {
    enum class Kind { TopK, TopP, TopKTopP };
    struct Case {
        std::string name;
        Kind        kind;
        int32_t     top_k;
        float       top_p;
    };

    constexpr int64_t trials       = 8192;
    constexpr int64_t batch_size   = 1;
    constexpr int64_t vocab_size   = 16;
    constexpr bool    deterministic = true;
    auto              stream       = at::cuda::getCurrentCUDAStream().stream();

    const std::vector<float> base_probs = {
        0.23f, 0.18f, 0.14f, 0.11f, 0.09f, 0.075f, 0.055f, 0.04f,
        0.025f, 0.018f, 0.012f, 0.009f, 0.006f, 0.004f, 0.003f, 0.001f};
    auto probs = torch::tensor(base_probs, torch::TensorOptions().dtype(torch::kFloat32))
                     .reshape({batch_size, vocab_size})
                     .to(torch::kCUDA);
    probs = probs.div(probs.sum(-1, /*keepdim=*/true)).contiguous();

    const std::vector<Case> cases = {
        {"top_k", Kind::TopK, 5, 1.0f},
        {"top_p", Kind::TopP, 0, 0.72f},
        {"top_k_top_p", Kind::TopKTopP, 8, 0.65f},
    };

    auto int_options  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    for (const auto& c : cases) {
        auto top_k_h = torch::full({batch_size}, c.top_k, torch::TensorOptions().dtype(torch::kInt32)).pin_memory();
        auto top_p_h =
            torch::full({batch_size}, c.top_p, torch::TensorOptions().dtype(torch::kFloat32)).pin_memory();
        auto top_k_d = top_k_h.to(torch::kCUDA, /*non_blocking=*/true).contiguous();
        auto top_p_d = top_p_h.to(torch::kCUDA, /*non_blocking=*/true).contiguous();

        auto reference = torch::empty_like(probs);
        if (c.kind == Kind::TopK) {
            rtp_llm::top_k_renorm_probs(probs, reference, top_k_d, 0, (int64_t)stream);
        } else if (c.kind == Kind::TopP) {
            rtp_llm::top_p_renorm_probs(probs, reference, top_p_d, 1.0, (int64_t)stream);
        } else {
            auto tmp = torch::empty_like(probs);
            rtp_llm::top_k_renorm_probs(probs, tmp, top_k_d, 1.0, (int64_t)stream);
            rtp_llm::top_p_renorm_probs(tmp, reference, top_p_d, 1.0, (int64_t)stream);
        }
        cudaDeviceSynchronize();
        auto reference_host = toHostFloat(reference);

        auto old_uniforms = torch::rand({trials, 32, batch_size}, probs.options()).contiguous();
        auto old_samples  = torch::empty({trials, batch_size}, int_options);
        auto old_valid    = torch::empty({trials, batch_size}, bool_options);
        auto new_samples  = torch::empty({trials, batch_size}, int_options);
        auto new_valid    = torch::empty({trials, batch_size}, bool_options);

        for (int64_t i = 0; i < trials; ++i) {
            auto old_uniform = old_uniforms.select(0, i);
            auto old_sample  = old_samples.slice(0, i, i + 1);
            auto old_ok      = old_valid.select(0, i);
            switch (c.kind) {
                case Kind::TopK:
                    ::top_k_sampling_from_probs(
                        probs, old_uniform, old_sample, old_ok, top_k_h, 0, deterministic, (int64_t)stream);
                    break;
                case Kind::TopP:
                    ::top_p_sampling_from_probs(
                        probs, old_uniform, old_sample, old_ok, top_p_h, 1.0, deterministic, (int64_t)stream);
                    break;
                case Kind::TopKTopP:
                    ::top_k_top_p_sampling_from_probs(probs,
                                                       old_uniform,
                                                       old_sample,
                                                       old_ok,
                                                       top_k_h,
                                                       0,
                                                       top_p_h,
                                                       1.0,
                                                       deterministic,
                                                       (int64_t)stream);
                    break;
            }

            auto new_sample = new_samples.select(0, i);
            auto new_ok     = new_valid.select(0, i);
            const auto seed = static_cast<uint64_t>(20260526);
            const auto offset = static_cast<uint64_t>(i * 32);
            switch (c.kind) {
                case Kind::TopK:
                    rtp_llm::top_k_sampling_from_probs(probs,
                                                       new_sample,
                                                       new_ok,
                                                       std::nullopt,
                                                       top_k_d,
                                                       0,
                                                       deterministic,
                                                       std::nullopt,
                                                       seed,
                                                       std::nullopt,
                                                       offset,
                                                       (int64_t)stream);
                    break;
                case Kind::TopP:
                    rtp_llm::top_p_sampling_from_probs(probs,
                                                       new_sample,
                                                       new_ok,
                                                       std::nullopt,
                                                       top_p_d,
                                                       1.0,
                                                       deterministic,
                                                       std::nullopt,
                                                       seed,
                                                       std::nullopt,
                                                       offset,
                                                       (int64_t)stream);
                    break;
                case Kind::TopKTopP:
                    rtp_llm::top_k_top_p_sampling_from_probs(probs,
                                                             new_sample,
                                                             new_ok,
                                                             std::nullopt,
                                                             top_k_d,
                                                             0,
                                                             top_p_d,
                                                             1.0,
                                                             deterministic,
                                                             std::nullopt,
                                                             seed,
                                                             std::nullopt,
                                                             offset,
                                                             (int64_t)stream);
                    break;
            }
        }
        cudaDeviceSynchronize();

        auto              old_valid_cpu = old_valid.cpu().contiguous();
        auto              new_valid_cpu = new_valid.cpu().contiguous();
        std::vector<bool> old_valid_host(old_valid_cpu.data_ptr<bool>(),
                                         old_valid_cpu.data_ptr<bool>() + old_valid_cpu.numel());
        std::vector<bool> new_valid_host(new_valid_cpu.data_ptr<bool>(),
                                         new_valid_cpu.data_ptr<bool>() + new_valid_cpu.numel());
        auto old_stats = compareEmpiricalToReference(toHostInt(old_samples), old_valid_host, reference_host);
        auto new_stats = compareEmpiricalToReference(toHostInt(new_samples), new_valid_host, reference_host);

        std::cout << "[sampling-accuracy] case=" << c.name << " trials=" << trials
                  << " old_l1=" << old_stats.l1 << " old_max_abs=" << old_stats.max_abs
                  << " old_kl=" << old_stats.kl << " old_invalid=" << old_stats.invalid_count
                  << " old_support_violations=" << old_stats.support_violations << " new_l1=" << new_stats.l1
                  << " new_max_abs=" << new_stats.max_abs << " new_kl=" << new_stats.kl
                  << " new_invalid=" << new_stats.invalid_count
                  << " new_support_violations=" << new_stats.support_violations << std::endl;

        assertSamplingStatsClose(c.name + "_old", old_stats);
        assertSamplingStatsClose(c.name + "_new", new_stats);
    }
}

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
    assertTokenIn(output_token_ids_host, 17, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    assertTokenIn(output_token_ids_host, 23, {7, 8});
}

TEST_F(CudaSamplerTest, testHy4GreedyRawArgmax) {
    const auto f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto i32 = f32.dtype(torch::kInt32);
    for (bool mixed : {false, true}) {
        for (bool return_probs : {false, true}) {
            auto logits = torch::zeros({4, 8192}, f32);
            logits[0][8191].fill_(1e-8f);
            logits[1][4097].fill_(std::numeric_limits<float>::quiet_NaN());
            logits[2].fill_(-std::numeric_limits<float>::infinity());
            logits[3].normal_();
            auto         expected = torch::argmax(logits, -1, true);
            GreedyParams params{};
            params.logits           = logits.clone();
            params.input_lengths    = torch::zeros({4}, i32);
            params.sequence_lengths = torch::zeros({4}, i32);
            params.token_ids        = torch::zeros({4, 1}, i32);
            params.step             = 0;
            params.top_k            = mixed ? pinnedIntTensor({1, 1, 1, 0}) : pinnedIntTensor({1, 1, 1, 1});
            params.top_p            = pinnedFloatTensor({0.2f, 0.4f, 0.6f, 0.8f});
            params.temperature      = pinnedFloatTensor({1, 1, 1, 1});
            params.force_topk       = 4096;
            if (return_probs)
                params.output_all_probs = torch::empty_like(logits);
            for (int row = 0; row < 4; ++row) {
                params.generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
                params.generator.back().set_current_seed(row + 123);
            }
            auto      result       = execSampleGreedy(params);
            const int greedy_count = mixed ? 3 : 4;
            EXPECT_TRUE(torch::equal(params.token_ids.narrow(0, 0, greedy_count).to(torch::kInt64),
                                     expected.narrow(0, 0, greedy_count)));
            for (int row = 0; row < greedy_count; ++row)
                EXPECT_EQ(params.generator[row].get_offset(), 0);
            if (return_probs) {
                auto point_mass = torch::zeros_like(logits).scatter_(1, expected, 1.0);
                EXPECT_TRUE(torch::equal(params.output_all_probs->narrow(0, 0, greedy_count),
                                         point_mass.narrow(0, 0, greedy_count)));
            }
            if (result.success.defined()) {
                EXPECT_TRUE(result.success.all().item<bool>());
            }
        }
    }
}

TEST_F(CudaSamplerTest, testOptimizedAndFallbackGreedyAndCappedReference) {
    const auto f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto i32 = f32.dtype(torch::kInt32);
    for (bool optimized : {false, true}) {
        for (bool greedy : {true, false}) {
            if (!greedy && !optimized)
                continue;
            auto logits = greedy ? torch::zeros({2, 8192}, f32) : torch::randn({2, 8192}, f32);
            if (greedy)
                logits.select(1, 8191).fill_(1e-8f);
            GreedyParams params{};
            params.logits                 = logits.clone();
            params.input_lengths          = torch::zeros({2}, i32);
            params.sequence_lengths       = torch::zeros({2}, i32);
            params.token_ids              = torch::zeros({2, 1}, i32);
            params.step                   = 0;
            params.top_k                  = greedy ? pinnedIntTensor({1, 1}) : pinnedIntTensor({32, 0});
            params.top_p                  = pinnedFloatTensor({1, 1});
            params.temperature            = pinnedFloatTensor({1, 1});
            params.use_optimized_sampling = optimized;
            params.force_topk             = !optimized ? 0 : 4096;
            params.output_all_probs       = torch::empty_like(logits);
            for (int row = 0; row < 2; ++row) {
                params.generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
                params.generator.back().set_current_seed(123 + row);
            }
            auto result = execSampleGreedy(params);
            if (greedy) {
                auto reference_logits = !optimized ? torch::softmax(logits, -1) : logits;
                auto ids              = torch::argmax(reference_logits, -1, true);
                EXPECT_TRUE(torch::equal(params.token_ids.to(torch::kInt64), ids));
                auto expected_probs = torch::zeros_like(logits).scatter_(1, ids, 1.f);
                if (!optimized) {
                    auto maximum   = std::get<0>(reference_logits.max(-1, true));
                    auto retained  = reference_logits * reference_logits.ge(maximum);
                    expected_probs = retained / retained.sum(-1, true);
                    EXPECT_TRUE(torch::equal(expected_probs, torch::full_like(logits, 1.f / 8192)));
                }
                EXPECT_TRUE(torch::equal(*params.output_all_probs, expected_probs));
                for (const auto& generator : params.generator)
                    EXPECT_EQ(generator.get_offset() == 0, optimized);
            } else {
                auto reference = torch::zeros_like(logits);
                for (int row = 0; row < 2; ++row) {
                    auto selected = torch::topk(logits[row], row == 0 ? 32 : 4096);
                    reference[row].scatter_(0, std::get<1>(selected), torch::softmax(std::get<0>(selected), -1));
                }
                EXPECT_TRUE(torch::allclose(*params.output_all_probs, reference, 1e-5, 1e-6));
                EXPECT_TRUE(reference.gather(1, params.token_ids.to(torch::kInt64)).gt(0).all().item<bool>());
            }
            if (result.success.defined()) {
                EXPECT_TRUE(result.success.all().item<bool>());
            }
        }
    }
}

TEST_F(CudaSamplerTest, testHy4ForceTopK) {
    const auto f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto i32 = f32.dtype(torch::kInt32);
    for (int cap : {1, 32, 4096, 8192, 0}) {
        for (int mode : {0, 1, 2, 3, 4}) {
            const int batch = 4, vocab = 8192;
            auto      logits = mode == 4 ? torch::zeros({batch, vocab}, f32) : torch::randn({batch, vocab}, f32);
            auto      ks     = mode == 1 ? pinnedIntTensor({1, 7, 0, 6000}) :
                               mode == 2 ? pinnedIntTensor({7, 7, 7, 7}) :
                               mode == 4 ? pinnedIntTensor({1, 1, 1, 1}) :
                                           pinnedIntTensor({0, 0, 0, 0});
            auto      ps =
                mode == 4 ? pinnedFloatTensor({0.2f, 0.4f, 0.6f, 0.8f}) : pinnedFloatTensor({0.8f, 0.8f, 0.8f, 0.8f});
            const bool truncate         = cap > 0 && cap < vocab && mode != 2 && mode != 3 && mode != 4;
            auto       selected         = torch::topk(logits, std::max(cap, 1), -1, true, false);
            auto       reference_logits = truncate ? std::get<0>(selected) : logits;
            auto       reference_ks     = ks.clone();
            if (truncate && mode == 1) {
                auto* data = reference_ks.data_ptr<int32_t>();
                for (int i = 0; i < batch; ++i)
                    data[i] = data[i] <= 0 ? cap : std::min(data[i], cap);
            }
            auto run = [&](const torch::Tensor& input, torch::Tensor top_k, int force, bool return_probs) {
                GreedyParams params{};
                params.logits           = input.clone();
                params.input_lengths    = torch::zeros({batch}, i32);
                params.sequence_lengths = torch::zeros({batch}, i32);
                params.token_ids        = torch::zeros({batch, 1}, i32);
                params.step             = 0;
                params.top_k            = top_k.clone();
                params.top_p            = ps.clone();
                params.temperature      = pinnedFloatTensor({1, 1, 1, 1});
                params.generator.resize(batch);
                params.force_topk = force;
                if (return_probs)
                    params.output_all_probs = torch::empty_like(input);
                if (mode == 3)
                    params.cum_log_probs = torch::zeros({batch}, f32);
                torch::manual_seed(1937);
                auto result = execSampleGreedy(params);
                if (result.success.defined()) {
                    EXPECT_TRUE(result.success.all().item<bool>());
                }
                return params;
            };
            auto actual    = run(logits, ks, cap, true);
            auto reference = run(reference_logits, reference_ks, 0, true);
            auto expected_probs =
                truncate ? torch::zeros_like(logits).scatter_(1, std::get<1>(selected), *reference.output_all_probs) :
                           *reference.output_all_probs;
            auto expected_ids = truncate ? std::get<1>(selected).gather(1, reference.token_ids.to(torch::kLong)) :
                                           reference.token_ids.to(torch::kLong);
            EXPECT_TRUE(torch::allclose(*actual.output_all_probs, expected_probs, 1e-5, 1e-7));
            if (!truncate) {
                EXPECT_TRUE(torch::equal(actual.token_ids.to(torch::kLong), expected_ids));
            }
            EXPECT_TRUE(actual.output_all_probs->gather(1, actual.token_ids.to(torch::kLong)).gt(0).all().item<bool>());
            EXPECT_TRUE(torch::allclose(actual.output_all_probs->sum(-1), torch::ones({batch}, f32)));
            if (mode == 3) {
                EXPECT_TRUE(torch::allclose(*actual.cum_log_probs, *reference.cum_log_probs));
            }
            auto without_probs = run(logits, ks, cap, false);
            EXPECT_TRUE(torch::equal(actual.token_ids, without_probs.token_ids));
        }
    }
}

TEST_F(CudaSamplerTest, testHy4ForceTopKDistribution) {
    const auto f32   = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto i32   = f32.dtype(torch::kInt32);
    const int  batch = 4096, vocab = 8;
    auto       logits = torch::tensor({0.4f, 0.3f, 0.2f, 0.1f, 0.f, 0.f, 0.f, 0.f}, f32).log().repeat({batch, 1});
    for (int mode : {0, 1, 2}) {
        SCOPED_TRACE(mode);
        const int    cap = mode == 2 ? 4 : 0;
        GreedyParams params{};
        params.logits           = logits.clone();
        params.input_lengths    = torch::zeros({batch}, i32);
        params.sequence_lengths = torch::zeros({batch}, i32);
        params.token_ids        = torch::zeros({batch, 1}, i32);
        params.step             = 0;
        params.top_k            = torch::full({batch}, 2, torch::kInt32).pin_memory();
        params.top_k[0]         = 3;
        params.top_p            = torch::full({batch}, 0.5f).pin_memory();
        params.temperature      = torch::ones({batch}).pin_memory();
        params.output_all_probs = torch::empty_like(logits);
        params.generator.resize(batch);
        params.force_topk             = cap;
        params.use_optimized_sampling = mode > 0;
        execSampleGreedy(params);
        const auto zero_probability_tokens =
            params.output_all_probs->gather(1, params.token_ids.to(torch::kLong)).eq(0).sum().item<int64_t>();
        std::cout << "HY4_SAMPLING_CONTRACT cap=" << cap << " optimized=" << (mode > 0)
                  << " zero_probability_tokens=" << zero_probability_tokens << "/" << batch << std::endl;
        if (mode > 0) {
            EXPECT_EQ(zero_probability_tokens, 0);
            auto expected = torch::zeros_like(logits);
            expected.select(1, 0).fill_(1);
            expected[0][0].fill_(4.0f / 7.0f);
            expected[0][1].fill_(3.0f / 7.0f);
            EXPECT_TRUE(torch::allclose(*params.output_all_probs, expected));
        } else {
            EXPECT_GT(zero_probability_tokens, 0);
        }
    }
    for (int k : {0, 6}) {
        GreedyParams params{};
        params.logits           = torch::arange(vocab, f32).repeat({2, 1});
        params.input_lengths    = torch::zeros({2}, i32);
        params.sequence_lengths = torch::zeros({2}, i32);
        params.token_ids        = torch::zeros({2, 1}, i32);
        params.step             = 0;
        params.top_k            = pinnedIntTensor({1, k});
        params.top_p            = pinnedFloatTensor({1, 1});
        params.temperature      = pinnedFloatTensor({1, 1});
        params.output_all_probs = torch::empty_like(params.logits);
        params.generator.resize(2);
        params.force_topk = 4;
        execSampleGreedy(params);
        EXPECT_EQ(params.output_all_probs->select(0, 1).gt(0).sum().item<int64_t>(), 4);
        EXPECT_EQ(params.output_all_probs->select(0, 0).gt(0).sum().item<int64_t>(), 1);
    }
}

TEST_F(CudaSamplerTest, testHy4GreedyBenchmark) {
    const auto     f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    cudaDeviceProp device;
    ASSERT_EQ(cudaGetDeviceProperties(&device, at::cuda::getCurrentCUDAStream().device_index()), cudaSuccess);
    std::cout << "HY4_TARGET_DEVICE name=" << device.name << " sm=" << device.major << device.minor << std::endl;
    auto flush_buffer = torch::zeros({kL2FlushBytes / sizeof(float)}, f32);
    for (int batch : {1, 16, 128}) {
        for (int greedy_percent : {100, 50}) {
            if (batch == 1 && greedy_percent != 100)
                continue;
            for (bool probabilities : {false, true}) {
                auto logits = torch::randn({batch, 154880}, f32);
                auto ks     = torch::zeros({batch}, torch::kInt32).pin_memory();
                ks.narrow(0, 0, batch * greedy_percent / 100).fill_(1);
                GreedyParams params{};
                params.input_lengths    = torch::zeros({batch}, f32.dtype(torch::kInt32));
                params.sequence_lengths = torch::zeros_like(params.input_lengths);
                params.token_ids        = torch::zeros({batch, 1}, f32.dtype(torch::kInt32));
                params.step             = 0;
                params.top_p            = torch::full({batch}, 0.9f).pin_memory();
                params.temperature      = torch::ones({batch}).pin_memory();
                if (probabilities)
                    params.output_all_probs = torch::empty_like(logits);
                params.generator.resize(batch);
                for (int trial = 0; trial < 3; ++trial) {
                    for (int side = 0; side < 2; ++side) {
                        const bool argmax             = (trial + side) % 2;
                        params.use_optimized_sampling = argmax;
                        auto run                      = [&] {
                            params.logits = logits.clone();
                            params.top_k  = ks.clone();
                            execSampleGreedy(params);
                        };
                        const auto gpu_us = benchmarkCudaEventUs(run, flush_buffer, 20, 100);
                        cudaDeviceSynchronize();
                        const auto start = std::chrono::steady_clock::now();
                        for (int i = 0; i < 20; ++i)
                            run();
                        cudaDeviceSynchronize();
                        const auto wall_us =
                            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()
                            / 20;
                        std::cout << "HY4_TARGET_GREEDY batch=" << batch << " greedy_percent=" << greedy_percent
                                  << " probabilities=" << probabilities << " argmax=" << argmax << " trial=" << trial
                                  << " gpu_us=" << gpu_us << " wall_us=" << wall_us << std::endl;
                    }
                }
            }
        }
    }
}

TEST_F(CudaSamplerTest, testHy4ForceTopKAccuracyImpact) {
    const auto f32   = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const int  batch = 16, vocab = 154880;
    auto       source = torch::randn({batch, vocab}, f32);
    for (float scale : {0.f, 1.f, 4.f, 8.f}) {
        auto                       logits       = source * scale;
        auto                       selected_ids = std::get<1>(torch::topk(logits, 4096, -1));
        auto                       retained     = torch::softmax(logits, -1).gather(1, selected_ids).sum(-1);
        std::vector<torch::Tensor> distributions;
        for (int cap : {0, 4096}) {
            GreedyParams params{};
            params.logits           = logits.clone();
            params.input_lengths    = torch::zeros({batch}, f32.dtype(torch::kInt32));
            params.sequence_lengths = torch::zeros_like(params.input_lengths);
            params.token_ids        = torch::zeros({batch, 1}, f32.dtype(torch::kInt32));
            params.step             = 0;
            params.top_k            = torch::zeros({batch}, torch::kInt32).pin_memory();
            params.top_p            = torch::full({batch}, 0.9f).pin_memory();
            params.temperature      = torch::ones({batch}).pin_memory();
            params.output_all_probs = torch::empty_like(logits);
            params.generator.resize(batch);
            params.force_topk = cap;
            auto result       = execSampleGreedy(params);
            ASSERT_TRUE(result.success.all().item<bool>());
            EXPECT_TRUE(params.output_all_probs->gather(1, params.token_ids.to(torch::kLong)).gt(0).all().item<bool>());
            distributions.push_back(*params.output_all_probs);
        }
        auto tv = (distributions[0] - distributions[1]).abs().sum(-1) * 0.5;
        std::cout << "HY4_FORCE_TOPK_ACCURACY synthetic_normal_scale=" << scale
                  << " retained_mass_mean=" << retained.mean().item<float>()
                  << " retained_mass_min=" << retained.min().item<float>()
                  << " total_variation_mean=" << tv.mean().item<float>()
                  << " total_variation_max=" << tv.max().item<float>() << std::endl;
    }
}

TEST_F(CudaSamplerTest, testHy4ForceTopKBenchmark) {
    const auto f32          = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto       flush_buffer = torch::zeros({kL2FlushBytes / sizeof(float)}, f32);
    for (int batch : {1, 16, 128}) {
        auto         logits = torch::randn({batch, 154880}, f32);
        GreedyParams params{};
        params.input_lengths    = torch::zeros({batch}, f32.dtype(torch::kInt32));
        params.sequence_lengths = torch::zeros_like(params.input_lengths);
        params.token_ids        = torch::zeros({batch, 1}, f32.dtype(torch::kInt32));
        params.step             = 0;
        params.top_k            = torch::zeros({batch}, torch::kInt32).pin_memory();
        params.top_p            = torch::full({batch}, 0.9f).pin_memory();
        params.temperature      = torch::ones({batch}).pin_memory();
        params.output_all_probs = torch::empty_like(logits);
        params.generator.resize(batch);
        for (int trial = 0; trial < 3; ++trial) {
            for (int side = 0; side < 4; ++side) {
                const int mode                   = (trial + side) % 4;
                const int cap                    = mode == 0 ? 0 : 4096;
                params.force_topk = cap;
                params.skip_logits_probs_restore = mode >= 2;
                params.output_all_probs          = torch::empty({batch, mode == 3 ? 4096 : 154880}, f32);
                params.output_all_probs_indices =
                    mode == 3 ? std::optional<torch::Tensor>(torch::empty({batch, 4096}, f32.dtype(torch::kInt32))) :
                                std::nullopt;
                const auto us     = benchmarkCudaEventUs(
                    [&] {
                        params.logits = logits.clone();
                        execSampleGreedy(params);
                    },
                    flush_buffer,
                    20,
                    100);
                std::cout << "HY4_FORCE_TOPK batch=" << batch << " vocab=154880 cap=" << cap << " trial=" << trial
                          << " mode=" << mode << " us=" << us << std::endl;
            }
        }
    }
}

TEST_F(CudaSamplerTest, testHy4CompactTargetProbabilities) {
    const auto f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto i32 = f32.dtype(torch::kInt32);
    for (int mode : {0, 1, 2}) {
        auto logits = torch::randn({4, 154880}, f32);
        auto run    = [&](bool compact) {
            GreedyParams params{};
            params.logits           = logits.clone();
            params.input_lengths    = torch::zeros({4}, i32);
            params.sequence_lengths = torch::zeros({4}, i32);
            params.token_ids        = torch::zeros({4, 1}, i32);
            params.step             = 0;
            params.top_k            = mode == 1 ? pinnedIntTensor({1, 7, 0, 6000}) : pinnedIntTensor({0, 0, 0, 0});
            params.top_p =
                mode == 2 ? pinnedFloatTensor({0.9f, 1.f, 0.f, 0.8f}) : pinnedFloatTensor({0.9f, 0.9f, 0.9f, 0.9f});
            params.temperature               = pinnedFloatTensor({1, 1, 1, 1});
            params.force_topk                = 4096;
            params.skip_logits_probs_restore = compact;
            params.output_all_probs          = torch::empty({4, compact ? 4096 : 154880}, f32);
            if (compact)
                params.output_all_probs_indices = torch::empty({4, 4096}, i32);
            params.generator.resize(4);
            torch::manual_seed(2438);
            auto output = execSampleGreedy(params);
            EXPECT_TRUE(output.success.all().item<bool>());
            return params;
        };
        auto dense   = run(false);
        auto compact = run(true);
        EXPECT_TRUE(torch::equal(compact.logits, logits));
        EXPECT_TRUE(torch::equal(dense.token_ids, compact.token_ids)) << dense.token_ids << compact.token_ids;
        auto ids = compact.output_all_probs_indices->to(torch::kLong);
        ASSERT_TRUE(ids.ge(-1).logical_and(ids.lt(154880)).all().item<bool>());
        EXPECT_FALSE(compact.output_all_probs->masked_select(ids.lt(0)).ne(0).any().item<bool>());
        auto restored = torch::zeros_like(logits).scatter_add_(1, ids.clamp_min(0), *compact.output_all_probs);
        EXPECT_TRUE(torch::allclose(restored, *dense.output_all_probs, 1e-5, 1e-7))
            << "max_diff=" << (restored - *dense.output_all_probs).abs().max().item<float>()
            << "l1=" << (restored - *dense.output_all_probs).abs().sum(-1);
        EXPECT_TRUE(torch::allclose(compact.output_all_probs->sum(-1), torch::ones({4}, f32)));
    }
}

TEST_F(CudaSamplerTest, testDeepSelectFallbackPreservesTopK) {
    const auto f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int mode = 0; mode < 4; ++mode) {
        SCOPED_TRACE(mode);
        auto      logits = torch::randn({4, mode == 3 ? 8193 : 8192}, f32);
        const int k      = mode == 0 ? 32 : 4096;
        if (mode == 1)
            logits = logits.to(torch::kFloat16);
        if (mode == 2)
            logits = logits.transpose(0, 1).contiguous().transpose(0, 1);
        bool used          = true;
        auto [values, ids] = torch_ext::deepselect_fp32_topk(logits, k, &used);
        EXPECT_FALSE(used);
        EXPECT_TRUE(torch::equal(values, logits.gather(1, ids)));
        EXPECT_TRUE(torch::equal(std::get<0>(values.sort(-1, true)), std::get<0>(torch::topk(logits, k, -1))));
    }
}

TEST_F(CudaSamplerTest, testHy4DeepSelectAccuracyAndBenchmark) {
    const auto f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int width : {8192, 154880}) {
        for (int pattern : {0, 1, 2, 3, 4, 5}) {
            auto logits = torch::randn({16, width}, f32);
            if (pattern == 1)
                logits.zero_();
            if (pattern == 2 || pattern == 3) {
                logits.fill_(-std::numeric_limits<float>::infinity());
                if (pattern == 2)
                    logits.narrow(1, width - 8, 8).normal_();
            }
            if (pattern == 4)
                logits.select(1, width - 1).fill_(std::numeric_limits<float>::infinity());
            if (pattern == 5)
                logits.select(1, width - 1).fill_(std::numeric_limits<float>::quiet_NaN());
            bool used          = false;
            auto [values, ids] = torch_ext::deepselect_fp32_topk(logits, 4096, &used);
            ASSERT_TRUE(used);
            ASSERT_TRUE(ids.ge(0).logical_and(ids.lt(width)).all().item<bool>());
            auto ordered_ids = std::get<0>(ids.sort(-1));
            EXPECT_TRUE(ordered_ids.slice(1, 1).gt(ordered_ids.slice(1, 0, 4095)).all().item<bool>());
            EXPECT_TRUE(torch::allclose(values, logits.gather(1, ids), 0, 0, true));
            auto reference = std::get<0>(torch::topk(logits, 4096, -1));
            EXPECT_TRUE(torch::allclose(std::get<0>(values.sort(-1, true)), reference, 0, 0, true));
        }
    }
    auto                       stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto                       input = torch::randn({16, 154880}, f32);
    torch_ext::deepselect_fp32_topk(input, 4096);
    ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    auto [graph_values, graph_ids] = torch_ext::deepselect_fp32_topk(input, 4096);
    graph.capture_end();
    for (int replay = 0; replay < 3; ++replay) {
        input.normal_();
        graph.replay();
        EXPECT_TRUE(torch::equal(graph_values, input.gather(1, graph_ids)));
        EXPECT_TRUE(torch::equal(std::get<0>(graph_values.sort(-1, true)), std::get<0>(torch::topk(input, 4096, -1))));
    }
    auto flush = torch::zeros({kL2FlushBytes / sizeof(float)}, f32);
    for (int batch : {1, 16, 128}) {
        auto logits = torch::randn({batch, 154880}, f32);
        for (int trial = 0; trial < 3; ++trial) {
            for (int side = 0; side < 2; ++side) {
                const bool deepselect = (trial + side) % 2;
                const auto us         = benchmarkCudaEventUs(
                    [&] {
                        if (deepselect) {
                            bool used = false;
                            torch_ext::deepselect_fp32_topk(logits, 4096, &used);
                            EXPECT_TRUE(used);
                        } else {
                            torch::topk(logits, 4096, -1, true, false);
                        }
                    },
                    flush,
                    20,
                    100);
                std::cout << "HY4_SELECTOR batch=" << batch << " deepselect=" << deepselect << " trial=" << trial
                          << " us=" << us << std::endl;
            }
        }
    }
}

TEST_F(CudaSamplerTest, testHy4GreedyAccuracyImpact) {
    const auto f32   = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto i32   = f32.dtype(torch::kInt32);
    const int  batch = 128, vocab = 154880;
    for (int pattern : {0, 1, 2}) {
        auto logits = torch::randn({batch, vocab}, f32);
        if (pattern != 0) {
            logits.zero_();
            if (pattern == 1)
                logits.select(1, vocab - 1).fill_(1e-8f);
        }
        GreedyParams params{};
        params.logits           = logits.clone();
        params.input_lengths    = torch::zeros({batch}, i32);
        params.sequence_lengths = torch::zeros({batch}, i32);
        params.token_ids        = torch::zeros({batch, 1}, i32);
        params.step             = 0;
        params.top_k            = torch::ones({batch}, torch::kInt32).pin_memory();
        params.top_p            = torch::ones({batch}).pin_memory();
        params.temperature      = torch::ones({batch}).pin_memory();
        params.output_all_probs = torch::empty_like(logits);
        execSampleGreedy(params);
        auto actual    = params.token_ids.to(torch::kLong);
        auto reference = torch::argmax(logits, -1, true);
        auto old       = torch::argmax(torch::softmax(logits, -1), -1, true);
        EXPECT_TRUE(torch::equal(actual, reference));
        std::cout << "HY4_TARGET_ACCURACY pattern=" << pattern << " rows=" << batch
                  << " raw_mismatches=" << actual.ne(reference).sum().item<int64_t>()
                  << " softmax_mismatches=" << actual.ne(old).sum().item<int64_t>() << std::endl;
        if (pattern == 0) {
            EXPECT_TRUE(torch::equal(actual, old));
            params.logits = logits.clone();
            params.top_k.index_put_({batch - 1}, 0);
            params.generator.resize(1);
            auto output = execSampleGreedy(params);
            EXPECT_TRUE(output.success.all().item<bool>());
            EXPECT_TRUE(params.output_all_probs->gather(1, params.token_ids.to(torch::kLong)).gt(0).all().item<bool>());
        }
    }
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
    assertTokenIn(output_token_ids_host, 17, {7, 8, 5, 0, 4, 3});
    assertTokenIn(output_token_ids_host, 23, {7, 8, 5, 0, 4, 3, 9, 1});
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
    assertTokenIn(output_token_ids_host, 11, {2, 8});
    assertTokenIn(output_token_ids_host, 17, {7, 8, 5, 0, 4, 3});
    assertTokenIn(output_token_ids_host, 23, {7, 8});
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
                         output_all_probs_t,
                         presence_penalty_t,
                         frequency_penalty_t,
                         nullopt,
                         generator});
    execSampleGreedy(params);
    check_cuda_error();

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    assertTokenIn(output_token_ids_host, 5, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    assertTokenIn(output_token_ids_host, 11, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    assertTokenIn(output_token_ids_host, 17, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    assertTokenIn(output_token_ids_host, 23, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

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
                         output_all_probs_t,
                         nullopt,
                         nullopt,
                         do_sample_t,
                         generator});
    execSampleGreedy(params);
    check_cuda_error();

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    assertTokenIn(output_token_ids_host, 5, {1, 2});
    assertTokenIn(output_token_ids_host, 11, {1, 2});
    assertTokenIn(output_token_ids_host, 17, {1, 2});
    assertTokenIn(output_token_ids_host, 23, {1, 2});

    auto output_all_probs_host = toHostFloat(output_all_probs_t);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.477515, 0.522485, 0, 0, 0, 0, 0, 0, 0,
                            0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.488752, 0.511248, 0, 0, 0, 0, 0, 0, 0}),
        1e-3);
    auto cum_log_probs_host = toHostFloat(cum_log_probs_t);
    assertCumLogProbMatches(cum_log_probs_host,
                            output_all_probs_host,
                            output_token_ids_host,
                            {-1.0, -2.0, -3.0, -3.0},
                            batch_size,
                            step,
                            10);
}
