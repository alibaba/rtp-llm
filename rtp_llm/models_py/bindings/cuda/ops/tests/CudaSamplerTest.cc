#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "3rdparty/flashinfer/flashinfer.h"
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

constexpr int64_t kL2FlushBytes = 512LL * 1024 * 1024;

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
    double  l1                 = 0.0;
    double  max_abs            = 0.0;
    double  kl                 = 0.0;
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
    void assertBatchTemperatureBeyondBlock(torch::ScalarType dtype) {
        constexpr int64_t vocab = 32;  // The kernel launches only 32 threads per block.
        for (const int64_t batch : {80, 128}) {
            SCOPED_TRACE(::testing::Message() << "batch=" << batch << " dtype=" << dtype);
            std::vector<float> logits_data(batch * vocab), temperature_data(batch), inverse_data(batch);
            for (int64_t row = 0; row < batch; ++row) {
                temperature_data[row] = 0.5f + static_cast<float>(row % 17) / 10.0f;
                inverse_data[row]     = 1.0f / (temperature_data[row] + 1e-6f);
                for (int64_t col = 0; col < vocab; ++col) {
                    logits_data[row * vocab + col] = static_cast<float>(1 + (row * 7 + col * 3) % 53) / 8.0f;
                }
            }
            auto input_cpu = torch::tensor(logits_data, torch::kFloat32).reshape({batch, vocab}).to(dtype);
            // The even-vocabulary FP16 path rounds the reciprocal to half before
            // multiplying in half2. Match both roundings, not float division.
            auto       inverse_cpu  = torch::tensor(inverse_data, torch::kFloat32).to(dtype).to(torch::kFloat32);
            auto       expected     = (input_cpu.to(torch::kFloat32) * inverse_cpu.unsqueeze(1)).to(dtype);
            auto       logits       = input_cpu.cuda();
            auto       temperatures = cudaTensor(temperature_data, {batch});
            const auto stream       = at::cuda::getCurrentCUDAStream().stream();
            if (dtype == torch::kFloat32) {
                invokeBatchApplyTemperaturePenalty<float>(
                    logits.data_ptr<float>(), nullptr, temperatures.data_ptr<float>(), batch, vocab, vocab, stream);
            } else {
                invokeBatchApplyTemperaturePenalty<half>(reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
                                                         nullptr,
                                                         temperatures.data_ptr<float>(),
                                                         batch,
                                                         vocab,
                                                         vocab,
                                                         stream);
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            auto actual = logits.cpu();
            ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
            if (dtype == torch::kFloat16) {
                ASSERT_TRUE(torch::equal(actual, expected));
            } else {
                ASSERT_TRUE(torch::allclose(actual, expected, /*rtol=*/1e-6, /*atol=*/1e-6));
            }
        }
    }

    // This is a lowest-layer equivalence check, not permission to omit history
    // for penalties, processors, beam search, or arbitrary Sampler consumers.
    void assertCompactHistoryEquivalent(int64_t batch,
                                        int64_t history,
                                        int     mode,
                                        bool    with_all_probs      = true,
                                        int64_t vocab               = 32,
                                        bool    exact_probabilities = true) {
        SCOPED_TRACE(::testing::Message() << "batch=" << batch << " history=" << history << " mode=" << mode
                                          << " all_probs=" << with_all_probs << " vocab=" << vocab);
        ASSERT_TRUE(with_all_probs || mode == 0);
        std::vector<float>   logits_data(batch * vocab);
        std::vector<int32_t> input_data(batch), sequence_data(batch), top_k_data(batch);
        std::vector<float>   top_p_data(batch), temperature_data(batch);
        for (int64_t row = 0; row < batch; ++row) {
            input_data[row]       = static_cast<int32_t>(history - 4 - row % 3);
            sequence_data[row]    = static_cast<int32_t>(history - row % 3);
            top_k_data[row]       = mode == 0 ? 1 : (row % 2 ? 8 : 0);
            top_p_data[row]       = mode == 0 ? 1.0f : (row % 2 ? 0.85f : 0.7f);
            temperature_data[row] = mode == 0 ? 1.0f : (row % 2 ? 0.8f : 1.2f);
            if (mode == 2 && row % 2 == 0) {
                // Match fillSamplerCommonInputs' do_sample=false normalization.
                top_k_data[row]       = 1;
                top_p_data[row]       = 1.0f;
                temperature_data[row] = 1.0f;
            }
            for (int64_t col = 0; col < vocab; ++col) {
                logits_data[row * vocab + col] = static_cast<float>((col * 7 + row * 3) % 31) / 9.0f;
            }
        }
        auto logits           = cudaTensor(logits_data, {batch, vocab});
        auto input_lengths    = torch::tensor(input_data, torch::kInt32);
        auto sequence_lengths = torch::tensor(sequence_data, torch::kInt32);
        auto input_before     = input_lengths.clone();
        auto sequence_before  = sequence_lengths.clone();
        auto do_sample        = torch::ones({batch}, torch::kBool).pin_memory();
        for (int64_t row = 0; row < batch; ++row) {
            do_sample.data_ptr<bool>()[row] = mode != 2 || row % 2 != 0;
        }
        std::vector<at::Generator> full_generators, compact_generators;
        for (int64_t row = 0; row < batch; ++row) {
            full_generators.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
            compact_generators.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
            full_generators.back().set_current_seed(9100 + row);
            compact_generators.back().set_current_seed(9100 + row);
        }
        // Two successive calls verify continuation, not just an equal first draw.
        for (int round = 0; round < 2; ++round) {
            SCOPED_TRACE(::testing::Message() << "round=" << round);
            auto device_i32     = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            auto full_tokens    = torch::full({batch, history + 1}, 7, device_i32);
            auto compact_tokens = torch::full({batch, 1}, -1, device_i32);
            full_tokens.select(1, history).fill_(-1);
            auto                       full_logits    = logits.clone();
            auto                       compact_logits = logits.clone();
            auto                       full_probs     = torch::zeros_like(logits);
            auto                       compact_probs  = torch::zeros_like(logits);
            auto                       full_cum       = torch::full({batch}, -0.5f, logits.options());
            auto                       compact_cum    = full_cum.clone();
            std::vector<torch::Tensor> before_rng;
            for (int64_t row = 0; row < batch; ++row) {
                before_rng.push_back(full_generators[row].get_state().clone());
            }

            auto make_params = [&](torch::Tensor                     l,
                                   torch::Tensor                     tokens,
                                   torch::Tensor                     probs,
                                   torch::Tensor                     cum,
                                   const std::vector<at::Generator>& generators) {
                return GreedyParams{l,
                                    input_lengths,
                                    sequence_lengths,
                                    tokens,
                                    static_cast<size_t>(history),
                                    pinnedIntTensor(top_k_data),
                                    pinnedFloatTensor(top_p_data),
                                    pinnedFloatTensor(temperature_data),
                                    pinnedFloatTensor(std::vector<float>(batch, 1.0f)),
                                    pinnedIntTensor(std::vector<int32_t>(batch, 0)),
                                    cum,
                                    std::nullopt,
                                    false,
                                    with_all_probs ? std::optional<torch::Tensor>(probs) : std::nullopt,
                                    pinnedFloatTensor(std::vector<float>(batch, 0.0f)),
                                    pinnedFloatTensor(std::vector<float>(batch, 0.0f)),
                                    do_sample,
                                    generators};
            };
            auto full_params = make_params(full_logits, full_tokens, full_probs, full_cum, full_generators);
            auto compact_params =
                make_params(compact_logits, compact_tokens, compact_probs, compact_cum, compact_generators);
            ASSERT_EQ(full_params.output_all_probs.has_value(), with_all_probs);
            ASSERT_EQ(compact_params.output_all_probs.has_value(), with_all_probs);
            // Logical step and lengths intentionally remain real, including 81920.
            ASSERT_EQ(full_params.step, compact_params.step);
            auto full_output    = execSampleGreedy(full_params);
            auto compact_output = execSampleGreedy(compact_params);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_TRUE(torch::equal(full_tokens.select(1, history), compact_tokens.select(1, 0)));
            ASSERT_TRUE(compact_tokens.ge(0).logical_and(compact_tokens.lt(vocab)).all().item<bool>());
            if (exact_probabilities) {
                ASSERT_TRUE(torch::equal(full_probs, compact_probs));
                ASSERT_TRUE(torch::equal(full_cum, compact_cum));
            } else {
                // Real-vocab multi-CTA renorm is not bitwise repeatable even
                // full/full with frozen input. Stage-isolation tests retain
                // that evidence. Bound distribution drift per row, not only
                // per-element absolute error (which hides broad mass shifts).
                ASSERT_TRUE(torch::isfinite(full_probs).all().item<bool>());
                ASSERT_TRUE(torch::isfinite(compact_probs).all().item<bool>());
                ASSERT_TRUE(torch::equal(full_probs.gt(0), compact_probs.gt(0)));
                ASSERT_LE((full_probs - compact_probs).abs().sum(-1).max().item<float>(), 1e-6f);
                ASSERT_LE((full_probs.sum(-1) - 1).abs().max().item<float>(), 1e-6f);
                ASSERT_LE((compact_probs.sum(-1) - 1).abs().max().item<float>(), 1e-6f);
                ASSERT_LE((full_cum - compact_cum).abs().max().item<float>(), 2e-6f);
            }
            ASSERT_TRUE(torch::equal(full_logits, compact_logits));
            ASSERT_EQ(full_output.success.defined(), compact_output.success.defined());
            if (full_output.success.defined()) {
                ASSERT_TRUE(torch::equal(full_output.success, compact_output.success));
                ASSERT_TRUE(full_output.success.all().item<bool>());
            }
            if (!with_all_probs) {
                // top1's no-all_probs fast path is argmax only: no softmax or RNG draw.
                ASSERT_FALSE(full_output.success.defined());
                ASSERT_FALSE(compact_output.success.defined());
                ASSERT_TRUE(torch::equal(compact_tokens.select(1, 0), logits.argmax(-1).to(torch::kInt32)));
                ASSERT_TRUE(torch::equal(full_logits, logits));
                ASSERT_TRUE(full_cum.eq(-0.5f).all().item<bool>());
            }
            ASSERT_TRUE(full_tokens.narrow(1, 0, history).eq(7).all().item<bool>());
            ASSERT_TRUE(torch::equal(input_lengths, input_before));
            ASSERT_TRUE(torch::equal(sequence_lengths, sequence_before));
            for (int64_t row = 0; row < batch; ++row) {
                ASSERT_TRUE(torch::equal(full_generators[row].get_state(), compact_generators[row].get_state()));
                if (!with_all_probs) {
                    ASSERT_TRUE(torch::equal(before_rng[row], full_generators[row].get_state()));
                    ASSERT_TRUE(torch::equal(before_rng[row], compact_generators[row].get_state()));
                }
            }
            // Even top1 with all_probs must retain the current RNG reservation.
            if (with_all_probs) {
                ASSERT_FALSE(torch::equal(before_rng.front(), full_generators.front().get_state()));
            }
        }
    }

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

TEST_F(CudaSamplerTest, DISABLED_benchmarkLatestFlashinferSamplingVsCurrentRtp) {
    enum class Kind {
        TopK,
        TopP,
        TopKTopP
    };
    struct Case {
        std::string name;
        Kind        kind;
        int64_t     batch_size;
        int64_t     vocab_size;
        int32_t     top_k;
        float       top_p;
    };

    constexpr int64_t          vocab_size  = 129280;
    const std::vector<int64_t> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 96, 128};
    std::vector<Case>          cases;
    cases.reserve(batch_sizes.size() * 3);
    for (auto batch_size : batch_sizes) {
        const auto suffix = "_b" + std::to_string(batch_size) + "_v" + std::to_string(vocab_size);
        cases.push_back({"top_k" + suffix, Kind::TopK, batch_size, vocab_size, 50, 1.0f});
        cases.push_back({"top_p" + suffix, Kind::TopP, batch_size, vocab_size, 0, 0.85f});
        cases.push_back({"top_k_top_p" + suffix, Kind::TopKTopP, batch_size, vocab_size, 100, 0.9f});
    }

    constexpr bool deterministic   = true;
    constexpr int  warmup          = 20;
    constexpr int  iterations      = 100;
    auto           stream          = at::cuda::getCurrentCUDAStream().stream();
    auto           l2_flush_buffer = torch::empty({kL2FlushBytes / static_cast<int64_t>(sizeof(float))},
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
        auto top_p_h = torch::full({c.batch_size}, c.top_p, torch::TensorOptions().dtype(torch::kFloat32)).pin_memory();
        auto top_k_d = top_k_h.to(torch::kCUDA, /*non_blocking=*/true).contiguous();
        auto top_p_d = top_p_h.to(torch::kCUDA, /*non_blocking=*/true).contiguous();
        auto seed_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        auto seed_d       = torch::full({c.batch_size}, 20260525, seed_options);
        auto offset_d     = torch::arange(0, c.batch_size * 32, 32, seed_options);

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
        double latest_us   = benchmarkCudaEventUs(latest_kernel, l2_flush_buffer, warmup, iterations);

        std::cout << "[sampling-benchmark] case=" << c.name << " batch=" << c.batch_size << " vocab=" << c.vocab_size
                  << " old_kernel_us=" << old_kernel_us << " old_uniform_rng_us=" << old_uniform_rng_us
                  << " old_with_uniform_rng_us=" << old_full_us << " latest_us=" << latest_us
                  << " speedup_vs_old_kernel=" << (old_kernel_us / latest_us)
                  << " speedup_vs_old_with_rng=" << (old_full_us / latest_us) << std::endl;
    }
}

TEST_F(CudaSamplerTest, DISABLED_compareLatestFlashinferSamplingAccuracyVsCurrentRtp) {
    enum class Kind {
        TopK,
        TopP,
        TopKTopP
    };
    struct Case {
        std::string name;
        Kind        kind;
        int32_t     top_k;
        float       top_p;
    };

    constexpr int64_t trials        = 8192;
    constexpr int64_t batch_size    = 1;
    constexpr int64_t vocab_size    = 16;
    constexpr bool    deterministic = true;
    auto              stream        = at::cuda::getCurrentCUDAStream().stream();

    const std::vector<float> base_probs = {0.23f,
                                           0.18f,
                                           0.14f,
                                           0.11f,
                                           0.09f,
                                           0.075f,
                                           0.055f,
                                           0.04f,
                                           0.025f,
                                           0.018f,
                                           0.012f,
                                           0.009f,
                                           0.006f,
                                           0.004f,
                                           0.003f,
                                           0.001f};
    auto                     probs      = torch::tensor(base_probs, torch::TensorOptions().dtype(torch::kFloat32))
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
        auto top_p_h = torch::full({batch_size}, c.top_p, torch::TensorOptions().dtype(torch::kFloat32)).pin_memory();
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

            auto       new_sample = new_samples.select(0, i);
            auto       new_ok     = new_valid.select(0, i);
            const auto seed       = static_cast<uint64_t>(20260526);
            const auto offset     = static_cast<uint64_t>(i * 32);
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

        std::cout << "[sampling-accuracy] case=" << c.name << " trials=" << trials << " old_l1=" << old_stats.l1
                  << " old_max_abs=" << old_stats.max_abs << " old_kl=" << old_stats.kl
                  << " old_invalid=" << old_stats.invalid_count
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
    assertTokenIn(output_token_ids_host, 17, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    assertTokenIn(output_token_ids_host, 23, {7, 8});
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

TEST_F(CudaSamplerTest, testSeededTopPIsInvariantToBatchPosition) {
    constexpr int64_t vocab_size = 100;
    constexpr int64_t step       = 1;

    auto run = [&](int64_t batch_size) {
        auto cuda_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto cuda_int   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto pinned_int = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
        auto pinned_f32 = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);

        auto x           = torch::arange(vocab_size, cuda_float);
        auto logits_row  = torch::sin(x * 0.017) + torch::cos(x * 0.003);
        auto logits      = logits_row.repeat({batch_size, 1});
        auto token_ids   = torch::zeros({batch_size, step + 1}, cuda_int);
        auto lengths     = torch::full({batch_size}, step, cuda_int);
        auto top_k       = torch::zeros({batch_size}, pinned_int);
        auto top_p       = torch::full({batch_size}, 0.9, pinned_f32);
        auto temperature = torch::ones({batch_size}, pinned_f32);

        std::vector<at::Generator> generators;
        generators.reserve(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
            auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
            generator.set_current_seed(314159);
            generators.push_back(generator);
        }

        GreedyParams params({logits,
                             lengths,
                             lengths,
                             token_ids,
                             step,
                             top_k,
                             top_p,
                             temperature,
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
        execSampleGreedy(params);
        check_cuda_error();
        return token_ids.select(1, step).cpu();
    };

    auto single  = run(1);
    auto batched = run(8);
    ASSERT_TRUE(batched.eq(single.item<int32_t>()).all().item<bool>())
        << "per-request seeded sampling changed with dynamic batch row";
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

TEST_F(CudaSamplerTest, BatchTemperaturePenaltyFloatBatchExceedsBlock) {
    assertBatchTemperatureBeyondBlock(torch::kFloat32);
}

TEST_F(CudaSamplerTest, BatchTemperaturePenaltyHalfBatchExceedsBlock) {
    assertBatchTemperatureBeyondBlock(torch::kFloat16);
}

TEST_F(CudaSamplerTest, CompactHistoryTop1AllProbsLong81920B80PreservesRng) {
    assertCompactHistoryEquivalent(80, 81920, 0);
}

TEST_F(CudaSamplerTest, CompactHistoryTop1NoAllProbsLong81920B80LeavesRngUnchanged) {
    assertCompactHistoryEquivalent(80, 81920, 0, /*with_all_probs=*/false);
}

TEST_F(CudaSamplerTest, CompactHistoryRandomTopKTopPPreservesRngAcrossShapes) {
    assertCompactHistoryEquivalent(80, 81920, 1);
    assertCompactHistoryEquivalent(5, 17, 1);
    assertCompactHistoryEquivalent(128, 31, 1);
}

TEST_F(CudaSamplerTest, CompactHistoryMixedDoSamplePreservesRngAcrossShapes) {
    assertCompactHistoryEquivalent(80, 81920, 2);
    assertCompactHistoryEquivalent(1, 9, 2);
    assertCompactHistoryEquivalent(80, 23, 2);
}

TEST_F(CudaSamplerTest, CompactHistoryRealVocab200064Top1AllProbsLong81920B80PreservesRng) {
    assertCompactHistoryEquivalent(80,
                                   81920,
                                   0,
                                   /*with_all_probs=*/true,
                                   /*vocab=*/200064,
                                   /*exact_probabilities=*/false);
}

TEST_F(CudaSamplerTest, CompactHistoryRealVocab200064Top1NoAllProbsLong81920B80LeavesRngUnchanged) {
    assertCompactHistoryEquivalent(80, 81920, 0, /*with_all_probs=*/false, /*vocab=*/200064);
}

TEST_F(CudaSamplerTest, CompactHistoryRealVocab200064RandomTopKTopPLong81920B80PreservesRng) {
    assertCompactHistoryEquivalent(80,
                                   81920,
                                   1,
                                   /*with_all_probs=*/true,
                                   /*vocab=*/200064,
                                   /*exact_probabilities=*/false);
}

TEST_F(CudaSamplerTest, CompactHistoryRealVocab200064MixedDoSampleLong81920B80PreservesRng) {
    assertCompactHistoryEquivalent(80,
                                   81920,
                                   2,
                                   /*with_all_probs=*/true,
                                   /*vocab=*/200064,
                                   /*exact_probabilities=*/false);
}

// Opt-in bitwise diagnostic: intentionally fails for multi-CTA FP32 renorm,
// including full/full controls. Kept to reproduce the numerical-bound evidence;
// enabled regressions above enforce exact tokens/logits/RNG and bounded mass.
TEST_F(CudaSamplerTest, DISABLED_DiagnosticCompactHistoryFullFullCompactSampling) {
    struct Case {
        int64_t batch;
        int64_t history;
        int     mode;
        bool    hold_seed_inputs = false;
        int64_t vocab            = 32;
    };
    const std::vector<Case> cases   = {{80, 81920, 1},
                                       {128, 31, 1},
                                       {80, 31, 1},
                                       {80, 81920, 2},
                                       {80, 23, 2},
                                       {80, 31, 3},
                                       {80, 81920, 1, true},
                                       {128, 31, 1, true},
                                       {80, 31, 1, true},
                                       {80, 81920, 2, true},
                                       {80, 23, 2, true},
                                       {80, 31, 3, true},
                                       {80, 81920, 0, true, 200064},
                                       {80, 81920, 1, true, 200064},
                                       {80, 81920, 2, true, 200064}};
    constexpr int           repeats = 5;
    for (const auto& c : cases) {
        const int64_t vocab = c.vocab;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            SCOPED_TRACE(::testing::Message()
                         << "diagnostic batch=" << c.batch << " history=" << c.history << " mode=" << c.mode
                         << " repeat=" << repeat << " hold_seed_inputs=" << c.hold_seed_inputs);
            std::cout << "COMPACT_DIAGNOSTIC batch=" << c.batch << " history=" << c.history << " mode=" << c.mode
                      << " repeat=" << repeat << " vocab=" << vocab << " first_call=" << repeat % 3
                      << " hold_seed_inputs=" << c.hold_seed_inputs << std::endl;
            std::vector<float>   data(c.batch * vocab), top_p(c.batch), temperature(c.batch);
            std::vector<int32_t> input(c.batch), sequence(c.batch), top_k(c.batch);
            for (int64_t row = 0; row < c.batch; ++row) {
                input[row]       = static_cast<int32_t>(c.history - 4 - row % 3);
                sequence[row]    = static_cast<int32_t>(c.history - row % 3);
                top_k[row]       = row % 2 ? 8 : 0;
                top_p[row]       = row % 2 ? 0.85f : 0.7f;
                temperature[row] = row % 2 ? 0.8f : 1.2f;
                if (c.mode == 0) {
                    top_k[row]       = 1;
                    top_p[row]       = 1.0f;
                    temperature[row] = 1.0f;
                }
                if (c.mode == 2 && row % 2 == 0) {
                    top_k[row]       = 1;
                    top_p[row]       = 1.0f;
                    temperature[row] = 1.0f;
                }
                // Dedicated all-zero top_k control exercises no-limit top-p,
                // distinct from mixed {0,8} rows in the combined top-k/top-p branch.
                if (c.mode == 3) {
                    top_k[row] = 0;
                }
                for (int64_t col = 0; col < vocab; ++col) {
                    data[row * vocab + col] = static_cast<float>((col * 7 + row * 3) % 31) / 9.0f;
                }
            }
            std::vector<GreedyParams> params;
            std::vector<GreedyOutput> outputs(3);
            // Caller-owned pinned inputs in params remain alive through sync in
            // both controls. Holders additionally retain internally created
            // seed/offset host tensors, matching the production ownership path.
            std::vector<TensorHolder> holders(3);
            for (int variant = 0; variant < 3; ++variant) {
                auto logits = cudaTensor(data, {c.batch, vocab});
                auto tokens = torch::full({c.batch, variant == 2 ? 1 : c.history + 1},
                                          7,
                                          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
                tokens.select(1, tokens.size(1) - 1).fill_(-1);
                auto                       do_sample = torch::ones({c.batch}, torch::kBool).pin_memory();
                std::vector<at::Generator> generators;
                for (int64_t row = 0; row < c.batch; ++row) {
                    do_sample.data_ptr<bool>()[row] = c.mode != 2 || row % 2 != 0;
                    generators.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
                    generators.back().set_current_seed(9100 + row);
                }
                params.push_back(GreedyParams{logits,
                                              torch::tensor(input, torch::kInt32),
                                              torch::tensor(sequence, torch::kInt32),
                                              tokens,
                                              static_cast<size_t>(c.history),
                                              pinnedIntTensor(top_k),
                                              pinnedFloatTensor(top_p),
                                              pinnedFloatTensor(temperature),
                                              pinnedFloatTensor(std::vector<float>(c.batch, 1.0f)),
                                              pinnedIntTensor(std::vector<int32_t>(c.batch, 0)),
                                              torch::full({c.batch}, -0.5f, logits.options()),
                                              std::nullopt,
                                              false,
                                              torch::zeros_like(logits),
                                              pinnedFloatTensor(std::vector<float>(c.batch, 0.0f)),
                                              pinnedFloatTensor(std::vector<float>(c.batch, 0.0f)),
                                              do_sample,
                                              generators,
                                              c.hold_seed_inputs ? &holders[variant] : nullptr});
            }
            for (int variant = 1; variant < 3; ++variant) {
                for (int64_t row = 0; row < c.batch; ++row) {
                    EXPECT_TRUE(
                        torch::equal(params[0].generator[row].get_state(), params[variant].generator[row].get_state()))
                        << "initial RNG";
                }
            }
            // Rotate submission order; retain all three independent inputs/outputs.
            for (int call = 0; call < 3; ++call) {
                const int variant = (call + repeat) % 3;
                outputs[variant]  = execSampleGreedy(params[variant]);
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            for (int variant = 0; variant < 3; ++variant) {
                std::cout << "COMPACT_DIAGNOSTIC variant=" << variant
                          << " held_host_tensors=" << holders[variant].tensors.size()
                          << " top_k_zero_after=" << params[variant].top_k.eq(0).sum().item<int64_t>()
                          << " top_k_no_limit_after=" << params[variant].top_k.eq(1 << 30).sum().item<int64_t>()
                          << std::endl;
            }
            auto report = [&](const char* pair, const char* field, const torch::Tensor& a, const torch::Tensor& b) {
                std::cout << "COMPACT_DIAGNOSTIC pair=" << pair << " field=" << field << " defined_a=" << a.defined()
                          << " defined_b=" << b.defined();
                if (!a.defined() || !b.defined()) {
                    std::cout << std::endl;
                    EXPECT_EQ(a.defined(), b.defined()) << pair << "/" << field;
                    return;
                }
                auto       x      = a.cpu().to(torch::kFloat64);
                auto       y      = b.cpu().to(torch::kFloat64);
                auto       finite = torch::isfinite(x).logical_and(torch::isfinite(y));
                auto       delta  = (x - y).abs().masked_select(finite);
                const bool equal  = torch::equal(a, b);
                std::cout << " exact=" << equal << " different=" << x.ne(y).sum().item<int64_t>()
                          << " finite_a=" << torch::isfinite(x).sum().item<int64_t>()
                          << " finite_b=" << torch::isfinite(y).sum().item<int64_t>()
                          << " nan_a=" << torch::isnan(x).sum().item<int64_t>()
                          << " nan_b=" << torch::isnan(y).sum().item<int64_t>()
                          << " inf_a=" << torch::isinf(x).sum().item<int64_t>()
                          << " inf_b=" << torch::isinf(y).sum().item<int64_t>()
                          << " finite_max_abs=" << (delta.numel() ? delta.max().item<double>() : 0.0)
                          << " finite_L1=" << delta.sum().item<double>() << std::endl;
                // Nonfatal: report every field and every repetition before failing.
                EXPECT_TRUE(equal) << pair << "/" << field;
            };
            for (int other = 1; other < 3; ++other) {
                const char* pair = other == 1 ? "fullA/fullB" : "fullA/compact";
                report(pair,
                       "tokens",
                       params[0].token_ids.select(1, c.history),
                       params[other].token_ids.select(1, params[other].token_ids.size(1) - 1));
                report(
                    pair, "probabilities", params[0].output_all_probs.value(), params[other].output_all_probs.value());
                report(pair, "logits", params[0].logits, params[other].logits);
                report(pair, "cum_log_probs", params[0].cum_log_probs.value(), params[other].cum_log_probs.value());
                report(pair, "success", outputs[0].success, outputs[other].success);
                int64_t rng_different_rows = 0;
                for (int64_t row = 0; row < c.batch; ++row) {
                    rng_different_rows +=
                        !torch::equal(params[0].generator[row].get_state(), params[other].generator[row].get_state());
                }
                std::cout << "COMPACT_DIAGNOSTIC pair=" << pair
                          << " field=post_rng different_rows=" << rng_different_rows << " total_rows=" << c.batch
                          << std::endl;
                EXPECT_EQ(rng_different_rows, 0) << pair;
            }
        }
    }
}

TEST_F(CudaSamplerTest, DiagnosticRealVocabSoftmaxAndRenormRepeatability) {
    constexpr int64_t batch = 80, vocab = 200064;
    auto              rows         = torch::arange(batch, torch::kInt64).unsqueeze(1);
    auto              cols         = torch::arange(vocab, torch::kInt64).unsqueeze(0);
    auto              logits       = ((cols * 7 + rows * 3).remainder(31).to(torch::kFloat32) / 9.0f).cuda();
    auto              frozen_probs = torch::softmax(logits, -1);
    const auto        stream       = at::cuda::getCurrentCUDAStream().stream();
    for (int64_t top_k : {int64_t{1}, int64_t{8}, int64_t{1 << 30}}) {
        auto reference = torch::empty_like(frozen_probs);
        top_k_renorm_probs(frozen_probs, reference, std::nullopt, top_k, (int64_t)stream);
        for (int repeat = 0; repeat < 5; ++repeat) {
            auto repeated_softmax = torch::softmax(logits, -1);
            ASSERT_TRUE(torch::equal(frozen_probs, repeated_softmax));
            auto actual = torch::empty_like(frozen_probs);
            top_k_renorm_probs(frozen_probs, actual, std::nullopt, top_k, (int64_t)stream);
            auto delta = (reference - actual).abs();
            std::cout << "RENORM_DIAGNOSTIC top_k=" << top_k << " repeat=" << repeat
                      << " softmax_exact=1 renorm_exact=" << torch::equal(reference, actual)
                      << " max_abs=" << delta.max().item<float>() << " max_row_l1=" << delta.sum(-1).max().item<float>()
                      << " row_sum_error=" << (actual.sum(-1) - 1).abs().max().item<float>() << std::endl;
            ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
            ASSERT_TRUE(actual.ge(0).all().item<bool>());
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
                         false,
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
