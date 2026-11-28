#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

namespace rtp_llm {
namespace {

constexpr int64_t kModelVocabSize   = 248320;
constexpr int32_t kGrammarVocabSize = 248046;
constexpr int64_t kDecodeSeqLen     = 2048;
constexpr int     kWarmupIters      = 4;
constexpr int     kBenchIters       = 16;

using Clock = std::chrono::steady_clock;

std::shared_ptr<XGrammarBackend> makeLargeBackend() {
    std::vector<std::string> vocab;
    vocab.reserve(kGrammarVocabSize);
    for (int32_t i = 0; i < kGrammarVocabSize; ++i) {
        if (i < 128) {
            vocab.emplace_back(1, static_cast<char>(i));
        } else {
            vocab.emplace_back("<t" + std::to_string(i) + ">");
        }
    }
    xgrammar::TokenizerInfo tok_info(
        vocab, xgrammar::VocabType::RAW, kGrammarVocabSize, std::vector<int32_t>{0});
    GrammarConfig cfg;
    cfg.num_workers                  = 2;
    cfg.compiler_cache_bytes         = -1;
    cfg.terminate_without_stop_token = true;
    return XGrammarBackend::create(tok_info.SerializeJSON(), cfg);
}

struct LatencyStats {
    double mean_us = 0.0;
    double p50_us  = 0.0;
    double p90_us  = 0.0;
};

LatencyStats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    auto pct = [&samples](double q) {
        const size_t rank = static_cast<size_t>(std::ceil(q * samples.size()));
        return samples[std::min(samples.size() - 1, std::max<size_t>(1, rank) - 1)];
    };
    LatencyStats out;
    out.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    out.p50_us  = pct(0.50);
    out.p90_us  = pct(0.90);
    return out;
}

std::vector<std::shared_ptr<GrammarLogitsProcessor>> makeMixedProcessors(
    const std::shared_ptr<XGrammarBackend>& backend,
    int64_t                                 batch_size) {
    const int64_t active_count      = (batch_size * 90 + 64) / 128;
    const int64_t unconstrained_cnt = (batch_size * 26 + 64) / 128;
    const int64_t terminated_count  = std::max<int64_t>(0, batch_size - active_count - unconstrained_cnt);

    std::vector<std::shared_ptr<GrammarLogitsProcessor>> procs;
    procs.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
        if (i < active_count) {
            auto matcher = backend->createMatcherFromKey({"regex", "[0-9]+a"});
            EXPECT_TRUE(matcher.ok()) << matcher.status().ToString();
            procs.push_back(std::make_shared<GrammarLogitsProcessor>(matcher.value(), /*eos_token_id=*/0));
        } else if (i < active_count + unconstrained_cnt) {
            auto matcher = backend->createMatcherFromKey(
                {"structural_tag", R"({"type":"structural_tag","format":{"type":"any_text"}})"});
            EXPECT_TRUE(matcher.ok()) << matcher.status().ToString();
            procs.push_back(std::make_shared<GrammarLogitsProcessor>(matcher.value(), /*eos_token_id=*/0));
        } else if (i < active_count + unconstrained_cnt + terminated_count) {
            auto matcher = backend->createMatcherFromKey({"regex", "a"});
            EXPECT_TRUE(matcher.ok()) << matcher.status().ToString();
            auto proc = std::make_shared<GrammarLogitsProcessor>(matcher.value(), /*eos_token_id=*/0);
            auto tok  = torch::tensor({static_cast<int32_t>('a')}, torch::kInt32).reshape({1, 1});
            EXPECT_FALSE(proc->updateStatus(tok, 1).has_value());
            procs.push_back(std::move(proc));
        }
    }
    return procs;
}

template<typename StepFn>
void runTimedLoop(const char* name, int64_t batch_size, StepFn&& step_fn) {
    for (int i = 0; i < kWarmupIters; ++i) {
        step_fn(i, /*advance_grammar=*/false);
    }
    cudaDeviceSynchronize();

    std::vector<double> cpu_us;
    cpu_us.reserve(kBenchIters);
    std::cerr << "[r15-counter-begin] " << name << " B=" << batch_size << " iters=" << kBenchIters << "\n";
    cudaProfilerStart();
    for (int i = 0; i < kBenchIters; ++i) {
        const auto t0 = Clock::now();
        step_fn(kWarmupIters + i, /*advance_grammar=*/true);
        const auto t1 = Clock::now();
        cpu_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    cudaProfilerStop();
    cudaDeviceSynchronize();

    const auto t_e2e_start = Clock::now();
    for (int i = 0; i < kBenchIters; ++i) {
        step_fn(kWarmupIters + kBenchIters + i, /*advance_grammar=*/true);
    }
    cudaDeviceSynchronize();
    const auto t_e2e_end = Clock::now();
    const double e2e_per_step_us =
        std::chrono::duration<double, std::micro>(t_e2e_end - t_e2e_start).count() / kBenchIters;

    const auto stats = summarize(std::move(cpu_us));
    std::cout << std::fixed << std::setprecision(1) << "[r15-perf] " << name << " B=" << batch_size
              << " cpu_mean_us=" << stats.mean_us << " cpu_p50_us=" << stats.p50_us
              << " cpu_p90_us=" << stats.p90_us << " e2e_mean_us=" << e2e_per_step_us << std::endl;
}

TEST(GrammarAndSamplerPerfTest, DISABLED_DecodeB16B64B128) {
    auto backend = makeLargeBackend();
    ASSERT_TRUE(backend);

    const auto cuda_f32    = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto pinned_i32  = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    const auto pinned_f32  = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
    const auto pinned_bool = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);

    constexpr int kTotalIters = kWarmupIters + 2 * kBenchIters;

    for (int64_t batch_size : {int64_t{16}, int64_t{64}, int64_t{128}}) {
        const int64_t active_count = (batch_size * 90 + 64) / 128;

        // 1. GrammarLogitsProcessor + LogitsProcessorStates::batchProcess
        {
            auto procs  = makeMixedProcessors(backend, batch_size);
            auto states = std::make_shared<LogitsProcessorStates>();
            for (int64_t i = 0; i < batch_size; ++i) {
                states->insert(procs[i], static_cast<size_t>(i), static_cast<size_t>(i + 1));
            }
            auto logits    = torch::zeros({batch_size, kModelVocabSize}, cuda_f32);
            auto tok_digit = torch::tensor({static_cast<int32_t>('5')}, torch::kInt32).reshape({1, 1});
            SamplerInputs inputs{};
            inputs.logits        = logits;
            inputs.vocab_size    = static_cast<size_t>(kModelVocabSize);
            inputs.batch_size    = static_cast<size_t>(batch_size);
            inputs.batch_size_out = static_cast<size_t>(batch_size);
            inputs.finished_mask = torch::zeros({batch_size}, pinned_bool);

            runTimedLoop("grammar_batch_process", batch_size, [&](int iter, bool advance) {
                (void)iter;
                if (advance) {
                    for (int64_t i = 0; i < active_count; ++i) {
                        ASSERT_FALSE(procs[i]->updateStatus(tok_digit, 1).has_value());
                    }
                }
                auto errs = states->batchProcess(inputs);
                for (const auto& e : errs) {
                    ASSERT_FALSE(e.has_value());
                }
            });
        }

        // 2. Sampler::forward (top_k=20, top_p=0.95, temperature=0.7, with mixed grammar)
        {
            Sampler sampler(SamplerInitParams{static_cast<size_t>(batch_size), false});
            auto    procs  = makeMixedProcessors(backend, batch_size);
            auto    states = std::make_shared<LogitsProcessorStates>();
            for (int64_t i = 0; i < batch_size; ++i) {
                states->insert(procs[i], static_cast<size_t>(i), static_cast<size_t>(i + 1));
            }
            auto                       logits_seed = torch::randn({batch_size, kModelVocabSize}, cuda_f32);
            std::vector<torch::Tensor> logits_pool;
            logits_pool.reserve(kTotalIters);
            for (int i = 0; i < kTotalIters; ++i) {
                logits_pool.push_back(logits_seed.clone());
            }
            cudaDeviceSynchronize();
            auto tok_digit = torch::tensor({static_cast<int32_t>('5')}, torch::kInt32).reshape({1, 1});

            SamplerInputs inputs{};
            inputs.logits                      = logits_pool[0];
            inputs.token_ids                   = torch::zeros({batch_size, kDecodeSeqLen + 1}, pinned_i32);
            inputs.input_lengths               = torch::full({batch_size}, kDecodeSeqLen, pinned_i32);
            inputs.sequence_lengths            = torch::full({batch_size}, kDecodeSeqLen, pinned_i32);
            inputs.logits_processor_states_ptr = states;
            inputs.vocab_size                  = kModelVocabSize;
            inputs.step                        = kDecodeSeqLen;
            inputs.batch_size                  = static_cast<size_t>(batch_size);
            inputs.batch_size_out              = static_cast<size_t>(batch_size);
            inputs.num_beams_in                = torch::ones({batch_size}, torch::kLong);
            inputs.num_beams_out               = torch::ones({batch_size}, torch::kLong);
            inputs.top_k                       = torch::full({batch_size}, 20, pinned_i32);
            inputs.top_p                       = torch::full({batch_size}, 0.95f, pinned_f32);
            inputs.temperature                 = torch::full({batch_size}, 0.7f, pinned_f32);
            inputs.do_sample                   = torch::ones({batch_size}, pinned_bool);
            inputs.finished_mask               = torch::zeros({batch_size}, pinned_bool);
            for (int64_t i = 0; i < batch_size; ++i) {
                auto gen = torch::make_generator<at::CUDAGeneratorImpl>();
                gen.set_current_seed(1000 + i);
                inputs.generator.push_back(std::move(gen));
            }

            runTimedLoop("sampler_plus_grammar_pinned_hist", batch_size, [&](int iter, bool advance) {
                inputs.logits = logits_pool[static_cast<size_t>(iter)];
                if (advance) {
                    for (int64_t i = 0; i < active_count; ++i) {
                        ASSERT_FALSE(procs[i]->updateStatus(tok_digit, 1).has_value());
                    }
                }
                auto out = sampler.forward(inputs);
                (void)out;
            });

            // Re-seed logits_pool for pure Sampler::forward with no grammar and [B, 1] CUDA token_ids
            for (int i = 0; i < kTotalIters; ++i) {
                logits_pool[static_cast<size_t>(i)].copy_(logits_seed);
            }
            cudaDeviceSynchronize();
            inputs.logits_processor_states_ptr = std::make_shared<LogitsProcessorStates>();
            inputs.token_ids                   = torch::empty({batch_size, 1}, cuda_i32);
            runTimedLoop("sampler_only_cuda_col", batch_size, [&](int iter, bool) {
                inputs.logits = logits_pool[static_cast<size_t>(iter)];
                auto out      = sampler.forward(inputs);
                (void)out;
            });
        }
    }
}

}  // namespace
}  // namespace rtp_llm
