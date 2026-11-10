#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>
#ifdef FAIR_GUMBEL_BRANCH
#include "pybind11/embed.h"
#endif

using namespace rtp_llm;
namespace {
class FairSamplingBench: public DeviceTestBase {};
TEST_F(FairSamplingBench, TargetThroughFinalOutput) {
    const bool    trace_shape = std::getenv("FAIR_TRACE_SHAPE") != nullptr;
    const int64_t vocab       = trace_shape ? 129280 : 154880;
    const int64_t steps = 5, rows = steps + 1, cap = 0;
    const float   top_p = 1.0f;
#ifdef FAIR_GUMBEL_BRANCH
    static pybind11::scoped_interpreter interpreter{};
    pybind11::exec(R"(
import os, sys
from pathlib import Path
r = Path(os.environ['TEST_SRCDIR'])
sys.path[:0] = [str(r), str(r / os.environ['TEST_WORKSPACE'])] + [str(p) for p in sorted(r.glob('pip*/site-packages'))]
import torch
)");
    setenv("RTP_LLM_USE_VLLM_REJECTION_SAMPLING", "1", 1);
    initializeSamplingRuntime(vocab);
    ASSERT_TRUE(supportsDeferredRejectionSampling());
    const std::string branch = "gumbel";
#else
    const std::string branch = "reference";
#endif
    const auto        fp        = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto        ip        = fp.dtype(torch::kInt32);
    const auto        hp        = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    const auto        hf        = hp.dtype(torch::kFloat32);
    const auto        stream    = at::cuda::getCurrentCUDAStream().stream();
    const std::string directory = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    std::ofstream     report(directory + "/fair_results.jsonl");
    const auto        batches = std::getenv("FAIR_NEARBY_BATCHES") ? std::vector<int64_t>{62, 63, 64} :
                                trace_shape                        ? std::vector<int64_t>{64} :
                                                                     std::vector<int64_t>{16, 41};
    for (int64_t batch : batches) {
        for (double probability : {0.0, 0.7, 1.0}) {
            std::vector<float> values(vocab);
            double             mass = 0;
            for (int64_t v = 1; v < vocab; ++v) {
                values[v] = -static_cast<float>(v) / static_cast<float>(vocab);
                if (cap == 0 || v < cap) {
                    mass += std::exp(static_cast<double>(values[v]));
                }
            }
            values[0] =
                probability == 0 ? -80.0f :
                probability == 1 ? 80.0f :
                                   static_cast<float>(std::log(mass * probability * top_p / (1 - probability * top_p)));
            auto pristine = torch::tensor(values).to(fp).unsqueeze(0).expand({batch * rows, vocab}).contiguous();
            auto logits   = pristine.clone();
            auto draft    = torch::zeros({batch, steps}, ip);
            auto tokens   = torch::zeros({batch * rows, 2}, ip);
            auto output   = torch::empty({batch, rows}, ip);
            auto lengths  = torch::empty({batch}, ip);
            auto sample   = torch::ones({batch}, fp.dtype(torch::kBool));
            auto uniform  = torch::empty({batch, rows}, fp);
            GreedyParams target;
            target.logits                    = logits;
            target.input_lengths             = torch::full({batch * rows}, 1, hp);
            target.sequence_lengths          = torch::full({batch * rows}, 1, hp);
            target.token_ids                 = tokens;
            target.step                      = 1;
            target.top_k                     = torch::zeros({batch * rows}, hp);
            target.top_p                     = torch::full({batch * rows}, top_p, hf);
            target.temperature               = torch::ones({batch * rows}, hf);
            target.skip_logits_probs_restore = true;
            for (int64_t i = 0; i < batch * rows; ++i) {
                auto g = torch::make_generator<at::CUDAGeneratorImpl>();
                g.set_current_seed(20261110 + i);
                target.generator.push_back(g);
            }
            RejectionSamplingParams reject;
            reject.draft_token_ids_d           = draft;
            reject.output_token_ids_d          = output;
            reject.output_accepted_token_num_d = lengths;
            reject.do_sample_d                 = sample;
#ifdef FAIR_GUMBEL_BRANCH
            // Deferred filtering does not use force_topk; express the same cap
            // through its explicit top_k input, including the filter in timing.
            if (cap) {
                target.top_k.fill_(cap);
            }
            target.defer_token_sampling = true;
            TensorHolder holder;
            target.buffer_holder           = &holder;
            reject.buffer_holder           = &holder;
            reject.target_logits_d         = logits.reshape({batch, rows, vocab});
            reject.deferred_token_sampling = true;
            reject.draft_point_mass        = true;
            for (int64_t i = 0; i < batch; ++i) {
                auto g = torch::make_generator<at::CUDAGeneratorImpl>();
                g.set_current_seed(20261110 + i);
                reject.generator.push_back(g);
            }
#else
            auto probs         = torch::empty({batch * rows, cap ? cap : vocab}, fp);
            auto candidate_ids = torch::empty({batch * rows, cap ? cap : vocab}, ip);
            if (cap) {
                target.force_topk               = cap;
                target.output_all_probs_indices = candidate_ids;
            }
            target.output_all_probs = probs;
            GreedySamplingBuffers buffers;
            const auto            hi      = hp.dtype(torch::kInt64);
            buffers.seed_host             = torch::empty({batch * rows}, hi);
            buffers.offset_host           = torch::empty({batch * rows}, hi);
            buffers.output_ids_ptrs_host  = torch::empty({batch * rows}, hi);
            buffers.seed_device           = torch::empty({batch * rows}, ip.dtype(torch::kInt64));
            buffers.offset_device         = torch::empty({batch * rows}, ip.dtype(torch::kInt64));
            buffers.max_batch_size        = batch * rows;
            target.sampling_buffers       = &buffers;
            reject.target_probs_d         = probs.reshape({batch, rows, cap ? cap : vocab});
            reject.target_token_ids_d     = tokens;
            reject.uniform_samples_d      = uniform;
            reject.draft_probs_point_mass = cap == 0;
            if (cap) {
                // Explicit singleton q is the same greedy draft distribution.
                reject.draft_probs_d          = torch::ones({batch, steps, 1}, fp);
                reject.draft_probs_indices_d  = torch::zeros({batch, steps, 1}, ip);
                reject.target_probs_indices_d = candidate_ids.reshape({batch, rows, cap});
            }
            reject.finalize_outputs = true;
#endif
            auto run = [&]() {
                {
                    at::RecordFunction scope(at::RecordScope::USER_SCOPE);
                    scope.before("fair.target_prepare_and_sample");
                    auto sampled = execSampleGreedy(target);
#ifdef FAIR_GUMBEL_BRANCH
                    TORCH_CHECK(sampled.deferred_token_sampling);
#else
                    (void)sampled;
#endif
                }
                {
                    at::RecordFunction scope(at::RecordScope::USER_SCOPE);
                    scope.before("fair.reject_and_writeback");
#ifndef FAIR_GUMBEL_BRANCH
                    uniform.uniform_(0, 1);
#endif
                    execRejectionSampling(reject);
                }
            };
            auto prepare = [&]() {
                logits.copy_(pristine);
#ifdef FAIR_GUMBEL_BRANCH
                holder.release();
#endif
            };
            for (int i = 0; i < 10; ++i) {
                prepare();
                run();
                cudaStreamSynchronize(stream);
            }
            std::vector<double> times;
            double              accepted = 0, reached = 0, emitted = 0;
            for (int repeat = 0; repeat < 5; ++repeat) {
                double sum = 0;
                for (int i = 0; i < 50; ++i) {
                    prepare();
                    cudaEvent_t start, end;
                    cudaEventCreate(&start);
                    cudaEventCreate(&end);
                    cudaEventRecord(start, stream);
                    run();
                    cudaEventRecord(end, stream);
                    cudaEventSynchronize(end);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, start, end);
                    sum += ms * 1000;
                    cudaEventDestroy(start);
                    cudaEventDestroy(end);
                    auto counts = lengths.cpu();
                    for (int64_t b = 0; b < batch; ++b) {
                        int n = counts.data_ptr<int32_t>()[b];
                        ASSERT_GE(n, 1);
                        ASSERT_LE(n, rows);
                        if (probability == 0) {
                            ASSERT_EQ(n, 1);
                        }
                        if (probability == 1) {
                            ASSERT_EQ(n, rows);
                        }
                        accepted += n - 1;
                        reached += std::min<int>(n, steps);
                        emitted += n;
                    }
                }
                times.push_back(sum / 50);
            }
            auto cpu_out = output.cpu();
            auto cpu_len = lengths.cpu();
            for (int64_t b = 0; b < batch; ++b) {
                int n = cpu_len.data_ptr<int32_t>()[b];
                for (int64_t j = 0; j < rows; ++j) {
                    auto token = cpu_out.data_ptr<int32_t>()[b * rows + j];
                    if (j >= n) {
                        ASSERT_EQ(token, 0);
                    } else {
                        ASSERT_GE(token, 0);
                        ASSERT_LT(token, vocab);
                    }
                    if (j < n - 1) {
                        ASSERT_EQ(token, 0);
                    }
                }
            }
            std::sort(times.begin(), times.end());
            std::ostringstream json;
            json << "{\"branch\":\"" << branch << "\",\"batch\":" << batch << ",\"steps\":5,\"vocab\":" << vocab
                 << ",\"candidate_cap\":" << cap << ",\"top_p\":" << top_p
                 << ",\"target_draft_probability\":" << probability << ",\"median_us\":" << times[2]
                 << ",\"acceptance_among_reached\":" << accepted / reached
                 << ",\"mean_emitted\":" << emitted / (250 * batch) << ",\"sorted_repeat_us\":[";
            for (size_t i = 0; i < times.size(); ++i)
                json << (i ? "," : "") << times[i];
            json << "]}";
            report << json.str() << std::endl;
            std::cout << "FAIR_SAMPLING " << json.str() << std::endl;
            namespace tap = torch::autograd::profiler;
            namespace tpi = torch::profiler::impl;
            tpi::ProfilerConfig               config(tpi::ProfilerState::KINETO, true);
            const std::set<tpi::ActivityType> activities{tpi::ActivityType::CPU, tpi::ActivityType::CUDA};
            tap::prepareProfiler(config, activities);
            tap::enableProfiler(config, activities);
            for (int i = 0; i < 5; ++i) {
                prepare();
                cudaStreamSynchronize(stream);
                {
                    at::RecordFunction scope(at::RecordScope::USER_SCOPE);
                    scope.before("fair.full_sampling");
                    run();
                }
                cudaStreamSynchronize(stream);
            }
            auto trace = tap::disableProfiler();
            trace->save(directory + "/fair_b" + std::to_string(batch) + "_p" + std::to_string(int(probability * 100))
                        + ".json");
        }
    }
}
}  // namespace
