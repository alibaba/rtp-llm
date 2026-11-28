#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/softmax_host.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/temperature_host.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/tests/AirTopPBenchHelper.h"

using namespace rtp_llm;

namespace {

template<typename Fn>
double measureCudaUs(cudaStream_t stream, int warmup, int iters, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (static_cast<double>(ms) * 1000.0) / iters;
}

}  // namespace

class SamplingBenchmarkTest: public DeviceTestBase {};

TEST_F(SamplingBenchmarkTest, testSubKernelAndEndToEndDecodeBenchmarks) {
    auto       stream   = at::cuda::getCurrentCUDAStream().stream();
    const auto f32_dev  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i32_dev  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto bool_dev = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    const auto i64_dev  = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    constexpr int64_t vocab   = 129280;
    constexpr int64_t seq_len = 256;
    constexpr int     warmup  = 5;
    constexpr int     iters   = 30;

    torch::manual_seed(20261128);
    const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    std::printf("\n=== SamplingBenchmarkTest Device: %s (SM %d.%d), V=%ld, seq_len=%ld ===\n",
                prop->name,
                prop->major,
                prop->minor,
                static_cast<long>(vocab),
                static_cast<long>(seq_len));

    std::printf("\n[1] Per-Subkernel Breakdown (BEFORE vs AFTER, us)\n");
    std::printf("%-4s | %-15s | %-15s | %-15s | %-15s | %-15s | %-15s\n",
                "B",
                "Temp (B->A)",
                "Smax+Cp (B->A)",
                "Penalty (B->A)",
                "TopP (B->A)",
                "JointRnSm(B->A)",
                "PureSm (B->A)");
    std::printf(
        "-----+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------\n");

    double b80_before_joint_pen_us = 0.0;
    double b80_after_joint_pen_us  = 0.0;
    double b80_e2e_after_us        = 0.0;

    for (int64_t batch : {24, 48, 80, 96}) {
        auto base_logits = torch::randn({batch, vocab}, f32_dev) * 2.0f;
        auto work_logits = base_logits.clone();
        auto renorm_out  = torch::empty_like(base_logits);
        auto probs_fixed = torch::softmax(base_logits, -1);

        auto temps_dev  = torch::full({batch}, 0.7f, f32_dev);
        auto rep_dev    = torch::full({batch}, 1.15f, f32_dev);
        auto pres_dev   = torch::full({batch}, 0.1f, f32_dev);
        auto freq_dev   = torch::full({batch}, 0.05f, f32_dev);
        auto top_p_dev  = torch::full({batch}, 0.95f, f32_dev);
        auto top_k_dev  = torch::full({batch}, 20, i32_dev);
        auto in_len_dev = torch::full({batch}, seq_len / 2, i32_dev);
        auto tokens_dev = torch::randint(0, vocab, {batch, seq_len + 1}, i32_dev);
        auto persist_ws = torch::zeros({batch * vocab}, i32_dev);
        auto air_ws     = torch::empty({static_cast<int64_t>(airTopPWorkspaceBytes(batch, vocab))},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

        auto sampled_id  = torch::empty({batch}, i32_dev);
        auto valid_out   = torch::empty({batch}, bool_dev);
        auto seeds_dev   = torch::full({batch}, 777LL, i64_dev);
        auto offsets_dev = torch::arange(0, batch * 32LL, 32LL, i64_dev);

        const double temp_before = measureCudaUs(stream, warmup, iters, [&]() {
            invokeBatchApplyTemperaturePenalty<float>(
                work_logits.data_ptr<float>(), nullptr, temps_dev.data_ptr<float>(), batch, vocab, vocab, stream);
        });
        work_logits.copy_(base_logits);
        const double temp_after =
            measureCudaUs(stream, warmup, iters, [&]() { applySamplerTemperature(work_logits, temps_dev, stream); });

        const double smax_before = measureCudaUs(stream, warmup, iters, [&]() {
            auto p = torch::softmax(base_logits, -1);
            work_logits.copy_(p, true);
        });
        work_logits.copy_(base_logits);
        const double copy_only_us = measureCudaUs(stream, warmup, iters, [&]() { work_logits.copy_(base_logits); });
        const double smax_after   = measureCudaUs(stream, warmup, iters, [&]() {
            work_logits.copy_(base_logits);
            samplerSoftmaxInplace(work_logits);
        });
        // Subtract base_logits DtoD copy inside smax_after loop to report pure splitSoftmax kernel time cleanly:
        const double smax_after_pure = std::max(0.0, smax_after - copy_only_us);

        const double pen_before = measureCudaUs(stream, warmup, iters, [&]() {
            auto tokens_t = tokens_dev.slice(1, 0, seq_len).transpose(0, 1).contiguous();
            auto ws       = torch::zeros({batch, vocab}, i32_dev);
            invokeBatchApplyRepetitionPenalty<float>(work_logits.data_ptr<float>(),
                                                     ws.data_ptr<int32_t>(),
                                                     rep_dev.data_ptr<float>(),
                                                     pres_dev.data_ptr<float>(),
                                                     freq_dev.data_ptr<float>(),
                                                     tokens_t.data_ptr<int32_t>(),
                                                     batch,
                                                     batch,
                                                     vocab,
                                                     in_len_dev.data_ptr<int32_t>(),
                                                     seq_len,
                                                     seq_len,
                                                     stream);
        });
        work_logits.copy_(base_logits);
        const double pen_after = measureCudaUs(stream, warmup, iters, [&]() {
            invokeBatchApplyRepetitionPenaltyRowMajorSparse<float>(work_logits.data_ptr<float>(),
                                                                   persist_ws.data_ptr<int32_t>(),
                                                                   rep_dev.data_ptr<float>(),
                                                                   pres_dev.data_ptr<float>(),
                                                                   freq_dev.data_ptr<float>(),
                                                                   tokens_dev.data_ptr<int32_t>(),
                                                                   batch,
                                                                   seq_len + 1,
                                                                   vocab,
                                                                   in_len_dev.data_ptr<int32_t>(),
                                                                   seq_len,
                                                                   seq_len,
                                                                   stream);
        });

        const double topp_before = measureCudaUs(stream, warmup, iters, [&]() {
            invokeAirTopPRenormBench(probs_fixed.data_ptr<float>(),
                                     renorm_out.data_ptr<float>(),
                                     top_p_dev.data_ptr<float>(),
                                     batch,
                                     1.0f,
                                     vocab,
                                     air_ws.data_ptr(),
                                     stream);
        });
        const double topp_after  = measureCudaUs(stream, warmup, iters, [&]() {
            top_p_renorm_probs(probs_fixed, renorm_out, top_p_dev, 1.0, reinterpret_cast<int64_t>(stream));
        });

        const double joint_rnsm_before = measureCudaUs(stream, warmup, iters, [&]() {
            invokeFlashInferTopKTopPSamplingBench(probs_fixed.data_ptr<float>(),
                                                  top_k_dev.data_ptr<int32_t>(),
                                                  top_p_dev.data_ptr<float>(),
                                                  sampled_id.data_ptr<int32_t>(),
                                                  valid_out.data_ptr<bool>(),
                                                  batch,
                                                  vocab,
                                                  reinterpret_cast<uint64_t*>(seeds_dev.data_ptr<int64_t>()),
                                                  reinterpret_cast<uint64_t*>(offsets_dev.data_ptr<int64_t>()),
                                                  stream);
            auto z = torch::zeros_like(renorm_out);
            invokeAirTopPRenormBench(probs_fixed.data_ptr<float>(),
                                     z.data_ptr<float>(),
                                     top_p_dev.data_ptr<float>(),
                                     batch,
                                     1.0f,
                                     vocab,
                                     air_ws.data_ptr(),
                                     stream);
            top_k_renorm_probs(z, z, top_k_dev, 0, reinterpret_cast<int64_t>(stream));
        });
        const double joint_rnsm_after  = measureCudaUs(stream, warmup, iters, [&]() {
            top_k_top_p_renorm_and_sample_probs(probs_fixed,
                                                renorm_out,
                                                sampled_id,
                                                valid_out,
                                                std::nullopt,
                                                top_k_dev,
                                                0,
                                                top_p_dev,
                                                1.0,
                                                true,
                                                seeds_dev,
                                                0,
                                                offsets_dev,
                                                0,
                                                reinterpret_cast<int64_t>(stream));
        });

        const double pure_sm_before = measureCudaUs(stream, warmup, iters, [&]() {
            invokeFlashInferSamplingFromProbBench(probs_fixed.data_ptr<float>(),
                                                  sampled_id.data_ptr<int32_t>(),
                                                  valid_out.data_ptr<bool>(),
                                                  batch,
                                                  vocab,
                                                  reinterpret_cast<uint64_t*>(seeds_dev.data_ptr<int64_t>()),
                                                  reinterpret_cast<uint64_t*>(offsets_dev.data_ptr<int64_t>()),
                                                  stream);
        });
        const double pure_sm_after  = measureCudaUs(stream, warmup, iters, [&]() {
            sampling_from_probs(probs_fixed,
                                sampled_id,
                                valid_out,
                                std::nullopt,
                                true,
                                seeds_dev,
                                0,
                                offsets_dev,
                                0,
                                reinterpret_cast<int64_t>(stream));
        });

        std::printf(
            "%-4ld | %6.1f -> %5.1f | %6.1f -> %5.1f | %6.1f -> %5.1f | %6.1f -> %5.1f | %6.1f -> %5.1f | %6.1f -> %5.1f\n",
            static_cast<long>(batch),
            temp_before,
            temp_after,
            smax_before,
            smax_after_pure,
            pen_before,
            pen_after,
            topp_before,
            topp_after,
            joint_rnsm_before,
            joint_rnsm_after,
            pure_sm_before,
            pure_sm_after);

        if (batch == 80) {
            b80_before_joint_pen_us = temp_before + smax_before + pen_before + joint_rnsm_before;
            b80_after_joint_pen_us  = temp_after + smax_after_pure + pen_after + joint_rnsm_after;
        }
    }

    std::printf(
        "\n[2] End-to-End execSampleGreedy Matrix (with persistent GreedySamplingBuffers, V=%ld, seq_len=%ld)\n",
        static_cast<long>(vocab),
        static_cast<long>(seq_len));
    std::printf("%-4s | %-4s | %-4s | %-3s | %-9s | %-9s\n", "B", "topP", "topK", "Pen", "E2E(noAll)", "E2E(allP)");
    std::printf("-----+------+------+-----+-----------+----------\n");

    GreedySamplingBuffers sampling_buffers;
    sampling_buffers.max_batch_size       = 128;
    auto pin_i64                          = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    sampling_buffers.seed_host            = torch::empty({128}, pin_i64);
    sampling_buffers.offset_host          = torch::empty({128}, pin_i64);
    sampling_buffers.output_ids_ptrs_host = torch::empty({128}, pin_i64);

    for (int64_t batch : {24, 48, 80, 96}) {
        auto pin_i32 = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
        auto pin_f32 = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);

        auto base_logits   = torch::randn({batch, vocab}, f32_dev) * 2.0f;
        auto work_logits   = torch::empty_like(base_logits);
        auto all_probs_out = torch::empty_like(base_logits);
        auto token_ids_dev = torch::randint(0, vocab, {batch, seq_len + 1}, i32_dev);
        auto in_lens_pin   = torch::full({batch}, seq_len / 2, pin_i32);
        auto seq_lens_pin  = torch::full({batch}, seq_len, pin_i32);

        std::vector<at::Generator> gens;
        gens.reserve(batch);
        for (int64_t i = 0; i < batch; ++i) {
            auto g = torch::make_generator<at::CUDAGeneratorImpl>();
            g.set_current_seed(1000 + i);
            gens.push_back(g);
        }

        const double restore_us = measureCudaUs(stream, warmup, iters, [&]() { work_logits.copy_(base_logits); });

        for (float top_p_val : {0.95f, 1.0f}) {
            for (int top_k_val : {0, 20, 50}) {
                for (bool pen_on : {false, true}) {
                    auto top_k_h = torch::full({batch}, top_k_val, pin_i32);
                    auto top_p_h = torch::full({batch}, top_p_val, pin_f32);
                    auto temp_h  = torch::full({batch}, 0.7f, pin_f32);
                    auto rep_h   = torch::full({batch}, pen_on ? 1.15f : 1.0f, pin_f32);
                    auto pres_h  = torch::full({batch}, pen_on ? 0.1f : 0.0f, pin_f32);
                    auto freq_h  = torch::full({batch}, pen_on ? 0.05f : 0.0f, pin_f32);

                    const double no_all_total = measureCudaUs(stream, warmup, iters, [&]() {
                        work_logits.copy_(base_logits);
                        GreedyParams p{work_logits,
                                       in_lens_pin,
                                       seq_lens_pin,
                                       token_ids_dev,
                                       static_cast<size_t>(seq_len),
                                       top_k_h,
                                       top_p_h,
                                       temp_h,
                                       rep_h,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       false,
                                       std::nullopt,
                                       pres_h,
                                       freq_h,
                                       std::nullopt,
                                       gens,
                                       &sampling_buffers};
                        execSampleGreedy(p);
                    });
                    const double all_total    = measureCudaUs(stream, warmup, iters, [&]() {
                        work_logits.copy_(base_logits);
                        GreedyParams p{work_logits,
                                       in_lens_pin,
                                       seq_lens_pin,
                                       token_ids_dev,
                                       static_cast<size_t>(seq_len),
                                       top_k_h,
                                       top_p_h,
                                       temp_h,
                                       rep_h,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       false,
                                       all_probs_out,
                                       pres_h,
                                       freq_h,
                                       std::nullopt,
                                       gens,
                                       &sampling_buffers};
                        execSampleGreedy(p);
                    });

                    const double e2e_no_all = std::max(0.0, no_all_total - restore_us);
                    const double e2e_all    = std::max(0.0, all_total - restore_us);
                    std::printf("%-4ld | %4.2f | %-4d | %-3s | %7.1f   | %7.1f\n",
                                static_cast<long>(batch),
                                top_p_val,
                                top_k_val,
                                pen_on ? "on" : "off",
                                e2e_no_all,
                                e2e_all);
                    if (batch == 80 && top_p_val == 0.95f && top_k_val == 20 && pen_on) {
                        b80_e2e_after_us = e2e_all;
                    }
                }
            }
        }
    }

    const double kernel_speedup = b80_before_joint_pen_us / std::max(b80_after_joint_pen_us, 1e-6);
    std::printf(
        "\nB=80 Joint+Pen Summary: Before=%.1f us, After(subkernels)=%.1f us (%.2fx), After(E2E allP)=%.1f us\n",
        b80_before_joint_pen_us,
        b80_after_joint_pen_us,
        kernel_speedup,
        b80_e2e_after_us);
    EXPECT_GT(kernel_speedup, 2.0);
    EXPECT_LT(b80_after_joint_pen_us, 400.0);
}
