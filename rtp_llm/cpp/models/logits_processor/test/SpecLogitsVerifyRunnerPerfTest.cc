#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {
namespace {

// Shape from kimi_k3_0920_1789886927_decode_wr0_1.json.
constexpr int64_t kStreamCount = 21;
constexpr int64_t kProposeStep = 3;
constexpr int64_t kVocabSize   = 163840;
constexpr int     kWarmupIters = 3;
constexpr int     kBenchIters  = 20;

class AllowAllSpecProcessor: public SpecLogitsProcessor {
public:
    bool isSpecVerifyEligible() const override {
        return true;
    }
    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        return request.propose_step;
    }
};

struct Stats {
    double mean_us;
    double p50_us;
    double p90_us;
};

Stats summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    auto percentile = [&values](double q) {
        const size_t rank = std::min(values.size() - 1, static_cast<size_t>(std::ceil(q * values.size())) - 1);
        return values[rank];
    };
    return {std::accumulate(values.begin(), values.end(), 0.0) / values.size(), percentile(0.5), percentile(0.9)};
}

void printStats(const char* label, const Stats& stats) {
    std::cout << std::fixed << std::setprecision(3) << "[spec-logits-perf] " << label
              << " mean_us=" << stats.mean_us << " p50_us=" << stats.p50_us << " p90_us=" << stats.p90_us
              << std::endl;
}

class SpecLogitsVerifyRunnerPerfTest: public DeviceTestBase {};

// Manual benchmark: --gtest_also_run_disabled_tests
TEST_F(SpecLogitsVerifyRunnerPerfTest, DISABLED_KimiK3DecodeB21P3V163840) {
    using Clock = std::chrono::steady_clock;
    constexpr int64_t rows         = kStreamCount * (kProposeStep + 1);
    constexpr int64_t words        = (kVocabSize + 31) / 32;
    constexpr int64_t packed_bytes = rows * words * static_cast<int64_t>(sizeof(int32_t));
    constexpr int64_t dense_bytes  = rows * kVocabSize * static_cast<int64_t>(sizeof(bool));

    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = kStreamCount;
    task.propose_step  = kProposeStep;
    task.vocab_size    = kVocabSize;
    task.draft_tokens  = torch::zeros({kStreamCount, kProposeStep}, torch::kInt32).to(torch::kCUDA);
    auto processor     = std::make_shared<AllowAllSpecProcessor>();
    for (int64_t i = 0; i < kStreamCount; ++i) {
        task.active.push_back({processor, static_cast<size_t>(i), 0, static_cast<uint64_t>(i), 0, 0});
    }

    auto benchmark = [](auto&& fn) {
        for (int i = 0; i < kWarmupIters; ++i) {
            fn();
        }
        std::vector<double> samples;
        for (int i = 0; i < kBenchIters; ++i) {
            const auto begin = Clock::now();
            fn();
            const auto end = Clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        }
        return summarize(std::move(samples));
    };

    SpecLogitsVerifyRunner runner;
    auto packed_stats = benchmark([&]() {
        auto result = runner.buildInline(task);
        result.ready_event->synchronize();
        result.consumed_event->record(cuda_graph::graphGetCurrentStream());
    });

    // Reproduce the removed path: packed-to-dense CPU expansion followed by
    // a dense bool H2D upload.
    auto dense_stats = benchmark([&]() {
        auto dense_cpu = torch::empty({rows, kVocabSize},
                                      torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(true));
        std::fill_n(dense_cpu.data_ptr<bool>(), dense_cpu.numel(), false);
        auto dense_gpu = torch::empty({rows, kVocabSize},
                                      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        dense_gpu.copy_(dense_cpu, /*non_blocking=*/true);
        torch::cuda::synchronize();
    });

    auto result = runner.buildInline(task);
    result.ready_event->synchronize();
    auto logits = torch::zeros({rows, kVocabSize}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto apply_stats = benchmark([&]() {
        SpecLogitsVerifyRunner::applyMaskToLogits(logits, result, kVocabSize);
        torch::cuda::synchronize();
    });
    result.consumed_event->record(cuda_graph::graphGetCurrentStream());

    std::cout << "[spec-logits-perf] shape B=" << kStreamCount << " P=" << kProposeStep << " V=" << kVocabSize
              << " rows=" << rows << " packed_bytes=" << packed_bytes << " dense_bytes=" << dense_bytes
              << " dense_over_packed=" << static_cast<double>(dense_bytes) / packed_bytes << std::endl;
    printStats("legacy_dense_build_h2d", dense_stats);
    printStats("packed_build_h2d", packed_stats);
    printStats("packed_apply_kernel", apply_stats);
}

}  // namespace
}  // namespace rtp_llm
