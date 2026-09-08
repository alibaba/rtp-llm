#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace {

// Perf shape matching normal_profiler_wr3_1.json:
// decode_stream_size=35, mtp_step=3, DeepSeek-V4 vocab_size=129280.
constexpr int64_t kStreamCount = 35;
constexpr int64_t kProposeStep = 3;
constexpr int64_t kVocabSize   = 129280;
constexpr int     kWarmupIters = 2;
constexpr int     kBenchIters  = 8;

using Clock = std::chrono::steady_clock;

// SpecLogitsVerifyRunner pre-fills the processor mask with allow-all.
// Keeping this processor intentionally cheap isolates runner overhead
// from XGrammar state-machine work.
class AllowAllSpecProcessor: public BaseLogitsProcessor {
public:
    std::optional<ErrorInfo> process(const SamplerInputs&, size_t, size_t) override {
        return std::nullopt;
    }
    void                     updateMultiSeqStatus(const std::vector<int>&) override {}
    std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t) override {
        return std::nullopt;
    }

    MtpProcessorCapability mtpCapability() const override {
        return {MtpProcessorMode::SPEC_VERIFY, "allow-all perf processor"};
    }

    ErrorResult<int> prepareSpeculative(const SpecLogitsProcessorRequest& request) override {
        return ErrorResult<int>(static_cast<int>(request.propose_step));
    }
};

struct LatencyStats {
    double min_us  = 0;
    double mean_us = 0;
    double p50_us  = 0;
    double p90_us  = 0;
    double max_us  = 0;
};

LatencyStats summarize(std::vector<double> samples) {
    EXPECT_FALSE(samples.empty());
    std::sort(samples.begin(), samples.end());
    auto percentile = [&samples](double quantile) {
        const auto rank = static_cast<size_t>(std::ceil(quantile * samples.size()));
        return samples[std::min(samples.size() - 1, std::max<size_t>(1, rank) - 1)];
    };

    LatencyStats stats;
    stats.min_us  = samples.front();
    stats.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    stats.p50_us  = percentile(0.50);
    stats.p90_us  = percentile(0.90);
    stats.max_us  = samples.back();
    return stats;
}

template<typename Func>
LatencyStats benchmarkCpu(Func&& func) {
    for (int i = 0; i < kWarmupIters; ++i) {
        func();
    }

    std::vector<double> samples;
    samples.reserve(kBenchIters);
    for (int i = 0; i < kBenchIters; ++i) {
        const auto begin = Clock::now();
        func();
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    return summarize(std::move(samples));
}

void printStats(const std::string& label, const LatencyStats& stats) {
    std::cout << std::fixed << std::setprecision(3) << "[spec-logits-perf] " << label << " min_us=" << stats.min_us
              << " mean_us=" << stats.mean_us << " p50_us=" << stats.p50_us << " p90_us=" << stats.p90_us
              << " max_us=" << stats.max_us << std::endl;
}

struct RunLatencyStats {
    LatencyStats run_cpu;
    LatencyStats ready_wait;
};

RunLatencyStats benchmarkRun(SpecLogitsVerifyRunner& runner, const SpecLogitsVerifyRunner::LaunchTask& task) {
    auto run_once = [&runner, &task]() {
        const auto begin  = Clock::now();
        auto       result = runner.run(task);
        const auto built  = Clock::now();

        EXPECT_TRUE(result.has_active_processor);
        if (result.ready_event) {
            result.ready_event->synchronize();
        }
        const auto ready = Clock::now();
        return std::pair<double, double>{
            std::chrono::duration<double, std::micro>(built - begin).count(),
            std::chrono::duration<double, std::micro>(ready - built).count(),
        };
    };

    for (int i = 0; i < kWarmupIters; ++i) {
        run_once();
    }

    std::vector<double> run_samples;
    std::vector<double> ready_samples;
    run_samples.reserve(kBenchIters);
    ready_samples.reserve(kBenchIters);
    for (int i = 0; i < kBenchIters; ++i) {
        auto [run_us, ready_us] = run_once();
        run_samples.push_back(run_us);
        ready_samples.push_back(ready_us);
    }
    return {summarize(std::move(run_samples)), summarize(std::move(ready_samples))};
}

SpecLogitsVerifyRunner::LaunchTask makeTask(size_t active_processor_count) {
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = kStreamCount;
    task.propose_step  = kProposeStep;
    task.vocab_size    = kVocabSize;
    task.draft_tokens  = torch::zeros({kStreamCount, kProposeStep}, torch::kInt32);

    auto processor = std::make_shared<AllowAllSpecProcessor>();
    task.active.reserve(active_processor_count);
    for (size_t i = 0; i < active_processor_count; ++i) {
        task.active.push_back({processor, /*stream_idx=*/i});
    }
    return task;
}

// Manual perf test. Run explicitly with --gtest_also_run_disabled_tests.
TEST(SpecLogitsVerifyRunnerPerfTest, DISABLED_DeepSeekFlashDecodeB35P3) {
    constexpr int64_t rows          = kStreamCount * (kProposeStep + 1);
    constexpr int64_t bitmask_words = (kVocabSize + 31) / 32;
    constexpr int64_t packed_bytes  = rows * bitmask_words * static_cast<int64_t>(sizeof(int32_t));
    // The pre-#1006 dense bool mask this shape would have uploaded; main keeps
    // masks packed on GPU, so this is only logged as the avoided-H2D baseline.
    constexpr int64_t dense_bytes = rows * kVocabSize * static_cast<int64_t>(sizeof(bool));

    static_assert(kStreamCount * kProposeStep * static_cast<int64_t>(sizeof(int32_t)) == 420,
                  "timeline draft-token D2H byte count changed");

    std::cout << "[spec-logits-perf] shape B=" << kStreamCount << " P=" << kProposeStep << " V=" << kVocabSize
              << " rows=" << rows << " packed_bytes=" << packed_bytes << " dense_bytes=" << dense_bytes
              << " dense_over_packed=" << static_cast<double>(dense_bytes) / packed_bytes << std::endl;

    SpecLogitsVerifyRunner runner;

    // One active processor exercises the sparse path: only its P+1 compact
    // rows are merged on CPU and uploaded.
    auto one_active_task  = makeTask(/*active_processor_count=*/1);
    auto one_active_stats = benchmarkRun(runner, one_active_task);
    printStats("run_one_active_cpu_enqueue", one_active_stats.run_cpu);
    printStats("run_one_active_ready_wait", one_active_stats.ready_wait);

    // All-active measures the full per-processor packed merge plus the
    // maximum compact-row H2D upload.
    auto all_active_task  = makeTask(/*active_processor_count=*/kStreamCount);
    auto all_active_stats = benchmarkRun(runner, all_active_task);
    printStats("run_all_active_cpu_enqueue", all_active_stats.run_cpu);
    printStats("run_all_active_ready_wait", all_active_stats.ready_wait);

    // Each benchmark iteration waits for ready_event, so the same pinned
    // scratch must be reused rather than reallocated every decode round. The
    // test target compiles with -fno-access-control, matching the existing
    // logits-processor test convention, so private scratch access here does
    // not change the production API.
    const void* pinned_scratch_ptr = runner.merged_bitmask_cpu_.data_ptr();
    auto        repeat_stats       = benchmarkRun(runner, all_active_task);
    printStats("run_all_active_repeat_cpu_enqueue", repeat_stats.run_cpu);
    EXPECT_EQ(pinned_scratch_ptr, runner.merged_bitmask_cpu_.data_ptr());

    // GPU-side application of the packed mask onto real-shaped logits; this is
    // the consumer-side cost that replaced the dev branch's packed-to-dense
    // CPU expansion + dense H2D upload.
    auto result = runner.run(all_active_task);
    if (result.ready_event) {
        result.ready_event->synchronize();
    }
    auto logits_options = torch::TensorOptions().dtype(torch::kFloat32);
#if USING_CUDA
    logits_options = logits_options.device(torch::kCUDA);
#endif
    auto logits      = torch::zeros({rows, kVocabSize}, logits_options);
    auto apply_stats = benchmarkCpu([&]() {
        SpecLogitsVerifyRunner::applyMaskToLogits(logits, result, kVocabSize);
#if USING_CUDA
        torch::cuda::synchronize();
#endif
    });
    printStats("apply_packed_mask", apply_stats);
}

}  // namespace
}  // namespace rtp_llm
