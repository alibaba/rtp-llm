#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

namespace rtp_llm {
namespace {

constexpr int64_t kStreamCount = 35;
constexpr int64_t kProposeStep = 3;
constexpr int64_t kVocabSize   = 129280;
constexpr int     kWarmupIters = 2;
constexpr int     kBenchIters  = 8;

using Clock = std::chrono::steady_clock;

class AllowAllSpecProcessor: public SpecLogitsProcessor {
public:
    bool isSpecVerifyEligible() const override {
        return true;
    }

    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        // SpecLogitsVerifyRunner pre-fills the processor mask with allow-all.
        // Keeping this processor intentionally cheap isolates runner overhead
        // from XGrammar state-machine work.
        return request.propose_step;
    }
};

class MaskTokenSpecProcessor: public SpecLogitsProcessor {
public:
    explicit MaskTokenSpecProcessor(int32_t token_id, int cap = -1): token_id_(token_id), cap_(cap) {}

    bool isSpecVerifyEligible() const override {
        return true;
    }

    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        const size_t word_idx = static_cast<size_t>(token_id_ / 32);
        const auto   bit      = static_cast<uint32_t>(1u << (token_id_ % 32));
        for (int offset = 0; offset <= request.propose_step; ++offset) {
            auto* row = request.bitmask_cpu_out + static_cast<size_t>(offset) * request.bitmask_size_int32;
            row[word_idx] &= static_cast<int32_t>(~bit);
        }
        return cap_ < 0 ? request.propose_step : cap_;
    }

private:
    int32_t token_id_;
    int     cap_;
};

class RecordingSpecProcessor: public SpecLogitsProcessor {
public:
    bool isSpecVerifyEligible() const override {
        return true;
    }

    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        observed_tokens.assign(request.draft_tokens, request.draft_tokens + request.propose_step);
        return request.propose_step;
    }

    std::vector<int32_t> observed_tokens;
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

struct BuildLatencyStats {
    LatencyStats build_cpu;
    LatencyStats ready_wait;
};

BuildLatencyStats benchmarkBuild(SpecLogitsVerifyRunner& runner, const SpecLogitsVerifyRunner::LaunchTask& task) {
    auto run_once = [&runner, &task]() {
        const auto begin  = Clock::now();
        auto       result = runner.buildInline(task);
        const auto built  = Clock::now();

        EXPECT_TRUE(result.has_active_processor);
        EXPECT_TRUE(result.ready_event != nullptr);
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

    std::vector<double> build_samples;
    std::vector<double> ready_samples;
    build_samples.reserve(kBenchIters);
    ready_samples.reserve(kBenchIters);
    for (int i = 0; i < kBenchIters; ++i) {
        auto [build_us, ready_us] = run_once();
        build_samples.push_back(build_us);
        ready_samples.push_back(ready_us);
    }
    return {summarize(std::move(build_samples)), summarize(std::move(ready_samples))};
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
        task.active.push_back({processor,
                               i,
                               /*processor_idx=*/0,
                               /*stream_id=*/i + 1,
                               /*base_seq_len=*/1024,
                               /*base_output_len=*/128});
    }
    return task;
}

TEST(SpecLogitsVerifyRunnerTest, PackedToDenseHandlesArbitraryBitsAndTail) {
    constexpr size_t rows          = 2;
    constexpr size_t vocab_size    = 35;
    constexpr size_t bitmask_words = 2;

    SpecLogitsVerifyRunner runner;
    runner.merged_bitmask_cpu_ = torch::tensor({1513931685, 5, -1513931686, 2}, torch::kInt32);
    auto mask = torch::empty({static_cast<int64_t>(rows), static_cast<int64_t>(vocab_size)}, torch::kBool);

    runner.unpackMergedBitmaskToVocabMask(mask, rows, vocab_size, bitmask_words);

    const auto* words = runner.merged_bitmask_cpu_.data_ptr<int32_t>();
    const auto* dense = mask.data_ptr<bool>();
    for (size_t row = 0; row < rows; ++row) {
        for (size_t token = 0; token < vocab_size; ++token) {
            const auto word     = static_cast<uint32_t>(words[row * bitmask_words + token / 32]);
            const bool expected = (word & (1u << (token % 32))) == 0u;
            EXPECT_EQ(dense[row * vocab_size + token], expected) << "row=" << row << " token=" << token;
        }
    }
}

TEST(SpecLogitsVerifyRunnerTest, SparseActiveRowsLeaveInactiveRowsUnmasked) {
    constexpr size_t  stream_count = 3;
    constexpr int     propose_step = 1;
    constexpr size_t  vocab_size   = 35;
    constexpr int32_t masked_token = 34;

    SpecLogitsVerifyRunner runner;
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = stream_count;
    task.propose_step  = propose_step;
    task.vocab_size    = vocab_size;
    task.draft_tokens  = torch::zeros({static_cast<int64_t>(stream_count), propose_step}, torch::kInt32);
    task.active.push_back({std::make_shared<MaskTokenSpecProcessor>(masked_token),
                           /*stream_idx=*/1,
                           /*processor_idx=*/0,
                           /*stream_id=*/17,
                           /*base_seq_len=*/0,
                           /*base_output_len=*/0});
    task.active.push_back({std::make_shared<MaskTokenSpecProcessor>(masked_token - 1, /*cap=*/0),
                           /*stream_idx=*/1,
                           /*processor_idx=*/1,
                           /*stream_id=*/17,
                           /*base_seq_len=*/0,
                           /*base_output_len=*/0});

    auto result = runner.buildInline(task);
    ASSERT_TRUE(result.ready_event != nullptr);
    result.ready_event->synchronize();
    auto mask_cpu = result.spec_vocab_mask_gpu.cpu();
    auto cap_cpu  = result.spec_cap_gpu.cpu();

    ASSERT_EQ(mask_cpu.dim(), 2);
    ASSERT_EQ(mask_cpu.size(0), 6);
    ASSERT_EQ(mask_cpu.size(1), static_cast<int64_t>(vocab_size));
    for (int64_t row = 0; row < mask_cpu.size(0); ++row) {
        const bool active_row = row == 2 || row == 3;
        EXPECT_EQ(mask_cpu[row][masked_token].item<bool>(), active_row) << "row=" << row;
        EXPECT_EQ(mask_cpu[row][masked_token - 1].item<bool>(), active_row) << "row=" << row;
        EXPECT_FALSE(mask_cpu[row][0].item<bool>()) << "row=" << row;
    }
    EXPECT_EQ(cap_cpu[0].item<int32_t>(), propose_step);
    EXPECT_EQ(cap_cpu[1].item<int32_t>(), 0);
    EXPECT_EQ(cap_cpu[2].item<int32_t>(), propose_step);
    ASSERT_EQ(result.applied_processors.size(), 2);
    EXPECT_EQ(result.applied_processors[0].stream_id, 17);
    EXPECT_EQ(result.applied_processors[0].processor_idx, 0);
    EXPECT_EQ(result.applied_processors[1].stream_id, 17);
    EXPECT_EQ(result.applied_processors[1].processor_idx, 1);
}

TEST(SpecLogitsVerifyRunnerTest, AllActiveRowsUseDenseFastPath) {
    constexpr size_t  stream_count = 2;
    constexpr int     propose_step = 2;
    constexpr size_t  vocab_size   = 35;
    constexpr int32_t first_token  = 0;
    constexpr int32_t second_token = 34;

    SpecLogitsVerifyRunner runner;
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = stream_count;
    task.propose_step  = propose_step;
    task.vocab_size    = vocab_size;
    task.draft_tokens  = torch::zeros({static_cast<int64_t>(stream_count), propose_step}, torch::kInt32);
    task.active.push_back({std::make_shared<MaskTokenSpecProcessor>(second_token, /*cap=*/1),
                           /*stream_idx=*/1,
                           /*processor_idx=*/7,
                           /*stream_id=*/22,
                           /*base_seq_len=*/0,
                           /*base_output_len=*/0});
    task.active.push_back({std::make_shared<MaskTokenSpecProcessor>(first_token),
                           /*stream_idx=*/0,
                           /*processor_idx=*/3,
                           /*stream_id=*/11,
                           /*base_seq_len=*/0,
                           /*base_output_len=*/0});

    auto result = runner.buildInline(task);
    ASSERT_TRUE(result.ready_event != nullptr);
    result.ready_event->synchronize();
    auto mask_cpu = result.spec_vocab_mask_gpu.cpu();
    auto cap_cpu  = result.spec_cap_gpu.cpu();

    ASSERT_EQ(mask_cpu.size(0), 6);
    ASSERT_EQ(mask_cpu.size(1), static_cast<int64_t>(vocab_size));
    for (int64_t row = 0; row < mask_cpu.size(0); ++row) {
        const bool first_stream = row < propose_step + 1;
        EXPECT_EQ(mask_cpu[row][first_token].item<bool>(), first_stream) << "row=" << row;
        EXPECT_EQ(mask_cpu[row][second_token].item<bool>(), !first_stream) << "row=" << row;
        EXPECT_FALSE(mask_cpu[row][1].item<bool>()) << "row=" << row;
    }
    EXPECT_EQ(cap_cpu[0].item<int32_t>(), propose_step);
    EXPECT_EQ(cap_cpu[1].item<int32_t>(), 1);
    ASSERT_EQ(result.applied_processors.size(), 2);
    EXPECT_EQ(result.applied_processors[0].stream_id, 22);
    EXPECT_EQ(result.applied_processors[0].processor_idx, 7);
    EXPECT_EQ(result.applied_processors[1].stream_id, 11);
    EXPECT_EQ(result.applied_processors[1].processor_idx, 3);
}

TEST(SpecLogitsVerifyRunnerTest, LeadingVerifyAnchorIsNotPassedAsProposal) {
    constexpr int    propose_step = 3;
    constexpr size_t vocab_size   = 35;

    auto processor = std::make_shared<RecordingSpecProcessor>();

    SpecLogitsVerifyRunner             runner;
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = 1;
    task.propose_step  = propose_step;
    task.vocab_size    = vocab_size;
    task.draft_tokens  = torch::tensor({99, 11, 12, 13}, torch::kInt32).reshape({1, propose_step + 1});
    task.active.push_back({processor,
                           /*stream_idx=*/0,
                           /*processor_idx=*/0,
                           /*stream_id=*/1,
                           /*base_seq_len=*/0,
                           /*base_output_len=*/0});

    auto result = runner.buildInline(task);
    ASSERT_TRUE(result.ready_event != nullptr);
    result.ready_event->synchronize();
    EXPECT_EQ((std::vector<int32_t>{11, 12, 13}), processor->observed_tokens);
}

// Manual perf test matching normal_profiler_wr3_1.json:
// decode_stream_size=35, mtp_step=3, DeepSeek-V4 vocab_size=129280.
// Run explicitly with --gtest_also_run_disabled_tests.
TEST(SpecLogitsVerifyRunnerPerfTest, DISABLED_DeepSeekFlashDecodeB35P3) {
    constexpr int64_t rows           = kStreamCount * (kProposeStep + 1);
    constexpr int64_t bitmask_words  = (kVocabSize + 31) / 32;
    constexpr int64_t dense_elements = rows * kVocabSize;
    constexpr int64_t packed_bytes   = rows * bitmask_words * sizeof(int32_t);
    constexpr int64_t dense_bytes    = dense_elements * sizeof(bool);

    static_assert(dense_bytes == 18099200, "timeline dense-mask H2D byte count changed");
    static_assert(kStreamCount * kProposeStep * static_cast<int64_t>(sizeof(int32_t)) == 420,
                  "timeline draft-token D2H byte count changed");

    std::cout << "[spec-logits-perf] shape B=" << kStreamCount << " P=" << kProposeStep << " V=" << kVocabSize
              << " rows=" << rows << " packed_bytes=" << packed_bytes << " dense_bytes=" << dense_bytes
              << " dense_over_packed=" << static_cast<double>(dense_bytes) / packed_bytes << std::endl;

    SpecLogitsVerifyRunner runner;

    // Isolate the packed-to-dense O(B * (P + 1) * V) loop used by buildInline.
    // The test target compiles with -fno-access-control, matching the existing
    // logits-processor test convention, so private scratch/method access here
    // does not change the production API.
    runner.merged_bitmask_cpu_ = torch::full({rows, bitmask_words},
                                             SpecLogitsProcessor::kBitmaskAllowAll,
                                             torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto dense_mask_cpu = torch::empty(
        {rows, kVocabSize}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(true));

    auto unpack_stats = benchmarkCpu([&]() {
        runner.unpackMergedBitmaskToVocabMask(dense_mask_cpu, static_cast<size_t>(rows), kVocabSize, bitmask_words);
    });
    printStats("packed_to_dense_cpu", unpack_stats);
    EXPECT_FALSE(dense_mask_cpu.data_ptr<bool>()[0]);
    EXPECT_FALSE(dense_mask_cpu.data_ptr<bool>()[dense_elements - 1]);

    // One active processor exercises the compact path: only its P+1 rows are
    // expanded on CPU and copied into an otherwise-false full GPU mask.
    auto one_active_task  = makeTask(/*active_processor_count=*/1);
    auto one_active_stats = benchmarkBuild(runner, one_active_task);
    printStats("build_inline_one_active_cpu_enqueue", one_active_stats.build_cpu);
    printStats("build_inline_one_active_ready_wait", one_active_stats.ready_wait);

    // All-active measures the dense fallback and per-processor packed work.
    auto all_active_task  = makeTask(/*active_processor_count=*/kStreamCount);
    auto all_active_stats = benchmarkBuild(runner, all_active_task);
    printStats("build_inline_all_active_cpu_enqueue", all_active_stats.build_cpu);
    printStats("build_inline_all_active_ready_wait", all_active_stats.ready_wait);

    // Each benchmark iteration waits for ready_event, so the same pinned source
    // slot should be reused rather than going through the pinned allocator every
    // decode round.
    EXPECT_EQ(runner.cpu_artifact_slots_.size(), 1);
}

}  // namespace
}  // namespace rtp_llm
