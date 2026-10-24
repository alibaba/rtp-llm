#include "rtp_llm/cpp/models/eplb/ExpertLoadMonitor.h"

#include <c10/core/DeviceGuard.h>
#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <torch/cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

class ExpertLoadMonitorTest: public ::testing::Test {
protected:
    void SetUp() override {
        if (const auto* previous = std::getenv("LOG_PATH")) {
            previous_log_path_ = previous;
        }
        ASSERT_TRUE(torch::cuda::is_available()) << "This test requires the GPU assigned by gpu_lock";
        device_guard_         = std::make_unique<c10::DeviceGuard>(c10::Device(c10::kCUDA, 0));
        const char* temporary = std::getenv("TEST_TMPDIR");
        ASSERT_NE(temporary, nullptr);
        const auto log_path =
            std::filesystem::path(temporary) / ::testing::UnitTest::GetInstance()->current_test_info()->name();
        directory_ = log_path / "eplb_snapshot";
        std::filesystem::remove_all(log_path);
        std::filesystem::create_directories(directory_);
        setenv("LOG_PATH", log_path.c_str(), 1);
        config_.eplb_metrics_report_interval    = 2;
        config_.eplb_metrics_snapshot_max_files = 100;
    }

    void TearDown() override {
        if (previous_log_path_) {
            setenv("LOG_PATH", previous_log_path_->c_str(), 1);
        } else {
            unsetenv("LOG_PATH");
        }
    }

    std::unique_ptr<ExpertLoadMonitor> monitor(size_t experts = 2) {
        return ExpertLoadMonitor::create(1, experts, ParallelismConfig{}, config_, 0, nullptr);
    }

    void step(ExpertLoadMonitor& monitor, int64_t left, int64_t right) {
        monitor.stepCountsBuffer().select(1, 0).add_(left);
        monitor.stepCountsBuffer().select(1, 1).add_(right);
        monitor.recordStep();
    }

    std::vector<rapidjson::Document> snapshots() {
        std::vector<rapidjson::Document> result;
        for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
            if (entry.path().extension() != ".json") {
                throw std::runtime_error("unexpected unfinished snapshot: " + entry.path().string());
            }
            std::ifstream file(entry.path());
            if (!file) {
                throw std::runtime_error("cannot read snapshot: " + entry.path().string());
            }
            rapidjson::IStreamWrapper stream(file);
            rapidjson::Document       document;
            document.ParseStream(stream);
            if (document.HasParseError() || !document.IsObject()) {
                throw std::runtime_error("invalid snapshot JSON: " + entry.path().string());
            }
            result.push_back(std::move(document));
        }
        std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
            return left["step_end"].GetUint64() < right["step_end"].GetUint64();
        });
        return result;
    }

    std::unique_ptr<c10::DeviceGuard> device_guard_;
    std::filesystem::path             directory_;
    std::optional<std::string>        previous_log_path_;
    EPLBConfig                        config_;
};

TEST_F(ExpertLoadMonitorTest, FixedOutputAndClearedZeroWindowHaveNoStaleLoad) {
    auto recorder = monitor();
    ASSERT_EQ(recorder->stepCountsBuffer().scalar_type(), torch::kInt32);
    auto* pointer = recorder->stepCountsBuffer().data_ptr<int32_t>();
    step(*recorder, 4, 0);
    EXPECT_TRUE(snapshots().empty());
    step(*recorder, 0, 4);
    ASSERT_EQ(snapshots().size(), 1u);
    step(*recorder, 0, 0);
    step(*recorder, 0, 0);
    EXPECT_EQ(recorder->stepCountsBuffer().data_ptr<int32_t>(), pointer);
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 2u);
    EXPECT_EQ(windows[0]["schema_version"].GetUint(), 4u);
    EXPECT_EQ(windows[0]["layers"][0]["total_activations"].GetInt64(), 8);
    EXPECT_DOUBLE_EQ(windows[0]["moe_lb"].GetDouble(), 1.0);
    EXPECT_DOUBLE_EQ(windows[1]["moe_lb"].GetDouble(), 1.0);
    EXPECT_DOUBLE_EQ(windows[1]["layers"][0]["moe_lb"].GetDouble(), 1.0);
    EXPECT_EQ(windows[1]["layers"][0]["total_activations"].GetInt64(), 0);
    EXPECT_EQ(windows[1]["rank_samples"][0][0][0].GetInt64(), 0);
    EXPECT_EQ(windows[1]["rank_samples"][0][1][0].GetInt64(), 0);
}

TEST_F(ExpertLoadMonitorTest, MultipleContributionsAccumulateUntilTheNextStep) {
    auto        recorder = monitor();
    const auto* pointer  = recorder->stepCountsBuffer().data_ptr<int32_t>();
    recorder->stepCountsBuffer().select(1, 0).add_(2);
    recorder->stepCountsBuffer().select(1, 0).add_(3);
    recorder->stepCountsBuffer().select(1, 1).add_(4);
    recorder->recordStep();
    EXPECT_EQ(recorder->stepCountsBuffer().count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(recorder->stepCountsBuffer().data_ptr<int32_t>(), pointer);
    recorder->stepCountsBuffer().select(1, 1).add_(6);
    recorder->recordStep();
    EXPECT_EQ(recorder->stepCountsBuffer().count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(recorder->stepCountsBuffer().data_ptr<int32_t>(), pointer);
    recorder->recordStep();
    recorder->recordStep();

    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 2u);
    EXPECT_EQ(windows[0]["rank_samples"][0][0][0].GetInt64(), 9);
    EXPECT_EQ(windows[0]["rank_samples"][0][1][0].GetInt64(), 6);
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 15);
    EXPECT_EQ(windows[1]["rank_samples"][0][0][0].GetInt64(), 0);
    EXPECT_EQ(windows[1]["rank_samples"][0][1][0].GetInt64(), 0);
}

TEST_F(ExpertLoadMonitorTest, LayerAndWindowTotalsUseInt64) {
    auto          recorder = monitor();
    const int64_t count    = std::numeric_limits<int32_t>::max();
    step(*recorder, count, count);
    step(*recorder, count, count);

    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0]["rank_samples"][0][0][0].GetInt64(), 2 * count);
    EXPECT_EQ(windows[0]["rank_samples"][0][1][0].GetInt64(), 2 * count);
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 4 * count);
}

TEST_F(ExpertLoadMonitorTest, UnwrittenFirstForwardAfterWarmupHasZeroLoad) {
    auto recorder = monitor();
    recorder->stepCountsBuffer().fill_(99);
    // Warmup clears without recording; an unsupported backend writes no counts
    // on the following real forward.
    recorder->clearRecordBuffer();
    recorder->recordStep();
    EXPECT_TRUE(snapshots().empty());
    step(*recorder, 4, 0);
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0]["step_begin"].GetUint64(), 1u);
    EXPECT_EQ(windows[0]["step_end"].GetUint64(), 2u);
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 4);
    EXPECT_EQ(windows[0]["rank_samples"][0][0][0].GetInt64(), 0);
    EXPECT_EQ(windows[0]["rank_samples"][0][1][0].GetInt64(), 4);
}

TEST_F(ExpertLoadMonitorTest, TargetAndDraftHaveIndependentBuffersAndWindows) {
    auto target                               = monitor();
    auto draft_config                         = config_;
    draft_config.eplb_metrics_report_interval = 3;
    auto draft = ExpertLoadMonitor::create(1, 2, ParallelismConfig{}, draft_config, 1, nullptr, "draft");
    EXPECT_NE(target->stepCountsBuffer().data_ptr<int32_t>(), draft->stepCountsBuffer().data_ptr<int32_t>());

    step(*target, 2, 3);
    step(*draft, 5, 6);
    step(*target, 3, 4);
    ASSERT_EQ(snapshots().size(), 1u);
    step(*draft, 6, 7);
    ASSERT_EQ(snapshots().size(), 1u);
    step(*target, 99, 99);
    step(*draft, 8, 9);

    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 2u);
    EXPECT_EQ(windows[0]["model_id"].GetUint64(), 0u);
    EXPECT_STREQ(windows[0]["model_type"].GetString(), "target");
    EXPECT_EQ(windows[0]["step_end"].GetUint64(), 2u);
    EXPECT_EQ(windows[0]["sample_count"].GetUint64(), 2u);
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 12);
    EXPECT_EQ(windows[1]["model_id"].GetUint64(), 1u);
    EXPECT_STREQ(windows[1]["model_type"].GetString(), "draft");
    EXPECT_EQ(windows[1]["step_end"].GetUint64(), 3u);
    EXPECT_EQ(windows[1]["sample_count"].GetUint64(), 3u);
    EXPECT_EQ(windows[1]["total_activations"].GetInt64(), 41);
}

TEST_F(ExpertLoadMonitorTest, DenseModelReportsOneWithoutExpertStorage) {
    auto recorder = monitor(0);
    EXPECT_EQ(recorder->stepCountsBuffer().size(0), 1);
    EXPECT_EQ(recorder->stepCountsBuffer().size(1), 0);
    recorder->recordStep();
    recorder->recordStep();

    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0]["physical_expert_count"].GetUint64(), 0u);
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 0);
    EXPECT_DOUBLE_EQ(windows[0]["moe_lb"].GetDouble(), 1.0);
    EXPECT_DOUBLE_EQ(windows[0]["layers"][0]["moe_lb"].GetDouble(), 1.0);
    EXPECT_EQ(windows[0]["rank_samples"][0][0][0].GetInt64(), 0);
    EXPECT_EQ(windows[0]["rank_samples"][0][1][0].GetInt64(), 0);
}

TEST_F(ExpertLoadMonitorTest, DenseContextParallelModelReportsOne) {
    ParallelismConfig parallelism;
    parallelism.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    auto recorder                        = ExpertLoadMonitor::create(1, 0, parallelism, config_, 0, nullptr);
    ASSERT_NE(recorder, nullptr);
    EXPECT_EQ(recorder->stepCountsBuffer().size(1), 0);
    recorder->recordStep();
    recorder->recordStep();

    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 0);
    EXPECT_DOUBLE_EQ(windows[0]["moe_lb"].GetDouble(), 1.0);
    EXPECT_EQ(windows[0]["rank_samples"][0][0][0].GetInt64(), 0);
    EXPECT_EQ(windows[0]["rank_samples"][0][1][0].GetInt64(), 0);
}

TEST_F(ExpertLoadMonitorTest, MoeContextParallelModelWithoutEpSkipsMonitoring) {
    ParallelismConfig parallelism;
    parallelism.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_EQ(ExpertLoadMonitor::create(1, 2, parallelism, config_, 0, nullptr), nullptr);
}

TEST_F(ExpertLoadMonitorTest, UnsupportedTopologiesSkipMonitoring) {
    ParallelismConfig pipeline;
    pipeline.pp_size = 2;
    EXPECT_EQ(ExpertLoadMonitor::create(1, 2, pipeline, config_, 0, nullptr), nullptr);
    EXPECT_EQ(ExpertLoadMonitor::create(1, 0, pipeline, config_, 0, nullptr), nullptr);

    ParallelismConfig disaggregated;
    disaggregated.ffn_disaggregate_config.enable_ffn_disaggregate = true;
    EXPECT_EQ(ExpertLoadMonitor::create(1, 2, disaggregated, config_, 0, nullptr), nullptr);

    ParallelismConfig partial_ep;
    partial_ep.world_size = 4;
    partial_ep.ep_size    = 2;
    partial_ep.dp_size    = 2;
    EXPECT_EQ(ExpertLoadMonitor::create(1, 4, partial_ep, config_, 0, nullptr), nullptr);
    EXPECT_TRUE(snapshots().empty());
}

TEST_F(ExpertLoadMonitorTest, InvalidDimensionsAndWindowRemainErrors) {
    ParallelismConfig parallelism;
    parallelism.ep_size = 0;
    EXPECT_THROW(ExpertLoadMonitor::create(1, 2, parallelism, config_, 0, nullptr), std::runtime_error);
    parallelism.ep_size = 2;
    EXPECT_THROW(ExpertLoadMonitor::create(1, 3, parallelism, config_, 0, nullptr), std::runtime_error);
    parallelism = ParallelismConfig{};
    EXPECT_THROW(ExpertLoadMonitor::create(0, 2, parallelism, config_, 0, nullptr), std::runtime_error);
    config_.eplb_metrics_report_interval = 0;
    EXPECT_THROW(ExpertLoadMonitor::create(1, 2, parallelism, config_, 0, nullptr), std::runtime_error);
    config_.eplb_metrics_report_interval    = 2;
    config_.eplb_metrics_snapshot_max_files = -1;
    EXPECT_THROW(ExpertLoadMonitor::create(1, 2, parallelism, config_, 0, nullptr), std::runtime_error);
}

TEST_F(ExpertLoadMonitorTest, DenseMultiRankWindowNeedsNoCollectiveAndOnlyRootReports) {
    ParallelismConfig parallelism;
    parallelism.world_size = 2;
    parallelism.ep_size    = 2;
    for (int rank : {1, 0}) {
        parallelism.world_rank = rank;
        parallelism.ep_rank    = rank;
        auto recorder          = ExpertLoadMonitor::create(1, 0, parallelism, config_, 0, nullptr);
        ASSERT_NE(recorder, nullptr);
        // This test binary's execAllGather stub throws on every call.
        recorder->recordStep();
        recorder->recordStep();
        EXPECT_EQ(snapshots().size(), rank == 0 ? 1u : 0u);
    }
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0]["world_size"].GetUint64(), 2u);
    EXPECT_EQ(windows[0]["rank_samples"].Size(), 2u);
    EXPECT_EQ(windows[0]["rank_activations"].Size(), 2u);
    for (const auto& rank : windows[0]["rank_samples"].GetArray()) {
        for (const auto& sample : rank.GetArray()) {
            EXPECT_EQ(sample[0].GetInt64(), 0);
        }
    }
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 0);
    EXPECT_DOUBLE_EQ(windows[0]["moe_lb"].GetDouble(), 1.0);
}

TEST_F(ExpertLoadMonitorTest, WarmupDoesNotAdvanceIdsOrEnterTheWindow) {
    auto recorder = monitor();
    step(*recorder, 1, 2);
    recorder->stepCountsBuffer().fill_(99);
    // Production callers discard warmup counts without recording a sample.
    recorder->clearRecordBuffer();
    EXPECT_TRUE(snapshots().empty());
    step(*recorder, 3, 4);
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0]["step_begin"].GetUint64(), 1u);
    EXPECT_EQ(windows[0]["step_end"].GetUint64(), 2u);
    EXPECT_EQ(windows[0]["sample_count"].GetUint64(), 2u);
    EXPECT_EQ(windows[0]["layers"][0]["total_activations"].GetInt64(), 10);
    EXPECT_EQ(windows[0]["rank_samples"][0][0][0].GetInt64(), 3);
    EXPECT_EQ(windows[0]["rank_samples"][0][1][0].GetInt64(), 7);
}

TEST_F(ExpertLoadMonitorTest, SynchronousReportingIncludesEveryForward) {
    auto recorder = monitor();
    for (int64_t id = 1; id <= 100; ++id) {
        step(*recorder, id, 2 * id);
    }
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 50u);
    int64_t id = 1;
    for (const auto& document : windows) {
        EXPECT_EQ(document["step_begin"].GetInt64(), id);
        ASSERT_EQ(document["rank_samples"].Size(), 1u);
        const auto& samples = document["rank_samples"][0];
        ASSERT_EQ(samples.Size(), 2u);
        for (const auto& sample : samples.GetArray()) {
            ASSERT_EQ(sample.Size(), 1u);
            EXPECT_EQ(sample[0].GetInt64(), 3 * id);
            ++id;
        }
        EXPECT_EQ(document["step_end"].GetInt64(), id - 1);
    }
    EXPECT_EQ(id, 101);
}

TEST_F(ExpertLoadMonitorTest, SingleEpOwnerHasUnitBalanceRegardlessOfExpertSkew) {
    auto recorder = monitor(5);
    recorder->stepCountsBuffer().fill_(128);
    recorder->stepCountsBuffer().select(1, 4).fill_(64);
    recorder->recordStep();
    recorder->stepCountsBuffer().select(1, 0).fill_(5);
    recorder->recordStep();
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    const auto& layer = windows[0]["layers"][0];
    EXPECT_DOUBLE_EQ(layer["moe_lb"].GetDouble(), 1.0);
    EXPECT_EQ(layer["total_activations"].GetInt64(), 581);
    EXPECT_EQ(layer["max_rank_activations"].GetInt64(), 581);
    EXPECT_EQ(layer["layer_id"].GetUint64(), 0u);
}

TEST_F(ExpertLoadMonitorTest, MappingChangeDropsOnlyTheUnsubmittedPartialWindow) {
    auto recorder = monitor();
    step(*recorder, 99, 99);
    recorder->beforeMappingChange();
    step(*recorder, 1, 2);
    step(*recorder, 3, 4);
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0]["mapping_epoch"].GetUint64(), 1u);
    EXPECT_EQ(windows[0]["step_begin"].GetUint64(), 2u);
    EXPECT_EQ(windows[0]["layers"][0]["total_activations"].GetInt64(), 10);
}

TEST_F(ExpertLoadMonitorTest, DestructionLeavesOnlyCompletedWindows) {
    {
        auto recorder = monitor();
        step(*recorder, 1, 2);
    }
    EXPECT_TRUE(snapshots().empty());
    {
        auto recorder = monitor();
        step(*recorder, 1, 2);
        step(*recorder, 3, 4);
    }
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 10);
}

TEST_F(ExpertLoadMonitorTest, ZeroSnapshotRetentionDoesNotWriteFiles) {
    config_.eplb_metrics_snapshot_max_files = 0;
    auto recorder                           = monitor();
    step(*recorder, 1, 2);
    step(*recorder, 3, 4);
    EXPECT_TRUE(snapshots().empty());
}

TEST_F(ExpertLoadMonitorTest, SnapshotWriteFailureDoesNotStopTheNextWindow) {
    auto recorder = monitor();
    std::filesystem::remove_all(directory_);
    std::ofstream(directory_).close();
    step(*recorder, 1, 0);
    EXPECT_NO_THROW(step(*recorder, 2, 0));

    std::filesystem::remove(directory_);
    step(*recorder, 3, 0);
    step(*recorder, 4, 0);
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0]["step_begin"].GetUint64(), 3u);
    EXPECT_EQ(windows[0]["step_end"].GetUint64(), 4u);
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 7);
}

TEST_F(ExpertLoadMonitorTest, SnapshotFilesRotateInConfiguredSlots) {
    config_.eplb_metrics_snapshot_max_files = 2;
    auto recorder                           = monitor();
    for (int64_t id = 1; id <= 4; ++id) {
        step(*recorder, id, 0);
    }
    ASSERT_EQ(snapshots().size(), 2u);
    std::vector<std::filesystem::path> slots;
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
        slots.push_back(entry.path());
    }

    for (int64_t id = 5; id <= 8; ++id) {
        step(*recorder, id, 0);
    }
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 2u);
    EXPECT_EQ(windows[0]["step_end"].GetUint64(), 6u);
    EXPECT_EQ(windows[0]["total_activations"].GetInt64(), 11);
    EXPECT_EQ(windows[1]["step_end"].GetUint64(), 8u);
    EXPECT_EQ(windows[1]["total_activations"].GetInt64(), 15);
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
        EXPECT_NE(std::find(slots.begin(), slots.end(), entry.path()), slots.end());
    }
}

TEST_F(ExpertLoadMonitorTest, SnapshotRotationKeepsOtherModelSessions) {
    config_.eplb_metrics_snapshot_max_files = 1;
    auto              first                 = monitor();
    auto              second                = monitor();
    auto draft = ExpertLoadMonitor::create(1, 2, ParallelismConfig{}, config_, 0, nullptr, "draft");
    for (int i = 0; i < 2; ++i) {
        step(*first, 1, 0);
        step(*second, 2, 0);
        step(*draft, 3, 0);
    }
    for (int i = 0; i < 2; ++i) {
        step(*first, 4, 0);
    }

    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 3u);
    std::vector<int64_t> totals;
    size_t               draft_windows = 0;
    for (const auto& window : windows) {
        EXPECT_EQ(window["model_id"].GetUint64(), 0u);
        totals.push_back(window["total_activations"].GetInt64());
        draft_windows += std::string(window["model_type"].GetString()) == "draft";
    }
    std::sort(totals.begin(), totals.end());
    EXPECT_EQ(totals, (std::vector<int64_t>{4, 6, 8}));
    EXPECT_EQ(draft_windows, 1u);
}

}  // namespace
}  // namespace rtp_llm
