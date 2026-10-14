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
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

class ExpertLoadMonitorTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "This test requires the GPU assigned by gpu_lock";
        device_guard_         = std::make_unique<c10::DeviceGuard>(c10::Device(c10::kCUDA, 0));
        const char* temporary = std::getenv("TEST_TMPDIR");
        ASSERT_NE(temporary, nullptr);
        directory_ = std::filesystem::path(temporary) / ::testing::UnitTest::GetInstance()->current_test_info()->name();
        std::filesystem::remove_all(directory_);
        std::filesystem::create_directories(directory_);
        config_.eplb_metrics_report_interval    = 2;
        config_.eplb_metrics_snapshot_dir       = directory_.string();
        config_.eplb_metrics_snapshot_max_files = 10;
    }

    std::unique_ptr<ExpertLoadMonitor> monitor() {
        return std::make_unique<ExpertLoadMonitor>(1, 2, ParallelismConfig{}, config_, 0, nullptr);
    }

    void step(ExpertLoadMonitor& monitor, int64_t left, int64_t right) {
        monitor.beginStep(false);
        monitor.stepCounts().select(1, 0).fill_(left);
        monitor.stepCounts().select(1, 1).fill_(right);
        monitor.finishStep();
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

    void expectWindow(const rapidjson::Document& document,
                      uint64_t                   begin,
                      int64_t                    left,
                      int64_t                    right,
                      int64_t                    first,
                      int64_t                    second) {
        EXPECT_EQ(document["step_begin"].GetUint64(), begin);
        EXPECT_EQ(document["step_end"].GetUint64(), begin + 1);
        EXPECT_EQ(document["sample_count"].GetUint64(), 2u);
        EXPECT_EQ(document["world_size"].GetUint64(), 1u);
        EXPECT_EQ(document["ep_size"].GetUint64(), 1u);
        ASSERT_EQ(document["physical_expert_activations"].Size(), 1u);
        const auto& experts = document["physical_expert_activations"][0];
        ASSERT_EQ(experts.Size(), 2u);
        EXPECT_EQ(experts[0].GetInt64(), left);
        EXPECT_EQ(experts[1].GetInt64(), right);
        ASSERT_EQ(document["rank_history"].Size(), 2u);
        EXPECT_EQ(document["rank_history"][0][0][0].GetInt64(), first);
        EXPECT_EQ(document["rank_history"][1][0][0].GetInt64(), second);
        ASSERT_EQ(document["layers"].Size(), 1u);
        const auto& layer = document["layers"][0];
        EXPECT_EQ(layer["total_activations"].GetInt64(), first + second);
        EXPECT_EQ(layer["rank_activations"][0].GetInt64(), first + second);
        EXPECT_EQ(layer["valid"].GetBool(), first + second > 0);
        if (first + second > 0) {
            EXPECT_DOUBLE_EQ(layer["moe_lb"].GetDouble(), 1.0);
            EXPECT_DOUBLE_EQ(layer["moe_lb_window"].GetDouble(), 1.0);
        } else {
            EXPECT_TRUE(layer["moe_lb"].IsNull());
            EXPECT_TRUE(layer["moe_lb_window"].IsNull());
        }
    }

    std::unique_ptr<c10::DeviceGuard> device_guard_;
    std::filesystem::path             directory_;
    EPLBConfig                        config_;
};

TEST_F(ExpertLoadMonitorTest, WindowsRotateAndZeroWindowHasNoStaleLoad) {
    auto recorder = monitor();
    step(*recorder, 4, 0);
    EXPECT_TRUE(snapshots().empty());
    step(*recorder, 0, 4);
    step(*recorder, 0, 0);
    ASSERT_EQ(snapshots().size(), 1u);
    step(*recorder, 0, 0);
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 2u);
    expectWindow(windows[0], 1, 4, 4, 4, 4);
    expectWindow(windows[1], 3, 0, 0, 0, 0);
}

TEST_F(ExpertLoadMonitorTest, WarmupAndAbortDoNotConsumeProductionSteps) {
    auto recorder = monitor();
    step(*recorder, 1, 2);
    recorder->beginStep(true);
    recorder->stepCounts().fill_(99);
    recorder->finishStep();
    recorder->beginStep(false);
    recorder->stepCounts().fill_(123);
    recorder->abortStep();
    EXPECT_TRUE(snapshots().empty());
    step(*recorder, 3, 4);
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    expectWindow(windows[0], 1, 4, 6, 3, 7);
}

TEST_F(ExpertLoadMonitorTest, FailedFlushDiscardsWindowAndNextWindowRecovers) {
    auto recorder = monitor();
    step(*recorder, 4, 4);
    recorder->beginStep(false);
    recorder->stepCounts().fill_(-1);
    ASSERT_THROW(recorder->finishStep(), std::invalid_argument);
    // Match the production forward guard after finishStep propagates failure.
    recorder->abortStep();
    EXPECT_TRUE(snapshots().empty());
    step(*recorder, 2, 2);
    EXPECT_TRUE(snapshots().empty());
    step(*recorder, 3, 3);
    const auto windows = snapshots();
    ASSERT_EQ(windows.size(), 1u);
    expectWindow(windows[0], 3, 5, 5, 4, 6);
}

}  // namespace
}  // namespace rtp_llm
