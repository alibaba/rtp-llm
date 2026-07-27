#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"

namespace rtp_llm::test {

TEST(RpcServerRuntimeMetaTest, EnqueuePendingReportsPendingPhase) {
    RpcServerRuntimeMeta meta;

    meta.enqueuePending(/*request_id=*/101, /*input_length=*/2048);

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_TRUE(info.finished_task_info_list.empty());
    EXPECT_EQ(info.running_task_info_list[0].request_id, 101);
    EXPECT_EQ(info.running_task_info_list[0].input_length, 2048);
    EXPECT_EQ(info.running_task_info_list[0].prefix_length, 0);
    EXPECT_EQ(info.running_task_info_list[0].phase, TaskPhase::PENDING);
}

TEST(RpcServerRuntimeMetaTest, FinishTaskMovesPendingToFinishedWithErrorDetails) {
    RpcServerRuntimeMeta meta;

    meta.enqueuePending(/*request_id=*/202, /*input_length=*/1024);
    meta.finishTask(/*request_id=*/202,
                    /*input_length=*/1024,
                    /*prefix_length=*/128,
                    /*error_code=*/13,
                    /*error_message=*/"decode alloc failed");

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    const auto& finished = info.finished_task_info_list[0];
    EXPECT_EQ(finished.request_id, 202);
    EXPECT_EQ(finished.input_length, 1024);
    EXPECT_EQ(finished.prefix_length, 128);
    EXPECT_EQ(finished.error_code, 13);
    EXPECT_EQ(finished.error_message, "decode alloc failed");
    EXPECT_GE(info.latest_finished_version, 0);
}

TEST(RpcServerRuntimeMetaTest, FinishTaskWithoutPendingStillReportsFailure) {
    RpcServerRuntimeMeta meta;

    meta.finishTask(/*request_id=*/303,
                    /*input_length=*/512,
                    /*prefix_length=*/0,
                    /*error_code=*/14,
                    /*error_message=*/"remote load failed");

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    const auto& finished = info.finished_task_info_list[0];
    EXPECT_EQ(finished.request_id, 303);
    EXPECT_EQ(finished.input_length, 512);
    EXPECT_EQ(finished.error_code, 14);
    EXPECT_EQ(finished.error_message, "remote load failed");
}

// Simulates a decode early failure after setStream(): the request is already tracked as running,
// then finishTask() must move it to finished with the error details and bump the version.
TEST(RpcServerRuntimeMetaTest, FinishTaskAfterEnqueueClearsRunningAndBumpsVersion) {
    RpcServerRuntimeMeta meta;

    meta.enqueuePending(/*request_id=*/404, /*input_length=*/256);
    auto before = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(before.running_task_info_list.size(), 1);
    ASSERT_TRUE(before.finished_task_info_list.empty());

    meta.finishTask(/*request_id=*/404,
                    /*input_length=*/256,
                    /*prefix_length=*/0,
                    /*error_code=*/602,
                    /*error_message=*/"decode allocate resource failed");

    auto after = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(after.running_task_info_list.empty());
    ASSERT_EQ(after.finished_task_info_list.size(), 1);
    const auto& finished = after.finished_task_info_list[0];
    EXPECT_EQ(finished.request_id, 404);
    EXPECT_EQ(finished.input_length, 256);
    EXPECT_EQ(finished.error_code, 602);
    EXPECT_EQ(finished.error_message, "decode allocate resource failed");
    EXPECT_GT(after.latest_finished_version, before.latest_finished_version);
}

// finishTask() erases the running entry first, so the fallback dequeue in ~GenerateContext()
// finds nothing and returns early: exactly one finished record is reported, no duplicates.
TEST(RpcServerRuntimeMetaTest, DequeueAfterFinishTaskDoesNotDuplicateReport) {
    RpcServerRuntimeMeta meta;

    meta.enqueuePending(/*request_id=*/505, /*input_length=*/640);
    meta.finishTask(/*request_id=*/505,
                    /*input_length=*/640,
                    /*prefix_length=*/0,
                    /*error_code=*/8302,
                    /*error_message=*/"decode load cache from prefill failed");
    // The destructor-path dequeue arrives after finishTask; the running entry is already gone so
    // it must be a no-op (the null stream is never dereferenced on this path).
    meta.dequeue(/*request_id=*/505, /*stream=*/nullptr);

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 505);
    EXPECT_EQ(info.finished_task_info_list[0].error_code, 8302);
    EXPECT_EQ(info.finished_task_info_list[0].error_message, "decode load cache from prefill failed");
}

// Engine execution time is the turnaround (finish - begin) minus the queue wait.
TEST(RpcServerRuntimeMetaTest, ComputeExecutionTimeExcludesQueueWait) {
    // Begin at 1000ms, finish at 1800ms → 800ms turnaround, of which 120ms was queued.
    EXPECT_EQ(RpcServerRuntimeMeta::computeExecutionTimeMs(
                  /*finish_time_ms=*/1800, /*begin_time_us=*/1'000'000, /*waiting_time_ms=*/120),
              680);
    // With no queue wait, execution time equals the full turnaround.
    EXPECT_EQ(RpcServerRuntimeMeta::computeExecutionTimeMs(
                  /*finish_time_ms=*/1800, /*begin_time_us=*/1'000'000, /*waiting_time_ms=*/0),
              800);
}

}  // namespace rtp_llm::test
