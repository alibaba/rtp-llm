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

}  // namespace rtp_llm::test
