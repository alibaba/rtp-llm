#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"

namespace rtp_llm::test {

namespace {

class RuntimeMetaTestStream: public GenerateStream {
public:
    explicit RuntimeMetaTestStream(const std::shared_ptr<GenerateInput>& input):
        GenerateStream(input, modelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}

    ErrorResult<GenerateOutputs> nextOutput() override {
        return ErrorResult<GenerateOutputs>(GenerateOutputs{});
    }

    void updateOutput(const StreamUpdateInfo&) override {}

private:
    static ModelConfig modelConfig() {
        ModelConfig config;
        config.max_seq_len = 4096;
        return config;
    }
};

}  // namespace

TEST(RpcServerRuntimeMetaTest, EnqueueReadsBatchIdFromStreamInput) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 101;
    input->group_id            = 77;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);

    meta.enqueue(input->request_id, stream);

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].request_id, 101);
    EXPECT_EQ(info.running_task_info_list[0].batch_id, 77);
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

TEST(RpcServerRuntimeMetaTest, PriorityCancelDecoratesExistingTaskWithoutDuplicateRuntimeEntry) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 404;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    meta.enqueue(input->request_id, stream);

    meta.markPriorityPreemptionCanceling(input->request_id);

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].request_id, input->request_id);
    EXPECT_EQ(info.running_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELING);
}

TEST(RpcServerRuntimeMetaTest, PriorityCanceledIsPublishedOnceAndClearsControlOverlay) {
    RpcServerRuntimeMeta meta;
    meta.markPriorityPreemptionCanceling(/*request_id=*/405);

    EXPECT_TRUE(meta.markPriorityPreemptionCanceled(
        /*request_id=*/405, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted"));
    EXPECT_FALSE(meta.markPriorityPreemptionCanceled(
        /*request_id=*/405, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "duplicate"));

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 405);
    EXPECT_EQ(info.finished_task_info_list[0].error_code,
              static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(info.finished_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELED);
}

TEST(RpcServerRuntimeMetaTest, PriorityCanceledReplacesRunningTaskWithSingleTypedFinishedRecord) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 406;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    meta.enqueue(input->request_id, stream);
    meta.markPriorityPreemptionCanceling(input->request_id);

    ASSERT_TRUE(meta.markPriorityPreemptionCanceled(
        input->request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted"));

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, input->request_id);
    EXPECT_EQ(info.finished_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(info.finished_task_info_list[0].error_code,
              static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(RpcServerRuntimeMetaTest, OrdinaryDequeueCannotRegressPriorityCancelingToUntypedFinished) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 407;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    meta.enqueue(input->request_id, stream);
    meta.markPriorityPreemptionCanceling(input->request_id);

    meta.dequeue(input->request_id, stream);

    auto canceling = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_TRUE(canceling.finished_task_info_list.empty());
    EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELING);

    ASSERT_TRUE(meta.markPriorityPreemptionCanceled(
        input->request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted"));
    auto canceled = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(canceled.running_task_info_list.empty());
    ASSERT_EQ(canceled.finished_task_info_list.size(), 1);
    EXPECT_EQ(canceled.finished_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(canceled.finished_task_info_list[0].error_code,
              static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
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
