#include <gtest/gtest.h>

#include <thread>

#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"

namespace rtp_llm::test {

namespace {

class RuntimeMetaTestStream: public GenerateStream {
public:
    explicit RuntimeMetaTestStream(const std::shared_ptr<GenerateInput>& input):
        GenerateStream(input, modelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}

    ErrorResult<GenerateOutputs> nextOutput(int64_t wait_timeout_ms = 0) override {
        (void)wait_timeout_ms;
        return ErrorResult<GenerateOutputs>(GenerateOutputs{});
    }

    void updateOutput(const StreamUpdateInfo&) override {}

    void setWaitTimeUsForTest(int64_t wait_time_us) {
        std::lock_guard<std::mutex> lock(*mutex_);
        wait_time_us_ = wait_time_us;
    }

private:
    static ModelConfig modelConfig() {
        ModelConfig config;
        config.max_seq_len = 4096;
        return config;
    }
};

class TestableRpcServerRuntimeMeta: public RpcServerRuntimeMeta {
public:
    void replaceBeforeCommittingDequeueSnapshot(int64_t                  request_id,
                                                const GenerateStreamPtr& stale_stream,
                                                const GenerateStreamPtr& replacement_stream) {
        const auto stream_snapshot = captureStreamRuntimeSnapshot(stale_stream);
        enqueue(request_id, replacement_stream);
        commitDequeueSnapshot(request_id, stale_stream, stream_snapshot);
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

TEST(RpcServerRuntimeMetaTest, EnqueueConvertsWaitTimeFromMicrosecondsToMilliseconds) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 103;
    input->group_id            = 78;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    stream->setWaitTimeUsForTest(123'456);

    meta.enqueue(input->request_id, stream);

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].waiting_time_ms, 123);
}

TEST(RpcServerRuntimeMetaTest, EnqueueKeepsEnvelopeBatchIdOnStreamMismatch) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 102;
    input->group_id            = 77;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);

    meta.enqueue(TaskIdentity{input->request_id, /*envelope batch_id=*/99}, stream);

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].request_id, 102);
    EXPECT_EQ(info.running_task_info_list[0].batch_id, 99);
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
    input->group_id            = 77;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    meta.enqueue(TaskIdentity{input->request_id, input->group_id}, stream);
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 1'000'000);
    stream->setWaitTimeUsForTest(456'789);

    // The running record and Cancel overlay are constructed from the same
    // request identity, so decoration cannot change its batch id.
    meta.markPriorityPreemptionCanceling(TaskIdentity{input->request_id, /*batch_id=*/77});

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].request_id, input->request_id);
    EXPECT_EQ(info.running_task_info_list[0].batch_id, 77);
    EXPECT_EQ(info.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);

    ASSERT_TRUE(meta.markPriorityPreemptionCanceled(
        input->request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted", stream));
    auto canceled = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceled.finished_task_info_list.size(), 1);
    EXPECT_EQ(canceled.finished_task_info_list[0].batch_id, 77);
    EXPECT_EQ(canceled.finished_task_info_list[0].waiting_time_ms, 456);
}

TEST(RpcServerRuntimeMetaTest, PriorityCanceledIsPublishedOnceAndClearsControlOverlay) {
    RpcServerRuntimeMeta meta;
    meta.markPriorityPreemptionCanceling(TaskIdentity{/*request_id=*/405, /*batch_id=*/-1});

    auto canceling = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, -1);

    EXPECT_TRUE(meta.markPriorityPreemptionCanceled(
        /*request_id=*/405, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted", nullptr));
    EXPECT_FALSE(meta.markPriorityPreemptionCanceled(
        /*request_id=*/405, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "duplicate", nullptr));

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 405);
    EXPECT_EQ(info.finished_task_info_list[0].batch_id, -1);
    EXPECT_EQ(info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(info.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
}

TEST(RpcServerRuntimeMetaTest, PriorityCanceledReplacesRunningTaskWithSingleTypedFinishedRecord) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 406;
    input->group_id            = 88;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    // Cancel can win before the local stream is enqueued. Both records are
    // created from the same immutable request identity.
    const TaskIdentity identity{input->request_id, input->group_id};
    meta.markPriorityPreemptionCanceling(identity);
    meta.enqueue(identity, stream);

    auto canceling = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, 88);

    ASSERT_TRUE(meta.markPriorityPreemptionCanceled(
        input->request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted", stream));

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, input->request_id);
    EXPECT_EQ(info.finished_task_info_list[0].batch_id, 88);
    EXPECT_EQ(info.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(RpcServerRuntimeMetaTest, ConcurrentEarlyCancelAndEnqueueKeepOneBatchIdentity) {
    constexpr int64_t request_id = 408;
    constexpr int64_t batch_id   = 90;

    for (int attempt = 0; attempt < 16; ++attempt) {
        RpcServerRuntimeMeta meta;
        auto                 input = std::make_shared<GenerateInput>();
        input->request_id          = request_id + attempt;
        input->group_id            = batch_id + attempt;
        input->generate_config     = std::make_shared<GenerateConfig>();
        input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
        auto stream                = std::make_shared<RuntimeMetaTestStream>(input);

        const TaskIdentity identity{input->request_id, input->group_id};
        std::atomic<int>   ready{0};
        std::atomic<bool>  start{false};
        const auto         await_start = [&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        };
        std::thread cancel_thread([&]() {
            await_start();
            meta.markPriorityPreemptionCanceling(identity);
        });
        std::thread enqueue_thread([&]() {
            await_start();
            meta.enqueue(identity, stream);
        });
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        cancel_thread.join();
        enqueue_thread.join();

        auto canceling = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
        ASSERT_EQ(canceling.running_task_info_list.size(), 1) << "attempt=" << attempt;
        EXPECT_EQ(canceling.running_task_info_list[0].request_id, identity.request_id) << "attempt=" << attempt;
        EXPECT_EQ(canceling.running_task_info_list[0].batch_id, identity.batch_id) << "attempt=" << attempt;
        EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress,
                  PriorityPreemptionProgress::CANCELING)
            << "attempt=" << attempt;

        ASSERT_TRUE(meta.markPriorityPreemptionCanceled(
            identity.request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted", stream));
        auto canceled = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
        ASSERT_EQ(canceled.finished_task_info_list.size(), 1) << "attempt=" << attempt;
        EXPECT_EQ(canceled.finished_task_info_list[0].batch_id, identity.batch_id) << "attempt=" << attempt;
        EXPECT_EQ(canceled.finished_task_info_list[0].priority_preemption_progress,
                  PriorityPreemptionProgress::CANCELED)
            << "attempt=" << attempt;
    }
}

TEST(RpcServerRuntimeMetaTest, OrdinaryDequeueCannotRegressPriorityCancelingToUntypedFinished) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 407;
    input->group_id            = 89;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    // Exercise the full early-Cancel -> late enqueue -> ordinary dequeue ->
    // typed CANCELED sequence with one immutable request identity.
    const TaskIdentity identity{input->request_id, input->group_id};
    meta.markPriorityPreemptionCanceling(identity);
    meta.enqueue(identity, stream);

    auto enqueued = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(enqueued.running_task_info_list.size(), 1);
    EXPECT_EQ(enqueued.running_task_info_list[0].batch_id, 89);

    meta.dequeue(input->request_id, stream);

    auto canceling = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_TRUE(canceling.finished_task_info_list.empty());
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, 89);
    EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);

    ASSERT_TRUE(meta.markPriorityPreemptionCanceled(
        input->request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted", stream));
    auto canceled = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(canceled.running_task_info_list.empty());
    ASSERT_EQ(canceled.finished_task_info_list.size(), 1);
    EXPECT_EQ(canceled.finished_task_info_list[0].batch_id, 89);
    EXPECT_EQ(canceled.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(canceled.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(RpcServerRuntimeMetaTest, SnapshotCommitDoesNotRemoveReplacementStream) {
    TestableRpcServerRuntimeMeta meta;
    auto                         stale_input = std::make_shared<GenerateInput>();
    stale_input->request_id                  = 409;
    stale_input->group_id                    = 91;
    stale_input->generate_config             = std::make_shared<GenerateConfig>();
    stale_input->input_ids                   = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stale_stream                        = std::make_shared<RuntimeMetaTestStream>(stale_input);

    auto replacement_input             = std::make_shared<GenerateInput>();
    replacement_input->request_id      = stale_input->request_id;
    replacement_input->group_id        = 92;
    replacement_input->generate_config = std::make_shared<GenerateConfig>();
    replacement_input->input_ids       = torch::tensor({4, 5, 6}, torch::kInt32);
    auto replacement_stream            = std::make_shared<RuntimeMetaTestStream>(replacement_input);

    meta.enqueue(stale_input->request_id, stale_stream);
    meta.replaceBeforeCommittingDequeueSnapshot(stale_input->request_id, stale_stream, replacement_stream);

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.finished_task_info_list.empty());
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].batch_id, replacement_input->group_id);
}

TEST(RpcServerRuntimeMetaTest, StalePriorityFinalizerDoesNotRemoveReplacementStream) {
    RpcServerRuntimeMeta meta;
    auto                 stale_input = std::make_shared<GenerateInput>();
    stale_input->request_id          = 410;
    stale_input->group_id            = 93;
    stale_input->generate_config     = std::make_shared<GenerateConfig>();
    stale_input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stale_stream                = std::make_shared<RuntimeMetaTestStream>(stale_input);

    auto replacement_input             = std::make_shared<GenerateInput>();
    replacement_input->request_id      = stale_input->request_id;
    replacement_input->group_id        = 94;
    replacement_input->generate_config = std::make_shared<GenerateConfig>();
    replacement_input->input_ids       = torch::tensor({4, 5, 6}, torch::kInt32);
    auto replacement_stream            = std::make_shared<RuntimeMetaTestStream>(replacement_input);

    meta.enqueue(stale_input->request_id, stale_stream);
    meta.markPriorityPreemptionCanceling(TaskIdentity{stale_input->request_id, stale_input->group_id});
    meta.enqueue(replacement_input->request_id, replacement_stream);

    EXPECT_TRUE(meta.markPriorityPreemptionCanceled(stale_input->request_id,
                                                    static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED),
                                                    "stale priority preemption",
                                                    stale_stream));

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].batch_id, stale_input->group_id);
    EXPECT_EQ(info.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].batch_id, replacement_input->group_id);
    EXPECT_EQ(info.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::NONE);
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
