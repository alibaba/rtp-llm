#include <gtest/gtest.h>

#include <thread>

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

    // The running record and Cancel overlay are constructed from the same
    // request identity, so decoration cannot change its batch id.
    meta.markPriorityPreemptionCanceling(TaskIdentity{input->request_id, /*batch_id=*/77});

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].request_id, input->request_id);
    EXPECT_EQ(info.running_task_info_list[0].batch_id, 77);
    EXPECT_EQ(info.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);

    ASSERT_TRUE(meta.markPriorityPreemptionCanceled(
        input->request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted"));
    auto canceled = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceled.finished_task_info_list.size(), 1);
    EXPECT_EQ(canceled.finished_task_info_list[0].batch_id, 77);
}

TEST(RpcServerRuntimeMetaTest, PriorityCanceledIsPublishedOnceAndClearsControlOverlay) {
    RpcServerRuntimeMeta meta;
    meta.markPriorityPreemptionCanceling(TaskIdentity{/*request_id=*/405, /*batch_id=*/-1});

    auto canceling = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, -1);

    EXPECT_TRUE(meta.markPriorityPreemptionCanceled(
        /*request_id=*/405, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted"));
    EXPECT_FALSE(meta.markPriorityPreemptionCanceled(
        /*request_id=*/405, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "duplicate"));

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
        input->request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted"));

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
            identity.request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted"));
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
        input->request_id, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "priority preempted"));
    auto canceled = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(canceled.running_task_info_list.empty());
    ASSERT_EQ(canceled.finished_task_info_list.size(), 1);
    EXPECT_EQ(canceled.finished_task_info_list[0].batch_id, 89);
    EXPECT_EQ(canceled.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(canceled.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
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

namespace {

std::shared_ptr<RuntimeMetaTestStream> makeBareStream(int64_t request_id) {
    auto input             = std::make_shared<GenerateInput>();
    input->request_id      = request_id;
    input->group_id        = 77;
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
    return std::make_shared<RuntimeMetaTestStream>(input);
}

}  // namespace

// Engine contract defaults: a stream without allocated KV reports
// kv_tokens = 0 and the running detail is complete — this is exactly the
// legacy-engine wire shape (field absent), so the Master-side default path
// must be indistinguishable from before.
TEST(RpcServerRuntimeMetaTest, RunningDetailDefaultsToUntruncatedWithZeroKvTokens) {
    RpcServerRuntimeMeta meta;
    meta.enqueue(TaskIdentity{/*request_id=*/601, /*batch_id=*/77}, makeBareStream(601));

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].kv_tokens, 0);
    EXPECT_FALSE(info.running_detail_truncated);
}

// Running detail below the cap is never truncated.
TEST(RpcServerRuntimeMetaTest, RunningDetailNotTruncatedBelowCap) {
    RpcServerRuntimeMeta meta;
    auto                 stream = makeBareStream(602);
    for (int i = 0; i < 3; ++i) {
        meta.enqueue(TaskIdentity{602 + i, /*batch_id=*/77}, stream);
    }

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_EQ(info.running_task_info_list.size(), 3);
    EXPECT_FALSE(info.running_detail_truncated);
}

// Exceeding the reporting cap truncates the running list and sets the
// completeness flag, so consumers can tell truncation-driven absence apart
// from actual completion.
TEST(RpcServerRuntimeMetaTest, RunningDetailTruncatedAtCap) {
    RpcServerRuntimeMeta meta;
    auto                 stream  = makeBareStream(603);
    const size_t         entries = RpcServerRuntimeMeta::kMaxRunningTaskDetailEntries + 1;
    for (size_t i = 0; i < entries; ++i) {
        meta.enqueue(TaskIdentity{static_cast<int64_t>(7000 + i), /*batch_id=*/77}, stream);
    }

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_EQ(info.running_task_info_list.size(), RpcServerRuntimeMeta::kMaxRunningTaskDetailEntries);
    EXPECT_TRUE(info.running_detail_truncated);
}

// Terminal delivery is state-based: a Master that lost an increment (cursor
// still behind) re-receives the same terminals from the finished window on
// the next pull; once its cursor advances past them, replay stops.
TEST(RpcServerRuntimeMetaTest, FinishedWindowReplaysTerminalsUntilCursorAdvances) {
    RpcServerRuntimeMeta meta;
    meta.finishTask(/*request_id=*/801, /*input_length=*/128);
    meta.finishTask(/*request_id=*/802, /*input_length=*/256);

    // First successful sync observes both terminals and the new cursor.
    auto       first    = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    const auto advanced = first.latest_finished_version;
    ASSERT_EQ(first.finished_task_info_list.size(), 2);

    // The Master lost that increment: with its cursor still at -1 the next
    // pull replays both terminals from the window — not one-shot delivery.
    auto replay = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(replay.finished_task_info_list.size(), 2);
    EXPECT_EQ(replay.latest_finished_version, advanced);

    // Durably advanced cursor: the window stops replaying.
    auto settled = meta.getEngineScheduleInfo(advanced);
    EXPECT_TRUE(settled.finished_task_info_list.empty());
}

}  // namespace rtp_llm::test
