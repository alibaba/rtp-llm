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

    // Terminal-state drivers for the finish-promotion tests: completion is
    // event-driven now, so these must drive the real state-machine FINISHED
    // transition through moveToNext() (as the scheduler does each step)
    // instead of writing the status word directly — that transition is what
    // fires the stream's finish callback.
    void forceFinished() {
        generate_status_->status.store(StreamState::RUNNING, std::memory_order_release);
        generate_status_->reportEvent(StreamEvents::GenerateDone);
        moveToNext();
    }

    void forceError(ErrorCode code, const std::string& msg) {
        generate_status_->reportEvent(StreamEvents::Error, code, msg);
        moveToNext();
    }

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

// Event-driven promotion: the finish callback registered by
// PrefillGenerateContext::setStream migrates the runtime-meta entry into
// the finished report at the stream's terminal transition, even while its
// Decode-side fetch()/dequeue() call has not arrived yet — the Master's
// next GetWorkerStatus poll settles the member without waiting for the
// P/D fetch link.
TEST(RpcServerRuntimeMetaTest, FinishCallbackPromotesFinishedStreamWithoutDequeue) {
    RpcServerRuntimeMeta meta;

    auto make_stream = [](int64_t request_id, int64_t batch_id) {
        auto input             = std::make_shared<GenerateInput>();
        input->request_id      = request_id;
        input->group_id        = batch_id;
        input->generate_config = std::make_shared<GenerateConfig>();
        input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
        return std::make_shared<RuntimeMetaTestStream>(input);
    };

    auto running = make_stream(/*request_id=*/501, /*batch_id=*/91);
    meta.enqueue(501, running);
    auto done = make_stream(/*request_id=*/502, /*batch_id=*/91);
    meta.enqueue(502, done);
    // Mirror PrefillGenerateContext::setStream's registration: only the
    // meta, the request id, and a weak stream reference are captured. The
    // callback fires synchronously inside forceFinished(), while the stack
    // meta object is still alive.
    std::weak_ptr<GenerateStream> weak_done = done;
    done->setFinishCallback([&meta, rid = int64_t{502}, weak_done]() {
        if (auto finished = weak_done.lock()) {
            meta.dequeue(rid, finished);
        }
    });
    done->forceFinished();

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].request_id, 501);
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 502);
    EXPECT_EQ(info.finished_task_info_list[0].batch_id, 91);
    EXPECT_EQ(info.finished_task_info_list[0].error_code, 0);

    // Incremental semantics: replaying with the returned version marker
    // reports nothing new; the promoted member does not duplicate.
    auto next = meta.getEngineScheduleInfo(info.latest_finished_version);
    EXPECT_EQ(next.running_task_info_list.size(), 1);
    EXPECT_TRUE(next.finished_task_info_list.empty());

    // A late fetch-driven dequeue of the already-promoted member is an
    // idempotent no-op: it must not publish a second finished record.
    meta.dequeue(502, done);
    auto after = meta.getEngineScheduleInfo(next.latest_finished_version);
    EXPECT_TRUE(after.finished_task_info_list.empty());
}

// Error-terminal promotion: the finish callback fires on an errored stream's
// FINISHED transition too (the Error event converges to FINISHED in
// moveToNext()), and the finished record carries the stream's error_code.
TEST(RpcServerRuntimeMetaTest, FinishCallbackPromotesErroredStreamWithErrorCode) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 503;
    input->group_id            = 92;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto failed                = std::make_shared<RuntimeMetaTestStream>(input);
    meta.enqueue(503, failed);
    std::weak_ptr<GenerateStream> weak_failed = failed;
    failed->setFinishCallback([&meta, rid = int64_t{503}, weak_failed]() {
        if (auto finished = weak_failed.lock()) {
            meta.dequeue(rid, finished);
        }
    });
    failed->forceError(ErrorCode::LONG_PROMPT_ERROR, "prompt exceeds context window");

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 503);
    EXPECT_EQ(info.finished_task_info_list[0].batch_id, 92);
    EXPECT_EQ(info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::LONG_PROMPT_ERROR));
}

}  // namespace rtp_llm::test
