#include <gtest/gtest.h>

#include <thread>

#include "autil/TimeUtility.h"
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

// P1 ledger backstop: an aged CANCELING overlay is swept with a typed CANCELED
// finished record (8429) and disappears from the running projection; a fresh
// overlay is never swept; the late finalizer stays idempotent afterwards.
TEST(RpcServerRuntimeMetaTest, SweepStalePriorityOverlayPublishesTypedCanceledAndClearsOverlay) {
    RpcServerRuntimeMeta meta;
    const TaskIdentity   identity{/*request_id=*/501, /*batch_id=*/91};
    meta.markPriorityPreemptionCanceling(identity);

    const int64_t now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    // Fresh overlay: below the age threshold, nothing is swept.
    EXPECT_EQ(meta.sweepStalePriorityOverlays(now_ms, /*max_age_ms=*/300000), 0u);
    auto fresh = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(fresh.running_task_info_list.size(), 1);
    EXPECT_EQ(fresh.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);
    EXPECT_TRUE(fresh.finished_task_info_list.empty());

    // Aged overlay: swept once, typed CANCELED published, overlay gone.
    EXPECT_EQ(meta.sweepStalePriorityOverlays(now_ms + 300001, /*max_age_ms=*/300000), 1u);
    // Second sweep is a no-op (nothing left to age out).
    EXPECT_EQ(meta.sweepStalePriorityOverlays(now_ms + 300002, /*max_age_ms=*/300000), 0u);

    auto swept = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(swept.running_task_info_list.empty());
    ASSERT_EQ(swept.finished_task_info_list.size(), 1);
    EXPECT_EQ(swept.finished_task_info_list[0].request_id, 501);
    EXPECT_EQ(swept.finished_task_info_list[0].batch_id, 91);
    EXPECT_EQ(swept.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(swept.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);

    // Idempotent with the finalizer chain: no overlay left, so a late
    // markPriorityPreemptionCanceled keeps its "not found -> false" semantics.
    EXPECT_FALSE(meta.markPriorityPreemptionCanceled(
        501, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "late finalizer"));
    auto after_late = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(after_late.finished_task_info_list.size(), 1);
}

// The sweep must never remove a live stream's running entry: the typed CANCELED
// record is published for the control ledger, while the live runtime entry
// stays for its own teardown path to close.
TEST(RpcServerRuntimeMetaTest, SweepStaleOverlayKeepsLiveRunningStreamEntry) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 502;
    input->group_id            = 92;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    // RuntimeMetaTestStream stays in its initial WAITING state: a live stream.

    const TaskIdentity identity{502, 92};
    meta.markPriorityPreemptionCanceling(identity);
    meta.enqueue(identity, stream);

    auto canceling = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);

    const int64_t now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    ASSERT_EQ(meta.sweepStalePriorityOverlays(now_ms + 300001, /*max_age_ms=*/300000), 1u);

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    // Ledger side: exactly one typed CANCELED finished record.
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 502);
    EXPECT_EQ(info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(info.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
    // Live stream side: the running entry survives, no longer CANCELING.
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].request_id, 502);
    EXPECT_EQ(info.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::NONE);
}

// The automatic entry point (getEngineScheduleInfo) must not sweep fresh
// overlays with the production default threshold, and a disabled sweep
// (max_age <= 0) never removes anything.
TEST(RpcServerRuntimeMetaTest, GetEngineScheduleInfoKeepsFreshOverlayAndHonorsDisabledSweep) {
    RpcServerRuntimeMeta meta;
    const TaskIdentity   identity{/*request_id=*/503, /*batch_id=*/93};
    meta.markPriorityPreemptionCanceling(identity);

    // Automatic path with the production threshold: the overlay is fresh, so
    // repeated snapshots keep the CANCELING control record.
    for (int i = 0; i < 3; ++i) {
        auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
        ASSERT_EQ(info.running_task_info_list.size(), 1);
        EXPECT_EQ(info.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);
    }

    // Disabled sweep (<=0) must not touch even a logically-aged overlay.
    const int64_t now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    EXPECT_EQ(meta.sweepStalePriorityOverlays(now_ms + 10'000'000, /*max_age_ms=*/0), 0u);
    EXPECT_EQ(meta.sweepStalePriorityOverlays(now_ms + 10'000'000, /*max_age_ms=*/-1), 0u);
    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(info.running_task_info_list.size(), 1);
    EXPECT_EQ(info.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);
}

// R3/S-1 single-record invariant: after the sweep publishes the typed CANCELED
// record while preserving a LIVE stream's running entry, that stream's own
// ordinary dequeue() must remove the entry WITHOUT emitting a second untyped
// finished record.
TEST(RpcServerRuntimeMetaTest, SweepKeepLiveThenOrdinaryDequeuePublishesSingleRecord) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 510;
    input->group_id            = 94;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    // RuntimeMetaTestStream stays WAITING: a live stream the sweep must keep.

    const TaskIdentity identity{510, 94};
    meta.markPriorityPreemptionCanceling(identity);
    meta.enqueue(identity, stream);

    const int64_t now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    ASSERT_EQ(meta.sweepStalePriorityOverlays(now_ms + 300001, /*max_age_ms=*/300000), 1u);

    // One typed CANCELED record from the sweep; the live entry survives.
    auto after_sweep = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(after_sweep.finished_task_info_list.size(), 1);
    ASSERT_EQ(after_sweep.running_task_info_list.size(), 1);

    // The live stream finishes normally and tears down via ordinary dequeue:
    // entry removed, and still exactly ONE finished record (no untyped dupe).
    meta.dequeue(input->request_id, stream);

    auto final_info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(final_info.running_task_info_list.empty());
    ASSERT_EQ(final_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(final_info.finished_task_info_list[0].request_id, 510);
    EXPECT_EQ(final_info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(final_info.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
}

// R2/P1-1 defect A: sweep consumes the overlay (keep_live preserves the running
// entry), then the LATE finalizer arrives. markPriorityPreemptionCanceled must
// clean that leftover entry (true) instead of leaking it forever — and must
// not publish a second record (the sweep already published the typed one).
TEST(RpcServerRuntimeMetaTest, LateFinalizerAfterSweepCleansLeftoverRunningEntry) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 511;
    input->group_id            = 95;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);

    const TaskIdentity identity{511, 95};
    meta.markPriorityPreemptionCanceling(identity);
    meta.enqueue(identity, stream);

    const int64_t now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    ASSERT_EQ(meta.sweepStalePriorityOverlays(now_ms + 300001, /*max_age_ms=*/300000), 1u);

    // Late finalizer: overlay gone, leftover live entry present -> cleaned.
    EXPECT_TRUE(meta.markPriorityPreemptionCanceled(
        511, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "late finalizer"));

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.running_task_info_list.empty());
    // Single record invariant holds: exactly the sweep's typed CANCELED record.
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 511);
    EXPECT_EQ(info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

// R2 defensive branch: overlay missing WITHOUT the sweep's canceled_published
// latch (no sweep ever ran — the entry was never priority-cancel-published).
// The entry must still not leak: it is closed with exactly one typed record.
TEST(RpcServerRuntimeMetaTest, MarkCanceledWithoutOverlayStillCleansRunningEntry) {
    RpcServerRuntimeMeta meta;
    auto                 input = std::make_shared<GenerateInput>();
    input->request_id          = 512;
    input->group_id            = 96;
    input->generate_config     = std::make_shared<GenerateConfig>();
    input->input_ids           = torch::tensor({1, 2, 3}, torch::kInt32);
    auto stream                = std::make_shared<RuntimeMetaTestStream>(input);
    // No markPriorityPreemptionCanceling: running entry exists, no overlay,
    // no canceled_published latch — the unreachable-in-practice corner.

    meta.enqueue(input->request_id, stream);

    EXPECT_TRUE(
        meta.markPriorityPreemptionCanceled(512, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "defensive"));

    auto info = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 512);
    EXPECT_EQ(info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(info.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);

    // Fully closed afterwards: nothing left for a second call to do.
    EXPECT_FALSE(
        meta.markPriorityPreemptionCanceled(512, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED), "second"));
}

}  // namespace rtp_llm::test
