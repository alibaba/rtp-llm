#include <memory>
#include <future>
#include <chrono>
#include "torch/all.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/engine_base/schedulers/GatherBatchScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class GatherBatchSchedulerTest: public DeviceTestBase {
public:
    GatherBatchSchedulerTest() {}

protected:
    // Build a scheduler with enough KV blocks to admit a few short streams. We do not need any
    // particular cache layout — we only care about the gather-vs-defer scheduling decision.
    std::shared_ptr<GatherBatchScheduler> makeScheduler(bool load_python_model) {
        cache_config_  = makeMhaCacheConfig(1, 32, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
        EXPECT_TRUE(cache_manager_->init());
        resource_context_.cache_manager = cache_manager_;

        model_config_.max_seq_len                                   = 8192;
        runtime_config_.max_generate_batch_size                     = 100;
        runtime_config_.fifo_scheduler_config.max_batch_tokens_size = 8192;
        model_specific_config_.load_python_model                    = load_python_model;

        return std::make_shared<GatherBatchScheduler>(runtime_config_,
                                                      model_config_,
                                                      pd_sep_config_,
                                                      parallelism_config_,
                                                      model_specific_config_,
                                                      cache_manager_,
                                                      /*metrics_reporter=*/nullptr);
    }

    std::shared_ptr<GenerateStream> makeStream(int max_new_tokens = 8192, int64_t request_id = 0) {
        auto query                             = std::make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
        query->generate_config                 = std::make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        query->request_id                      = request_id;
        return std::make_shared<NormalGenerateStream>(
            query, model_config_, runtime_config_, resource_context_, nullptr);
    }

    void setGatherBatchSize(GatherBatchScheduler& scheduler, int batch_size) {
        // GatherBatchScheduler parses {"batch_size": N} JSON into gather_batch_size_.
        scheduler.updateSchedulerInfo(R"({"batch_size":)" + std::to_string(batch_size) + "}");
    }

    CacheConfig                     cache_config_;
    std::shared_ptr<KVCacheManager> cache_manager_;
    ResourceContext                 resource_context_;
    ModelConfig                     model_config_;
    RuntimeConfig                   runtime_config_;
    PDSepConfig                     pd_sep_config_;
    ParallelismConfig               parallelism_config_;
    ModelSpecificConfig             model_specific_config_;
};

// Baseline: a single stream is gathered into running on the first schedule call.
TEST_F(GatherBatchSchedulerTest, testSingleStreamGather) {
    auto scheduler = makeScheduler(/*load_python_model=*/false);
    auto stream    = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    auto streams_status = scheduler->schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

// updateSchedulerInfo({"batch_size":N}) makes the scheduler wait until N streams accumulate
// before gathering them all in one shot. With only one stream waiting, no gather happens.
TEST_F(GatherBatchSchedulerTest, testGatherBatchAccumulatesBeforeRunning) {
    auto scheduler = makeScheduler(/*load_python_model=*/false);
    setGatherBatchSize(*scheduler, 2);

    auto stream1 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());

    // Only one stream waiting — gather threshold (2) not yet met, so nothing should run yet.
    auto streams_status1 = scheduler->schedule();
    ASSERT_TRUE(streams_status1.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);

    auto stream2 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());

    // Threshold reached: both streams gathered into running together.
    auto streams_status2 = scheduler->schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 2);
}

TEST_F(GatherBatchSchedulerTest, testEnqueueGroupReturnsUnsupported) {
    auto                           scheduler = makeScheduler(/*load_python_model=*/true);
    std::vector<GenerateStreamPtr> streams   = {makeStream(), makeStream()};

    auto [enqueue_successes, returned_streams] = scheduler->enqueueGroup(streams);
    ASSERT_EQ(enqueue_successes, std::vector<bool>({false, false}));
    ASSERT_EQ(returned_streams, streams);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    for (const auto& stream : streams) {
        ASSERT_TRUE(stream->hasError());
        ASSERT_EQ(stream->statusInfo().code(), ErrorCode::UNKNOWN_ERROR);
    }
}

// The load_python_model flag no longer changes scheduling behavior. Even when callers pass
// false, a new prefill must wait until the running batch drains.
TEST_F(GatherBatchSchedulerTest, testRemovedPyModelFlagStillDefersGatherWhileRunning) {
    auto scheduler = makeScheduler(/*load_python_model=*/false);

    auto stream1 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    auto streams_status1 = scheduler->schedule();
    ASSERT_TRUE(streams_status1.ok());
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);

    // Stream1 stays in running (still "decoding"). Enqueue a second stream and re-schedule.
    auto stream2 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());

    auto streams_status2 = scheduler->schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

// load_python_model=true with empty running is the happy path — the guard does not fire and
// the new stream is gathered immediately.
TEST_F(GatherBatchSchedulerTest, testPyModelGuardAllowsGatherWhenRunningEmpty) {
    auto scheduler = makeScheduler(/*load_python_model=*/true);

    auto stream = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    auto streams_status = scheduler->schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

// The core regression test for the fix: with load_python_model=true and a stream already
// running (i.e. potentially decoding), a freshly enqueued stream must NOT be gathered. Mixing
// a context (prefill) stream into a running decode batch is what triggered the
// `output with shape [1] doesn't match the broadcast shape [2]` crash inside
// PyWrappedModel::buildPyAttentionInputs.
TEST_F(GatherBatchSchedulerTest, testPyModelGuardDefersGatherWhileRunningBusy) {
    auto scheduler = makeScheduler(/*load_python_model=*/true);

    // Step 1: stream1 enters running.
    auto stream1 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    auto streams_status1 = scheduler->schedule();
    ASSERT_TRUE(streams_status1.ok());
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);

    // Step 2: stream1 stays in running (simulating it's still in decode). Enqueue stream2.
    auto stream2 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());

    // Schedule must NOT add stream2 to running while stream1 is still there.
    auto streams_status2 = scheduler->schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
    // The returned list is the unchanged running set (still just stream1).
    ASSERT_EQ(streams_status2.value().size(), 1);
}

// After the guard defers a gather, the deferred stream must be picked up as soon as the
// previously-running stream finishes — otherwise the scheduler would be stuck.
TEST_F(GatherBatchSchedulerTest, testPyModelGuardResumesGatherAfterRunningDrains) {
    auto scheduler = makeScheduler(/*load_python_model=*/true);

    auto stream1 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);

    auto stream2 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());

    // Guard fires — stream2 deferred.
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);

    // stream1 finishes. Next schedule should retire stream1 and admit stream2.
    stream1->reportEvent(StreamEvents::GenerateDone);
    auto streams_status = scheduler->schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
    // The single running stream is stream2 (not stream1).
    ASSERT_EQ(streams_status.value().front()->streamId(), stream2->streamId());
}

// A pure prompt_batch arrival (the smoke-test scenario): N streams enqueued at once with
// gather_batch_size=N. With load_python_model=true and running empty, the guard does not fire
// and all N streams are gathered together as a single prefill batch.
TEST_F(GatherBatchSchedulerTest, testPyModelGuardAllowsBatchGatherWhenRunningEmpty) {
    auto scheduler = makeScheduler(/*load_python_model=*/true);
    setGatherBatchSize(*scheduler, 3);

    auto stream1 = makeStream();
    auto stream2 = makeStream();
    auto stream3 = makeStream();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());
    ASSERT_TRUE(scheduler->enqueue(stream3).ok());

    auto streams_status = scheduler->schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 3);
}

TEST_F(GatherBatchSchedulerTest, testMixedIndependentEnqueuesSplitOnlyTheFullGatherWindow) {
    for (bool zero_first : {false, true}) {
        SCOPED_TRACE(zero_first);
        auto scheduler = makeScheduler(/*load_python_model=*/true);
        setGatherBatchSize(*scheduler, 3);
        auto first  = makeStream(zero_first ? 0 : 2, 30);
        auto second = makeStream(zero_first ? 2 : 0, 20);
        auto third  = makeStream(zero_first ? 0 : 2, 10);
        auto fourth = makeStream(zero_first ? 0 : 2, 0);
        for (const auto& stream : {first, second, third, fourth}) {
            ASSERT_TRUE(scheduler->enqueue(stream).ok());
        }
        auto result = scheduler->schedule();
        ASSERT_TRUE(result.ok());
        // Preserve streamId sorting, but never pull fourth across the original N=3 window.
        EXPECT_EQ(std::vector<GenerateStreamPtr>(result->begin(), result->end()),
                  (std::vector<GenerateStreamPtr>{third, first}));
        EXPECT_EQ(scheduler->gather_batch_size_, 1);
        EXPECT_EQ(scheduler->waitingStreamsSize(), 2);
        EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
        EXPECT_FALSE(fourth->hasEvent(StreamEvents::CanRun));
        EXPECT_FALSE(hasMixedExecutionModes(*result));
        // A still-running batch must not absorb either deferred mode.
        result = scheduler->schedule();
        ASSERT_TRUE(result.ok());
        EXPECT_EQ(result->size(), 2u);
        first->reportEvent(StreamEvents::GenerateDone);
        third->reportEvent(StreamEvents::GenerateDone);
        result = scheduler->schedule();
        ASSERT_TRUE(result.ok());
        ASSERT_EQ(result->size(), 1u);
        EXPECT_EQ(result->front(), second);
        second->reportEvent(StreamEvents::GenerateDone);
        result = scheduler->schedule();
        ASSERT_TRUE(result.ok());
        ASSERT_EQ(result->size(), 1u);
        EXPECT_EQ(result->front(), fourth);
        EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
        for (const auto& stream : {first, second, third, fourth}) {
            EXPECT_FALSE(stream->hasError());
        }
    }
}

TEST_F(GatherBatchSchedulerTest, testMixedQueueStillWaitsForConfiguredGatherSize) {
    auto scheduler = makeScheduler(/*load_python_model=*/true);
    setGatherBatchSize(*scheduler, 3);
    auto zero     = makeStream(0);
    auto positive = makeStream(2);
    ASSERT_TRUE(scheduler->enqueue(zero).ok());
    ASSERT_TRUE(scheduler->enqueue(positive).ok());
    auto pending = std::async(std::launch::async, [&] { return scheduler->schedule(); });
    EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    auto peer = makeStream(0);
    // Always unblock the future, including when the assertion above reports a regression.
    EXPECT_TRUE(scheduler->enqueue(peer).ok());
    auto result = pending.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 2u);
    EXPECT_FALSE(hasMixedExecutionModes(*result));
    EXPECT_EQ(positive->getStatus(), StreamState::WAITING);
}

TEST_F(GatherBatchSchedulerTest, testNewGatherSettingStillGatesDeferredMixedTail) {
    auto scheduler = makeScheduler(/*load_python_model=*/true);
    setGatherBatchSize(*scheduler, 2);
    auto zero     = makeStream(0);
    auto positive = makeStream(2);
    ASSERT_TRUE(scheduler->enqueue(zero).ok());
    ASSERT_TRUE(scheduler->enqueue(positive).ok());
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ(result->front(), zero);
    setGatherBatchSize(*scheduler, 2);
    zero->reportEvent(StreamEvents::GenerateDone);
    result = scheduler->schedule();  // running work wakes the existing wait predicate
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_EQ(scheduler->gather_batch_size_, 2);
    EXPECT_FALSE(positive->hasEvent(StreamEvents::CanRun));
    auto peer = makeStream(2);
    ASSERT_TRUE(scheduler->enqueue(peer).ok());
    result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 2u);
    EXPECT_EQ(scheduler->gather_batch_size_, 1);
}

TEST_F(GatherBatchSchedulerTest, testExplicitMixedGroupRemainsUnsupportedAndAtomic) {
    auto                           scheduler = makeScheduler(/*load_python_model=*/true);
    std::vector<GenerateStreamPtr> streams{makeStream(0), makeStream(2)};
    auto                           result = scheduler->enqueueGroup(streams);
    EXPECT_EQ(result.first, (std::vector<bool>{false, false}));
    EXPECT_EQ(result.second, streams);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    for (const auto& stream : streams) {
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::UNKNOWN_ERROR);
        EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
        EXPECT_EQ(stream->curBlocksNum(), 0u);
    }
}

}  // namespace rtp_llm
