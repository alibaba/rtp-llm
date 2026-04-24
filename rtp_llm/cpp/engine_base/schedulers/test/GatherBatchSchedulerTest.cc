#include <memory>
#include "torch/all.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/engine_base/schedulers/GatherBatchScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/core/Types.h"
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

    std::shared_ptr<GenerateStream> makeStream() {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
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

// Without load_python_model the scheduler is allowed to add new streams to a non-empty running
// set (the prefill+decode mix is fine for the regular C++ path). This pins down that the new
// guard does not regress the non-py_model behaviour.
TEST_F(GatherBatchSchedulerTest, testNonPyModelAllowsGatherWhileRunning) {
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
    // Without the py_model guard, stream2 is allowed to join running alongside stream1.
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 2);
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

}  // namespace rtp_llm
