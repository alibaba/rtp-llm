#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class FIFOSchedulerTest: public DeviceTestBase {
public:
    FIFOSchedulerTest() {}
};

class ObservableFIFOScheduler: public FIFOScheduler {
public:
    using FIFOScheduler::FIFOScheduler;

    bool waitUntilContextBatchCoalescing() {
        std::unique_lock<std::mutex> lock(coalescing_lock_);
        return coalescing_cond_.wait_for(
            lock, std::chrono::seconds(1), [this] { return context_batch_coalescing_wait_entered_; });
    }

    bool contextBatchCoalescingWaitEntered() {
        std::lock_guard<std::mutex> lock(coalescing_lock_);
        return context_batch_coalescing_wait_entered_;
    }

protected:
    void onContextBatchCoalescingWait() override {
        {
            std::lock_guard<std::mutex> lock(coalescing_lock_);
            context_batch_coalescing_wait_entered_ = true;
        }
        coalescing_cond_.notify_all();
    }

private:
    std::mutex              coalescing_lock_;
    std::condition_variable coalescing_cond_;
    bool                    context_batch_coalescing_wait_entered_ = false;
};

class ContextBatchSchedulerHarness {
public:
    ContextBatchSchedulerHarness(int64_t  max_context_batch_size,
                                 int64_t  coalescing_window_ms,
                                 RoleType role_type = RoleType::PDFUSION) {
        auto cache_config =
            rtp_llm::test::makeSimpleMhaCacheConfig(1, 65, 8, rtp_llm::DataType::TYPE_FP16, 1, 4);
        cache_manager     = std::make_shared<KVCacheManager>(cache_config);
        if (!cache_manager->init()) {
            throw std::runtime_error("failed to initialize test cache manager");
        }
        resource_context.cache_manager = cache_manager;
        resource_context.role_type     = role_type;

        model_config.max_seq_len = 8192;
        runtime_config.max_generate_batch_size = 32;
        runtime_config.fifo_scheduler_config.max_context_batch_size = max_context_batch_size;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size  = 8192;
        runtime_config.fifo_scheduler_config.context_batch_coalescing_window_ms = coalescing_window_ms;
        pd_sep_config.role_type = role_type;
        scheduler = std::make_unique<ObservableFIFOScheduler>(runtime_config,
                                                              model_config,
                                                              pd_sep_config,
                                                              parallelism_config,
                                                              model_specific_config,
                                                              cache_manager);
    }

    GenerateStreamPtr makeStream(bool context_stream = true) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor({1}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        query->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = std::make_shared<NormalGenerateStream>(
            query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(context_stream);
        return stream;
    }

    GenerateStreamPtr makeStreamWithBatchSize(int batch_size) {
        auto query                                  = std::make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = std::make_shared<GenerateConfig>();
        query->generate_config->num_return_sequences = batch_size;
        query->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = std::make_shared<NormalGenerateStream>(
            query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(true);
        return stream;
    }

    GenerateStreamPtr makeExpiredForceBatchStream(int batch_size) {
        auto query                                      = std::make_shared<GenerateInput>();
        query->input_ids                                = torch::tensor({1}, torch::kInt32);
        query->generate_config                          = std::make_shared<GenerateConfig>();
        query->generate_config->num_return_sequences    = batch_size;
        query->generate_config->force_batch             = true;
        query->generate_config->batch_group_timeout     = 1;
        query->batch_group_id                           = 17;
        query->batch_group_size                         = batch_size + 1;
        query->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - 1000 * 1000;
        auto stream = std::make_shared<NormalGenerateStream>(
            query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(true);
        return stream;
    }

    GenerateStreamPtr makeAgedCompleteForceBatchStream(int64_t group_id, int group_size, int batch_size) {
        auto query                                      = std::make_shared<GenerateInput>();
        query->input_ids                                = torch::tensor({1}, torch::kInt32);
        query->generate_config                          = std::make_shared<GenerateConfig>();
        query->generate_config->num_return_sequences    = batch_size;
        query->generate_config->force_batch             = true;
        query->generate_config->batch_group_timeout     = 1;
        query->batch_group_id                           = group_id;
        query->batch_group_size                         = group_size;
        query->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - 1000 * 1000;
        auto stream = std::make_shared<NormalGenerateStream>(
            query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(true);
        return stream;
    }

    std::shared_ptr<KVCacheManager> cache_manager;
    ResourceContext                 resource_context;
    ModelConfig                     model_config;
    RuntimeConfig                   runtime_config;
    PDSepConfig                     pd_sep_config;
    ParallelismConfig               parallelism_config;
    ModelSpecificConfig             model_specific_config;
    std::unique_ptr<ObservableFIFOScheduler> scheduler;
};

TEST_F(FIFOSchedulerTest, testIdleContextBatchReleasesImmediatelyWhenFull) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/3, /*coalescing_window_ms=*/5000);
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());

    auto schedule_future =
        std::async(std::launch::async, [&harness] { return harness.scheduler->schedule(); });
    ASSERT_TRUE(harness.scheduler->waitUntilContextBatchCoalescing());

    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());
    ASSERT_EQ(schedule_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = schedule_future.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 3);
    EXPECT_EQ(harness.scheduler->waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testIdleContextBatchFallsBackWhenWindowExpires) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/4, /*coalescing_window_ms=*/100);
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());

    auto schedule_future =
        std::async(std::launch::async, [&harness] { return harness.scheduler->schedule(); });
    ASSERT_TRUE(harness.scheduler->waitUntilContextBatchCoalescing());
    ASSERT_EQ(schedule_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = schedule_future.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 1);
}

TEST_F(FIFOSchedulerTest, testContextBatchCoalescingDisabledKeepsImmediateScheduling) {
    for (const auto [max_context_batch_size, coalescing_window_ms] :
         std::vector<std::pair<int64_t, int64_t>>{{4, 0}, {1, 5000}}) {
        ContextBatchSchedulerHarness harness(max_context_batch_size, coalescing_window_ms);
        ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());
        ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());

        auto schedule_future =
            std::async(std::launch::async, [&harness] { return harness.scheduler->schedule(); });
        const auto wait_status = schedule_future.wait_for(std::chrono::seconds(1));
        if (wait_status != std::future_status::ready) {
            ASSERT_TRUE(harness.scheduler->stop().ok());
        }
        ASSERT_EQ(wait_status, std::future_status::ready);
        auto result = schedule_future.get();
        ASSERT_TRUE(result.ok());
        EXPECT_EQ(result->size(), 2);
    }
}

TEST_F(FIFOSchedulerTest, testStopWakesContextBatchCoalescingWait) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/4, /*coalescing_window_ms=*/10000);
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());

    auto schedule_future =
        std::async(std::launch::async, [&harness] { return harness.scheduler->schedule(); });
    ASSERT_TRUE(harness.scheduler->waitUntilContextBatchCoalescing());
    ASSERT_TRUE(harness.scheduler->stop().ok());
    ASSERT_EQ(schedule_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = schedule_future.get();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
}

TEST_F(FIFOSchedulerTest, testCancellationBreaksActiveContextBatchCoalescingWait) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/4, /*coalescing_window_ms=*/10000);
    auto                         live_stream      = harness.makeStream();
    auto                         cancelled_stream = harness.makeStream();
    ASSERT_TRUE(harness.scheduler->enqueue(live_stream).ok());
    ASSERT_TRUE(harness.scheduler->enqueue(cancelled_stream).ok());

    auto schedule_future =
        std::async(std::launch::async, [&harness] { return harness.scheduler->schedule(); });
    ASSERT_TRUE(harness.scheduler->waitUntilContextBatchCoalescing());
    cancelled_stream->reportError(ErrorCode::CANCELLED, "cancelled while waiting for batch");
    ASSERT_EQ(schedule_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = schedule_future.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), live_stream);
    EXPECT_TRUE(cancelled_stream->isFinished());
}

TEST_F(FIFOSchedulerTest, testEnabledContextCoalescingCapsBatchAtConfiguredSize) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/2, /*coalescing_window_ms=*/5000);
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());

    auto result = harness.scheduler->schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 2);
    EXPECT_EQ(harness.scheduler->waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, testFirstOversizedContextStreamMakesProgressUnderSoftCap) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/2, /*coalescing_window_ms=*/5000);
    auto                         oversized_stream = harness.makeStreamWithBatchSize(/*batch_size=*/3);
    ASSERT_TRUE(harness.scheduler->enqueue(oversized_stream).ok());
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());

    auto result = harness.scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), oversized_stream);
    EXPECT_EQ(oversized_stream->currentBatchSize(), 3);
    EXPECT_EQ(harness.scheduler->waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerTest, testExpiredForceBatchFallbackRespectsContextBatchCap) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/2, /*coalescing_window_ms=*/5000);
    auto normal_stream        = harness.makeStream();
    auto expired_force_stream = harness.makeExpiredForceBatchStream(/*batch_size=*/2);
    harness.scheduler->waiting_streams_.push_back(normal_stream);
    harness.scheduler->waiting_streams_.push_back(expired_force_stream);

    harness.scheduler->evaluateWaitingStreams(harness.scheduler->waiting_streams_);

    EXPECT_TRUE(normal_stream->hasEvent(StreamEvents::CanRun));
    EXPECT_FALSE(expired_force_stream->hasEvent(StreamEvents::CanRun));
    harness.scheduler->waiting_streams_.clear();
}

TEST_F(FIFOSchedulerTest, testExpiredForceBatchFallbackCompletesCoalescingBatch) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/2, /*coalescing_window_ms=*/5000);
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream()).ok());
    ASSERT_TRUE(harness.scheduler->enqueue(harness.makeExpiredForceBatchStream(/*batch_size=*/1)).ok());

    auto schedule_future =
        std::async(std::launch::async, [&harness] { return harness.scheduler->schedule(); });
    const auto wait_status = schedule_future.wait_for(std::chrono::seconds(1));
    if (wait_status != std::future_status::ready) {
        ASSERT_TRUE(harness.scheduler->stop().ok());
    }
    ASSERT_EQ(wait_status, std::future_status::ready);
    auto result = schedule_future.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 2);
    EXPECT_FALSE(harness.scheduler->contextBatchCoalescingWaitEntered());
}

TEST_F(FIFOSchedulerTest, testAgedCompleteForceBatchRemainsAtomicAboveContextBatchCap) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/2, /*coalescing_window_ms=*/5000);
    constexpr int64_t            group_id = 23;
    auto first = harness.makeAgedCompleteForceBatchStream(group_id, /*group_size=*/2, /*batch_size=*/2);
    auto second = harness.makeAgedCompleteForceBatchStream(group_id, /*group_size=*/2, /*batch_size=*/2);
    ASSERT_TRUE(harness.scheduler->enqueue(first).ok());
    ASSERT_TRUE(harness.scheduler->enqueue(second).ok());

    auto result = harness.scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 2);
    EXPECT_EQ(result->front(), first);
    EXPECT_EQ(result->back(), second);
    EXPECT_EQ(harness.scheduler->waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testSchedulingRoundKeepsContextAndDecodePhasesSeparate) {
    for (const bool first_is_context : {true, false}) {
        ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/4, /*coalescing_window_ms=*/0);
        ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream(first_is_context)).ok());
        ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream(!first_is_context)).ok());

        auto result = harness.scheduler->schedule();
        ASSERT_TRUE(result.ok());
        ASSERT_EQ(result->size(), 1);
        EXPECT_EQ(result->front()->isContextStream(), first_is_context);
        EXPECT_EQ(harness.scheduler->waitingStreamsSize(), 1);
    }
}

TEST_F(FIFOSchedulerTest, testLoadingStreamDoesNotConstrainPDFusionExecutionPhase) {
    ContextBatchSchedulerHarness harness(/*max_context_batch_size=*/4, /*coalescing_window_ms=*/0);
    auto loading_context = harness.makeStream(/*context_stream=*/true);
    auto waiting_decode  = harness.makeStream(/*context_stream=*/false);
    auto waiting_context = harness.makeStream(/*context_stream=*/true);
    harness.scheduler->loading_cache_streams_.push_back(loading_context);
    harness.scheduler->waiting_streams_.push_back(waiting_decode);
    harness.scheduler->waiting_streams_.push_back(waiting_context);

    harness.scheduler->evaluateWaitingStreams(harness.scheduler->waiting_streams_);

    EXPECT_TRUE(waiting_decode->hasEvent(StreamEvents::CanRun));
    EXPECT_FALSE(waiting_context->hasEvent(StreamEvents::CanRun));
    harness.scheduler->loading_cache_streams_.clear();
    harness.scheduler->waiting_streams_.clear();
}

TEST_F(FIFOSchedulerTest, testContextCoalescingIsDisabledForDedicatedRoles) {
    for (const auto role_type : {RoleType::PREFILL, RoleType::DECODE}) {
        ContextBatchSchedulerHarness harness(
            /*max_context_batch_size=*/4, /*coalescing_window_ms=*/5000, role_type);
        ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream(/*context_stream=*/true)).ok());
        ASSERT_TRUE(harness.scheduler->enqueue(harness.makeStream(/*context_stream=*/false)).ok());

        auto schedule_future =
            std::async(std::launch::async, [&harness] { return harness.scheduler->schedule(); });
        const auto wait_status = schedule_future.wait_for(std::chrono::seconds(1));
        if (wait_status != std::future_status::ready) {
            ASSERT_TRUE(harness.scheduler->stop().ok());
        }
        ASSERT_EQ(wait_status, std::future_status::ready);
        auto result = schedule_future.get();
        ASSERT_TRUE(result.ok());
        if (role_type == RoleType::DECODE) {
            EXPECT_TRUE(result->empty());
            result = harness.scheduler->schedule();
            ASSERT_TRUE(result.ok());
        }
        EXPECT_EQ(result->size(), 2);
        EXPECT_FALSE(harness.scheduler->contextBatchCoalescingWaitEntered());
    }
}

TEST_F(FIFOSchedulerTest, testContextCoalescingWindowBounds) {
    EXPECT_NO_THROW((ContextBatchSchedulerHarness(/*max_context_batch_size=*/4, /*coalescing_window_ms=*/60000)));
    EXPECT_THROW((ContextBatchSchedulerHarness(/*max_context_batch_size=*/4, /*coalescing_window_ms=*/-1)),
                 std::invalid_argument);
    EXPECT_THROW((ContextBatchSchedulerHarness(/*max_context_batch_size=*/4, /*coalescing_window_ms=*/60001)),
                 std::invalid_argument);
}

TEST_F(FIFOSchedulerTest, testSimple) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = torch::tensor({1}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false without enable_memory_cache)
    // Since no cache loading is needed, stream transitions directly to RUNNING in one schedule call
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    stream->reportEvent(StreamEvents::GenerateDone);

    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);
}

TEST_F(FIFOSchedulerTest, testInitKVCacheLackMem) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 2, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = torch::tensor({1, 2, 3}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    // In the new code, checkInputLength rejects at enqueue time
    ASSERT_FALSE(scheduler.enqueue(stream).ok());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->stopReason(), "input len 3 is greater than kv cache max available tokens num 2");
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);
}

TEST_F(FIFOSchedulerTest, testIncrKVCacheLackMem) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 3, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false)
    // Since no cache loading is needed, stream transitions directly to RUNNING in one schedule call
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 1);
    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(stream->stopReason(), "");
    ASSERT_EQ(cache_manager->freeBlocksNum(), 0);

    stream->setSeqLength(stream->seqLength() + 1);
    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 0);
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->stopReason(), "incrKVBlock failed: LACK MEM");
    ASSERT_EQ(cache_manager->freeBlocksNum(), 2);
}

TEST_F(FIFOSchedulerTest, testInitKVCacheRejectedByReserveBlocks) {
    CacheConfig cache_config = makeMhaCacheConfig(/*layer_num=*/1,
                                                  /*block_num=*/11,
                                                  /*local_head_num_kv=*/1,
                                                  /*size_per_head=*/4,
                                                  /*tokens_per_block=*/1,
                                                  rtp_llm::DataType::TYPE_FP16);

    KVCacheConfig kv_cache_config;
    kv_cache_config.reserve_block_ratio = 50;  // reserve = 50% * available(10) = 5 blocks

    std::shared_ptr<KVCacheManager> cache_manager =
        std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, nullptr, kv_cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 10);

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Need 6 blocks. With reserve=5 blocks and available=10 blocks, init malloc should be rejected.
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = torch::tensor({1, 2, 3, 4, 5, 6}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();

    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 0);
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->stopReason(), "LACK MEM");
    ASSERT_EQ(cache_manager->freeBlocksNum(), 10);
    ASSERT_EQ(cache_manager->availableBlocksNum(), 10);
}

TEST_F(FIFOSchedulerTest, testReserveBlocksOnlyAffectInitMallocNotIncrMalloc) {
    CacheConfig cache_config = makeMhaCacheConfig(/*layer_num=*/1,
                                                  /*block_num=*/11,
                                                  /*local_head_num_kv=*/1,
                                                  /*size_per_head=*/4,
                                                  /*tokens_per_block=*/1,
                                                  rtp_llm::DataType::TYPE_FP16);

    KVCacheConfig kv_cache_config;
    kv_cache_config.reserve_block_ratio = 50;  // reserve = 5 blocks

    std::shared_ptr<KVCacheManager> cache_manager =
        std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, nullptr, kv_cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 10);

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = false;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    // Init need 4 blocks, should pass: 10 >= 4 + 5.
    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();

    shared_ptr<GenerateStream> stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false)
    // Since no cache loading is needed, stream transitions directly to RUNNING in one schedule call
    auto streams_status1 = scheduler.schedule();
    ASSERT_TRUE(streams_status1.ok());
    ASSERT_EQ(streams_status1.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_FALSE(stream->hasError());

    stream->setSeqLength(9);
    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(streams_status2.value().size(), 1);
    ASSERT_FALSE(stream->hasError());
}

TEST_F(FIFOSchedulerTest, testReuseCache) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 10);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = true;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
    query->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream1 =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream1).ok());

    // Single schedule: stream calls initKVBlock and asyncLoadCache (returns false without enable_memory_cache)
    // Since no cache loading is needed, stream transitions directly to RUNNING in one schedule call
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    // Stream is already running, no need for second schedule
    auto streams_status2 = scheduler.schedule();
    ASSERT_TRUE(streams_status2.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 7);

    stream1->reportEvent(StreamEvents::GenerateDone);
    auto streams_status3 = scheduler.schedule();

    ASSERT_TRUE(streams_status3.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 8);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
    query2->generate_config               = make_shared<GenerateConfig>();
    shared_ptr<GenerateStream> stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    ASSERT_TRUE(scheduler.enqueue(stream2).ok());

    // Third schedule for stream2: transitions to RUNNING in single call (no cache loading needed)
    auto streams_status4 = scheduler.schedule();
    ASSERT_TRUE(streams_status4.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 6);

    stream2->reportEvent(StreamEvents::GenerateDone);
    auto streams_status6 = scheduler.schedule();
    ASSERT_TRUE(streams_status6.ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 0);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 7);
}

TEST_F(FIFOSchedulerTest, testMaxContextBatchSize) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;
    resource_context.reuse_cache   = true;

    ModelConfig model_config;
    model_config.max_seq_len = 100;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 1;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size  = 100;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    {
        // test normalcase
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream1 =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream1).ok());

        // Single schedule: transitions to RUNNING (no cache loading needed)
        auto streams_status = scheduler.schedule();
        ASSERT_TRUE(streams_status.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 1);

        stream1->reportEvent(StreamEvents::GenerateDone);
        auto streams_status2 = scheduler.schedule();

        ASSERT_TRUE(streams_status2.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    }

    {
        // test normal case with tile num
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        query->generate_config->num_beams    = 2;
        shared_ptr<GenerateStream> stream1 =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream1).ok());

        // Single schedule: transitions to RUNNING (no cache loading needed)
        auto streams_status = scheduler.schedule();
        ASSERT_TRUE(streams_status.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 1);

        stream1->reportEvent(StreamEvents::GenerateDone);
        auto streams_status2 = scheduler.schedule();

        ASSERT_TRUE(streams_status2.ok());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
        ASSERT_EQ(scheduler.runningStreamsSize(), 0);
        ASSERT_EQ(cache_manager->freeBlocksNum(), 20);
    }

    {
        // After fix (commit 2238d50cf): checkInputLength no longer rejects when
        // inputLength * currentBatchSize > max_batch_tokens_size. Such requests must
        // be admitted to the waiting queue; the per-round token-budget constraint is
        // enforced later in evaluateRunningMemory using contextLength (not batch_size).
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        std::shared_ptr<GenerateInput> query2         = make_shared<GenerateInput>();
        query2->input_ids                             = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query2->generate_config                       = make_shared<GenerateConfig>();
        query2->generate_config->num_return_sequences = 20;
        shared_ptr<GenerateStream> stream2 =
            make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
        // input_len 7 * batch_size 20 = 140 > max_batch_tokens_size 100, but enqueue still succeeds.
        ASSERT_TRUE(scheduler.enqueue(stream2).ok());
        ASSERT_FALSE(stream2->hasError());
        ASSERT_EQ(scheduler.waitingStreamsSize(), 1);
    }
}

// Regression test for commit 2238d50cf: removing the
// `inputLength * currentBatchSize() > max_batch_tokens_size_` rejection in
// FIFOScheduler::checkInputLength. The check was wrong because max_batch_tokens_size
// bounds per-scheduling-round token usage (computed from contextLength in
// evaluateRunningMemory), not the multi-sequence batch fan-out at enqueue.
TEST_F(FIFOSchedulerTest, testCheckInputLengthIgnoresBatchSizeFanOut) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    // max_seq_len must exceed the longest input below (maxAvailableTokensNum + 1) so that
    // stream construction does not throw on its own seq_length check; the scheduler-level
    // checkInputLength is what we actually want to exercise here.
    model_config.max_seq_len = 1024;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 100;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    {
        // num_return_sequences fan-out: input_len 7 * batch 20 = 140 > 100, but accepted.
        std::shared_ptr<GenerateInput> query         = make_shared<GenerateInput>();
        query->input_ids                             = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->num_return_sequences = 20;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        ASSERT_FALSE(stream->hasError());
    }

    {
        // num_beams fan-out: input_len 10 * batch 16 = 160 > 100, but accepted.
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        query->generate_config->num_beams    = 16;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
        ASSERT_FALSE(stream->hasError());
    }

    // The KV-cache-bound check still rejects (input_len > maxAvailableTokensNum).
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        // Cache has 20 blocks * 8 tokens/block - 1 reserved tail = 159 max available; pick a length above it.
        std::vector<int32_t> ids(int(cache_manager->maxAvailableTokensNum()) + 1, 1);
        query->input_ids       = torch::tensor(ids, torch::kInt32);
        query->generate_config = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_FALSE(scheduler.enqueue(stream).ok());
        ASSERT_TRUE(stream->hasError());
    }
}

TEST_F(FIFOSchedulerTest, testBatchEnqueue) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 4, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ASSERT_EQ(cache_manager->freeBlocksNum(), 3);
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    vector<GenerateStreamPtr> streams;
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        streams.push_back(stream);
    }
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        streams.push_back(stream);
    }
    auto enqueued = scheduler.batchEnqueue(streams);
    ASSERT_EQ(enqueued.size(), streams.size());

    // Single schedule: both streams transition to RUNNING (no cache loading needed)
    auto streams_status = scheduler.schedule();
    ASSERT_TRUE(streams_status.ok());
    ASSERT_EQ(streams_status.value().size(), 2);
    ASSERT_EQ(cache_manager->freeBlocksNum(), 1);

    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

TEST_F(FIFOSchedulerTest, testForceBatchGroupComplete) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    int64_t group_id   = 100;
    int     group_size = 3;

    // Enqueue only 2 of 3 — group incomplete, should not be scheduled
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // First schedule: streams stay in WAITING (group incomplete, cannot run yet)
    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 0);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);

    // Enqueue the 3rd — group complete, all 3 should be scheduled together
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // Second schedule: group complete, all 3 streams transition to RUNNING in single call
    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 3);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 3);
}

TEST_F(FIFOSchedulerTest, testForceBatchTimeout) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    int64_t group_id   = 200;
    int     group_size = 3;
    int     timeout_ms = 10;
    int64_t past_time  = autil::TimeUtility::currentTimeInMicroSeconds() - (timeout_ms + 100) * 1000;

    // Enqueue only 2 of 3 with begin_time far in the past so timeout has expired
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = timeout_ms;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = past_time;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = timeout_ms;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = past_time;
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // Single schedule: timeout expired, streams transition to RUNNING
    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerTest, testForceBatchIsolation) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 11, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    int64_t group_id   = 300;
    int     group_size = 2;

    // Enqueue: normal stream first, then a complete force batch group
    shared_ptr<GenerateStream> normal_stream;
    {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids                     = torch::tensor({1}, torch::kInt32);
        query->generate_config               = make_shared<GenerateConfig>();
        query->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
        normal_stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(normal_stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        shared_ptr<GenerateStream> stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // Round 1: normal stream transitions to RUNNING (force batch streams skipped due to batch isolation)
    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 1);

    // Finish the normal stream
    normal_stream->reportEventWithoutLock(StreamEvents::GenerateDone);

    // Round 2: force batch group transitions to RUNNING
    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

// Two different complete force batch groups: only one group per scheduling round
TEST_F(FIFOSchedulerTest, testTwoForceBatchGroupsIsolation) {
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 21, 1, 4, 8, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    int64_t group_id_a = 500;
    int64_t group_id_b = 600;
    int     group_size = 2;

    // Enqueue group A (2 streams), then group B (2 streams), both complete
    vector<shared_ptr<GenerateStream>> group_a_streams;
    for (int i = 0; i < group_size; i++) {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id_a;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        group_a_streams.push_back(stream);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }
    for (int i = 0; i < group_size; i++) {
        std::shared_ptr<GenerateInput> query        = make_shared<GenerateInput>();
        query->input_ids                            = torch::tensor({1}, torch::kInt32);
        query->generate_config                      = make_shared<GenerateConfig>();
        query->generate_config->force_batch         = true;
        query->generate_config->batch_group_timeout = 10;
        query->batch_group_id                       = group_id_b;
        query->batch_group_size                     = group_size;
        query->begin_time_us                        = autil::TimeUtility::currentTimeInMicroSeconds();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        ASSERT_TRUE(scheduler.enqueue(stream).ok());
    }

    // Round 1: group A transitions to RUNNING (group B skipped due to batch isolation)
    auto result1 = scheduler.schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 2);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);

    // Finish group A
    for (auto& s : group_a_streams) {
        s->reportEventWithoutLock(StreamEvents::GenerateDone);
    }

    // Round 2: group B transitions to RUNNING
    auto result2 = scheduler.schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 2);
    ASSERT_EQ(scheduler.waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler.runningStreamsSize(), 2);
}

}  // namespace rtp_llm
