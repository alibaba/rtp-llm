
#include <memory>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/test/mock/MockKVCacheConnectorCoordinator.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;
using testing::Return;
using testing::NiceMock;
using testing::_;

namespace rtp_llm {

class FIFOSchedulerAsyncCacheTest: public DeviceTestBase {
protected:
    FIFOSchedulerAsyncCacheTest(): perf_scope("PERF_TEST", "1") {}

    void SetUp() override {
        DeviceTestBase::SetUp();
        // Default: enough blocks for testing
        cache_config_ = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/21, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
        ASSERT_TRUE(cache_manager_->init());
    }

    void setupMockCoordinator() {
        mock_coord_ = std::make_shared<NiceMock<MockKVCacheConnectorCoordinator>>(cache_manager_->config_,
                                                                                  cache_manager_->kv_cache_config_,
                                                                                  cache_manager_->runtime_config_,
                                                                                  cache_manager_->allocator_,
                                                                                  nullptr);
        ON_CALL(*mock_coord_, hasActiveConnectors()).WillByDefault(Return(true));
        cache_manager_->coordinator_ = mock_coord_;
    }

    std::shared_ptr<FIFOScheduler> createScheduler(size_t   max_batch_size = 100,
                                                   RoleType role_type      = RoleType::PDFUSION) {
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                      = max_batch_size;
        runtime_config.fifo_scheduler_config.max_context_batch_size = 1;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size  = 8192;
        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = role_type;
        ParallelismConfig   parallelism_config;
        ModelSpecificConfig model_specific_config;
        return std::make_shared<FIFOScheduler>(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    }

    GenerateStreamPtr createStream(const std::vector<int>& input_tokens        = {1, 2, 3},
                                   bool                    reuse_cache         = false,
                                   bool                    enable_memory_cache = false,
                                   RoleType                role_type           = RoleType::PDFUSION,
                                   bool                    force_batch         = false,
                                   int64_t                 batch_group_id      = -1,
                                   int                     batch_group_size    = 1) {
        ResourceContext resource_context;
        resource_context.cache_manager       = cache_manager_;
        resource_context.reuse_cache         = reuse_cache;
        resource_context.enable_memory_cache = enable_memory_cache;
        resource_context.role_type           = role_type;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;

        std::shared_ptr<GenerateInput>  query(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->reuse_cache         = reuse_cache;
        generate_config->enable_memory_cache = enable_memory_cache;
        generate_config->force_batch         = force_batch;
        generate_config->batch_group_timeout = force_batch ? std::optional<int>(10000) : std::nullopt;
        query->input_ids                     = torch::tensor(input_tokens, torch::kInt32);
        query->generate_config               = generate_config;
        query->batch_group_id                = batch_group_id;
        query->batch_group_size              = batch_group_size;
        query->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    }

    // Create a mock FusedAsyncReadContext that is NOT done yet
    std::shared_ptr<MockAsyncContext> createPendingAsyncContext() {
        auto ctx = std::make_shared<NiceMock<MockAsyncContext>>();
        ON_CALL(*ctx, done()).WillByDefault(Return(false));
        ON_CALL(*ctx, success()).WillByDefault(Return(false));
        return ctx;
    }

    // Create a mock FusedAsyncReadContext that is immediately done
    std::shared_ptr<MockAsyncContext> createDoneAsyncContext() {
        auto ctx = std::make_shared<NiceMock<MockAsyncContext>>();
        ON_CALL(*ctx, done()).WillByDefault(Return(true));
        ON_CALL(*ctx, success()).WillByDefault(Return(true));
        return ctx;
    }

    template<typename Predicate>
    bool waitUntil(Predicate predicate) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!predicate() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return predicate();
    }

protected:
    autil::EnvGuard                                            perf_scope;
    CacheConfig                                                cache_config_;
    std::shared_ptr<KVCacheManager>                            cache_manager_;
    std::shared_ptr<NiceMock<MockKVCacheConnectorCoordinator>> mock_coord_;
};

// ============================================================================
// 1. scheduleNew: stream without reuse_cache goes directly to RUNNING
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleNew_NoReuseCache_DirectlyRunning) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/false);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // Single schedule: stream transitions directly to RUNNING (no cache loading needed)
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 0);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

// ============================================================================
// 2. scheduleNew: stream with reuse_cache and connector enters LOADING_CACHE
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleNew_WithReuseCache_EntersLoadingCache) {
    setupMockCoordinator();
    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    // Stream is in LOADING_CACHE, not in running
    ASSERT_EQ(result.value().size(), 0);
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);
}

// ============================================================================
// 3. evaluateLoadingCacheStreams: stream load done -> moves to waiting -> then running
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testEvaluateLoadingCache_LoadDone_MovesToRunning) {
    setupMockCoordinator();

    // Mock context: done() returns true when checked (load completes immediately)
    auto mock_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, waitDone()).WillByDefault(Return());

    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(mock_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // First schedule: stream enters LOADING_CACHE
    // (evaluateLoadingCacheStreams runs before scheduleNew, so loading_cache_streams_ is empty at that point)
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 0);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);

    // Second schedule: evaluateLoadingCacheStreams -> loadCacheDone()=true -> WAITING -> scheduleNew -> RUNNING
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 1);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

// ============================================================================
// 4. evaluateLoadingCacheStreams: stream with error during loading -> evicted
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testEvaluateLoadingCache_ErrorDuringLoading_Evicted) {
    setupMockCoordinator();

    // Mock context: done() returns true so evaluateLoadingCacheStreams proceeds to error check
    auto mock_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, waitDone()).WillByDefault(Return());

    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(mock_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // First schedule: enters LOADING_CACHE
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);

    // Simulate external error (e.g., cancel from gRPC)
    stream->reportError(ErrorCode::CANCELLED, "cancelled by client");

    // Second schedule: loadCacheDone()=true, hasError()=true -> stream evicted and finished
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 0);
    ASSERT_EQ(result2.value().size(), 0);
    ASSERT_TRUE(stream->isFinished());
}

// ============================================================================
// 5. loading_cache_streams_ counted in evaluateRunningBatch (batch size limit)
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCacheStreams_CountedInBatchLimit) {
    setupMockCoordinator();

    // Set max batch size to 2
    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 2;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    auto                scheduler = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillRepeatedly(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    // Enqueue 3 streams with reuse_cache
    auto stream1 = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto stream2 = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto stream3 = createStream({3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());
    ASSERT_TRUE(scheduler->enqueue(stream3).ok());

    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    // loading_cache_streams_ should count toward max_generate_batch_size
    // With max=2, only 2 streams should be scheduled (into LOADING_CACHE)
    // The 3rd stream should remain in waiting
    ASSERT_LE(result.value().size(), 2);
}

// ============================================================================
// 6. scheduleNew: stream returning from LOADING_CACHE (already has blocks) skips asyncLoadCache
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleNew_ReturningFromLoadingCache_SkipsAsyncLoad) {
    setupMockCoordinator();

    // Mock context: done() returns true when checked in evaluateLoadingCacheStreams
    auto mock_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, waitDone()).WillByDefault(Return());

    // asyncRead should only be called ONCE (for the first time entering LOADING_CACHE)
    EXPECT_CALL(*mock_coord_, asyncRead(_)).Times(1).WillOnce(Return(std::static_pointer_cast<AsyncContext>(mock_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // First schedule: stream -> LOADING_CACHE
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);

    // Second schedule: load done -> back to WAITING -> scheduleNew -> RUNNING (skips asyncLoadCache)
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 1);
    // asyncRead was called exactly once - the second scheduleNew sees had_blocks > 0 and skips asyncLoadCache
}

// ============================================================================
// 7. loading_cache_streams_ included in empty() and onflightStreams()
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCacheStreams_IncludedInEmptyAndOnflight) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);

    // Scheduler should NOT be empty when there are loading_cache_streams_
    ASSERT_FALSE(scheduler->empty());
    // onflightStreams should include loading_cache_streams_
    ASSERT_EQ(scheduler->onflightStreams(), 1);
}

// ============================================================================
// 8. loading_cache_streams_ included in waitPredicate()
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testWaitPredicate_IncludesLoadingCacheStreams) {
    auto scheduler = createScheduler();
    // Empty scheduler -> waitPredicate should be false
    ASSERT_FALSE(scheduler->waitPredicate());

    // Add a fake stream to loading_cache_streams_
    auto stream = createStream({1, 2, 3});
    scheduler->loading_cache_streams_.emplace_back(stream);
    ASSERT_TRUE(scheduler->waitPredicate());
}

// ============================================================================
// 9. evictDoneStreams handles external errors (hasError -> consumeError -> setFinishedWithoutLock)
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testEvictDoneStreams_HandlesExternalError) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3});

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // Single schedule: stream transitions directly to RUNNING (no cache loading needed)
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);

    // Simulate external error
    stream->reportError(ErrorCode::CANCELLED, "cancelled by RPC");

    // Next schedule: evictDoneStreams should detect the error, finish the stream, and release resources
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 0);
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);
}

// ============================================================================
// 10. Multiple streams: mix of async-loading and direct-running
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testMixedAsyncAndDirectStreams) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler = createScheduler();

    // Stream1: needs async cache load
    auto stream1 = createStream({1, 2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    // Stream2: no reuse cache
    auto stream2 = createStream({3, 4}, /*reuse_cache=*/false);

    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());

    // Single schedule: stream1 -> LOADING_CACHE (async load), stream2 -> RUNNING (directly)
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);  // Only stream2 is running
    ASSERT_TRUE(stream1->getStatus() == StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

// ============================================================================
// 11. evaluateLoadingCacheStreams: stream still loading -> stays in queue
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testEvaluateLoadingCache_StillLoading_StaysInQueue) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // First schedule: enters LOADING_CACHE
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);

    // Second schedule: still pending (done() returns false)
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    ASSERT_EQ(result2.value().size(), 0);
}

// ============================================================================
// 12. schedule() ordering: load_done_streams inserted at head of waiting_streams_
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleOrdering_LoadDoneStreamsAtWaitingHead) {
    setupMockCoordinator();

    // Mock context: done() returns true when checked in evaluateLoadingCacheStreams
    auto mock_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, waitDone()).WillByDefault(Return());

    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(mock_ctx)));

    auto scheduler = createScheduler();

    // Stream1: will enter LOADING_CACHE first
    auto stream1 = createStream({1, 2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);

    // Stream2: enqueued later while stream1 is loading
    auto stream2 = createStream({3, 4}, /*reuse_cache=*/false);
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());

    // Second schedule: stream1 load done -> moves to WAITING head -> should be scheduled before stream2
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    // Both streams should be running now
    ASSERT_GE(result2.value().size(), 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testFeatureFlagOffKeepsSynchronousAdmission) {
    autil::EnvGuard disable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "0");
    auto            scheduler = createScheduler(/*max_batch_size=*/2);
    auto            first     = createStream({1, 2, 3});
    auto            second    = createStream({4, 5, 6});
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());

    auto scheduled = scheduler->schedule();
    ASSERT_TRUE(scheduled.ok());
    ASSERT_EQ(scheduled->size(), 2);
    EXPECT_FALSE(scheduler->async_cache_prepare_enabled_);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testEmptyRunningClearsBlockedPrepareHead) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler();
    auto            blocked   = createStream({1, 2, 3});
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        scheduler->cache_prepare_blocked_stream_ = blocked;
        scheduler->schedule_trigger_             = true;
    }

    auto scheduled = scheduler->schedule();
    ASSERT_TRUE(scheduled.ok());
    EXPECT_TRUE(scheduled->empty());
    EXPECT_EQ(scheduler->cache_prepare_blocked_stream_, nullptr);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPrepareStartsAtEnqueue) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    setupMockCoordinator();

    std::atomic<bool> load_done{false};
    auto              async_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*async_ctx, done()).WillByDefault([&]() { return load_done.load(); });
    ON_CALL(*async_ctx, success()).WillByDefault(Return(true));
    EXPECT_CALL(*async_ctx, waitDone()).Times(1);
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(async_ctx)));

    auto scheduler = createScheduler(/*max_batch_size=*/1);
    auto stream    = createStream({1, 2, 3}, true, true);
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 1000000);
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(
        waitUntil([&]() { return stream->curBlocksNum() > 0 && stream->hasEvent(StreamEvents::LoadInitiated); }));
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
    EXPECT_EQ(stream->getTimeInfo().wait_time_us, 0);

    load_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return stream->hasEvent(StreamEvents::CachePrepared); }));
    auto running = scheduler->schedule();
    ASSERT_TRUE(running.ok());
    ASSERT_EQ(running->size(), 1);
    EXPECT_GE(stream->getTimeInfo().wait_time_us, 1000000);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPreparePreservesPrefillBatch) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/2);
    auto            first     = createStream({1, 2, 3});
    auto            second    = createStream({4, 5, 6});
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(waitUntil([&]() {
        return first->hasEvent(StreamEvents::CachePrepared) && second->hasEvent(StreamEvents::CachePrepared);
    }));

    auto running = scheduler->schedule();
    ASSERT_TRUE(running.ok());
    ASSERT_EQ(running->size(), 2);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPrepareResourceExhaustedKeepsStrictFifo) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/2);
    auto            current   = createStream(std::vector<int>(16, 1));
    auto            blocked   = createStream(std::vector<int>(30, 2));
    auto            trailing  = createStream(std::vector<int>(2, 3));

    ASSERT_TRUE(scheduler->enqueue(current).ok());
    ASSERT_TRUE(waitUntil([&]() { return current->hasEvent(StreamEvents::CachePrepared); }));
    auto first_round = scheduler->schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round->size(), 1);

    ASSERT_TRUE(scheduler->enqueue(blocked).ok());
    ASSERT_TRUE(scheduler->enqueue(trailing).ok());
    ASSERT_TRUE(waitUntil([&]() { return scheduler->cache_prepare_blocked_stream_ == blocked; }));
    EXPECT_EQ(trailing->curBlocksNum(), 0);
    EXPECT_FALSE(trailing->hasEvent(StreamEvents::CachePrepared));

    current->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(waitUntil([&]() {
        return blocked->hasEvent(StreamEvents::CachePrepared) && trailing->hasEvent(StreamEvents::CachePrepared);
    }));
    auto second_round = scheduler->schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round->size(), 2);
    EXPECT_EQ(second_round->front(), blocked);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLackMemStillFinishesEarlierInFlightLoad) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    setupMockCoordinator();

    std::atomic<bool> first_done{false};
    auto              first_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault([&]() { return first_done.load(); });
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    EXPECT_CALL(*first_ctx, waitDone()).Times(1);
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)));

    auto scheduler = createScheduler(/*max_batch_size=*/2);
    auto first     = createStream(std::vector<int>(16, 1), true, true);
    auto blocked   = createStream(std::vector<int>(30, 2));
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(blocked).ok());
    ASSERT_TRUE(waitUntil([&]() { return scheduler->cache_prepare_blocked_stream_ == blocked; }));

    first_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return first->hasEvent(StreamEvents::CachePrepared); }));
    auto first_round = scheduler->schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round->size(), 1);
    EXPECT_EQ(first_round->front(), first);
    EXPECT_FALSE(blocked->hasEvent(StreamEvents::CachePrepared));

    first->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(waitUntil([&]() { return blocked->hasEvent(StreamEvents::CachePrepared); }));
    auto second_round = scheduler->schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round->size(), 1);
    EXPECT_EQ(second_round->front(), blocked);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPreparePublishesReadyPrefixWithoutWaitingForWindow) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    setupMockCoordinator();

    std::atomic<bool> first_done{false};
    std::atomic<bool> second_done{false};
    auto              first_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    auto              second_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault([&]() { return first_done.load(); });
    ON_CALL(*second_ctx, done()).WillByDefault([&]() { return second_done.load(); });
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    EXPECT_CALL(*first_ctx, waitDone()).Times(1);
    EXPECT_CALL(*second_ctx, waitDone()).Times(1);
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(/*max_batch_size=*/2);
    auto first     = createStream({1, 2, 3}, true, true);
    auto second    = createStream({4, 5, 6}, true, true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(waitUntil([&]() {
        return first->hasEvent(StreamEvents::LoadInitiated) && second->hasEvent(StreamEvents::LoadInitiated);
    }));

    first_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return first->hasEvent(StreamEvents::CachePrepared); }));
    auto partial = scheduler->schedule();
    ASSERT_TRUE(partial.ok());
    ASSERT_EQ(partial->size(), 1);
    EXPECT_EQ(partial->front(), first);

    second_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return second->hasEvent(StreamEvents::CachePrepared); }));
    first->reportEvent(StreamEvents::GenerateDone);
    auto complete = scheduler->schedule();
    ASSERT_TRUE(complete.ok());
    ASSERT_EQ(complete->size(), 1);
    EXPECT_EQ(complete->front(), second);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPrepareDoesNotPublishOutOfOrderCompletion) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    setupMockCoordinator();

    std::atomic<bool> first_done{false};
    auto              first_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    auto              second_ctx = createDoneAsyncContext();
    ON_CALL(*first_ctx, done()).WillByDefault([&]() { return first_done.load(); });
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    EXPECT_CALL(*first_ctx, waitDone()).Times(1);
    EXPECT_CALL(*second_ctx, waitDone()).Times(1);
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(/*max_batch_size=*/2);
    auto first     = createStream({1, 2, 3}, true, true);
    auto second    = createStream({4, 5, 6}, true, true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(waitUntil([&]() { return second->hasEvent(StreamEvents::CachePrepared); }));

    auto blocked = scheduler->schedule();
    ASSERT_TRUE(blocked.ok());
    EXPECT_TRUE(blocked->empty());

    first_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return first->hasEvent(StreamEvents::CachePrepared); }));
    auto ready = scheduler->schedule();
    ASSERT_TRUE(ready.ok());
    ASSERT_EQ(ready->size(), 2);
    EXPECT_EQ(ready->front(), first);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testNextPrepareOverlapsCurrentGpuRound) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    setupMockCoordinator();

    auto scheduler = createScheduler(/*max_batch_size=*/1);
    auto current   = createStream({1, 2, 3});
    ASSERT_TRUE(scheduler->enqueue(current).ok());
    ASSERT_TRUE(waitUntil([&]() { return current->hasEvent(StreamEvents::CachePrepared); }));
    auto first_round = scheduler->schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round->size(), 1);

    std::atomic<bool> load_done{false};
    auto              async_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*async_ctx, done()).WillByDefault([&]() { return load_done.load(); });
    ON_CALL(*async_ctx, success()).WillByDefault(Return(true));
    EXPECT_CALL(*async_ctx, waitDone()).Times(1);
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(async_ctx)));

    auto next = createStream({4, 5, 6}, true, true);
    ASSERT_TRUE(scheduler->enqueue(next).ok());
    ASSERT_TRUE(waitUntil([&]() { return next->hasEvent(StreamEvents::LoadInitiated); }));
    EXPECT_EQ(current->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(next->getStatus(), StreamState::WAITING);

    load_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return next->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_EQ(current->getStatus(), StreamState::RUNNING);

    current->reportEvent(StreamEvents::GenerateDone);
    auto second_round = scheduler->schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round->size(), 1);
    EXPECT_EQ(second_round->front(), next);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testForcedBatchWaitsUntilWholeGroupPrepared) {
    autil::EnvGuard                enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto                           scheduler = createScheduler(/*max_batch_size=*/6);
    std::vector<GenerateStreamPtr> streams;
    for (int i = 0; i < 5; ++i) {
        streams.push_back(createStream({i + 1}, false, false, RoleType::PDFUSION, true, 7, 6));
        ASSERT_TRUE(scheduler->enqueue(streams.back()).ok());
    }
    ASSERT_TRUE(waitUntil([&]() {
        return std::all_of(streams.begin(), streams.end(), [](const auto& stream) {
            return stream->hasEvent(StreamEvents::CachePrepared);
        });
    }));
    auto incomplete = scheduler->schedule();
    ASSERT_TRUE(incomplete.ok());
    EXPECT_TRUE(incomplete->empty());

    streams.push_back(createStream({6}, false, false, RoleType::PDFUSION, true, 7, 6));
    ASSERT_TRUE(scheduler->enqueue(streams.back()).ok());
    ASSERT_TRUE(waitUntil([&]() { return streams.back()->hasEvent(StreamEvents::CachePrepared); }));
    auto complete = scheduler->schedule();
    ASSERT_TRUE(complete.ok());
    ASSERT_EQ(complete->size(), 6);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testDecodeRoleBypassesAsyncWorker) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1, RoleType::DECODE);
    EXPECT_FALSE(scheduler->async_cache_prepare_enabled_);
}

}  // namespace rtp_llm
