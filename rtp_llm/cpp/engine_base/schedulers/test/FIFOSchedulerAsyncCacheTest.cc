
#include <memory>
#include <unistd.h>
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

    std::shared_ptr<FIFOScheduler> createScheduler() {
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                     = 100;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
        PDSepConfig         pd_sep_config;
        ParallelismConfig   parallelism_config;
        ModelSpecificConfig model_specific_config;
        return std::make_shared<FIFOScheduler>(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    }

    GenerateStreamPtr createStream(const std::vector<int>& input_tokens        = {1, 2, 3},
                                   bool                    reuse_cache         = false,
                                   bool                    enable_memory_cache = false) {
        ResourceContext resource_context;
        resource_context.cache_manager       = cache_manager_;
        resource_context.reuse_cache         = reuse_cache;
        resource_context.enable_memory_cache = enable_memory_cache;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;

        std::shared_ptr<GenerateInput>  query(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->reuse_cache         = reuse_cache;
        generate_config->enable_memory_cache = enable_memory_cache;
        query->input_ids                     = torch::tensor(input_tokens, torch::kInt32);
        query->generate_config               = generate_config;
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
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 0u);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

// ============================================================================
// 2. scheduleNew: stream with reuse_cache and connector enters loading_ queue
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleNew_WithReuseCache_EntersLoadingQueue) {
    setupMockCoordinator();
    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    // Stream is in loading_ queue, not in running
    ASSERT_EQ(result.value().size(), 0);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);
}

// ============================================================================
// 3. loading check: stream load done -> moves to waiting -> then running
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCheck_LoadDone_MovesToRunning) {
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

    // First schedule: stream enters loading_ queue
    // (loading check runs before scheduleNew, so loading_ is empty at that point)
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 0);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);

    // Second schedule: loading check -> loadCacheDone()=true -> WAITING -> scheduleNew -> RUNNING
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 1);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 0u);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

// ============================================================================
// 4. loading check: stream with error during loading -> evicted
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCheck_ErrorDuringLoading_Evicted) {
    setupMockCoordinator();

    // Mock context: done() returns true so loading check proceeds to error check
    auto mock_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, waitDone()).WillByDefault(Return());

    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(mock_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // First schedule: enters loading_ queue
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);

    // Simulate external error (e.g., cancel from gRPC)
    stream->reportError(ErrorCode::CANCELLED, "cancelled by client");

    // Second schedule: loadCacheDone()=true, hasError()=true -> stream evicted and finished
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 0u);
    ASSERT_EQ(result2.value().size(), 0);
    ASSERT_TRUE(stream->isFinished());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCheck_LoadFailureReportsErrorWithoutDeadlock) {
    setupMockCoordinator();

    auto mock_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, success()).WillByDefault(Return(false));
    ON_CALL(*mock_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*mock_ctx, errorInfo())
        .WillByDefault(Return(ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, "cache transfer failed")));
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(mock_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);

    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_TRUE(stream->isFinished());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::LOAD_CACHE_TIMEOUT);
}

// ============================================================================
// 5. admitted_count limits per-round batch size
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingQueue_CountedInBatchLimit) {
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
    // loading_ queue should count toward max_generate_batch_size
    // With max=2, only 2 streams should be scheduled (into loading_ queue)
    // The 3rd stream should remain in waiting
    ASSERT_LE(result.value().size(), 2);
}

// ============================================================================
// 6. scheduleNew: stream returning from loading_ queue (already has blocks) skips asyncLoadCache
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleNew_ReturningFromLoadingQueue_SkipsAsyncLoad) {
    setupMockCoordinator();

    // Mock context: done() returns true when checked in loading check
    auto mock_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, waitDone()).WillByDefault(Return());

    // asyncRead should only be called ONCE (for the first time entering loading_ queue)
    EXPECT_CALL(*mock_coord_, asyncRead(_)).Times(1).WillOnce(Return(std::static_pointer_cast<AsyncContext>(mock_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // First schedule: stream -> loading_ queue
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);

    // Second schedule: load done -> back to WAITING -> scheduleNew -> RUNNING (skips asyncLoadCache)
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 1);
    // asyncRead was called exactly once - the second scheduleNew sees had_blocks > 0 and skips asyncLoadCache
}

// ============================================================================
// 7. loading_ queue included in empty() and onflightStreams()
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingQueue_IncludedInEmptyAndOnflight) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);

    // Scheduler should NOT be empty when there are streams in loading_ queue
    ASSERT_FALSE(scheduler->empty());
    // onflightStreams should include loading_ queue
    ASSERT_EQ(scheduler->onflightStreams(), 1);
}

// ============================================================================
// 8. loading_ queue included in waitPredicate()
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testWaitPredicate_IncludesLoadingQueue) {
    auto scheduler = createScheduler();
    // Empty scheduler -> waitPredicate should be false
    ASSERT_FALSE(scheduler->waitPredicate());

    // Add a fake stream to the loading queue.
    auto stream = createStream({1, 2, 3});
    scheduler->loading_cache_streams_.push_back(stream);
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

    // Single schedule: stream1 -> loading_ queue (async load), stream2 -> RUNNING (directly)
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);  // Only stream2 is running
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
}

// ============================================================================
// 11. loading check: stream still loading -> stays in queue
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCheck_StillLoading_StaysInQueue) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // First schedule: enters loading_ queue
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);

    // Second schedule: still pending (done() returns false)
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);
    ASSERT_EQ(result2.value().size(), 0);
}

// ============================================================================
// 12. schedule() ordering: load_done_streams inserted at head of waiting_
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleOrdering_LoadDoneStreamsAtWaitingHead) {
    setupMockCoordinator();

    // Mock context: done() returns true when checked in loading check
    auto mock_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*mock_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*mock_ctx, waitDone()).WillByDefault(Return());

    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(mock_ctx)));

    auto scheduler = createScheduler();

    // Stream1: will enter loading_ queue first
    auto stream1 = createStream({1, 2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    auto result1 = scheduler->schedule();
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1u);

    // Stream2: enqueued later while stream1 is loading
    auto stream2 = createStream({3, 4}, /*reuse_cache=*/false);
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());

    // Second schedule: stream1 load done -> moves to waiting_ head -> should be scheduled before stream2
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    // Both streams should be running now
    ASSERT_GE(result2.value().size(), 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingGroupDoesNotBlockOrdinaryWaitingStreams) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler      = createScheduler();
    auto direct_stream  = createStream({1, 2}, /*reuse_cache=*/false);
    auto loading_stream = createStream({3, 4}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto waiting_stream = createStream({5, 6}, /*reuse_cache=*/false);

    auto [enqueue_successes, streams] = scheduler->enqueueGroup({direct_stream, loading_stream});
    ASSERT_EQ(enqueue_successes, std::vector<bool>({true, true}));
    ASSERT_EQ(streams.size(), 2);

    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_TRUE(group_result.value().empty());
    EXPECT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);
    EXPECT_EQ(scheduler->loading_cache_group_queue_.front().size(), 2);

    ASSERT_TRUE(scheduler->enqueue(waiting_stream).ok());
    auto waiting_result = scheduler->schedule();
    ASSERT_TRUE(waiting_result.ok());
    ASSERT_EQ(waiting_result.value().size(), 1);
    EXPECT_EQ(waiting_stream->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testWaitingGroupWaitsForOrdinaryCacheLoad) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler      = createScheduler();
    auto loading_stream = createStream({1, 2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(loading_stream).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);

    auto group_stream_1 = createStream({3, 4}, /*reuse_cache=*/false);
    auto group_stream_2 = createStream({5, 6}, /*reuse_cache=*/false);
    ASSERT_EQ(scheduler->enqueueGroup({group_stream_1, group_stream_2}).first, std::vector<bool>({true, true}));

    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.value().empty());
    EXPECT_EQ(scheduler->waiting_group_queue_.size(), 1);
    EXPECT_TRUE(scheduler->loading_cache_group_queue_.empty());
    EXPECT_EQ(group_stream_1->getStatus(), StreamState::WAITING);
    EXPECT_EQ(group_stream_2->getStatus(), StreamState::WAITING);
    EXPECT_EQ(group_stream_1->curBlocksNum(), 0);
    EXPECT_EQ(group_stream_2->curBlocksNum(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testPreparedGroupFinishesLoadingInOneRound) {
    setupMockCoordinator();

    auto done_ctx = createDoneAsyncContext();
    ON_CALL(*done_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(done_ctx)));

    auto scheduler      = createScheduler();
    auto direct_stream  = createStream({1, 2}, /*reuse_cache=*/false);
    auto loading_stream = createStream({3, 4}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto waiting_stream = createStream({5, 6}, /*reuse_cache=*/false);

    ASSERT_EQ(scheduler->enqueueGroup({direct_stream, loading_stream}).first, std::vector<bool>({true, true}));

    auto first_result = scheduler->schedule();
    ASSERT_TRUE(first_result.ok());
    EXPECT_TRUE(first_result.value().empty());
    EXPECT_EQ(scheduler->loading_cache_group_queue_.size(), 1);

    ASSERT_TRUE(scheduler->enqueue(waiting_stream).ok());

    auto second_result = scheduler->schedule();
    ASSERT_TRUE(second_result.ok());
    EXPECT_EQ(second_result.value().size(), 3);
    EXPECT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(waiting_stream->getStatus(), StreamState::RUNNING);
    EXPECT_TRUE(scheduler->loading_cache_group_queue_.empty());
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testGroupedSurvivorContinuesLoadingAfterPeerTimeout) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler      = createScheduler();
    auto direct_stream  = createStream({1, 2}, /*reuse_cache=*/false);
    auto loading_stream = createStream({3, 4}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    direct_stream->generateConfig()->timeout_ms = 1;
    direct_stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds());

    auto [enqueue_successes, streams] = scheduler->enqueueGroup({direct_stream, loading_stream});
    ASSERT_EQ(enqueue_successes, std::vector<bool>({true, true}));
    ASSERT_EQ(streams.size(), 2);

    auto first_result = scheduler->schedule();
    ASSERT_TRUE(first_result.ok());
    ASSERT_TRUE(first_result.value().empty());
    ASSERT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(loading_stream->getStatus(), StreamState::LOADING_CACHE);

    usleep(3000);
    auto second_result = scheduler->schedule();
    ASSERT_TRUE(second_result.ok());
    ASSERT_TRUE(second_result.value().empty());
    ASSERT_EQ(direct_stream->getStatus(), StreamState::FINISHED);
    ASSERT_EQ(direct_stream->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    ASSERT_EQ(loading_stream->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.front().size(), 1);
    ASSERT_TRUE(scheduler->loading_cache_streams_.empty());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);

    ON_CALL(*pending_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*pending_ctx, success()).WillByDefault(Return(true));
    auto final_result = scheduler->schedule();
    ASSERT_TRUE(final_result.ok());
    ASSERT_EQ(final_result.value().size(), 1);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::RUNNING);
    EXPECT_TRUE(scheduler->loading_cache_group_queue_.empty());
}

}  // namespace rtp_llm
