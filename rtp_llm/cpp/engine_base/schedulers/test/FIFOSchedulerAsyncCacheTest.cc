
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <thread>
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

namespace {

bool enqueueIndividually(FIFOScheduler& scheduler, const vector<GenerateStreamPtr>& streams) {
    return std::all_of(
        streams.begin(), streams.end(), [&scheduler](const auto& stream) { return scheduler.enqueue(stream).ok(); });
}

}  // namespace

class FIFOSchedulerAsyncCacheTest: public DeviceTestBase {
protected:
    FIFOSchedulerAsyncCacheTest():
        perf_scope("PERF_TEST", "1"), async_prepare_scope("RTP_LLM_ASYNC_PREPARE_CACHE", "0") {}

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

    std::shared_ptr<FIFOScheduler> createSchedulerWithInitedLimit(size_t max_inited_kv_cache_streams,
                                                                  size_t max_batch_size = 100) {
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                           = max_batch_size;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 8192;
        runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = max_inited_kv_cache_streams;
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
        resource_context.role_type           = RoleType::PDFUSION;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;

        std::shared_ptr<GenerateInput>  query(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->reuse_cache         = reuse_cache;
        generate_config->enable_memory_cache = enable_memory_cache;
        query->input_ids                     = torch::tensor(input_tokens, torch::kInt32);
        query->generate_config               = generate_config;
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
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!predicate() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return predicate();
    }

    bool blockedPrepareStreamIs(const std::shared_ptr<FIFOScheduler>& scheduler, const GenerateStreamPtr& stream) {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        return scheduler->cache_prepare_blocked_stream_ == stream;
    }

protected:
    autil::EnvGuard                                            perf_scope;
    autil::EnvGuard                                            async_prepare_scope;
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
// 5. LOADING_CACHE only consumes initialized-KV quota, not runtime batch quota
// ============================================================================

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCacheStreams_DoNotConsumeRuntimeBatchQuota) {
    setupMockCoordinator();

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 1;
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
    EXPECT_TRUE(result->empty());
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 3u);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCacheStreamsSupportBatchedCpPrefill) {
    setupMockCoordinator();

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    auto scheduler                              = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillRepeatedly(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    ASSERT_TRUE(scheduler->enqueue(createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true)).ok());
    ASSERT_TRUE(scheduler->enqueue(createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true)).ok());

    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 2u);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testWithoutCacheQuotaBoundsEachLoadingAdmissionRound) {
    setupMockCoordinator();

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                              = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size          = 8192;
    runtime_config.fifo_scheduler_config.max_batch_tokens_without_cache = 1;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    auto                scheduler = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillRepeatedly(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto stream1 = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto stream2 = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto stream3 = createStream({3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(enqueueIndividually(*scheduler, {stream1, stream2, stream3}));

    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 1u);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 2);
    EXPECT_EQ(stream2->getStatus(), StreamState::WAITING);
    EXPECT_EQ(stream2->curBlocksNum(), 0);

    auto next_result = scheduler->schedule();
    ASSERT_TRUE(next_result.ok());
    EXPECT_TRUE(next_result->empty());
    // The quota is per schedule round: existing LOADING_CACHE streams are not
    // carried into a new round's uncached admission accounting, so one more
    // stream can make progress on each pass without exceeding that round's cap.
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 2u);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 1);
    EXPECT_EQ(stream2->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_GT(stream2->curBlocksNum(), 0);
    EXPECT_EQ(stream3->getStatus(), StreamState::WAITING);
    EXPECT_EQ(stream3->curBlocksNum(), 0);

    auto third_result = scheduler->schedule();
    ASSERT_TRUE(third_result.ok());
    EXPECT_TRUE(third_result->empty());
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 3u);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(stream3->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_GT(stream3->curBlocksNum(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCacheStreams_RespectInitializedKVCacheQuota) {
    setupMockCoordinator();

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                           = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 1;
    runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = 2;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    auto                scheduler = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillRepeatedly(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    ASSERT_TRUE(scheduler->enqueue(createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true)).ok());
    ASSERT_TRUE(scheduler->enqueue(createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true)).ok());
    ASSERT_TRUE(scheduler->enqueue(createStream({3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true)).ok());

    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 2u);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 1);
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

TEST_F(FIFOSchedulerAsyncCacheTest, testOrdinaryCacheLoadDoesNotStarveWaitingGroup) {
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

    ON_CALL(*pending_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*pending_ctx, success()).WillByDefault(Return(true));
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    EXPECT_EQ(result.value().front(), loading_stream);
    EXPECT_EQ(scheduler->waiting_group_queue_.size(), 1);
    EXPECT_TRUE(scheduler->loading_cache_group_queue_.empty());
    EXPECT_EQ(group_stream_1->getStatus(), StreamState::WAITING);
    EXPECT_EQ(group_stream_2->getStatus(), StreamState::WAITING);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::RUNNING);

    loading_stream->reportEvent(StreamEvents::GenerateDone);
    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_EQ(group_result.value().size(), 2);
    EXPECT_EQ(group_stream_1->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(group_stream_2->getStatus(), StreamState::RUNNING);
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
    // The ready explicit group owns this execution boundary. The ordinary
    // stream remains queued instead of being mixed into the group batch.
    EXPECT_EQ(second_result.value().size(), 2);
    EXPECT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(waiting_stream->getStatus(), StreamState::WAITING);
    EXPECT_TRUE(scheduler->loading_cache_group_queue_.empty());
    EXPECT_EQ(scheduler->waitingStreamsSize(), 1);

    direct_stream->reportEvent(StreamEvents::GenerateDone);
    loading_stream->reportEvent(StreamEvents::GenerateDone);
    auto third_result = scheduler->schedule();
    ASSERT_TRUE(third_result.ok());
    ASSERT_EQ(third_result.value().size(), 1);
    EXPECT_EQ(waiting_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testReadyLoadingGroupDrainsNormalLaneBeforeDispatch) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler      = createScheduler();
    auto direct_stream  = createStream({1, 2}, /*reuse_cache=*/false);
    auto loading_stream = createStream({3, 4}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto normal_stream  = createStream({5, 6}, /*reuse_cache=*/false);
    auto normal_tail    = createStream({7, 8}, /*reuse_cache=*/false);

    ASSERT_EQ(scheduler->enqueueGroup({direct_stream, loading_stream}).first, std::vector<bool>({true, true}));
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);

    ASSERT_TRUE(scheduler->enqueue(normal_stream).ok());
    auto normal_result = scheduler->schedule();
    ASSERT_TRUE(normal_result.ok());
    ASSERT_EQ(normal_result.value().size(), 1);
    EXPECT_EQ(normal_stream->getStatus(), StreamState::RUNNING);

    // Cache completion makes the group ready while an ordinary batch is still
    // executing. A continuously arriving normal tail must not be admitted into
    // that batch; otherwise the group can never reach an isolated boundary.
    ON_CALL(*pending_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*pending_ctx, success()).WillByDefault(Return(true));
    ASSERT_TRUE(scheduler->enqueue(normal_tail).ok());
    auto drain_result = scheduler->schedule();
    ASSERT_TRUE(drain_result.ok());
    ASSERT_EQ(drain_result.value().size(), 1);
    EXPECT_EQ(normal_tail->getStatus(), StreamState::WAITING);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);

    normal_stream->reportEvent(StreamEvents::GenerateDone);
    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_EQ(group_result.value().size(), 2);
    EXPECT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(normal_tail->getStatus(), StreamState::WAITING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testOrdinaryCacheCompletionDoesNotMixIntoRunningGroup) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler        = createScheduler();
    auto ordinary_loading = createStream({1, 2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(ordinary_loading).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(ordinary_loading->getStatus(), StreamState::LOADING_CACHE);

    auto group_stream_1 = createStream({3, 4}, /*reuse_cache=*/false);
    auto group_stream_2 = createStream({5, 6}, /*reuse_cache=*/false);
    ASSERT_EQ(scheduler->enqueueGroup({group_stream_1, group_stream_2}).first, std::vector<bool>({true, true}));
    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_EQ(group_result->size(), 2);

    ON_CALL(*pending_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*pending_ctx, success()).WillByDefault(Return(true));
    auto isolated_result = scheduler->schedule();
    ASSERT_TRUE(isolated_result.ok());
    ASSERT_EQ(isolated_result->size(), 2);
    EXPECT_EQ(ordinary_loading->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 1);

    group_stream_1->reportEvent(StreamEvents::GenerateDone);
    group_stream_2->reportEvent(StreamEvents::GenerateDone);
    auto ordinary_result = scheduler->schedule();
    ASSERT_TRUE(ordinary_result.ok());
    ASSERT_EQ(ordinary_result->size(), 1);
    EXPECT_EQ(ordinary_loading->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testOrdinaryLoadingReleasesInitedLimitBeforeWaitingGroup) {
    setupMockCoordinator();

    auto pending_ctx = createPendingAsyncContext();
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(pending_ctx)));

    auto scheduler        = createSchedulerWithInitedLimit(1);
    auto ordinary_loading = createStream({1, 2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(ordinary_loading).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(ordinary_loading->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->countInitedKVCacheStreams(), 1);

    auto group_stream = createStream({3, 4}, /*reuse_cache=*/false);
    ASSERT_EQ(scheduler->enqueueGroup({group_stream}).first, std::vector<bool>({true}));

    // The group cannot initialize while the ordinary load owns the sole slot.
    // Advancing that load must still be allowed; otherwise both queues deadlock.
    ON_CALL(*pending_ctx, done()).WillByDefault(Return(true));
    ON_CALL(*pending_ctx, success()).WillByDefault(Return(true));
    auto ordinary_result = scheduler->schedule();
    ASSERT_TRUE(ordinary_result.ok());
    ASSERT_EQ(ordinary_result->size(), 1);
    EXPECT_EQ(ordinary_loading->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(group_stream->getStatus(), StreamState::WAITING);

    ordinary_loading->reportEvent(StreamEvents::GenerateDone);
    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_EQ(group_result->size(), 1);
    EXPECT_EQ(group_stream->getStatus(), StreamState::RUNNING);
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

TEST_F(FIFOSchedulerAsyncCacheTest, testBlockedPrepareUsesBoundedFallbackWithoutBusyRetry) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            holder    = createStream(std::vector<int>(16, 1));
    auto            blocked   = createStream(std::vector<int>(30, 2));

    ASSERT_TRUE(scheduler->enqueue(holder).ok());
    ASSERT_TRUE(waitUntil([&]() { return holder->hasEvent(StreamEvents::CachePrepared); }));
    auto holder_round = scheduler->schedule();
    ASSERT_TRUE(holder_round.ok());
    ASSERT_EQ(holder_round->size(), 1);

    ASSERT_TRUE(scheduler->enqueue(blocked).ok());
    ASSERT_TRUE(waitUntil([&]() { return blockedPrepareStreamIs(scheduler, blocked); }));

    std::chrono::steady_clock::time_point retry_at;
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        retry_at = scheduler->cache_prepare_retry_at_;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        EXPECT_EQ(scheduler->cache_prepare_blocked_stream_, blocked);
        EXPECT_EQ(scheduler->cache_prepare_retry_at_, retry_at);
    }

    // Resource snapshots are polled frequently, but stable shortage only
    // performs another allocation attempt after the backoff expires. A failed
    // retry re-arms the deadline instead of spinning.
    ASSERT_TRUE(waitUntil([&]() {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        return scheduler->cache_prepare_blocked_stream_ == blocked && scheduler->cache_prepare_retry_at_ > retry_at;
    }));

    holder->reportEvent(StreamEvents::GenerateDone);
    auto release_round = scheduler->schedule();
    ASSERT_TRUE(release_round.ok());
    EXPECT_TRUE(release_round->empty());
    ASSERT_TRUE(waitUntil([&]() { return blocked->hasEvent(StreamEvents::CachePrepared); }));
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
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(
        waitUntil([&]() { return stream->hasEvent(StreamEvents::LoadInitiated) && stream->curBlocksNum() > 0; }));
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);

    load_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return stream->hasEvent(StreamEvents::CachePrepared); }));
    auto running = scheduler->schedule();
    ASSERT_TRUE(running.ok());
    ASSERT_EQ(running->size(), 1);
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

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPrepareGroupStartsAtEnqueueAndWaitsForEveryMember) {
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
    auto first     = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto second    = createStream({4, 5, 6}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto [enqueue_successes, returned_streams] = scheduler->enqueueGroup({first, second});
    ASSERT_EQ(enqueue_successes, std::vector<bool>({true, true}));
    ASSERT_EQ(returned_streams, std::vector<GenerateStreamPtr>({first, second}));

    // Group preparation starts from enqueueGroup(), before schedule() is called.
    ASSERT_TRUE(waitUntil([&]() {
        return first->hasEvent(StreamEvents::LoadInitiated) && second->hasEvent(StreamEvents::LoadInitiated);
    }));
    EXPECT_EQ(first->getStatus(), StreamState::WAITING);
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);

    // A partially prepared group must not leak a partial execution batch.
    first_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return first->hasEvent(StreamEvents::CachePrepared); }));
    auto partial = scheduler->schedule();
    ASSERT_TRUE(partial.ok());
    EXPECT_TRUE(partial->empty());
    EXPECT_EQ(first->getStatus(), StreamState::WAITING);
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);

    second_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return second->hasEvent(StreamEvents::CachePrepared); }));
    auto complete = scheduler->schedule();
    ASSERT_TRUE(complete.ok());
    ASSERT_EQ(complete->size(), 2);
    EXPECT_EQ(first->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(second->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testPendingGroupCacheLoadYieldsToNormalFallback) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    setupMockCoordinator();

    std::atomic<bool> group_done{false};
    auto              group_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*group_ctx, done()).WillByDefault([&]() { return group_done.load(); });
    ON_CALL(*group_ctx, success()).WillByDefault(Return(true));
    EXPECT_CALL(*group_ctx, waitDone()).Times(1);
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(group_ctx)));

    auto scheduler = createScheduler(/*max_batch_size=*/1);
    auto running   = createStream({1, 2, 3});
    auto group     = createStream({4, 5, 6}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto tail      = createStream({7, 8, 9});

    ASSERT_TRUE(scheduler->enqueue(running).ok());
    ASSERT_TRUE(waitUntil([&]() { return running->hasEvent(StreamEvents::CachePrepared); }));
    ASSERT_EQ(scheduler->schedule()->size(), 1);

    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::LoadInitiated); }));
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(tail->hasEvent(StreamEvents::CachePrepared));

    running->reportEvent(StreamEvents::GenerateDone);
    auto group_not_ready_round = scheduler->schedule();
    ASSERT_TRUE(group_not_ready_round.ok());
    EXPECT_TRUE(group_not_ready_round->empty());

    ASSERT_TRUE(waitUntil([&]() { return tail->hasEvent(StreamEvents::CachePrepared); }));
    auto tail_round = scheduler->schedule();
    ASSERT_TRUE(tail_round.ok());
    ASSERT_EQ(tail_round->size(), 1);
    EXPECT_EQ(tail_round->front(), tail);
    EXPECT_EQ(group->getStatus(), StreamState::WAITING);

    tail->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    group_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared); }));
    auto group_round = scheduler->schedule();
    ASSERT_TRUE(group_round.ok());
    ASSERT_EQ(group_round->size(), 1);
    EXPECT_EQ(group_round->front(), group);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPrepareGroupSplitsAtKvShortageWithoutDeadlock) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/2);
    auto            prepared  = createStream(std::vector<int>(16, 1));
    auto            blocked   = createStream(std::vector<int>(30, 2));
    ASSERT_EQ(scheduler->enqueueGroup({prepared, blocked}).first, std::vector<bool>({true, true}));

    ASSERT_TRUE(waitUntil([&]() { return blockedPrepareStreamIs(scheduler, blocked); }));
    ASSERT_TRUE(prepared->hasEvent(StreamEvents::CachePrepared));
    ASSERT_FALSE(blocked->hasEvent(StreamEvents::CachePrepared));

    // Running the prepared prefix releases the KV blocks needed by the residual member.
    auto first_round = scheduler->schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round->size(), 1);
    EXPECT_EQ(first_round->front(), prepared);
    EXPECT_EQ(blocked->getStatus(), StreamState::WAITING);

    prepared->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(waitUntil([&]() { return blocked->hasEvent(StreamEvents::CachePrepared); }));

    auto second_round = scheduler->schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round->size(), 1);
    EXPECT_EQ(second_round->front(), blocked);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPrepareGroupShortageStillPreparesLaterSmallerMembers) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    cache_config_ = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/6, /*tokens_per_block=*/1, rtp_llm::DataType::TYPE_INT8);
    cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
    ASSERT_TRUE(cache_manager_->init());
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 5u);

    auto scheduler = createScheduler(/*max_batch_size=*/4);
    auto first     = createStream({1, 2, 3, 4});
    auto blocked   = createStream({5, 6, 7});
    auto smaller   = createStream({8});
    auto deferred  = createStream({9, 10});
    ASSERT_EQ(scheduler->enqueueGroup({first, blocked, smaller, deferred}).first,
              std::vector<bool>({true, true, true, true}));

    ASSERT_TRUE(waitUntil([&]() {
        return first->hasEvent(StreamEvents::CachePrepared) && smaller->hasEvent(StreamEvents::CachePrepared)
               && blockedPrepareStreamIs(scheduler, blocked);
    }));
    auto first_round = scheduler->schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round->size(), 2);
    EXPECT_EQ(first->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(smaller->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(blocked->getStatus(), StreamState::WAITING);
    EXPECT_EQ(deferred->getStatus(), StreamState::WAITING);

    first->reportEvent(StreamEvents::GenerateDone);
    smaller->reportEvent(StreamEvents::GenerateDone);
    auto release_round = scheduler->schedule();
    ASSERT_TRUE(release_round.ok());
    EXPECT_TRUE(release_round->empty());
    ASSERT_TRUE(waitUntil([&]() {
        return blocked->hasEvent(StreamEvents::CachePrepared) && deferred->hasEvent(StreamEvents::CachePrepared);
    }));
    auto second_round = scheduler->schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round->size(), 2);
    EXPECT_EQ(blocked->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(deferred->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLaterBlockerWakesFromGroupScanLowWaterMark) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    cache_config_ = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/6, /*tokens_per_block=*/1, rtp_llm::DataType::TYPE_INT8);
    cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
    ASSERT_TRUE(cache_manager_->init());
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 5u);

    // Hold three blocks outside the scheduler. The group then observes:
    // large fails at available=2, runnable consumes two, small fails at 0.
    auto external = createStream({1, 2, 3});
    ASSERT_EQ(external->prepareCache(), CachePrepareResult::DONE);
    ASSERT_EQ(cache_manager_->freeBlocksNum(), 2u);

    auto scheduler = createScheduler(/*max_batch_size=*/3);
    auto large     = createStream({4, 5, 6});
    auto runnable  = createStream({7, 8});
    auto small     = createStream({9});
    ASSERT_EQ(scheduler->enqueueGroup({large, runnable, small}).first, std::vector<bool>({true, true, true}));
    ASSERT_TRUE(waitUntil([&]() {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        return runnable->hasEvent(StreamEvents::CachePrepared) && scheduler->cache_prepare_blocked_streams_.size() == 2
               && scheduler->isCachePrepareBlocked(large) && scheduler->isCachePrepareBlocked(small);
    }));

    std::chrono::steady_clock::time_point retry_at;
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        ASSERT_EQ(scheduler->cache_prepare_blocked_available_blocks_, std::vector<size_t>({0}));
        retry_at = scheduler->cache_prepare_retry_at_;
    }
    auto runnable_round = scheduler->schedule();
    ASSERT_TRUE(runnable_round.ok());
    ASSERT_EQ(runnable_round->size(), 1);
    EXPECT_EQ(runnable_round->front(), runnable);

    runnable->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    while (!small->hasEvent(StreamEvents::CachePrepared) && std::chrono::steady_clock::now() < retry_at) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(small->hasEvent(StreamEvents::CachePrepared));

    external->releaseResource();
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPrepareAdvancesPrequeuedGroupsInOrder) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            first     = createStream({1, 2, 3});
    auto            second    = createStream({4, 5, 6});

    ASSERT_EQ(scheduler->enqueueGroup({first}).first, std::vector<bool>({true}));
    ASSERT_EQ(scheduler->enqueueGroup({second}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return first->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_FALSE(second->hasEvent(StreamEvents::CachePrepared));

    auto first_round = scheduler->schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round->size(), 1);
    EXPECT_EQ(first_round->front(), first);

    // Popping the first group must wake the worker for the group that was
    // already queued behind it; there is no new enqueue to provide a wakeup.
    ASSERT_TRUE(waitUntil([&]() { return second->hasEvent(StreamEvents::CachePrepared); }));
    first->reportEvent(StreamEvents::GenerateDone);
    auto second_round = scheduler->schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round->size(), 1);
    EXPECT_EQ(second_round->front(), second);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPrepareAlternatesNormalBetweenPrequeuedGroups) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            first     = createStream({1, 2, 3});
    auto            second    = createStream({4, 5, 6});
    auto            normal    = createStream({7, 8, 9});

    ASSERT_EQ(scheduler->enqueueGroup({first}).first, std::vector<bool>({true}));
    ASSERT_EQ(scheduler->enqueueGroup({second}).first, std::vector<bool>({true}));
    ASSERT_TRUE(scheduler->enqueue(normal).ok());
    ASSERT_TRUE(waitUntil([&]() { return normal->hasEvent(StreamEvents::CachePrepared); }));
    auto normal_round = scheduler->schedule();
    ASSERT_TRUE(normal_round.ok());
    ASSERT_EQ(normal_round->size(), 1);
    EXPECT_EQ(normal_round->front(), normal);

    ASSERT_TRUE(waitUntil([&]() { return first->hasEvent(StreamEvents::CachePrepared); }));
    normal->reportEvent(StreamEvents::GenerateDone);
    auto first_round = scheduler->schedule();
    ASSERT_TRUE(first_round.ok());
    ASSERT_EQ(first_round->size(), 1);
    EXPECT_EQ(first_round->front(), first);

    auto tail = createStream({10, 11, 12});
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    ASSERT_TRUE(waitUntil([&]() { return tail->hasEvent(StreamEvents::CachePrepared); }));
    first->reportEvent(StreamEvents::GenerateDone);
    auto tail_round = scheduler->schedule();
    ASSERT_TRUE(tail_round.ok());
    ASSERT_EQ(tail_round->size(), 1);
    EXPECT_EQ(tail_round->front(), tail);

    ASSERT_TRUE(waitUntil([&]() { return second->hasEvent(StreamEvents::CachePrepared); }));
    tail->reportEvent(StreamEvents::GenerateDone);
    auto second_round = scheduler->schedule();
    ASSERT_TRUE(second_round.ok());
    ASSERT_EQ(second_round->size(), 1);
    EXPECT_EQ(second_round->front(), second);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPrepareAdvancesAfterCancelledHeadGroup) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            cancelled = createStream({1, 2, 3});
    auto            next      = createStream({4, 5, 6});

    ASSERT_EQ(scheduler->enqueueGroup({cancelled}).first, std::vector<bool>({true}));
    ASSERT_EQ(scheduler->enqueueGroup({next}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return cancelled->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_FALSE(next->hasEvent(StreamEvents::CachePrepared));

    cancelled->reportError(ErrorCode::CANCELLED, "cancelled before group admission");
    auto cancelled_round = scheduler->schedule();
    ASSERT_TRUE(cancelled_round.ok());
    EXPECT_TRUE(cancelled_round->empty());

    ASSERT_TRUE(waitUntil([&]() { return next->hasEvent(StreamEvents::CachePrepared); }));
    auto next_round = scheduler->schedule();
    ASSERT_TRUE(next_round.ok());
    ASSERT_EQ(next_round->size(), 1);
    EXPECT_EQ(next_round->front(), next);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testCancelledPreferredGroupDoesNotLeakTurnAcrossIdleBoundary) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            running   = createStream({1, 2, 3});
    auto            cancelled = createStream({4, 5, 6});

    ASSERT_TRUE(scheduler->enqueue(running).ok());
    ASSERT_TRUE(waitUntil([&]() { return running->hasEvent(StreamEvents::CachePrepared); }));
    ASSERT_EQ(scheduler->schedule()->size(), 1);
    ASSERT_EQ(scheduler->enqueueGroup({cancelled}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return cancelled->hasEvent(StreamEvents::CachePrepared); }));

    auto barrier_round = scheduler->schedule();
    ASSERT_TRUE(barrier_round.ok());
    ASSERT_EQ(barrier_round->size(), 1);
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        ASSERT_TRUE(scheduler->prefer_group_next_);
    }

    cancelled->reportError(ErrorCode::CANCELLED, "cancel sole preferred group");
    running->reportEvent(StreamEvents::GenerateDone);
    auto idle_round = scheduler->schedule();
    ASSERT_TRUE(idle_round.ok());
    EXPECT_TRUE(idle_round->empty());
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        EXPECT_FALSE(scheduler->prefer_group_next_);
    }
}

TEST_F(FIFOSchedulerAsyncCacheTest, testCancelledPreparedGroupReleasesKvWhileNormalBatchRuns) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            running   = createStream({1, 2, 3});
    auto            group     = createStream({4, 5, 6});

    ASSERT_TRUE(scheduler->enqueue(running).ok());
    ASSERT_TRUE(waitUntil([&]() { return running->hasEvent(StreamEvents::CachePrepared); }));
    auto running_round = scheduler->schedule();
    ASSERT_TRUE(running_round.ok());
    ASSERT_EQ(running_round->size(), 1);

    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared) && group->curBlocksNum() > 0; }));
    group->reportError(ErrorCode::CANCELLED, "cancel prepared group while normal batch is running");

    auto cancel_round = scheduler->schedule();
    ASSERT_TRUE(cancel_round.ok());
    ASSERT_EQ(cancel_round->size(), 1);
    EXPECT_EQ(cancel_round->front(), running);
    EXPECT_EQ(group->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(group->curBlocksNum(), 0);
    EXPECT_TRUE(scheduler->waiting_group_queue_.empty());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testCancelledPreparedNormalTailReleasesKvWhileGroupRuns) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            group     = createStream({1, 2, 3});
    auto            tail      = createStream({4, 5, 6});

    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared); }));
    auto group_round = scheduler->schedule();
    ASSERT_TRUE(group_round.ok());
    ASSERT_EQ(group_round->size(), 1);

    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    ASSERT_TRUE(waitUntil([&]() { return tail->hasEvent(StreamEvents::CachePrepared) && tail->curBlocksNum() > 0; }));
    tail->reportError(ErrorCode::CANCELLED, "cancel prepared normal tail while group is running");

    auto cancel_round = scheduler->schedule();
    ASSERT_TRUE(cancel_round.ok());
    ASSERT_EQ(cancel_round->size(), 1);
    EXPECT_EQ(cancel_round->front(), group);
    EXPECT_EQ(tail->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(tail->curBlocksNum(), 0);
    EXPECT_TRUE(scheduler->waiting_streams_.empty());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testTokenRejectedGroupDoesNotReserveKvAndYieldsCachePrepareToNormal) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 3;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    auto                scheduler = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    auto group  = createStream({1, 2, 3});
    auto normal = createStream({4});

    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(group->hasEvent(StreamEvents::CachePrepared));
    EXPECT_EQ(group->curBlocksNum(), 0);
    auto blocked_round = scheduler->schedule();
    ASSERT_TRUE(blocked_round.ok());
    EXPECT_TRUE(blocked_round->empty());
    EXPECT_EQ(group->getStatus(), StreamState::WAITING);

    // The valid group remains waiting at the strict per-round token boundary,
    // but it must not prevent later ordinary work from preparing and running.
    ASSERT_TRUE(scheduler->enqueue(normal).ok());
    ASSERT_TRUE(waitUntil([&]() { return normal->hasEvent(StreamEvents::CachePrepared); }));
    auto normal_round = scheduler->schedule();
    ASSERT_TRUE(normal_round.ok());
    ASSERT_EQ(normal_round->size(), 1);
    EXPECT_EQ(normal_round->front(), normal);
    EXPECT_EQ(group->getStatus(), StreamState::WAITING);
    EXPECT_EQ(group->curBlocksNum(), 0);

    // Periodic polling must still finalize a statically blocked group on its
    // request timeout; otherwise it would remain invisible once no work runs.
    group->generateConfig()->timeout_ms = 1;
    group->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 10'000);
    normal->reportEvent(StreamEvents::GenerateDone);
    auto timeout_round = scheduler->schedule();
    ASSERT_TRUE(timeout_round.ok());
    EXPECT_TRUE(timeout_round->empty());
    EXPECT_EQ(group->getStatus(), StreamState::FINISHED);
    EXPECT_TRUE(scheduler->waiting_group_queue_.empty());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testStaticRejectedGroupMemberDoesNotBlockSurvivor) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 2;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 3;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    auto                scheduler = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    auto rejected = createStream({1, 2, 3});
    auto survivor = createStream({4});

    ASSERT_EQ(scheduler->enqueueGroup({rejected, survivor}).first, std::vector<bool>({true, true}));
    ASSERT_TRUE(waitUntil([&]() { return survivor->hasEvent(StreamEvents::CachePrepared); }));
    auto survivor_round = scheduler->schedule();
    ASSERT_TRUE(survivor_round.ok());
    ASSERT_EQ(survivor_round->size(), 1);
    EXPECT_EQ(survivor_round->front(), survivor);
    EXPECT_EQ(rejected->getStatus(), StreamState::WAITING);
    EXPECT_EQ(rejected->curBlocksNum(), 0);

    rejected->generateConfig()->timeout_ms = 1;
    rejected->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 10'000);
    auto timeout_round = scheduler->schedule();
    ASSERT_TRUE(timeout_round.ok());
    ASSERT_EQ(timeout_round->size(), 1);
    EXPECT_EQ(timeout_round->front(), survivor);
    EXPECT_EQ(rejected->getStatus(), StreamState::FINISHED);
    EXPECT_TRUE(scheduler->waiting_group_queue_.empty());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testStaticRejectedResidualDoesNotBlockNextNormalLane) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 2;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 3;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    auto                scheduler = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    auto rejected = createStream({1, 2, 3});
    auto survivor = createStream({4});
    auto tail     = createStream({6});

    ASSERT_EQ(scheduler->enqueueGroup({rejected, survivor}).first, std::vector<bool>({true, true}));
    ASSERT_TRUE(waitUntil([&]() { return survivor->hasEvent(StreamEvents::CachePrepared); }));
    auto survivor_round = scheduler->schedule();
    ASSERT_TRUE(survivor_round.ok());
    ASSERT_EQ(survivor_round->size(), 1);
    EXPECT_EQ(survivor_round->front(), survivor);

    rejected->generateConfig()->timeout_ms = 1;
    rejected->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 10'000);
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    ASSERT_TRUE(waitUntil([&]() { return tail->hasEvent(StreamEvents::CachePrepared); }));
    auto timeout_round = scheduler->schedule();
    ASSERT_TRUE(timeout_round.ok());
    ASSERT_EQ(timeout_round->size(), 1);
    EXPECT_EQ(timeout_round->front(), survivor);
    EXPECT_EQ(rejected->getStatus(), StreamState::FINISHED);

    survivor->reportEvent(StreamEvents::GenerateDone);
    auto tail_round = scheduler->schedule();
    ASSERT_TRUE(tail_round.ok());
    ASSERT_EQ(tail_round->size(), 1);
    EXPECT_EQ(tail_round->front(), tail);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testStaticRejectedNormalDoesNotBlockRunnableTail) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 2;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 3;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PDFUSION;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    auto                scheduler = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    auto rejected = createStream({1, 2, 3});
    auto tail     = createStream({4});

    ASSERT_TRUE(scheduler->enqueue(rejected).ok());
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    ASSERT_TRUE(waitUntil([&]() { return tail->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_FALSE(rejected->hasEvent(StreamEvents::CachePrepared));
    EXPECT_EQ(rejected->curBlocksNum(), 0);

    auto tail_round = scheduler->schedule();
    ASSERT_TRUE(tail_round.ok());
    ASSERT_EQ(tail_round->size(), 1);
    EXPECT_EQ(tail_round->front(), tail);
    EXPECT_EQ(rejected->getStatus(), StreamState::WAITING);

    rejected->generateConfig()->timeout_ms = 1;
    rejected->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 10'000);
    tail->reportEvent(StreamEvents::GenerateDone);
    auto timeout_round = scheduler->schedule();
    ASSERT_TRUE(timeout_round.ok());
    EXPECT_TRUE(timeout_round->empty());
    EXPECT_EQ(rejected->getStatus(), StreamState::FINISHED);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPreparePreservesNormalFirstLaneWhenGroupEnqueuedFirst) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            group     = createStream({1, 2, 3});
    auto            normal    = createStream({4, 5, 6});

    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    ASSERT_TRUE(scheduler->enqueue(normal).ok());
    ASSERT_TRUE(waitUntil([&]() { return normal->hasEvent(StreamEvents::CachePrepared); }));

    auto normal_round = scheduler->schedule();
    ASSERT_TRUE(normal_round.ok());
    ASSERT_EQ(normal_round->size(), 1);
    EXPECT_EQ(normal_round->front(), normal);

    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared); }));

    normal->reportEvent(StreamEvents::GenerateDone);
    auto group_round = scheduler->schedule();
    ASSERT_TRUE(group_round.ok());
    ASSERT_EQ(group_round->size(), 1);
    EXPECT_EQ(group_round->front(), group);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testFreshNormalKeepsDefaultBoundaryAheadOfPreparedGroup) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            group     = createStream({1, 2, 3});
    auto            normal    = createStream({4, 5, 6});

    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        return group->hasEvent(StreamEvents::CachePrepared) && !scheduler->cache_prepare_group_scan_in_progress_;
    }));
    {
        // Publish NORMAL without waking the worker so schedule() observes the
        // default-lane acquisition window deterministically.
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        scheduler->waiting_streams_.push_back(normal);
        scheduler->schedule_trigger_ = true;
    }

    auto protected_boundary = scheduler->schedule();
    ASSERT_TRUE(protected_boundary.ok());
    EXPECT_TRUE(protected_boundary->empty());
    EXPECT_EQ(group->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(normal->hasEvent(StreamEvents::CachePrepared));

    // enqueue() would normally provide this wakeup; the direct queue publish
    // above intentionally withheld it until after the protected boundary.
    scheduler->cond_.notify_all();
    ASSERT_TRUE(waitUntil([&]() { return normal->hasEvent(StreamEvents::CachePrepared); }));
    auto normal_round = scheduler->schedule();
    ASSERT_TRUE(normal_round.ok());
    ASSERT_EQ(normal_round->size(), 1);
    EXPECT_EQ(normal_round->front(), normal);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testBlockedHeadGroupKeepsAllocationBarrierAgainstNormalTail) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            holder    = createStream(std::vector<int>(16, 1));
    auto            group     = createStream(std::vector<int>(30, 2));
    auto            tail      = createStream(std::vector<int>(2, 3));

    ASSERT_TRUE(scheduler->enqueue(holder).ok());
    ASSERT_TRUE(waitUntil([&]() { return holder->hasEvent(StreamEvents::CachePrepared); }));
    auto holder_round = scheduler->schedule();
    ASSERT_TRUE(holder_round.ok());
    ASSERT_EQ(holder_round->size(), 1);

    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return blockedPrepareStreamIs(scheduler, group); }));
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(tail->hasEvent(StreamEvents::CachePrepared));
    EXPECT_EQ(tail->curBlocksNum(), 0);

    holder->reportEvent(StreamEvents::GenerateDone);
    auto release_round = scheduler->schedule();
    ASSERT_TRUE(release_round.ok());
    EXPECT_TRUE(release_round->empty());
    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared); }));

    auto group_round = scheduler->schedule();
    ASSERT_TRUE(group_round.ok());
    ASSERT_EQ(group_round->size(), 1);
    EXPECT_EQ(group_round->front(), group);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testFreshFrontGroupKeepsBoundaryAheadOfPreparedNormalTail) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            runner    = createStream({1, 2, 3});
    auto            tail      = createStream({4, 5, 6});
    auto            group     = createStream({7, 8, 9});

    ASSERT_TRUE(scheduler->enqueue(runner).ok());
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    ASSERT_TRUE(waitUntil([&]() {
        return runner->hasEvent(StreamEvents::CachePrepared) && tail->hasEvent(StreamEvents::CachePrepared);
    }));
    auto runner_round = scheduler->schedule();
    ASSERT_TRUE(runner_round.ok());
    ASSERT_EQ(runner_round->size(), 1);
    ASSERT_EQ(runner_round->front(), runner);

    runner->reportEvent(StreamEvents::GenerateDone);
    {
        // Publish the group without waking the worker so schedule() observes
        // the acquisition-before-worker window deterministically.
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        scheduler->waiting_group_queue_.push_back({group});
        scheduler->schedule_trigger_ = true;
    }

    auto protected_boundary = scheduler->schedule();
    ASSERT_TRUE(protected_boundary.ok());
    EXPECT_TRUE(protected_boundary->empty());
    EXPECT_EQ(tail->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(group->hasEvent(StreamEvents::CachePrepared));

    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared); }));
    auto group_round = scheduler->schedule();
    ASSERT_TRUE(group_round.ok());
    ASSERT_EQ(group_round->size(), 1);
    EXPECT_EQ(group_round->front(), group);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testInflightPrepareReservesGroupAllocationSlot) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createSchedulerWithInitedLimit(/*max_inited_kv_cache_streams=*/1);
    auto            reserved  = createStream({1, 2, 3});
    auto            group     = createStream({4, 5, 6});

    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        scheduler->cache_prepare_inflight_stream_ = reserved;
        EXPECT_EQ(scheduler->countInitedKVCacheStreams(), 1);
    }
    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));

    auto blocked_round = scheduler->schedule();
    ASSERT_TRUE(blocked_round.ok());
    EXPECT_TRUE(blocked_round->empty());
    EXPECT_EQ(group->getStatus(), StreamState::WAITING);
    EXPECT_EQ(group->curBlocksNum(), 0);
    ASSERT_TRUE(waitUntil([&]() { return blockedPrepareStreamIs(scheduler, group); }));

    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        scheduler->cache_prepare_inflight_stream_.reset();
        scheduler->clearCachePrepareBlocked();
        scheduler->schedule_trigger_ = true;
    }
    scheduler->cond_.notify_all();

    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared); }));
    auto admitted_round = scheduler->schedule();
    ASSERT_TRUE(admitted_round.ok());
    ASSERT_EQ(admitted_round->size(), 1);
    EXPECT_EQ(admitted_round->front(), group);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncPendingNormalReservesInitedSlotBeforeGroup) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    setupMockCoordinator();

    std::atomic<bool> load_done{false};
    auto              async_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*async_ctx, done()).WillByDefault([&]() { return load_done.load(); });
    ON_CALL(*async_ctx, success()).WillByDefault(Return(true));
    EXPECT_CALL(*async_ctx, waitDone()).Times(1);
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(async_ctx)));

    auto scheduler = createSchedulerWithInitedLimit(/*max_inited_kv_cache_streams=*/1);
    auto normal    = createStream({1, 2, 3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto group     = createStream({4, 5, 6});
    ASSERT_TRUE(scheduler->enqueue(normal).ok());
    ASSERT_TRUE(
        waitUntil([&]() { return normal->hasEvent(StreamEvents::LoadInitiated) && normal->curBlocksNum() > 0; }));
    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return blockedPrepareStreamIs(scheduler, group); }));
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        EXPECT_EQ(scheduler->countInitedKVCacheStreams(), 1);
    }
    EXPECT_FALSE(group->hasEvent(StreamEvents::CachePrepared));
    EXPECT_EQ(group->curBlocksNum(), 0);

    load_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return normal->hasEvent(StreamEvents::CachePrepared); }));
    auto normal_round = scheduler->schedule();
    ASSERT_TRUE(normal_round.ok());
    ASSERT_EQ(normal_round->size(), 1);
    EXPECT_EQ(normal_round->front(), normal);
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        EXPECT_EQ(scheduler->countInitedKVCacheStreams(), 1);
    }

    normal->reportEvent(StreamEvents::GenerateDone);
    auto release_round = scheduler->schedule();
    ASSERT_TRUE(release_round.ok());
    EXPECT_TRUE(release_round->empty());
    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared); }));
    auto group_round = scheduler->schedule();
    ASSERT_TRUE(group_round.ok());
    ASSERT_EQ(group_round->size(), 1);
    EXPECT_EQ(group_round->front(), group);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testWaitingHolderCancellationRetriesBlockedGroup) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createSchedulerWithInitedLimit(/*max_inited_kv_cache_streams=*/2,
                                                    /*max_batch_size=*/1);
    auto            runner    = createStream({0, 1, 2});
    auto            holder    = createStream({1, 2, 3});
    auto            group     = createStream({4, 5, 6});

    ASSERT_TRUE(scheduler->enqueue(runner).ok());
    ASSERT_TRUE(scheduler->enqueue(holder).ok());
    ASSERT_TRUE(waitUntil([&]() {
        return runner->hasEvent(StreamEvents::CachePrepared) && holder->hasEvent(StreamEvents::CachePrepared)
               && holder->curBlocksNum() > 0;
    }));
    auto runner_round = scheduler->schedule();
    ASSERT_TRUE(runner_round.ok());
    ASSERT_EQ(runner_round->size(), 1);
    ASSERT_EQ(runner_round->front(), runner);

    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return blockedPrepareStreamIs(scheduler, group); }));
    EXPECT_EQ(holder->getStatus(), StreamState::WAITING);
    EXPECT_EQ(scheduler->runningStreamsSize(), 1);

    // Releasing an initialized WAITING stream is a real resource change even
    // though the active NORMAL batch does not shrink.
    holder->reportError(ErrorCode::CANCELLED, "cancel waiting holder");
    auto cancelled_round = scheduler->schedule();
    ASSERT_TRUE(cancelled_round.ok());
    ASSERT_EQ(cancelled_round->size(), 1);
    EXPECT_EQ(cancelled_round->front(), runner);
    EXPECT_EQ(holder->getStatus(), StreamState::FINISHED);

    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared); }));
    runner->reportEvent(StreamEvents::GenerateDone);
    auto group_round = scheduler->schedule();
    ASSERT_TRUE(group_round.ok());
    ASSERT_EQ(group_round->size(), 1);
    EXPECT_EQ(group_round->front(), group);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testDelayedPdRefReleaseRetriesBlockedGroup) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            holder    = createStream(std::vector<int>(16, 1));
    auto            group     = createStream(std::vector<int>(30, 2));

    ASSERT_TRUE(scheduler->enqueue(holder).ok());
    ASSERT_TRUE(waitUntil([&]() { return holder->hasEvent(StreamEvents::CachePrepared); }));
    holder->holdKVCacheForPDSep();
    auto holder_round = scheduler->schedule();
    ASSERT_TRUE(holder_round.ok());
    ASSERT_EQ(holder_round->size(), 1);

    ASSERT_EQ(scheduler->enqueueGroup({group}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitUntil([&]() { return blockedPrepareStreamIs(scheduler, group); }));
    holder->reportEvent(StreamEvents::GenerateDone);
    auto release_request_ref_round = scheduler->schedule();
    ASSERT_TRUE(release_request_ref_round.ok());
    EXPECT_TRUE(release_request_ref_round->empty());

    // Request ownership is gone, but the delayed PD connector ref still holds
    // the blocks. The worker retries once, records a new zero-inited baseline,
    // and must then wait for allocator availability to increase.
    ASSERT_TRUE(waitUntil([&]() {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        return scheduler->cache_prepare_blocked_stream_ == group
               && scheduler->cache_prepare_blocked_inited_streams_ == 0;
    }));
    EXPECT_FALSE(group->hasEvent(StreamEvents::CachePrepared));

    holder->releaseKVCacheForPDSep();
    ASSERT_TRUE(waitUntil([&]() { return group->hasEvent(StreamEvents::CachePrepared); }));
    auto group_round = scheduler->schedule();
    ASSERT_TRUE(group_round.ok());
    ASSERT_EQ(group_round->size(), 1);
    EXPECT_EQ(group_round->front(), group);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testCancellationBehindUnpreparedHeadReleasesKvAndRetriesBlocker) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    setupMockCoordinator();

    std::atomic<bool> head_done{false};
    auto              head_ctx = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*head_ctx, done()).WillByDefault([&]() { return head_done.load(); });
    ON_CALL(*head_ctx, success()).WillByDefault(Return(true));
    EXPECT_CALL(*head_ctx, waitDone()).Times(1);
    EXPECT_CALL(*mock_coord_, asyncRead(_)).WillOnce(Return(std::static_pointer_cast<AsyncContext>(head_ctx)));

    auto scheduler = createScheduler(/*max_batch_size=*/2);
    auto head      = createStream(std::vector<int>(16, 1), /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto cancelled = createStream(std::vector<int>(16, 2));
    ASSERT_TRUE(scheduler->enqueue(head).ok());
    ASSERT_TRUE(scheduler->enqueue(cancelled).ok());
    ASSERT_TRUE(waitUntil([&]() {
        return head->hasEvent(StreamEvents::LoadInitiated) && cancelled->hasEvent(StreamEvents::CachePrepared)
               && cancelled->curBlocksNum() > 0;
    }));

    // Model a later retry of the FIFO head being resource-blocked by the
    // already-prepared tail. Pin the worker at the head while no prepare call
    // is in flight, then cancel the tail.
    ASSERT_TRUE(waitUntil([&]() {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        if (scheduler->cache_prepare_inflight_stream_) {
            return false;
        }
        scheduler->markCachePrepareBlocked(head);
        return true;
    }));
    cancelled->reportError(ErrorCode::CANCELLED, "cancel prepared stream behind blocked FIFO head");
    auto cancel_round = scheduler->schedule();
    ASSERT_TRUE(cancel_round.ok());
    EXPECT_TRUE(cancel_round->empty());
    EXPECT_EQ(cancelled->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(cancelled->curBlocksNum(), 0);

    head_done.store(true);
    ASSERT_TRUE(waitUntil([&]() { return head->hasEvent(StreamEvents::CachePrepared); }));
    auto head_round = scheduler->schedule();
    ASSERT_TRUE(head_round.ok());
    ASSERT_EQ(head_round->size(), 1);
    EXPECT_EQ(head_round->front(), head);
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
    ASSERT_TRUE(waitUntil([&]() { return blockedPrepareStreamIs(scheduler, blocked); }));
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
    ASSERT_TRUE(waitUntil([&]() { return blockedPrepareStreamIs(scheduler, blocked); }));

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

TEST_F(FIFOSchedulerAsyncCacheTest, testStopWakeDoesNotDispatchPreparedWaitingStream) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1);
    auto            stream    = createStream({1, 2, 3});

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(waitUntil([&]() { return stream->hasEvent(StreamEvents::CachePrepared); }));
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        // Model stop()'s join window: the stream remains queued after stop_ is
        // published, and a blocked schedule thread is awakened by the stop.
        scheduler->schedule_trigger_ = false;
    }

    bool        schedule_ok   = false;
    size_t      scheduled_num = std::numeric_limits<size_t>::max();
    std::thread schedule_thread([&]() {
        auto result = scheduler->schedule();
        schedule_ok = result.ok();
        if (result.ok()) {
            scheduled_num = result->size();
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    {
        std::lock_guard<std::mutex> lock(scheduler->lock_);
        scheduler->stop_ = true;
    }
    scheduler->cond_.notify_all();
    schedule_thread.join();

    EXPECT_TRUE(schedule_ok);
    EXPECT_EQ(scheduled_num, 0u);
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testDecodeRoleBypassesAsyncWorker) {
    autil::EnvGuard enable_async_prepare("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(/*max_batch_size=*/1, RoleType::DECODE);
    EXPECT_FALSE(scheduler->async_cache_prepare_enabled_);
}

}  // namespace rtp_llm
