
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <vector>
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

class ObservableAsyncFIFOScheduler: public FIFOScheduler {
public:
    using FIFOScheduler::FIFOScheduler;

    bool waitUntilContextBatchWaitCount(size_t expected) {
        std::unique_lock<std::mutex> lock(wait_lock_);
        return wait_cond_.wait_for(
            lock, std::chrono::seconds(1), [this, expected] { return context_batch_wait_count_ >= expected; });
    }

    size_t contextBatchWaitCount() {
        std::lock_guard<std::mutex> lock(wait_lock_);
        return context_batch_wait_count_;
    }

    std::chrono::steady_clock::time_point contextBatchWaitEntryTime(size_t count) {
        std::lock_guard<std::mutex> lock(wait_lock_);
        return context_batch_wait_entry_times_.at(count - 1);
    }

protected:
    void onContextBatchCoalescingWait() override {
        {
            std::lock_guard<std::mutex> lock(wait_lock_);
            ++context_batch_wait_count_;
            context_batch_wait_entry_times_.push_back(std::chrono::steady_clock::now());
        }
        wait_cond_.notify_all();
    }

private:
    std::mutex              wait_lock_;
    std::condition_variable wait_cond_;
    size_t                  context_batch_wait_count_ = 0;
    std::vector<std::chrono::steady_clock::time_point> context_batch_wait_entry_times_;
};

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

    std::shared_ptr<ObservableAsyncFIFOScheduler> createScheduler(
        RoleType role_type = RoleType::PDFUSION, int64_t max_context_batch_size = 1, int64_t coalescing_window_ms = 0) {
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                     = 100;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
        runtime_config.fifo_scheduler_config.max_context_batch_size = max_context_batch_size;
        runtime_config.fifo_scheduler_config.context_batch_coalescing_window_ms = coalescing_window_ms;
        PDSepConfig         pd_sep_config;
        pd_sep_config.role_type = role_type;
        ParallelismConfig   parallelism_config;
        ModelSpecificConfig model_specific_config;
        return std::make_shared<ObservableAsyncFIFOScheduler>(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    }

    GenerateStreamPtr createStream(const std::vector<int>& input_tokens        = {1, 2, 3},
                                   bool                    reuse_cache         = false,
                                   bool                    enable_memory_cache = false,
                                   RoleType                role_type           = RoleType::PDFUSION,
                                   int                     batch_size          = 1) {
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
        generate_config->num_return_sequences = batch_size;
        query->input_ids                     = torch::tensor(input_tokens, torch::kInt32);
        query->generate_config               = generate_config;
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    }

    GenerateStreamPtr createForceStream(int64_t group_id,
                                        int     group_size,
                                        bool    reuse_cache,
                                        bool    aged = false,
                                        int     timeout_ms = 1) {
        ResourceContext resource_context;
        resource_context.cache_manager       = cache_manager_;
        resource_context.reuse_cache         = reuse_cache;
        resource_context.enable_memory_cache = reuse_cache;
        resource_context.role_type           = RoleType::PDFUSION;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;

        auto query                                      = std::make_shared<GenerateInput>();
        query->input_ids                                = torch::tensor({1}, torch::kInt32);
        query->generate_config                          = std::make_shared<GenerateConfig>();
        query->generate_config->reuse_cache             = reuse_cache;
        query->generate_config->enable_memory_cache     = reuse_cache;
        query->generate_config->force_batch             = true;
        query->generate_config->batch_group_timeout     = timeout_ms;
        query->batch_group_id                           = group_id;
        query->batch_group_size                         = group_size;
        query->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - (aged ? 1000 * 1000 : 0);
        auto stream = std::make_shared<NormalGenerateStream>(
            query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(true);
        return stream;
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

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadedStreamRequiresReadmissionAtBatchLimit) {
    setupMockCoordinator();

    ModelConfig model_config;
    model_config.max_seq_len = 8192;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    auto                scheduler = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);

    std::atomic<bool> load_done{false};
    auto              load_context = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*load_context, done()).WillByDefault(testing::Invoke([&load_done] { return load_done.load(); }));
    ON_CALL(*load_context, success()).WillByDefault(Return(true));
    ON_CALL(*load_context, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(1)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(load_context)));

    auto loaded_stream  = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto running_stream = createStream({2}, /*reuse_cache=*/false, /*enable_memory_cache=*/false);

    ASSERT_TRUE(scheduler->enqueue(loaded_stream).ok());
    auto first_result = scheduler->schedule();
    ASSERT_TRUE(first_result.ok());
    ASSERT_EQ(first_result.value().size(), 0);
    ASSERT_EQ(loaded_stream->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_TRUE(loaded_stream->hasEvent(StreamEvents::CanRun));

    ASSERT_TRUE(scheduler->enqueue(running_stream).ok());
    auto second_result = scheduler->schedule();
    ASSERT_TRUE(second_result.ok());
    ASSERT_EQ(second_result.value().size(), 1);
    ASSERT_EQ(running_stream->getStatus(), StreamState::RUNNING);

    load_done.store(true);
    auto third_result = scheduler->schedule();
    ASSERT_TRUE(third_result.ok());
    ASSERT_EQ(third_result.value().size(), 1);
    ASSERT_EQ(running_stream->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(loaded_stream->getStatus(), StreamState::WAITING);
    ASSERT_FALSE(loaded_stream->hasEvent(StreamEvents::CanRun));
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);

    running_stream->reportEvent(StreamEvents::GenerateDone);
    auto fourth_result = scheduler->schedule();
    ASSERT_TRUE(fourth_result.ok());
    ASSERT_EQ(fourth_result.value().size(), 1);
    ASSERT_EQ(loaded_stream->getStatus(), StreamState::RUNNING);

    loaded_stream->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadedContextDoesNotBypassOppositePhaseAdmission) {
    setupMockCoordinator();

    std::atomic<bool> load_done{false};
    auto              load_context = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*load_context, done()).WillByDefault(testing::Invoke([&load_done] { return load_done.load(); }));
    ON_CALL(*load_context, success()).WillByDefault(Return(true));
    ON_CALL(*load_context, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(1)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(load_context)));

    auto scheduler      = createScheduler();
    auto loaded_context = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto waiting_decode = createStream({2});
    loaded_context->setIsContextStream(true);
    waiting_decode->setIsContextStream(false);

    ASSERT_TRUE(scheduler->enqueue(loaded_context).ok());
    auto first_result = scheduler->schedule();
    ASSERT_TRUE(first_result.ok());
    ASSERT_TRUE(first_result->empty());
    ASSERT_EQ(loaded_context->getStatus(), StreamState::LOADING_CACHE);

    ASSERT_TRUE(scheduler->enqueue(waiting_decode).ok());
    load_done.store(true);
    auto second_result = scheduler->schedule();
    ASSERT_TRUE(second_result.ok());
    ASSERT_EQ(second_result->size(), 1);
    EXPECT_EQ(second_result->front(), waiting_decode);
    EXPECT_EQ(waiting_decode->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(loaded_context->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(loaded_context->hasEvent(StreamEvents::CanRun));

    waiting_decode->reportEvent(StreamEvents::GenerateDone);
    auto third_result = scheduler->schedule();
    ASSERT_TRUE(third_result.ok());
    ASSERT_EQ(third_result->size(), 1);
    EXPECT_EQ(third_result->front(), loaded_context);

    loaded_context->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncLoadCohortRegroupsStaggeredCompletions) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    constexpr int64_t window_ms = 500;
    auto scheduler = createScheduler(RoleType::PDFUSION, /*max_context_batch_size=*/4, window_ms);
    auto first     = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto second    = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());

    auto initial = scheduler->schedule();
    ASSERT_TRUE(initial.ok());
    ASSERT_TRUE(initial->empty());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 2);
    ASSERT_EQ(scheduler->contextBatchWaitCount(), 1);

    first_done->store(true);
    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_TRUE(scheduler->waitUntilContextBatchWaitCount(2));
    second_done->store(true);
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 2);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncLoadCohortDeadlineReleasesCompletedSubset) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    constexpr int64_t window_ms = 120;
    auto scheduler = createScheduler(RoleType::PDFUSION, /*max_context_batch_size=*/4, window_ms);
    auto first     = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto second    = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 2);
    ASSERT_EQ(scheduler->contextBatchWaitCount(), 1);

    first_done->store(true);
    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_TRUE(scheduler->waitUntilContextBatchWaitCount(2));
    const auto completion_wait_started = scheduler->contextBatchWaitEntryTime(2);
    EXPECT_EQ(completion.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), first);
    EXPECT_GE(std::chrono::steady_clock::now() - completion_wait_started,
              std::chrono::milliseconds(window_ms - 40));
}

TEST_F(FIFOSchedulerAsyncCacheTest, testSingleAsyncLoadCompletionDoesNotWaitForSecondWindow) {
    setupMockCoordinator();

    auto load_done = std::make_shared<std::atomic<bool>>(false);
    auto load_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*load_ctx, done()).WillByDefault(testing::Invoke([load_done] { return load_done->load(); }));
    ON_CALL(*load_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*load_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(1)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(load_ctx)));

    constexpr int64_t window_ms = 200;
    auto scheduler = createScheduler(RoleType::PDFUSION, /*max_context_batch_size=*/4, window_ms);
    auto stream    = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->contextBatchWaitCount(), 1);

    load_done->store(true);
    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 1);
    EXPECT_EQ(scheduler->contextBatchWaitCount(), 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncLoadCohortErrorBreaksCompletionWait) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    constexpr int64_t window_ms = 500;
    auto scheduler = createScheduler(RoleType::PDFUSION, /*max_context_batch_size=*/4, window_ms);
    auto first     = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto second    = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 2);
    ASSERT_EQ(scheduler->contextBatchWaitCount(), 1);

    first_done->store(true);
    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_TRUE(scheduler->waitUntilContextBatchWaitCount(2));
    second->reportError(ErrorCode::CANCELLED, "cancelled while loading");
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), first);
    EXPECT_TRUE(second->isFinished());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testCancellingReadyCohortMemberBreaksCompletionWait) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto third_done  = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    auto third_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*third_ctx, done()).WillByDefault(testing::Invoke([third_done] { return third_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*third_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*third_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(3)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(third_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/500);
    auto first = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto second = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto third = createStream({3}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->enqueue(third).ok());
    ASSERT_TRUE(scheduler->schedule().ok());

    first_done->store(true);
    second_done->store(true);
    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_TRUE(scheduler->waitUntilContextBatchWaitCount(2));
    first->reportError(ErrorCode::CANCELLED, "cancelled after cache load");
    ASSERT_EQ(completion.wait_for(std::chrono::milliseconds(250)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), second);
    EXPECT_TRUE(first->isFinished());
    EXPECT_EQ(second->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(third->getStatus(), StreamState::LOADING_CACHE);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testStopBreaksAsyncLoadCohortCompletionWait) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/500);
    auto first = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto second = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());

    first_done->store(true);
    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_TRUE(scheduler->waitUntilContextBatchWaitCount(2));
    ASSERT_TRUE(scheduler->stop().ok());
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_TRUE(first->isFinished());
    EXPECT_TRUE(second->isFinished());
    EXPECT_TRUE(scheduler->context_load_cohort_.empty());
    EXPECT_FALSE(scheduler->context_load_cohort_deadline_.has_value());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncLoadCohortReadyStreamsTakePriorityOverLateArrival) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/500);
    auto first = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto second = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 2);
    ASSERT_EQ(scheduler->contextBatchWaitCount(), 1);

    auto late = createStream({3},
                             /*reuse_cache=*/false,
                             /*enable_memory_cache=*/false,
                             RoleType::PDFUSION,
                             /*batch_size=*/3);
    ASSERT_TRUE(scheduler->enqueue(late).ok());
    auto decode = createStream({4});
    decode->setIsContextStream(false);
    ASSERT_TRUE(scheduler->enqueue(decode).ok());
    first_done->store(true);
    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_TRUE(scheduler->waitUntilContextBatchWaitCount(2));
    second_done->store(true);
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 2);
    EXPECT_EQ(result->front(), first);
    EXPECT_EQ(result->back(), second);
    EXPECT_EQ(late->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(late->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(decode->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(decode->hasEvent(StreamEvents::CanRun));
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncLoadCohortDoesNotWaitWhileExecutionIsRunning) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/500);
    auto first = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto second = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->contextBatchWaitCount(), 1);

    auto running = createStream({3});
    running->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(running->moveToNext(), StreamState::RUNNING);
    scheduler->running_streams_.push_back(running);
    first_done->store(true);

    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), running);
    EXPECT_EQ(scheduler->contextBatchWaitCount(), 1);
    EXPECT_FALSE(scheduler->context_load_cohort_deadline_.has_value());
    EXPECT_EQ(first->getStatus(), StreamState::WAITING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testEarlierDecodePhaseBypassesContextLoadCohortWait) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/500);
    auto first = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    auto second = createStream({2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->contextBatchWaitCount(), 1);

    auto decode = createStream({3});
    decode->setIsContextStream(false);
    ASSERT_TRUE(scheduler->enqueue(decode).ok());
    first_done->store(true);

    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), decode);
    EXPECT_EQ(scheduler->contextBatchWaitCount(), 1);
    EXPECT_FALSE(scheduler->context_load_cohort_deadline_.has_value());
    EXPECT_EQ(first->getStatus(), StreamState::WAITING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testEarlierDecodePrecedesCompletedSingleLoadCohort) {
    setupMockCoordinator();

    auto load_done = std::make_shared<std::atomic<bool>>(false);
    auto load_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*load_ctx, done()).WillByDefault(testing::Invoke([load_done] { return load_done->load(); }));
    ON_CALL(*load_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*load_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(1)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(load_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/200);
    auto loaded = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(loaded).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->contextBatchWaitCount(), 1);

    auto decode = createStream({2});
    decode->setIsContextStream(false);
    ASSERT_TRUE(scheduler->enqueue(decode).ok());
    load_done->store(true);

    auto completion = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = completion.get();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), decode);
    EXPECT_EQ(scheduler->contextBatchWaitCount(), 1);
    EXPECT_FALSE(scheduler->context_load_cohort_deadline_.has_value());
    EXPECT_EQ(loaded->getStatus(), StreamState::WAITING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAsyncLoadPhaseFlipClearsContextCohortMembership) {
    setupMockCoordinator();

    auto load_done = std::make_shared<std::atomic<bool>>(false);
    auto load_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*load_ctx, done()).WillByDefault(testing::Invoke([load_done] { return load_done->load(); }));
    ON_CALL(*load_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*load_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(1)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(load_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/200);
    auto loaded = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    ASSERT_TRUE(scheduler->enqueue(loaded).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(loaded->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->context_load_cohort_.count(loaded.get()), 1);

    auto late = createStream({2},
                             /*reuse_cache=*/false,
                             /*enable_memory_cache=*/false,
                             RoleType::PDFUSION,
                             /*batch_size=*/4);
    ASSERT_TRUE(scheduler->enqueue(late).ok());
    loaded->setIsContextStream(false);
    load_done->store(true);

    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), late);
    EXPECT_EQ(loaded->getStatus(), StreamState::WAITING);
    EXPECT_TRUE(scheduler->context_load_cohort_.empty());
    EXPECT_FALSE(scheduler->context_load_cohort_deadline_.has_value());

    late->reportEvent(StreamEvents::GenerateDone);
    auto decode_result = scheduler->schedule();
    ASSERT_TRUE(decode_result.ok());
    ASSERT_EQ(decode_result->size(), 1);
    EXPECT_EQ(decode_result->front(), loaded);
    EXPECT_TRUE(scheduler->context_load_cohort_.empty());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAtomicForceBatchWaitsForAllAsyncLoads) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/100);
    constexpr int64_t group_id = 71;
    auto first  = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    auto second = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    auto initial = scheduler->schedule();
    ASSERT_TRUE(initial.ok());
    ASSERT_TRUE(initial->empty());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 2);

    first_done->store(true);
    auto partial = scheduler->schedule();
    ASSERT_TRUE(partial.ok());
    EXPECT_TRUE(partial->empty());
    EXPECT_EQ(first->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(first->hasEvent(StreamEvents::CanRun));

    second_done->store(true);
    auto complete = scheduler->schedule();
    ASSERT_TRUE(complete.ok());
    ASSERT_EQ(complete->size(), 2);
    EXPECT_EQ(first->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(second->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testBrokenAgedAtomicForceBatchFallsBackWithoutDeadlock) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/100);
    constexpr int64_t group_id = 72;
    auto first  = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    auto second = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());

    first_done->store(true);
    second->reportError(ErrorCode::CANCELLED, "force peer cancelled while loading");
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->front(), first);
    EXPECT_TRUE(second->isFinished());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testUnagedBrokenForceBatchDoesNotReleaseHealthyPeerEarly) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/100);
    constexpr int64_t group_id = 76;
    auto first = createForceStream(
        group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/false, /*timeout_ms=*/10000);
    auto second = createForceStream(
        group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/false, /*timeout_ms=*/10000);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());

    first_done->store(true);
    second_done->store(true);
    scheduler->evaluateAndUpdateStreams(scheduler->loading_cache_streams_);
    ASSERT_EQ(first->getStatus(), StreamState::WAITING);
    ASSERT_EQ(second->getStatus(), StreamState::WAITING);
    ASSERT_EQ(scheduler->force_batch_admission_by_member_.size(), 2);

    first->reportError(ErrorCode::CANCELLED, "force peer cancelled after cache load");
    auto before_deadline = scheduler->schedule();
    ASSERT_TRUE(before_deadline.ok());
    EXPECT_TRUE(before_deadline->empty());
    EXPECT_TRUE(first->isFinished());
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
    ASSERT_EQ(scheduler->force_batch_admission_by_member_.size(), 1);
    EXPECT_TRUE(scheduler->force_batch_admission_by_member_.at(second.get())->broken);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testActiveAtomicForceBatchRejectsNewSameGroupGeneration) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/100);
    constexpr int64_t group_id = 73;
    auto first  = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    auto second = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());

    auto next_first  = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/false);
    auto next_second = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/false);
    ASSERT_TRUE(scheduler->enqueue(next_first).ok());
    ASSERT_TRUE(scheduler->enqueue(next_second).ok());
    first_done->store(true);

    auto partial = scheduler->schedule();
    ASSERT_TRUE(partial.ok());
    EXPECT_TRUE(partial->empty());
    EXPECT_EQ(first->getStatus(), StreamState::WAITING);
    EXPECT_EQ(next_first->getStatus(), StreamState::WAITING);
    EXPECT_EQ(next_second->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(next_first->hasEvent(StreamEvents::CanRun));
    EXPECT_FALSE(next_second->hasEvent(StreamEvents::CanRun));

    second_done->store(true);
    auto complete = scheduler->schedule();
    ASSERT_TRUE(complete.ok());
    ASSERT_EQ(complete->size(), 2);
    EXPECT_EQ(complete->front(), first);
    EXPECT_EQ(complete->back(), second);
    EXPECT_EQ(next_first->getStatus(), StreamState::WAITING);
    EXPECT_EQ(next_second->getStatus(), StreamState::WAITING);
    EXPECT_TRUE(scheduler->force_batch_admission_by_member_.empty());
    EXPECT_TRUE(scheduler->active_force_batch_group_ids_.empty());

    first->reportEvent(StreamEvents::GenerateDone);
    second->reportEvent(StreamEvents::GenerateDone);
    auto next = scheduler->schedule();
    ASSERT_TRUE(next.ok());
    ASSERT_EQ(next->size(), 2);
    EXPECT_EQ(next->front(), next_first);
    EXPECT_EQ(next->back(), next_second);
    EXPECT_TRUE(scheduler->force_batch_admission_by_member_.empty());
    EXPECT_TRUE(scheduler->active_force_batch_group_ids_.empty());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testPartialMemoryAdmissionDoesNotCreateForceIdentity) {
    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/100);
    scheduler->max_generate_batch_size_ = 1;

    constexpr int64_t group_id = 74;
    auto first  = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/false);
    auto second = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/false);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());

    scheduler->evaluateWaitingStreams(scheduler->waiting_streams_);
    EXPECT_TRUE(first->hasEvent(StreamEvents::CanRun));
    EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
    EXPECT_TRUE(scheduler->force_batch_admission_by_member_.empty());
    EXPECT_TRUE(scheduler->active_force_batch_group_ids_.empty());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAtomicForceBatchMixedPhaseFailsClosed) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/100);
    constexpr int64_t group_id = 75;
    auto first  = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    auto second = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());

    second->setIsContextStream(false);
    first_done->store(true);
    second_done->store(true);
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_EQ(first->getStatus(), StreamState::WAITING);
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(first->hasEvent(StreamEvents::CanRun));
    EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(scheduler->force_batch_admission_by_member_.size(), 2);
    EXPECT_EQ(scheduler->active_force_batch_group_ids_.count(group_id), 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testAtomicForceBatchAllDecodePhaseRemainsAtomic) {
    setupMockCoordinator();

    auto first_done  = std::make_shared<std::atomic<bool>>(false);
    auto second_done = std::make_shared<std::atomic<bool>>(false);
    auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
    auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
    ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
    ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
    ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
    ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(2)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

    auto scheduler = createScheduler(
        RoleType::PDFUSION, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/100);
    constexpr int64_t group_id = 77;
    auto first  = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    auto second = createForceStream(group_id, /*group_size=*/2, /*reuse_cache=*/true, /*aged=*/true);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(scheduler->schedule().ok());

    first->setIsContextStream(false);
    second->setIsContextStream(false);
    first_done->store(true);
    second_done->store(true);
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 2);
    EXPECT_EQ(result->front(), first);
    EXPECT_EQ(result->back(), second);
    EXPECT_TRUE(scheduler->force_batch_admission_by_member_.empty());
    EXPECT_TRUE(scheduler->active_force_batch_group_ids_.empty());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testDedicatedRolesDoNotWaitForAsyncLoadCohort) {
    setupMockCoordinator();

    for (const auto role_type : {RoleType::PREFILL, RoleType::DECODE}) {
        auto first_done  = std::make_shared<std::atomic<bool>>(false);
        auto second_done = std::make_shared<std::atomic<bool>>(false);
        auto first_ctx   = std::make_shared<NiceMock<MockAsyncContext>>();
        auto second_ctx  = std::make_shared<NiceMock<MockAsyncContext>>();
        ON_CALL(*first_ctx, done()).WillByDefault(testing::Invoke([first_done] { return first_done->load(); }));
        ON_CALL(*second_ctx, done()).WillByDefault(testing::Invoke([second_done] { return second_done->load(); }));
        ON_CALL(*first_ctx, success()).WillByDefault(Return(true));
        ON_CALL(*second_ctx, success()).WillByDefault(Return(true));
        ON_CALL(*first_ctx, waitDone()).WillByDefault(Return());
        ON_CALL(*second_ctx, waitDone()).WillByDefault(Return());
        EXPECT_CALL(*mock_coord_, asyncRead(_))
            .Times(2)
            .WillOnce(Return(std::static_pointer_cast<AsyncContext>(first_ctx)))
            .WillOnce(Return(std::static_pointer_cast<AsyncContext>(second_ctx)));

        auto scheduler = createScheduler(role_type, /*max_context_batch_size=*/4, /*coalescing_window_ms=*/500);
        auto first = createStream(
            {1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true, role_type);
        auto second = createStream(
            {2}, /*reuse_cache=*/true, /*enable_memory_cache=*/true, role_type);
        ASSERT_TRUE(scheduler->enqueue(first).ok());
        ASSERT_TRUE(scheduler->enqueue(second).ok());
        auto initial = scheduler->schedule();
        ASSERT_TRUE(initial.ok());
        ASSERT_TRUE(initial->empty());
        ASSERT_EQ(scheduler->loading_cache_streams_.size(), 2);
        ASSERT_EQ(scheduler->contextBatchWaitCount(), 0);

        first_done->store(true);
        auto completion = scheduler->schedule();
        ASSERT_TRUE(completion.ok());
        ASSERT_EQ(completion->size(), 1);
        EXPECT_EQ(completion->front(), first);
        EXPECT_EQ(second->getStatus(), StreamState::LOADING_CACHE);
        EXPECT_EQ(scheduler->contextBatchWaitCount(), 0);
        EXPECT_TRUE(first->hasEvent(StreamEvents::CanRun));
    }
}

TEST_F(FIFOSchedulerAsyncCacheTest, testNonFIFOLoadLoopRetainsCanRunAdmission) {
    setupMockCoordinator();

    std::atomic<bool> load_done{false};
    auto              load_context = std::make_shared<NiceMock<MockAsyncContext>>();
    ON_CALL(*load_context, done()).WillByDefault(testing::Invoke([&load_done] { return load_done.load(); }));
    ON_CALL(*load_context, success()).WillByDefault(Return(true));
    ON_CALL(*load_context, waitDone()).WillByDefault(Return());
    EXPECT_CALL(*mock_coord_, asyncRead(_))
        .Times(1)
        .WillOnce(Return(std::static_pointer_cast<AsyncContext>(load_context)));

    auto stream = createStream({1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true);
    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::LOADING_CACHE);
    ASSERT_TRUE(stream->hasEvent(StreamEvents::CanRun));

    load_done.store(true);
    ASSERT_EQ(stream->moveToNext(), StreamState::WAITING);
    ASSERT_TRUE(stream->hasEvent(StreamEvents::CanRun));
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);

    stream->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream->moveToNext(), StreamState::FINISHED);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testDedicatedFIFOLoadCompletionKeepsExistingAdmission) {
    setupMockCoordinator();

    for (const auto role_type : {RoleType::PREFILL, RoleType::DECODE}) {
        auto load_done    = std::make_shared<std::atomic<bool>>(false);
        auto load_context = std::make_shared<NiceMock<MockAsyncContext>>();
        ON_CALL(*load_context, done()).WillByDefault(testing::Invoke([load_done] { return load_done->load(); }));
        ON_CALL(*load_context, success()).WillByDefault(Return(true));
        ON_CALL(*load_context, waitDone()).WillByDefault(Return());
        EXPECT_CALL(*mock_coord_, asyncRead(_))
            .Times(1)
            .WillOnce(Return(std::static_pointer_cast<AsyncContext>(load_context)));

        auto scheduler = createScheduler(role_type);
        auto stream = createStream(
            {1}, /*reuse_cache=*/true, /*enable_memory_cache=*/true, role_type);
        ASSERT_TRUE(scheduler->enqueue(stream).ok());

        auto first_result = scheduler->schedule();
        ASSERT_TRUE(first_result.ok());
        ASSERT_TRUE(first_result->empty());
        ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
        ASSERT_TRUE(stream->hasEvent(StreamEvents::CanRun));

        load_done->store(true);
        auto second_result = scheduler->schedule();
        ASSERT_TRUE(second_result.ok());
        ASSERT_EQ(second_result->size(), 1);
        EXPECT_EQ(second_result->front(), stream);
        EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::CanRun));

        stream->reportEvent(StreamEvents::GenerateDone);
        ASSERT_TRUE(scheduler->schedule().ok());
    }
}

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

}  // namespace rtp_llm
