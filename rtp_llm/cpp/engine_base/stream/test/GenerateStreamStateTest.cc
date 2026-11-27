#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/utils/test/DecodePublicationTestStartup.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <chrono>
#include <future>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"

#define private public
#define protected public
#include "rtp_llm/cpp/metrics/ServiceStatus.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockCoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/test/MockAsyncContext.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

namespace {

thread_local const AsyncContext* tracked_abort_context = nullptr;
thread_local size_t              tracked_abort_count   = 0;

void expectQps(kmonitor::MutableMetric* metric, const kmonitor::MetricsTags& tags, double expected) {
    ASSERT_NE(metric, nullptr);
    auto* series = metric->DeclareMetric(&tags);
    ASSERT_NE(series, nullptr);
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    series->Snapshot(&record, 1000);
    EXPECT_TRUE(metric->UndeclareMetric(series));
    ASSERT_EQ(record.Values().size(), 1u);
    EXPECT_DOUBLE_EQ(std::stod(record.Values().front()->Value()), expected);
}

}  // namespace

// KVCacheManager::abortPendingLoad is intentionally non-virtual. This target-local
// GNU ld --wrap seam counts only the AsyncContext selected by the current test
// and forwards every call to the real implementation. The Itanium ABI symbol is
// _ZN7rtp_llm14KVCacheManager16abortPendingLoadERKSt10shared_ptrINS_12AsyncContextEE.
extern "C" bool
realKVCacheManagerAbortPendingLoad(KVCacheManager* manager, const std::shared_ptr<AsyncContext>& context) asm(
    "__real__ZN7rtp_llm14KVCacheManager16abortPendingLoadERKSt10shared_ptrINS_12AsyncContextEE");

extern "C" bool
wrappedKVCacheManagerAbortPendingLoad(KVCacheManager* manager, const std::shared_ptr<AsyncContext>& context) asm(
    "__wrap__ZN7rtp_llm14KVCacheManager16abortPendingLoadERKSt10shared_ptrINS_12AsyncContextEE");

extern "C" bool wrappedKVCacheManagerAbortPendingLoad(KVCacheManager*                      manager,
                                                      const std::shared_ptr<AsyncContext>& context) {
    if (context.get() == tracked_abort_context) {
        ++tracked_abort_count;
    }
    return realKVCacheManagerAbortPendingLoad(manager, context);
}

class GenerateStreamStateTest: public DeviceTestBase {
protected:
    GenerateStreamStateTest(): perf_scope("PERF_TEST", "1") {
        // 降低显存配置以避免 OOM
        max_seq_len_ = 2048;
    }

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    GenerateStreamPtr createStream(const std::vector<int>& input_tokens = {1, 2, 3, 4, 5, 6},
                                   bool                    reuse_cache  = false,
                                   RoleType                role_type    = RoleType::PDFUSION,
                                   const CacheConfig*      config = nullptr) {
        cache_manager_ =
            std::make_shared<KVCacheManager>(config ? *config : init_config(), /*warmup=*/false, /*metrics_reporter=*/nullptr);
        EXPECT_TRUE(cache_manager_->init());
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache   = reuse_cache;
        resource_context.role_type     = role_type;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 1;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_tokens.begin(), input_tokens.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
    }

    std::shared_ptr<testing::NiceMock<MockCoordinatorCacheManager>> installRetryableInitMalloc() {
        auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
        ON_CALL(*allocator, totalBlocksNum()).WillByDefault(testing::Return(64));
        ON_CALL(*allocator, getNeedBlocks(testing::_)).WillByDefault(testing::Return(1));
        ON_CALL(*allocator, initMallocForCommonLen(testing::_))
            .WillByDefault(testing::Return(MallocResult{false, 0, 0, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED}));
        cache_manager_->coordinator_manager_ = allocator;
        return allocator;
    }

protected:
    autil::EnvGuard                 perf_scope;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

// ============================================================================
// 1. Initial state and basic state checks
// ============================================================================

TEST_F(GenerateStreamStateTest, testInitialStateIsWaiting) {
    auto stream = createStream();
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    ASSERT_FALSE(stream->isFinished());
    ASSERT_FALSE(stream->getStatus() == StreamState::RUNNING);
    ASSERT_FALSE(stream->getStatus() == StreamState::LOADING_CACHE);
}

TEST_F(GenerateStreamStateTest, testMallocRetryMetricsIndependentOfTerminalOutcome) {
    startKmonServiceStatus(nullptr);
    setKmonServiceServing(true);
    struct StopTestMetrics {
        ~StopTestMetrics() {
            stopKmonServiceStatus();
        }
    } stop_test_metrics;

    const struct {
        ErrorCode error;
        int       retries;
    } cases[] = {{ErrorCode::NONE_ERROR, 3},
                 {ErrorCode::MALLOC_FAILED, 3},
                 {ErrorCode::DECODE_MALLOC_FAILED, 3},
                 {ErrorCode::CANCELLED, 1},
                 {ErrorCode::GENERATE_TIMEOUT, 3},
                 {ErrorCode::MALLOC_FAILED, 0},
                 {ErrorCode::DECODE_MALLOC_FAILED, 0}};
    for (const auto& test : cases) {
        SCOPED_TRACE(testing::Message() << "error=" << static_cast<int>(test.error) << " retries=" << test.retries);
        auto tags     = kmonTagsWithServiceStatus(nullptr);
        auto reporter = std::make_shared<kmonitor::MetricsReporter>("", "", tags);
        auto stream   = createStream({1, 2, 3}, /*reuse_cache=*/false, RoleType::PREFILL);
        stream->setMetricsReporter(reporter);
        auto  allocator  = installRetryableInitMalloc();
        auto* operations = reporter->getMetricsGroup<RtpLLMCacheOperationMetrics>();
        stream->reportEvent(StreamEvents::CanRun);
        for (int i = 0; i < test.retries; ++i) {
            EXPECT_EQ(stream->moveToNext(), StreamState::WAITING);
        }
        // Retry decisions are visible while waiting, even if cancelled before the next attempt.
        expectQps(operations->malloc_retry_qps_metric, tags, test.retries);
        if (test.error == ErrorCode::NONE_ERROR) {
            ON_CALL(*allocator, initMallocForCommonLen(testing::_))
                .WillByDefault(testing::Return(MallocResult{true, 0}));
            ON_CALL(*allocator, incrMalloc(testing::_)).WillByDefault(testing::Return(MallocResult{true, 0}));
            EXPECT_EQ(stream->moveToNext(), StreamState::RUNNING);
            stream->reportEvent(StreamEvents::GenerateDone);
        } else {
            stream->reportError(test.error, "terminal error");
        }
        EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
        stream->reportMetricOnce();
        stream->reportMetricOnce();
        const auto stream_tag_map =
            buildStreamMetricTagMap(stream->priority(), test.error == ErrorCode::GENERATE_TIMEOUT);
        const kmonitor::MetricsTags stream_tags(stream_tag_map);
        const auto                  serving_stream_tags = kmonTagsWithServiceStatus(&stream_tags);
        stream.reset();
        expectQps(reporter->getMetricsGroup<RtpLLMStreamMetrics>()->kv_cache_malloc_failed_qps_metric,
                  serving_stream_tags,
                  test.error == ErrorCode::MALLOC_FAILED || test.error == ErrorCode::DECODE_MALLOC_FAILED);
        expectQps(operations->malloc_retry_qps_metric, tags, 0);
    }
}

TEST_F(GenerateStreamStateTest, testDecodeInitialCapacityPressureFailsImmediately) {
    auto stream    = createStream({1, 2, 3}, false, RoleType::DECODE);
    auto allocator = installRetryableInitMalloc();
    EXPECT_CALL(*allocator, initMallocForCommonLen(testing::_)).Times(1);
    stream->reportEvent(StreamEvents::CanRun);

    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
}

TEST_F(GenerateStreamStateTest, testDirectStateManipulation) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE via direct manipulation
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    ASSERT_FALSE(stream->getStatus() == StreamState::WAITING);
    ASSERT_FALSE(stream->getStatus() == StreamState::RUNNING);
    ASSERT_FALSE(stream->isFinished());
    ASSERT_EQ(stream->generate_status_->status, StreamState::LOADING_CACHE);
}

TEST_F(GenerateStreamStateTest, testLoadingCacheToWaitingTransition) {
    auto stream = createStream();
    // Set to LOADING_CACHE
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    // Transition back to WAITING
    stream->generate_status_->status = StreamState::WAITING;
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    ASSERT_FALSE(stream->getStatus() == StreamState::LOADING_CACHE);
}

// ============================================================================
// 2. WAITING -> LOADING_CACHE -> WAITING -> RUNNING -> FINISHED paths
// ============================================================================

TEST_F(GenerateStreamStateTest, testWaitingToLoadingCacheToWaitingPath) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    // LOADING_CACHE -> WAITING (load done, back to waiting for scheduling)
    stream->generate_status_->status = StreamState::WAITING;
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    // WAITING -> RUNNING (normal scheduling)
    stream->generate_status_->status = StreamState::RUNNING;
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);
}

TEST_F(GenerateStreamStateTest, testLoadingCacheToFinishedViaError) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    // Report error and set FINISHED during LOADING_CACHE
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testLoadingCacheToFinishedViaTimeout) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    // Error during LOADING_CACHE transitions to FINISHED
    stream->reportError(ErrorCode::GENERATE_TIMEOUT, "timeout during loading");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::GENERATE_TIMEOUT);
}

// ============================================================================
// 3. RUNNING -> FINISHED transitions
// ============================================================================

TEST_F(GenerateStreamStateTest, testRunningToFinishedNormal) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->generate_status_->status = StreamState::FINISHED;
    stream->fillSubGenerateStatus(StreamState::FINISHED);
    ASSERT_TRUE(stream->isFinished());
    ASSERT_FALSE(stream->getStatus() == StreamState::RUNNING);
}

TEST_F(GenerateStreamStateTest, testRunningToFinishedViaError) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->reportError(ErrorCode::MALLOC_FAILED, "OOM");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_TRUE(stream->hasError());
}

// ============================================================================
// 4. reportEvent(Error) / hasError / statusInfo
// ============================================================================

TEST_F(GenerateStreamStateTest, testReportErrorDoesNotChangeState) {
    auto stream = createStream();
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    // reportEvent(Error) only stores error, does NOT directly change state
    stream->reportError(ErrorCode::CANCELLED, "cancel from RPC");
    ASSERT_TRUE(stream->hasError());
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);  // state unchanged
    ASSERT_EQ(stream->generate_status_->status, StreamState::WAITING);
}

TEST_F(GenerateStreamStateTest, testReportErrorFirstWins) {
    auto stream = createStream();
    stream->reportError(ErrorCode::CANCELLED, "first error");
    stream->reportError(ErrorCode::GENERATE_TIMEOUT, "second error");
    auto err = stream->statusInfo();
    // First error should win
    ASSERT_EQ(err.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testReportErrorAndSetFinished_Waiting) {
    auto stream = createStream();
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    stream->reportError(ErrorCode::MALLOC_FAILED, "OOM");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->generate_status_->status, StreamState::FINISHED);
}

TEST_F(GenerateStreamStateTest, testReportErrorAndReleaseResource) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    auto& resource                   = stream->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_GT(resource.curBlocksNum(), 0);

    stream->reportError(ErrorCode::MALLOC_FAILED, "OOM");
    stream->generate_status_->status = StreamState::FINISHED;
    stream->releaseResource();
    ASSERT_TRUE(stream->isFinished());
    ASSERT_TRUE(stream->hasError());
}

// ============================================================================
// 5. reportEvent(Error) + state set replaces cancelIfNotRunning
// ============================================================================

TEST_F(GenerateStreamStateTest, testReportErrorAndSetFinished_LoadingCache) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testReportError_Running_StateUnchanged) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    // reportEvent(Error) should NOT change state
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);
    ASSERT_TRUE(stream->hasError());
}

TEST_F(GenerateStreamStateTest, testReportErrorAndSetFinished) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

// ============================================================================
// 6. Use reportEvent(Error) to cancel (external error reporting)
// ============================================================================

TEST_F(GenerateStreamStateTest, testCancelUsesReportError) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    ASSERT_TRUE(stream->hasError());
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);
    auto err = stream->statusInfo();
    ASSERT_EQ(err.code(), ErrorCode::CANCELLED);
}

// ============================================================================
// 7. Complete state machine cycle: WAITING -> LOADING_CACHE -> WAITING -> RUNNING -> FINISHED
// ============================================================================

TEST_F(GenerateStreamStateTest, testFullStateMachineCycle) {
    auto stream = createStream();
    // Initial: WAITING
    ASSERT_EQ(stream->generate_status_->status, StreamState::WAITING);

    // WAITING -> LOADING_CACHE
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_EQ(stream->generate_status_->status, StreamState::LOADING_CACHE);

    // LOADING_CACHE -> WAITING (load done)
    stream->generate_status_->status = StreamState::WAITING;
    ASSERT_EQ(stream->generate_status_->status, StreamState::WAITING);

    // WAITING -> RUNNING
    stream->generate_status_->status = StreamState::RUNNING;
    ASSERT_EQ(stream->generate_status_->status, StreamState::RUNNING);

    // RUNNING -> FINISHED
    stream->generate_status_->status = StreamState::FINISHED;
    stream->fillSubGenerateStatus(StreamState::FINISHED);
    ASSERT_EQ(stream->generate_status_->status, StreamState::FINISHED);
}

// ============================================================================
// 8. StreamStateToString
// ============================================================================

TEST_F(GenerateStreamStateTest, testStreamStateToString) {
    ASSERT_EQ(StreamStateToString(StreamState::WAITING), "WAITING");
    ASSERT_EQ(StreamStateToString(StreamState::LOADING_CACHE), "LOADING_CACHE");
    ASSERT_EQ(StreamStateToString(StreamState::RUNNING), "RUNNING");
    ASSERT_EQ(StreamStateToString(StreamState::FINISHED), "FINISHED");
}

// ============================================================================
// 9. LoadInitiated event: Verify Decode mode cache load fix
// ============================================================================

TEST_F(GenerateStreamStateTest, testLoadInitiatedPreventsDuplicateInitKVBlock) {
    auto stream = createStream();
    ASSERT_EQ(stream->getStatus(), StreamState::WAITING);

    // Simulate DecodeRpcServer: call initKVBlock directly and set LoadInitiated
    auto& resource = stream->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    stream->reportEvent(StreamEvents::LoadInitiated);

    // FIFOScheduler calls moveToNext, should skip initKVBlock and asyncLoadCache
    auto new_state = stream->moveToNext();
    // Should stay in WAITING because CanRun is not set yet
    ASSERT_EQ(new_state, StreamState::WAITING);

    // Now simulate FIFOScheduler setting CanRun
    stream->reportEvent(StreamEvents::CanRun);
    new_state = stream->moveToNext();
    ASSERT_EQ(new_state, StreamState::RUNNING);
}

TEST_F(GenerateStreamStateTest, testLoadInitiatedSkipsAsyncLoadCache) {
    auto stream = createStream();
    ASSERT_EQ(stream->getStatus(), StreamState::WAITING);

    // Simulate DecodeRpcServer: only initKVBlock, no asyncLoadCache
    auto& resource = stream->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    stream->reportEvent(StreamEvents::LoadInitiated);

    // No allocator-owned load is in flight.
    ASSERT_FALSE(resource.allocator_load_context_);

    // moveToNext should not trigger asyncLoadCache because LoadInitiated is set
    stream->reportEvent(StreamEvents::CanRun);
    auto new_state = stream->moveToNext();
    ASSERT_EQ(new_state, StreamState::RUNNING);

    // Still no allocator load context.
    ASSERT_FALSE(resource.allocator_load_context_);
}

TEST_F(GenerateStreamStateTest, testPrefillFallbackDecodeGrowsBlocksAfterContext) {
    auto stream = createStream({1, 2}, /*reuse_cache=*/false, RoleType::PREFILL);

    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    ASSERT_TRUE(stream->isContextStream());
    ASSERT_EQ(stream->curBlocksNum(), 1u);

    // PD fallback can continue decoding in the PREFILL role. The next decode
    // token crosses the two-token test block boundary and needs another column.
    stream->setIsContextStream(false);
    stream->setSeqLength(3);

    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    EXPECT_EQ(stream->curBlocksNum(), 2u);
}

TEST_F(GenerateStreamStateTest, testNormalAsyncHostCommitPrecedesPageBoundaryReservation) {
    auto stream = createStream({1, 2});
    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    ASSERT_EQ(stream->curBlocksNum(), 1u);
    stream->setIsContextStream(false);
    stream->incPendingAsyncBookkeeping();
    EXPECT_FALSE(stream->prepareMtpCacheForNextStep());

    std::promise<void> scheduling_started;
    auto               scheduling_ready = scheduling_started.get_future();
    auto               scheduling       = std::async(std::launch::async, [stream, &scheduling_started] {
        scheduling_started.set_value();
        return stream->moveToNext();
    });
    scheduling_ready.get();
    EXPECT_EQ(scheduling.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);

    // The previous dispatch worker commits a token across the two-token page.
    // This happens before dropping its resource claim, as in NormalExecutor.
    stream->setSeqLength(3);
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
    ASSERT_EQ(scheduling.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(scheduling.get(), StreamState::RUNNING);
    EXPECT_EQ(stream->curBlocksNum(), 2u);
}

TEST_F(GenerateStreamStateTest, testNormalAsyncPublishedLengthReservesBeforeHostCommit) {
    auto stream = createStream({1, 2});
    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    ASSERT_EQ(stream->curBlocksNum(), 1u);
    stream->setIsContextStream(false);

    GenerateStream::NormalAsyncDeviceState state;
    state.last_real_seq_len = 2;
    state.next_real_seq_len = 3;
    stream->setNormalAsyncDeviceState(std::move(state));
    stream->incPendingAsyncBookkeeping();
    EXPECT_TRUE(stream->prepareMtpCacheForNextStep());
    EXPECT_EQ(stream->moveToNext(), StreamState::RUNNING);
    EXPECT_EQ(stream->seqLength(), 2u);
    EXPECT_EQ(stream->curBlocksNum(), 2u);
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
}

TEST_F(GenerateStreamStateTest, testIncrementalAsyncAllocationTerminatesBeforeModelAndReleasesOnce) {
    using ::testing::_;
    using ::testing::Return;

    auto stream = createStream({1, 2}, /*reuse_cache=*/false, RoleType::PREFILL);
    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    ASSERT_EQ(stream->curBlocksNum(), 1u);

    const auto real_allocator = cache_manager_->coordinator_manager_;
    ASSERT_EQ(real_allocator->groupBlockPools().size(), 1u);
    const auto device_pool = real_allocator->groupBlockPools().front();
    ASSERT_NE(device_pool, nullptr);
    const auto& request_blocks = stream->streamCacheResource().kvCache().blocks(/*batch_id=*/0, "default");
    ASSERT_EQ(request_blocks.size(), 1u);
    const BlockIdxType request_block = request_blocks.front();
    ASSERT_EQ(device_pool->refCount(request_block), 1u);
    const size_t free_after_request_alloc = device_pool->freeBlocksNum();

    auto   async_context  = std::make_shared<MockAsyncContext>();
    auto   mock_allocator = std::make_shared<MockCoordinatorCacheManager>(init_config());
    size_t free_count     = 0;
    EXPECT_CALL(*mock_allocator, incrMalloc(_))
        .WillOnce(Return(MallocResult{/*success=*/true,
                                      /*reuse_len=*/0,
                                      /*match_cost_time_us=*/0,
                                      async_context}));
    EXPECT_CALL(*mock_allocator, free(_)).WillOnce([&](const FreeInfo& free_info) {
        ++free_count;
        real_allocator->free(free_info);
    });
    cache_manager_->coordinator_manager_ = mock_allocator;

    stream->setIsContextStream(false);
    stream->setSeqLength(3);
    tracked_abort_context = async_context.get();
    tracked_abort_count   = 0;

    // A model executor may only consume RUNNING streams. The state-machine
    // caller must make the unexpected incremental async result terminal first.
    size_t     model_entry_count     = 0;
    const auto run_model_if_eligible = [&]() {
        if (stream->getStatus() == StreamState::RUNNING) {
            ++model_entry_count;
        }
    };

    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    run_model_if_eligible();
    EXPECT_TRUE(stream->isFinished());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    EXPECT_EQ(stream->firstError().error.code(), ErrorCode::MALLOC_FAILED);
    stream->reportError(ErrorCode::CANCELLED, "later cancellation");
    EXPECT_EQ(stream->firstError().error.code(), ErrorCode::MALLOC_FAILED);
    EXPECT_NE(stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(model_entry_count, 0u);
    EXPECT_EQ(tracked_abort_count, 1u);
    EXPECT_EQ(free_count, 1u);
    EXPECT_EQ(stream->curBlocksNum(), 0u);
    EXPECT_FALSE(device_pool->isAllocated(request_block));
    EXPECT_EQ(device_pool->freeBlocksNum(), free_after_request_alloc + 1);
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());

    // Re-entering the terminal state and an explicit stream release are both
    // idempotent: neither the observer nor request blocks are released twice.
    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    stream->releaseResource();
    EXPECT_EQ(tracked_abort_count, 1u);
    EXPECT_EQ(free_count, 1u);
    EXPECT_FALSE(device_pool->isAllocated(request_block));
    EXPECT_EQ(device_pool->freeBlocksNum(), free_after_request_alloc + 1);

    tracked_abort_context = nullptr;
    tracked_abort_count   = 0;
}

TEST_F(GenerateStreamStateTest, testNormalPathTriggersAsyncLoadCache) {
    // Create stream with reuse_cache enabled to trigger asyncLoadCache
    auto stream = createStream({1, 2, 3, 4, 5, 6}, /*reuse_cache=*/true);
    ASSERT_EQ(stream->getStatus(), StreamState::WAITING);

    // Normal path: moveToNext should trigger initKVBlock + asyncLoadCache
    auto new_state = stream->moveToNext();

    // Should transition to LOADING_CACHE if asyncLoadCache was initiated
    // or stay in WAITING if no connectors are available
    ASSERT_TRUE(new_state == StreamState::LOADING_CACHE || new_state == StreamState::WAITING);
}


TEST_F(GenerateStreamStateTest, AsyncRequestLifecycleEndsOnlyAtCommittedTerminal) {
    auto session = AsyncProfileSession::start(32);
    auto stream = createStream();
    stream->reportError(ErrorCode::CANCELLED, "trace cancellation");
    EXPECT_NE(stream->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    stream->moveToNext();
    stream.reset();
    const auto events = session->finish();
    size_t roots = 0, cancelled = 0, released = 0;
    for (const auto& e : events) {
        if (e.record.path != "request") continue;
        roots += e.record.stage == "request_lifetime";
        cancelled += e.record.stage == "cancelled";
        released += e.record.stage == "released_without_terminal";
        EXPECT_FALSE(e.end_clipped);
    }
    EXPECT_EQ(roots, 1u);
    EXPECT_EQ(cancelled, 1u);
    EXPECT_EQ(released, 0u);
}
TEST_F(GenerateStreamStateTest, UnknownWriterWithoutNativeCustodyRetainsRequestReferences) {
    auto stream = createStream({1, 2, 3, 4}, false, RoleType::DECODE);
    stream->generate_status_->status = StreamState::RUNNING;
    ASSERT_TRUE(stream->initKVBlock().ok());
    auto& resource = stream->streamCacheResource();
    const auto blocks = resource.kvCache().blocks(0, "default");
    auto manager = resource.resource_context_.cache_manager;
    block_tree_cache_detail::ScopeRollback restore([&] {
        resource.resource_context_.cache_manager = manager;
        stream->unknown_decode_publication_.reset();
        stream->async_bookkeeping_->defer_release.store(false);
        stream->releaseResource();
    });
    resource.resource_context_.cache_manager.reset(); // Model-writer freeze cannot be proven.
    stream->freezeUnknownDecodeWriter({});
    EXPECT_EQ(stream->unknown_decode_publication_.get(), stream.get());
    EXPECT_FALSE(stream->unknown_decode_kv_frozen_);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::Error));
    const auto first_error = stream->firstError();
    EXPECT_EQ(first_error.error.code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_GT(first_error.order, 0u);
    stream->reportError(ErrorCode::CANCELLED, "later cancellation");
    EXPECT_EQ(stream->firstError().order, first_error.order);
    EXPECT_EQ(stream->firstError().error.code(), ErrorCode::EXECUTION_EXCEPTION);
    stream->releaseResource();
    EXPECT_FALSE(resource.isResourceReleased());
    EXPECT_EQ(resource.kvCache().blocks(0, "default"), blocks);
}

TEST_F(GenerateStreamStateTest, LastSubmissionNotifiesCleanupAfterDeferredReleaseWithoutWorker) {
    auto stream = createStream({1, 2, 3, 4}, false, RoleType::DECODE);
    stream->generate_status_->status = StreamState::RUNNING;
    ASSERT_TRUE(stream->initKVBlock().ok());
    ASSERT_TRUE(stream->beginDecodePublicationSubmission());
    std::vector<bool> pending;
    std::weak_ptr<GenerateStream> weak_stream = stream;
    stream->onLocalCleanupProgress([&pending, weak_stream] {
        if (const auto owner = weak_stream.lock()) pending.push_back(owner->hasPendingLocalCleanup());
    });
    stream->reportError(ErrorCode::CANCELLED, "cancel during submission");
    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    ASSERT_FALSE(pending.empty());
    EXPECT_TRUE(pending.back());
    const auto notifications = pending.size();
    stream->endDecodePublicationSubmission();
    EXPECT_GT(pending.size(), notifications);
    EXPECT_FALSE(pending.back());
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
    EXPECT_FALSE(stream->isDeferredReleasePending());
    stream.reset();
}

TEST_F(GenerateStreamStateTest, SubmissionAloneDefersSchedulerReleaseUntilItsHandoffEnds) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    test::withDecodePublicationStartup([&] {
        for (bool cancelled : {false, true}) {
            SCOPED_TRACE(cancelled);
            auto stream = createStream({1, 2, 3, 4}, false, RoleType::DECODE);
            stream->generate_status_->status = StreamState::RUNNING;
            auto& resource = stream->streamCacheResource();
            ASSERT_TRUE(resource.initKVBlock().ok());
            ASSERT_EQ(cache_manager_->coordinator_manager_->groupBlockPools().size(), 1u);
            auto pool = cache_manager_->coordinator_manager_->groupBlockPools().front();
            const auto blocks = resource.kvCache().blocks(0, "default");
            ASSERT_FALSE(blocks.empty());
            ASSERT_FALSE(stream->hasPendingAsyncBookkeeping());
            ASSERT_TRUE(stream->beginDecodePublicationSubmission());
            auto submission = std::shared_ptr<void>(nullptr, [stream](void*) {
                stream->endDecodePublicationSubmission();
            }); // Also drains the test-only claim after a fatal assertion.
            if (cancelled) stream->reportError(ErrorCode::CANCELLED, "cancel during handoff");
            else stream->reportEvent(StreamEvents::GenerateDone);
            EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
            EXPECT_FALSE(resource.isResourceReleased());
            EXPECT_TRUE(stream->isDeferredReleasePending());
            stream->releaseResource(); // No bookkeeping wait; the submission still protects the pages.
            EXPECT_FALSE(resource.isResourceReleased());
            EXPECT_EQ(resource.kvCache().blocks(0, "default"), blocks);
            for (auto block : blocks) {
                EXPECT_TRUE(pool->isAllocated(block));
                EXPECT_GT(pool->refCount(block), 0u);
            }
            submission.reset(); // The actual end-submission path now performs the deferred release.
            EXPECT_TRUE(resource.isResourceReleased());
            for (auto block : blocks) EXPECT_FALSE(pool->isAllocated(block));
            stream->releaseResource();
        }
    }, [&] { TearDown(); cache_manager_.reset(); });
}

TEST_F(GenerateStreamStateTest, UnknownWriterReleasesOrdinaryResourcesButKeepsNativeFrozenPagesAndTensorOwner) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    test::withDecodePublicationStartup([&] {
        auto config = init_config();
        config.finalizeBlockNums(32, RuntimeConfig{}); // Preserve the native frozen-ratio guard.
        for (bool cancelled : {false, true}) for (bool pending_worker : {false, true})
        for (bool externally_frozen : {false, true}) for (bool pending_submission : {false, true}) {
            SCOPED_TRACE(cancelled);
            SCOPED_TRACE(pending_worker);
            SCOPED_TRACE(externally_frozen);
            SCOPED_TRACE(pending_submission);
            auto stream = createStream({1, 2, 3, 4}, false, RoleType::DECODE, &config);
            stream->generate_status_->status = StreamState::RUNNING;
            auto& resource = stream->streamCacheResource();
            ASSERT_TRUE(resource.initKVBlock().ok());
            ASSERT_EQ(cache_manager_->coordinator_manager_->groupBlockPools().size(), 1u);
            auto pool = cache_manager_->coordinator_manager_->groupBlockPools().front();
            const auto blocks = resource.kvCache().blocks(0, "default");
            ASSERT_FALSE(blocks.empty());
            if (externally_frozen) for (auto block : blocks) pool->freezeBlock(block);
            // No executor or GPU writer is submitted in this test. Only this
            // test-owned proof permits clearing its simulated UNKNOWN at scope
            // exit; no production reset/quiescence API is added.
            auto release_frozen = pool->prepareFrozenReferenceRelease(blocks);
            auto mock_quiescence = std::shared_ptr<void>(nullptr, [stream, pool, blocks, release_frozen](void*) {
                bool all_frozen = true;
                for (auto block : blocks) all_frozen = all_frozen && pool->isFrozen(block);
                if (all_frozen) release_frozen(); // Preserve assertion diagnostics if freezing regresses.
                {
                    std::lock_guard<std::mutex> lock(*stream->mutex_);
                    stream->unknown_decode_publication_.reset();
                }
                stream->releaseResource();
            });
            std::shared_ptr<void> worker;
            std::shared_ptr<void> submission;
            if (pending_submission) {
                ASSERT_TRUE(stream->beginDecodePublicationSubmission());
                submission = std::shared_ptr<void>(nullptr, [stream](void*) {
                    stream->endDecodePublicationSubmission();
                });
            }
            if (pending_worker) {
                stream->incPendingAsyncBookkeeping();
                worker = std::shared_ptr<void>(nullptr, [stream](void*) {
                    stream->decPendingAsyncBookkeepingAndMaybeRelease();
                });
            }
            auto external_recovery = std::shared_ptr<void>(nullptr, [pool, blocks, externally_frozen](void*) {
                if (externally_frozen)
                    for (auto block : blocks) pool->decTreeRef(block, BlockTreeRefType::FROZEN);
            }); // Drops before mock_quiescence on assertion early return.
            if (cancelled) stream->reportError(ErrorCode::CANCELLED, "original cancellation");
            stream->retainUnknownDecodePublication(stream);
            const auto expected = cancelled ? ErrorCode::CANCELLED : ErrorCode::EXECUTION_EXCEPTION;
            const auto first_error = stream->firstError();
            EXPECT_EQ(first_error.error.code(), expected);
            EXPECT_GT(first_error.order, 0u);
            EXPECT_TRUE(stream->hasError());
            EXPECT_TRUE(stream->hasEvent(StreamEvents::Error));
            auto output = stream->nextOutput(/*wait_timeout_ms=*/1);
            EXPECT_FALSE(output.ok());
            EXPECT_EQ(output.status().code(), expected);
            if (cancelled) {
                EXPECT_EQ(output.status().ToString(), "original cancellation");
            }
            worker.reset(); // Native FROZEN must protect pages after ordinary host users retire.
            ASSERT_FALSE(stream->hasPendingAsyncBookkeeping());
            EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
            EXPECT_EQ(resource.isResourceReleased(), !pending_submission);
            EXPECT_EQ(stream->isDeferredReleasePending(), pending_submission);
            stream->releaseResource();
            EXPECT_EQ(resource.isResourceReleased(), !pending_submission);
            submission.reset();
            EXPECT_TRUE(resource.isResourceReleased());
            EXPECT_FALSE(stream->beginDecodePublicationSubmission());
            EXPECT_EQ(stream->unknown_decode_publication_.get(), stream.get());
            EXPECT_TRUE(resource.kvCache().blocks(0, "default").empty());
            EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REQUEST), 0u);
            EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 0u);
            for (auto block : blocks) {
                EXPECT_TRUE(pool->isAllocated(block));
                EXPECT_EQ(pool->refCount(block), 1u); // Aggregate native tree hold, not an ordinary pin.
            }
            stream->retainUnknownDecodePublication(stream); // Duplicate notification preserves the first error/hold.
            EXPECT_EQ(stream->statusInfo().code(), expected);
            EXPECT_EQ(stream->firstError().order, first_error.order);
            EXPECT_EQ(stream->firstError().error.code(), expected);
            external_recovery.reset(); // Another operation's recovery must not remove this writer's protection.
            for (auto block : blocks) {
                EXPECT_TRUE(pool->isFrozen(block));
                EXPECT_EQ(pool->treeRefCount(block), 1u);
            }
            mock_quiescence.reset();
            EXPECT_TRUE(resource.isResourceReleased());
            for (auto block : blocks) EXPECT_FALSE(pool->isAllocated(block));
            std::weak_ptr<GenerateStream> weak = stream;
            stream.reset();
            EXPECT_TRUE(weak.expired()); // Test cleanup broke only its simulated hold.
        }
        // Idle DP streams own placeholder zero IDs, never allocated pages.
        auto fake = createStream({1, 2}, false, RoleType::DECODE);
        fake->streamCacheResource().fakeInitKVBlock(2);
        auto mock_quiescence = std::shared_ptr<void>(nullptr, [fake](void*) {
            fake->unknown_decode_publication_.reset(); // No real writer exists in this test.
            fake->releaseResource();
        });
        fake->retainUnknownDecodePublication(fake);
        EXPECT_TRUE(fake->hasError());
        for (const auto& pool : cache_manager_->coordinator_manager_->groupBlockPools())
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    }, [&] { TearDown(); cache_manager_.reset(); });
}

TEST_F(GenerateStreamStateTest, UnknownWriterFreezesRetainedEndpointPagesOutsideTheCurrentTableOnce) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    test::withDecodePublicationStartup([&] {
        auto config = init_config();
        config.finalizeBlockNums(32, RuntimeConfig{});
        auto stream = createStream({1, 2}, false, RoleType::DECODE, &config);
        stream->generate_status_->status = StreamState::RUNNING;
        auto& resource = stream->streamCacheResource();
        ASSERT_TRUE(resource.initKVBlock().ok());
        auto pool = cache_manager_->coordinator_manager_->groupBlockPools().front();
        const auto current = resource.kvCache().blocks(0, "default");
        ASSERT_FALSE(current.empty());
        const auto extra = pool->malloc();
        ASSERT_TRUE(extra);
        pool->incRef(*extra);
        const auto pending_extra = pool->malloc();
        ASSERT_TRUE(pending_extra);
        pool->incRef(*pending_extra);
        auto extra_requests = std::shared_ptr<void>(nullptr, [pool, confirmed = *extra, pending = *pending_extra](void*) {
            pool->decRef(confirmed);
            pool->decRef(pending);
        });
        auto endpoint = std::make_shared<DecodeEndpoint>();
        // A retained endpoint can own a physical page no longer in the current
        // table (STATE/SWA), and share other pages with the current candidate.
        auto protected_blocks = current;
        protected_blocks.push_back(*extra);
        for (auto block : protected_blocks) {
            auto page = std::make_shared<DecodeEndpoint::PageHold>();
            page->pool = pool;
            page->block = block;
            page->pin = pool->pinForStorageIo(block);
            endpoint->pages["default"].push_back(std::move(page));
        }
        stream->kvCacheMutable().decode_endpoint = endpoint;
        auto pending = std::make_shared<DecodeEndpoint>();
        pending->pages["default"].push_back(endpoint->pages.at("default").front());
        auto pending_page = std::make_shared<DecodeEndpoint::PageHold>();
        pending_page->pool = pool;
        pending_page->block = *pending_extra;
        pending_page->pin = pool->pinForStorageIo(*pending_extra);
        pending->pages["default"].push_back(std::move(pending_page));
        stream->kvCacheMutable().decode_provisional_endpoints.push_back(std::move(pending));
        protected_blocks.push_back(*pending_extra);
        endpoint.reset();
        extra_requests.reset(); // The out-of-table allocations now have only their endpoint pins.
        std::vector<std::function<void()>> releases;
        for (auto block : protected_blocks)
            releases.push_back(pool->prepareFrozenReferenceRelease({block}));
        auto mock_quiescence = std::shared_ptr<void>(nullptr, [stream, pool, protected_blocks, releases](void*) {
            // No real writer was submitted. Recover only this test's native holds.
            for (size_t i = 0; i < protected_blocks.size(); ++i)
                if (pool->isAllocated(protected_blocks[i]) && pool->isFrozen(protected_blocks[i])) releases[i]();
            stream->unknown_decode_publication_.reset();
            stream->releaseResource();
        });
        ASSERT_TRUE(stream->beginDecodePublicationSubmission());
        auto submission = std::shared_ptr<void>(nullptr, [stream](void*) { stream->endDecodePublicationSubmission(); });
        stream->retainUnknownDecodePublication(stream);
        stream->releaseResource();
        EXPECT_FALSE(resource.isResourceReleased());
        submission.reset();
        EXPECT_TRUE(resource.isResourceReleased());
        EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REQUEST), 0u);
        EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 0u);
        for (auto block : protected_blocks) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_TRUE(pool->isFrozen(block));
            EXPECT_EQ(pool->treeRefCount(block), 1u); // Current + repeated endpoint references deduplicate.
            EXPECT_EQ(pool->refCount(block), 1u);
        }
        stream->retainUnknownDecodePublication(stream);
        for (auto block : protected_blocks) EXPECT_EQ(pool->treeRefCount(block), 1u);
        EXPECT_FALSE(stream->beginDecodePublicationSubmission());
        EXPECT_EQ(stream->unknown_decode_publication_.get(), stream.get());
        mock_quiescence.reset();
        for (auto block : protected_blocks) EXPECT_FALSE(pool->isAllocated(block));
    }, [&] { TearDown(); cache_manager_.reset(); });
}


}  // namespace rtp_llm
