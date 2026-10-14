
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <chrono>
#include <future>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"

#define private public
#define protected public
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
                                   const CacheConfig*      cache_config = nullptr) {
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config ? *cache_config : init_config(),
                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr);
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

TEST_F(GenerateStreamStateTest, testMtpSnapshotSkipsBookkeepingJoinWithinReservedCapacity) {
    auto stream = createStream({1, 2, 3});
    stream->setReserveStep(0);
    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    stream->setIsContextStream(false);

    GenerateStream::MtpAsyncDeviceState state;
    state.next_seq_len_upper_bound          = 4;
    const auto cuda_i32                     = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    state.next_kv_cache_block_id_gpu        = torch::ones({1, 1, 2}, cuda_i32);
    state.next_kv_cache_kernel_block_id_gpu = torch::ones({1, 1, 2}, cuda_i32);
    stream->setMtpAsyncDeviceState(std::move(state));
    stream->incPendingAsyncBookkeeping();
    auto pending_guard =
        std::shared_ptr<void>(nullptr, [stream](void*) { stream->decPendingAsyncBookkeepingAndMaybeRelease(); });

    ASSERT_EQ(stream->streamCacheResource().singleBatchNeedBlocks(4, 0), 0);
    ASSERT_FALSE(stream->needsAsyncBookkeepingJoinForKvReservation());
    EXPECT_EQ(stream->moveToNext(), StreamState::RUNNING);
    EXPECT_TRUE(stream->hasPendingAsyncBookkeeping());
    EXPECT_TRUE(stream->hasMtpCacheSnapshot());
    EXPECT_EQ(stream->seqLength(), 3);

    // A terminal transition still defers physical release until the pending
    // worker has stopped touching the stream, even on this scheduling path.
    stream->reportError(ErrorCode::CANCELLED, "cancel overlapped decode");
    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(stream->isDeferredReleasePending());
    EXPECT_FALSE(stream->streamCacheResource().isResourceReleased());
    pending_guard.reset();
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
}

TEST_F(GenerateStreamStateTest, testMtpSnapshotJoinsBookkeepingAtCapacityBoundary) {
    auto stream = createStream({1, 2, 3});
    stream->setReserveStep(0);
    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    stream->setIsContextStream(false);

    GenerateStream::MtpAsyncDeviceState state;
    state.next_seq_len_upper_bound          = 5;
    const auto cuda_i32                     = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    state.next_kv_cache_block_id_gpu        = torch::ones({1, 1, 2}, cuda_i32);
    state.next_kv_cache_kernel_block_id_gpu = torch::ones({1, 1, 2}, cuda_i32);
    stream->setMtpAsyncDeviceState(std::move(state));
    stream->incPendingAsyncBookkeeping();
    auto pending_guard =
        std::shared_ptr<void>(nullptr, [stream](void*) { stream->decPendingAsyncBookkeepingAndMaybeRelease(); });
    ASSERT_GT(stream->streamCacheResource().singleBatchNeedBlocks(5, 0), 0);
    ASSERT_TRUE(stream->needsAsyncBookkeepingJoinForKvReservation());

    auto next = std::async(std::launch::async, [stream] { return stream->moveToNext(); });
    EXPECT_EQ(next.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    // The join must happen before taking the stream mutex, so a worker can
    // commit its accepted length and release its claim without deadlocking.
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->setSeqLength(5);
    }
    pending_guard.reset();
    ASSERT_EQ(next.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(next.get(), StreamState::RUNNING);
    EXPECT_FALSE(stream->hasMtpCacheSnapshot());
    EXPECT_EQ(stream->streamCacheResource().singleBatchNeedBlocks(5, 0), 0);
}

TEST_F(GenerateStreamStateTest, testMtpUpperBoundReservationKeepsEpochAndDeviceMetadata) {
    auto stream = createStream({1, 2, 3});
    stream->setReserveStep(0);
    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    stream->setIsContextStream(false);

    const auto                          cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    GenerateStream::MtpAsyncDeviceState state;
    state.allow_async_cache_reservation     = true;
    state.max_mtp_advance                   = 2;
    state.previous_seq_len_upper_bound      = 3;
    state.next_seq_len_upper_bound          = 5;
    state.next_seq_len_gpu                  = torch::full({1}, 4, cuda_i32);  // Partial acceptance, below the bound.
    state.next_position_ids_gpu             = torch::tensor({3, 3, 3}, cuda_i32);
    state.engram_token_window_gpu           = torch::tensor({3, 2, 1, 0}, cuda_i32).reshape({1, 4});
    state.next_kv_cache_block_id_gpu        = torch::ones({1, 1, 2}, cuda_i32);
    state.next_kv_cache_kernel_block_id_gpu = torch::ones({1, 1, 2}, cuda_i32);
    const auto epoch                        = stream->setMtpAsyncDeviceState(state);
    stream->incPendingAsyncBookkeeping();
    auto pending_guard =
        std::shared_ptr<void>(nullptr, [stream](void*) { stream->decPendingAsyncBookkeepingAndMaybeRelease(); });
    ASSERT_GT(stream->streamCacheResource().singleBatchNeedBlocks(5, 0), 0);
    ASSERT_FALSE(stream->needsAsyncBookkeepingJoinForKvReservation());
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    EXPECT_TRUE(stream->hasPendingAsyncBookkeeping());
    const auto grown = stream->getMtpAsyncDeviceState();
    EXPECT_EQ(grown.epoch, epoch);
    EXPECT_EQ(grown.next_seq_len_upper_bound, 5);
    EXPECT_EQ(grown.next_seq_len_gpu.data_ptr(), state.next_seq_len_gpu.data_ptr());
    EXPECT_EQ(grown.next_position_ids_gpu.data_ptr(), state.next_position_ids_gpu.data_ptr());
    EXPECT_EQ(grown.engram_token_window_gpu.data_ptr(), state.engram_token_window_gpu.data_ptr());
    EXPECT_EQ(grown.next_kv_cache_block_id_gpu.size(2), 3);
    EXPECT_EQ(grown.next_kv_cache_kernel_block_id_gpu.size(2), 3);
    EXPECT_EQ(grown.cache_snapshot_host_buffers.size(), 2);
    EXPECT_TRUE(grown.cache_snapshot_host_buffers.front().is_pinned());
    EXPECT_TRUE(torch::equal(grown.next_kv_cache_block_id_gpu.cpu(), grown.cache_snapshot_host_buffers[0]));
    EXPECT_TRUE(torch::equal(grown.next_kv_cache_kernel_block_id_gpu.cpu(), grown.cache_snapshot_host_buffers[1]));
    EXPECT_EQ(state.next_kv_cache_block_id_gpu.size(2), 2);
    EXPECT_TRUE(torch::equal(state.next_kv_cache_block_id_gpu.cpu(), torch::ones({1, 1, 2}, torch::kInt32)));

    // A subsequent published acceptance invalidates replacement by a stale
    // epoch. Its shorter actual prefix must survive the next upper-bound reset.
    auto next_state                         = grown;
    next_state.previous_seq_len_upper_bound = 4;
    next_state.next_seq_len_upper_bound     = 6;
    const auto next_epoch                   = stream->setMtpAsyncDeviceState(next_state);
    EXPECT_FALSE(stream->replaceMtpCacheSnapshot(epoch, {}, {}, {}));
    EXPECT_EQ(stream->getMtpAsyncDeviceState().epoch, next_epoch);
    EXPECT_EQ(stream->getMtpAsyncDeviceState().next_seq_len_upper_bound, 6);
    EXPECT_TRUE(stream->hasMtpCacheSnapshot());
}

TEST_F(GenerateStreamStateTest, testMtpUpperBoundSwaGrowthPrunesOnlyBeyondRetainedTail) {
    auto config                          = test::makeSimpleMhaCacheConfig(1, 16, 32, DataType::TYPE_INT8);
    auto groups                          = config.topology().groups();
    groups[0].policy                     = defaultCacheGroupPolicy(CacheGroupType::SWA);
    groups[0].policy.sliding_window_size = 4;
    groups[0].policy.active_tail_blocks  = 2;
    config.setTopology(std::move(groups), config.topology().layers());
    auto stream = createStream(std::vector<int>(31, 1), false, RoleType::PDFUSION, &config);
    stream->setReserveStep(1);
    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    stream->setIsContextStream(false);
    const auto cuda_i32       = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto tag            = config.groupTags().front();
    bool       observed_prune = false;
    int        host_length    = 31;
    for (int round = 0; round < 48; ++round) {
        GenerateStream::MtpAsyncDeviceState state = stream->getMtpAsyncDeviceState();
        state.allow_async_cache_reservation       = true;
        state.max_mtp_advance                     = 6;
        state.previous_seq_len_upper_bound        = host_length;
        state.next_seq_len_upper_bound            = host_length + 6;
        if (!state.next_kv_cache_block_id_gpu.defined()) {
            state.next_kv_cache_block_id_gpu        = torch::ones({1, 1, 1}, cuda_i32);
            state.next_kv_cache_kernel_block_id_gpu = torch::ones({1, 1, 1}, cuda_i32);
        }
        stream->setMtpAsyncDeviceState(state);
        stream->incPendingAsyncBookkeeping();
        auto pending_guard =
            std::shared_ptr<void>(nullptr, [stream](void*) { stream->decPendingAsyncBookkeepingAndMaybeRelease(); });
        ASSERT_TRUE(stream->canReserveMtpCacheWithoutBookkeepingJoin(state));
        ASSERT_FALSE(stream->needsAsyncBookkeepingJoinForKvReservation());
        ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
        ASSERT_TRUE(stream->hasPendingAsyncBookkeeping());
        const auto& blocks = stream->streamCacheResource().kvCache().blocks(0, tag);
        int         live   = 0;
        for (size_t slot = 0; slot < blocks.size(); ++slot) {
            if (!isNullBlockIdx(blocks[slot])) {
                ++live;
            } else {
                observed_prune = true;
                EXPECT_LT(slot + 3, blocks.size());  // Two active pages plus reserve.
            }
        }
        EXPECT_LE(live, 3);
        // Alternate partial/full acceptance; the next bound starts at the
        // committed prefix, never at the previous round's conservative bound.
        host_length += round % 2 == 0 ? 1 : 6;
        stream->setSeqLength(host_length);
    }
    EXPECT_TRUE(observed_prune);

    auto unsafe_state            = stream->getMtpAsyncDeviceState();
    unsafe_state.max_mtp_advance = 64;
    EXPECT_FALSE(stream->canReserveMtpCacheWithoutBookkeepingJoin(unsafe_state));
    unsafe_state                               = stream->getMtpAsyncDeviceState();
    unsafe_state.allow_async_cache_reservation = false;
    EXPECT_FALSE(stream->canReserveMtpCacheWithoutBookkeepingJoin(unsafe_state));
}

TEST_F(GenerateStreamStateTest, testMallocRetryMetricsIndependentOfTerminalOutcome) {
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
        kmonitor::MetricsTags tags;
        auto                  reporter = std::make_shared<kmonitor::MetricsReporter>("", "", tags);
        auto                  stream   = createStream({1, 2, 3}, /*reuse_cache=*/false, RoleType::PREFILL);
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
        stream.reset();
        expectQps(reporter->getMetricsGroup<RtpLLMStreamMetrics>()->kv_cache_malloc_failed_qps_metric,
                  tags,
                  test.error == ErrorCode::MALLOC_FAILED || test.error == ErrorCode::DECODE_MALLOC_FAILED);
        expectQps(operations->malloc_retry_qps_metric, tags, 0);
    }
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

}  // namespace rtp_llm
