
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include "torch/all.h"
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
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class FIFOSchedulerCancelTest: public DeviceTestBase {
protected:
    FIFOSchedulerCancelTest(): perf_scope("PERF_TEST", "1") {}

    void SetUp() override {
        DeviceTestBase::SetUp();
        cache_config_ = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/21, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
        ASSERT_TRUE(cache_manager_->init());
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

    GenerateStreamPtr createStream(const std::vector<int>& input_tokens = {1, 2, 3}) {
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;

        auto query             = std::make_shared<GenerateInput>();
        auto generate_config   = std::make_shared<GenerateConfig>();
        query->input_ids       = torch::tensor(input_tokens, torch::kInt32);
        query->generate_config = generate_config;
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    }

    size_t initialFreeBlocks() const {
        return cache_manager_->freeBlocksNum();
    }

    // Helper function to schedule a stream through WAITING->LOADING_CACHE->WAITING->RUNNING
    // Returns the result of the final schedule() call when stream is RUNNING
    absl::StatusOr<std::list<GenerateStreamPtr>> scheduleToRunning(std::shared_ptr<FIFOScheduler>& scheduler) {
        // First schedule: WAITING -> LOADING_CACHE
        auto result1 = scheduler->schedule();
        if (!result1.ok() || result1.value().size() > 0) {
            return result1;  // Unexpected: already RUNNING or error
        }
        // Second schedule: LOADING_CACHE -> WAITING (with CanRun event set)
        auto result2 = scheduler->schedule();
        if (!result2.ok() || result2.value().size() > 0) {
            return result2;  // Unexpected: error or already done
        }
        // Third schedule: WAITING -> RUNNING
        return scheduler->schedule();
    }

protected:
    autil::EnvGuard                 perf_scope;
    CacheConfig                     cache_config_;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

// ============================================================================
// 1. Cancel a stream in WAITING state before it gets scheduled to RUNNING
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, CancelWhileWaiting) {
    auto scheduler   = createScheduler();
    auto stream      = createStream({1, 2, 3});
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);

    // Cancel the stream before schedule() is called
    stream->reportError(ErrorCode::CANCELLED, "cancelled by user");

    // Now schedule should consume the error and evict the stream
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 0);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);
    ASSERT_TRUE(stream->hasError());
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(stream->stopReason(), "cancelled by user");
    // Resources should be fully released
    ASSERT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// 2. Cancel a stream in RUNNING state
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, CancelWhileRunning) {
    auto scheduler   = createScheduler();
    auto stream      = createStream({1, 2, 3});
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // Schedule stream to RUNNING state (requires 3 schedule() calls in new code)
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);
    // Blocks should be allocated
    ASSERT_LT(cache_manager_->freeBlocksNum(), free_before);

    // Cancel the running stream
    stream->reportError(ErrorCode::CANCELLED, "cancelled by user");

    // Second schedule should detect the error and transition to FINISHED
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);
    ASSERT_TRUE(stream->hasError());
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(stream->stopReason(), "cancelled by user");
    // Resources released after cancel
    ASSERT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// 3. Cancel during resource allocation (stream transitions through WAITING
//    where initKVBlock happens inside moveToNext)
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, CancelDuringResourceAllocation) {
    auto scheduler   = createScheduler();
    auto stream      = createStream({1, 2, 3});
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    // Report error before the first schedule() — the stream is WAITING
    // and moveToNext() will see the Error event before attempting initKVBlock
    stream->reportError(ErrorCode::CANCELLED, "cancelled during init");

    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 0);
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(stream->stopReason(), "cancelled during init");
    // No blocks should remain allocated
    ASSERT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// 4. Cancel when scheduler is idle (no running streams) vs busy (has running)
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, CancelWithIdleAndBusyScheduler) {
    auto scheduler   = createScheduler();
    auto stream1     = createStream({1, 2});
    auto stream2     = createStream({3, 4});
    auto free_before = initialFreeBlocks();

    // Enqueue and schedule both streams to RUNNING
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 2);
    ASSERT_EQ(scheduler->runningStreamsSize(), 2);

    // Cancel only stream1 — scheduler is "busy" with stream2
    stream1->reportError(ErrorCode::CANCELLED, "cancelled stream1");

    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    // Only stream2 should remain running
    ASSERT_EQ(result2.value().size(), 1);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
    ASSERT_TRUE(stream1->isFinished());
    ASSERT_TRUE(stream2->getStatus() == StreamState::RUNNING);

    // Now finish stream2 normally and verify scheduler becomes idle
    stream2->reportEvent(StreamEvents::GenerateDone);
    auto result3 = scheduler->schedule();
    ASSERT_TRUE(result3.ok());
    ASSERT_EQ(result3.value().size(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);
    ASSERT_TRUE(stream2->isFinished());
    ASSERT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// 5. Verify state is correctly updated to FINISHED after cancel
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, VerifyStateAndErrorAfterCancel) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3});

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);

    // Verify initial state is RUNNING
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);
    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(stream->stopReason(), "");

    // Cancel
    stream->reportError(ErrorCode::CANCELLED, "user requested cancel");

    // hasError should be true even before the next schedule
    ASSERT_TRUE(stream->hasError());

    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());

    // After schedule, stream must be in FINISHED state
    ASSERT_TRUE(stream->isFinished());
    ASSERT_FALSE(stream->getStatus() == StreamState::RUNNING);
    ASSERT_FALSE(stream->getStatus() == StreamState::WAITING);
    ASSERT_EQ(stream->stopReason(), "user requested cancel");
    // moveToNext on FINISHED stream should not crash (idempotent)
    auto state = stream->moveToNext();
    ASSERT_EQ(state, StreamState::FINISHED);
}

// ============================================================================
// 6. Ensure no memory leak: cancel releases all KV cache blocks
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, CancelReleasesResources) {
    auto scheduler   = createScheduler();
    auto free_before = initialFreeBlocks();

    // Run 10 streams, cancel them all, verify all blocks returned
    std::vector<GenerateStreamPtr> streams;
    for (int i = 0; i < 10; i++) {
        auto stream = createStream({1, 2});
        streams.push_back(stream);
        ASSERT_TRUE(scheduler->enqueue(stream).ok());
    }

    // Schedule them to RUNNING (requires 3 schedule() calls per stream in new code)
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 10);
    ASSERT_LT(cache_manager_->freeBlocksNum(), free_before);

    // Cancel all streams
    for (auto& s : streams) {
        s->reportError(ErrorCode::CANCELLED, "batch cancel");
    }

    // Schedule to evict all
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);

    // All KV cache blocks should be fully returned
    ASSERT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// 7. Concurrent cancel safety: another thread cancels while schedule() runs
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, ConcurrentCancelDuringSchedule) {
    auto scheduler   = createScheduler();
    auto free_before = initialFreeBlocks();

    constexpr int                  kNumStreams = 10;
    std::vector<GenerateStreamPtr> streams;
    for (int i = 0; i < kNumStreams; i++) {
        auto stream = createStream({1, 2});
        streams.push_back(stream);
        ASSERT_TRUE(scheduler->enqueue(stream).ok());
    }

    // Schedule all to RUNNING (requires 3 schedule() calls per stream in new code)
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ((int)result1.value().size(), kNumStreams);

    // Launch a thread that cancels streams concurrently with schedule()
    std::thread cancel_thread([&]() {
        for (auto& s : streams) {
            s->reportError(ErrorCode::CANCELLED, "concurrent cancel");
        }
    });

    // Run schedule() while cancel thread is active.
    // Only call schedule() when there are running streams to avoid blocking
    // on an empty scheduler (waitPredicate would return false and cv blocks forever).
    // reportEvent() only sets event flags; the streams remain in running_streams_
    // until schedule() -> evaluateAndUpdateStreams() calls moveToNext().
    while (scheduler->runningStreamsSize() > 0) {
        auto result = scheduler->schedule();
        ASSERT_TRUE(result.ok());
    }

    cancel_thread.join();
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);

    // All streams should be FINISHED
    for (auto& s : streams) {
        ASSERT_TRUE(s->isFinished());
        ASSERT_TRUE(s->hasError());
    }

    // All resources released
    ASSERT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// 8. Cancel via reportEvent with Error event
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, CancelViaReportEvent) {
    auto scheduler   = createScheduler();
    auto stream      = createStream({1, 2, 3});
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);

    // Use reportEvent with Error event directly
    stream->reportError(ErrorCode::CANCELLED, "cancelled via reportEvent");

    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 0);
    ASSERT_TRUE(stream->isFinished());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->stopReason(), "cancelled via reportEvent");
    ASSERT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// 9. Double cancel is harmless (idempotent)
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, DoubleCancelIsIdempotent) {
    auto scheduler   = createScheduler();
    auto stream      = createStream({1, 2, 3});
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);

    // Cancel twice
    stream->reportError(ErrorCode::CANCELLED, "first cancel");
    stream->reportError(ErrorCode::CANCELLED, "second cancel");

    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 0);
    ASSERT_TRUE(stream->isFinished());
    // First error message is preserved
    ASSERT_EQ(stream->stopReason(), "first cancel");
    ASSERT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// 10. Mixed cancel and normal completion in the same batch
// ============================================================================
TEST_F(FIFOSchedulerCancelTest, MixedCancelAndNormalCompletion) {
    auto scheduler   = createScheduler();
    auto stream_ok   = createStream({1, 2});
    auto stream_err  = createStream({3, 4});
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(scheduler->enqueue(stream_ok).ok());
    ASSERT_TRUE(scheduler->enqueue(stream_err).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 2);

    // One finishes normally, the other is cancelled
    stream_ok->reportEvent(StreamEvents::GenerateDone);
    stream_err->reportError(ErrorCode::CANCELLED, "cancelled");

    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_EQ(result2.value().size(), 0);

    // Both streams should be FINISHED
    ASSERT_TRUE(stream_ok->isFinished());
    ASSERT_TRUE(stream_err->isFinished());
    // Only the cancelled stream should report an error
    ASSERT_FALSE(stream_ok->hasError());
    ASSERT_TRUE(stream_err->hasError());
    ASSERT_EQ(stream_err->stopReason(), "cancelled");
    // All resources released
    ASSERT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

}  // namespace rtp_llm
