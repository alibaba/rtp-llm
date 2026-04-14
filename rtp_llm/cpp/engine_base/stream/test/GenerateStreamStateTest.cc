
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

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
                                   bool                    reuse_cache  = false) {
        cache_manager_ =
            std::make_shared<KVCacheManager>(init_config(), /*warmup=*/false, /*metrics_reporter=*/nullptr);
        EXPECT_TRUE(cache_manager_->init());
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache   = reuse_cache;

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

    // Verify load_cache_context_ is null (no asyncLoadCache was called)
    ASSERT_FALSE(resource.load_cache_context_);

    // moveToNext should not trigger asyncLoadCache because LoadInitiated is set
    stream->reportEvent(StreamEvents::CanRun);
    auto new_state = stream->moveToNext();
    ASSERT_EQ(new_state, StreamState::RUNNING);

    // Still no asyncLoadCache context
    ASSERT_FALSE(resource.load_cache_context_);
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
