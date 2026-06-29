// Regression tests for the onErrorReported() wakeup hook introduced to
// eliminate the up-to-1s nextOutput() stall after a stream is flipped to Error.
//
// Three wakeup paths are exercised:
//   1) GenerateStream::reportError()                 -- external caller, takes mutex_
//   2) GenerateStream::reportErrorWithoutLock()      -- internal path, mutex_ already held
//   3) GenerateStateMachine::reportErrorAndWakeup()  -- state-machine internal Error path
//                                                       (handleWaiting / handleRunning MALLOC_FAILED)
//
// Each test measures wall-clock latency from the moment Error is reported to the
// moment nextOutput() returns. Assertion threshold is set well below
// SynchronizedQueue::DEF_WAIT_TIME (1s) so any future regression that bypasses
// the hook is caught deterministically.

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <thread>

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStateMachine.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class GenerateStreamErrorWakeupTest: public DeviceTestBase {
protected:
    GenerateStreamErrorWakeupTest(): perf_scope_("PERF_TEST", "1") {}

    CacheConfig initCacheConfig() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    std::shared_ptr<NormalGenerateStream> createStream() {
        cache_manager_ =
            std::make_shared<KVCacheManager>(initCacheConfig(), /*warmup=*/false, /*metrics_reporter=*/nullptr);
        EXPECT_TRUE(cache_manager_->init());
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache   = false;

        auto generate_input             = std::make_shared<GenerateInput>();
        auto generate_config            = std::make_shared<GenerateConfig>();
        generate_input->generate_config = generate_config;
        std::vector<int32_t> tokens     = {1, 2, 3, 4, 5, 6};
        generate_input->input_ids       = torch::tensor(tokens, torch::kInt32);

        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
    }

    // Spawn a thread that blocks on nextOutput(); return latency (ms) from the
    // moment the trigger callback runs to the moment nextOutput() returns. The
    // 50ms park-delay lets the consumer reach waitNotEmpty() before we fire.
    template <typename Trigger>
    int64_t measureWakeupLatencyMs(NormalGenerateStream& stream, Trigger&& trigger) {
        std::atomic<bool>                              returned{false};
        std::chrono::steady_clock::time_point          done_at;
        std::thread waiter([&]() {
            auto result = stream.nextOutput();
            done_at     = std::chrono::steady_clock::now();
            returned.store(true, std::memory_order_release);
            EXPECT_FALSE(result.ok());
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_FALSE(returned.load(std::memory_order_acquire))
            << "nextOutput() returned before Error was reported -- park-delay too short or queue had data";

        auto fire_at = std::chrono::steady_clock::now();
        trigger();
        waiter.join();

        return std::chrono::duration_cast<std::chrono::milliseconds>(done_at - fire_at).count();
    }

    autil::EnvGuard                 perf_scope_;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

// 1) Public reportError() path: external thread reports, waiter wakes quickly.
TEST_F(GenerateStreamErrorWakeupTest, reportErrorWakesNextOutput) {
    auto stream = createStream();
    auto latency_ms = measureWakeupLatencyMs(*stream, [&]() {
        stream->reportError(ErrorCode::MALLOC_FAILED, "test wakeup via reportError");
    });
    // SynchronizedQueue cv timeout is ~1s; allow generous 300ms ceiling so the
    // test is robust on slow CI without losing the regression signal.
    EXPECT_LT(latency_ms, 300) << "nextOutput() did not wake within 300ms after reportError(); "
                                  "suspect onErrorReported() hook regression";
    EXPECT_TRUE(stream->hasError());
}

// 2) Internal reportErrorWithoutLock() path: caller already holds *mutex_,
//    e.g. update()/specUpdate() chains. Lock order *mutex_ -> queue._cond
//    must hold; the wakeup hook must still fire.
TEST_F(GenerateStreamErrorWakeupTest, reportErrorWithoutLockWakesNextOutput) {
    auto stream = createStream();
    auto latency_ms = measureWakeupLatencyMs(*stream, [&]() {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->reportErrorWithoutLock(ErrorCode::OUTPUT_QUEUE_FULL, "test wakeup via reportErrorWithoutLock");
    });
    EXPECT_LT(latency_ms, 300);
    EXPECT_TRUE(stream->hasError());
}

// 3) GenerateStateMachine internal Error path: handleWaiting/handleRunning
//    MALLOC_FAILED previously used reportEvent(StreamEvents::Error, ...) which
//    bypassed the hook. The reportErrorAndWakeup() helper now routes through
//    the owner stream so nextOutput() wakes immediately.
TEST_F(GenerateStreamErrorWakeupTest, stateMachineReportErrorAndWakeupWakesNextOutput) {
    auto stream = createStream();
    auto latency_ms = measureWakeupLatencyMs(*stream, [&]() {
        // State machine reports under the stream mutex (matches the moveToNext()
        // call site that drives handleRunning).
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->generate_status_->reportErrorAndWakeup(ErrorCode::MALLOC_FAILED,
                                                       "test wakeup via state machine");
    });
    EXPECT_LT(latency_ms, 300);
    EXPECT_TRUE(stream->hasError());
    // Verify error_info / events_ were also registered (parity with the old
    // reportEvent(Error,...) write).
    EXPECT_FALSE(stream->generate_status_->error_info.ok());
    EXPECT_TRUE(stream->generate_status_->hasEvent(StreamEvents::Error));
}

// 4) Misuse guards: reportEvent / reportEventWithoutLock with StreamEvents::Error
//    must be rejected by the RTP_LLM_CHECK_WITH_INFO so the wakeup hook can
//    never be silently bypassed. myAssert() throws (default) -- catch via
//    EXPECT_ANY_THROW. If user_ft_core_dump_on_exception is on the binary
//    would abort instead, which is also an acceptable failure signal.
TEST_F(GenerateStreamErrorWakeupTest, reportEventErrorIsForbidden) {
    auto stream = createStream();
    EXPECT_ANY_THROW({
        stream->reportEvent(StreamEvents::Error, ErrorCode::MALLOC_FAILED, "should be rejected");
    });
}

TEST_F(GenerateStreamErrorWakeupTest, reportEventWithoutLockErrorIsForbidden) {
    auto stream = createStream();
    EXPECT_ANY_THROW({
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->reportEventWithoutLock(StreamEvents::Error, ErrorCode::MALLOC_FAILED, "should be rejected");
    });
}

}  // namespace rtp_llm
