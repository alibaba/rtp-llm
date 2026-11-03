#include <any>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <future>
#include <memory>
#include <string>
#include "torch/all.h"

#define private public
#define protected public
#if USING_CUDA
#include "rtp_llm/cpp/normal_engine/accuracy/AccuracyChecker.h"
#endif
#include "rtp_llm/cpp/normal_engine/accuracy/AttentionAccuracyCheckResourceRelease.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#undef protected
#undef private
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

class GatedDecodeStream: public NormalGenerateStream {
public:
    using NormalGenerateStream::NormalGenerateStream;

    void updateOutput(const StreamUpdateInfo& update_info) override {
        // Generation reaches the outer AsyncRunner on decode; prefill-only finishes on its first update.
        const int gated_update = generateConfig()->isPrefillOnly() ? 1 : 2;
        if (++update_count_ == gated_update) {
            decode_started.set_value();
            release.wait();
            NormalGenerateStream::updateOutput(update_info);
            decode_completed = true;
        } else {
            NormalGenerateStream::updateOutput(update_info);
        }
    }

    std::promise<void>       decode_started;
    std::atomic<bool>        decode_completed{false};
    std::shared_future<void> release;

private:
    int update_count_ = 0;
};

class EngineStopTest: public DeviceTestBase, public ::testing::WithParamInterface<int> {
protected:
    void checkStopWaitsForDispatch(int max_new_tokens);
};

// Separate Bazel targets set RTP_LLM_STREAM_ASYNC before process startup because
// NormalExecutor caches this flag. Both modes also exercise worker counts 0/2.
void EngineStopTest::checkStopWaitsForDispatch(int max_new_tokens) {
    const char* async_env = std::getenv("RTP_LLM_STREAM_ASYNC");
    ASSERT_NE(async_env, nullptr);
    const bool stream_async = std::string(async_env) == "1";

    CustomConfig config;
    config.output_dispatcher_worker_count = GetParam();
    auto  engine                          = createMockEngine(config);
    auto* executor                        = dynamic_cast<NormalExecutor*>(engine->executor_.get());
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(executor->useStreamAsync(), stream_async);
    EXPECT_EQ(executor->batch_stream_processor_->output_dispatcher_->thread_pool_ != nullptr, GetParam() > 0);

    auto query                             = std::make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
    query->generate_config                 = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = max_new_tokens;
    query->generate_config->is_streaming   = false;

    std::promise<void> release;
    auto               stream = std::make_shared<GatedDecodeStream>(
        query, engine->model_config_, engine->runtime_config, engine->resourceContext(), nullptr);
    stream->release                 = release.get_future().share();
    auto              started       = stream->decode_started.get_future();
    GenerateStreamPtr queued_stream = stream;
    engine->enqueue(queued_stream);

    // No fatal assertions while the gate is held: always unblock the worker
    // before destroying the stop future or engine, including failure paths.
    EXPECT_EQ(started.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    EXPECT_EQ(stream->hasPendingAsyncBookkeeping(), stream_async && max_new_tokens > 0);
    EXPECT_FALSE(stream->decode_completed.load());
    std::promise<void> stop_started;
    auto               stopping = stop_started.get_future();
    auto               stopped  = std::async(std::launch::async, [&]() {
        stop_started.set_value();
        return engine->stop();
    });
    EXPECT_EQ(stopping.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_FALSE(stream->decode_completed.load());
    EXPECT_EQ(stream->hasPendingAsyncBookkeeping(), stream_async && max_new_tokens > 0);
    release.set_value();

    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    EXPECT_TRUE(stopped.get().ok());
    EXPECT_TRUE(stream->decode_completed.load());
    EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
    EXPECT_EQ(engine->executor_, nullptr);
    // stop() cancels scheduled streams, so nextOutput() need not succeed.
    // A stopped prefill-only request must not manufacture a generated token.
    EXPECT_EQ(stream->seqLength(), 2 + max_new_tokens);
}

TEST_P(EngineStopTest, StopWaitsForInFlightDecodeDispatch) {
    checkStopWaitsForDispatch(2);
}

TEST_P(EngineStopTest, StopWaitsForInFlightPrefillOnlyDispatch) {
    checkStopWaitsForDispatch(0);
}

INSTANTIATE_TEST_SUITE_P(SerialAndParallel, EngineStopTest, ::testing::Values(0, 2));

TEST(AccuracyCheckExecutorTest, DisablesProductionSideEffects) {
    CustomConfig config;
    config.output_dispatcher_worker_count = 2;
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params               = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    NormalExecutor::test_model_factory = [vocab = model_config.vocab_size](const GptModelInitParams&) {
        return std::make_unique<MockModel>(vocab);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    NormalExecutor executor(params, nullptr, false, false, 0, MlaOpsType::AUTO, nullptr, nullptr, false, true);
    EXPECT_FALSE(executor.useStreamAsync());
    EXPECT_EQ(executor.metrics_reporter_, nullptr);
    EXPECT_EQ(executor.model_inputs_logger_, nullptr);
    EXPECT_EQ(executor.batch_stream_processor_->output_dispatcher_->thread_pool_, nullptr);
}

#if USING_CUDA
TEST(AccuracyCheckCacheTest, RejectsZeroBlockSize) {
    KVCacheConfig kv_cache_config;
    EXPECT_THROW(AccuracyChecker::computeTemporaryBlockNum(0, kv_cache_config), std::exception);
}
#endif

TEST(AccuracyCheckResourceReleaseTest, ReleasesExecutorOwnedTemporaryKvCacheBeforeCudaCleanup) {
    struct TemporaryCache {};
    struct TemporaryExecutor {
        explicit TemporaryExecutor(std::shared_ptr<TemporaryCache> cache): cache(std::move(cache)) {}

        std::shared_ptr<TemporaryCache> cache;
    };

    auto cache                       = std::make_shared<TemporaryCache>();
    auto weak_cache                  = std::weak_ptr<TemporaryCache>(cache);
    auto executor                    = std::make_unique<TemporaryExecutor>(cache);
    int  synchronize_count           = 0;
    int  empty_cache_count           = 0;
    bool cache_released_before_sync  = false;
    bool cache_released_before_empty = false;

    attention_accuracy_check::releaseResources(
        executor,
        cache,
        [&]() {
            ++synchronize_count;
            cache_released_before_sync = weak_cache.expired();
        },
        [&]() {
            ++empty_cache_count;
            cache_released_before_empty = weak_cache.expired();
        });

    EXPECT_EQ(executor, nullptr);
    EXPECT_EQ(cache, nullptr);
    EXPECT_EQ(synchronize_count, 1);
    EXPECT_EQ(empty_cache_count, 1);
    EXPECT_TRUE(cache_released_before_sync);
    EXPECT_TRUE(cache_released_before_empty);
}

#if USING_CUDA
TEST(AccuracyCheckPromptTest, IsDeterministicAndLengthDependent) {
    ModelConfig       model_config;
    RuntimeConfig     runtime_config;
    ParallelismConfig parallelism_config;
    model_config.vocab_size     = 1024;
    model_config.embedding_size = 512;
    AccuracyChecker checker(model_config, runtime_config, parallelism_config);

    const auto first  = checker.makeCheckPrompt(128);
    const auto second = checker.makeCheckPrompt(128);
    const auto longer = checker.makeCheckPrompt(129);

    EXPECT_TRUE(torch::equal(first, second));
    EXPECT_FALSE(torch::equal(first.slice(0, 0, 128), longer.slice(0, 0, 128)));
    EXPECT_GE(first.min().item<int32_t>(), 0);
    EXPECT_LT(first.max().item<int32_t>(), 512);
}
#endif

}  // namespace
}  // namespace rtp_llm
