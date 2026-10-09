#include <any>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include "torch/all.h"

#define private public
#define protected public
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
        // Prefill must finish first: only decode uses the outer AsyncRunner.
        if (++update_count_ == 2) {
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

class EngineStopTest: public DeviceTestBase, public ::testing::WithParamInterface<int> {};

// Separate Bazel targets set RTP_LLM_STREAM_ASYNC before process startup because
// NormalExecutor caches this flag. Both modes also exercise worker counts 0/2.
TEST_P(EngineStopTest, StopWaitsForInFlightDecodeDispatch) {
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
    query->generate_config->max_new_tokens = 2;
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
    EXPECT_EQ(stream->hasPendingAsyncBookkeeping(), stream_async);
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
    EXPECT_EQ(stream->hasPendingAsyncBookkeeping(), stream_async);
    release.set_value();

    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    EXPECT_TRUE(stopped.get().ok());
    EXPECT_TRUE(stream->decode_completed.load());
    EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
    EXPECT_EQ(engine->executor_, nullptr);
    // stop() cancels scheduled streams, so nextOutput() need not succeed.
    // The gated update must still have committed both generated tokens.
    EXPECT_EQ(stream->seqLength(), 4);
}

INSTANTIATE_TEST_SUITE_P(SerialAndParallel, EngineStopTest, ::testing::Values(0, 2));

}  // namespace
}  // namespace rtp_llm
