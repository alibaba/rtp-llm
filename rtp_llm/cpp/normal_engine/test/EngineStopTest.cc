#include <any>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <functional>
#include <stdexcept>
#include <memory>
#include <string>
#include "torch/all.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"

#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
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

class ObservedBatchScheduler: public BatchDecodeScheduler {
public:
    ObservedBatchScheduler(NormalEngine* engine, bool fail_stop):
        BatchDecodeScheduler(
            engine->runtime_config, engine->resourceContext().cache_manager, nullptr, 0, RoleType::PREFILL),
        engine_(engine),
        fail_stop_(fail_stop) {
        batch_size_ = 1;
    }

    absl::Status stop() override {
        const auto result = BatchDecodeScheduler::stop();
        if (!stop_observed_.exchange(true)) {
            stop_returning.set_value();
        }
        return fail_stop_ ? absl::InternalError("injected scheduler stop error") : result;
    }

    absl::Status finalizeStop() override {
        loop_joined       = engine_->loop_thread_ == nullptr;
        executor_drained  = engine_->executor_ == nullptr;
        const auto status = BatchDecodeScheduler::finalizeStop();
        ++finalize_calls;
        return status;
    }

    std::promise<void> stop_returning;
    std::atomic<bool>  loop_joined{false};
    std::atomic<bool>  executor_drained{false};
    std::atomic<int>   finalize_calls{0};

private:
    NormalEngine*     engine_;
    bool              fail_stop_;
    std::atomic<bool> stop_observed_{false};
};

// Keep the original FIFO fixture intact. Only these new tests replace the
// scheduler, before startLoop, and reuse the real NormalExecutor/dispatcher/KV.
std::shared_ptr<NormalEngine>
createPausedBatchEngine(const CustomConfig& config, MockModel::ForwardHook hook = {}, bool fail_stop = false) {
    ModelConfig   model;
    RuntimeConfig runtime;
    KVCacheConfig cache;
    auto          params               = createEngineInitParams(config, model, runtime, cache);
    NormalExecutor::test_model_factory = [vocab = model.vocab_size, hook](const GptModelInitParams&) {
        return std::make_unique<MockModel>(vocab, hook);
    };
    struct ResetFactory {
        ~ResetFactory() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } reset_factory;
    auto engine        = std::make_shared<NormalEngine>(params, nullptr, false);
    engine->scheduler_ = std::make_unique<ObservedBatchScheduler>(engine.get(), fail_stop);
    return engine;
}

class EngineStopTest: public DeviceTestBase, public ::testing::WithParamInterface<int> {
protected:
    void checkStopWaitsForDispatch(int max_new_tokens, bool batch_scheduler = false);
    void checkStopWaitsForForward(bool fail_stop);
};

// Separate Bazel targets set RTP_LLM_STREAM_ASYNC before process startup because
// NormalExecutor caches this flag. Both modes also exercise worker counts 0/2.
void EngineStopTest::checkStopWaitsForDispatch(int max_new_tokens, bool batch_scheduler) {
    const char* async_env = std::getenv("RTP_LLM_STREAM_ASYNC");
    ASSERT_NE(async_env, nullptr);
    const bool stream_async = std::string(async_env) == "1";

    CustomConfig config;
    config.output_dispatcher_worker_count = GetParam();
    auto  engine   = batch_scheduler ? createPausedBatchEngine(config) : createMockEngine(config);
    auto* observed = dynamic_cast<ObservedBatchScheduler*>(engine->scheduler_.get());
    if (observed) {
        ASSERT_TRUE(engine->startLoop().ok());
    }
    auto* executor = dynamic_cast<NormalExecutor*>(engine->executor_.get());
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(executor->useStreamAsync(), stream_async);
    EXPECT_FALSE(executor->useDeviceInput());
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
    if (stream_async && max_new_tokens > 0) {
        // Async dispatch also works with the explicitly selected host-input path.
        const auto state = stream->getNormalAsyncDeviceState();
        EXPECT_FALSE(state.next_seq_len_gpu.defined());
    }
    const size_t held_blocks = stream->curBlocksNum();
    if (observed) {
        EXPECT_GT(held_blocks, 0);
        EXPECT_FALSE(stream->stream_cache_resource_->isResourceReleased());
    }
    std::promise<void> stop_started;
    auto               stopping = stop_started.get_future();
    auto               stopped  = std::async(std::launch::async, [&]() {
        stop_started.set_value();
        return engine->stop();
    });
    EXPECT_EQ(stopping.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    if (observed) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!observed->stop_requested_.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_TRUE(observed->stop_requested_.load());
        EXPECT_EQ(observed->finalize_calls.load(), 0);
        // updateOutput holds the stream mutex while gated: check errors after release.
        EXPECT_EQ(stream->curBlocksNum(), held_blocks);
        EXPECT_FALSE(stream->stream_cache_resource_->isResourceReleased());
    }
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
    if (observed) {
        EXPECT_EQ(observed->finalize_calls.load(), 1);
        EXPECT_TRUE(observed->loop_joined.load());
        EXPECT_TRUE(observed->executor_drained.load());
        EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        EXPECT_FALSE(stream->hasError());
        EXPECT_TRUE(stream->stream_cache_resource_->isResourceReleased());
        EXPECT_EQ(stream->curBlocksNum(), 0);
        EXPECT_EQ(observed->onflightStreams(), 0);
    }
}

TEST_P(EngineStopTest, StopWaitsForInFlightDecodeDispatch) {
    checkStopWaitsForDispatch(2);
}

TEST_P(EngineStopTest, StopWaitsForInFlightPrefillOnlyDispatch) {
    checkStopWaitsForDispatch(0);
}

void EngineStopTest::checkStopWaitsForForward(bool fail_stop) {
    CustomConfig config;
    config.output_dispatcher_worker_count = GetParam();
    std::atomic<bool>  armed{false};
    std::promise<void> in_forward;
    auto               entered = in_forward.get_future();
    std::promise<void> release;
    auto               released = release.get_future().share();
    auto               engine   = createPausedBatchEngine(
        config,
        [&](const GptModelInputs&) {
            if (armed.exchange(false)) {
                in_forward.set_value();
                released.wait();
            }
        },
        fail_stop);
    auto* observed = dynamic_cast<ObservedBatchScheduler*>(engine->scheduler_.get());
    ASSERT_NE(observed, nullptr);
    auto stop_returning                    = observed->stop_returning.get_future();
    auto query                             = std::make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
    query->generate_config                 = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 0;
    query->generate_config->is_streaming   = false;
    auto stream                            = std::make_shared<NormalGenerateStream>(
        query, engine->model_config_, engine->runtime_config, engine->resourceContext(), nullptr);
    // Do not hold an external owning copy of the executor: reset must really
    // destroy/drain it, and the observer verifies this before finalizeStop.
    armed = true;
    ASSERT_TRUE(engine->startLoop().ok());
    GenerateStreamPtr queued = stream;
    engine->enqueue(queued);
    EXPECT_EQ(entered.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    const size_t held_blocks = stream->curBlocksNum();
    EXPECT_GT(held_blocks, 0);
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    auto stopped = std::async(std::launch::async, [&] { return engine->stop(); });
    EXPECT_EQ(stop_returning.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_EQ(observed->finalize_calls.load(), 0);
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(stream->curBlocksNum(), held_blocks);
    EXPECT_FALSE(stream->stream_cache_resource_->isResourceReleased());
    release.set_value();
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    const auto status = stopped.get();
    if (fail_stop) {
        EXPECT_EQ(status, absl::InternalError("injected scheduler stop error"));
    } else {
        EXPECT_TRUE(status.ok()) << status;
    }
    EXPECT_TRUE(observed->loop_joined.load());
    EXPECT_TRUE(observed->executor_drained.load());
    EXPECT_EQ(observed->finalize_calls.load(), 1);
    EXPECT_EQ(engine->executor_, nullptr);
    EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(stream->seqLength(), 2);
    EXPECT_TRUE(stream->stream_cache_resource_->isResourceReleased());
    EXPECT_EQ(stream->curBlocksNum(), 0);
    EXPECT_EQ(observed->onflightStreams(), 0);
}

TEST_P(EngineStopTest, BatchStopKeepsKvAliveUntilSynchronousForwardReturns) {
    checkStopWaitsForForward(false);
}

TEST_P(EngineStopTest, BatchStopErrorStillJoinsDrainsAndFinalizes) {
    checkStopWaitsForForward(true);
}

TEST_P(EngineStopTest, BatchStopFinalizesAfterInFlightDecodeDispatch) {
    checkStopWaitsForDispatch(2, true);
}

TEST_P(EngineStopTest, BatchStopFinalizesAfterInFlightPrefillOnlyDispatch) {
    checkStopWaitsForDispatch(0, true);
}

// These tests exercise the real NormalEngine::loop and step dispatch. They set
// the already-claimed per-engine protocol flag deliberately: public UDS frame
// validation/claim is covered separately by CpuTpBroadcasterTest. No fork occurs
// in this GPU fixture, and no production test hook is introduced.
class CpuTpLoopExecutor: public Executor {
public:
    explicit CpuTpLoopExecutor(std::function<absl::Status()> process): process_(std::move(process)) {}

    absl::Status process(const std::list<GenerateStreamPtr>&, int64_t) override {
        return process_();
    }

private:
    std::function<absl::Status()> process_;
};

class CpuTpEngineLoopTest: public DeviceTestBase {
protected:
    std::shared_ptr<NormalEngine> follower() {
        CustomConfig config;
        auto         engine                = createPausedBatchEngine(config);
        engine->parallelism_config.tp_rank = 1;
        engine->parallelism_config.tp_size = 2;
        engine->parallelism_config.dp_size = 1;
        engine->cpu_tp_shutdown_           = true;
        engine->running_                   = false;
        return engine;
    }

    void finish(const std::shared_ptr<NormalEngine>& engine) {
        auto* observed = static_cast<ObservedBatchScheduler*>(engine->scheduler_.get());
        // A terminal frame ends dispatch; it must not itself free executor/KV.
        EXPECT_NE(engine->executor_, nullptr);
        EXPECT_EQ(observed->finalize_calls.load(), 0);
        EXPECT_TRUE(engine->stop().ok());
        EXPECT_EQ(engine->executor_, nullptr);
        EXPECT_TRUE(observed->loop_joined.load());
        EXPECT_TRUE(observed->executor_drained.load());
        EXPECT_EQ(observed->finalize_calls.load(), 1);
    }
};

TEST_F(CpuTpEngineLoopTest, FollowerAlreadyLocallyStoppedConsumesDataBeforeTerminal) {
    auto engine       = follower();
    int  calls        = 0;
    engine->executor_ = std::make_unique<CpuTpLoopExecutor>([&]() -> absl::Status {
        if (++calls <= 2) {
            return absl::OkStatus();
        }
        throw CpuTpShutdown();
    });
    engine->loop();
    EXPECT_EQ(calls, 3);
    EXPECT_FALSE(engine->running_.load());
    finish(engine);
}

TEST_F(CpuTpEngineLoopTest, LocalStopInsideCompleteStepStillReceivesRootTerminal) {
    auto engine       = follower();
    engine->running_  = true;
    int calls         = 0;
    engine->executor_ = std::make_unique<CpuTpLoopExecutor>([&]() -> absl::Status {
        if (++calls == 1) {
            engine->running_ = false;
            return absl::OkStatus();
        }
        throw CpuTpShutdown();
    });
    engine->loop();
    EXPECT_EQ(calls, 2);
    EXPECT_FALSE(engine->running_.load());
    finish(engine);
}

TEST_F(CpuTpEngineLoopTest, PausedLocallyStoppedFollowerReachesTerminalReceive) {
    auto engine = follower();
    // Written before loop entry, avoiding concurrent access to existing pause_.
    engine->pause_    = true;
    int calls         = 0;
    engine->executor_ = std::make_unique<CpuTpLoopExecutor>([&]() -> absl::Status {
        ++calls;
        throw CpuTpShutdown();
    });
    engine->loop();
    EXPECT_EQ(calls, 1);
    EXPECT_FALSE(engine->running_.load());
    engine->pause_ = false;
    finish(engine);
}

TEST_F(CpuTpEngineLoopTest, OrdinaryTransportErrorStillEscapesLocallyStoppedFollower) {
    auto engine       = follower();
    int  calls        = 0;
    engine->executor_ = std::make_unique<CpuTpLoopExecutor>([&]() -> absl::Status {
        ++calls;
        throw std::runtime_error("injected ordinary transport EOF");
    });
    EXPECT_THROW(engine->loop(), std::runtime_error);
    EXPECT_EQ(calls, 1);
    EXPECT_FALSE(engine->running_.load());
    finish(engine);
}

TEST_F(CpuTpEngineLoopTest, TerminalOutsideCoordinatedFollowerRemainsAnError) {
    auto engine              = follower();
    engine->cpu_tp_shutdown_ = false;
    engine->running_         = true;
    int calls                = 0;
    engine->executor_        = std::make_unique<CpuTpLoopExecutor>([&]() -> absl::Status {
        ++calls;
        throw CpuTpShutdown();
    });
    bool rejected            = false;
    try {
        engine->loop();
    } catch (const CpuTpShutdown&) {
        ADD_FAILURE() << "unexpected typed terminal must be converted to a protocol error";
    } catch (const std::exception&) {
        rejected = true;
    }
    engine->running_ = false;
    EXPECT_TRUE(rejected);
    EXPECT_EQ(calls, 1);
    finish(engine);
}

INSTANTIATE_TEST_SUITE_P(SerialAndParallel, EngineStopTest, ::testing::Values(0, 2));

}  // namespace
}  // namespace rtp_llm
