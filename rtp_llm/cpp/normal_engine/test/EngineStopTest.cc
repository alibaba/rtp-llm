#include <any>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <memory>
#include <string>
#include "torch/all.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"

#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#undef protected
#undef private
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

class PeerClosedExecutor: public Executor {
public:
    PeerClosedExecutor(std::atomic<bool>& running, bool stopping): running_(running), stopping_(stopping) {}

    absl::Status process(const ScheduleOutput&, int64_t) override {
        if (stopping_) {
            running_ = false;
        }
        throw CpuTpPeerClosed("peer closed at the next input frame");
    }

private:
    std::atomic<bool>& running_;
    bool               stopping_;
};

class EnginePeerClosedTest: public DeviceTestBase {
protected:
    void checkPeerClose(bool stopping) {
        auto engine = createMockEngine({});
        ASSERT_TRUE(engine->stop().ok());
        // Use the follower path so the real loop reaches process() with no requests.
        engine->parallelism_config.tp_rank = 1;
        engine->executor_                  = std::make_unique<PeerClosedExecutor>(engine->running_, stopping);
        engine->running_                   = true;
        if (stopping) {
            EXPECT_NO_THROW(engine->loop());
        } else {
            EXPECT_THROW(engine->loop(), CpuTpPeerClosed);
        }
        engine->running_ = false;
    }
};

// Exercise NormalEngine::loop/stop rather than only the rendezvous class. The
// mock collective represents a model round shared by both local DP replicas.
struct StopCohort {
    std::mutex              mutex;
    std::condition_variable cv;
    int                     arrivals = 0, generation = 0;
    std::atomic<int>        progress{0}, drained{0}, released_early{0};
    void                    step() {
        std::unique_lock<std::mutex> lock(mutex);
        const int                    round = generation;
        if (++arrivals == 8) {
            arrivals = 0;
            ++generation;
            cv.notify_all();
        } else if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return generation != round; })) {
            throw std::runtime_error("engine stop lost a model participant");
        }
        ++progress;
    }
};

class StopCohortExecutor: public Executor {
public:
    StopCohortExecutor(StopCohort& cohort, int rank): cohort_(cohort), rank_(rank) {}
    ~StopCohortExecutor() override {
        if (cohort_.drained.load() != 8) {
            ++cohort_.released_early;
        }
    }
    absl::Status process(const ScheduleOutput&, int64_t) override {
        cohort_.step();
        return absl::OkStatus();
    }
    void drainForShutdown() override {
        if (rank_ == 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        ++cohort_.drained;
    }

private:
    StopCohort& cohort_;
    int         rank_;
};

TEST_F(EnginePeerClosedTest, LocalDpStopDrainsAllPeersBeforeReleasingAnyExecutor) {
    char directory[] = "/tmp/rtp_engine_cohort_XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    const std::string                            path = std::string(directory) + "/state";
    StopCohort                                   cohort;
    std::array<std::shared_ptr<NormalEngine>, 8> engines;
    std::array<std::future<absl::Status>, 8>     stops;
    for (int rank = 0; rank < 8; ++rank) {
        engines[rank] = createMockEngine({});
        ASSERT_TRUE(engines[rank]->stop().ok());
    }
    for (int rank = 0; rank < 8; ++rank) {
        // No scheduler inputs are needed for the mock collective executor.
        engines[rank]->parallelism_config.tp_rank = 1;
        engines[rank]->executor_                  = std::make_unique<StopCohortExecutor>(cohort, rank);
        engines[rank]->loop_coordinator_          = std::make_unique<EngineLoopCoordinator>(path, rank, 8);
        engines[rank]->running_                   = true;
    }
    for (auto& engine : engines) {
        auto* pointer        = engine.get();
        engine->loop_thread_ = autil::Thread::createThread([pointer] { pointer->loop(); }, "stop_cohort");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (cohort.progress.load() < 80 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stops[5] = std::async(std::launch::async, [&] { return engines[5]->stop(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    for (int rank = 0; rank < 8; ++rank) {
        if (rank != 5) {
            stops[rank] = std::async(std::launch::async, [&, rank] { return engines[rank]->stop(); });
        }
    }
    for (auto& stop : stops) {
        EXPECT_TRUE(stop.get().ok());
    }
    EXPECT_EQ(cohort.drained.load(), 8);
    EXPECT_EQ(cohort.released_early.load(), 0);
    const auto rounds = engines[0]->loop_coordinator_->completedSteps();
    for (auto& engine : engines) {
        EXPECT_EQ(engine->loop_coordinator_->completedSteps(), rounds);
        EXPECT_EQ(engine->executor_, nullptr);
    }
    engines.fill(nullptr);
    EXPECT_EQ(rmdir(directory), 0);
}

TEST_F(EnginePeerClosedTest, StopAllowsCleanPeerClose) {
    checkPeerClose(true);
}

TEST_F(EnginePeerClosedTest, RunningEngineRejectsCleanPeerClose) {
    checkPeerClose(false);
}

TEST_F(EnginePeerClosedTest, ResidentPromptRoundsAreCountedButWarmupIsNot) {
    auto engine = createMockEngine({});
    ASSERT_TRUE(engine->stop().ok());
    class CountingExecutor: public Executor {
    public:
        absl::Status process(const ScheduleOutput&, int64_t) override {
            return absl::OkStatus();
        }
    };
    engine->executor_ = std::make_unique<CountingExecutor>();
    char directory[]  = "/tmp/rtp_prompt_round_XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    struct Cleanup {
        NormalEngine& engine;
        std::string   directory;
        ~Cleanup() {
            engine.loop_coordinator_.reset();
            unlink((directory + "/state").c_str());
            rmdir(directory.c_str());
        }
    } cleanup{*engine, directory};
    engine->loop_coordinator_ = std::make_unique<EngineLoopCoordinator>(std::string(directory) + "/state", 0, 2);
    auto query                = std::make_shared<GenerateInput>();
    query->input_ids          = torch::tensor({1, 2, 3}, torch::kInt32);
    query->generate_config    = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 1;
    ASSERT_TRUE(engine->preRun(query, preRunMode::prefill_warm_up).ok());
    ASSERT_TRUE(engine->preRun(query, preRunMode::decode_warm_up).ok());
    EXPECT_EQ(engine->loop_coordinator_->completedSteps(), 0u);
    ASSERT_TRUE(engine->preRun(query, preRunMode::build_system_prompt).ok());
    EXPECT_EQ(engine->loop_coordinator_->completedSteps(), 1u);
}

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

    std::shared_ptr<NormalEngine> createConfiguredEngine(const CustomConfig&                    config,
                                                         std::function<void(EngineInitParams&)> configure) {
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig cache_config;
        auto          params = createEngineInitParams(config, model_config, runtime_config, cache_config);
        configure(params);
        NormalExecutor::test_model_factory = [vocab = model_config.vocab_size](const GptModelInitParams&) {
            return std::make_unique<MockModel>(vocab);
        };
        struct FactoryResetGuard {
            ~FactoryResetGuard() {
                NormalExecutor::test_model_factory = nullptr;
            }
        } guard;
        return std::make_shared<NormalEngine>(params, nullptr);
    }
};

struct FinishState {
    std::promise<std::thread::id> started;
    std::shared_future<void>      release;
    std::atomic<int>              calls{0};
    std::atomic<bool>             completed{false};
    std::atomic<bool>             destroyed{false};
    std::atomic<bool>             destroyed_after_finish{false};
};

class GatedFinishExecutor: public Executor {
public:
    GatedFinishExecutor(std::shared_ptr<FinishState> state, const std::atomic<bool>& running):
        state_(std::move(state)), running_(running) {}

    absl::Status process(const ScheduleOutput&, int64_t) override {
        return absl::OkStatus();
    }

    absl::Status finish() override {
        EXPECT_FALSE(running_.load());
        ++state_->calls;
        state_->started.set_value(std::this_thread::get_id());
        state_->release.wait();
        state_->completed = true;
        return absl::OkStatus();
    }

    ~GatedFinishExecutor() override {
        state_->destroyed_after_finish = state_->completed.load();
        state_->destroyed              = true;
    }

private:
    std::shared_ptr<FinishState> state_;
    const std::atomic<bool>&     running_;
};

class EngineFinishTest: public DeviceTestBase {};

TEST_F(EngineFinishTest, StopJoinsLoopFinishBeforeDestroyingExecutor) {
    for (int pp_size : {1, 3}) {
        SCOPED_TRACE(pp_size);
        auto engine = createMockEngine(CustomConfig{});
        ASSERT_TRUE(engine->stop().ok());
        auto               state = std::make_shared<FinishState>();
        std::promise<void> release;
        state->release                     = release.get_future().share();
        auto started                       = state->started.get_future();
        engine->executor_                  = std::make_unique<GatedFinishExecutor>(state, engine->running_);
        engine->parallelism_config.pp_size = pp_size;
        /** Keep the scheduler stopped: this test observes loop teardown without admitting new requests. */
        ASSERT_TRUE(engine->startLoop().ok());
        std::thread::id stop_thread;
        auto            stopped = std::async(std::launch::async, [&] {
            stop_thread = std::this_thread::get_id();
            return engine->stop();
        });

        /** Always release finish before destroying the stop future, including assertion failures. */
        const auto ready = started.wait_for(std::chrono::seconds(10));
        EXPECT_EQ(ready, std::future_status::ready);
        if (ready == std::future_status::ready) {
            EXPECT_NE(started.get(), stop_thread);
        }
        EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
        EXPECT_FALSE(state->destroyed.load());
        release.set_value();
        EXPECT_EQ(stopped.wait_for(std::chrono::seconds(30)), std::future_status::ready);
        EXPECT_TRUE(stopped.get().ok());
        EXPECT_EQ(state->calls.load(), 1);
        EXPECT_TRUE(state->completed.load());
        EXPECT_TRUE(state->destroyed_after_finish.load());
        EXPECT_EQ(engine->executor_, nullptr);
        EXPECT_TRUE(engine->stop().ok());
        EXPECT_EQ(state->calls.load(), 1);
    }
}

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
    EXPECT_TRUE(engine->stop().ok());
    engine.reset();
}

TEST_P(EngineStopTest, StopWaitsForInFlightDecodeDispatch) {
    checkStopWaitsForDispatch(2);
}

TEST_P(EngineStopTest, StopWaitsForInFlightPrefillOnlyDispatch) {
    checkStopWaitsForDispatch(0);
}

TEST_P(EngineStopTest, SchedulerStopErrorStillReleasesExecutor) {
    auto engine = createConfiguredEngine({}, [](EngineInitParams& params) {
        params.runtime_config.use_batch_decode_scheduler = true;
        // Keep the empty scheduler nonblocking so this exercises cleanup, not its idle timeout.
        params.runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 0;
    });
    EXPECT_TRUE(absl::IsUnimplemented(engine->stop()));
    EXPECT_EQ(engine->executor_, nullptr);
}

TEST_P(EngineStopTest, SystemPromptsDoNotConsumeProfilingWindow) {
    CustomConfig config;
    config.multi_task_prompt_tokens = {{"one", {1, 2}}, {"two", {3, 4}}, {"three", {5, 6}}};
    config.reuse_cache              = true;
    auto engine                     = createConfiguredEngine(config, [](EngineInitParams& params) {
        params.profiling_debug_logging_config.gen_timeline_sync   = true;
        params.profiling_debug_logging_config.timeline_start_step = 1000;
    });
    ASSERT_TRUE(engine->stop().ok());
    EXPECT_TRUE(engine->step_profiler_.enabled());
    EXPECT_EQ(engine->step_profiler_.waited_steps_, 0);
    EXPECT_EQ(engine->step_profiler_.profiled_steps_, 0);
    EXPECT_EQ(engine->step_profiler_.profiler_, nullptr);
}

INSTANTIATE_TEST_SUITE_P(SerialAndParallel, EngineStopTest, ::testing::Values(0, 2));

}  // namespace
}  // namespace rtp_llm
