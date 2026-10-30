#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "autil/EnvUtil.h"
#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;

// Only the executor/transport is substituted: the real engine loop, FIFO,
// arm/cancel/commit APIs and thread joins run on both ranks.
struct TerminalTransport {
    std::mutex                      mutex;
    std::condition_variable         changed;
    std::vector<bool>               inputs;
    bool                            hold_process    = false;
    bool                            shutdown        = false;
    bool                            release_on_stop = true;
    std::atomic<int>                stop_notifications{0};
    std::array<std::atomic<int>, 2> entered{{0, 0}};
    std::array<std::atomic<int>, 2> completed{{0, 0}};
    std::atomic<int>                terminals{0};
};

class TerminalExecutor: public Executor {
public:
    TerminalExecutor(TerminalTransport& transport, int rank): transport_(transport), rank_(rank) {}
    bool supportsTerminalStop() const override {
        return true;
    }
    void requestTerminalStop() override {
        requested_ = true;
    }
    bool terminalStopObserved() const override {
        return observed_;
    }
    void notifyStop() override {
        std::lock_guard<std::mutex> lock(transport_.mutex);
        ++transport_.stop_notifications;
        if (!transport_.release_on_stop) {
            return;
        }
        transport_.shutdown = true;
        transport_.changed.notify_all();
    }
    absl::Status process(const std::list<GenerateStreamPtr>&, int64_t = 0) override {
        if (!error.ok()) {
            return error;
        }
        if (throw_error) {
            throw std::runtime_error("injected process exception");
        }
        std::unique_lock<std::mutex> lock(transport_.mutex);
        if (rank_ == 0) {
            transport_.inputs.push_back(requested_);
            if (requested_) {
                ++transport_.terminals;
            }
            transport_.changed.notify_all();
        }
        transport_.changed.wait(lock, [&]() { return transport_.shutdown || index_ < transport_.inputs.size(); });
        if (transport_.shutdown) {
            return absl::OkStatus();
        }
        const bool terminal = transport_.inputs[index_++];
        ++transport_.entered[rank_];
        transport_.changed.notify_all();
        transport_.changed.wait(lock, [&]() { return transport_.shutdown || !transport_.hold_process; });
        observed_ = terminal;
        ++transport_.completed[rank_];
        return absl::OkStatus();
    }
    absl::Status error       = absl::OkStatus();
    bool         throw_error = false;

private:
    TerminalTransport& transport_;
    const int          rank_;
    size_t             index_     = 0;
    bool               requested_ = false;
    bool               observed_  = false;
};

class NormalEngineTerminalStopTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        autil::EnvGuard startup_warmup("RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS", "");
        for (int rank = 0; rank < 2; ++rank) {
            ModelConfig   model;
            RuntimeConfig runtime;
            KVCacheConfig cache;
            auto          params               = createEngineInitParams(CustomConfig{}, model, runtime, cache);
            NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
                return std::make_unique<MockModel>(model.vocab_size);
            };
            try {
                engines_[rank] = std::make_shared<NormalEngine>(params, nullptr, true);
            } catch (...) {
                NormalExecutor::test_model_factory = nullptr;
                throw;
            }
            NormalExecutor::test_model_factory = nullptr;
            // Model construction is single-device; only executor transport is
            // replaced for the two-rank engine control-plane test.
            auto& pc   = engines_[rank]->parallelism_config;
            pc.tp_size = pc.world_size = pc.ep_size = pc.ffn_tp_size = 2;
            pc.tp_rank = pc.world_rank = pc.ep_rank = pc.ffn_tp_rank = rank;
            pc.prefill_cp_config.prefill_cp_size                     = 2;
            auto executor             = std::make_unique<TerminalExecutor>(transport_, rank);
            executors_[rank]          = executor.get();
            engines_[rank]->executor_ = std::move(executor);
        }
    }

    void TearDown() override {
        // Failure cleanup only; successful late-root tests must finish their
        // cancellation ACKs and loop joins with transport stop-release disabled.
        {
            std::lock_guard<std::mutex> lock(transport_.mutex);
            transport_.release_on_stop = true;
        }
        for (auto& engine : engines_) {
            if (engine) {
                engine->requestForceStop();
            }
        }
        for (auto& thread : controls_) {
            thread.join();
        }
        for (auto& engine : engines_) {
            if (engine) {
                (void)engine->forceStop();
                engine.reset();
            }
        }
        DeviceTestBase::TearDown();
    }

    std::future<absl::Status> arm(int rank) {
        std::packaged_task<absl::Status()> task([&, rank]() { return engines_[rank]->armStopAtStep(0, 2000); });
        auto                               result = task.get_future();
        controls_.emplace_back(std::move(task));
        return result;
    }
    void expectReady(std::future<absl::Status>& result) {
        ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
        EXPECT_TRUE(result.get().ok());
    }
    void start() {
        for (auto& engine : engines_) {
            ASSERT_TRUE(engine->startLoop().ok());
        }
    }
    void commit() {
        for (auto& engine : engines_) {
            EXPECT_TRUE(engine->stopAtStep(0).ok());
        }
    }

    TerminalTransport                            transport_;
    std::array<std::shared_ptr<NormalEngine>, 2> engines_;
    std::array<TerminalExecutor*, 2>             executors_;
    std::vector<std::thread>                     controls_;
};

TEST_F(NormalEngineTerminalStopTest, IdleFifoStopsWithOneTerminalAndNoIdleBudget) {
    start();
    EXPECT_EQ(engines_[0]->completedSteps(), EngineBase::kTerminalStopCapability);
    auto root     = arm(0);
    auto follower = arm(1);
    expectReady(root);
    expectReady(follower);
    commit();
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
    EXPECT_TRUE(engines_[0]->stopAtStep(0).ok());
}

TEST_F(NormalEngineTerminalStopTest, PausedEnginesReachTerminalBoundaryOnArm) {
    for (auto& engine : engines_) {
        engine->pause();
    }
    start();
    std::this_thread::sleep_for(20ms);
    EXPECT_EQ(transport_.completed[0], 0);
    EXPECT_EQ(transport_.completed[1], 0);
    auto follower = arm(1);
    auto root     = arm(0);
    expectReady(root);
    expectReady(follower);
    commit();
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
}

TEST_F(NormalEngineTerminalStopTest, FollowerTimeoutAndLateRootAbortReachCommonBoundaryBeforeCancellation) {
    transport_.release_on_stop = false;
    start();
    // Both loops are live: follower is in input sync and root is in idle FIFO.
    ASSERT_EQ(engines_[1]->armStopAtStep(0, 20).code(), absl::StatusCode::kDeadlineExceeded);
    EXPECT_EQ(transport_.terminals, 0);
    EXPECT_EQ(engines_[0]->cancelArmedStop(20).code(), absl::StatusCode::kFailedPrecondition);

    // Shared abort recovery re-arms the timed-out follower and the late root.
    // No notifyStop/force-stop can release the transport in this test.
    auto follower = arm(1);
    EXPECT_EQ(follower.wait_for(20ms), std::future_status::timeout);
    engines_[0]->pause();  // abort recovery must also release a paused late root
    auto root = arm(0);
    ASSERT_EQ(root.wait_for(3s), std::future_status::ready);
    ASSERT_TRUE(root.get().ok());
    ASSERT_EQ(follower.wait_for(3s), std::future_status::ready);
    ASSERT_TRUE(follower.get().ok());
    for (auto& engine : engines_) {
        ASSERT_TRUE(engine->cancelArmedStop(2000).ok());
    }
    EXPECT_EQ(transport_.stop_notifications, 0);
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
    for (auto& engine : engines_) {
        EXPECT_TRUE(engine->forceStop().ok());
    }
    EXPECT_FALSE(transport_.shutdown);  // loop joins did not unblock a fake collective
}

TEST_F(NormalEngineTerminalStopTest, ArmWaitsForInflightProcessBeforeTerminalBroadcast) {
    transport_.hold_process = true;
    start();
    static_cast<FIFOScheduler&>(engines_[0]->getScheduler()).notifyControlWakeup();
    {
        std::unique_lock<std::mutex> lock(transport_.mutex);
        ASSERT_TRUE(transport_.changed.wait_for(
            lock, 2s, [&]() { return transport_.entered[0] == 1 && transport_.entered[1] == 1; }));
    }
    auto root     = arm(0);
    auto follower = arm(1);
    EXPECT_EQ(root.wait_for(20ms), std::future_status::timeout);
    EXPECT_EQ(follower.wait_for(20ms), std::future_status::timeout);
    EXPECT_EQ(transport_.terminals, 0);
    EXPECT_EQ(transport_.completed[0], 0);
    {
        std::lock_guard<std::mutex> lock(transport_.mutex);
        transport_.hold_process = false;
        transport_.changed.notify_all();
    }
    expectReady(root);
    expectReady(follower);
    commit();
    EXPECT_EQ(transport_.terminals, 1);
    EXPECT_EQ(transport_.completed[0], 2);
    EXPECT_EQ(transport_.completed[1], 2);
}

TEST_F(NormalEngineTerminalStopTest, LateFollowerArmAndCommitDoNotReenterProcess) {
    start();
    auto root = arm(0);
    expectReady(root);
    // Root's broadcast may beat this rank's control-plane arm.
    auto follower = arm(1);
    expectReady(follower);
    EXPECT_TRUE(engines_[0]->stopAtStep(0).ok());
    std::this_thread::sleep_for(20ms);
    EXPECT_EQ(transport_.completed[1], 1);
    EXPECT_TRUE(engines_[1]->armStopAtStep(0, 20).ok());
    EXPECT_TRUE(engines_[1]->stopAtStep(0).ok());
    EXPECT_EQ(transport_.terminals, 1);
}

TEST_F(NormalEngineTerminalStopTest, CancellationParksAtBoundaryUntilForceStop) {
    start();
    auto root     = arm(0);
    auto follower = arm(1);
    expectReady(root);
    expectReady(follower);
    for (auto& engine : engines_) {
        EXPECT_TRUE(engine->cancelArmedStop(2000).ok());
        EXPECT_FALSE(engine->stopAtStep(0).ok());
        EXPECT_FALSE(engine->armStopAtStep(0, 20).ok());
    }
    EXPECT_EQ(transport_.completed[0], 1);
    EXPECT_EQ(transport_.completed[1], 1);
    for (auto& engine : engines_) {
        EXPECT_TRUE(engine->forceStop().ok());
        EXPECT_FALSE(engine->stopAtStep(0).ok());  // stale observation cannot authorize a new commit
    }
    EXPECT_EQ(transport_.terminals, 1);
}

TEST_F(NormalEngineTerminalStopTest, InvalidTopologyAndOtherModesRejectArmWithoutExecuting) {
    auto&      engine                                = *engines_[0];
    const auto pc                                    = engine.parallelism_config;
    const auto runtime                               = engine.runtime_config;
    engine.running_                                  = true;
    const std::vector<std::function<void()>> invalid = {
        [&]() { engine.parallelism_config.world_size = 4; },
        [&]() { engine.parallelism_config.pp_size = 2; },
        [&]() { engine.parallelism_config.dp_size = 2; },
        [&]() { engine.parallelism_config.tp_rank = 1; },
        [&]() { engine.parallelism_config.ep_size = 4; },
        [&]() { engine.parallelism_config.ffn_tp_size = 0; },
        [&]() { engine.parallelism_config.prefill_cp_config.prefill_cp_size = 4; },
        [&]() { engine.parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate = true; },
        [&]() { engine.runtime_config.use_batch_decode_scheduler = true; },
        [&]() { engine.runtime_config.use_gather_batch_scheduler = true; },
    };
    for (const auto& change : invalid) {
        engine.parallelism_config = pc;
        engine.runtime_config     = runtime;
        change();
        EXPECT_EQ(engine.armStopAtStep(0, 1).code(), absl::StatusCode::kFailedPrecondition);
        EXPECT_EQ(engine.stopAtStep(0).code(), absl::StatusCode::kFailedPrecondition);
    }
    engine.parallelism_config                              = pc;
    engine.runtime_config                                  = runtime;
    engine.ffn_disaggregate_config.enable_ffn_disaggregate = true;
    EXPECT_FALSE(engine.armStopAtStep(0, 1).ok());
    engine.ffn_disaggregate_config.enable_ffn_disaggregate = false;
    class UnsupportedExecutor: public Executor {
    public:
        absl::Status process(const std::list<GenerateStreamPtr>&, int64_t = 0) override {
            return absl::InternalError("unsupported executor must not run");
        }
    };
    auto executor    = std::move(engine.executor_);
    engine.executor_ = std::make_unique<UnsupportedExecutor>();
    EXPECT_EQ(engine.armStopAtStep(0, 1).code(), absl::StatusCode::kFailedPrecondition);
    engine.executor_ = std::move(executor);
    auto scheduler   = std::move(engine.scheduler_);
    EXPECT_EQ(engine.armStopAtStep(0, 1).code(), absl::StatusCode::kFailedPrecondition);
    engine.scheduler_ = std::move(scheduler);
    EXPECT_EQ(engine.armStopAtStep(1, 1).code(), absl::StatusCode::kInvalidArgument);
    engine.running_ = false;
    EXPECT_FALSE(engine.armStopAtStep(0, 1).ok());  // loop not started
    EXPECT_EQ(transport_.terminals, 0);
    engine.propose_params_            = std::make_unique<ProposeModelEngineInitParams>();
    engine.published_completed_steps_ = 64;
    EXPECT_EQ(engine.completedSteps(), 64);  // MTP retains step snapshots.
    engine.propose_params_.reset();
}

TEST_F(NormalEngineTerminalStopTest, ProcessErrorsCannotAcknowledgeTerminalOrCommit) {
    auto& engine    = *engines_[0];
    engine.running_ = true;
    EXPECT_EQ(engine.armStopAtStep(0, 1).code(), absl::StatusCode::kDeadlineExceeded);
    executors_[0]->error = absl::InternalError("injected process failure");
    EXPECT_EQ(engine.step().code(), absl::StatusCode::kInternal);
    EXPECT_FALSE(engine.stopAtStep(0).ok());
    EXPECT_EQ(engine.cancelArmedStop(1).code(), absl::StatusCode::kDeadlineExceeded);
    executors_[0]->error       = absl::OkStatus();
    executors_[0]->throw_error = true;
    static_cast<FIFOScheduler&>(engine.getScheduler()).notifyControlWakeup();
    EXPECT_THROW(engine.step(), std::runtime_error);
    EXPECT_EQ(transport_.terminals, 0);
    engine.requestForceStop();
    EXPECT_FALSE(engine.armStopAtStep(0, 1).ok());
    EXPECT_FALSE(engine.stopAtStep(0).ok());
}

}  // namespace
}  // namespace rtp_llm
