#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"
#include "rtp_llm/cpp/engine_base/sleep/SleepMemoryPolicy.h"

namespace rtp_llm {

namespace {

class ScopedEnvironment {
public:
    ScopedEnvironment(const char* name, const char* value): name_(name) {
        const char* old_value = std::getenv(name_);
        if (old_value != nullptr) {
            had_old_value_ = true;
            old_value_     = old_value;
        }
        if (value != nullptr) {
            setenv(name_, value, 1);
        } else {
            unsetenv(name_);
        }
    }

    ~ScopedEnvironment() {
        if (had_old_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool        had_old_value_{false};
    std::string old_value_;
};

SleepOptions gracefulOptions() {
    SleepOptions opt;
    opt.mode       = "wait";
    opt.timeout_ms = 1000;
    return opt;
}

}  // namespace

TEST(SleepMemoryPolicyTest, CudaGraphVmmRegionIsEnabledOnlyForSleepLevelOneOrTwo) {
    for (const char* level : {"1", "2"}) {
        ScopedEnvironment enabled("ENABLE_SLEEP_MODE", "1");
        ScopedEnvironment configured_level("SLEEP_MODE_LEVEL", level);
        EXPECT_TRUE(sleep_memory_policy::useCudaGraphVmmRegionFromEnvironment());
    }

    {
        ScopedEnvironment enabled("ENABLE_SLEEP_MODE", "1");
        ScopedEnvironment configured_level("SLEEP_MODE_LEVEL", "3");
        EXPECT_FALSE(sleep_memory_policy::useCudaGraphVmmRegionFromEnvironment());
    }

    {
        ScopedEnvironment enabled("ENABLE_SLEEP_MODE", "0");
        ScopedEnvironment configured_level("SLEEP_MODE_LEVEL", "3");
        EXPECT_FALSE(sleep_memory_policy::useCudaGraphVmmRegionFromEnvironment());
    }

    {
        ScopedEnvironment enabled("ENABLE_SLEEP_MODE", nullptr);
        ScopedEnvironment configured_level("SLEEP_MODE_LEVEL", "1");
        EXPECT_FALSE(sleep_memory_policy::useCudaGraphVmmRegionFromEnvironment());
    }

    {
        ScopedEnvironment enabled("ENABLE_SLEEP_MODE", "1");
        ScopedEnvironment configured_level("SLEEP_MODE_LEVEL", nullptr);
        EXPECT_TRUE(sleep_memory_policy::useCudaGraphVmmRegionFromEnvironment());
    }
}

TEST(SleepMemoryPolicyTest, LevelThreeDoesNotManageCudaGraphVmmBacking) {
    EXPECT_TRUE(sleep_memory_policy::manageCudaGraphVmmBacking(1));
    EXPECT_TRUE(sleep_memory_policy::manageCudaGraphVmmBacking(2));
    EXPECT_FALSE(sleep_memory_policy::manageCudaGraphVmmBacking(3));
}

TEST(SleepLifecycleControllerTest, InitialStateIsRunning) {
    SleepLifecycleController controller(true);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_TRUE(controller.admit());
    EXPECT_TRUE(controller.enabled());
    EXPECT_TRUE(controller.effective());
    EXPECT_EQ(controller.sleepEpoch(), 0);

    const auto status = controller.status();
    EXPECT_TRUE(status.sleep_mode_enabled);
    EXPECT_TRUE(status.effective);
    EXPECT_EQ(status.supported_levels, std::vector<int32_t>{1});
    EXPECT_EQ(status.state, SleepState::RUNNING);
    EXPECT_TRUE(status.device_kv_cache_valid);
    EXPECT_EQ(status.kv_memory_state, "ACTIVE");
}

TEST(SleepLifecycleControllerTest, DisabledByDefaultRejectsSleepAndReportsCapability) {
    SleepLifecycleController controller;
    EXPECT_FALSE(controller.enabled());
    EXPECT_FALSE(controller.effective());

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.code, SleepResult::Code::DISABLED);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_TRUE(controller.admit());

    const auto status = controller.status();
    EXPECT_FALSE(status.sleep_mode_enabled);
    EXPECT_FALSE(status.effective);
    EXPECT_TRUE(status.supported_levels.empty());
    EXPECT_FALSE(status.disabled_reason.empty());
}

TEST(SleepLifecycleControllerTest, RuntimeUnsupportedReportsNotEffectiveEvenWhenEnabled) {
    SleepLifecycleController controller(true);
    controller.setRuntimeSupport(false, "torch_memory_saver preload shim is not available");

    EXPECT_TRUE(controller.enabled());
    EXPECT_FALSE(controller.runtimeSupported());
    EXPECT_FALSE(controller.effective());

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.code, SleepResult::Code::DISABLED);
    EXPECT_NE(result.message.find("torch_memory_saver"), std::string::npos);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);

    const auto status = controller.status();
    EXPECT_TRUE(status.sleep_mode_enabled);
    EXPECT_FALSE(status.effective);
    EXPECT_TRUE(status.supported_levels.empty());
    EXPECT_TRUE(status.supported_modes.empty());
    EXPECT_NE(status.disabled_reason.find("torch_memory_saver"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, SleepWithDefaultHooksReachesSleeping) {
    SleepLifecycleController controller(true);
    const auto               result = controller.sleep(gracefulOptions());
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    EXPECT_FALSE(controller.admit());
    EXPECT_EQ(controller.sleepEpoch(), 1);

    const auto status = controller.status();
    // Empty hooks are no-op success for core state-machine unit tests. Without
    // an injected KV release hook, resource-specific KV status stays active.
    EXPECT_EQ(status.kv_memory_state, "ACTIVE");
    EXPECT_TRUE(status.device_kv_cache_valid);
    EXPECT_EQ(status.gpu_resource_state, "RELEASED");
}

TEST(SleepLifecycleControllerTest, LevelZeroIsDefinedButUnimplemented) {
    SleepLifecycleController controller(true);
    auto                     opt = gracefulOptions();
    opt.level                    = 0;

    const auto result = controller.sleep(opt);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.code, SleepResult::Code::UNIMPLEMENTED);
    EXPECT_NE(result.message.find("level=0"), std::string::npos);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_EQ(controller.status().supported_levels, std::vector<int32_t>{1});
}

TEST(SleepLifecycleControllerTest, DefaultModeRejectsLevelTwo) {
    SleepLifecycleController controller(true);
    auto                     opt = gracefulOptions();
    opt.level                    = 2;

    const auto result = controller.sleep(opt);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.code, SleepResult::Code::INVALID_ARGUMENT);
    EXPECT_NE(result.message.find("level=2"), std::string::npos);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_EQ(controller.status().supported_levels, std::vector<int32_t>{1});
}

TEST(SleepLifecycleControllerTest, DiscardModeSupportsLevelTwo) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(2);

    EXPECT_TRUE(controller.discardWeights());
    EXPECT_EQ(controller.status().supported_levels, std::vector<int32_t>{2});

    auto opt          = gracefulOptions();
    opt.level         = 2;
    const auto result = controller.sleep(opt);
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    EXPECT_EQ(controller.activeSleepLevel(), 2);
}

TEST(SleepLifecycleControllerTest, DiscardModeRejectsLevelOne) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(2);

    auto opt          = gracefulOptions();
    opt.level         = 1;
    const auto result = controller.sleep(opt);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.code, SleepResult::Code::INVALID_ARGUMENT);
    EXPECT_NE(result.message.find("level=1"), std::string::npos);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
}

TEST(SleepLifecycleControllerTest, LevelThreeIsStrictAndDiscardsWeights) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    EXPECT_EQ(controller.configuredLevel(), 3);
    EXPECT_TRUE(controller.discardWeights());
    EXPECT_EQ(controller.status().supported_levels, std::vector<int32_t>{3});

    for (const int32_t mismatched_level : {1, 2}) {
        auto opt  = gracefulOptions();
        opt.level = mismatched_level;
        EXPECT_EQ(controller.sleep(opt).code, SleepResult::Code::INVALID_ARGUMENT);
        EXPECT_EQ(controller.state(), SleepState::RUNNING);
    }
}

TEST(SleepLifecycleControllerTest, InvalidConfiguredLevelIsRejected) {
    SleepLifecycleController controller(true);
    EXPECT_THROW(controller.setConfiguredLevel(0), std::invalid_argument);
    EXPECT_THROW(controller.setConfiguredLevel(4), std::invalid_argument);
    EXPECT_EQ(controller.configuredLevel(), 1);
}

TEST(SleepLifecycleControllerTest, LevelThreeHooksRunInDependencyOrder) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.freezeExternalTransfers = [&calls](const SleepOptions& opt) {
        EXPECT_EQ(opt.level, 3);
        calls.push_back("freeze_rdma");
        return true;
    };
    hooks.quiesceEngine = [&calls](const SleepOptions&) {
        calls.push_back("quiesce");
        return true;
    };
    hooks.teardownRdmaTransports = [&calls](const SleepOptions& opt) {
        EXPECT_EQ(opt.level, 3);
        calls.push_back("teardown_rdma");
        return true;
    };
    hooks.synchronizeAndDeregisterMr = [&calls](const SleepOptions&) {
        calls.push_back("deregister_mr");
        return true;
    };
    hooks.teardownCollectives = [&calls](const SleepOptions& opt) {
        EXPECT_EQ(opt.level, 3);
        calls.push_back("teardown_collectives");
        return true;
    };
    hooks.releaseKvMemoryBacking = [&calls](const SleepOptions&) {
        calls.push_back("release_kv");
        return true;
    };
    hooks.releaseRestorableGpuMemory = [&calls](const SleepOptions&) {
        calls.push_back("release_weights");
        return true;
    };
    hooks.restoreRestorableGpuMemory = [&calls]() {
        calls.push_back("restore_weights");
        return true;
    };
    hooks.restoreKvMemoryBackingAndResetMetadata = [&calls]() {
        calls.push_back("restore_kv");
        return true;
    };
    hooks.rebuildCollectives = [&calls]() {
        calls.push_back("rebuild_collectives");
        return true;
    };
    hooks.rebuildRdmaTransports = [&calls]() {
        calls.push_back("rebuild_rdma");
        return true;
    };
    hooks.registerMr = [&calls]() {
        calls.push_back("register_mr");
        return true;
    };
    hooks.recaptureCollectiveGraphs = [&calls]() {
        calls.push_back("recapture_graphs");
        return true;
    };
    hooks.restartEngine = [&calls]() {
        calls.push_back("restart");
        return true;
    };
    hooks.resumeExternalTransfers = [&calls]() {
        calls.push_back("resume_rdma");
        return true;
    };
    hooks.coordinateResourcePhase = [&calls](const std::string& phase, int64_t epoch, bool local_success) {
        EXPECT_EQ(epoch, 1);
        EXPECT_TRUE(local_success);
        calls.push_back(phase);
        return local_success;
    };
    controller.setHooks(hooks);

    auto sleep_opt  = gracefulOptions();
    sleep_opt.level = 3;
    ASSERT_TRUE(controller.sleep(sleep_opt).ok);

    WakeUpOptions wake_prepare;
    wake_prepare.prepare_only = true;
    ASSERT_TRUE(controller.wakeUp(wake_prepare).ok);
    EXPECT_EQ(calls.back(), "register_mr");

    WakeUpOptions wake_commit;
    wake_commit.commit_only = true;
    ASSERT_TRUE(controller.wakeUp(wake_commit).ok);

    EXPECT_EQ(calls,
              (std::vector<std::string>{"freeze_rdma",
                                        "quiesce",
                                        "teardown_rdma",
                                        "deregister_mr",
                                        "collective_teardown_ready",
                                        "teardown_collectives",
                                        "collective_teardown_done",
                                        "release_kv",
                                        "release_weights",
                                        "restore_weights",
                                        "restore_kv",
                                        "collective_rebuild_ready",
                                        "rebuild_collectives",
                                        "collective_rebuild_done",
                                        "rebuild_rdma",
                                        "register_mr",
                                        "graph_recapture_ready",
                                        "recapture_graphs",
                                        "graph_recapture_done",
                                        "restart",
                                        "resume_rdma"}));
}

TEST(SleepLifecycleControllerTest, LevelThreeTeardownReadyCoordinatesLocalFailure) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    std::vector<std::tuple<std::string, int64_t, bool>> phases;
    int                                                 collective_called = 0;
    SleepHooks                                          hooks;
    hooks.teardownRdmaTransports  = [](const SleepOptions&) { return false; };
    hooks.coordinateResourcePhase = [&phases](const std::string& phase, int64_t epoch, bool local_success) {
        phases.emplace_back(phase, epoch, local_success);
        return local_success;
    };
    hooks.teardownCollectives = [&collective_called](const SleepOptions&) {
        ++collective_called;
        return true;
    };
    controller.setHooks(hooks);

    auto opt  = gracefulOptions();
    opt.level = 3;
    EXPECT_FALSE(controller.sleep(opt).ok);

    ASSERT_EQ(phases.size(), 1);
    EXPECT_EQ(std::get<0>(phases[0]), "collective_teardown_ready");
    EXPECT_EQ(std::get<1>(phases[0]), 1);
    EXPECT_FALSE(std::get<2>(phases[0]));
    EXPECT_EQ(collective_called, 0);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
}

TEST(SleepLifecycleControllerTest, LevelThreeTeardownDoneCoordinatesLocalFailure) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    std::vector<std::tuple<std::string, int64_t, bool>> phases;
    SleepHooks                                          hooks;
    hooks.coordinateResourcePhase = [&phases](const std::string& phase, int64_t epoch, bool local_success) {
        phases.emplace_back(phase, epoch, local_success);
        return local_success;
    };
    hooks.teardownCollectives = [](const SleepOptions&) { return false; };
    controller.setHooks(hooks);

    auto opt  = gracefulOptions();
    opt.level = 3;
    EXPECT_FALSE(controller.sleep(opt).ok);

    ASSERT_EQ(phases.size(), 2);
    EXPECT_EQ(std::get<0>(phases[0]), "collective_teardown_ready");
    EXPECT_TRUE(std::get<2>(phases[0]));
    EXPECT_EQ(std::get<0>(phases[1]), "collective_teardown_done");
    EXPECT_EQ(std::get<1>(phases[1]), 1);
    EXPECT_FALSE(std::get<2>(phases[1]));
    EXPECT_EQ(controller.state(), SleepState::ERROR);
}

TEST(SleepLifecycleControllerTest, LevelThreeRebuildReadyCoordinatesLocalFailure) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    std::vector<std::tuple<std::string, int64_t, bool>> phases;
    int                                                 rebuild_called = 0;
    SleepHooks                                          hooks;
    hooks.coordinateResourcePhase = [&phases](const std::string& phase, int64_t epoch, bool local_success) {
        phases.emplace_back(phase, epoch, local_success);
        return local_success;
    };
    hooks.restoreRestorableGpuMemory = []() { return false; };
    hooks.rebuildCollectives         = [&rebuild_called]() {
        ++rebuild_called;
        return true;
    };
    controller.setHooks(hooks);

    auto opt  = gracefulOptions();
    opt.level = 3;
    ASSERT_TRUE(controller.sleep(opt).ok);
    phases.clear();

    EXPECT_FALSE(controller.wakeUp().ok);
    ASSERT_FALSE(phases.empty());
    EXPECT_EQ(std::get<0>(phases[0]), "collective_rebuild_ready");
    EXPECT_EQ(std::get<1>(phases[0]), 1);
    EXPECT_FALSE(std::get<2>(phases[0]));
    EXPECT_EQ(rebuild_called, 0);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
}

TEST(SleepLifecycleControllerTest, LevelThreeRebuildDoneCoordinatesLocalFailure) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    std::vector<std::tuple<std::string, int64_t, bool>> phases;
    SleepHooks                                          hooks;
    hooks.coordinateResourcePhase = [&phases](const std::string& phase, int64_t epoch, bool local_success) {
        phases.emplace_back(phase, epoch, local_success);
        return local_success;
    };
    hooks.rebuildCollectives = []() { return false; };
    controller.setHooks(hooks);

    auto opt  = gracefulOptions();
    opt.level = 3;
    ASSERT_TRUE(controller.sleep(opt).ok);
    phases.clear();

    EXPECT_FALSE(controller.wakeUp().ok);
    ASSERT_GE(phases.size(), 2);
    EXPECT_EQ(std::get<0>(phases[0]), "collective_rebuild_ready");
    EXPECT_TRUE(std::get<2>(phases[0]));
    EXPECT_EQ(std::get<0>(phases[1]), "collective_rebuild_done");
    EXPECT_EQ(std::get<1>(phases[1]), 1);
    EXPECT_FALSE(std::get<2>(phases[1]));
    EXPECT_EQ(controller.state(), SleepState::ERROR);
}

TEST(SleepLifecycleControllerTest, LevelThreeGraphReadyCoordinatesLocalFailure) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    std::vector<std::tuple<std::string, int64_t, bool>> phases;
    int                                                 graph_called = 0;
    SleepHooks                                          hooks;
    hooks.coordinateResourcePhase = [&phases](const std::string& phase, int64_t epoch, bool local_success) {
        phases.emplace_back(phase, epoch, local_success);
        return local_success;
    };
    hooks.registerMr                = []() { return false; };
    hooks.recaptureCollectiveGraphs = [&graph_called]() {
        ++graph_called;
        return true;
    };
    controller.setHooks(hooks);

    auto opt  = gracefulOptions();
    opt.level = 3;
    ASSERT_TRUE(controller.sleep(opt).ok);
    phases.clear();

    EXPECT_FALSE(controller.wakeUp().ok);
    ASSERT_GE(phases.size(), 3);
    const auto& graph_ready = phases.back();
    EXPECT_EQ(std::get<0>(graph_ready), "graph_recapture_ready");
    EXPECT_EQ(std::get<1>(graph_ready), 1);
    EXPECT_FALSE(std::get<2>(graph_ready));
    EXPECT_EQ(graph_called, 0);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
}

TEST(SleepLifecycleControllerTest, LevelThreeGraphDoneCoordinatesLocalFailure) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    std::vector<std::tuple<std::string, int64_t, bool>> phases;
    SleepHooks                                          hooks;
    hooks.coordinateResourcePhase = [&phases](const std::string& phase, int64_t epoch, bool local_success) {
        phases.emplace_back(phase, epoch, local_success);
        return local_success;
    };
    hooks.recaptureCollectiveGraphs = []() { return false; };
    controller.setHooks(hooks);

    auto opt  = gracefulOptions();
    opt.level = 3;
    ASSERT_TRUE(controller.sleep(opt).ok);
    phases.clear();

    EXPECT_FALSE(controller.wakeUp().ok);
    ASSERT_GE(phases.size(), 4);
    const auto& graph_ready = phases[phases.size() - 2];
    const auto& graph_done  = phases.back();
    EXPECT_EQ(std::get<0>(graph_ready), "graph_recapture_ready");
    EXPECT_TRUE(std::get<2>(graph_ready));
    EXPECT_EQ(std::get<0>(graph_done), "graph_recapture_done");
    EXPECT_EQ(std::get<1>(graph_done), 1);
    EXPECT_FALSE(std::get<2>(graph_done));
    EXPECT_EQ(controller.state(), SleepState::ERROR);
}

TEST(SleepLifecycleControllerTest, LevelThreeArmFailureIsTerminalAndFailClosed) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    bool                     transfers_frozen = false;
    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.freezeExternalTransfers = [&transfers_frozen, &calls](const SleepOptions&) {
        transfers_frozen = true;
        calls.push_back("freeze");
        return true;
    };
    hooks.armEngineQuiesce = [&calls](const SleepOptions&) {
        calls.push_back("arm");
        return false;
    };
    hooks.drain = [&calls](const SleepOptions&, const DrainCancellationPredicate&) {
        calls.push_back("drain");
        return true;
    };
    hooks.quiesceEngine = [&calls](const SleepOptions&) {
        calls.push_back("quiesce");
        return true;
    };
    controller.setHooks(hooks);

    auto opt          = gracefulOptions();
    opt.level         = 3;
    const auto result = controller.sleep(opt);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admit());
    EXPECT_TRUE(transfers_frozen);
    EXPECT_EQ(calls, (std::vector<std::string>{"drain", "freeze", "drain", "arm"}));
    EXPECT_NE(controller.status().last_error.find("armEngineQuiesce"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, LevelThreeArmExceptionIsTerminalAndFailClosed) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    bool       transfers_frozen = false;
    int        drain_called     = 0;
    SleepHooks hooks;
    hooks.freezeExternalTransfers = [&transfers_frozen](const SleepOptions&) {
        transfers_frozen = true;
        return true;
    };
    hooks.armEngineQuiesce = [](const SleepOptions&) -> bool { throw std::runtime_error("arm failed"); };
    hooks.drain            = [&drain_called](const SleepOptions&, const DrainCancellationPredicate&) {
        ++drain_called;
        return true;
    };
    controller.setHooks(hooks);

    auto opt          = gracefulOptions();
    opt.level         = 3;
    const auto result = controller.sleep(opt);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_TRUE(transfers_frozen);
    EXPECT_EQ(drain_called, 2);
    EXPECT_NE(controller.status().last_error.find("armEngineQuiesce"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, LevelOneAndTwoKeepBestEffortArmBehavior) {
    for (const int32_t level : {1, 2}) {
        SCOPED_TRACE(level);
        SleepLifecycleController controller(true);
        controller.setConfiguredLevel(level);

        int        drain_called = 0;
        SleepHooks hooks;
        hooks.armEngineQuiesce = [](const SleepOptions&) { return false; };
        hooks.drain            = [&drain_called](const SleepOptions&, const DrainCancellationPredicate&) {
            ++drain_called;
            return true;
        };
        controller.setHooks(hooks);

        auto opt          = gracefulOptions();
        opt.level         = level;
        const auto result = controller.sleep(opt);
        EXPECT_TRUE(result.ok) << result.message;
        EXPECT_EQ(controller.state(), SleepState::SLEEPING);
        EXPECT_EQ(drain_called, 1);
    }
}

TEST(SleepLifecycleControllerTest, FreezeFailureAfterInitialDrainCanBeCancelledOnEveryLevel) {
    for (const int32_t level : {1, 2, 3}) {
        SCOPED_TRACE(level);
        SleepLifecycleController controller(true);
        controller.setConfiguredLevel(level);
        std::atomic<int> drain_called{0};
        std::atomic<int> resume_called{0};
        SleepHooks       hooks;
        hooks.freezeExternalTransfers = [level](const SleepOptions& opt) {
            EXPECT_EQ(opt.level, level);
            return false;
        };
        hooks.drain = [&drain_called](const SleepOptions&, const DrainCancellationPredicate&) {
            ++drain_called;
            return true;
        };
        hooks.resumeExternalTransfers = [&resume_called]() {
            ++resume_called;
            return true;
        };
        controller.setHooks(hooks);

        auto opt  = gracefulOptions();
        opt.level = level;
        EXPECT_FALSE(controller.sleep(opt).ok);
        EXPECT_EQ(controller.state(), SleepState::DRAINING);
        EXPECT_EQ(drain_called.load(), 1);
        EXPECT_TRUE(controller.wakeUp().ok);
        EXPECT_EQ(resume_called.load(), 1);
        EXPECT_EQ(controller.state(), SleepState::RUNNING);
    }
}

TEST(SleepLifecycleControllerTest, LevelThreeRdmaTeardownFailureStopsBeforeMrDeregistration) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);
    std::atomic<int> deregister_called{0};
    std::atomic<int> collective_teardown_called{0};
    SleepHooks       hooks;
    hooks.teardownRdmaTransports     = [](const SleepOptions&) { return false; };
    hooks.synchronizeAndDeregisterMr = [&deregister_called](const SleepOptions&) {
        ++deregister_called;
        return true;
    };
    hooks.teardownCollectives = [&collective_teardown_called](const SleepOptions&) {
        ++collective_teardown_called;
        return true;
    };
    controller.setHooks(hooks);

    auto opt          = gracefulOptions();
    opt.level         = 3;
    const auto result = controller.sleep(opt);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_EQ(deregister_called.load(), 0);
    EXPECT_EQ(collective_teardown_called.load(), 0);
    EXPECT_NE(controller.status().last_error.find("teardownRdmaTransports"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, LevelOneAndTwoSkipRdmaTeardownAndKeepMrReleaseOrder) {
    for (const int32_t level : {1, 2}) {
        SCOPED_TRACE(level);
        SleepLifecycleController controller(true);
        controller.setConfiguredLevel(level);

        std::vector<std::string> calls;
        SleepHooks               hooks;
        hooks.teardownRdmaTransports = [&calls](const SleepOptions&) {
            calls.push_back("teardown_rdma");
            return true;
        };
        hooks.synchronizeAndDeregisterMr = [&calls](const SleepOptions&) {
            calls.push_back("deregister_mr");
            return true;
        };
        hooks.releaseKvMemoryBacking = [&calls](const SleepOptions&) {
            calls.push_back("release_kv");
            return true;
        };
        controller.setHooks(hooks);

        auto opt          = gracefulOptions();
        opt.level         = level;
        const auto result = controller.sleep(opt);
        EXPECT_TRUE(result.ok) << result.message;
        EXPECT_EQ(calls, (std::vector<std::string>{"deregister_mr", "release_kv"}));
    }
}

TEST(SleepLifecycleControllerTest, LevelThreeTeardownFailureStopsBeforeMemoryRelease) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);
    std::atomic<int> release_called{0};
    SleepHooks       hooks;
    hooks.teardownCollectives    = [](const SleepOptions&) { return false; };
    hooks.releaseKvMemoryBacking = [&release_called](const SleepOptions&) {
        ++release_called;
        return true;
    };
    controller.setHooks(hooks);

    auto opt          = gracefulOptions();
    opt.level         = 3;
    const auto result = controller.sleep(opt);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_EQ(release_called.load(), 0);
    EXPECT_NE(controller.status().last_error.find("teardownCollectives"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, LevelThreeRebuildFailureStopsBeforeMrRegistration) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);
    std::atomic<int> register_called{0};
    SleepHooks       hooks;
    hooks.rebuildCollectives = []() { return false; };
    hooks.registerMr         = [&register_called]() {
        ++register_called;
        return true;
    };
    controller.setHooks(hooks);

    auto opt  = gracefulOptions();
    opt.level = 3;
    ASSERT_TRUE(controller.sleep(opt).ok);
    const auto result = controller.wakeUp();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_EQ(register_called.load(), 0);
    EXPECT_NE(controller.status().last_error.find("rebuildCollectives"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, LevelThreeRdmaRebuildFailureStopsBeforeMrRegistration) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);
    std::atomic<int> register_called{0};
    SleepHooks       hooks;
    hooks.rebuildRdmaTransports = []() { return false; };
    hooks.registerMr            = [&register_called]() {
        ++register_called;
        return true;
    };
    controller.setHooks(hooks);

    auto opt  = gracefulOptions();
    opt.level = 3;
    ASSERT_TRUE(controller.sleep(opt).ok);
    const auto result = controller.wakeUp();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_EQ(register_called.load(), 0);
    EXPECT_NE(controller.status().last_error.find("rebuildRdmaTransports"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, LevelOneAndTwoGateTransfersWithoutRunningLevelThreeLifecycle) {
    for (const int32_t level : {1, 2}) {
        SCOPED_TRACE(level);
        SleepLifecycleController controller(true);
        controller.setConfiguredLevel(level);
        bool                     transfers_frozen = false;
        std::atomic<int>         level_three_calls{0};
        std::vector<std::string> calls;
        SleepHooks               hooks;
        hooks.freezeExternalTransfers = [&transfers_frozen, &calls, level](const SleepOptions& opt) {
            EXPECT_EQ(opt.level, level);
            transfers_frozen = true;
            calls.push_back("freeze");
            return true;
        };
        hooks.drain = [&transfers_frozen, &calls](const SleepOptions&, const DrainCancellationPredicate&) {
            calls.push_back(transfers_frozen ? "drain_frozen" : "drain_open");
            return true;
        };
        hooks.teardownCollectives = [&level_three_calls](const SleepOptions&) {
            ++level_three_calls;
            return true;
        };
        hooks.rebuildCollectives = [&level_three_calls]() {
            ++level_three_calls;
            return true;
        };
        hooks.teardownRdmaTransports = [&level_three_calls](const SleepOptions&) {
            ++level_three_calls;
            return true;
        };
        hooks.rebuildRdmaTransports = [&level_three_calls]() {
            ++level_three_calls;
            return true;
        };
        hooks.recaptureCollectiveGraphs = [&level_three_calls]() {
            ++level_three_calls;
            return true;
        };
        hooks.restartEngine = [&transfers_frozen, &calls]() {
            EXPECT_TRUE(transfers_frozen);
            calls.push_back("restart");
            return true;
        };
        hooks.warmupAndHealthCheck = [&transfers_frozen, &calls]() {
            EXPECT_TRUE(transfers_frozen);
            calls.push_back("warmup");
            return true;
        };
        hooks.resumeExternalTransfers = [&transfers_frozen, &calls]() {
            EXPECT_TRUE(transfers_frozen);
            transfers_frozen = false;
            calls.push_back("resume");
            return true;
        };
        controller.setHooks(hooks);

        auto opt  = gracefulOptions();
        opt.level = level;
        ASSERT_TRUE(controller.sleep(opt).ok);
        ASSERT_TRUE(controller.wakeUp().ok);
        EXPECT_FALSE(transfers_frozen);
        EXPECT_EQ(level_three_calls.load(), 0);
        EXPECT_EQ(calls,
                  (std::vector<std::string>{
                      "drain_open", "freeze", "drain_frozen", "restart", "warmup", "resume"}));
    }
}

TEST(SleepLifecycleControllerTest, AdmittedRequestCanFinishDependentTransferBeforeTransferGateCloses) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(1);

    bool                     request_active   = true;
    bool                     transfer_active  = false;
    bool                     transfers_frozen = false;
    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.drain = [&](const SleepOptions&, const DrainCancellationPredicate&) {
        calls.push_back(transfers_frozen ? "drain_frozen" : "drain_open");
        if (!transfers_frozen) {
            // Model the production race: inference was admitted before DRAINING,
            // but its downstream KV transfer has not acquired a lease yet.
            EXPECT_TRUE(request_active);
            transfer_active = true;
            transfer_active = false;
            request_active  = false;
        }
        return !request_active && !transfer_active;
    };
    hooks.freezeExternalTransfers = [&](const SleepOptions&) {
        EXPECT_FALSE(request_active);
        EXPECT_FALSE(transfer_active);
        transfers_frozen = true;
        calls.push_back("freeze");
        return true;
    };
    controller.setHooks(hooks);

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    EXPECT_EQ(calls, (std::vector<std::string>{"drain_open", "freeze", "drain_frozen"}));
}

TEST(SleepLifecycleControllerTest, PostFreezeDrainCatchesTransferAcquiredAtGateBoundary) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(1);

    bool                     transfer_active  = false;
    bool                     transfers_frozen = false;
    int                      drain_calls      = 0;
    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.drain = [&](const SleepOptions&, const DrainCancellationPredicate&) {
        ++drain_calls;
        calls.push_back(transfers_frozen ? "drain_frozen" : "drain_open");
        if (transfers_frozen) {
            EXPECT_TRUE(transfer_active);
            transfer_active = false;
        }
        return !transfer_active;
    };
    hooks.freezeExternalTransfers = [&](const SleepOptions&) {
        // Model a transfer that acquired its lease just before close() won the
        // admission lock. It is allowed to finish, but no later lease can start.
        transfer_active  = true;
        transfers_frozen = true;
        calls.push_back("freeze");
        return true;
    };
    controller.setHooks(hooks);

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    EXPECT_EQ(drain_calls, 2);
    EXPECT_FALSE(transfer_active);
    EXPECT_EQ(calls, (std::vector<std::string>{"drain_open", "freeze", "drain_frozen"}));
}

TEST(SleepLifecycleControllerTest, WakeResumeFailureRefreezesTransfersOnEveryLevel) {
    for (const int32_t level : {1, 2, 3}) {
        SCOPED_TRACE(level);
        SleepLifecycleController controller(true);
        controller.setConfiguredLevel(level);
        bool                     transfers_frozen = false;
        std::vector<std::string> calls;
        SleepHooks               hooks;
        hooks.freezeExternalTransfers = [&transfers_frozen, &calls, level](const SleepOptions& opt) {
            EXPECT_EQ(opt.level, level);
            transfers_frozen = true;
            calls.push_back("freeze");
            return true;
        };
        hooks.resumeExternalTransfers = [&transfers_frozen, &calls]() {
            EXPECT_TRUE(transfers_frozen);
            transfers_frozen = false;
            calls.push_back("resume");
            return false;
        };
        controller.setHooks(hooks);

        auto opt  = gracefulOptions();
        opt.level = level;
        ASSERT_TRUE(controller.sleep(opt).ok);
        calls.clear();

        EXPECT_FALSE(controller.wakeUp().ok);
        EXPECT_TRUE(transfers_frozen);
        EXPECT_EQ(controller.state(), SleepState::ERROR);
        EXPECT_EQ(calls, (std::vector<std::string>{"resume", "freeze"}));
    }
}

TEST(SleepLifecycleControllerTest, WakeUpFromSleepingReachesRunning) {
    SleepLifecycleController controller(true);
    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);

    const auto result = controller.wakeUp();
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_TRUE(controller.admit());
    // epoch is bumped by sleep, not by wake_up.
    EXPECT_EQ(controller.sleepEpoch(), 1);
    EXPECT_EQ(controller.status().kv_memory_state, "ACTIVE");
    EXPECT_TRUE(controller.status().device_kv_cache_valid);
}

TEST(SleepLifecycleControllerTest, SleepIsIdempotent) {
    SleepLifecycleController controller(true);
    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), SleepState::SLEEPING);

    const auto again = controller.sleep(gracefulOptions());
    EXPECT_TRUE(again.ok) << again.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    // Idempotent repeat must NOT bump the epoch.
    EXPECT_EQ(controller.sleepEpoch(), 1);
}

TEST(SleepLifecycleControllerTest, WakeUpIsIdempotent) {
    SleepLifecycleController controller(true);
    EXPECT_TRUE(controller.wakeUp().ok);  // RUNNING -> wake_up == no-op success
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
}

TEST(SleepLifecycleControllerTest, EpochIsMonotonicAcrossCycles) {
    SleepLifecycleController controller(true);
    for (int64_t i = 1; i <= 3; ++i) {
        ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
        EXPECT_EQ(controller.sleepEpoch(), i);
        ASSERT_TRUE(controller.wakeUp().ok);
        EXPECT_EQ(controller.sleepEpoch(), i);
    }
}

TEST(SleepLifecycleControllerTest, DrainTimeoutKeepsDraining) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&, const DrainCancellationPredicate&) { return false; };  // simulate timeout
    controller.setHooks(hooks);

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_FALSE(result.ok);
    // Per design: graceful drain timeout keeps DRAINING, does not release GPU.
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_TRUE(controller.status().device_kv_cache_valid);
}

TEST(SleepLifecycleControllerTest, WakeUpCancelsInflightDrainWithoutWaitingForTimeout) {
    SleepLifecycleController controller(true);
    std::promise<void>       drain_started;
    std::atomic<int>         quiesce_called{0};
    std::atomic<int>         release_called{0};
    SleepHooks               hooks;
    hooks.drain = [&drain_started](const SleepOptions&, const DrainCancellationPredicate& cancelled) {
        drain_started.set_value();
        while (!cancelled()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    };
    hooks.quiesceEngine = [&quiesce_called](const SleepOptions&) {
        ++quiesce_called;
        return true;
    };
    hooks.releaseKvMemoryBacking = [&release_called](const SleepOptions&) {
        ++release_called;
        return true;
    };
    controller.setHooks(hooks);

    auto drain_ready = drain_started.get_future();
    auto sleeper     = std::async(std::launch::async, [&controller]() {
        auto opt       = gracefulOptions();
        opt.timeout_ms = 60000;
        return controller.sleep(opt);
    });
    ASSERT_EQ(drain_ready.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    const auto start       = std::chrono::steady_clock::now();
    const auto wake_result = controller.wakeUp();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    EXPECT_TRUE(wake_result.ok) << wake_result.message;
    EXPECT_LT(elapsed_ms, 500);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_FALSE(sleeper.get().ok);
    EXPECT_EQ(quiesce_called.load(), 0);
    EXPECT_EQ(release_called.load(), 0);
}

TEST(SleepLifecycleControllerTest, WakeUpJoinsInflightAbortHookBeforeReturningRunning) {
    SleepLifecycleController controller(true);
    std::promise<void>       abort_hook_started;
    std::atomic<bool>        allow_hook_exit{false};
    std::atomic<bool>        hook_exited{false};
    std::atomic<int>         late_abort_side_effects{0};
    SleepHooks               hooks;
    hooks.drain = [&](const SleepOptions& opt, const DrainCancellationPredicate& cancelled) {
        EXPECT_EQ(opt.mode, "abort");
        EXPECT_FALSE(cancelled());
        abort_hook_started.set_value();
        while (!allow_hook_exit.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Simulates an abort hook that passed its predicate check immediately
        // before wake invalidated the token, then completed a cancellation side effect.
        ++late_abort_side_effects;
        hook_exited.store(true, std::memory_order_release);
        return false;
    };
    controller.setHooks(hooks);

    auto hook_ready = abort_hook_started.get_future();
    auto sleeper    = std::async(std::launch::async, [&controller]() {
        SleepOptions abort;
        abort.mode       = "abort";
        abort.timeout_ms = 60000;
        return controller.sleep(abort);
    });
    ASSERT_EQ(hook_ready.wait_for(std::chrono::seconds(1)), std::future_status::ready);

    auto waker = std::async(std::launch::async, [&controller]() { return controller.wakeUp(); });
    EXPECT_EQ(waker.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_FALSE(hook_exited.load(std::memory_order_acquire));

    allow_hook_exit.store(true, std::memory_order_release);
    const auto wake_result = waker.get();
    EXPECT_TRUE(wake_result.ok) << wake_result.message;
    EXPECT_TRUE(hook_exited.load(std::memory_order_acquire));
    EXPECT_EQ(late_abort_side_effects.load(), 1);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_FALSE(sleeper.get().ok);
}

TEST(SleepLifecycleControllerTest, AbortRetryCancelsOlderDrainAndOwnsRelease) {
    SleepLifecycleController controller(true);
    std::promise<void>       wait_drain_started;
    std::atomic<int>         abort_seen{0};
    std::atomic<int>         quiesce_called{0};
    std::atomic<int>         release_called{0};
    SleepHooks               hooks;
    hooks.drain = [&wait_drain_started, &abort_seen](const SleepOptions&               opt,
                                                     const DrainCancellationPredicate& cancelled) {
        if (opt.mode == "abort") {
            ++abort_seen;
            return !cancelled();
        }
        wait_drain_started.set_value();
        while (!cancelled()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    };
    hooks.quiesceEngine = [&quiesce_called](const SleepOptions&) {
        ++quiesce_called;
        return true;
    };
    hooks.releaseKvMemoryBacking = [&release_called](const SleepOptions&) {
        ++release_called;
        return true;
    };
    controller.setHooks(hooks);

    auto wait_drain_ready = wait_drain_started.get_future();
    auto old_sleeper      = std::async(std::launch::async, [&controller]() {
        auto opt       = gracefulOptions();
        opt.timeout_ms = 60000;
        return controller.sleep(opt);
    });
    ASSERT_EQ(wait_drain_ready.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    SleepOptions abort;
    abort.mode        = "abort";
    abort.timeout_ms  = 1000;
    const auto result = controller.sleep(abort);

    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_FALSE(old_sleeper.get().ok);
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    EXPECT_EQ(controller.sleepEpoch(), 1);
    EXPECT_EQ(abort_seen.load(), 1);
    EXPECT_EQ(quiesce_called.load(), 1);
    EXPECT_EQ(release_called.load(), 1);
}

TEST(SleepLifecycleControllerTest, LeaseAcquiredBeforeDrainMustReleaseBeforeSleepProgresses) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [&controller](const SleepOptions&, const DrainCancellationPredicate&) {
        return controller.activeAdmissionCount() == 0;
    };
    controller.setHooks(hooks);

    SleepResult first_sleep;
    {
        auto admission = controller.acquireAdmission();
        ASSERT_TRUE(admission.admitted());
        EXPECT_EQ(controller.activeAdmissionCount(), 1);

        first_sleep = controller.sleep(gracefulOptions());
        EXPECT_FALSE(first_sleep.ok);
        EXPECT_EQ(controller.state(), SleepState::DRAINING);
    }

    EXPECT_EQ(controller.activeAdmissionCount(), 0);
    const auto retry = controller.sleep(gracefulOptions());
    EXPECT_TRUE(retry.ok) << retry.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
}

TEST(SleepLifecycleControllerTest, SleepRetryFromDrainingCanComplete) {
    SleepLifecycleController controller(true);
    std::atomic<bool>        busy{true};
    SleepHooks               hooks;
    hooks.drain = [&busy](const SleepOptions&, const DrainCancellationPredicate&) { return !busy.load(); };
    controller.setHooks(hooks);

    EXPECT_FALSE(controller.sleep(gracefulOptions()).ok);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_EQ(controller.sleepEpoch(), 1);

    busy             = false;
    const auto retry = controller.sleep(gracefulOptions());
    EXPECT_TRUE(retry.ok) << retry.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    EXPECT_EQ(controller.sleepEpoch(), 1);
}

TEST(SleepLifecycleControllerTest, PrepareOnlyStaysDrainingUntilCommit) {
    SleepLifecycleController controller(true);
    std::atomic<int>         release_kv_called{0};
    std::atomic<int>         quiesce_called{0};
    std::atomic<int>         sync_dereg_called{0};
    SleepHooks               hooks;
    hooks.quiesceEngine = [&quiesce_called](const SleepOptions&) {
        quiesce_called++;
        return true;
    };
    hooks.synchronizeAndDeregisterMr = [&sync_dereg_called](const SleepOptions&) {
        sync_dereg_called++;
        return true;
    };
    hooks.releaseKvMemoryBacking = [&release_kv_called](const SleepOptions&) {
        release_kv_called++;
        return true;
    };
    controller.setHooks(hooks);

    SleepOptions prepare = gracefulOptions();
    prepare.prepare_only = true;
    const auto prepared  = controller.sleep(prepare);
    EXPECT_TRUE(prepared.ok) << prepared.message;
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_FALSE(controller.admit());
    EXPECT_EQ(controller.sleepEpoch(), 1);
    EXPECT_TRUE(controller.status().device_kv_cache_valid);
    EXPECT_EQ(quiesce_called.load(), 0);
    EXPECT_EQ(sync_dereg_called.load(), 0);
    EXPECT_EQ(release_kv_called.load(), 0);

    SleepOptions commit  = gracefulOptions();
    commit.commit_only   = true;
    const auto committed = controller.sleep(commit);
    EXPECT_TRUE(committed.ok) << committed.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    EXPECT_EQ(quiesce_called.load(), 1);
    EXPECT_EQ(sync_dereg_called.load(), 1);
    EXPECT_EQ(release_kv_called.load(), 1);
}

TEST(SleepLifecycleControllerTest, PrepareAndCommitCannotAcquireStragglerAdmission) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [&controller](const SleepOptions&, const DrainCancellationPredicate&) {
        return controller.activeAdmissionCount() == 0;
    };
    controller.setHooks(hooks);

    SleepOptions prepare = gracefulOptions();
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(prepare).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    auto straggler = controller.acquireAdmission();
    EXPECT_FALSE(straggler.admitted());
    EXPECT_EQ(straggler.state, SleepState::DRAINING);
    EXPECT_EQ(controller.activeAdmissionCount(), 0);

    SleepOptions commit = gracefulOptions();
    commit.commit_only  = true;
    const auto result   = controller.sleep(commit);
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
}

TEST(SleepLifecycleControllerTest, CommitOnlyRequiresSuccessfulPrepareDrain) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&, const DrainCancellationPredicate&) { return false; };
    controller.setHooks(hooks);

    ASSERT_FALSE(controller.sleep(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    SleepOptions commit = gracefulOptions();
    commit.commit_only  = true;
    const auto result   = controller.sleep(commit);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_NE(controller.status().last_error.find("admitted work was not drained"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, WakeUpFromPreparedDrainingDoesNotRestartRunningEngine) {
    SleepLifecycleController controller(true);
    std::atomic<int>         cancel_called{0};
    SleepHooks               hooks;
    hooks.cancelQuiesceAndRestartEngine = [&cancel_called]() {
        cancel_called++;
        return true;
    };
    controller.setHooks(hooks);

    SleepOptions prepare = gracefulOptions();
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(prepare).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    const auto result = controller.wakeUp();
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_TRUE(controller.admit());
    EXPECT_EQ(controller.sleepEpoch(), 1);
    EXPECT_EQ(cancel_called.load(), 0);
}

TEST(SleepLifecycleControllerTest, LevelThreePrepareRollbackLeavesEngineAndTransferGateUntouched) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    int        drain_called   = 0;
    int        freeze_called  = 0;
    int        arm_called     = 0;
    int        quiesce_called = 0;
    int        restart_called = 0;
    int        resume_called  = 0;
    SleepHooks hooks;
    hooks.drain = [&drain_called](const SleepOptions&, const DrainCancellationPredicate&) {
        ++drain_called;
        return true;
    };
    hooks.freezeExternalTransfers = [&freeze_called](const SleepOptions&) {
        ++freeze_called;
        return true;
    };
    hooks.armEngineQuiesce = [&arm_called](const SleepOptions&) {
        ++arm_called;
        return true;
    };
    hooks.quiesceEngine = [&quiesce_called](const SleepOptions&) {
        ++quiesce_called;
        return true;
    };
    hooks.cancelQuiesceAndRestartEngine = [&restart_called]() {
        ++restart_called;
        return true;
    };
    hooks.resumeExternalTransfers = [&resume_called]() {
        ++resume_called;
        return true;
    };
    controller.setHooks(hooks);

    SleepOptions prepare = gracefulOptions();
    prepare.level        = 3;
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(prepare).ok);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_EQ(drain_called, 1);
    EXPECT_EQ(freeze_called, 0);
    EXPECT_EQ(arm_called, 0);
    EXPECT_EQ(quiesce_called, 0);

    const auto result = controller.wakeUp();
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_EQ(restart_called, 0);
    EXPECT_EQ(resume_called, 0);
}

TEST(SleepLifecycleControllerTest, CommitFailureRollbackRestartsBeforeResumingTransfers) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    bool                     transfers_frozen = false;
    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&, const DrainCancellationPredicate&) { return true; };
    hooks.freezeExternalTransfers = [&transfers_frozen, &calls](const SleepOptions&) {
        transfers_frozen = true;
        calls.push_back("freeze");
        return true;
    };
    hooks.armEngineQuiesce = [&calls](const SleepOptions&) {
        calls.push_back("arm");
        return true;
    };
    hooks.quiesceEngine = [&calls](const SleepOptions&) {
        calls.push_back("quiesce");
        return false;
    };
    hooks.cancelQuiesceAndRestartEngine = [&transfers_frozen, &calls]() {
        EXPECT_TRUE(transfers_frozen);
        calls.push_back("restart");
        return true;
    };
    hooks.resumeExternalTransfers = [&transfers_frozen, &calls]() {
        EXPECT_TRUE(transfers_frozen);
        transfers_frozen = false;
        calls.push_back("resume");
        return true;
    };
    controller.setHooks(hooks);

    SleepOptions prepare = gracefulOptions();
    prepare.level        = 3;
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(prepare).ok);

    SleepOptions commit = gracefulOptions();
    commit.level        = 3;
    commit.commit_only  = true;
    ASSERT_FALSE(controller.sleep(commit).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    const auto result = controller.wakeUp();
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_FALSE(transfers_frozen);
    EXPECT_EQ(calls, (std::vector<std::string>{"freeze", "arm", "quiesce", "restart", "resume"}));
}

TEST(SleepLifecycleControllerTest, CommitFreezeFailureRollbackResumesGateWithoutRestart) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    bool                     transfers_frozen = false;
    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&, const DrainCancellationPredicate&) { return true; };
    hooks.freezeExternalTransfers = [&transfers_frozen, &calls](const SleepOptions&) {
        transfers_frozen = true;
        calls.push_back("freeze");
        return false;
    };
    hooks.cancelQuiesceAndRestartEngine = [&calls]() {
        calls.push_back("restart");
        return true;
    };
    hooks.resumeExternalTransfers = [&transfers_frozen, &calls]() {
        EXPECT_TRUE(transfers_frozen);
        transfers_frozen = false;
        calls.push_back("resume");
        return true;
    };
    controller.setHooks(hooks);

    SleepOptions prepare = gracefulOptions();
    prepare.level        = 3;
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(prepare).ok);

    SleepOptions commit = gracefulOptions();
    commit.level        = 3;
    commit.commit_only  = true;
    ASSERT_FALSE(controller.sleep(commit).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    const auto result = controller.wakeUp();
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_FALSE(transfers_frozen);
    EXPECT_EQ(calls, (std::vector<std::string>{"freeze", "resume"}));
}

TEST(SleepLifecycleControllerTest, CommitRollbackResumeFailureRefreezesTransfers) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    bool                     transfers_frozen = false;
    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&, const DrainCancellationPredicate&) { return true; };
    hooks.freezeExternalTransfers = [&transfers_frozen, &calls](const SleepOptions&) {
        transfers_frozen = true;
        calls.push_back("freeze");
        return true;
    };
    hooks.armEngineQuiesce = [](const SleepOptions&) { return true; };
    hooks.quiesceEngine    = [](const SleepOptions&) { return false; };
    hooks.cancelQuiesceAndRestartEngine = [&calls]() {
        calls.push_back("restart");
        return true;
    };
    hooks.resumeExternalTransfers = [&transfers_frozen, &calls]() {
        transfers_frozen = false;
        calls.push_back("resume");
        return false;
    };
    controller.setHooks(hooks);

    SleepOptions prepare = gracefulOptions();
    prepare.level        = 3;
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(prepare).ok);

    SleepOptions commit = gracefulOptions();
    commit.level        = 3;
    commit.commit_only  = true;
    ASSERT_FALSE(controller.sleep(commit).ok);
    calls.clear();

    const auto result = controller.wakeUp();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_TRUE(transfers_frozen);
    EXPECT_EQ(calls, (std::vector<std::string>{"restart", "resume", "freeze"}));
    EXPECT_NE(controller.status().last_error.find("resumeExternalTransfers"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, WakeUpPrepareOnlyStaysWakingUpUntilCommit) {
    SleepLifecycleController controller(true);
    std::atomic<int>         restore_kv_called{0};
    std::atomic<int>         restore_weights_called{0};
    std::atomic<int>         register_mr_called{0};
    std::atomic<int>         restart_called{0};
    SleepHooks               hooks;
    hooks.restoreKvMemoryBackingAndResetMetadata = [&restore_kv_called]() {
        restore_kv_called++;
        return true;
    };
    hooks.restoreRestorableGpuMemory = [&restore_weights_called]() {
        restore_weights_called++;
        return true;
    };
    hooks.registerMr = [&register_mr_called]() {
        register_mr_called++;
        return true;
    };
    hooks.restartEngine = [&restart_called]() {
        restart_called++;
        return true;
    };
    controller.setHooks(hooks);

    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), SleepState::SLEEPING);

    WakeUpOptions prepare;
    prepare.prepare_only = true;
    const auto prepared  = controller.wakeUp(prepare);
    EXPECT_TRUE(prepared.ok) << prepared.message;
    EXPECT_EQ(controller.state(), SleepState::WAKING_UP);
    EXPECT_FALSE(controller.admit());
    EXPECT_EQ(restore_kv_called.load(), 1);
    EXPECT_EQ(restore_weights_called.load(), 1);
    EXPECT_EQ(register_mr_called.load(), 1);
    EXPECT_EQ(restart_called.load(), 0);

    WakeUpOptions commit;
    commit.commit_only   = true;
    const auto committed = controller.wakeUp(commit);
    EXPECT_TRUE(committed.ok) << committed.message;
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_TRUE(controller.admit());
    EXPECT_EQ(restart_called.load(), 1);
    EXPECT_TRUE(controller.status().device_kv_cache_valid);
}

TEST(SleepLifecycleControllerTest, ControlPlaneSmokeFlowExposesExpectedIntermediateStates) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.quiesceEngine                          = [](const SleepOptions&) { return true; };
    hooks.synchronizeAndDeregisterMr             = [](const SleepOptions&) { return true; };
    hooks.releaseKvMemoryBacking                 = [](const SleepOptions&) { return true; };
    hooks.releaseRestorableGpuMemory             = [](const SleepOptions&) { return true; };
    hooks.restoreKvMemoryBackingAndResetMetadata = []() { return true; };
    hooks.restoreRestorableGpuMemory             = []() { return true; };
    hooks.registerMr                             = []() { return true; };
    hooks.restartEngine                          = []() { return true; };
    hooks.warmupAndHealthCheck                   = []() { return true; };
    controller.setHooks(hooks);

    auto status = controller.status();
    EXPECT_EQ(status.state, SleepState::RUNNING);
    EXPECT_TRUE(controller.admit());
    EXPECT_EQ(status.gpu_resource_state, "ACTIVE");

    SleepOptions sleep_prepare = gracefulOptions();
    sleep_prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(sleep_prepare).ok);
    status = controller.status();
    EXPECT_EQ(status.state, SleepState::DRAINING);
    EXPECT_FALSE(controller.admit());
    EXPECT_EQ(status.gpu_resource_state, "ACTIVE");
    EXPECT_TRUE(status.device_kv_cache_valid);

    SleepOptions sleep_commit = gracefulOptions();
    sleep_commit.commit_only  = true;
    ASSERT_TRUE(controller.sleep(sleep_commit).ok);
    status = controller.status();
    EXPECT_EQ(status.state, SleepState::SLEEPING);
    EXPECT_EQ(status.gpu_resource_state, "RELEASED");
    EXPECT_EQ(status.kv_memory_state, "PAUSED");
    EXPECT_FALSE(status.device_kv_cache_valid);

    WakeUpOptions wake_prepare;
    wake_prepare.prepare_only = true;
    ASSERT_TRUE(controller.wakeUp(wake_prepare).ok);
    status = controller.status();
    EXPECT_EQ(status.state, SleepState::WAKING_UP);
    EXPECT_EQ(status.gpu_resource_state, "RESTORING");
    EXPECT_FALSE(controller.admit());

    WakeUpOptions wake_commit;
    wake_commit.commit_only = true;
    ASSERT_TRUE(controller.wakeUp(wake_commit).ok);
    status = controller.status();
    EXPECT_EQ(status.state, SleepState::RUNNING);
    EXPECT_EQ(status.gpu_resource_state, "ACTIVE");
    EXPECT_EQ(status.kv_memory_state, "ACTIVE");
    EXPECT_TRUE(status.device_kv_cache_valid);
    EXPECT_TRUE(controller.admit());
}

TEST(SleepLifecycleControllerTest, WakeUpPrepareFailureDoesNotRestartEngine) {
    SleepLifecycleController controller(true);
    std::atomic<int>         restart_called{0};
    SleepHooks               hooks;
    hooks.restoreRestorableGpuMemory = []() { return false; };
    hooks.restartEngine              = [&restart_called]() {
        restart_called++;
        return true;
    };
    controller.setHooks(hooks);

    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);

    WakeUpOptions prepare;
    prepare.prepare_only = true;
    const auto result    = controller.wakeUp(prepare);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_EQ(restart_called.load(), 0);
}

TEST(SleepLifecycleControllerTest, SleepRetryFromDrainingCanEscalateToAbort) {
    SleepLifecycleController controller(true);
    std::atomic<int>         abort_seen{0};
    SleepHooks               hooks;
    hooks.drain = [&abort_seen](const SleepOptions& opt, const DrainCancellationPredicate&) {
        if (opt.mode == "abort") {
            abort_seen++;
            return true;
        }
        return false;
    };
    controller.setHooks(hooks);

    EXPECT_FALSE(controller.sleep(gracefulOptions()).ok);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);

    SleepOptions abort;
    abort.mode       = "abort";
    abort.timeout_ms = 1000;
    const auto retry = controller.sleep(abort);
    EXPECT_TRUE(retry.ok) << retry.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    EXPECT_EQ(controller.sleepEpoch(), 1);
    EXPECT_EQ(abort_seen.load(), 1);
}

TEST(SleepLifecycleControllerTest, SleepHookFailureGoesToError) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.releaseKvMemoryBacking = [](const SleepOptions&) { return false; };
    controller.setHooks(hooks);

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admit());
    EXPECT_FALSE(controller.status().last_error.empty());
}

TEST(SleepLifecycleControllerTest, SleepHalfReleasedFailureGoesToError) {
    SleepLifecycleController controller(true);
    std::atomic<int>         release_kv_called{0};
    SleepHooks               hooks;
    hooks.releaseKvMemoryBacking = [&release_kv_called](const SleepOptions&) {
        release_kv_called++;
        return true;
    };
    hooks.releaseRestorableGpuMemory = [](const SleepOptions&) { return false; };
    controller.setHooks(hooks);

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admit());
    EXPECT_EQ(release_kv_called.load(), 1);
    EXPECT_EQ(controller.status().kv_memory_state, "PAUSED");
    EXPECT_FALSE(controller.status().device_kv_cache_valid);
    EXPECT_EQ(controller.status().gpu_resource_state, "UNKNOWN");
}

TEST(SleepLifecycleControllerTest, WakeUpFailureGoesToError) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.warmupAndHealthCheck = []() { return false; };
    controller.setHooks(hooks);

    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
    const auto result = controller.wakeUp();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admit());
}

TEST(SleepLifecycleControllerTest, WakeUpFailureDoesNotRunImplicitRollback) {
    SleepLifecycleController controller(true);
    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);

    std::atomic<int> release_kv_called{0};
    SleepHooks       hooks;
    hooks.restoreKvMemoryBackingAndResetMetadata = []() { return true; };
    hooks.restoreRestorableGpuMemory             = []() { return false; };
    hooks.releaseKvMemoryBacking                 = [&release_kv_called](const SleepOptions&) {
        release_kv_called++;
        return true;
    };
    controller.setHooks(hooks);

    const auto result = controller.wakeUp();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_EQ(release_kv_called.load(), 0);
}

TEST(SleepLifecycleControllerTest, WakeUpHookExceptionGoesToError) {
    SleepLifecycleController controller(true);
    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);

    SleepHooks hooks;
    hooks.restoreKvMemoryBackingAndResetMetadata = []() -> bool { throw std::runtime_error("boom"); };
    controller.setHooks(hooks);

    const auto result = controller.wakeUp();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admit());
}

TEST(SleepLifecycleControllerTest, ErrorIsTerminalAndRejectsWakeUp) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.releaseKvMemoryBacking = [](const SleepOptions&) { return false; };
    controller.setHooks(hooks);
    ASSERT_FALSE(controller.sleep(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), SleepState::ERROR);

    controller.setHooks(SleepHooks{});
    const auto result = controller.wakeUp();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.code, SleepResult::Code::FAILED_PRECONDITION);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admit());
}

TEST(SleepLifecycleControllerTest, WakeUpWhileDrainingAbortsSleep) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&, const DrainCancellationPredicate&) { return false; };
    controller.setHooks(hooks);
    ASSERT_FALSE(controller.sleep(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    const auto result = controller.wakeUp();
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_TRUE(controller.admit());
}

TEST(SleepLifecycleControllerTest, StatusExposesLiveCounters) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.activeRequestCount       = []() { return 7; };
    hooks.activeCacheTransferCount = []() { return 3; };
    controller.setHooks(hooks);

    const auto status = controller.status();
    EXPECT_EQ(status.active_request_count, 7);
    EXPECT_EQ(status.active_cache_transfer_count, 3);
}

TEST(SleepLifecycleControllerTest, ConcurrentSleepWakeUpIsSerializedAndConsistent) {
    SleepLifecycleController controller(true);
    std::atomic<int>         ok_sleeps{0};

    std::vector<std::thread> threads;
    threads.reserve(8);
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&controller, &ok_sleeps]() {
            if (controller.sleep(gracefulOptions()).ok) {
                ok_sleeps.fetch_add(1);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    // All callers either performed or idempotently observed the sleep.
    EXPECT_EQ(ok_sleeps.load(), 8);
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    // Exactly one real sleep happened.
    EXPECT_EQ(controller.sleepEpoch(), 1);
}

}  // namespace rtp_llm
