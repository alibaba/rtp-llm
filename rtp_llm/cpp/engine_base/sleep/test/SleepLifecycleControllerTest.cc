#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"

namespace rtp_llm {

namespace {

SleepOptions gracefulOptions() {
    SleepOptions opt;
    opt.mode       = "wait";
    opt.timeout_ms = 1000;
    return opt;
}

}  // namespace

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
    hooks.drain = [](const SleepOptions&) { return false; };  // simulate timeout
    controller.setHooks(hooks);

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_FALSE(result.ok);
    // Per design: graceful drain timeout keeps DRAINING, does not release GPU.
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_TRUE(controller.status().device_kv_cache_valid);
}

TEST(SleepLifecycleControllerTest, LeaseAcquiredBeforeDrainMustReleaseBeforeSleepProgresses) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [&controller](const SleepOptions&) { return controller.activeAdmissionCount() == 0; };
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
    hooks.drain = [&busy](const SleepOptions&) { return !busy.load(); };
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
    EXPECT_EQ(quiesce_called.load(), 1);
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
    hooks.drain = [&controller](const SleepOptions&) { return controller.activeAdmissionCount() == 0; };
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

TEST(SleepLifecycleControllerTest, CommitOnlyRequiresPreparedQuiesce) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&) { return false; };
    controller.setHooks(hooks);

    ASSERT_FALSE(controller.sleep(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    SleepOptions commit = gracefulOptions();
    commit.commit_only  = true;
    const auto result   = controller.sleep(commit);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_NE(controller.status().last_error.find("engine is not quiesced"), std::string::npos);
}

TEST(SleepLifecycleControllerTest, WakeUpFromPreparedDrainingAbortsSleep) {
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
    EXPECT_EQ(cancel_called.load(), 1);
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
    hooks.drain = [&abort_seen](const SleepOptions& opt) {
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
    hooks.drain = [](const SleepOptions&) { return false; };
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

// The collective sleep-quiesce consensus MUST be armed BEFORE the (rank-asymmetric)
// drain, not after it: only the rank holding an in-flight request blocks in drain, so
// arming after drain would leave the busy rank unarmed while idle peers issue unmatched
// consensus rounds. This locks the ordering the multi-rank drain fix depends on.
TEST(SleepLifecycleControllerTest, ArmEngineQuiesceRunsBeforeDrain) {
    SleepLifecycleController controller(true);
    std::mutex               order_mu;
    std::vector<std::string> order;
    SleepHooks               hooks;
    hooks.armEngineQuiesce = [&](const SleepOptions&) {
        std::lock_guard<std::mutex> l(order_mu);
        order.push_back("arm");
        return true;
    };
    hooks.drain = [&](const SleepOptions&) {
        std::lock_guard<std::mutex> l(order_mu);
        order.push_back("drain");
        return true;
    };
    controller.setHooks(hooks);

    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
    ASSERT_EQ(order.size(), 2u);
    EXPECT_EQ(order[0], "arm");
    EXPECT_EQ(order[1], "drain");
}

// commit_only is the second phase of the two-phase sleep; the consensus was already
// armed during prepare. Re-arming on commit would double-issue and is gated out.
TEST(SleepLifecycleControllerTest, ArmEngineQuiesceNotCalledOnCommitOnly) {
    SleepLifecycleController controller(true);
    std::atomic<int>         arm_called{0};
    SleepHooks               hooks;
    hooks.armEngineQuiesce = [&arm_called](const SleepOptions&) {
        arm_called++;
        return true;
    };
    controller.setHooks(hooks);

    SleepOptions prepare = gracefulOptions();
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(prepare).ok);
    EXPECT_EQ(arm_called.load(), 1);  // armed during prepare

    SleepOptions commit = gracefulOptions();
    commit.commit_only  = true;
    ASSERT_TRUE(controller.sleep(commit).ok);
    EXPECT_EQ(arm_called.load(), 1);  // commit_only must NOT re-arm
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
}

// The restore hooks run at most once per wake cycle: a prepare_only wake applies them and
// stays in WAKING_UP, so a following FLAGLESS wake (neither prepare_only nor commit_only)
// must finish to RUNNING WITHOUT re-running the restores -- a second VMM resume / level-2
// reload could fail into ERROR. This is the prepare->flagless path the commit_only-based
// test cannot cover.
TEST(SleepLifecycleControllerTest, WakeUpPrepareThenFlaglessDoesNotReRestore) {
    SleepLifecycleController controller(true);
    std::atomic<int>         restore_kv{0};
    std::atomic<int>         restore_weights{0};
    std::atomic<int>         register_mr{0};
    std::atomic<int>         restart_called{0};
    SleepHooks               hooks;
    hooks.restoreKvMemoryBackingAndResetMetadata = [&restore_kv]() {
        restore_kv++;
        return true;
    };
    hooks.restoreRestorableGpuMemory = [&restore_weights]() {
        restore_weights++;
        return true;
    };
    hooks.registerMr = [&register_mr]() {
        register_mr++;
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
    ASSERT_TRUE(controller.wakeUp(prepare).ok);
    EXPECT_EQ(controller.state(), SleepState::WAKING_UP);
    EXPECT_EQ(restore_kv.load(), 1);
    EXPECT_EQ(restore_weights.load(), 1);
    EXPECT_EQ(register_mr.load(), 1);
    EXPECT_EQ(restart_called.load(), 0);

    // Flagless wake: completes the cycle without re-restoring.
    const auto flagless = controller.wakeUp();
    EXPECT_TRUE(flagless.ok) << flagless.message;
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_EQ(restore_kv.load(), 1);       // NOT re-run
    EXPECT_EQ(restore_weights.load(), 1);  // NOT re-run
    EXPECT_EQ(register_mr.load(), 1);      // NOT re-run
    EXPECT_EQ(restart_called.load(), 1);   // final restart ran once
}

// The latch is per wake cycle: a fresh SLEEPING->WAKING_UP transition clears it, so a
// new sleep/wake cycle re-runs the restores (they were released by the intervening sleep).
TEST(SleepLifecycleControllerTest, WakeUpRestoreRerunsOnFreshCycle) {
    SleepLifecycleController controller(true);
    std::atomic<int>         restore_weights{0};
    SleepHooks               hooks;
    hooks.restoreRestorableGpuMemory = [&restore_weights]() {
        restore_weights++;
        return true;
    };
    controller.setHooks(hooks);

    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
    ASSERT_TRUE(controller.wakeUp().ok);  // cycle 1: restore #1
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_EQ(restore_weights.load(), 1);

    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
    ASSERT_TRUE(controller.wakeUp().ok);  // cycle 2: fresh latch -> restore #2
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_EQ(restore_weights.load(), 2);
}

}  // namespace rtp_llm
