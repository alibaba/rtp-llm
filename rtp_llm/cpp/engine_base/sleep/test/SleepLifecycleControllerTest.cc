#include "rtp_llm/cpp/engine_base/sleep/test/BoundSleepLifecycleController.h"
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"

namespace rtp_llm {

TEST(SleepLifecycleMetricsTest, CommitPausesAndExplicitWakeResumeRestoresReporting) {
    BoundSleepLifecycleController controller(true);
    std::vector<bool>        changes;
    SleepHooks               hooks;
    hooks.setMetricsReportingEnabled = [&](bool enabled) {
        changes.push_back(enabled);
        return true;
    };
    controller.setHooks(hooks);
    SleepOptions prepare;
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(prepare).ok);
    EXPECT_TRUE(changes.empty());
    SleepOptions commit;
    commit.commit_only = true;
    ASSERT_TRUE(controller.sleep(commit).ok);
    EXPECT_EQ(changes, std::vector<bool>({false}));
    ASSERT_TRUE(controller.sleep(commit).ok);
    EXPECT_EQ(changes.size(), 1);
    WakeUpOptions wake_prepare;
    wake_prepare.prepare_only = true;
    ASSERT_TRUE(controller.wakeUp(wake_prepare).ok);
    EXPECT_EQ(changes.size(), 1);
    WakeUpOptions wake_commit;
    wake_commit.commit_only = true;
    ASSERT_TRUE(controller.wakeUp(wake_commit).ok);
    EXPECT_EQ(changes, std::vector<bool>({false}));
    WakeUpOptions resume;
    resume.resume_metrics_only  = true;
    resume.expected_incarnation = controller.status().worker_incarnation;
    resume.expected_sleep_epoch = controller.sleepEpoch();
    ASSERT_TRUE(controller.wakeUp(resume).ok);
    EXPECT_EQ(changes, std::vector<bool>({false, true}));
    ASSERT_TRUE(controller.wakeUp().ok);
    EXPECT_EQ(changes.size(), 2);
}

TEST(SleepLifecycleMetricsTest, RejectsStaleOrPrematureResume) {
    BoundSleepLifecycleController controller(true);
    std::vector<bool>        changes;
    SleepHooks               hooks;
    hooks.setMetricsReportingEnabled = [&](bool enabled) {
        changes.push_back(enabled);
        return true;
    };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    WakeUpOptions resume;
    resume.resume_metrics_only  = true;
    resume.expected_incarnation = controller.status().worker_incarnation;
    resume.expected_sleep_epoch = controller.sleepEpoch();
    EXPECT_FALSE(controller.wakeUp(resume).ok);
    WakeUpOptions prepare, commit;
    prepare.prepare_only = true;
    commit.commit_only   = true;
    ASSERT_TRUE(controller.wakeUp(prepare).ok);
    ASSERT_TRUE(controller.wakeUp(commit).ok);
    auto wrong                 = resume;
    wrong.expected_incarnation = "replaced-worker";
    EXPECT_FALSE(controller.wakeUp(wrong).ok);
    wrong             = resume;
    wrong.commit_only = true;
    EXPECT_FALSE(controller.wakeUp(wrong).ok);
    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    ASSERT_TRUE(controller.wakeUp(prepare).ok);
    ASSERT_TRUE(controller.wakeUp(commit).ok);
    EXPECT_FALSE(controller.wakeUp(resume).ok);
    EXPECT_EQ(changes, std::vector<bool>({false, false}));
    resume.expected_sleep_epoch = controller.sleepEpoch();
    EXPECT_TRUE(controller.wakeUp(resume).ok);
}

TEST(SleepLifecycleMetricsTest, MetricsResumeFailureKeepsEngineRunningAndCanBeRetried) {
    for (bool throws : {false, true}) {
        BoundSleepLifecycleController controller(true);
        bool                     fail     = true;
        int                      restarts = 0;
        SleepHooks               hooks;
        hooks.restartEngine = [&] {
            ++restarts;
            return true;
        };
        hooks.setMetricsReportingEnabled = [&](bool enabled) {
            if (enabled && fail && throws) {
                throw std::runtime_error("monitor unavailable");
            }
            return !enabled || !fail;
        };
        controller.setHooks(hooks);
        ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
        EXPECT_FALSE(controller.wakeUp().ok);
        EXPECT_EQ(controller.state(), SleepState::RUNNING);
        EXPECT_EQ(controller.status().kv_memory_state, "ACTIVE");
        fail = false;
        EXPECT_TRUE(controller.wakeUp().ok);
        EXPECT_EQ(restarts, 1);
    }
}

TEST(SleepLifecycleMetricsTest, PartialPauseFailureRollsBackBeforeResourceReleaseAndCanRetry) {
    for (bool throws : {false, true}) {
        BoundSleepLifecycleController controller(true);
        bool                          cpp_enabled = true, python_enabled = true, fail_pause = true;
        std::vector<std::string>      releases;
        SleepHooks                    hooks;
        hooks.setMetricsReportingEnabled = [&](bool enabled) {
            cpp_enabled = enabled;
            if (!enabled && fail_pause) {
                if (throws) {
                    throw std::runtime_error("Python reporting failed after C++ pause");
                }
                return false;
            }
            python_enabled = enabled;
            return true;
        };
        hooks.synchronizeAndDeregisterMr = [&](const SleepOptions&) {
            EXPECT_FALSE(cpp_enabled);
            EXPECT_FALSE(python_enabled);
            releases.push_back("mr");
            return true;
        };
        hooks.releaseKvMemoryBacking = [&](const SleepOptions&) {
            releases.push_back("kv");
            return true;
        };
        hooks.releaseRestorableGpuMemory = [&](const SleepOptions&) {
            releases.push_back("weights");
            return true;
        };
        controller.setHooks(hooks);
        SleepOptions prepare, commit;
        prepare.prepare_only = true;
        commit.commit_only   = true;
        ASSERT_TRUE(controller.sleep(prepare).ok);
        const auto epoch = controller.sleepEpoch();
        EXPECT_FALSE(controller.sleep(commit).ok);
        EXPECT_EQ(controller.state(), SleepState::DRAINING);
        EXPECT_EQ(controller.status().kv_memory_state, "ACTIVE");
        EXPECT_TRUE(controller.status().device_kv_cache_valid);
        EXPECT_TRUE(releases.empty());
        EXPECT_TRUE(cpp_enabled);
        EXPECT_TRUE(python_enabled);
        EXPECT_FALSE(controller.admit());

        fail_pause = false;
        ASSERT_TRUE(controller.sleep(commit).ok);
        EXPECT_EQ(controller.state(), SleepState::SLEEPING);
        EXPECT_EQ(controller.sleepEpoch(), epoch);
        EXPECT_EQ(releases, std::vector<std::string>({"mr", "kv", "weights"}));
        ASSERT_TRUE(controller.wakeUp().ok);
        EXPECT_TRUE(cpp_enabled);
        EXPECT_TRUE(python_enabled);
    }
}

TEST(SleepLifecycleMetricsTest, FailedPauseCompensationRemainsRetryableThroughDrainCancellation) {
    for (bool throws : {false, true}) {
        BoundSleepLifecycleController controller(true);
        bool                          cpp_enabled = true, python_enabled = true, fail_resume = true;
        int                           restarts = 0;
        SleepHooks                    hooks;
        hooks.setMetricsReportingEnabled = [&](bool enabled) {
            if (!enabled) {
                cpp_enabled = false;
                throw std::runtime_error("partial pause");
            }
            if (fail_resume) {
                if (throws) {
                    throw std::runtime_error("compensation unavailable");
                }
                return false;
            }
            cpp_enabled = python_enabled = true;
            return true;
        };
        hooks.cancelQuiesceAndRestartEngine = [&] {
            ++restarts;
            return true;
        };
        hooks.releaseKvMemoryBacking = [](const SleepOptions&) {
            ADD_FAILURE() << "metrics failure must precede GPU release";
            return true;
        };
        controller.setHooks(hooks);
        EXPECT_FALSE(controller.sleep(SleepOptions{}).ok);
        EXPECT_EQ(controller.state(), SleepState::DRAINING);
        EXPECT_FALSE(cpp_enabled);
        EXPECT_TRUE(python_enabled);
        EXPECT_FALSE(controller.wakeUp().ok);
        EXPECT_EQ(controller.state(), SleepState::RUNNING);
        EXPECT_EQ(controller.status().kv_memory_state, "ACTIVE");
        WakeUpOptions cancel;
        cancel.cancel_quiesce_token = "late-cancel";
        EXPECT_FALSE(controller.wakeUp(cancel).ok);
        fail_resume = false;
        EXPECT_TRUE(controller.wakeUp(cancel).ok);
        EXPECT_TRUE(cpp_enabled);
        EXPECT_TRUE(python_enabled);
        EXPECT_EQ(restarts, 1);
    }
}

TEST(SleepLifecycleMetricsTest, FailedReleaseRestoresReportingAndFailedWakeDoesNotReconnect) {
    for (bool fail_sleep : {false, true}) {
        BoundSleepLifecycleController controller(true);
        std::vector<bool>        changes;
        SleepHooks               hooks;
        hooks.releaseRestorableGpuMemory = [&](const SleepOptions&) { return !fail_sleep; };
        hooks.warmupAndHealthCheck       = [] { return false; };
        hooks.setMetricsReportingEnabled = [&](bool enabled) {
            changes.push_back(enabled);
            return true;
        };
        controller.setHooks(hooks);
        if (fail_sleep) {
            EXPECT_FALSE(controller.sleep(SleepOptions{}).ok);
            EXPECT_EQ(changes, std::vector<bool>({false, true}));
        } else {
            ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
            EXPECT_FALSE(controller.wakeUp().ok);
            EXPECT_EQ(changes, std::vector<bool>({false}));
        }
    }
}

TEST(SleepLifecycleMetricsTest, DisabledModeNeverSwitchesReporting) {
    BoundSleepLifecycleController controller(false);
    SleepHooks               hooks;
    hooks.setMetricsReportingEnabled = [](bool) {
        ADD_FAILURE() << "disabled sleep must not change reporting";
        return true;
    };
    controller.setHooks(hooks);
    EXPECT_FALSE(controller.sleep(SleepOptions{}).ok);
    EXPECT_FALSE(controller.wakeUp().ok);
}

TEST(SleepLifecycleControllerConcurrencyTest, ConcurrentAdmissionCannotEscapeClosedGate) {
    BoundSleepLifecycleController controller(true);
    std::atomic<bool>        stop{false};
    std::atomic<int>         accepted{0};
    SleepHooks               hooks;
    hooks.drain = [&](const SleepOptions&) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (controller.activeAdmissionCount() != 0) {
            if (std::chrono::steady_clock::now() > deadline) {
                return false;
            }
            std::this_thread::yield();
        }
        return true;
    };
    hooks.releaseKvMemoryBacking = [&](const SleepOptions&) {
        EXPECT_EQ(controller.activeAdmissionCount(), 0);
        EXPECT_FALSE(controller.admission()->admit().accepted);
        return true;
    };
    controller.setHooks(hooks);
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&] {
            while (!stop.load()) {
                auto result = controller.admission()->admit();
                if (result.accepted) {
                    accepted.fetch_add(1);
                    // Let close race with requests holding actual leases.
                    std::this_thread::yield();
                    result.complete();
                }
            }
        });
    }
    while (accepted.load() == 0) {
        std::this_thread::yield();
    }
    for (int cycle = 0; cycle < 50; ++cycle) {
        EXPECT_TRUE(controller.sleep(SleepOptions{}).ok);
        EXPECT_EQ(controller.state(), SleepState::SLEEPING);
        EXPECT_EQ(controller.activeAdmissionCount(), 0);
        for (int check = 0; check < 20; ++check) {
            EXPECT_FALSE(controller.admission()->admit().accepted);
        }
        EXPECT_TRUE(controller.wakeUp().ok);
    }
    stop.store(true);
    for (auto& worker : workers) {
        worker.join();
    }
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
}

namespace {

SleepOptions gracefulOptions() {
    SleepOptions opt;
    opt.mode       = "wait";
    opt.timeout_ms = 1000;
    return opt;
}

}  // namespace

static SleepOptions coordinatedDrain(const BoundSleepLifecycleController& controller, const std::string& token) {
    auto options                 = gracefulOptions();
    options.prepare_only         = true;
    options.drain_only           = true;
    options.quiesce_token        = token;
    options.expected_incarnation = controller.status().worker_incarnation;
    options.expected_sleep_epoch = controller.sleepEpoch();
    return options;
}

TEST(SleepLifecycleControllerTest, CoordinatedDrainFreezeCatchupAndCommitAreSeparate) {
    BoundSleepLifecycleController controller(true);
    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.requiresCoordinatedQuiesce = true;
    hooks.drain                      = [&](const SleepOptions&) {
        calls.emplace_back("drain");
        return true;
    };
    hooks.freezeEngineRounds = [&] {
        calls.emplace_back("freeze");
        return uint64_t{5};
    };
    hooks.quiesceEngineAtRound = [&](uint64_t round, int64_t) {
        EXPECT_EQ(round, 7);
        calls.emplace_back("quiesce");
        return true;
    };
    hooks.releaseKvMemoryBacking = [&](const SleepOptions&) {
        calls.emplace_back("release");
        return true;
    };
    controller.setHooks(hooks);
    EXPECT_FALSE(controller.sleep(gracefulOptions()).ok);
    EXPECT_TRUE(calls.empty());
    const auto drain = coordinatedDrain(controller, "operation-1");
    ASSERT_TRUE(controller.sleep(drain).ok);
    EXPECT_EQ(calls, std::vector<std::string>({"drain"}));
    auto commit         = drain;
    commit.prepare_only = commit.drain_only = false;
    commit.commit_only                      = true;
    EXPECT_FALSE(controller.sleep(commit).ok);
    uint64_t            frozen = 0;
    SleepQuiesceOptions quiesce{"operation-1", true, 0, 100};
    ASSERT_TRUE(controller.quiesce(quiesce, frozen).ok);
    EXPECT_EQ(frozen, 5);
    ASSERT_TRUE(controller.quiesce(quiesce, frozen).ok);
    EXPECT_EQ(calls, std::vector<std::string>({"drain", "drain", "freeze"}));
    quiesce.freeze_only  = false;
    quiesce.target_round = 4;
    EXPECT_FALSE(controller.quiesce(quiesce, frozen).ok);
    quiesce.target_round = 7;
    ASSERT_TRUE(controller.quiesce(quiesce, frozen).ok);
    ASSERT_TRUE(controller.quiesce(quiesce, frozen).ok);
    quiesce.target_round = 8;
    EXPECT_FALSE(controller.quiesce(quiesce, frozen).ok);
    EXPECT_EQ(calls, std::vector<std::string>({"drain", "drain", "freeze", "quiesce", "drain"}));
    ASSERT_TRUE(controller.sleep(commit).ok);
    EXPECT_EQ(calls, std::vector<std::string>({"drain", "drain", "freeze", "quiesce", "drain", "release"}));
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    WakeUpOptions cancel;
    cancel.cancel_quiesce_token = "operation-1";
    EXPECT_FALSE(controller.wakeUp(cancel).ok);
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
}

TEST(SleepLifecycleControllerTest, CancelBeforeDelayedInitialDrainFencesThatRequest) {
    BoundSleepLifecycleController controller(true);
    const auto               old_drain = coordinatedDrain(controller, "old");
    WakeUpOptions            cancel;
    cancel.cancel_quiesce_token = "old";
    ASSERT_TRUE(controller.wakeUp(cancel).ok);
    const auto epoch = controller.sleepEpoch();
    ASSERT_TRUE(controller.wakeUp(cancel).ok);
    EXPECT_EQ(controller.sleepEpoch(), epoch);
    EXPECT_FALSE(controller.sleep(old_drain).ok);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_TRUE(controller.sleep(coordinatedDrain(controller, "new")).ok);
}

TEST(SleepLifecycleControllerTest, OldWorkerIncarnationCannotPrepareANewWorker) {
    BoundSleepLifecycleController old_worker(true), new_worker(true);
    EXPECT_NE(old_worker.status().worker_incarnation, new_worker.status().worker_incarnation);
    EXPECT_FALSE(new_worker.sleep(coordinatedDrain(old_worker, "old-process")).ok);
    EXPECT_EQ(new_worker.state(), SleepState::RUNNING);
}

TEST(SleepLifecycleControllerTest, OldFreezeTargetCommitAndCancelCannotAffectANewDrain) {
    BoundSleepLifecycleController controller(true);
    auto                     old = coordinatedDrain(controller, "old");
    ASSERT_TRUE(controller.sleep(old).ok);
    WakeUpOptions cancel;
    cancel.cancel_quiesce_token = "old";
    ASSERT_TRUE(controller.wakeUp(cancel).ok);
    ASSERT_TRUE(controller.sleep(coordinatedDrain(controller, "new")).ok);
    uint64_t round = 0;
    EXPECT_FALSE(controller.quiesce({"old", true, 0, 100}, round).ok);
    EXPECT_FALSE(controller.quiesce({"old", false, 7, 100}, round).ok);
    old.drain_only = old.prepare_only = false;
    old.commit_only                   = true;
    EXPECT_FALSE(controller.sleep(old).ok);
    EXPECT_FALSE(controller.wakeUp(cancel).ok);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_TRUE(controller.quiesce({"new", true, 0, 100}, round).ok);
}

TEST(SleepLifecycleControllerTest, DrainTimeoutNeverFreezesAndCanRollBack) {
    BoundSleepLifecycleController controller(true);
    int                      frozen = 0, released = 0;
    SleepHooks               hooks;
    hooks.drain              = [](const SleepOptions&) { return false; };
    hooks.freezeEngineRounds = [&] {
        ++frozen;
        return uint64_t{1};
    };
    hooks.releaseKvMemoryBacking = [&](const SleepOptions&) {
        ++released;
        return true;
    };
    controller.setHooks(hooks);
    EXPECT_FALSE(controller.sleep(coordinatedDrain(controller, "timeout")).ok);
    uint64_t round = 0;
    EXPECT_FALSE(controller.quiesce({"timeout", true, 0, 100}, round).ok);
    WakeUpOptions cancel;
    cancel.cancel_quiesce_token = "timeout";
    EXPECT_TRUE(controller.wakeUp(cancel).ok);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_EQ(frozen, 0);
    EXPECT_EQ(released, 0);
}

TEST(SleepLifecycleControllerTest, FailedAsyncQuiesceCannotCommitResources) {
    BoundSleepLifecycleController controller(true);
    int                      released = 0;
    SleepHooks               hooks;
    hooks.quiesceEngineAtRound   = [](uint64_t, int64_t) -> bool { throw std::runtime_error("async CPU error"); };
    hooks.releaseKvMemoryBacking = [&](const SleepOptions&) {
        ++released;
        return true;
    };
    controller.setHooks(hooks);
    auto drain = coordinatedDrain(controller, "failed");
    ASSERT_TRUE(controller.sleep(drain).ok);
    uint64_t round = 0;
    ASSERT_TRUE(controller.quiesce({"failed", true, 0, 100}, round).ok);
    EXPECT_FALSE(controller.quiesce({"failed", false, 0, 100}, round).ok);
    drain.drain_only = drain.prepare_only = false;
    drain.commit_only                     = true;
    EXPECT_FALSE(controller.sleep(drain).ok);
    EXPECT_EQ(released, 0);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
}

TEST(SleepLifecycleControllerTest, InitialStateIsRunning) {
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller;
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
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

TEST(SleepLifecycleControllerTest, NonEmptyTagsRejectBeforeDrainOrRelease) {
    for (const auto& tag : {"weights", "kv_cache"}) {
        for (const auto phase : {0, 1, 2}) {
            BoundSleepLifecycleController controller(true);
            int                      hook_calls = 0;
            SleepHooks               hooks;
            hooks.drain = [&](const SleepOptions&) {
                ++hook_calls;
                return true;
            };
            hooks.releaseRestorableGpuMemory = [&](const SleepOptions&) {
                ++hook_calls;
                return true;
            };
            controller.setHooks(hooks);
            auto opt         = gracefulOptions();
            opt.tags         = {tag};
            opt.prepare_only = phase == 1;
            opt.commit_only  = phase == 2;

            const auto result = controller.sleep(opt);

            EXPECT_FALSE(result.ok);
            EXPECT_EQ(result.code, SleepResult::Code::INVALID_ARGUMENT);
            EXPECT_NE(result.message.find("tags"), std::string::npos);
            EXPECT_EQ(hook_calls, 0);
            EXPECT_EQ(controller.state(), SleepState::RUNNING);
            EXPECT_EQ(controller.sleepEpoch(), 0);
            EXPECT_TRUE(controller.admit());
        }
    }
}

TEST(SleepLifecycleControllerTest, LevelZeroIsDefinedButUnimplemented) {
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), SleepState::SLEEPING);

    const auto again = controller.sleep(gracefulOptions());
    EXPECT_TRUE(again.ok) << again.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    // Idempotent repeat must NOT bump the epoch.
    EXPECT_EQ(controller.sleepEpoch(), 1);
}

TEST(SleepLifecycleControllerTest, WakeUpIsIdempotent) {
    BoundSleepLifecycleController controller(true);
    EXPECT_TRUE(controller.wakeUp().ok);  // RUNNING -> wake_up == no-op success
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
}

TEST(SleepLifecycleControllerTest, EpochIsMonotonicAcrossCycles) {
    BoundSleepLifecycleController controller(true);
    for (int64_t i = 1; i <= 3; ++i) {
        ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
        EXPECT_EQ(controller.sleepEpoch(), i);
        ASSERT_TRUE(controller.wakeUp().ok);
        EXPECT_EQ(controller.sleepEpoch(), i);
    }
}

TEST(SleepLifecycleControllerTest, DrainTimeoutKeepsDraining) {
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [&controller](const SleepOptions&) { return controller.activeAdmissionCount() == 0; };
    controller.setHooks(hooks);

    SleepResult first_sleep;
    {
        auto admission = controller.admission()->admit();
        ASSERT_TRUE(admission.accepted);
        EXPECT_EQ(controller.activeAdmissionCount(), 1);

        first_sleep = controller.sleep(gracefulOptions());
        EXPECT_FALSE(first_sleep.ok);
        EXPECT_EQ(controller.state(), SleepState::DRAINING);
        admission.complete();
    }

    EXPECT_EQ(controller.activeAdmissionCount(), 0);
    const auto retry = controller.sleep(gracefulOptions());
    EXPECT_TRUE(retry.ok) << retry.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
}

TEST(SleepLifecycleControllerTest, SleepRetryFromDrainingCanComplete) {
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [&controller](const SleepOptions&) { return controller.activeAdmissionCount() == 0; };
    controller.setHooks(hooks);

    SleepOptions prepare = gracefulOptions();
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.sleep(prepare).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);

    auto straggler = controller.admission()->admit();
    EXPECT_FALSE(straggler.accepted);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_EQ(controller.activeAdmissionCount(), 0);

    SleepOptions commit = gracefulOptions();
    commit.commit_only  = true;
    const auto result   = controller.sleep(commit);
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
}

TEST(SleepLifecycleControllerTest, CommitOnlyRequiresPreparedQuiesce) {
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
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

TEST(SleepLifecycleControllerTest, WakeHealthCheckCompletesWhileExecutionIsStillParked) {
    BoundSleepLifecycleController controller(true);
    bool                          execution_running = false;
    int                           health_checks     = 0;
    SleepHooks                    hooks;
    hooks.warmupAndHealthCheck = [&]() {
        EXPECT_FALSE(execution_running);
        ++health_checks;
        return !execution_running;
    };
    hooks.restartEngine = [&]() {
        execution_running = true;
        return true;
    };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    WakeUpOptions prepare;
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.wakeUp(prepare).ok);
    EXPECT_EQ(health_checks, 1);
    EXPECT_FALSE(execution_running);
    WakeUpOptions commit;
    commit.commit_only = true;
    EXPECT_TRUE(controller.wakeUp(commit).ok);
    EXPECT_EQ(health_checks, 1);
    EXPECT_TRUE(execution_running);
}

TEST(SleepLifecycleControllerTest, WakePreparedIsACompletionFactNotTheWakingState) {
    BoundSleepLifecycleController controller(true);
    std::promise<void>            entered, release;
    auto                          released = release.get_future().share();
    std::atomic<int>              restores{0};
    SleepHooks                    hooks;
    hooks.restoreRestorableGpuMemory = [&]() {
        ++restores;
        entered.set_value();
        released.wait();
        return true;
    };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    WakeUpOptions prepare;
    prepare.prepare_only         = true;
    prepare.expected_incarnation = controller.status().worker_incarnation;
    prepare.expected_sleep_epoch = controller.sleepEpoch();
    auto preparing               = std::async(std::launch::async, [&] { return controller.wakeUp(prepare); });
    entered.get_future().wait();
    const auto during_restore = controller.status();
    EXPECT_EQ(during_restore.state, SleepState::WAKING_UP);
    EXPECT_FALSE(during_restore.wake_prepared);
    release.set_value();
    ASSERT_TRUE(preparing.get().ok);
    EXPECT_TRUE(controller.status().wake_prepared);
    ASSERT_TRUE(controller.wakeUp(prepare).ok);
    EXPECT_EQ(restores.load(), 1);
    WakeUpOptions commit = prepare;
    commit.prepare_only  = false;
    commit.commit_only   = true;
    ASSERT_TRUE(controller.wakeUp(commit).ok);
    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    EXPECT_FALSE(controller.status().wake_prepared);
    EXPECT_FALSE(controller.wakeUp(commit).ok);  // Previous epoch cannot resume.
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
}

TEST(SleepLifecycleControllerTest, StaleWakeAndTerminalIntentDoNotRunRestoreHooks) {
    BoundSleepLifecycleController controller(true);
    int                           restores = 0;
    SleepHooks                    hooks;
    hooks.restoreRestorableGpuMemory = [&]() {
        ++restores;
        return true;
    };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    WakeUpOptions prepare;
    prepare.prepare_only         = true;
    prepare.expected_incarnation = "old-worker";
    prepare.expected_sleep_epoch = controller.sleepEpoch();
    EXPECT_FALSE(controller.wakeUp(prepare).ok);
    EXPECT_EQ(restores, 0);
    prepare.expected_incarnation = controller.status().worker_incarnation;
    controller.admission()->beginTermination();
    EXPECT_FALSE(controller.wakeUp(prepare).ok);
    EXPECT_EQ(restores, 0);
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
}

TEST(SleepLifecycleControllerTest, TerminalIntentDuringRealPrepareKeepsAdmissionClosedAndCommitRejected) {
    for (bool restore_succeeds : {true, false}) {
        SCOPED_TRACE(restore_succeeds);
        BoundSleepLifecycleController controller(true);
        std::promise<void>            entered, release;
        auto                          entered_future = entered.get_future();
        auto                          released       = release.get_future().share();
        std::atomic<int>              restores{0}, kv_restores{0}, registrations{0}, checks{0}, restarts{0};
        SleepHooks                    hooks;
        hooks.restoreRestorableGpuMemory = [&]() {
            ++restores;
            entered.set_value();
            released.wait();
            return restore_succeeds;
        };
        hooks.restoreKvMemoryBackingAndResetMetadata = [&]() {
            ++kv_restores;
            return true;
        };
        hooks.registerMr = [&]() {
            ++registrations;
            return true;
        };
        hooks.warmupAndHealthCheck = [&]() {
            ++checks;
            return true;
        };
        hooks.restartEngine = [&]() {
            ++restarts;
            return true;
        };
        controller.setHooks(hooks);
        ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
        WakeUpOptions prepare;
        prepare.prepare_only         = true;
        prepare.expected_incarnation = controller.status().worker_incarnation;
        prepare.expected_sleep_epoch = controller.sleepEpoch();
        auto       preparing         = std::async(std::launch::async, [&] { return controller.wakeUp(prepare); });
        const bool restore_entered   = entered_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        if (!restore_entered) {
            // Always unblock a late hook before destroying its async future.
            release.set_value();
            preparing.wait();
            FAIL() << "prepare never entered the controlled restore hook";
        }
        EXPECT_EQ(controller.state(), SleepState::WAKING_UP);
        EXPECT_FALSE(controller.status().wake_prepared);
        EXPECT_EQ(preparing.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);

        // This is the real first step in RtpLLMOp::beginShutdown. It does not
        // acquire the controller transition mutex held by the blocked restore.
        controller.admission()->beginTermination();
        EXPECT_TRUE(controller.admission()->terminating());
        EXPECT_FALSE(controller.admission()->admit().accepted);
        EXPECT_FALSE(controller.admission()->admit(true).accepted);
        EXPECT_EQ(restarts.load(), 0);
        EXPECT_FALSE(controller.status().wake_prepared);

        release.set_value();
        const auto prepare_result = preparing.get();
        EXPECT_EQ(prepare_result.ok, restore_succeeds);
        // Terminal intent does not cancel already-entered resource work. Its
        // successful prepare may finish, but cannot authorize a later commit.
        const auto expected_state = restore_succeeds ? SleepState::WAKING_UP : SleepState::ERROR;
        EXPECT_EQ(controller.state(), expected_state);
        EXPECT_EQ(controller.status().wake_prepared, restore_succeeds);
        EXPECT_FALSE(controller.admit());
        EXPECT_FALSE(controller.admission()->admit().accepted);
        EXPECT_FALSE(controller.admission()->admit(true).accepted);

        auto commit         = prepare;
        commit.prepare_only = false;
        commit.commit_only  = true;
        for (const auto& retry : {commit, prepare}) {
            const auto result = controller.wakeUp(retry);
            EXPECT_FALSE(result.ok);
            EXPECT_EQ(result.code, SleepResult::Code::FAILED_PRECONDITION);
            EXPECT_NE(result.message.find("termination intent"), std::string::npos);
        }
        EXPECT_EQ(controller.state(), expected_state);
        EXPECT_EQ(controller.sleepEpoch(), prepare.expected_sleep_epoch);
        EXPECT_EQ(restores.load(), 1);
        EXPECT_EQ(kv_restores.load(), restore_succeeds ? 1 : 0);
        EXPECT_EQ(registrations.load(), restore_succeeds ? 1 : 0);
        EXPECT_EQ(checks.load(), restore_succeeds ? 1 : 0);
        EXPECT_EQ(restarts.load(), 0);
        EXPECT_FALSE(controller.admission()->admit().accepted);
        EXPECT_FALSE(controller.admission()->admit(true).accepted);
    }
}

TEST(SleepLifecycleControllerTest, FailedWakeHealthCheckDoesNotPublishPreparedOrRestart) {
    BoundSleepLifecycleController controller(true);
    int                           restarts = 0;
    SleepHooks                    hooks;
    hooks.warmupAndHealthCheck = [] { return false; };
    hooks.restartEngine        = [&] {
        ++restarts;
        return true;
    };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    WakeUpOptions prepare;
    prepare.prepare_only = true;
    EXPECT_FALSE(controller.wakeUp(prepare).ok);
    EXPECT_FALSE(controller.status().wake_prepared);
    EXPECT_EQ(restarts, 0);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
}

TEST(SleepLifecycleControllerTest, WakeUpPrepareOnlyStaysWakingUpUntilCommit) {
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
    std::atomic<int>         abort_seen{0};
    std::vector<std::string> modes;
    SleepHooks               hooks;
    hooks.drain = [&abort_seen, &modes](const SleepOptions& opt) {
        modes.push_back(opt.mode);
        if (opt.mode == "abort") {
            abort_seen++;
            return true;
        }
        return abort_seen.load() > 0;  // Cancellation has already drained work.
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
    EXPECT_EQ(modes, std::vector<std::string>({"wait", "abort", "wait", "wait"}));
}

TEST(SleepLifecycleControllerTest, SleepHookFailureGoesToError) {
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
    std::atomic<int>         release_kv_called{0};
    SleepHooks               hooks;
    hooks.releaseKvMemoryBacking = [&release_kv_called](const SleepOptions&) {
        release_kv_called++;
        return true;
    };
    hooks.releaseRestorableGpuMemory = [](const SleepOptions&) { return false; };
    hooks.hookFailureDetail          = [](const char* hook_name) {
        EXPECT_STREQ(hook_name, "releaseRestorableGpuMemory");
        return std::string("[NcclMemory] [sleep] ncclCommSuspend failed for tp(rc=3); restart the instance");
    };
    controller.setHooks(hooks);

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.code, SleepResult::Code::FAILED_PRECONDITION);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admit());
    EXPECT_EQ(release_kv_called.load(), 1);
    EXPECT_EQ(controller.status().kv_memory_state, "PAUSED");
    EXPECT_FALSE(controller.status().device_kv_cache_valid);
    EXPECT_EQ(controller.status().gpu_resource_state, "UNKNOWN");
    // The hook's own message survives, and the NCCL detail is appended to it.
    EXPECT_EQ(
        result.message,
        "releaseRestorableGpuMemory failed: [NcclMemory] [sleep] ncclCommSuspend failed for tp(rc=3); restart the instance");
    EXPECT_EQ(controller.status().last_error, result.message);
}

TEST(SleepLifecycleControllerTest, WakeUpFailureGoesToError) {
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);

    std::atomic<int> release_kv_called{0};
    SleepHooks       hooks;
    hooks.restoreKvMemoryBackingAndResetMetadata = []() { return true; };
    hooks.restoreRestorableGpuMemory             = []() { return false; };
    hooks.hookFailureDetail                      = [](const char* hook_name) {
        EXPECT_STREQ(hook_name, "restoreRestorableGpuMemory");
        return std::string("[NcclMemory] [wake] ncclCommResume failed for tp(rc=3); restart the instance");
    };
    hooks.releaseKvMemoryBacking = [&release_kv_called](const SleepOptions&) {
        release_kv_called++;
        return true;
    };
    controller.setHooks(hooks);

    const auto result = controller.wakeUp();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.code, SleepResult::Code::FAILED_PRECONDITION);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_EQ(release_kv_called.load(), 0);
    EXPECT_EQ(
        result.message,
        "restoreRestorableGpuMemory failed: [NcclMemory] [wake] ncclCommResume failed for tp(rc=3); restart the instance");
    EXPECT_EQ(controller.status().last_error, result.message);
}

// hookFailureDetail is a diagnostic on the failure path, so every way it can come
// back short has to leave the verdict and the operator-visible message exactly as
// they would be without it. An empty string is the *normal* answer -- NCCL is only
// one of several reasons releaseRestorableGpuMemory can fail -- so appending a bare
// ": " there would be a permanent cosmetic regression, not a corner case.
TEST(SleepLifecycleControllerTest, SleepHookFailureDetailFallbacks) {
    const std::string expected = "releaseRestorableGpuMemory failed";

    {  // No provider installed at all: the pre-existing message, unchanged.
        BoundSleepLifecycleController controller(true);
        SleepHooks               hooks;
        hooks.releaseRestorableGpuMemory = [](const SleepOptions&) { return false; };
        controller.setHooks(hooks);

        const auto result = controller.sleep(gracefulOptions());
        EXPECT_FALSE(result.ok);
        EXPECT_EQ(result.code, SleepResult::Code::FAILED_PRECONDITION);
        EXPECT_EQ(result.message, expected);
    }

    {  // NCCL is not the cause: no separator, no trailing colon.
        BoundSleepLifecycleController controller(true);
        std::atomic<int>         detail_calls{0};
        SleepHooks               hooks;
        hooks.releaseRestorableGpuMemory = [](const SleepOptions&) { return false; };
        hooks.hookFailureDetail          = [&detail_calls](const char*) {
            detail_calls++;
            return std::string();
        };
        controller.setHooks(hooks);

        const auto result = controller.sleep(gracefulOptions());
        EXPECT_FALSE(result.ok);
        EXPECT_EQ(result.message, expected);
        EXPECT_EQ(detail_calls.load(), 1);
    }

    {  // The provider throws std::exception: warn, fall back, verdict unchanged.
        BoundSleepLifecycleController controller(true);
        SleepHooks               hooks;
        hooks.releaseRestorableGpuMemory = [](const SleepOptions&) { return false; };
        hooks.hookFailureDetail          = [](const char*) -> std::string { throw std::runtime_error("gil deadlock"); };
        controller.setHooks(hooks);

        const auto result = controller.sleep(gracefulOptions());
        EXPECT_FALSE(result.ok);
        EXPECT_EQ(result.code, SleepResult::Code::FAILED_PRECONDITION);
        EXPECT_EQ(controller.state(), SleepState::ERROR);
        EXPECT_EQ(result.message, expected);
    }

    {  // A non-std throw (pybind11's error_already_set does not derive from
       // std::exception on every toolchain) must not escape either.
        BoundSleepLifecycleController controller(true);
        SleepHooks               hooks;
        hooks.releaseRestorableGpuMemory = [](const SleepOptions&) { return false; };
        hooks.hookFailureDetail          = [](const char*) -> std::string { throw 42; };
        controller.setHooks(hooks);

        const auto result = controller.sleep(gracefulOptions());
        EXPECT_FALSE(result.ok);
        EXPECT_EQ(result.code, SleepResult::Code::FAILED_PRECONDITION);
        EXPECT_EQ(controller.state(), SleepState::ERROR);
        EXPECT_EQ(result.message, expected);
    }

    {  // Healthy sleep: a diagnostic must never be invoked when nothing failed.
        BoundSleepLifecycleController controller(true);
        std::atomic<int>         detail_calls{0};
        SleepHooks               hooks;
        hooks.hookFailureDetail = [&detail_calls](const char*) {
            detail_calls++;
            return std::string("should not be reached");
        };
        controller.setHooks(hooks);

        EXPECT_TRUE(controller.sleep(gracefulOptions()).ok);
        EXPECT_TRUE(controller.wakeUp().ok);
        EXPECT_EQ(detail_calls.load(), 0);
    }
}

TEST(SleepLifecycleControllerTest, WakeUpHookExceptionGoesToError) {
    BoundSleepLifecycleController controller(true);
    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);

    SleepHooks hooks;
    hooks.restoreKvMemoryBackingAndResetMetadata = []() -> bool { throw std::runtime_error("boom"); };
    controller.setHooks(hooks);

    const auto result = controller.wakeUp();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admit());
}

TEST(SleepLifecycleControllerTest, SleepReleaseHookExceptionGoesToError) {
    BoundSleepLifecycleController controller(true);
    SleepHooks hooks;
    hooks.releaseKvMemoryBacking = [](const SleepOptions&) -> bool {
        throw std::runtime_error("live cache references after quiesce");
    };
    controller.setHooks(hooks);

    const auto result = controller.sleep(gracefulOptions());
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admit());
    EXPECT_FALSE(controller.wakeUp().ok);
}

TEST(SleepLifecycleControllerTest, ErrorIsTerminalAndRejectsWakeUp) {
    BoundSleepLifecycleController controller(true);
    SleepHooks                    hooks;
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.activeRequestCount       = []() { return 7; };
    hooks.activeCacheTransferCount = []() { return 3; };
    controller.setHooks(hooks);

    const auto status = controller.status();
    EXPECT_EQ(status.active_request_count, 7);
    EXPECT_EQ(status.active_cache_transfer_count, 3);
}

TEST(SleepLifecycleControllerTest, ConcurrentSleepWakeUpIsSerializedAndConsistent) {
    BoundSleepLifecycleController controller(true);
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

TEST(SleepContinuationTest, EarlyDrainedPeerAcceptsBusyRanksLateKvLoadUntilFreeze) {
    BoundSleepLifecycleController root(true), peer(true);
    for (auto* controller : {&root, &peer}) {
        SleepHooks hooks;
        hooks.requiresCoordinatedQuiesce = true;
        hooks.drain = [controller](const SleepOptions&) { return controller->activeAdmissionCount() == 0; };
        hooks.releaseKvMemoryBacking     = [controller](const SleepOptions&) {
            EXPECT_EQ(controller->activeAdmissionCount(), 0);
            EXPECT_FALSE(controller->admission()->admit(true).accepted);
            EXPECT_FALSE(controller->admission()->admit().accepted);
            return true;
        };
        controller->setHooks(hooks);
    }
    auto parent     = root.admission()->admit();
    auto root_drain = coordinatedDrain(root, "drain");
    auto peer_drain = coordinatedDrain(peer, "drain");
    ASSERT_TRUE(peer.sleep(peer_drain).ok);
    EXPECT_FALSE(root.sleep(root_drain).ok);  // admitted parent is still alive
    EXPECT_FALSE(root.admission()->admit().accepted);
    EXPECT_FALSE(peer.admission()->admit().accepted);
    auto child = peer.admission()->admit(true);
    ASSERT_TRUE(child.accepted);
    EXPECT_EQ(peer.state(), SleepState::DRAINING);
    EXPECT_EQ(peer.activeAdmissionCount(), 1);
    EXPECT_FALSE(root.sleep(root_drain).ok);
    child.complete();
    parent.complete();
    ASSERT_TRUE(root.sleep(root_drain).ok);
    uint64_t round = 0;
    // Model the coordinator's barriers: all freezes, all quiesces, all commits.
    for (auto* controller : {&root, &peer}) {
        ASSERT_TRUE(controller->quiesce({"drain", true, 0, 100}, round).ok);
        EXPECT_FALSE(controller->admission()->admit(true).accepted);
    }
    for (auto* controller : {&root, &peer}) {
        ASSERT_TRUE(controller->quiesce({"drain", false, round, 100}, round).ok);
    }
    for (auto* controller : {&root, &peer}) {
        auto commit         = controller == &root ? root_drain : peer_drain;
        commit.prepare_only = commit.drain_only = false;
        commit.commit_only                      = true;
        EXPECT_TRUE(controller->sleep(commit).ok);
        EXPECT_EQ(controller->state(), SleepState::SLEEPING);
        EXPECT_FALSE(controller->admission()->admit(true).accepted);
    }
    for (auto* controller : {&root, &peer}) {
        ASSERT_TRUE(controller->wakeUp().ok);
        auto next = controller->admission()->admit(true);
        EXPECT_TRUE(next.accepted);
        next.complete();
    }
}

TEST(SleepContinuationTest, LateCleanupBlocksFreezeAndRetryDrainsBeforeAcknowledgement) {
    BoundSleepLifecycleController controller(true);
    int                      freezes = 0;
    SleepHooks               hooks;
    hooks.requiresCoordinatedQuiesce = true;
    hooks.drain                      = [&](const SleepOptions&) { return controller.activeAdmissionCount() == 0; };
    hooks.freezeEngineRounds         = [&] {
        ++freezes;
        EXPECT_EQ(controller.activeAdmissionCount(), 0);
        EXPECT_FALSE(controller.admission()->admit(true).accepted);
        return uint64_t{3};
    };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(coordinatedDrain(controller, "cleanup")).ok);
    auto late = controller.admission()->admit(true);
    ASSERT_TRUE(late.accepted);
    uint64_t round = 0;
    EXPECT_FALSE(controller.quiesce({"cleanup", true, 0, 1}, round).ok);
    EXPECT_EQ(freezes, 0);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_FALSE(controller.admission()->admit(true).accepted);
    EXPECT_EQ(controller.activeAdmissionCount(), 1);
    late.complete();
    ASSERT_TRUE(controller.quiesce({"cleanup", true, 0, 100}, round).ok);
    EXPECT_EQ(freezes, 1);
    EXPECT_EQ(round, 3);
    ASSERT_TRUE(controller.quiesce({"cleanup", true, 0, 100}, round).ok);
    EXPECT_EQ(freezes, 1);
    WakeUpOptions cancel;
    cancel.cancel_quiesce_token = "cleanup";
    ASSERT_TRUE(controller.wakeUp(cancel).ok);
    EXPECT_TRUE(admitAndComplete(controller.admission()->admit()));
    EXPECT_TRUE(admitAndComplete(controller.admission()->admit(true)));
    EXPECT_FALSE(controller.quiesce({"cleanup", true, 0, 100}, round).ok);
}

TEST(SleepContinuationTest, StaleTokenCannotCloseOrReopenContinuationGate) {
    BoundSleepLifecycleController controller(true);
    SleepHooks                    hooks;
    hooks.drain = [](const SleepOptions&) { return true; };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(coordinatedDrain(controller, "current")).ok);
    uint64_t round = 0;
    EXPECT_FALSE(controller.quiesce({"old", true, 0, 100}, round).ok);
    EXPECT_TRUE(admitAndComplete(controller.admission()->admit(true)));
    ASSERT_TRUE(controller.quiesce({"current", true, 0, 100}, round).ok);
    WakeUpOptions old;
    old.cancel_quiesce_token = "old";
    EXPECT_FALSE(controller.wakeUp(old).ok);
    EXPECT_FALSE(controller.admission()->admit(true).accepted);
}

TEST(SleepContinuationTest, NoncoordinatedPrepareClosesAndRedrainsBeforeQuiesce) {
    BoundSleepLifecycleController controller(true);
    std::function<void()> racing_child;
    int                      drains = 0;
    int                      pauses = 0;
    SleepHooks               hooks;
    hooks.drain = [&](const SleepOptions&) {
        if (++drains == 1) {
            // A child wins admission immediately after a local zero observation.
            racing_child = std::move(controller.admission()->admit(true).complete);
            EXPECT_TRUE(static_cast<bool>(racing_child));
            return true;
        }
        return controller.activeAdmissionCount() == 0;
    };
    hooks.quiesceEngine = [&](const SleepOptions&) {
        ++pauses;
        return true;
    };
    controller.setHooks(hooks);
    auto options         = gracefulOptions();
    options.prepare_only = true;
    EXPECT_FALSE(controller.sleep(options).ok);
    EXPECT_EQ(pauses, 0);
    EXPECT_FALSE(controller.admission()->admit(true).accepted);
    EXPECT_EQ(controller.activeAdmissionCount(), 1);
    racing_child();
    ASSERT_TRUE(controller.sleep(options).ok);
    EXPECT_EQ(pauses, 1);
    EXPECT_FALSE(controller.admission()->admit(true).accepted);
}

TEST(SleepContinuationTest, DrainExceptionsKeepGateClosedAndResourcesIntact) {
    BoundSleepLifecycleController controller(true);
    int                      drains  = 0;
    int                      freezes = 0;
    SleepHooks               hooks;
    hooks.drain = [&](const SleepOptions&) {
        if (++drains > 1) {
            throw std::runtime_error("continuation cleanup failed");
        }
        return true;
    };
    hooks.freezeEngineRounds = [&] {
        ++freezes;
        return uint64_t{0};
    };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(coordinatedDrain(controller, "throw")).ok);
    uint64_t round = 0;
    EXPECT_FALSE(controller.quiesce({"throw", true, 0, 1}, round).ok);
    EXPECT_EQ(freezes, 0);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_TRUE(controller.status().device_kv_cache_valid);
    EXPECT_FALSE(controller.admission()->admit(true).accepted);
    EXPECT_NE(controller.status().last_error.find("continuation cleanup failed"), std::string::npos);
}

TEST(SleepContinuationTest, CancellationWaitsForFreezeDrainThenReopensWithoutReleasingResources) {
    BoundSleepLifecycleController controller(true);
    std::promise<void>       drain_entered;
    std::promise<void>       finish_drain;
    auto                     drain_entered_future = drain_entered.get_future();
    auto                     finish_drain_future  = finish_drain.get_future();
    std::atomic<int>         drains{0}, freezes{0}, releases{0}, restarts{0};
    SleepHooks               hooks;
    hooks.drain = [&](const SleepOptions& options) {
        if (++drains == 1) {
            return true;
        }
        EXPECT_EQ(options.timeout_ms, 60000);
        EXPECT_EQ(options.mode, "wait");
        drain_entered.set_value();
        // The test controls completion rather than sleeping for the production
        // drain budget. The bound also prevents a failing test from hanging.
        EXPECT_EQ(finish_drain_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        return false;  // Model the bounded continuation drain timing out.
    };
    hooks.freezeEngineRounds = [&] {
        ++freezes;
        return uint64_t{0};
    };
    hooks.releaseKvMemoryBacking = [&](const SleepOptions&) {
        ++releases;
        return true;
    };
    hooks.cancelQuiesceAndRestartEngine = [&] {
        ++restarts;
        return true;
    };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(coordinatedDrain(controller, "cancel-freeze")).ok);
    auto child = controller.admission()->admit(true);
    ASSERT_TRUE(child.accepted);

    auto freeze = std::async(std::launch::async, [&] {
        uint64_t round = 0;
        return controller.quiesce({"cancel-freeze", true, 0, 60000}, round);
    });
    EXPECT_EQ(drain_entered_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_FALSE(controller.admission()->admit().accepted);
    EXPECT_FALSE(controller.admission()->admit(true).accepted);

    std::promise<void> cancel_started;
    auto               cancel_started_future = cancel_started.get_future();
    auto               cancel                = std::async(std::launch::async, [&] {
        WakeUpOptions options;
        options.cancel_quiesce_token = "cancel-freeze";
        cancel_started.set_value();
        return controller.wakeUp(options);
    });
    EXPECT_EQ(cancel_started_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    // Current contract: cancellation serializes behind the blocking drain; it
    // cannot interrupt the hook or reopen admission before the hook returns.
    EXPECT_EQ(cancel.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    EXPECT_EQ(restarts.load(), 0);
    EXPECT_EQ(releases.load(), 0);
    finish_drain.set_value();

    EXPECT_FALSE(freeze.get().ok);
    EXPECT_TRUE(cancel.get().ok);
    EXPECT_EQ(freezes.load(), 0);
    EXPECT_EQ(releases.load(), 0);
    EXPECT_EQ(restarts.load(), 1);
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_TRUE(controller.status().device_kv_cache_valid);
    EXPECT_EQ(controller.activeAdmissionCount(), 1);
    EXPECT_TRUE(admitAndComplete(controller.admission()->admit()));
    EXPECT_TRUE(admitAndComplete(controller.admission()->admit(true)));
    child.complete();
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
}

TEST(SleepContinuationTest, ConcurrentCloseCannotMissPrecloseLeaseOrAdmitPostcloseWork) {
    BoundSleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [&](const SleepOptions&) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (controller.activeAdmissionCount() != 0) {
            if (std::chrono::steady_clock::now() > deadline) {
                return false;
            }
            std::this_thread::yield();
        }
        return true;
    };
    controller.setHooks(hooks);
    ASSERT_TRUE(controller.sleep(coordinatedDrain(controller, "race")).ok);
    std::atomic<bool>        stop{false};
    std::atomic<int>         accepted{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            while (!stop.load()) {
                auto child = controller.admission()->admit(true);
                if (child.accepted) {
                    ++accepted;
                    std::this_thread::yield();
                    child.complete();
                }
            }
        });
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (accepted.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    uint64_t   round  = 0;
    const auto result = controller.quiesce({"race", true, 0, 1000}, round);
    stop.store(true);
    for (auto& t : threads) {
        t.join();
    }
    EXPECT_GT(accepted.load(), 0);
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
    for (int i = 0; i < 100; ++i) {
        EXPECT_FALSE(controller.admission()->admit(true).accepted);
        EXPECT_FALSE(controller.admission()->admit().accepted);
    }
}

TEST(SleepContinuationTest, SuccessfulDrainHookCannotReleaseWithALiveLease) {
    BoundSleepLifecycleController controller(true);
    SleepHooks               hooks;
    int                      releases = 0;
    // Deliberately incomplete provider: the controller must guard its own count.
    hooks.drain                  = [](const SleepOptions&) { return true; };
    hooks.releaseKvMemoryBacking = [&](const SleepOptions&) {
        ++releases;
        return true;
    };
    controller.setHooks(hooks);
    auto live = controller.admission()->admit();
    EXPECT_FALSE(controller.sleep(gracefulOptions()).ok);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_EQ(releases, 0);
    EXPECT_FALSE(controller.admission()->admit(true).accepted);
    live.complete();
    ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
    EXPECT_EQ(releases, 1);
}

TEST(SleepContinuationTest, TwoPhaseWakeKeepsContinuationClosedUntilCommitAcrossCycles) {
    BoundSleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [&](const SleepOptions&) { return controller.activeAdmissionCount() == 0; };
    hooks.restoreKvMemoryBackingAndResetMetadata = [&] {
        EXPECT_EQ(controller.state(), SleepState::WAKING_UP);
        EXPECT_FALSE(controller.admission()->admit().accepted);
        EXPECT_FALSE(controller.admission()->admit(true).accepted);
        return true;
    };
    controller.setHooks(hooks);
    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(controller.sleep(gracefulOptions()).ok);
        EXPECT_EQ(controller.sleepEpoch(), i + 1);
        EXPECT_FALSE(controller.admission()->admit(true).accepted);
        WakeUpOptions prepare;
        prepare.prepare_only = true;
        ASSERT_TRUE(controller.wakeUp(prepare).ok);
        EXPECT_FALSE(controller.admission()->admit(true).accepted);
        WakeUpOptions commit;
        commit.commit_only = true;
        ASSERT_TRUE(controller.wakeUp(commit).ok);
        EXPECT_TRUE(admitAndComplete(controller.admission()->admit()));
        EXPECT_TRUE(admitAndComplete(controller.admission()->admit(true)));
        EXPECT_EQ(controller.activeAdmissionCount(), 0);
    }
}

TEST(SleepContinuationTest, ResourceFailureNeverReopensContinuationGate) {
    BoundSleepLifecycleController controller(true);
    SleepHooks                    hooks;
    hooks.drain                  = [](const SleepOptions&) { return true; };
    hooks.releaseKvMemoryBacking = [](const SleepOptions&) { return false; };
    controller.setHooks(hooks);
    EXPECT_FALSE(controller.sleep(gracefulOptions()).ok);
    EXPECT_EQ(controller.state(), SleepState::ERROR);
    EXPECT_FALSE(controller.admission()->admit(true).accepted);
    EXPECT_FALSE(controller.wakeUp().ok);
    EXPECT_FALSE(controller.admission()->admit(true).accepted);
}

TEST(SleepContinuationTest, DisabledContinuationIsTrackedAndCompletesExactlyOnce) {
    BoundSleepLifecycleController disabled(false);
    auto                          tracked = disabled.admission()->admit(true);
    EXPECT_TRUE(tracked.accepted);
    EXPECT_TRUE(static_cast<bool>(tracked.complete));
    EXPECT_EQ(disabled.activeAdmissionCount(), 1);
    tracked.complete();
    EXPECT_EQ(disabled.activeAdmissionCount(), 0);
    BoundSleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&) { return false; };
    controller.setHooks(hooks);
    EXPECT_FALSE(controller.sleep(gracefulOptions()).ok);
    auto first  = controller.admission()->admit(true);
    auto second = controller.admission()->admit(true);
    EXPECT_EQ(controller.activeAdmissionCount(), 2);
    auto duplicate = first.complete;
    first.complete();
    EXPECT_EQ(controller.activeAdmissionCount(), 1);
    duplicate();
    EXPECT_EQ(controller.activeAdmissionCount(), 1);
    second.complete();
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
}

namespace {

bool primePostQuiesceTest(BoundSleepLifecycleController& controller, bool coordinated) {
    if (!coordinated) {
        return true;
    }
    uint64_t round = 0;
    return controller.sleep(coordinatedDrain(controller, "late-cleanup")).ok
           && controller.quiesce({"late-cleanup", true, 0, 1000}, round).ok;
}

SleepResult preparePostQuiesceTest(BoundSleepLifecycleController& controller, bool coordinated, int64_t timeout_ms) {
    if (coordinated) {
        uint64_t round = 0;
        return controller.quiesce({"late-cleanup", false, 0, timeout_ms}, round);
    }
    auto options         = gracefulOptions();
    options.prepare_only = true;
    options.timeout_ms   = timeout_ms;
    return controller.sleep(options);
}

SleepResult commitPostQuiesceTest(BoundSleepLifecycleController& controller, bool coordinated) {
    auto options         = coordinated ? coordinatedDrain(controller, "late-cleanup") : gracefulOptions();
    options.prepare_only = options.drain_only = false;
    options.commit_only                       = true;
    return controller.sleep(options);
}

}  // namespace

TEST(SleepPostQuiesceDrainTest, LateTransferBlocksAckAndResourcesUntilRetryDrains) {
    for (bool coordinated : {false, true}) {
        SCOPED_TRACE(coordinated);
        BoundSleepLifecycleController controller(true);
        bool                          late_transfer = false;
        int                           pauses = 0, final_drains = 0, releases = 0;
        SleepHooks                    hooks;
        hooks.requiresCoordinatedQuiesce = coordinated;
        hooks.drain                      = [&](const SleepOptions& options) {
            if (pauses > 0) {
                ++final_drains;
                EXPECT_EQ(options.mode, "wait");
            }
            return !late_transfer;
        };
        auto pause = [&] {
            // A final async runner cleanup creates a connector write only
            // after the pre-freeze counters have already reached zero.
            if (++pauses == 1) {
                late_transfer = true;
            }
            return true;
        };
        hooks.quiesceEngine              = [&](const SleepOptions&) { return pause(); };
        hooks.quiesceEngineAtRound       = [&](uint64_t, int64_t) { return pause(); };
        hooks.synchronizeAndDeregisterMr = [&](const SleepOptions&) {
            ++releases;
            return true;
        };
        controller.setHooks(hooks);
        ASSERT_TRUE(primePostQuiesceTest(controller, coordinated));
        EXPECT_FALSE(preparePostQuiesceTest(controller, coordinated, 1000).ok);
        EXPECT_GT(final_drains, 0);
        EXPECT_EQ(controller.state(), SleepState::DRAINING);
        EXPECT_TRUE(controller.status().device_kv_cache_valid);
        EXPECT_FALSE(commitPostQuiesceTest(controller, coordinated).ok);
        EXPECT_EQ(releases, 0);
        EXPECT_FALSE(controller.admission()->admit(true).accepted);

        late_transfer              = false;
        const int old_final_drains = final_drains;
        ASSERT_TRUE(preparePostQuiesceTest(controller, coordinated, 1000).ok);
        EXPECT_GT(final_drains, old_final_drains);
        EXPECT_TRUE(commitPostQuiesceTest(controller, coordinated).ok);
        EXPECT_EQ(releases, 1);
    }
}

TEST(SleepPostQuiesceDrainTest, ThrowingLateDrainCannotPublishPreparedAck) {
    for (bool coordinated : {false, true}) {
        SCOPED_TRACE(coordinated);
        BoundSleepLifecycleController controller(true);
        bool                          paused   = false;
        int                           releases = 0;
        SleepHooks                    hooks;
        hooks.requiresCoordinatedQuiesce = coordinated;
        hooks.drain                      = [&](const SleepOptions&) {
            if (paused) {
                throw std::runtime_error("late connector cleanup failed");
            }
            return true;
        };
        hooks.quiesceEngine              = [&](const SleepOptions&) { return paused = true; };
        hooks.quiesceEngineAtRound       = [&](uint64_t, int64_t) { return paused = true; };
        hooks.synchronizeAndDeregisterMr = [&](const SleepOptions&) {
            ++releases;
            return true;
        };
        controller.setHooks(hooks);
        ASSERT_TRUE(primePostQuiesceTest(controller, coordinated));
        EXPECT_FALSE(preparePostQuiesceTest(controller, coordinated, 1000).ok);
        EXPECT_FALSE(commitPostQuiesceTest(controller, coordinated).ok);
        EXPECT_EQ(controller.state(), SleepState::DRAINING);
        EXPECT_EQ(releases, 0);
        EXPECT_NE(controller.status().last_error.find("quiesced"), std::string::npos);
    }
}

TEST(SleepPostQuiesceDrainTest, FinalDrainUsesRemainingQuiesceBudget) {
    for (bool coordinated : {false, true}) {
        SCOPED_TRACE(coordinated);
        BoundSleepLifecycleController controller(true);
        bool                          paused       = false;
        int64_t                       final_budget = -1;
        SleepHooks                    hooks;
        hooks.requiresCoordinatedQuiesce = coordinated;
        hooks.drain                      = [&](const SleepOptions& options) {
            if (paused) {
                final_budget = options.timeout_ms;
                EXPECT_EQ(options.mode, "wait");
            }
            return true;
        };
        auto pause = [&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            return paused = true;
        };
        hooks.quiesceEngine        = [&](const SleepOptions&) { return pause(); };
        hooks.quiesceEngineAtRound = [&](uint64_t, int64_t) { return pause(); };
        controller.setHooks(hooks);
        ASSERT_TRUE(primePostQuiesceTest(controller, coordinated));
        EXPECT_TRUE(preparePostQuiesceTest(controller, coordinated, 1000).ok);
        EXPECT_GT(final_budget, 0);
        EXPECT_LE(final_budget, 970);
    }
}

TEST(SleepPostQuiesceDrainTest, ExhaustedQuiesceBudgetCannotRestartDrainTimeout) {
    for (bool coordinated : {false, true}) {
        SCOPED_TRACE(coordinated);
        BoundSleepLifecycleController controller(true);
        bool                          paused       = false;
        int                           final_drains = 0, releases = 0;
        SleepHooks                    hooks;
        hooks.requiresCoordinatedQuiesce = coordinated;
        hooks.drain                      = [&](const SleepOptions&) {
            if (paused) {
                ++final_drains;
            }
            return true;
        };
        auto pause = [&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            return paused = true;
        };
        hooks.quiesceEngine              = [&](const SleepOptions&) { return pause(); };
        hooks.quiesceEngineAtRound       = [&](uint64_t, int64_t) { return pause(); };
        hooks.synchronizeAndDeregisterMr = [&](const SleepOptions&) {
            ++releases;
            return true;
        };
        controller.setHooks(hooks);
        ASSERT_TRUE(primePostQuiesceTest(controller, coordinated));
        EXPECT_FALSE(preparePostQuiesceTest(controller, coordinated, 1).ok);
        EXPECT_EQ(final_drains, 0);
        EXPECT_FALSE(commitPostQuiesceTest(controller, coordinated).ok);
        EXPECT_EQ(releases, 0);
    }
}

TEST(SleepPostQuiesceDrainTest, LateSuccessAfterDrainDeadlineCannotPublishPreparedAck) {
    for (bool coordinated : {false, true}) {
        SCOPED_TRACE(coordinated);
        BoundSleepLifecycleController controller(true);
        bool                          paused       = false;
        int                           final_drains = 0, releases = 0;
        SleepHooks                    hooks;
        hooks.requiresCoordinatedQuiesce = coordinated;
        hooks.drain                      = [&](const SleepOptions& options) {
            if (paused) {
                ++final_drains;
                std::this_thread::sleep_for(std::chrono::milliseconds(options.timeout_ms + 3));
            }
            return true;
        };
        hooks.quiesceEngine              = [&](const SleepOptions&) { return paused = true; };
        hooks.quiesceEngineAtRound       = [&](uint64_t, int64_t) { return paused = true; };
        hooks.synchronizeAndDeregisterMr = [&](const SleepOptions&) {
            ++releases;
            return true;
        };
        controller.setHooks(hooks);
        ASSERT_TRUE(primePostQuiesceTest(controller, coordinated));
        EXPECT_FALSE(preparePostQuiesceTest(controller, coordinated, 20).ok);
        EXPECT_EQ(final_drains, 1);
        EXPECT_FALSE(commitPostQuiesceTest(controller, coordinated).ok);
        EXPECT_EQ(controller.state(), SleepState::DRAINING);
        EXPECT_EQ(releases, 0);
    }
}

TEST(SleepPostQuiesceDrainTest, ZeroTimeoutUsesEngineDefaultAndAbortIsNotRepeated) {
    for (bool coordinated : {false, true}) {
        SCOPED_TRACE(coordinated);
        BoundSleepLifecycleController controller(true);
        bool                          paused       = false;
        int                           final_drains = 0;
        SleepHooks                    hooks;
        hooks.requiresCoordinatedQuiesce = coordinated;
        hooks.drain                      = [&](const SleepOptions& options) {
            if (paused) {
                ++final_drains;
                EXPECT_EQ(options.mode, "wait");
                EXPECT_GT(options.timeout_ms, 0);
                EXPECT_LE(options.timeout_ms, 60000);
            }
            return true;
        };
        hooks.quiesceEngine        = [&](const SleepOptions&) { return paused = true; };
        hooks.quiesceEngineAtRound = [&](uint64_t, int64_t) { return paused = true; };
        controller.setHooks(hooks);
        ASSERT_TRUE(primePostQuiesceTest(controller, coordinated));
        if (coordinated) {
            EXPECT_TRUE(preparePostQuiesceTest(controller, coordinated, 0).ok);
        } else {
            auto options         = gracefulOptions();
            options.mode         = "abort";
            options.prepare_only = true;
            options.timeout_ms   = 0;
            EXPECT_TRUE(controller.sleep(options).ok);
        }
        EXPECT_EQ(final_drains, 1);
    }
}

}  // namespace rtp_llm
