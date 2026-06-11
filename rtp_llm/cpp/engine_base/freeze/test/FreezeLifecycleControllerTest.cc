#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/engine_base/freeze/FreezeLifecycleController.h"

namespace rtp_llm {

namespace {

FreezeOptions gracefulOptions() {
    FreezeOptions opt;
    opt.mode             = "graceful";
    opt.drain_timeout_ms = 1000;
    return opt;
}

}  // namespace

TEST(FreezeLifecycleControllerTest, InitialStateIsRunning) {
    FreezeLifecycleController controller;
    EXPECT_EQ(controller.state(), FreezeState::RUNNING);
    EXPECT_TRUE(controller.admit());
    EXPECT_EQ(controller.freezeEpoch(), 0);

    const auto status = controller.status();
    EXPECT_EQ(status.state, FreezeState::RUNNING);
    EXPECT_TRUE(status.device_kv_cache_valid);
    EXPECT_EQ(status.kv_memory_state, "ACTIVE");
}

TEST(FreezeLifecycleControllerTest, FreezeWithDefaultHooksReachesFrozen) {
    FreezeLifecycleController controller;
    const auto                result = controller.freeze(gracefulOptions());
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), FreezeState::FROZEN);
    EXPECT_FALSE(controller.admit());
    EXPECT_EQ(controller.freezeEpoch(), 1);

    const auto status = controller.status();
    EXPECT_EQ(status.kv_memory_state, "PAUSED");
    EXPECT_FALSE(status.device_kv_cache_valid);
    EXPECT_EQ(status.gpu_resource_state, "RELEASED");
}

TEST(FreezeLifecycleControllerTest, ResumeFromFrozenReachesRunning) {
    FreezeLifecycleController controller;
    ASSERT_TRUE(controller.freeze(gracefulOptions()).ok);

    const auto result = controller.resume();
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), FreezeState::RUNNING);
    EXPECT_TRUE(controller.admit());
    // epoch is bumped by freeze, not by resume.
    EXPECT_EQ(controller.freezeEpoch(), 1);
}

TEST(FreezeLifecycleControllerTest, FreezeIsIdempotent) {
    FreezeLifecycleController controller;
    ASSERT_TRUE(controller.freeze(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), FreezeState::FROZEN);

    const auto again = controller.freeze(gracefulOptions());
    EXPECT_TRUE(again.ok) << again.message;
    EXPECT_EQ(controller.state(), FreezeState::FROZEN);
    // Idempotent repeat must NOT bump the epoch.
    EXPECT_EQ(controller.freezeEpoch(), 1);
}

TEST(FreezeLifecycleControllerTest, ResumeIsIdempotent) {
    FreezeLifecycleController controller;
    EXPECT_TRUE(controller.resume().ok);  // RUNNING -> resume == no-op success
    EXPECT_EQ(controller.state(), FreezeState::RUNNING);
}

TEST(FreezeLifecycleControllerTest, EpochIsMonotonicAcrossCycles) {
    FreezeLifecycleController controller;
    for (int64_t i = 1; i <= 3; ++i) {
        ASSERT_TRUE(controller.freeze(gracefulOptions()).ok);
        EXPECT_EQ(controller.freezeEpoch(), i);
        ASSERT_TRUE(controller.resume().ok);
        EXPECT_EQ(controller.freezeEpoch(), i);
    }
}

TEST(FreezeLifecycleControllerTest, DrainTimeoutKeepsDraining) {
    FreezeLifecycleController controller;
    FreezeHooks               hooks;
    hooks.drain = [](const FreezeOptions&) { return false; };  // simulate timeout
    controller.setHooks(hooks);

    const auto result = controller.freeze(gracefulOptions());
    EXPECT_FALSE(result.ok);
    // Per design: graceful drain timeout keeps DRAINING, does not release GPU.
    EXPECT_EQ(controller.state(), FreezeState::DRAINING);
    EXPECT_TRUE(controller.status().device_kv_cache_valid);
}

TEST(FreezeLifecycleControllerTest, FreezeRetryFromDrainingCanComplete) {
    FreezeLifecycleController controller;
    std::atomic<bool>         busy{true};
    FreezeHooks               hooks;
    hooks.drain = [&busy](const FreezeOptions&) { return !busy.load(); };
    controller.setHooks(hooks);

    EXPECT_FALSE(controller.freeze(gracefulOptions()).ok);
    EXPECT_EQ(controller.state(), FreezeState::DRAINING);
    EXPECT_EQ(controller.freezeEpoch(), 1);

    busy             = false;
    const auto retry = controller.freeze(gracefulOptions());
    EXPECT_TRUE(retry.ok) << retry.message;
    EXPECT_EQ(controller.state(), FreezeState::FROZEN);
    EXPECT_EQ(controller.freezeEpoch(), 1);
}

TEST(FreezeLifecycleControllerTest, FreezeRetryFromDrainingCanEscalateToForce) {
    FreezeLifecycleController controller;
    std::atomic<int>          force_seen{0};
    FreezeHooks               hooks;
    hooks.drain = [&force_seen](const FreezeOptions& opt) {
        if (opt.force || opt.mode == "force") {
            force_seen++;
            return true;
        }
        return false;
    };
    controller.setHooks(hooks);

    EXPECT_FALSE(controller.freeze(gracefulOptions()).ok);
    EXPECT_EQ(controller.state(), FreezeState::DRAINING);

    FreezeOptions force;
    force.mode             = "force";
    force.force            = true;
    force.drain_timeout_ms = 1000;
    const auto retry       = controller.freeze(force);
    EXPECT_TRUE(retry.ok) << retry.message;
    EXPECT_EQ(controller.state(), FreezeState::FROZEN);
    EXPECT_EQ(controller.freezeEpoch(), 1);
    EXPECT_EQ(force_seen.load(), 1);
}

TEST(FreezeLifecycleControllerTest, FreezeHookFailureGoesToError) {
    FreezeLifecycleController controller;
    FreezeHooks               hooks;
    hooks.pauseKvMemory = [](const FreezeOptions&) { return false; };
    controller.setHooks(hooks);

    const auto result = controller.freeze(gracefulOptions());
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), FreezeState::ERROR);
    EXPECT_FALSE(controller.admit());
    EXPECT_FALSE(controller.status().last_error.empty());
}

TEST(FreezeLifecycleControllerTest, ResumeFailureRevertsToFrozen) {
    FreezeLifecycleController controller;
    FreezeHooks               hooks;
    hooks.warmupAndHealthCheck = []() { return false; };
    controller.setHooks(hooks);

    ASSERT_TRUE(controller.freeze(gracefulOptions()).ok);
    const auto result = controller.resume();
    EXPECT_FALSE(result.ok);
    // Per design: resume failure keeps FROZEN, never half-available.
    EXPECT_EQ(controller.state(), FreezeState::FROZEN);
    EXPECT_FALSE(controller.admit());
}

TEST(FreezeLifecycleControllerTest, ResumeFromErrorIsAllowedAsRecovery) {
    FreezeLifecycleController controller;
    FreezeHooks               hooks;
    hooks.pauseKvMemory = [](const FreezeOptions&) { return false; };
    controller.setHooks(hooks);
    ASSERT_FALSE(controller.freeze(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), FreezeState::ERROR);

    controller.setHooks(FreezeHooks{});  // recovery: hooks behave now
    const auto result = controller.resume();
    EXPECT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), FreezeState::RUNNING);
}

TEST(FreezeLifecycleControllerTest, ResumeRejectedWhileDraining) {
    FreezeLifecycleController controller;
    FreezeHooks               hooks;
    hooks.drain = [](const FreezeOptions&) { return false; };
    controller.setHooks(hooks);
    ASSERT_FALSE(controller.freeze(gracefulOptions()).ok);
    ASSERT_EQ(controller.state(), FreezeState::DRAINING);

    const auto result = controller.resume();
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), FreezeState::DRAINING);
}

TEST(FreezeLifecycleControllerTest, StatusExposesLiveCounters) {
    FreezeLifecycleController controller;
    FreezeHooks               hooks;
    hooks.activeRequestCount       = []() { return 7; };
    hooks.activeCacheTransferCount = []() { return 3; };
    controller.setHooks(hooks);

    const auto status = controller.status();
    EXPECT_EQ(status.active_request_count, 7);
    EXPECT_EQ(status.active_cache_transfer_count, 3);
}

TEST(FreezeLifecycleControllerTest, ConcurrentFreezeResumeIsSerializedAndConsistent) {
    FreezeLifecycleController controller;
    std::atomic<int>          ok_freezes{0};

    std::vector<std::thread> threads;
    threads.reserve(8);
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&controller, &ok_freezes]() {
            if (controller.freeze(gracefulOptions()).ok) {
                ok_freezes.fetch_add(1);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    // All callers either performed or idempotently observed the freeze.
    EXPECT_EQ(ok_freezes.load(), 8);
    EXPECT_EQ(controller.state(), FreezeState::FROZEN);
    // Exactly one real freeze happened.
    EXPECT_EQ(controller.freezeEpoch(), 1);
}

}  // namespace rtp_llm
