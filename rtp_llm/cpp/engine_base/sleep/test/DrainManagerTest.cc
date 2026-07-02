#include "rtp_llm/cpp/engine_base/sleep/DrainManager.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {

namespace {

int64_t elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

class DrainManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        manager_.setPollIntervalMs(1);
    }

    DrainManager manager_;
};

// §3 M3: drained is true only when every registered counter reads zero.
TEST_F(DrainManagerTest, DrainedOnlyWhenAllCountersZero) {
    // No counters registered yet: trivially drained.
    EXPECT_TRUE(manager_.drained());

    std::atomic<size_t> scheduler_onflight{0};
    std::atomic<size_t> rpc_onflight{0};
    std::atomic<size_t> connector_inflight{0};
    manager_.registerCounter("scheduler_onflight", [&]() { return scheduler_onflight.load(); });
    manager_.registerCounter("rpc_onflight", [&]() { return rpc_onflight.load(); });
    manager_.registerCounter(
        "connector_inflight", [&]() { return connector_inflight.load(); }, DrainManager::CounterKind::CACHE_TRANSFER);

    EXPECT_TRUE(manager_.drained());

    // Any single non-zero counter flips drained to false.
    scheduler_onflight = 1;
    EXPECT_FALSE(manager_.drained());
    scheduler_onflight = 0;
    EXPECT_TRUE(manager_.drained());

    rpc_onflight = 3;
    EXPECT_FALSE(manager_.drained());
    connector_inflight = 2;
    EXPECT_FALSE(manager_.drained());

    rpc_onflight = 0;
    EXPECT_FALSE(manager_.drained());
    connector_inflight = 0;
    EXPECT_TRUE(manager_.drained());
}

TEST_F(DrainManagerTest, RegisterReplaceAndUnregister) {
    std::atomic<size_t> count{5};
    manager_.registerCounter("frontend_active", [&]() { return count.load(); });
    EXPECT_FALSE(manager_.drained());

    // Re-registering the same name replaces the provider.
    manager_.registerCounter("frontend_active", []() { return size_t(0); });
    EXPECT_TRUE(manager_.drained());

    manager_.registerCounter("frontend_active", [&]() { return count.load(); });
    EXPECT_FALSE(manager_.drained());

    manager_.unregisterCounter("frontend_active");
    EXPECT_TRUE(manager_.drained());

    // Null provider must be rejected (not registered, not crash).
    manager_.registerCounter("null_provider", nullptr);
    EXPECT_TRUE(manager_.drained());
}

// §3 M3: graceful drain timeout returns false (caller stays DRAINING).
TEST_F(DrainManagerTest, WaitDrainedTimesOutWhileBusy) {
    std::atomic<size_t> inflight{1};
    manager_.registerCounter("scheduler_onflight", [&]() { return inflight.load(); });

    const auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(manager_.waitDrained(50));
    const int64_t elapsed = elapsedMs(start);
    EXPECT_GE(elapsed, 50);
    EXPECT_LT(elapsed, 2000);  // returned promptly after timeout, not stuck

    // timeout_ms <= 0 means a single immediate check.
    EXPECT_FALSE(manager_.waitDrained(0));
    EXPECT_FALSE(manager_.waitDrained(-1));
    inflight = 0;
    EXPECT_TRUE(manager_.waitDrained(0));
}

TEST_F(DrainManagerTest, WaitDrainedReturnsPromptlyAfterCountersReachZero) {
    std::atomic<size_t> inflight{2};
    manager_.registerCounter("rpc_onflight", [&]() { return inflight.load(); });

    std::thread worker([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        inflight = 0;
        manager_.notifyDrainProgress();
    });

    const auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(manager_.waitDrained(10000));
    // Must return way before the 10s timeout once counters hit zero.
    EXPECT_LT(elapsedMs(start), 5000);
    worker.join();
}

// §3 M3: force invokes the injected cancel callback (which cancels
// non-streaming requests only) and then keeps waiting for full drain;
// streaming requests finish naturally.
TEST_F(DrainManagerTest, ForceDrainInvokesCancelAndKeepsWaitingForStreaming) {
    std::atomic<size_t> non_streaming{3};
    std::atomic<size_t> streaming{1};
    std::atomic<int>    cancel_called{0};

    manager_.registerCounter("non_streaming_requests", [&]() { return non_streaming.load(); });
    manager_.registerCounter("streaming_requests", [&]() { return streaming.load(); });
    // Cancel callback provider is responsible for the streaming exemption:
    // it only cancels non-streaming requests.
    manager_.setCancelCallback([&]() {
        cancel_called++;
        non_streaming = 0;
    });

    std::thread streaming_finisher([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        streaming = 0;  // streaming request ends naturally
        manager_.notifyDrainProgress();
    });

    SleepOptions opt;
    opt.mode             = "abort";
    opt.timeout_ms = 10000;
    EXPECT_TRUE(manager_.drain(opt));

    EXPECT_EQ(cancel_called.load(), 1);
    EXPECT_EQ(non_streaming.load(), 0u);
    EXPECT_EQ(streaming.load(), 0u);
    streaming_finisher.join();
}

TEST_F(DrainManagerTest, ForceDrainWithoutCancelCallbackStillWaits) {
    std::atomic<size_t> inflight{1};
    manager_.registerCounter("rpc_onflight", [&]() { return inflight.load(); });

    SleepOptions opt;
    opt.mode             = "abort";
    opt.timeout_ms = 30;
    // No cancel callback injected: force degrades to graceful wait and times out.
    EXPECT_FALSE(manager_.drain(opt));

    inflight = 0;
    EXPECT_TRUE(manager_.drain(opt));
}

TEST_F(DrainManagerTest, GracefulDrainDoesNotInvokeCancel) {
    std::atomic<int> cancel_called{0};
    manager_.setCancelCallback([&]() { cancel_called++; });

    SleepOptions opt;
    opt.mode             = "wait";
    opt.timeout_ms = 10;
    EXPECT_TRUE(manager_.drain(opt));
    EXPECT_EQ(cancel_called.load(), 0);
}

// §3 M3 + M1 status(): aggregate values reported per counter kind.
TEST_F(DrainManagerTest, AggregateCountsByKind) {
    std::atomic<size_t> frontend{2};
    std::atomic<size_t> scheduler{3};
    std::atomic<size_t> connector{4};
    std::atomic<size_t> cache_store{5};

    manager_.registerCounter("frontend_active", [&]() { return frontend.load(); }, DrainManager::CounterKind::REQUEST);
    manager_.registerCounter(
        "scheduler_onflight", [&]() { return scheduler.load(); }, DrainManager::CounterKind::REQUEST);
    manager_.registerCounter(
        "connector_inflight", [&]() { return connector.load(); }, DrainManager::CounterKind::CACHE_TRANSFER);
    manager_.registerCounter(
        "cache_store_active_transfers",
        [&]() { return cache_store.load(); },
        DrainManager::CounterKind::CACHE_TRANSFER);

    EXPECT_EQ(manager_.activeRequestCount(), 5);
    EXPECT_EQ(manager_.activeCacheTransferCount(), 9);

    frontend    = 0;
    scheduler   = 0;
    connector   = 0;
    cache_store = 1;
    EXPECT_EQ(manager_.activeRequestCount(), 0);
    EXPECT_EQ(manager_.activeCacheTransferCount(), 1);
    EXPECT_FALSE(manager_.drained());  // cache transfer still in flight blocks drain

    cache_store = 0;
    EXPECT_TRUE(manager_.drained());
}

// DrainManager as M1's SleepHooks drain provider.
TEST_F(DrainManagerTest, InstallHooksDrivesSleepLifecycleController) {
    std::atomic<size_t> requests{1};
    std::atomic<size_t> transfers{2};
    std::atomic<int>    cancel_called{0};
    manager_.registerCounter("rpc_onflight", [&]() { return requests.load(); }, DrainManager::CounterKind::REQUEST);
    manager_.registerCounter(
        "connector_inflight", [&]() { return transfers.load(); }, DrainManager::CounterKind::CACHE_TRANSFER);
    manager_.setCancelCallback([&]() { cancel_called++; });

    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    manager_.installHooks(hooks);
    controller.setHooks(hooks);

    // Counters flow into controller status().
    auto status = controller.status();
    EXPECT_EQ(status.active_request_count, 1);
    EXPECT_EQ(status.active_cache_transfer_count, 2);

    // Graceful sleep with busy counters: drain hook fails, stays DRAINING.
    SleepOptions opt;
    opt.timeout_ms = 20;
    auto result          = controller.sleep(opt);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_EQ(cancel_called.load(), 0);

    // Counters reach zero: idempotent sleep retry now drains and sleeps.
    requests  = 0;
    transfers = 0;
    EXPECT_TRUE(controller.sleep(opt).ok);
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);

    status = controller.status();
    EXPECT_EQ(status.active_request_count, 0);
    EXPECT_EQ(status.active_cache_transfer_count, 0);
}

// §3 M3: concurrent register/unregister/query must be race-free.
TEST_F(DrainManagerTest, ConcurrentRegisterAndQuery) {
    std::atomic<bool>   stop{false};
    std::atomic<size_t> shared_count{1};

    std::vector<std::thread> threads;
    // Writers: register/replace/unregister counters concurrently.
    for (int w = 0; w < 4; ++w) {
        threads.emplace_back([&, w]() {
            for (int i = 0; i < 500; ++i) {
                const std::string name = "counter_" + std::to_string(w) + "_" + std::to_string(i % 7);
                manager_.registerCounter(
                    name,
                    [&]() { return shared_count.load(); },
                    (i % 2 == 0) ? DrainManager::CounterKind::REQUEST : DrainManager::CounterKind::CACHE_TRANSFER);
                if (i % 3 == 0) {
                    manager_.unregisterCounter(name);
                }
            }
        });
    }
    // Readers: drained / aggregates / short waits concurrently.
    for (int r = 0; r < 4; ++r) {
        threads.emplace_back([&]() {
            while (!stop.load()) {
                (void)manager_.drained();
                (void)manager_.activeRequestCount();
                (void)manager_.activeCacheTransferCount();
                (void)manager_.waitDrained(1);
            }
        });
    }
    // One thread toggling the cancel callback + invoking force cancel.
    threads.emplace_back([&]() {
        while (!stop.load()) {
            manager_.setCancelCallback([]() {});
            manager_.forceCancel();
        }
    });

    for (int w = 0; w < 4; ++w) {
        threads[w].join();
    }
    stop = true;
    for (size_t i = 4; i < threads.size(); ++i) {
        threads[i].join();
    }

    // Deterministic end state: all remaining counters read zero.
    shared_count = 0;
    EXPECT_TRUE(manager_.drained());
    EXPECT_EQ(manager_.activeRequestCount(), 0);
    EXPECT_EQ(manager_.activeCacheTransferCount(), 0);
}

}  // namespace rtp_llm
