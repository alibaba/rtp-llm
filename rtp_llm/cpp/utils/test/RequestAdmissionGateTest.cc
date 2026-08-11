#include "rtp_llm/cpp/utils/RequestAdmissionGate.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

TEST(RequestAdmissionGateTest, CloseRejectsNewAdmissionsAndIsIdempotent) {
    RequestAdmissionGate gate;
    auto                 permit = gate.tryAcquire();

    ASSERT_TRUE(permit);
    EXPECT_FALSE(gate.isClosed());

    gate.close();
    gate.close();

    EXPECT_TRUE(gate.isClosed());
    EXPECT_FALSE(gate.tryAcquire());
    EXPECT_FALSE(gate.waitFor(0ms));

    permit.reset();

    EXPECT_TRUE(gate.waitUntil(std::chrono::steady_clock::now()));
    EXPECT_FALSE(gate.tryAcquire());
}

TEST(RequestAdmissionGateTest, HeldPermitBlocksDrainAndFinalReleaseWakesWaiter) {
    RequestAdmissionGate gate;
    auto                 first = gate.tryAcquire();
    auto                 last  = gate.tryAcquire();
    ASSERT_TRUE(first);
    ASSERT_TRUE(last);

    gate.close();
    EXPECT_FALSE(gate.waitUntil(std::chrono::steady_clock::now()));

    std::promise<void> waiter_started;
    auto               waiter_started_future = waiter_started.get_future();
    std::promise<bool> waiter_result;
    auto               waiter_result_future = waiter_result.get_future();
    std::thread waiter([&gate, &waiter_started, &waiter_result]() {
        waiter_started.set_value();
        waiter_result.set_value(gate.waitUntil(std::chrono::steady_clock::now() + 5s));
    });

    waiter_started_future.wait();
    first.reset();
    EXPECT_FALSE(gate.waitFor(0ms));

    last.reset();
    EXPECT_TRUE(waiter_result_future.get());
    waiter.join();
}

TEST(RequestAdmissionGateTest, CloseAcquireRaceNeverAdmitsAfterClose) {
    constexpr size_t kThreadCount = 32;

    RequestAdmissionGate gate;
    std::atomic<size_t>  ready{0};
    std::atomic<size_t>  attempts{0};
    std::atomic<size_t>  late_admissions{0};
    std::atomic<bool>    start{false};
    std::atomic<bool>    close_returned{false};
    std::vector<std::thread> workers;
    workers.reserve(kThreadCount);

    for (size_t i = 0; i < kThreadCount; ++i) {
        workers.emplace_back([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            {
                auto permit = gate.tryAcquire();
                attempts.fetch_add(1, std::memory_order_release);
            }
            while (!close_returned.load(std::memory_order_acquire)) {
                auto permit = gate.tryAcquire();
            }

            if (gate.tryAcquire()) {
                late_admissions.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != kThreadCount) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    while (attempts.load(std::memory_order_acquire) < kThreadCount) {
        std::this_thread::yield();
    }

    gate.close();
    close_returned.store(true, std::memory_order_release);

    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_EQ(late_admissions.load(std::memory_order_relaxed), 0);
    EXPECT_TRUE(gate.waitFor(0ms));
}

TEST(RequestAdmissionGateTest, PermitMoveTransfersExactlyOneAdmission) {
    static_assert(!std::is_copy_constructible_v<RequestAdmissionGate::Permit>);
    static_assert(!std::is_copy_assignable_v<RequestAdmissionGate::Permit>);
    static_assert(std::is_nothrow_move_constructible_v<RequestAdmissionGate::Permit>);
    static_assert(std::is_nothrow_move_assignable_v<RequestAdmissionGate::Permit>);

    RequestAdmissionGate gate;
    auto                 first  = gate.tryAcquire();
    auto                 second = gate.tryAcquire();
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);

    RequestAdmissionGate::Permit moved(std::move(first));
    EXPECT_FALSE(first);
    EXPECT_TRUE(moved);

    second = std::move(moved);
    EXPECT_FALSE(moved);
    EXPECT_TRUE(second);

    gate.close();
    EXPECT_FALSE(gate.waitFor(0ms));

    moved.reset();
    EXPECT_FALSE(gate.waitFor(0ms));

    second.reset();
    second.reset();
    EXPECT_TRUE(gate.waitFor(0ms));
}

TEST(RequestAdmissionGateTest, PermitCanOutliveGate) {
    RequestAdmissionGate::Permit permit;
    {
        auto gate = std::make_unique<RequestAdmissionGate>();
        permit    = gate->tryAcquire();
        ASSERT_TRUE(permit);
        gate->close();
        EXPECT_FALSE(gate->waitFor(0ms));
    }

    permit.reset();
    permit.reset();
    EXPECT_FALSE(permit);
}

}  // namespace
}  // namespace rtp_llm
