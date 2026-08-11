#include "rtp_llm/cpp/model_rpc/RemoteLoadLeaseRetainer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

RemoteLoadLeaseRetainer::Config testConfig(size_t max_jobs = 4) {
    return RemoteLoadLeaseRetainer::Config{
        max_jobs,
        1ms,
        5ms,
        1s,
        1,
    };
}

template<typename Predicate>
bool waitUntil(Predicate&& predicate, std::chrono::milliseconds timeout = 1s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return predicate();
        }
        std::this_thread::yield();
    }
    return true;
}

struct ReentrantDestructionProbe {
    RemoteLoadLeaseRetainer* retainer{nullptr};
    std::atomic<bool>        enabled{false};
    std::atomic<int>         calls{0};

    void run() {
        if (enabled.exchange(false)) {
            (void)retainer->activeJobsForTest();
            ++calls;
        }
    }
};

struct ReentrantLease {
    explicit ReentrantLease(std::shared_ptr<ReentrantDestructionProbe> probe): probe(std::move(probe)) {}
    ~ReentrantLease() {
        probe->run();
    }

    std::shared_ptr<ReentrantDestructionProbe> probe;
};

struct ReentrantQuiesce {
    explicit ReentrantQuiesce(std::shared_ptr<ReentrantDestructionProbe> probe): probe(std::move(probe)) {}
    ~ReentrantQuiesce() {
        probe->run();
    }

    bool operator()() const {
        return true;
    }

    std::shared_ptr<ReentrantDestructionProbe> probe;
};

struct ThrowOnCopyQuiesce {
    ThrowOnCopyQuiesce(std::shared_ptr<std::atomic<bool>> throw_on_copy,
                       std::shared_ptr<std::atomic<int>>  calls):
        throw_on_copy(std::move(throw_on_copy)), calls(std::move(calls)) {}

    ThrowOnCopyQuiesce(const ThrowOnCopyQuiesce& other):
        throw_on_copy(other.throw_on_copy), calls(other.calls) {
        if (throw_on_copy->load()) {
            throw std::runtime_error("unexpected quiesce callback copy");
        }
    }

    ThrowOnCopyQuiesce(ThrowOnCopyQuiesce&&) noexcept = default;

    bool operator()() const {
        ++*calls;
        return true;
    }

    std::shared_ptr<std::atomic<bool>> throw_on_copy;
    std::shared_ptr<std::atomic<int>>  calls;
};

TEST(RemoteLoadLeaseRetainerTest, NeverStartedTicketReleasesLeaseWithoutQuiesce) {
    std::atomic<int> quiesce_calls{0};
    auto             lease = std::make_shared<int>(1);
    std::weak_ptr<int> weak_lease = lease;
    RemoteLoadLeaseRetainer retainer(testConfig());

    auto ticket = retainer.reserve("allocation-a", lease, [&]() {
        ++quiesce_calls;
        return true;
    });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    lease.reset();
    ticket->reset();

    EXPECT_TRUE(weak_lease.expired());
    EXPECT_EQ(quiesce_calls.load(), 0);
    EXPECT_EQ(retainer.activeJobsForTest(), 0);
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, StartedTicketRetriesUntilQuiesced) {
    std::atomic<int> attempts{0};
    auto             lease = std::make_shared<int>(1);
    std::weak_ptr<int> weak_lease = lease;
    RemoteLoadLeaseRetainer retainer(testConfig());

    auto ticket = retainer.reserve("allocation-a", lease, [&]() { return ++attempts >= 3; });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    lease.reset();
    ticket->reset();

    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (!weak_lease.expired() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(weak_lease.expired());
    EXPECT_GE(attempts.load(), 3);
    EXPECT_EQ(retainer.activeJobsForTest(), 0);
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, ExplicitCompletionReleasesExactlyOnce) {
    struct CountedLease {
        explicit CountedLease(std::atomic<int>& destroyed): destroyed(destroyed) {}
        ~CountedLease() {
            ++destroyed;
        }
        std::atomic<int>& destroyed;
    };

    std::atomic<int> destroyed{0};
    auto             lease = std::make_shared<CountedLease>(destroyed);
    RemoteLoadLeaseRetainer retainer(testConfig());
    auto ticket = retainer.reserve("allocation-a", lease, []() { return true; });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    lease.reset();

    EXPECT_TRUE((*ticket)->complete());
    EXPECT_FALSE((*ticket)->complete());
    EXPECT_EQ(destroyed.load(), 1);
    ticket->reset();
    EXPECT_EQ(destroyed.load(), 1);
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, CapacityIsReservedBeforeTransferStarts) {
    RemoteLoadLeaseRetainer retainer(testConfig(/*max_jobs=*/1));
    auto first = retainer.reserve("allocation-a", std::make_shared<int>(1), []() { return true; });
    ASSERT_TRUE(first.ok()) << first.status();

    auto second = retainer.reserve("allocation-b", std::make_shared<int>(2), []() { return true; });
    ASSERT_FALSE(second.ok());
    EXPECT_EQ(second.status().code(), absl::StatusCode::kResourceExhausted);

    first->reset();
    auto retry = retainer.reserve("allocation-b", std::make_shared<int>(2), []() { return true; });
    ASSERT_TRUE(retry.ok()) << retry.status();
    retry->reset();
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, AbandonDestroysLeaseAndCallbackOutsideManagerMutex) {
    RemoteLoadLeaseRetainer retainer(testConfig());
    auto lease_probe = std::make_shared<ReentrantDestructionProbe>();
    auto callback_probe = std::make_shared<ReentrantDestructionProbe>();
    lease_probe->retainer = &retainer;
    callback_probe->retainer = &retainer;

    auto lease = std::make_shared<ReentrantLease>(lease_probe);
    RemoteLoadLeaseRetainer::Quiesce quiesce{ReentrantQuiesce(callback_probe)};
    auto ticket = retainer.reserve("allocation-a", lease, std::move(quiesce));
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    lease.reset();

    lease_probe->enabled = true;
    callback_probe->enabled = true;
    ticket->reset();

    EXPECT_EQ(lease_probe->calls.load(), 1);
    EXPECT_EQ(callback_probe->calls.load(), 1);
    EXPECT_EQ(retainer.activeJobsForTest(), 0);
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, CompletionDestroysCallbackOutsideManagerMutex) {
    RemoteLoadLeaseRetainer retainer(testConfig());
    auto callback_probe = std::make_shared<ReentrantDestructionProbe>();
    callback_probe->retainer = &retainer;

    RemoteLoadLeaseRetainer::Quiesce quiesce{ReentrantQuiesce(callback_probe)};
    auto ticket = retainer.reserve("allocation-a", std::make_shared<int>(1), std::move(quiesce));
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());

    callback_probe->enabled = true;
    EXPECT_TRUE((*ticket)->complete());
    EXPECT_EQ(callback_probe->calls.load(), 1);
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, WorkerDoesNotCopyCallbackDuringRetry) {
    auto throw_on_copy = std::make_shared<std::atomic<bool>>(false);
    auto calls = std::make_shared<std::atomic<int>>(0);
    auto lease = std::make_shared<int>(1);
    std::weak_ptr<int> weak_lease = lease;
    RemoteLoadLeaseRetainer retainer(testConfig());

    RemoteLoadLeaseRetainer::Quiesce quiesce{ThrowOnCopyQuiesce(throw_on_copy, calls)};
    auto ticket = retainer.reserve("allocation-a", lease, std::move(quiesce));
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    lease.reset();
    throw_on_copy->store(true);
    ticket->reset();

    EXPECT_TRUE(waitUntil([&]() { return weak_lease.expired(); }));
    EXPECT_EQ(calls->load(), 1);
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, ThrowingCallbackIsRetriedWithoutReleasingLease) {
    std::atomic<int>  attempts{0};
    std::atomic<bool> allow_success{false};
    auto              lease = std::make_shared<int>(1);
    std::weak_ptr<int> weak_lease = lease;
    RemoteLoadLeaseRetainer retainer(testConfig());

    auto ticket = retainer.reserve("allocation-a", lease, [&]() {
        if (++attempts == 1) {
            throw std::runtime_error("retry");
        }
        while (!allow_success.load()) {
            std::this_thread::yield();
        }
        return true;
    });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    lease.reset();
    ticket->reset();

    const bool retry_started = waitUntil([&]() { return attempts.load() >= 2; });
    EXPECT_TRUE(retry_started);
    EXPECT_FALSE(weak_lease.expired());
    allow_success = true;
    EXPECT_TRUE(waitUntil([&]() { return weak_lease.expired(); }));
    EXPECT_EQ(attempts.load(), 2);
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, BoundedStopTimeoutRetainsLeaseAndRejectsNewReservations) {
    std::atomic<bool> allow_quiesce{false};
    std::atomic<int>  failures{0};
    auto              lease = std::make_shared<int>(1);
    std::weak_ptr<int> weak_lease = lease;
    RemoteLoadLeaseRetainer retainer(testConfig(), [&](const std::string&) { ++failures; });

    auto ticket = retainer.reserve("allocation-a", lease, [&]() { return allow_quiesce.load(); });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    lease.reset();

    EXPECT_FALSE(retainer.stop(0ms));
    EXPECT_EQ(failures.load(), 0);
    EXPECT_FALSE(weak_lease.expired());
    auto rejected = retainer.reserve("allocation-b", std::make_shared<int>(2), []() { return true; });
    ASSERT_FALSE(rejected.ok());
    EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);

    allow_quiesce = true;
    ticket->reset();
    EXPECT_TRUE(waitUntil([&]() { return weak_lease.expired(); }));
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, BoundedStopTimeoutReturnsWithDefaultFailurePolicy) {
    EXPECT_EXIT(
        {
            std::atomic<bool> allow_quiesce{false};
            auto              lease = std::make_shared<int>(1);
            std::weak_ptr<int> weak_lease = lease;
            RemoteLoadLeaseRetainer retainer(testConfig());

            auto ticket = retainer.reserve("allocation-a", lease, [&]() { return allow_quiesce.load(); });
            if (!ticket.ok() || !(*ticket)->markStarted()) {
                std::exit(1);
            }
            lease.reset();
            if (retainer.stop(0ms) || weak_lease.expired() || retainer.activeJobsForTest() != 1) {
                std::exit(2);
            }

            allow_quiesce = true;
            ticket->reset();
            if (!retainer.stop(100ms) || !weak_lease.expired()) {
                std::exit(3);
            }
            std::exit(0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST(RemoteLoadLeaseRetainerTest, DestructorFailsClosedWithUnresolvedStartedLease) {
    auto config       = testConfig();
    config.stop_grace = 0ms;

    EXPECT_EXIT(
        {
            RemoteLoadLeaseRetainer retainer(config);
            auto ticket = retainer.reserve("allocation-a", std::make_shared<int>(1), []() { return false; });
            if (!ticket.ok() || !(*ticket)->markStarted()) {
                std::exit(1);
            }
            ticket->reset();
        },
        ::testing::KilledBySignal(SIGABRT),
        "");
}

TEST(RemoteLoadLeaseRetainerTest, ShutdownPreventsReservedTicketFromStarting) {
    std::atomic<bool> stop_result{false};
    RemoteLoadLeaseRetainer retainer(testConfig(/*max_jobs=*/2), [](const std::string&) {});
    auto ticket = retainer.reserve("allocation-a", std::make_shared<int>(1), []() { return true; });
    ASSERT_TRUE(ticket.ok()) << ticket.status();

    std::thread stopper([&]() { stop_result = retainer.stop(5s); });
    bool        stopping_observed = false;
    size_t      probe_id = 0;
    while (!stopping_observed) {
        auto probe = retainer.reserve(
            "probe-" + std::to_string(probe_id++), std::make_shared<int>(2), []() { return true; });
        if (!probe.ok()) {
            EXPECT_EQ(probe.status().code(), absl::StatusCode::kFailedPrecondition);
            stopping_observed = true;
        }
    }

    EXPECT_FALSE((*ticket)->markStarted());
    ticket->reset();
    stopper.join();
    EXPECT_TRUE(stop_result.load());
}

TEST(RemoteLoadLeaseRetainerTest, ConcurrentReservationsNeverExceedCapacity) {
    constexpr size_t kCapacity = 8;
    constexpr size_t kThreadCount = 32;
    RemoteLoadLeaseRetainer retainer(testConfig(kCapacity));
    std::atomic<size_t> ready{0};
    std::atomic<size_t> attempted{0};
    std::atomic<size_t> accepted{0};
    std::atomic<bool>   go{false};
    std::atomic<bool>   release{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);

    for (size_t i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&, i]() {
            ++ready;
            while (!go.load()) {
                std::this_thread::yield();
            }
            auto ticket = retainer.reserve(
                "allocation-" + std::to_string(i), std::make_shared<int>(1), []() { return true; });
            if (ticket.ok()) {
                ++accepted;
            }
            ++attempted;
            while (!release.load()) {
                std::this_thread::yield();
            }
        });
    }

    const bool all_ready = waitUntil([&]() { return ready.load() == kThreadCount; });
    go = true;
    const bool all_attempted = waitUntil([&]() { return attempted.load() == kThreadCount; });
    EXPECT_TRUE(all_ready);
    EXPECT_TRUE(all_attempted);
    EXPECT_EQ(accepted.load(), kCapacity);
    EXPECT_EQ(retainer.activeJobsForTest(), kCapacity);

    release = true;
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(retainer.activeJobsForTest(), 0);
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, DifferentJobsRunConcurrentlyUpToConfiguredLimit) {
    auto config         = testConfig(/*max_jobs=*/8);
    config.worker_count = 3;
    RemoteLoadLeaseRetainer retainer(config);

    std::mutex              mutex;
    std::condition_variable changed;
    size_t                  entered    = 0;
    size_t                  active     = 0;
    size_t                  max_active = 0;
    bool                    release    = false;

    for (size_t index = 0; index < config.max_jobs; ++index) {
        auto ticket = retainer.reserve(
            "allocation-" + std::to_string(index), std::make_shared<int>(1), [&]() {
                std::unique_lock<std::mutex> lock(mutex);
                ++entered;
                ++active;
                max_active = std::max(max_active, active);
                changed.notify_all();
                changed.wait(lock, [&]() { return release; });
                --active;
                return true;
            });
        ASSERT_TRUE(ticket.ok()) << ticket.status();
        ASSERT_TRUE((*ticket)->markStarted());
        ticket->reset();
    }

    {
        std::unique_lock<std::mutex> lock(mutex);
        const bool all_workers_entered =
            changed.wait_for(lock, 1s, [&]() { return entered == config.worker_count; });
        EXPECT_TRUE(all_workers_entered);
        EXPECT_EQ(active, config.worker_count);
        EXPECT_EQ(max_active, config.worker_count);
        EXPECT_EQ(retainer.activeJobsForTest(), config.max_jobs);
        release = true;
    }
    changed.notify_all();

    EXPECT_TRUE(waitUntil([&]() { return retainer.activeJobsForTest() == 0; }));
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_EQ(entered, config.max_jobs);
        EXPECT_EQ(active, 0);
        EXPECT_EQ(max_active, config.worker_count);
    }
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, SameJobRetriesNeverOverlapAcrossWorkers) {
    auto config         = testConfig(/*max_jobs=*/1);
    config.worker_count = 4;
    RemoteLoadLeaseRetainer retainer(config);

    std::mutex              mutex;
    std::condition_variable changed;
    size_t                  calls      = 0;
    size_t                  active     = 0;
    size_t                  max_active = 0;
    bool                    release_first_attempt = false;

    auto ticket = retainer.reserve("allocation-a", std::make_shared<int>(1), [&]() {
        std::unique_lock<std::mutex> lock(mutex);
        const auto                  call = ++calls;
        ++active;
        max_active = std::max(max_active, active);
        changed.notify_all();
        if (call == 1) {
            changed.wait(lock, [&]() { return release_first_attempt; });
        }
        --active;
        changed.notify_all();
        return call > 1;
    });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    ticket->reset();

    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(changed.wait_for(lock, 1s, [&]() { return calls == 1 && active == 1; }));
        EXPECT_EQ(max_active, 1);
        release_first_attempt = true;
    }
    changed.notify_all();

    EXPECT_TRUE(waitUntil([&]() { return retainer.activeJobsForTest() == 0; }));
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_EQ(calls, 2);
        EXPECT_EQ(active, 0);
        EXPECT_EQ(max_active, 1);
    }
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, BackgroundRetirementDestroysResourcesOutsideManagerMutex) {
    auto config         = testConfig();
    config.worker_count = 2;
    RemoteLoadLeaseRetainer retainer(config);
    auto lease_probe    = std::make_shared<ReentrantDestructionProbe>();
    auto callback_probe = std::make_shared<ReentrantDestructionProbe>();
    lease_probe->retainer    = &retainer;
    callback_probe->retainer = &retainer;

    auto lease = std::make_shared<ReentrantLease>(lease_probe);
    RemoteLoadLeaseRetainer::Quiesce quiesce{ReentrantQuiesce(callback_probe)};
    auto ticket = retainer.reserve("allocation-a", lease, std::move(quiesce));
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    lease.reset();

    lease_probe->enabled    = true;
    callback_probe->enabled = true;
    ticket->reset();

    EXPECT_TRUE(waitUntil([&]() {
        return lease_probe->calls.load() == 1 && callback_probe->calls.load() == 1;
    }));
    EXPECT_EQ(retainer.activeJobsForTest(), 0);
    EXPECT_TRUE(retainer.stop(100ms));
}

TEST(RemoteLoadLeaseRetainerTest, StopFromOwnWorkerReturnsWithoutSelfJoin) {
    auto config         = testConfig();
    config.worker_count = 2;
    RemoteLoadLeaseRetainer retainer(config);
    std::atomic<bool>        stop_called{false};
    std::atomic<bool>        stop_result{true};

    auto ticket = retainer.reserve("allocation-a", std::make_shared<int>(1), [&]() {
        stop_result = retainer.stop(100ms);
        stop_called = true;
        return true;
    });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    ticket->reset();

    EXPECT_TRUE(waitUntil([&]() { return stop_called.load() && retainer.activeJobsForTest() == 0; }));
    EXPECT_FALSE(stop_result.load());
    EXPECT_TRUE(retainer.stop(100ms));
}

}  // namespace
}  // namespace rtp_llm
