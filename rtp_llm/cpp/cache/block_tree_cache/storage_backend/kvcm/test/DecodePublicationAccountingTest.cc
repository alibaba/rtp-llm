#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DecodePublicationAccounting.h"
#include "rtp_llm/cpp/utils/DecodePublicationLifecycle.h"
#include <gtest/gtest.h>
#include <future>
#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtp_llm::kvcm {
namespace {
using Accounting = DecodePublicationAccounting;
using Clock      = Accounting::Clock;

TEST(DecodePublicationAccountingTest, BudgetsCoverLeaseCopiesAndFrozenPages) {
    auto state = std::make_shared<Accounting>(100, 2);
    auto first = state->tryAcquire(60, Clock::now());
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(state->tryAcquire(41, Clock::now()), nullptr);
    auto second = state->tryAcquire(40, Clock::now());
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(state->tryAcquire(0, Clock::now()), nullptr);
    first->onQueued();
    second->onRunning();
    auto view = state->snapshot();
    EXPECT_EQ(view.bytes, 100u);
    EXPECT_EQ(view.jobs, 2u);
    EXPECT_EQ(view.queued, 1u);
    EXPECT_EQ(view.running, 1u);
    auto copy = first;
    first.reset();
    EXPECT_EQ(state->snapshot().bytes, 100u);
    copy->onQuarantined();
    copy->onRunning();
    view = state->snapshot();
    EXPECT_EQ(view.quarantined, 1u);
    EXPECT_EQ(view.running, 1u);
    EXPECT_EQ(view.queued, 0u);
    second.reset();
    EXPECT_EQ(state->snapshot().bytes, 60u);
    copy.reset();
    EXPECT_EQ(state->snapshot().bytes, 0u);
    EXPECT_EQ(state->snapshot().jobs, 0u);
}

TEST(DecodePublicationAccountingTest, RetriesAndQuarantineKeepFirstAdmissionAge) {
    auto state = std::make_shared<Accounting>(100, 2);
    auto start = Clock::now();
    auto lease = state->tryAcquire(60, start, start);
    ASSERT_NE(lease, nullptr);
    lease->onQueued();
    lease->onRunning();
    lease->onQueued();
    lease->onQuarantined();
    EXPECT_EQ(state->snapshot(start + std::chrono::seconds(7)).oldest_age_us, 7000000);
    EXPECT_EQ(state->snapshot().quarantined, 1u);
    lease.reset();
    EXPECT_EQ(state->snapshot(start + std::chrono::seconds(10)).oldest_age_us, 0);
}

TEST(DecodePublicationAccountingTest, OutcomeIsOncePerAdmissionAndIncludesEosQueueTime) {
    std::vector<std::pair<std::string, std::optional<int64_t>>> events;
    auto                                                        state =
        std::make_shared<Accounting>(100, 2, Accounting::SnapshotObserver{}, [&](const char* result, auto latency) {
            events.emplace_back(result, latency);
        });
    auto lease = state->tryAcquire(60, Clock::now() - std::chrono::seconds(2));
    ASSERT_NE(lease, nullptr);
    lease->recordOutcome("committed", true);
    lease->recordOutcome("committed", true);
    lease.reset();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].first, "committed");
    ASSERT_TRUE(events[0].second);
    EXPECT_GE(*events[0].second, 2000000);
    auto cancelled = state->tryAcquire(60, Clock::now());
    cancelled.reset();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[1].first, "failed");
    EXPECT_FALSE(events[1].second);
}

TEST(DecodePublicationAccountingTest, IdleSamplerObservesStuckAgeWithoutInferenceAndStops) {
    std::promise<int64_t> observed;
    auto                  result = observed.get_future();
    std::once_flag        once;
    auto                  state = std::make_shared<Accounting>(100, 2, [&](const auto& value) {
        if (value.jobs == 1)
            std::call_once(once, [&] { observed.set_value(value.oldest_age_us); });
    });
    auto                  lease = state->tryAcquire(60, Clock::now(), Clock::now() - std::chrono::seconds(3));
    state->startSampling();
    ASSERT_EQ(result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_GE(result.get(), 3000000);
    state->stopSampling();
    lease.reset();
    EXPECT_EQ(state->snapshot().bytes, 0u);
}

TEST(DecodePublicationAccountingTest, TelemetryExceptionsCannotLeakAdmission) {
    auto state = std::make_shared<Accounting>(100, 2, Accounting::SnapshotObserver{}, [](const char*, auto) {
        throw std::runtime_error("telemetry unavailable");
    });
    auto lease = state->tryAcquire(60, Clock::now());
    ASSERT_NE(lease, nullptr);
    lease->recordOutcome("committed", true);
    lease.reset();
    EXPECT_EQ(state->snapshot().jobs, 0u);
    EXPECT_EQ(state->snapshot().bytes, 0u);
}

TEST(DecodePublicationAccountingTest, ConcurrentAdmissionCannotOversubscribeBudgets) {
    auto                     state = std::make_shared<Accounting>(32, 3);
    std::mutex               mutex;
    std::condition_variable  cv;
    size_t                   attempted = 0;
    bool                     release   = false;
    std::atomic<size_t>      accepted{0};
    std::vector<std::thread> workers;
    for (size_t i = 0; i < 16; ++i)
        workers.emplace_back([&] {
            auto lease = state->tryAcquire(8, Clock::now());
            if (lease)
                ++accepted;
            std::unique_lock<std::mutex> lock(mutex);
            ++attempted;
            cv.notify_all();
            cv.wait(lock, [&] { return release; });
        });
    bool all_attempted;
    {
        std::unique_lock<std::mutex> lock(mutex);
        all_attempted = cv.wait_for(lock, std::chrono::seconds(3), [&] { return attempted == 16; });
        EXPECT_LE(state->snapshot().bytes, 24u);
        EXPECT_LE(state->snapshot().jobs, 3u);
        release = true;
    }
    cv.notify_all();
    for (auto& worker : workers)
        worker.join();
    EXPECT_TRUE(all_attempted);
    EXPECT_EQ(accepted.load(), 3u);
    EXPECT_EQ(state->snapshot().jobs, 0u);
    EXPECT_EQ(state->snapshot().bytes, 0u);
}

TEST(DecodePublicationAccountingTest, DelayedRetryAfterBudgetRefusalKeepsOriginalEosAndReportsSkipOnce) {
    std::vector<std::pair<std::string, std::optional<int64_t>>> events;
    auto                                                        state =
        std::make_shared<Accounting>(100, 1, Accounting::SnapshotObserver{}, [&](const char* result, auto latency) {
            events.emplace_back(result, latency);
        });
    auto occupied = state->tryAcquire(100, std::nullopt);
    ASSERT_NE(occupied, nullptr);
    DecodePublicationLifecycle lifecycle;
    const auto                 eos = Clock::now() - std::chrono::seconds(2);
    lifecycle.recordEos(eos);
    for (size_t retry = 0; retry < 2; ++retry) {
        EXPECT_EQ(state->tryAcquire(10, lifecycle.eosAt()), nullptr);
        if (lifecycle.recordFirstBudgetSkip())
            state->recordBudgetSkip();
        lifecycle.recordEos(Clock::now());  // Duplicate/fallback cannot reset EOS.
    }
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].first, "budget_skipped");
    EXPECT_EQ(lifecycle.eosAt(), eos);
    occupied.reset();
    auto retried = state->tryAcquire(10, lifecycle.eosAt());
    ASSERT_NE(retried, nullptr);
    retried->recordOutcome("committed", true);
    retried.reset();
    ASSERT_EQ(events.size(), 3u);  // skip, unrelated cancelled occupant, commit.
    EXPECT_EQ(events.back().first, "committed");
    ASSERT_TRUE(events.back().second);
    EXPECT_GE(*events.back().second, 2000000);
}

TEST(DecodePublicationAccountingTest, MissingActualEosNeverFabricatesLatencyAtAdmission) {
    std::optional<int64_t> observed;
    bool                   acknowledged = false;
    auto                   state =
        std::make_shared<Accounting>(100, 1, Accounting::SnapshotObserver{}, [&](const char* result, auto latency) {
            acknowledged = std::string(result) == "committed";
            observed     = latency;
        });
    auto lease = state->tryAcquire(10, std::nullopt);
    ASSERT_NE(lease, nullptr);
    lease->recordOutcome("committed", true);
    EXPECT_TRUE(acknowledged);
    EXPECT_FALSE(observed);
}
}  // namespace
TEST(DecodePublicationAccountingTest, StopAdmissionPreservesExistingLeaseUntilItIsReleased) {
    auto state = std::make_shared<DecodePublicationAccounting>(128, 2);
    auto lease = state->tryAcquire(64, std::nullopt);
    ASSERT_TRUE(lease);
    state->stopAdmission();
    EXPECT_FALSE(state->tryAcquire(1, std::nullopt));
    EXPECT_EQ(state->snapshot().bytes, 64u);
    lease.reset();
    EXPECT_EQ(state->snapshot().bytes, 0u);
    EXPECT_FALSE(state->tryAcquire(1, std::nullopt));
}

}  // namespace rtp_llm::kvcm
