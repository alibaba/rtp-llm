#include "rtp_llm/cpp/engine_base/schedulers/SchedulerAdmission.h"

#include <atomic>
#include <future>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

namespace rtp_llm {

TEST(SchedulerBaseLifecycleTest, LedgerOwnsAdmissionUntilExplicitFinalCleanup) {
    SchedulerAdmission    ledger;
    std::function<void()> complete;
    {
        auto admitted = ledger.admit();
        ASSERT_TRUE(admitted.accepted);
        complete = admitted.complete;
    }
    EXPECT_EQ(ledger.activeCount(), 1);
    complete();
    EXPECT_EQ(ledger.activeCount(), 0);
    complete();
    EXPECT_EQ(ledger.activeCount(), 0);
}

TEST(SchedulerBaseLifecycleTest, CloseAllowsContinuationsUntilSeal) {
    SchedulerAdmission ledger;
    auto               root = ledger.admit();
    ledger.closeRoots();
    EXPECT_FALSE(ledger.admit().accepted);
    auto child = ledger.admit(true);
    ASSERT_TRUE(child.accepted);
    EXPECT_EQ(ledger.activeCount(), 2);
    ledger.sealContinuations();
    EXPECT_FALSE(ledger.admit(true).accepted);
    root.complete();
    EXPECT_EQ(ledger.activeCount(), 1);
    child.complete();
    EXPECT_EQ(ledger.activeCount(), 0);
    EXPECT_TRUE(ledger.reopen());
    auto next = ledger.admit();
    EXPECT_TRUE(next.accepted);
    next.complete();
}

TEST(SchedulerBaseLifecycleTest, TerminationCannotReopenButStillAllowsFinalContinuations) {
    SchedulerAdmission ledger;
    ledger.beginTermination();
    EXPECT_TRUE(ledger.terminating());
    EXPECT_FALSE(ledger.rootsOpen());
    EXPECT_FALSE(ledger.reopen());
    auto child = ledger.admit(true);
    ASSERT_TRUE(child.accepted);
    ledger.sealContinuations();
    EXPECT_FALSE(ledger.admit(true).accepted);
    child.complete();
    EXPECT_FALSE(ledger.reopen());
}

TEST(SchedulerBaseLifecycleTest, RetryAndConcurrentChildrenHaveDistinctLocalExecutionIdentities) {
    SchedulerAdmission ledger;
    auto               first = ledger.admit(true);
    auto               retry = ledger.admit(true);
    EXPECT_NE(first.execution_id, retry.execution_id);
    first.complete();
    auto reused_request_id = ledger.admit(true);
    EXPECT_NE(first.execution_id, reused_request_id.execution_id);
    first.complete();
    EXPECT_EQ(ledger.activeCount(), 2);
    retry.complete();
    reused_request_id.complete();
    EXPECT_EQ(ledger.activeCount(), 0);
}

TEST(SchedulerBaseLifecycleTest, LateCompletionCannotTouchDestroyedOrReplacementScheduler) {
    std::function<void()> late;
    {
        SchedulerAdmission old;
        late = old.admit().complete;
    }
    SchedulerAdmission replacement;
    auto               active = replacement.admit();
    late();
    EXPECT_EQ(replacement.activeCount(), 1);
    active.complete();
}

TEST(SchedulerBaseLifecycleTest, ConcurrentDuplicateCompletionRemovesExactlyOneRecord) {
    SchedulerAdmission       ledger;
    auto                     first  = ledger.admit();
    auto                     second = ledger.admit();
    std::promise<void>       start;
    auto                     ready = start.get_future().share();
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&] {
            ready.wait();
            first.complete();
        });
    }
    start.set_value();
    for (auto& worker : workers)
        worker.join();
    EXPECT_EQ(ledger.activeCount(), 1);
    second.complete();
}

TEST(SchedulerBaseLifecycleTest, ConcurrentSealCountsEveryWinningAdmission) {
    SchedulerAdmission ledger;
    ledger.closeRoots();
    std::promise<void>                      start;
    auto                                    ready = start.get_future().share();
    std::vector<SchedulerAdmission::Result> results(64);
    std::vector<std::thread>                workers;
    for (size_t i = 0; i < results.size(); ++i) {
        workers.emplace_back([&, i] {
            ready.wait();
            results[i] = ledger.admit(true);
        });
    }
    start.set_value();
    ledger.sealContinuations();
    for (auto& worker : workers)
        worker.join();
    size_t accepted = 0;
    for (const auto& result : results)
        accepted += result.accepted;
    EXPECT_EQ(ledger.activeCount(), accepted);
    EXPECT_FALSE(ledger.admit(true).accepted);
    for (const auto& result : results)
        if (result.complete)
            result.complete();
    EXPECT_EQ(ledger.activeCount(), 0);
}

TEST(SchedulerBaseLifecycleTest, RepeatedCyclesReclaimAllRecords) {
    SchedulerAdmission ledger;
    for (int i = 0; i < 100; ++i) {
        auto root = ledger.admit();
        ledger.closeRoots();
        auto child = ledger.admit(true);
        root.complete();
        ledger.sealContinuations();
        EXPECT_EQ(ledger.activeCount(), 1);
        child.complete();
        EXPECT_EQ(ledger.activeCount(), 0);
        EXPECT_TRUE(ledger.reopen());
    }
}

}  // namespace rtp_llm
