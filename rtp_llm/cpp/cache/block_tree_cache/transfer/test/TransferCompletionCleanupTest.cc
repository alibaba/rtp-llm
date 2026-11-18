#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionCleanup.h"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

using Cleanup = TransferCompletionCleanup;

TransferCompletionTarget localTarget() {
    return {false, {{0, "local"}}};
}

TransferCompletionTarget remoteTarget() {
    return {true, {{0, "worker-0:100"}, {1, "worker-1:100"}}};
}

TransferCompletionForgetResult doneResult(const std::vector<TransferExecutionTrace>& traces) {
    TransferCompletionForgetResult result;
    for (const auto& trace : traces) {
        TransferCompletionForgetEntryResult entry;
        entry.trace_id = trace;
        entry.all_done = true;
        result.entries.push_back(std::move(entry));
    }
    return result;
}

TEST(TransferCompletionCleanupTest, FailureDiagnosticsAreRateLimitedWithoutBlockingRetries) {
    auto cleanup = std::make_shared<Cleanup>();
    cleanup->prepare(localTarget(), uint64_t{99}).submit();
    size_t calls = 0;
    const Cleanup::Forget forget = [&](const auto&, const auto&, int) -> TransferCompletionForgetResult {
        ++calls;
        throw std::runtime_error("network unavailable");
    };
    EXPECT_TRUE(cleanup->drain(forget, 100).failed);
    const auto deadline = cleanup->next_failure_log_;
    EXPECT_GT(deadline, std::chrono::steady_clock::now());
    EXPECT_TRUE(cleanup->drain(forget, 100).failed);
    EXPECT_EQ(cleanup->next_failure_log_, deadline);
    EXPECT_EQ(calls, 2u);
    EXPECT_EQ(cleanup->size(), 1u);
    cleanup->next_failure_log_ = {};
    EXPECT_TRUE(cleanup->drain(forget, 100).failed);
    EXPECT_GT(cleanup->next_failure_log_, std::chrono::steady_clock::now());
    EXPECT_EQ(calls, 3u);
}

TEST(TransferCompletionCleanupTest, PreparedNodesRequireExplicitSubmissionAndMoveOnlyOnce) {
    static_assert(!std::is_copy_constructible_v<Cleanup::Prepared>);
    static_assert(std::is_nothrow_move_constructible_v<Cleanup::Prepared>);
    static_assert(std::is_nothrow_move_assignable_v<Cleanup::Prepared>);
    auto queue = std::make_shared<Cleanup>();
    auto prepared = queue->prepare(localTarget(), uint64_t{1});
    EXPECT_FALSE(prepared.empty());
    EXPECT_EQ(queue->size(), 0u);
    auto moved = std::move(prepared);
    EXPECT_TRUE(prepared.empty());
    prepared.submit();
    EXPECT_EQ(queue->size(), 0u);
    moved.submit();
    moved.submit();
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(queue->size(), 1u);
    {
        auto discarded = queue->prepare(localTarget(), uint64_t{2});
        EXPECT_FALSE(discarded.empty());
    }
    EXPECT_EQ(queue->size(), 1u);
    const auto summary = queue->drain([](const auto&, const auto& traces, int) {
        EXPECT_EQ(traces, (std::vector<TransferExecutionTrace>{uint64_t{1}}));
        return doneResult(traces);
    }, 10);
    EXPECT_EQ(summary.attempted, 1u);
    EXPECT_EQ(summary.forgotten, 1u);
    EXPECT_EQ(summary.pending, 0u);
    EXPECT_FALSE(summary.failed);
}

TEST(TransferCompletionCleanupTest, PreparationRejectsInvalidIdentityAndRoutingBeforeQueueing) {
    auto queue = std::make_shared<Cleanup>();
    EXPECT_THROW(queue->prepare(localTarget(), uint64_t{0}), std::invalid_argument);
    EXPECT_THROW(queue->prepare(localTarget(), std::string{}), std::invalid_argument);
    EXPECT_THROW(queue->prepare({}, uint64_t{1}), std::invalid_argument);
    EXPECT_THROW(queue->prepare({false, {{0, "local"}, {1, "other"}}}, uint64_t{1}), std::invalid_argument);
    EXPECT_THROW(queue->prepare({true, {{0, "a"}, {0, "b"}}}, uint64_t{1}), std::invalid_argument);
    EXPECT_THROW(queue->prepare({true, {{1, "a"}}}, uint64_t{1}), std::invalid_argument);
    EXPECT_THROW(queue->prepare({true, {{0, ""}}}, uint64_t{1}), std::invalid_argument);
    EXPECT_EQ(queue->size(), 0u);
}

TEST(TransferCompletionCleanupTest, BatchesByOriginalRoutingAndNeverExceedsProtocolLimit) {
    auto queue = std::make_shared<Cleanup>();
    constexpr size_t count = 2 * kTransferCompletionForgetBatchSize + 3;
    for (uint64_t trace = 1; trace <= count; ++trace) {
        auto target = remoteTarget();
        if (trace % 2 == 0)
            std::swap(target.peers[0], target.peers[1]);
        queue->prepare(std::move(target), trace).submit();
    }
    queue->prepare(localTarget(), std::string("1")).submit();
    queue->prepare({true, {{0, "another-worker:100"}}}, uint64_t{1}).submit();
    size_t calls = 0;
    size_t remote_batches = 0;
    const auto summary = queue->drain([&](const auto& target, const auto& traces, int timeout) {
        ++calls;
        EXPECT_EQ(timeout, 37);
        EXPECT_FALSE(traces.empty());
        EXPECT_LE(traces.size(), kTransferCompletionForgetBatchSize);
        for (size_t rank = 0; rank < target.peers.size(); ++rank)
            EXPECT_EQ(target.peers[rank].rank, rank);
        if (target.peers.size() == 2) {
            ++remote_batches;
            EXPECT_TRUE(target.remote);
            EXPECT_EQ(target.peers[0].address, "worker-0:100");
            EXPECT_EQ(target.peers[1].address, "worker-1:100");
        } else if (!target.remote) {
            EXPECT_EQ(traces, (std::vector<TransferExecutionTrace>{std::string("1")}));
            EXPECT_EQ(target.peers.front().address, "local");
        } else {
            EXPECT_EQ(traces, (std::vector<TransferExecutionTrace>{uint64_t{1}}));
            EXPECT_EQ(target.peers.front().address, "another-worker:100");
        }
        return doneResult(traces);
    }, 37);
    EXPECT_EQ(calls, 5u);
    EXPECT_EQ(remote_batches, 3u);
    EXPECT_EQ(summary.attempted, count + 2);
    EXPECT_EQ(summary.forgotten, count + 2);
    EXPECT_EQ(queue->size(), 0u);
}

TEST(TransferCompletionCleanupTest, PartialSuccessRemovesOnlyConfirmedEntriesAndRetriesNextDrain) {
    auto queue = std::make_shared<Cleanup>();
    const std::vector<TransferExecutionTrace> traces{uint64_t{1}, std::string("1"), uint64_t{2}};
    for (const auto& trace : traces)
        queue->prepare(localTarget(), trace).submit();
    size_t calls = 0;
    const auto partial = queue->drain([&](const auto&, const auto& batch, int) {
        ++calls;
        EXPECT_EQ(queue->size(), traces.size());
        EXPECT_EQ(batch, traces);
        auto result = doneResult(batch);
        result.error = "one key remains running and another response was lost";
        result.entries[1].all_done = false;
        result.entries[2].all_done = false;
        return result;
    }, 10);
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(partial.attempted, 3u);
    EXPECT_EQ(partial.forgotten, 1u);
    EXPECT_EQ(partial.pending, 2u);
    EXPECT_TRUE(partial.failed);
    const auto retry = queue->drain([](const auto&, const auto& batch, int) {
        EXPECT_EQ(batch, (std::vector<TransferExecutionTrace>{std::string("1"), uint64_t{2}}));
        return doneResult(batch);
    }, 10);
    EXPECT_EQ(retry.forgotten, 2u);
    EXPECT_EQ(retry.pending, 0u);
}

TEST(TransferCompletionCleanupTest, MissingOrMismatchedAcknowledgementsCannotRemoveEntries) {
    auto queue = std::make_shared<Cleanup>();
    queue->prepare(localTarget(), uint64_t{7}).submit();
    const auto missing = queue->drain([](const auto&, const auto&, int) {
        return TransferCompletionForgetResult{};
    }, 10);
    EXPECT_TRUE(missing.failed);
    EXPECT_EQ(missing.forgotten, 0u);
    EXPECT_EQ(missing.pending, 1u);
    const auto wrong_namespace = queue->drain([](const auto&, const auto&, int) {
        return doneResult({std::string("7")});
    }, 10);
    EXPECT_TRUE(wrong_namespace.failed);
    EXPECT_EQ(wrong_namespace.forgotten, 0u);
    EXPECT_EQ(wrong_namespace.pending, 1u);
    const auto complete = queue->drain([](const auto&, const auto& batch, int) { return doneResult(batch); }, 10);
    EXPECT_EQ(complete.forgotten, 1u);
    EXPECT_EQ(complete.pending, 0u);
}

TEST(TransferCompletionCleanupTest, CallbackExceptionsAndLostAcknowledgementsRemainRetryable) {
    auto queue = std::make_shared<Cleanup>();
    TransferTaskRegistry registry;
    ASSERT_TRUE(registry.begin(uint64_t{11}));
    ASSERT_TRUE(registry.finish(uint64_t{11}, true));
    queue->prepare(localTarget(), uint64_t{11}).submit();
    size_t calls = 0;
    const auto lost = queue->drain([&](const auto&, const auto&, int) -> TransferCompletionForgetResult {
        ++calls;
        EXPECT_EQ(registry.forget(uint64_t{11}), TransferTaskRegistry::ForgetResult::DONE);
        throw std::runtime_error("response lost after worker deletion");
    }, 10);
    EXPECT_EQ(calls, 1u);
    EXPECT_TRUE(lost.failed);
    EXPECT_EQ(lost.pending, 1u);
    EXPECT_EQ(registry.size(), 0u);
    const auto retry = queue->drain([&](const auto&, const auto& batch, int) {
        EXPECT_EQ(registry.forget(uint64_t{11}), TransferTaskRegistry::ForgetResult::DONE);
        return doneResult(batch);
    }, 10);
    EXPECT_EQ(retry.forgotten, 1u);
    EXPECT_EQ(retry.pending, 0u);
}

TEST(TransferCompletionCleanupTest, ReentrantSubmissionsWaitForTheNextDrain) {
    auto queue = std::make_shared<Cleanup>();
    queue->prepare(localTarget(), uint64_t{1}).submit();
    size_t calls = 0;
    const auto first = queue->drain([&](const auto&, const auto& traces, int) {
        ++calls;
        EXPECT_EQ(queue->size(), 1u);
        queue->prepare(localTarget(), uint64_t{2}).submit();
        EXPECT_EQ(queue->size(), 2u);
        return doneResult(traces);
    }, 10);
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(first.forgotten, 1u);
    EXPECT_EQ(first.pending, 1u);
    const auto second = queue->drain([](const auto&, const auto& traces, int) {
        EXPECT_EQ(traces, (std::vector<TransferExecutionTrace>{uint64_t{2}}));
        return doneResult(traces);
    }, 10);
    EXPECT_EQ(second.forgotten, 1u);
    EXPECT_EQ(second.pending, 0u);
}

TEST(TransferCompletionCleanupTest, StopAndInvalidDrainArgumentsKeepEveryUnprocessedNode) {
    auto queue = std::make_shared<Cleanup>();
    queue->prepare(localTarget(), uint64_t{1}).submit();
    queue->prepare(remoteTarget(), uint64_t{2}).submit();
    size_t calls = 0;
    bool stop = true;
    const Cleanup::Forget forget = [&](const auto&, const auto& traces, int) {
        ++calls;
        stop = true;
        return doneResult(traces);
    };
    const auto stopped = queue->drain(forget, 10, [&] { return stop; });
    EXPECT_TRUE(stopped.stopped);
    EXPECT_EQ(stopped.attempted, 0u);
    EXPECT_EQ(stopped.pending, 2u);
    EXPECT_TRUE(queue->drain({}, 10).failed);
    EXPECT_TRUE(queue->drain(forget, 0).failed);
    EXPECT_EQ(calls, 0u);
    stop = false;
    const auto partial = queue->drain(forget, 10, [&] { return stop; });
    EXPECT_TRUE(partial.stopped);
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(partial.forgotten, 1u);
    EXPECT_EQ(partial.pending, 1u);
    const auto predicate_failed = queue->drain(forget, 10, []() -> bool {
        throw std::runtime_error("shutdown predicate unavailable");
    });
    EXPECT_TRUE(predicate_failed.failed);
    EXPECT_TRUE(predicate_failed.stopped);
    EXPECT_EQ(predicate_failed.pending, 1u);
    EXPECT_EQ(calls, 1u);
}

TEST(TransferCompletionCleanupTest, ConcurrentSubmissionRetainsEveryEntryBeyondWarningThreshold) {
    auto queue = std::make_shared<Cleanup>();
    constexpr size_t producers = 8;
    constexpr size_t per_producer = 1025;
    std::vector<std::thread> threads;
    for (size_t producer = 0; producer < producers; ++producer) {
        threads.emplace_back([&, producer] {
            for (uint64_t offset = 1; offset <= per_producer; ++offset)
                queue->prepare(localTarget(), producer * per_producer + offset).submit();
        });
    }
    for (auto& thread : threads)
        thread.join();
    EXPECT_EQ(queue->size(), producers * per_producer);
    std::vector<bool> seen(producers * per_producer, false);
    const auto summary = queue->drain([&](const auto&, const auto& traces, int) {
        for (const auto& trace : traces) {
            const auto index = std::get<uint64_t>(trace) - 1;
            EXPECT_LT(index, seen.size());
            if (index < seen.size()) {
                EXPECT_FALSE(seen[index]);
                seen[index] = true;
            }
        }
        return doneResult(traces);
    }, 10);
    EXPECT_EQ(summary.forgotten, seen.size());
    EXPECT_EQ(summary.pending, 0u);
    for (const auto present : seen)
        EXPECT_TRUE(present);
}

}  // namespace
}  // namespace rtp_llm
