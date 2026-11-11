#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskRecovery.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <stdexcept>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"

namespace rtp_llm {
namespace {

using State = TransferTaskRegistry::State;
using block_tree_cache_test::BoundedThread;
using namespace std::chrono_literals;

class RecoveryPool final: public IBlockPool {
public:
    explicit RecoveryPool(std::shared_ptr<const BlockPoolConfigBase> config): IBlockPool(std::move(config)) {
        markInitialized();
    }
    size_t blockSizeBytes() const override { return 0; }
};

std::shared_ptr<RecoveryPool> makePool() {
    auto config = std::make_shared<BlockPoolConfigBase>();
    config->pool_name = "frozen-recovery";
    config->pool_type = BlockPoolType::HOST;
    config->physical_block_count = 16;
    config->frozen_block_abort_lower_ratio = 0.0;
    config->frozen_block_abort_upper_ratio = 0.0;
    return std::make_shared<RecoveryPool>(std::move(config));
}

BlockIdxType hold(const std::shared_ptr<RecoveryPool>& pool) {
    const auto block = pool->malloc();
    EXPECT_TRUE(block.has_value());
    if (!block)
        return NULL_BLOCK_IDX;
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    return *block;
}

std::shared_ptr<FrozenTask> makeTask(TransferExecutionTrace trace, const std::shared_ptr<RecoveryPool>& pool,
                                    BlockIdxType block, bool remote = false,
                                    CacheCopyError error = CacheCopyError::COPY_FAILED) {
    FrozenTask::ExecutionInfo execution;
    execution.trace_id = std::move(trace);
    execution.operation = "load";
    execution.remote = remote;
    execution.expected_peers = remote ? std::vector<FrozenTask::ExpectedPeer>{{0, "worker-0"}, {1, "worker-1"}} :
                                        std::vector<FrozenTask::ExpectedPeer>{{0, "local"}};
    execution.expected_rank_count = execution.expected_peers.size();
    execution.descriptors.push_back({0, 0, Tier::REMOTE, Tier::DEVICE, {}, {{pool, block, 0}}});
    TransferResult failure(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "original business failure"), error);
    auto diagnostics = std::make_shared<TransferExecutionDiagnostics>();
    diagnostics->trace_id = execution.trace_id;
    diagnostics->failure_reason = "existing_failure";
    failure.execution_diagnostics = std::move(diagnostics);
    return std::make_shared<FrozenTask>(FrozenTask::prepare(std::move(execution)), std::move(failure));
}

TransferCompletionQueryResult proof(const FrozenTask::ExecutionInfo& execution,
                                    State last_state = State::BUFFER_SAFE) {
    TransferCompletionQueryResult result;
    result.trace_id = execution.trace_id;
    result.all_safe = last_state == State::BUFFER_SAFE;
    for (const auto& peer : execution.expected_peers) {
        TransferCompletionRankResult row;
        row.rank = peer.rank;
        row.server_address = peer.address;
        row.state = peer.rank + 1 == execution.expected_rank_count ? last_state : State::BUFFER_SAFE;
        if (execution.remote) {
            row.cq_received = true;
            row.cq_ok = true;
            row.grpc_code = 0;
        }
        result.ranks.push_back(std::move(row));
    }
    return result;
}

TEST(FrozenTaskRecoveryTest, LocalNumericAndStringProofsRecoverNonRpcFailuresWithoutChangingBusinessResults) {
    auto pool = makePool();
    const auto first_block = hold(pool);
    const auto second_block = hold(pool);
    auto first = makeTask(uint64_t{71}, pool, first_block, false, CacheCopyError::IO_FAILED);
    auto second = makeTask(std::string("71"), pool, second_block, false, CacheCopyError::CRC_MISMATCH);
    first->freeze();
    second->freeze();
    const auto original = first->failure().execution_diagnostics;
    TransferTaskRegistry ordinary;
    TransferTaskRegistry kvcm;
    ASSERT_TRUE(ordinary.begin(uint64_t{71}));
    ASSERT_TRUE(kvcm.begin(std::string("71")));
    TransferCompletionQuery local(nullptr, [&](const TransferExecutionTrace& trace) {
        if (const auto* numeric = std::get_if<uint64_t>(&trace))
            return ordinary.queryLocal(std::vector<uint64_t>{*numeric}).front();
        return kvcm.queryLocal(std::vector<std::string>{std::get<std::string>(trace)}).front();
    });
    FrozenTaskRecovery recovery([&] { return std::vector<std::shared_ptr<FrozenTask>>{first, second}; },
                                [&](const auto& execution, int timeout) { return local.query(execution, timeout); });
    const auto pending = recovery.runOnce(1000);
    EXPECT_EQ(pending.queried, 2u);
    EXPECT_EQ(pending.pending, 2u);
    EXPECT_EQ(pending.released, 0u);
    ASSERT_TRUE(ordinary.finish(uint64_t{71}, true));
    ASSERT_TRUE(kvcm.finish(std::string("71"), true));
    const auto released = recovery.runOnce(1000);
    EXPECT_EQ(released.released, 2u);
    EXPECT_EQ(released.query_failed, 0u);
    EXPECT_EQ(first->snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_EQ(second->snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_EQ(first->failure().status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    EXPECT_EQ(first->failure().copy_error, CacheCopyError::IO_FAILED);
    EXPECT_EQ(second->failure().copy_error, CacheCopyError::CRC_MISMATCH);
    EXPECT_FALSE(first->failure().buffer_safe);
    EXPECT_EQ(first->failure().execution_diagnostics, original);
    EXPECT_EQ(first->failure().execution_diagnostics->failure_reason, "existing_failure");
    EXPECT_EQ(first->recoverySnapshot().query_count, 2u);
    ASSERT_NE(first->recoverySnapshot().last_query, nullptr);
    EXPECT_TRUE(first->recoverySnapshot().last_query->all_safe);
    EXPECT_EQ(recovery.runOnce(1000).queried, 0u);
    EXPECT_EQ(pool->treeRefCount(first_block), 1u);
    EXPECT_EQ(pool->treeRefCount(second_block), 1u);
    EXPECT_EQ(ordinary.size(), 1u);
    EXPECT_EQ(kvcm.size(), 1u);
}

TEST(FrozenTaskRecoveryTest, RemoteRecoveryWaitsForEveryRankAndPreservesPendingQuerySnapshots) {
    auto pool = makePool();
    const auto block = hold(pool);
    auto task = makeTask(uint64_t{81}, pool, block, true);
    task->freeze();
    State last = State::RUNNING;
    FrozenTaskRecovery recovery([&] { return std::vector<std::shared_ptr<FrozenTask>>{task}; },
                                [&](const auto& execution, int) { return proof(execution, last); });
    EXPECT_EQ(recovery.runOnce(1000).released, 0u);
    const auto running_snapshot = task->recoverySnapshot().last_query;
    ASSERT_NE(running_snapshot, nullptr);
    EXPECT_EQ(running_snapshot->ranks[1].state, State::RUNNING);
    last = State::UNKNOWN;
    EXPECT_EQ(recovery.runOnce(1000).released, 0u);
    EXPECT_TRUE(pool->isFrozen(block));
    last = State::BUFFER_SAFE;
    EXPECT_EQ(recovery.runOnce(1000).released, 1u);
    EXPECT_FALSE(pool->isFrozen(block));
    EXPECT_EQ(task->recoverySnapshot().query_count, 3u);
    EXPECT_EQ(running_snapshot->ranks[1].state, State::RUNNING);
}

TEST(FrozenTaskRecoveryTest, InconsistentOrMismatchedInjectedProofNeverReleasesTargets) {
    auto pool = makePool();
    const auto block = hold(pool);
    auto task = makeTask(uint64_t{91}, pool, block, true);
    task->freeze();
    int malformed = 0;
    FrozenTaskRecovery recovery([&] { return std::vector<std::shared_ptr<FrozenTask>>{task}; },
        [&](const auto& execution, int) {
            auto result = proof(execution);
            switch (malformed) {
                case 0: result.trace_id = std::string("91"); break;
                case 1: result.trace_id = uint64_t{92}; break;
                case 2: result.ranks.pop_back(); break;
                case 3: result.ranks[1].rank = 0; break;
                case 4: result.ranks.push_back(result.ranks.back()); break;
                case 5: result.ranks[0].server_address = "another-worker"; break;
                case 6: result.ranks[0].state = State::RUNNING; break;
                case 7: result.all_safe = false; break;
                case 8: result.error = "query deadline exceeded"; break;
                case 9: result.ranks[0].error = "malformed response"; break;
                case 10: result.ranks[0].state = static_cast<State>(99); break;
                case 11: result.ranks[0].cq_received = false; break;
                case 12: result.ranks[0].cq_ok = false; break;
                case 13: result.ranks[0].grpc_code.reset(); break;
                case 14: result.ranks[0].grpc_code = 14; break;
            }
            return result;
        });
    for (; malformed < 15; ++malformed) {
        SCOPED_TRACE(malformed);
        const auto result = recovery.runOnce(1000);
        EXPECT_EQ(result.released, 0u);
        EXPECT_EQ(result.query_failed, 1u);
        EXPECT_TRUE(pool->isFrozen(block));
        EXPECT_EQ(task->snapshot().state, FrozenTask::State::FROZEN);
        ASSERT_NE(task->recoverySnapshot().last_query, nullptr);
        EXPECT_FALSE(task->recoverySnapshot().last_query->all_safe);
    }
}

TEST(FrozenTaskRecoveryTest, QueryAndSourceExceptionsDoNotPreventLaterTasksOrPasses) {
    auto pool = makePool();
    std::vector<std::shared_ptr<FrozenTask>> tasks;
    for (uint64_t trace = 1; trace <= 3; ++trace) {
        tasks.push_back(makeTask(trace, pool, hold(pool)));
        tasks.back()->freeze();
    }
    bool source_fails = true;
    bool query_fails = true;
    FrozenTaskRecovery recovery([&] {
        if (source_fails)
            throw std::runtime_error("source unavailable");
        return tasks;
    }, [&](const auto& execution, int) {
        const auto trace = std::get<uint64_t>(execution.trace_id);
        if (query_fails && trace == 1)
            throw std::runtime_error("query unavailable");
        if (query_fails && trace == 2)
            throw 7;
        return proof(execution);
    });
    EXPECT_TRUE(recovery.runOnce(1000).source_failed);
    source_fails = false;
    const auto first = recovery.runOnce(1000);
    EXPECT_EQ(first.queried, 3u);
    EXPECT_EQ(first.query_failed, 2u);
    EXPECT_EQ(first.released, 1u);
    EXPECT_EQ(first.pending, 2u);
    ASSERT_NE(tasks[0]->recoverySnapshot().last_query, nullptr);
    EXPECT_FALSE(tasks[0]->recoverySnapshot().last_query->all_safe);
    EXPECT_EQ(tasks[0]->recoverySnapshot().last_query->error, "query unavailable");
    query_fails = false;
    const auto second = recovery.runOnce(1000);
    EXPECT_EQ(second.queried, 2u);
    EXPECT_EQ(second.released, 2u);
    EXPECT_FALSE(second.skipped_running);
}

TEST(FrozenTaskRecoveryTest, SharedBlockRemainsHeldUntilEachExecutionHasSafeProof) {
    auto pool = makePool();
    const auto block = hold(pool);
    auto first = makeTask(uint64_t{1}, pool, block);
    auto second = makeTask(uint64_t{2}, pool, block);
    first->freeze();
    second->freeze();
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->treeRefCount(block), 2u);
    bool second_safe = false;
    FrozenTaskRecovery recovery([&] { return std::vector<std::shared_ptr<FrozenTask>>{first, second}; },
        [&](const auto& execution, int) {
            return proof(execution, std::get<uint64_t>(execution.trace_id) == 1 || second_safe ?
                                        State::BUFFER_SAFE : State::RUNNING);
        });
    EXPECT_EQ(recovery.runOnce(1000).released, 1u);
    EXPECT_TRUE(pool->isFrozen(block));
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->treeRefCount(block), 1u);
    second_safe = true;
    EXPECT_EQ(recovery.runOnce(1000).released, 1u);
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_EQ(recovery.runOnce(1000).released, 0u);
}

TEST(FrozenTaskRecoveryTest, ConcurrentPassIsSkippedAndManualReleaseRaceIsCountedExactlyOnce) {
    for (bool manual_release : {false, true}) {
        auto pool = makePool();
        const auto block = hold(pool);
        auto task = makeTask(uint64_t{1}, pool, block);
        task->freeze();
        auto entered = std::make_shared<std::promise<void>>();
        auto entered_future = entered->get_future();
        auto release = std::make_shared<std::promise<void>>();
        auto released = release->get_future().share();
        auto calls = std::make_shared<std::atomic<size_t>>(0);
        auto recovery = std::make_shared<FrozenTaskRecovery>(
            [task] { return std::vector<std::shared_ptr<FrozenTask>>{task}; },
            [entered, released, calls](const auto& execution, int) {
                ++*calls;
                entered->set_value();
                if (released.wait_for(5s) != std::future_status::ready)
                    throw std::runtime_error("test query gate timed out");
                return proof(execution);
            });
        BoundedThread<FrozenTaskRecovery::Summary> first([recovery] { return recovery->runOnce(1000); });
        ASSERT_EQ(entered_future.wait_for(2s), std::future_status::ready);
        const auto concurrent = recovery->runOnce(1000);
        EXPECT_TRUE(concurrent.skipped_running);
        EXPECT_EQ(concurrent.queried, 0u);
        EXPECT_EQ(calls->load(), 1u);
        if (manual_release) {
            EXPECT_TRUE(task->unfreeze());
        }
        release->set_value();
        ASSERT_EQ(first.waitFor(2s), std::future_status::ready);
        EXPECT_EQ(first.get().released, manual_release ? 0u : 1u);
        EXPECT_EQ(task->snapshot().state, FrozenTask::State::RELEASED);
        EXPECT_EQ(pool->treeRefCount(block), 1u);
        EXPECT_FALSE(task->unfreeze());
        EXPECT_FALSE(recovery->runOnce(1000).skipped_running);
    }
}

TEST(FrozenTaskRecoveryTest, StopBeforeQueryAfterQueryAndImmediatelyBeforeReleaseIsConservative) {
    for (int phase = 0; phase < 4; ++phase) {
        SCOPED_TRACE(phase);
        auto pool = makePool();
        const auto block = hold(pool);
        auto task = makeTask(uint64_t{1}, pool, block);
        task->freeze();
        bool queried = false;
        size_t checks_after_query = 0;
        FrozenTaskRecovery recovery([&] { return std::vector<std::shared_ptr<FrozenTask>>{task}; },
            [&](const auto& execution, int) { queried = true; return proof(execution); });
        const auto stopped = recovery.runOnce(1000, [&] {
            if (phase == 0)
                return true;
            if (phase == 3)
                throw std::runtime_error("shutdown predicate failed");
            if (!queried)
                return false;
            return ++checks_after_query >= (phase == 1 ? 1u : 2u);
        });
        EXPECT_TRUE(stopped.stopped);
        EXPECT_EQ(stopped.queried, phase == 0 || phase == 3 ? 0u : 1u);
        EXPECT_EQ(stopped.released, 0u);
        EXPECT_TRUE(pool->isFrozen(block));
        EXPECT_EQ(recovery.runOnce(1000).released, 1u);
    }
}

TEST(FrozenTaskRecoveryTest, InvalidTimeoutAndNonFrozenRecordsNeverInvokeQuery) {
    auto pool = makePool();
    auto prepared = makeTask(uint64_t{1}, pool, hold(pool));
    auto released = makeTask(uint64_t{2}, pool, hold(pool));
    released->freeze();
    ASSERT_TRUE(released->unfreeze());
    auto frozen = makeTask(uint64_t{3}, pool, hold(pool));
    frozen->freeze();
    size_t sources = 0;
    size_t queries = 0;
    FrozenTaskRecovery recovery([&] {
        ++sources;
        return std::vector<std::shared_ptr<FrozenTask>>{nullptr, prepared, released, frozen};
    }, [&](const auto& execution, int timeout) {
        ++queries;
        EXPECT_EQ(timeout, 23);
        return proof(execution);
    });
    EXPECT_TRUE(recovery.runOnce(0).invalid_timeout);
    EXPECT_TRUE(recovery.runOnce(-1).invalid_timeout);
    EXPECT_EQ(sources, 0u);
    const auto result = recovery.runOnce(23);
    EXPECT_EQ(result.inspected, 3u);
    EXPECT_EQ(result.queried, 1u);
    EXPECT_EQ(result.released, 1u);
    EXPECT_EQ(queries, 1u);
    EXPECT_EQ(prepared->snapshot().state, FrozenTask::State::PREPARED);
}


template<typename Predicate>
bool waitForRecoveryCondition(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline)
            return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

TEST(FrozenTaskRecoveryTest, BackgroundRetriesFailureAndRunningUntilSafeWithoutRepeatingRelease) {
    auto pool = makePool();
    const auto block = hold(pool);
    auto task = makeTask(std::string("background-read"), pool, block);
    task->freeze();
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto recovery = std::make_shared<FrozenTaskRecovery>(
        [task] { return std::vector<std::shared_ptr<FrozenTask>>{task}; },
        [calls](const auto& execution, int timeout) {
            EXPECT_EQ(timeout, 200);
            const auto attempt = calls->fetch_add(1);
            if (attempt == 0)
                throw std::runtime_error("transient query failure");
            return proof(execution, attempt == 1 ? State::RUNNING : State::BUFFER_SAFE);
        });
    ASSERT_TRUE(recovery->start(10, 200));
    ASSERT_TRUE(waitForRecoveryCondition([&] { return task->snapshot().state == FrozenTask::State::RELEASED; }));
    recovery->stop();
    EXPECT_EQ(calls->load(), 3u);
    EXPECT_EQ(task->recoverySnapshot().query_count, 3u);
    EXPECT_EQ(pool->treeRefCount(block), 1u);
    EXPECT_EQ(task->failure().status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    EXPECT_FALSE(task->failure().buffer_safe);
    EXPECT_FALSE(task->unfreeze());
}

TEST(FrozenTaskRecoveryTest, BackgroundEmptySourceWaitIsInterruptibleAndStartIsSingleUse) {
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto queried = std::make_shared<std::atomic<size_t>>(0);
    auto entered = std::make_shared<std::promise<void>>();
    auto first_pass = entered->get_future();
    auto recovery = std::make_shared<FrozenTaskRecovery>(
        [calls, entered] {
            if (calls->fetch_add(1) == 0)
                entered->set_value();
            return std::vector<std::shared_ptr<FrozenTask>>{};
        },
        [queried](const auto& execution, int) { ++*queried; return proof(execution); });
    EXPECT_FALSE(recovery->start(0, 1000));
    EXPECT_FALSE(recovery->start(-1, 1000));
    EXPECT_FALSE(recovery->start(10, 0));
    EXPECT_FALSE(recovery->start(10, -1));
    EXPECT_EQ(calls->load(), 0u);
    ASSERT_TRUE(recovery->start(60000, 1000));
    EXPECT_FALSE(recovery->start(10, 1000));
    ASSERT_EQ(first_pass.wait_for(2s), std::future_status::ready);
    // A stop also covers the gap between finishing a pass and entering wait.
    BoundedThread<void> stopped([recovery] { recovery->stop(); });
    ASSERT_EQ(stopped.waitFor(1s), std::future_status::ready);
    stopped.get();
    recovery->stop();
    EXPECT_FALSE(recovery->start(10, 1000));
    EXPECT_EQ(calls->load(), 1u);
    EXPECT_EQ(queried->load(), 0u);

    FrozenTaskRecovery never_started([] { return std::vector<std::shared_ptr<FrozenTask>>{}; }, {});
    never_started.stop();
    never_started.stop();
    EXPECT_FALSE(never_started.start(10, 1000));
}

TEST(FrozenTaskRecoveryTest, StopDuringQueryWaitsForExitAndDiscardsSafeReleasePermission) {
    auto pool = makePool();
    const auto block = hold(pool);
    auto task = makeTask(uint64_t{51}, pool, block);
    task->freeze();
    auto entered = std::make_shared<std::promise<void>>();
    auto in_query = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto released = release->get_future().share();
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto recovery = std::make_shared<FrozenTaskRecovery>(
        [task] { return std::vector<std::shared_ptr<FrozenTask>>{task}; },
        [entered, released, calls](const auto& execution, int) {
            ++*calls;
            entered->set_value();
            if (released.wait_for(5s) != std::future_status::ready)
                throw std::runtime_error("test query gate timed out");
            return proof(execution);
        });
    ASSERT_TRUE(recovery->start(10, 1000));
    ASSERT_EQ(in_query.wait_for(2s), std::future_status::ready);
    BoundedThread<void> stopped([recovery] { recovery->stop(); });
    // Observe only the stop barrier to make this race deterministic; the
    // behavioral assertion is that a later complete proof cannot release.
    ASSERT_TRUE(waitForRecoveryCondition([&] { return recovery->stop_requested_.load(); }));
    EXPECT_EQ(stopped.waitFor(20ms), std::future_status::timeout);
    EXPECT_TRUE(pool->isFrozen(block));
    release->set_value();
    ASSERT_EQ(stopped.waitFor(2s), std::future_status::ready);
    stopped.get();
    EXPECT_EQ(calls->load(), 1u);
    EXPECT_EQ(task->snapshot().state, FrozenTask::State::FROZEN);
    EXPECT_EQ(pool->treeRefCount(block), 2u);
    const auto snapshot = task->recoverySnapshot();
    ASSERT_NE(snapshot.last_query, nullptr);
    EXPECT_TRUE(snapshot.last_query->all_safe);
    EXPECT_EQ(snapshot.query_count, 1u);
    recovery->stop();
    EXPECT_EQ(task->snapshot().state, FrozenTask::State::FROZEN);
}

TEST(FrozenTaskRecoveryTest, DestructionJoinsSleepingWorkerBeforeReleasingCallbackDependencies) {
    auto dependency = std::make_shared<int>(7);
    std::weak_ptr<int> weak_dependency = dependency;
    auto entered = std::make_shared<std::promise<void>>();
    auto first_pass = entered->get_future();
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto recovery = std::make_unique<FrozenTaskRecovery>(
        [dependency, entered, calls] {
            EXPECT_EQ(*dependency, 7);
            if (calls->fetch_add(1) == 0)
                entered->set_value();
            return std::vector<std::shared_ptr<FrozenTask>>{};
        },
        [](const auto& execution, int) { return proof(execution); });
    dependency.reset();
    ASSERT_TRUE(recovery->start(60000, 1000));
    ASSERT_EQ(first_pass.wait_for(2s), std::future_status::ready);
    EXPECT_FALSE(weak_dependency.expired());
    BoundedThread<void> destroyed([recovery = std::move(recovery)]() mutable { recovery.reset(); });
    ASSERT_EQ(destroyed.waitFor(1s), std::future_status::ready);
    destroyed.get();
    EXPECT_TRUE(weak_dependency.expired());
    EXPECT_EQ(calls->load(), 1u);
}

}  // namespace
}  // namespace rtp_llm
