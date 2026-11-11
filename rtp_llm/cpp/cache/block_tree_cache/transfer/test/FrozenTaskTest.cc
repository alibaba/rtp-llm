#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionTypes.h"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

class FrozenTestPool final: public IBlockPool {
public:
    explicit FrozenTestPool(std::shared_ptr<const BlockPoolConfigBase> config): IBlockPool(std::move(config)) {
        markInitialized();
    }
    size_t blockSizeBytes() const override {
        return 0;
    }
};

std::shared_ptr<FrozenTestPool> makePool(std::string name) {
    auto config                            = std::make_shared<BlockPoolConfigBase>();
    config->pool_name                      = std::move(name);
    config->pool_type                      = BlockPoolType::HOST;
    config->physical_block_count           = 16;
    config->frozen_block_abort_lower_ratio  = 0.0;
    return std::make_shared<FrozenTestPool>(std::move(config));
}

BlockIdxType heldBlock(const std::shared_ptr<FrozenTestPool>& pool) {
    const auto block = pool->malloc();
    EXPECT_TRUE(block.has_value());
    if (!block)
        return NULL_BLOCK_IDX;
    pool->incTreeRef(*block, BlockTreeRefType::CACHE);
    return *block;
}

FrozenTask::ExecutionInfo execution(uint64_t                           trace,
                                    std::vector<FrozenTask::BlockInfo> sources,
                                    std::vector<FrozenTask::BlockInfo> targets) {
    FrozenTask::ExecutionInfo info;
    info.trace_id           = trace;
    info.operation          = "load";
    info.request_id         = 73;
    info.workflow_id        = 11;
    info.parent_trace_id    = 12;
    info.remote             = true;
    info.expected_rank_count = 2;
    info.expected_peers     = {{0, "worker-0:100"}, {1, "worker-1:100"}};
    info.descriptors.push_back({3, 9, Tier::HOST, Tier::DEVICE, std::move(sources), std::move(targets)});
    return info;
}

TransferResult unknownResult() {
    return TransferResult(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "RPC timed out"),
                          CacheCopyError::RPC_FAILED);
}

TEST(FrozenTaskTest, PreparationDoesNotAcquireReferencesAndOnlyTargetsFreeze) {
    auto source_pool = makePool("source");
    auto target_pool = makePool("target");
    const auto source = heldBlock(source_pool);
    const auto target = heldBlock(target_pool);
    auto info = execution(21, {{source_pool, source, 0}}, {{target_pool, target, 1}});
    auto prepared = FrozenTask::prepare(std::move(info));
    EXPECT_EQ(source_pool->treeRefCount(source), 1u);
    EXPECT_EQ(target_pool->treeRefCount(target), 1u);

    FrozenTask task(std::move(prepared), unknownResult());
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::PREPARED);
    EXPECT_FALSE(task.unfreeze());
    task.freeze();
    task.freeze();
    EXPECT_FALSE(source_pool->isFrozen(source));
    EXPECT_EQ(source_pool->treeRefCount(source), 1u);
    EXPECT_TRUE(target_pool->isFrozen(target));
    EXPECT_EQ(target_pool->treeRefCount(target), 2u);
    const auto frozen = task.snapshot();
    EXPECT_EQ(frozen.state, FrozenTask::State::FROZEN);
    EXPECT_EQ(frozen.target_block_count, 1u);
    EXPECT_GT(frozen.created_at_us, 0);
    EXPECT_GT(frozen.frozen_at_us, 0);
    EXPECT_EQ(frozen.released_at_us, 0);

    EXPECT_TRUE(task.unfreeze());
    const auto released = task.snapshot();
    EXPECT_EQ(released.state, FrozenTask::State::RELEASED);
    EXPECT_GT(released.released_at_us, 0);
    EXPECT_FALSE(task.unfreeze());
    task.freeze();
    EXPECT_EQ(task.snapshot().released_at_us, released.released_at_us);
    EXPECT_EQ(task.snapshot().frozen_at_us, frozen.frozen_at_us);
    EXPECT_FALSE(target_pool->isFrozen(target));
    EXPECT_EQ(target_pool->treeRefCount(target), 1u);
    EXPECT_EQ(source_pool->treeRefCount(source), 1u);
}

TEST(FrozenTaskTest, DeduplicatesTargetSlotsAcrossDescriptorsButKeepsCompleteMetadata) {
    auto first_pool  = makePool("first");
    auto second_pool = makePool("second");
    const auto first = heldBlock(first_pool);
    const auto second = heldBlock(second_pool);
    ASSERT_EQ(first, second);
    auto info = execution(22, {{first_pool, first, 0}}, {{first_pool, first, 0}, {first_pool, first, 1}});
    info.descriptors.push_back({4,
                                10,
                                Tier::DEVICE,
                                Tier::HOST,
                                {{first_pool, first, 0}},
                                {{first_pool, first, 0}, {second_pool, second, 0}}});
    FrozenTask task(FrozenTask::prepare(std::move(info)), unknownResult());
    task.freeze();
    EXPECT_EQ(task.snapshot().target_block_count, 2u);
    EXPECT_EQ(first_pool->treeRefCount(first), 2u);
    EXPECT_EQ(second_pool->treeRefCount(second), 2u);
    ASSERT_EQ(task.execution().descriptors.size(), 2u);
    EXPECT_EQ(task.execution().descriptors[0].targets.size(), 2u);
    EXPECT_EQ(task.execution().descriptors[1].targets.size(), 2u);
    EXPECT_EQ(task.execution().descriptors[0].sources[0].pool_name, "first");
    EXPECT_EQ(task.execution().descriptors[1].targets[1].pool_name, "second");
    task.unfreeze();
    EXPECT_EQ(first_pool->treeRefCount(first), 1u);
    EXPECT_EQ(second_pool->treeRefCount(second), 1u);
}

TEST(FrozenTaskTest, SharedTargetsKeepIndependentTaskReferencesAcrossBusinessRelease) {
    auto pool = makePool("shared");
    const auto block = heldBlock(pool);
    FrozenTask first(FrozenTask::prepare(execution(23, {}, {{pool, block, 0}})), unknownResult());
    FrozenTask second(FrozenTask::prepare(execution(24, {}, {{pool, block, 0}})), unknownResult());
    first.freeze();
    second.freeze();
    EXPECT_EQ(pool->treeRefCount(block), 3u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);

    pool->decTreeRef(block, BlockTreeRefType::CACHE);
    EXPECT_TRUE(first.unfreeze());
    EXPECT_FALSE(first.unfreeze());
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_TRUE(pool->isFrozen(block));
    EXPECT_EQ(pool->treeRefCount(block), 1u);
    EXPECT_TRUE(second.unfreeze());
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
}

TEST(FrozenTaskTest, DestructionDoesNotReleaseFrozenTargets) {
    auto pool = makePool("retained");
    const auto block = heldBlock(pool);
    {
        FrozenTask task(FrozenTask::prepare(execution(25, {}, {{pool, block, 0}})), unknownResult());
        task.freeze();
    }
    EXPECT_EQ(pool->treeRefCount(block), 2u);
    EXPECT_TRUE(pool->isFrozen(block));
    // Test cleanup releases the intentionally retained reference explicitly.
    pool->decTreeRef(block, BlockTreeRefType::FROZEN);
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
}

TEST(FrozenTaskTest, PoolOwnershipOutlivesCallerAndSourceAllocation) {
    auto source_pool = makePool("source-lifetime");
    auto target_pool = makePool("target-lifetime");
    const auto source = heldBlock(source_pool);
    const auto target = heldBlock(target_pool);
    std::weak_ptr<FrozenTestPool> weak_source = source_pool;
    std::weak_ptr<FrozenTestPool> weak_target = target_pool;
    {
        FrozenTask task(FrozenTask::prepare(execution(26, {{source_pool, source, 0}}, {{target_pool, target, 0}})),
                        unknownResult());
        task.freeze();
        source_pool->decTreeRef(source, BlockTreeRefType::CACHE);
        EXPECT_FALSE(source_pool->isAllocated(source));
        target_pool->decTreeRef(target, BlockTreeRefType::CACHE);
        source_pool.reset();
        target_pool.reset();
        EXPECT_FALSE(weak_source.expired());
        EXPECT_FALSE(weak_target.expired());
        EXPECT_NE(task.debugString().find("source-lifetime"), std::string::npos);
        task.unfreeze();
        EXPECT_EQ(weak_target.lock()->freeBlocksNum(), weak_target.lock()->totalBlocksNum());
    }
    EXPECT_TRUE(weak_source.expired());
    EXPECT_TRUE(weak_target.expired());
}

TEST(FrozenTaskTest, ThrowingCapacityNotificationsCannotInterruptOrRepeatBatchRelease) {
    auto first_pool = makePool("first-notify");
    auto second_pool = makePool("second-notify");
    const auto first = heldBlock(first_pool);
    const auto other = heldBlock(first_pool);
    const auto second = heldBlock(second_pool);
    size_t first_notifications = 0;
    size_t second_notifications = 0;
    first_pool->setCapacityChangeCallback([&] {
        ++first_notifications;
        throw std::runtime_error("notification failed after release");
    });
    second_pool->setCapacityChangeCallback([&] { ++second_notifications; });
    FrozenTask task(FrozenTask::prepare(execution(27,
                                                 {},
                                                 {{first_pool, first, 0},
                                                  {first_pool, other, 1},
                                                  {first_pool, first, 0},
                                                  {second_pool, second, 0}})),
                    unknownResult());
    task.freeze();
    first_pool->decTreeRef({first, other}, BlockTreeRefType::CACHE);
    second_pool->decTreeRef(second, BlockTreeRefType::CACHE);
    EXPECT_TRUE(task.unfreeze());
    EXPECT_FALSE(task.unfreeze());
    EXPECT_EQ(first_notifications, 1u);
    EXPECT_EQ(second_notifications, 1u);
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_EQ(first_pool->freeBlocksNum(), first_pool->totalBlocksNum());
    EXPECT_EQ(second_pool->freeBlocksNum(), second_pool->totalBlocksNum());
}

TEST(FrozenTaskTest, ReleaseNotificationsCanInspectAndReenterTaskWithoutRepeatingReferences) {
    auto pool = makePool("reentrant-notify");
    const auto block = heldBlock(pool);
    FrozenTask* observed = nullptr;
    size_t notifications = 0;
    pool->setCapacityChangeCallback([&] {
        ++notifications;
        ASSERT_NE(observed, nullptr);
        EXPECT_EQ(observed->snapshot().state, FrozenTask::State::RELEASING);
        EXPECT_NE(observed->debugString().find("state=RELEASING"), std::string::npos);
        EXPECT_FALSE(observed->unfreeze());
        observed->freeze();
    });
    FrozenTask task(FrozenTask::prepare(execution(28, {}, {{pool, block, 0}})), unknownResult());
    observed = &task;
    task.freeze();
    EXPECT_TRUE(task.unfreeze());
    EXPECT_EQ(notifications, 1u);
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_EQ(pool->treeRefCount(block), 1u);
}

TEST(FrozenTaskTest, ConcurrentRepeatedFreezeAndUnfreezeOwnOneReference) {
    auto pool = makePool("concurrent");
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    const auto block = heldBlock(pool);
    std::atomic<size_t> notifications{0};
    pool->setCapacityChangeCallback([&] { ++notifications; });
    FrozenTask task(FrozenTask::prepare(execution(29, {}, {{pool, block, 0}})), unknownResult(),
                    cleanup->prepare({true, {{0, "worker-0:100"}, {1, "worker-1:100"}}}, uint64_t{29}));
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 8; ++i) {
        threads.emplace_back([&] { task.freeze(); });
    }
    for (auto& thread : threads)
        thread.join();
    EXPECT_EQ(pool->treeRefCount(block), 2u);
    threads.clear();
    std::atomic<size_t> releases{0};
    for (size_t i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            if (task.unfreeze()) {
                ++releases;
            }
        });
    }
    for (auto& thread : threads)
        thread.join();
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_EQ(pool->treeRefCount(block), 1u);
    EXPECT_EQ(notifications.load(), 1u);
    EXPECT_EQ(releases.load(), 1u);
    EXPECT_EQ(cleanup->size(), 1u);
}

TEST(FrozenTaskTest, CleanupIsSubmittedOnlyAfterEveryPhysicalReleaseFinishes) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto first_pool = makePool("cleanup-first");
    auto second_pool = makePool("cleanup-second");
    const auto first = heldBlock(first_pool);
    const auto second = heldBlock(second_pool);
    size_t notifications = 0;
    FrozenTask* observed = nullptr;
    first_pool->setCapacityChangeCallback([&] {
        ++notifications;
        ASSERT_NE(observed, nullptr);
        EXPECT_EQ(observed->snapshot().state, FrozenTask::State::RELEASING);
        EXPECT_FALSE(observed->unfreeze());
        EXPECT_EQ(cleanup->size(), 0u);
        EXPECT_FALSE(first_pool->isAllocated(first));
        EXPECT_TRUE(second_pool->isFrozen(second));
    });
    second_pool->setCapacityChangeCallback([&] {
        ++notifications;
        ASSERT_NE(observed, nullptr);
        EXPECT_EQ(observed->snapshot().state, FrozenTask::State::RELEASING);
        EXPECT_EQ(cleanup->size(), 0u);
        EXPECT_FALSE(first_pool->isAllocated(first));
        EXPECT_FALSE(second_pool->isAllocated(second));
    });
    FrozenTask task(FrozenTask::prepare(execution(36, {}, {{first_pool, first, 0}, {second_pool, second, 0}})),
                    unknownResult(),
                    cleanup->prepare({true, {{0, "worker-0:100"}, {1, "worker-1:100"}}}, uint64_t{36}));
    observed = &task;
    EXPECT_FALSE(task.unfreeze());
    EXPECT_EQ(cleanup->size(), 0u);
    task.freeze();
    first_pool->decTreeRef(first, BlockTreeRefType::CACHE);
    second_pool->decTreeRef(second, BlockTreeRefType::CACHE);
    EXPECT_TRUE(task.unfreeze());
    EXPECT_FALSE(task.unfreeze());
    EXPECT_EQ(notifications, 2u);
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_EQ(cleanup->size(), 1u);
    const auto summary = cleanup->drain([&](const auto& target, const auto& traces, int) {
        EXPECT_TRUE(target.remote);
        EXPECT_EQ(target.peers.size(), 2u);
        EXPECT_EQ(traces, (std::vector<TransferExecutionTrace>{uint64_t{36}}));
        EXPECT_FALSE(first_pool->isFrozen(first));
        EXPECT_FALSE(second_pool->isFrozen(second));
        TransferCompletionForgetResult result;
        TransferCompletionForgetEntryResult entry;
        entry.trace_id = traces.front();
        entry.all_done = true;
        result.entries.push_back(std::move(entry));
        return result;
    }, 10);
    EXPECT_EQ(summary.forgotten, 1u);
    EXPECT_EQ(summary.pending, 0u);
}

TEST(FrozenTaskTest, DestructionNeverSubmitsCleanupForAnUnreleasedTask) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto pool = makePool("cleanup-unreleased");
    const auto block = heldBlock(pool);
    {
        FrozenTask task(FrozenTask::prepare(execution(37, {}, {{pool, block, 0}})), unknownResult(),
                        cleanup->prepare({true, {{0, "worker-0:100"}, {1, "worker-1:100"}}}, uint64_t{37}));
        task.freeze();
    }
    EXPECT_EQ(cleanup->size(), 0u);
    EXPECT_TRUE(pool->isFrozen(block));
    pool->decTreeRef(block, BlockTreeRefType::FROZEN);
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
}

TEST(FrozenTaskTest, PendingCleanupDoesNotRetainTheReleasedTaskOrPools) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    std::weak_ptr<FrozenTask> weak_task;
    std::weak_ptr<FrozenTestPool> weak_source;
    std::weak_ptr<FrozenTestPool> weak_target;
    {
        auto source_pool = makePool("cleanup-source-lifetime");
        auto target_pool = makePool("cleanup-target-lifetime");
        weak_source = source_pool;
        weak_target = target_pool;
        const auto source = heldBlock(source_pool);
        const auto target = heldBlock(target_pool);
        auto task = std::make_shared<FrozenTask>(
            FrozenTask::prepare(execution(38, {{source_pool, source, 0}}, {{target_pool, target, 0}})),
            unknownResult(),
            cleanup->prepare({true, {{0, "worker-0:100"}, {1, "worker-1:100"}}}, uint64_t{38}));
        weak_task = task;
        task->freeze();
        source_pool->decTreeRef(source, BlockTreeRefType::CACHE);
        target_pool->decTreeRef(target, BlockTreeRefType::CACHE);
        EXPECT_TRUE(task->unfreeze());
        EXPECT_EQ(cleanup->size(), 1u);
    }
    EXPECT_TRUE(weak_task.expired());
    EXPECT_TRUE(weak_source.expired());
    EXPECT_TRUE(weak_target.expired());
    EXPECT_EQ(cleanup->size(), 1u);
}

TEST(FrozenTaskTest, RecoveryQueriesPreserveOriginalFailureAndDoNotReleaseTargets) {
    auto pool = makePool("recovery-observation");
    const auto block = heldBlock(pool);
    auto info = execution(34, {}, {{pool, block, 0}});
    auto diagnostics = std::make_shared<TransferExecutionDiagnostics>();
    diagnostics->trace_id = info.trace_id;
    diagnostics->failure_reason = "worker_completion_unknown";
    TransferRankSnapshot original_rank;
    original_rank.rank = 1;
    original_rank.server_address = "worker-1:100";
    original_rank.grpc_code = 4;
    diagnostics->ranks.push_back(original_rank);
    auto failure = unknownResult();
    failure.execution_diagnostics = diagnostics;
    const auto original_error = failure.status.ToString();
    FrozenTask task(FrozenTask::prepare(std::move(info)), std::move(failure));
    task.freeze();

    task.recordRecoveryQuery(nullptr);
    const auto initial = task.recoverySnapshot();
    EXPECT_EQ(initial.query_count, 0u);
    EXPECT_EQ(initial.last_query_at_us, 0);
    EXPECT_EQ(initial.last_query, nullptr);

    TransferCompletionQueryResult pending;
    pending.trace_id = uint64_t{34};
    pending.error = "rank 1 query timed out";
    pending.ranks = {{0, "worker-0:100", TransferTaskRegistry::State::RUNNING, 0, "", true, true},
                     {1, "worker-1:100", TransferTaskRegistry::State::UNKNOWN, 4, "deadline", false, false}};
    auto pending_result = std::make_shared<const TransferCompletionQueryResult>(std::move(pending));
    task.recordRecoveryQuery(pending_result);
    const auto first = task.recoverySnapshot();
    EXPECT_EQ(first.query_count, 1u);
    EXPECT_GT(first.last_query_at_us, 0);
    EXPECT_EQ(first.last_query, pending_result);
    const auto pending_log = task.debugString();
    for (const auto* field : {"recovery_query_count=1", "last_recovery_query={trace=34,all_safe=false",
                              "error=\"rank 1 query timed out\"", "state=RUNNING,grpc_code=0",
                              "state=UNKNOWN,grpc_code=4,error=\"deadline\",cq_received=false,cq_ok=false"}) {
        EXPECT_NE(pending_log.find(field), std::string::npos) << field << '\n' << pending_log;
    }

    TransferCompletionQueryResult safe;
    safe.trace_id = uint64_t{34};
    safe.all_safe = true;
    safe.ranks = {{0, "worker-0:100", TransferTaskRegistry::State::BUFFER_SAFE, 0, "", true, true},
                  {1, "worker-1:100", TransferTaskRegistry::State::BUFFER_SAFE, 0, "", true, true}};
    auto safe_result = std::make_shared<const TransferCompletionQueryResult>(std::move(safe));
    task.recordRecoveryQuery(safe_result);
    task.recordRecoveryQuery(nullptr);
    const auto latest = task.recoverySnapshot();
    EXPECT_EQ(latest.query_count, 2u);
    EXPECT_GT(latest.last_query_at_us, 0);
    EXPECT_EQ(latest.last_query, safe_result);
    EXPECT_FALSE(first.last_query->all_safe);
    EXPECT_EQ(first.last_query->error, "rank 1 query timed out");
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::FROZEN);
    EXPECT_EQ(pool->treeRefCount(block), 2u);
    EXPECT_EQ(task.failure().status.ToString(), original_error);
    EXPECT_FALSE(task.failure().buffer_safe);
    ASSERT_EQ(task.failure().execution_diagnostics, diagnostics);
    EXPECT_EQ(task.failure().execution_diagnostics->failure_reason, "worker_completion_unknown");
    ASSERT_EQ(task.failure().execution_diagnostics->ranks.size(), 1u);
    EXPECT_EQ(task.failure().execution_diagnostics->ranks.front().grpc_code, 4);
    const auto safe_log = task.debugString();
    for (const auto* field : {"recovery_query_count=2", "last_recovery_query={trace=34,all_safe=true",
                              "state=BUFFER_SAFE", "failure_reason=\"worker_completion_unknown\""}) {
        EXPECT_NE(safe_log.find(field), std::string::npos) << field << '\n' << safe_log;
    }
    EXPECT_TRUE(task.unfreeze());
    EXPECT_EQ(task.recoverySnapshot().last_query, safe_result);
}

TEST(FrozenTaskTest, ConcurrentRecoveryQueriesRetainCountAndLatestStringTraceObservation) {
    auto pool = makePool("concurrent-observation");
    const auto block = heldBlock(pool);
    auto info = execution(35, {}, {{pool, block, 0}});
    info.trace_id = std::string("read-35");
    FrozenTask task(FrozenTask::prepare(std::move(info)), unknownResult());
    task.freeze();
    TransferCompletionQueryResult query;
    query.trace_id = std::string("read-35");
    query.ranks = {{0, "", TransferTaskRegistry::State::UNKNOWN, std::nullopt, "not observed", false, false}};
    auto result = std::make_shared<const TransferCompletionQueryResult>(std::move(query));
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            for (size_t j = 0; j < 8; ++j) {
                task.recordRecoveryQuery(result);
                const auto observed = task.recoverySnapshot();
                EXPECT_GT(observed.query_count, 0u);
                EXPECT_GT(observed.last_query_at_us, 0);
                EXPECT_EQ(observed.last_query, result);
            }
        });
    }
    for (auto& thread : threads)
        thread.join();
    EXPECT_EQ(task.recoverySnapshot().query_count, 64u);
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::FROZEN);
    const auto log = task.debugString();
    EXPECT_NE(log.find("last_recovery_query={trace=\"read-35\",all_safe=false"), std::string::npos);
    EXPECT_NE(log.find("state=UNKNOWN,grpc_code=unknown,error=\"not observed\""), std::string::npos);
    EXPECT_TRUE(task.unfreeze());
}

TEST(FrozenTaskTest, InvalidTargetPreparationDoesNotAcquireAnyFrozenHold) {
    auto pool = makePool("invalid");
    const auto block = heldBlock(pool);
    EXPECT_ANY_THROW(FrozenTask::prepare(execution(30, {}, {{pool, block, 0}, {pool, NULL_BLOCK_IDX, 0}})));
    EXPECT_EQ(pool->treeRefCount(block), 1u);
    EXPECT_FALSE(pool->isFrozen(block));
    EXPECT_ANY_THROW(FrozenTask::prepare(execution(30, {}, {{nullptr, block, 0}})));
    EXPECT_ANY_THROW(FrozenTask::prepare(execution(0, {}, {{pool, block, 0}})));
    EXPECT_ANY_THROW(FrozenTask(FrozenTask::prepare(execution(30, {}, {{pool, block, 0}})),
                                TransferResult(TransferStatus::OK)));
    EXPECT_EQ(pool->treeRefCount(block), 1u);
}

TEST(FrozenTaskTest, DiagnosticsRetainEveryRankOutcomeAndAllEndpointRoles) {
    auto target_pool = makePool("target-log");
    const auto target = heldBlock(target_pool);
    auto info = execution(31, {{nullptr, 7, 2, "unknown-source-pool"}}, {{target_pool, target, 4}});
    auto diagnostics = std::make_shared<TransferExecutionDiagnostics>();
    diagnostics->trace_id = info.trace_id;
    TransferRankSnapshot first;
    first.rank = 0;
    first.server_address = "worker-0:100";
    first.dispatch_state = TransferRpcDispatchState::SUBMITTED;
    first.cq_received = true;
    first.cq_ok = true;
    first.grpc_code = 0;
    first.response_received = true;
    first.response_valid = true;
    first.worker_error_code = static_cast<int64_t>(ErrorCode::EXECUTION_EXCEPTION);
    first.worker_error_message = "completed CRC failure";
    first.worker_copy_error = CacheCopyError::CRC_MISMATCH;
    first.block_errors = {CacheCopyError::CRC_MISMATCH};
    first.buffer_safe = true;
    TransferRankSnapshot second;
    second.rank = 1;
    second.server_address = "worker-1:100";
    second.dispatch_state = TransferRpcDispatchState::POSSIBLY_SUBMITTED;
    second.dispatch_error = "submission completion unknown";
    second.cq_received = true;
    second.cq_ok = false;
    second.grpc_code = 4;
    second.grpc_message = "RPC deadline";
    second.response_error = "no response";
    diagnostics->ranks = {first, second};
    auto failure = unknownResult();
    failure.block_errors = {CacheCopyError::RPC_FAILED};
    failure.execution_diagnostics = diagnostics;
    FrozenTask task(FrozenTask::prepare(std::move(info)), std::move(failure));
    task.freeze();
    const auto log = task.debugString();
    for (const auto* expected : {"trace=31",
                                 "operation=\"load\"",
                                 "request=73",
                                 "workflow=11",
                                 "parent_trace=12",
                                 "expected_rank_count=2",
                                 "state=FROZEN",
                                 "target_block_count=1",
                                 "source_tier=HOST",
                                 "target_tier=DEVICE",
                                 "unknown-source-pool",
                                 "target-log",
                                 "group_set=3",
                                 "path_index=9",
                                 "member=4",
                                 "dispatch_state=SUBMITTED",
                                 "dispatch_state=POSSIBLY_SUBMITTED",
                                 "dispatch_error=\"submission completion unknown\"",
                                 "cq_received=true",
                                 "cq_ok=false",
                                 "grpc_code=4",
                                 "grpc_message=\"RPC deadline\"",
                                 "response_received=true",
                                 "response_valid=true",
                                 "response_error=\"no response\"",
                                 "worker_error_message=\"completed CRC failure\"",
                                 "worker_copy_error=CRC_MISMATCH",
                                 "block_errors=[CRC_MISMATCH]",
                                 "buffer_safe=true",
                                 "buffer_safe=unknown"}) {
        EXPECT_NE(log.find(expected), std::string::npos) << expected << '\n' << log;
    }
    EXPECT_EQ(task.failure().execution_diagnostics->ranks.size(), 2u);
    task.unfreeze();
    EXPECT_NE(task.debugString().find("state=RELEASED"), std::string::npos);
}

TEST(FrozenTaskTest, RemoteReadPreservesStringIdentityAndRejectsAnotherTraceNamespace) {
    auto pool = makePool("remote-target");
    const auto block = heldBlock(pool);
    auto info = execution(33, {}, {{pool, block, 0}});
    info.trace_id = std::string("33");
    info.operation = "remote_read";
    info.descriptors.front().source_tier = Tier::REMOTE;
    info.descriptors.front().rank = 0;
    info.descriptors.front().group_tag = "default";
    info.descriptors.front().source_uri = "pace://original-source";
    auto diagnostics = std::make_shared<TransferExecutionDiagnostics>();
    diagnostics->trace_id = uint64_t{33};
    auto failure = unknownResult();
    failure.execution_diagnostics = diagnostics;
    EXPECT_ANY_THROW(FrozenTask(FrozenTask::prepare(info), failure));
    EXPECT_FALSE(pool->isFrozen(block));

    auto invalid = info;
    invalid.trace_id = std::string{};
    EXPECT_ANY_THROW(FrozenTask::prepare(std::move(invalid)));
    diagnostics->trace_id = info.trace_id;
    diagnostics->failure_reason = "rpc_timeout";
    TransferRankSnapshot rank;
    rank.rank = 0;
    rank.worker_transfer_status = 3;
    rank.sdk_failed = false;
    diagnostics->ranks.push_back(rank);
    FrozenTask task(FrozenTask::prepare(std::move(info)), std::move(failure));
    task.freeze();
    EXPECT_EQ(std::get<std::string>(task.execution().trace_id), "33");
    EXPECT_EQ(pool->treeRefCount(block), 2u);
    const auto log = task.debugString();
    for (const auto* field : {"trace=\"33\"", "execution_type=kvcm_read", "failure_reason=\"rpc_timeout\"",
                              "source_uri=\"pace://original-source\"", "group_tag=\"default\"",
                              "worker_transfer_status=3", "sdk_failed=false"}) {
        EXPECT_NE(log.find(field), std::string::npos) << field << '\n' << log;
    }
    task.unfreeze();
    task.unfreeze();
    EXPECT_EQ(pool->treeRefCount(block), 1u);
}

}  // namespace
}  // namespace rtp_llm
