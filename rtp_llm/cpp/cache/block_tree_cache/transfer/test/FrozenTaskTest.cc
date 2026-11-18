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
                                    std::vector<FrozenTask::TargetBlock> targets) {
    FrozenTask::ExecutionInfo info;
    info.trace_id           = std::to_string(trace);

    info.uses_rpc             = true;

    info.participant_ranks     = {{0, "worker-0:100"}, {1, "worker-1:100"}};
    info.target_blocks = std::move(targets);
    return info;
}

TransferResult unknownResult() {
    return TransferResult(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "RPC timed out"),
                          CacheCopyError::RPC_FAILED);
}

TEST(FrozenTaskTest, PreparationDoesNotAcquireReferencesAndOnlyTargetsFreeze) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto source_pool = makePool("source");
    auto target_pool = makePool("target");
    const auto source = heldBlock(source_pool);
    const auto target = heldBlock(target_pool);
    auto info = execution(21, {{target_pool, target}});
    auto prepared = FrozenTask::prepare(std::move(info));
    EXPECT_EQ(source_pool->treeRefCount(source), 1u);
    EXPECT_EQ(target_pool->treeRefCount(target), 1u);

    FrozenTask task(std::move(prepared), unknownResult());
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::PREPARED);
    EXPECT_FALSE(task.completeRecovery(*cleanup));
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

    EXPECT_TRUE(task.completeRecovery(*cleanup));
    const auto released = task.snapshot();
    EXPECT_EQ(released.state, FrozenTask::State::RELEASED);
    EXPECT_GT(released.released_at_us, 0);
    EXPECT_FALSE(task.completeRecovery(*cleanup));
    task.freeze();
    EXPECT_EQ(task.snapshot().released_at_us, released.released_at_us);
    EXPECT_EQ(task.snapshot().frozen_at_us, frozen.frozen_at_us);
    EXPECT_FALSE(target_pool->isFrozen(target));
    EXPECT_EQ(target_pool->treeRefCount(target), 1u);
    EXPECT_EQ(source_pool->treeRefCount(source), 1u);
}

TEST(FrozenTaskTest, DeduplicatesTargetSlotsAcrossPools) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto first_pool  = makePool("first");
    auto second_pool = makePool("second");
    const auto first = heldBlock(first_pool);
    const auto second = heldBlock(second_pool);
    ASSERT_EQ(first, second);
    auto info = execution(22, {{first_pool, first}, {first_pool, first}});
    info.target_blocks.push_back({first_pool, first});
    info.target_blocks.push_back({second_pool, second});
    FrozenTask task(FrozenTask::prepare(std::move(info)), unknownResult());
    task.freeze();
    EXPECT_EQ(task.snapshot().target_block_count, 2u);
    EXPECT_EQ(first_pool->treeRefCount(first), 2u);
    EXPECT_EQ(second_pool->treeRefCount(second), 2u);
    task.completeRecovery(*cleanup);
    EXPECT_EQ(first_pool->treeRefCount(first), 1u);
    EXPECT_EQ(second_pool->treeRefCount(second), 1u);
}

TEST(FrozenTaskTest, SharedTargetsKeepIndependentTaskReferencesAcrossBusinessRelease) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto pool = makePool("shared");
    const auto block = heldBlock(pool);
    FrozenTask first(FrozenTask::prepare(execution(23, {{pool, block}})), unknownResult());
    FrozenTask second(FrozenTask::prepare(execution(24, {{pool, block}})), unknownResult());
    first.freeze();
    second.freeze();
    EXPECT_EQ(pool->treeRefCount(block), 3u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);

    pool->decTreeRef(block, BlockTreeRefType::CACHE);
    EXPECT_TRUE(first.completeRecovery(*cleanup));
    EXPECT_FALSE(first.completeRecovery(*cleanup));
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_TRUE(pool->isFrozen(block));
    EXPECT_EQ(pool->treeRefCount(block), 1u);
    EXPECT_TRUE(second.completeRecovery(*cleanup));
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
}

TEST(FrozenTaskTest, DestructionDoesNotReleaseFrozenTargets) {
    auto pool = makePool("retained");
    const auto block = heldBlock(pool);
    {
        FrozenTask task(FrozenTask::prepare(execution(25, {{pool, block}})), unknownResult());
        task.freeze();
    }
    EXPECT_EQ(pool->treeRefCount(block), 2u);
    EXPECT_TRUE(pool->isFrozen(block));
    // Test cleanup releases the intentionally retained reference explicitly.
    pool->decTreeRef(block, BlockTreeRefType::FROZEN);
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
}

TEST(FrozenTaskTest, PoolOwnershipOutlivesCallerAndSourceAllocation) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto source_pool = makePool("source-lifetime");
    auto target_pool = makePool("target-lifetime");
    const auto source = heldBlock(source_pool);
    const auto target = heldBlock(target_pool);
    std::weak_ptr<FrozenTestPool> weak_source = source_pool;
    std::weak_ptr<FrozenTestPool> weak_target = target_pool;
    {
        FrozenTask task(FrozenTask::prepare(execution(26, {{target_pool, target}})),
                        unknownResult());
        task.freeze();
        source_pool->decTreeRef(source, BlockTreeRefType::CACHE);
        EXPECT_FALSE(source_pool->isAllocated(source));
        target_pool->decTreeRef(target, BlockTreeRefType::CACHE);
        source_pool.reset();
        target_pool.reset();
        EXPECT_TRUE(weak_source.expired());
        EXPECT_FALSE(weak_target.expired());
        task.completeRecovery(*cleanup);
        EXPECT_EQ(weak_target.lock()->freeBlocksNum(), weak_target.lock()->totalBlocksNum());
    }
    EXPECT_TRUE(weak_source.expired());
    EXPECT_TRUE(weak_target.expired());
}

TEST(FrozenTaskTest, ThrowingCapacityNotificationsCannotInterruptOrRepeatBatchRelease) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
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
                                                 {{first_pool, first},
                                                  {first_pool, other},
                                                  {first_pool, first},
                                                  {second_pool, second}})),
                    unknownResult());
    task.freeze();
    first_pool->decTreeRef({first, other}, BlockTreeRefType::CACHE);
    second_pool->decTreeRef(second, BlockTreeRefType::CACHE);
    EXPECT_TRUE(task.completeRecovery(*cleanup));
    EXPECT_FALSE(task.completeRecovery(*cleanup));
    EXPECT_EQ(first_notifications, 1u);
    EXPECT_EQ(second_notifications, 1u);
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_EQ(first_pool->freeBlocksNum(), first_pool->totalBlocksNum());
    EXPECT_EQ(second_pool->freeBlocksNum(), second_pool->totalBlocksNum());
}

TEST(FrozenTaskTest, ReleaseNotificationsCanInspectAndReenterTaskWithoutRepeatingReferences) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto pool = makePool("reentrant-notify");
    const auto block = heldBlock(pool);
    FrozenTask* observed = nullptr;
    size_t notifications = 0;
    pool->setCapacityChangeCallback([&] {
        ++notifications;
        ASSERT_NE(observed, nullptr);
        EXPECT_EQ(observed->snapshot().state, FrozenTask::State::RELEASING);
        EXPECT_EQ(observed->snapshot().state, FrozenTask::State::RELEASING);
        EXPECT_FALSE(observed->completeRecovery(*cleanup));
        observed->freeze();
    });
    FrozenTask task(FrozenTask::prepare(execution(28, {{pool, block}})), unknownResult());
    observed = &task;
    task.freeze();
    EXPECT_TRUE(task.completeRecovery(*cleanup));
    EXPECT_EQ(notifications, 1u);
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_EQ(pool->treeRefCount(block), 1u);
}

TEST(FrozenTaskTest, ConcurrentRepeatedFreezeAndRecoveryOwnOneReference) {
    auto pool = makePool("concurrent");
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    const auto block = heldBlock(pool);
    std::atomic<size_t> notifications{0};
    pool->setCapacityChangeCallback([&] { ++notifications; });
    FrozenTask task(FrozenTask::prepare(execution(29, {{pool, block}})), unknownResult(),
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
            if (task.completeRecovery(*cleanup)) {
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
        EXPECT_FALSE(observed->completeRecovery(*cleanup));
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
    FrozenTask task(FrozenTask::prepare(execution(36, {{first_pool, first}, {second_pool, second}})),
                    unknownResult(),
                    cleanup->prepare({true, {{0, "worker-0:100"}, {1, "worker-1:100"}}}, uint64_t{36}));
    observed = &task;
    EXPECT_FALSE(task.completeRecovery(*cleanup));
    EXPECT_EQ(cleanup->size(), 0u);
    task.freeze();
    first_pool->decTreeRef(first, BlockTreeRefType::CACHE);
    second_pool->decTreeRef(second, BlockTreeRefType::CACHE);
    EXPECT_TRUE(task.completeRecovery(*cleanup));
    EXPECT_FALSE(task.completeRecovery(*cleanup));
    EXPECT_EQ(notifications, 2u);
    EXPECT_EQ(task.snapshot().state, FrozenTask::State::RELEASED);
    EXPECT_EQ(cleanup->size(), 1u);
    const auto summary = cleanup->drain([&](const auto& target, const auto& traces, int) {
        EXPECT_TRUE(target.remote);
        EXPECT_EQ(target.peers.size(), 2u);
        EXPECT_EQ(traces, (std::vector<TransferCompletionKey>{uint64_t{36}}));
        EXPECT_FALSE(first_pool->isFrozen(first));
        EXPECT_FALSE(second_pool->isFrozen(second));
        TransferCompletionForgetResult result;
        TransferCompletionForgetEntryResult entry;
        entry.registry_type = traces.front().registry_type;
        entry.trace_id = traces.front().trace_id;
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
        FrozenTask task(FrozenTask::prepare(execution(37, {{pool, block}})), unknownResult(),
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
            FrozenTask::prepare(execution(38, {{target_pool, target}})),
            unknownResult(),
            cleanup->prepare({true, {{0, "worker-0:100"}, {1, "worker-1:100"}}}, uint64_t{38}));
        weak_task = task;
        task->freeze();
        source_pool->decTreeRef(source, BlockTreeRefType::CACHE);
        target_pool->decTreeRef(target, BlockTreeRefType::CACHE);
        EXPECT_TRUE(task->completeRecovery(*cleanup));
        EXPECT_EQ(cleanup->size(), 1u);
    }
    EXPECT_TRUE(weak_task.expired());
    EXPECT_TRUE(weak_source.expired());
    EXPECT_TRUE(weak_target.expired());
    EXPECT_EQ(cleanup->size(), 1u);
}

TEST(FrozenTaskTest, RecoveryQueriesPreserveOriginalFailureAndDoNotReleaseTargets) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto pool = makePool("recovery-observation");
    const auto block = heldBlock(pool);
    auto info = execution(34, {{pool, block}});
    auto failure = unknownResult();
    const auto original_error = failure.status.ToString();
    FrozenTask task(FrozenTask::prepare(std::move(info)), std::move(failure));
    task.freeze();

    task.recordRecoveryQuery(nullptr);
    const auto initial = task.recoverySnapshot();
    EXPECT_EQ(initial.query_count, 0u);
    EXPECT_EQ(initial.last_query_at_us, 0);
    EXPECT_EQ(initial.last_query, nullptr);

    TransferCompletionQueryResult pending;
    pending.trace_id = "34";
    pending.error = "rank 1 query timed out";
    pending.ranks = {{0, "worker-0:100", TransferTaskRegistry::State::RUNNING, 0, "", true, true},
                     {1, "worker-1:100", TransferTaskRegistry::State::UNKNOWN, 4, "deadline", false, false}};
    auto pending_result = std::make_shared<const TransferCompletionQueryResult>(std::move(pending));
    task.recordRecoveryQuery(pending_result);
    const auto first = task.recoverySnapshot();
    EXPECT_EQ(first.query_count, 1u);
    EXPECT_GT(first.last_query_at_us, 0);
    EXPECT_EQ(first.last_query, pending_result);
    TransferCompletionQueryResult safe;
    safe.trace_id = "34";
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
    EXPECT_TRUE(task.completeRecovery(*cleanup));
    EXPECT_EQ(task.recoverySnapshot().last_query, safe_result);
}

TEST(FrozenTaskTest, ConcurrentRecoveryQueriesRetainCountAndLatestStringTraceObservation) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto pool = makePool("concurrent-observation");
    const auto block = heldBlock(pool);
    auto info = execution(35, {{pool, block}});
    info.registry_type = RegistryType::KVCM_READ;
    info.trace_id = std::string("read-35");
    FrozenTask task(FrozenTask::prepare(std::move(info)), unknownResult());
    task.freeze();
    TransferCompletionQueryResult query;
    query.registry_type = RegistryType::KVCM_READ;
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
    EXPECT_TRUE(task.completeRecovery(*cleanup));
}

TEST(FrozenTaskTest, InvalidTargetPreparationDoesNotAcquireAnyFrozenHold) {
    auto pool = makePool("invalid");
    const auto block = heldBlock(pool);
    EXPECT_ANY_THROW(FrozenTask::prepare(execution(30, {{pool, block}, {pool, NULL_BLOCK_IDX}})));
    EXPECT_EQ(pool->treeRefCount(block), 1u);
    EXPECT_FALSE(pool->isFrozen(block));
    EXPECT_ANY_THROW(FrozenTask::prepare(execution(30, {{nullptr, block}})));
    EXPECT_ANY_THROW(FrozenTask::prepare(execution(0, {{pool, block}})));
    EXPECT_ANY_THROW(FrozenTask(FrozenTask::prepare(execution(30, {{pool, block}})),
                                TransferResult(TransferStatus::OK)));
    EXPECT_EQ(pool->treeRefCount(block), 1u);
}

TEST(FrozenTaskTest, RemoteReadPreservesStringIdentityAndRejectsEmptyTrace) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto pool = makePool("remote-target");
    const auto block = heldBlock(pool);
    auto info = execution(33, {{pool, block}});
    info.registry_type = RegistryType::KVCM_READ;
    info.trace_id = std::string("33");

    auto failure = unknownResult();
    EXPECT_FALSE(pool->isFrozen(block));

    auto invalid = info;
    invalid.trace_id = std::string{};
    EXPECT_ANY_THROW(FrozenTask::prepare(std::move(invalid)));
    FrozenTask task(FrozenTask::prepare(std::move(info)), std::move(failure));
    task.freeze();
    EXPECT_EQ(task.execution().trace_id, "33");
    EXPECT_EQ(pool->treeRefCount(block), 2u);
    task.completeRecovery(*cleanup);
    task.completeRecovery(*cleanup);
    EXPECT_EQ(pool->treeRefCount(block), 1u);
}

}  // namespace
}  // namespace rtp_llm
