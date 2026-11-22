#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionTypes.h"

#include <memory>
#include <stdexcept>
#include <string>
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
    info.key = trace;
    info.uses_rpc = true;
    info.rank_count = 2;
    info.target_blocks = std::move(targets);
    return info;
}

TransferResult unknownResult() {
    return TransferResult(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "RPC timed out"),
                          CacheCopyError::RPC_FAILED);
}

TEST(FrozenTaskTest, OnlyTheLastPendingWorkerReleasesTheTrace) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    FrozenTaskReclaimer coordinator(cleanup);
    const auto pool = makePool("pending-workers");
    const auto block = heldBlock(pool);
    const TransferCompletionKey key{uint64_t{41}};
    const auto result = unknownResult();
    coordinator.freezeTargetBlocksWithTraceIDAndStillIOWorkerID({key, {true, 3}, result}, {{pool, block}}, {1, 2, 1});
    EXPECT_FALSE(coordinator.unfreezeTargetBlocksInOneWorkerWithTraceID(key, 0));
    EXPECT_FALSE(coordinator.unfreezeTargetBlocksInOneWorkerWithTraceID(key, 2));
    EXPECT_FALSE(coordinator.unfreezeTargetBlocksInOneWorkerWithTraceID(key, 2));
    EXPECT_TRUE(pool->isFrozen(block));
    EXPECT_EQ(cleanup->size(), 0u);
    EXPECT_TRUE(coordinator.unfreezeTargetBlocksInOneWorkerWithTraceID(key, 1));
    EXPECT_FALSE(coordinator.unfreezeTargetBlocksInOneWorkerWithTraceID(key, 1));
    EXPECT_FALSE(pool->isFrozen(block));
    EXPECT_TRUE(coordinator.frozenTasks().empty());
    EXPECT_EQ(cleanup->size(), 1u);
    pool->decTreeRef(block, BlockTreeRefType::CACHE);
}

TEST(FrozenTaskTest, ReleasesEachOwnedTargetOnceBeforeSubmittingForget) {
    TransferCompletionCleanup cleanup;
    auto primary = makePool("primary");
    auto secondary = makePool("secondary");
    const auto shared = heldBlock(primary);
    const auto unique = heldBlock(primary);
    const auto other = heldBlock(secondary);
    ASSERT_EQ(shared, other);
    size_t notifications = 0;
    primary->setCapacityChangeCallback([&] {
        ++notifications;
        EXPECT_EQ(cleanup.size(), 0u);
        EXPECT_FALSE(primary->isAllocated(unique));
        EXPECT_TRUE(primary->isFrozen(shared));
        throw std::runtime_error("capacity notification failed");
    });
    secondary->setCapacityChangeCallback([&] {
        ++notifications;
        EXPECT_EQ(cleanup.size(), 0u);
        EXPECT_FALSE(secondary->isAllocated(other));
    });
    const auto info = execution(36, {{primary, shared}, {primary, shared}, {primary, unique}, {secondary, other}});
    FrozenTask first(FrozenTask::prepare(info), unknownResult(),
                     cleanup.prepare({true, 2}, uint64_t{36}));
    primary->setCapacityChangeCallback({});
    FrozenTask second(FrozenTask::prepare(execution(37, {{primary, shared}})), unknownResult());
    first.freeze();
    second.freeze();
    EXPECT_EQ(first.snapshot().target_block_count, 3u);
    primary->decTreeRef({unique, shared}, BlockTreeRefType::CACHE);
    secondary->decTreeRef(other, BlockTreeRefType::CACHE);
    EXPECT_TRUE(first.completeRecovery(cleanup));
    EXPECT_FALSE(first.completeRecovery(cleanup));
    EXPECT_EQ(notifications, 2u);
    EXPECT_EQ(cleanup.size(), 1u);
    EXPECT_EQ(primary->treeRefCount(shared), 1u);
    EXPECT_TRUE(second.completeRecovery(cleanup));
    EXPECT_FALSE(primary->isAllocated(shared));
}
}
}
