#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskRecovery.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionCleanup.h"

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

TEST(FrozenTaskRecoveryTest, StopDuringQueryWaitsForExitAndDiscardsSafeReleasePermission) {
    auto cleanup = std::make_shared<TransferCompletionCleanup>();
    auto pool = makePool();
    const auto allocated = pool->malloc();
    ASSERT_TRUE(allocated.has_value());
    const auto block = *allocated;
    pool->incTreeRef(block, BlockTreeRefType::CACHE);
    FrozenTaskReclaimer coordinator(cleanup);
    TransferResult failure(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "local I/O pending"), CacheCopyError::COPY_FAILED);
    coordinator.freezeTargetBlocksWithTraceIDAndStillIOWorkerID({uint64_t{51}, {false, 1}, failure}, {{pool, block}}, {0});
    auto task = coordinator.frozenTasks().front();
    auto entered = std::make_shared<std::promise<void>>();
    auto in_query = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto released = release->get_future().share();
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto recovery = std::make_shared<FrozenTaskRecovery>(coordinator,
        [entered, released, calls](const auto& execution, const auto&, int) {
            ++*calls;
            entered->set_value();
            if (released.wait_for(5s) != std::future_status::ready)
                throw std::runtime_error("test query gate timed out");
            TransferCompletionQueryResult safe;
            safe.trace_id = execution.trace_id;
            safe.registry_type = execution.registry_type;
            safe.ranks.push_back({0, State::BUFFER_SAFE});
            safe.all_safe = true;
            return safe;
        });
    ASSERT_TRUE(recovery->start(10, 1000));
    ASSERT_EQ(in_query.wait_for(2s), std::future_status::ready);
    BoundedThread<void> stopped([recovery] { recovery->stop(); });
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

}  // namespace
}  // namespace rtp_llm
