#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadOperation.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

#include <atomic>
#include <thread>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::makeLoadOperationForTest;

TEST(LoadOperationTest, CompletionIsOnceAndSupportsLateReentrantSubscribers) {
    auto operation     = makeLoadOperationForTest();
    int  notifications = 0;
    operation->subscribe(2, [&](ErrorInfo error) {
        EXPECT_FALSE(error.ok());
        EXPECT_TRUE(operation->done_);
        operation->complete(true);
        operation->subscribe(3, [&](ErrorInfo late) {
            EXPECT_FALSE(late.ok());
            ++notifications;
        });
        ++notifications;
    });
    EXPECT_FALSE(operation->done_);
    operation->complete(false);
    operation->complete(true);
    EXPECT_EQ(notifications, 2);
    EXPECT_FALSE(operation->error_.ok());
    EXPECT_EQ(operation->error_.code(), ErrorCode::EXECUTION_EXCEPTION);
}

TEST(LoadOperationTest, SubscribeRacingCompletionDeliversEveryCallbackOnce) {
    for (int iteration = 0; iteration < 32; ++iteration) {
        auto             operation = makeLoadOperationForTest();
        std::atomic<int> notifications{0};
        std::thread      producer([&] { operation->complete(true); });
        for (int subscriber = 0; subscriber < 32; ++subscriber) {
            operation->subscribe(subscriber + 1, [&](ErrorInfo error) {
                EXPECT_TRUE(error.ok());
                ++notifications;
            });
        }
        producer.join();
        EXPECT_EQ(notifications.load(), 32);
        EXPECT_TRUE(operation->error_.ok());
    }
}

class SharedLoadOperationTest: public ::testing::Test {
protected:
    void SetUp() override {
        coordinator_ = std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    }
    void TearDown() override {
        coordinator_->shutdown();
    }

    std::shared_ptr<LoadAsyncContext> join(const std::vector<std::shared_ptr<LoadOperation>>& operations,
                                           size_t                                             local_matched_blocks  = 1,
                                           size_t                                             matched_device_blocks = 0,
                                           size_t matched_device_host_blocks = 0) {
        std::vector<TransferDescriptor> descriptors;
        for (const auto& operation : operations) {
            descriptors.push_back(operation->descriptor());
        }
        auto context = coordinator_->create(std::move(descriptors),
                                            std::vector<bool>(operations.size(), true),
                                            operations,
                                            local_matched_blocks,
                                            matched_device_blocks,
                                            matched_device_host_blocks);
        EXPECT_NE(context, nullptr);
        return context;
    }

    std::shared_ptr<LoadContextCoordinator> coordinator_;
};

TEST_F(SharedLoadOperationTest, AbortingOneSubscriberDoesNotCancelSharedLoad) {
    auto operation = makeLoadOperationForTest();
    auto aborted   = join({operation});
    auto waiting   = join({operation});
    ASSERT_EQ(operation->callbacks_.size(), 2u);
    ASSERT_TRUE(aborted->abortPending());
    EXPECT_EQ(operation->callbacks_.size(), 1u);
    ASSERT_TRUE(waiting->commit());
    EXPECT_FALSE(operation->done_);
    operation->complete(true);
    EXPECT_FALSE(aborted->success());
    EXPECT_TRUE(waiting->success());
}

TEST_F(SharedLoadOperationTest, CompletedSubscriberSkipsOperationLocksDuringNotificationAndDestruction) {
    for (bool success : {false, true}) {
        SCOPED_TRACE(success);
        auto first   = makeLoadOperationForTest();
        auto second  = makeLoadOperationForTest();
        auto context = join({first, second});
        ASSERT_TRUE(context->commit());
        size_t notifications = 0;
        context->onDone([&](ErrorInfo error) {
            EXPECT_EQ(error.ok(), success);
            EXPECT_EQ(context->remaining_join_count_.load(), 0u);
            ++notifications;
        });
        first->complete(success);
        std::unique_lock<std::mutex>               first_lock(first->mutex_);
        block_tree_cache_test::BoundedThread<void> completion([second, success] { second->complete(success); });
        auto                                       completion_status = completion.waitFor(std::chrono::seconds(1));
        first_lock.unlock();
        EXPECT_EQ(completion_status, std::future_status::ready);
        completion.get();
        EXPECT_TRUE(context->done());
        EXPECT_EQ(notifications, 1u);

        first_lock.lock();
        std::unique_lock<std::mutex>               second_lock(second->mutex_);
        std::weak_ptr<LoadAsyncContext>            weak = context;
        block_tree_cache_test::BoundedThread<void> destruction(
            [context = std::move(context)]() mutable { context.reset(); });
        auto destruction_status = destruction.waitFor(std::chrono::seconds(1));
        first_lock.unlock();
        second_lock.unlock();
        EXPECT_EQ(destruction_status, std::future_status::ready);
        destruction.get();
        EXPECT_TRUE(weak.expired());
    }
}

TEST_F(SharedLoadOperationTest, JoinedResultsAreCountedUnderContextLock) {
    auto first   = makeLoadOperationForTest();
    auto second  = makeLoadOperationForTest();
    auto context = join({first, second});
    ASSERT_TRUE(context->commit());
    auto                                       barrier = std::make_shared<block_tree_cache_test::CallbackBarrier>();
    std::unique_lock<std::mutex>               lock(context->mutex_);
    block_tree_cache_test::BoundedThread<void> first_completion([first, barrier] {
        barrier->enterAndWait();
        first->complete(true);
    });
    block_tree_cache_test::BoundedThread<void> second_completion([second, barrier] {
        barrier->enterAndWait();
        second->complete(true);
    });
    EXPECT_TRUE(barrier->waitUntilEnteredFor(2, std::chrono::seconds(1)));
    barrier->release();
    EXPECT_EQ(first_completion.waitFor(std::chrono::milliseconds(50)), std::future_status::timeout);
    EXPECT_EQ(second_completion.waitFor(std::chrono::milliseconds(50)), std::future_status::timeout);
    EXPECT_EQ(context->remaining_join_count_.load(), 2u);
    EXPECT_EQ(context->remaining_transfer_count_, 2u);
    EXPECT_FALSE(context->join_trace_finished_.load());
    lock.unlock();
    ASSERT_EQ(first_completion.waitFor(std::chrono::seconds(1)), std::future_status::ready);
    first_completion.get();
    ASSERT_EQ(second_completion.waitFor(std::chrono::seconds(1)), std::future_status::ready);
    second_completion.get();
    EXPECT_EQ(context->remaining_join_count_.load(), 0u);
    EXPECT_TRUE(context->join_trace_finished_.load());
    EXPECT_TRUE(context->success());
}

TEST_F(SharedLoadOperationTest, CancelledSubscriberDestructionSkipsOperationLocks) {
    auto first     = makeLoadOperationForTest();
    auto second    = makeLoadOperationForTest();
    auto cancelled = join({first, second});
    auto waiting   = join({first, second});
    ASSERT_TRUE(waiting->commit());
    ASSERT_TRUE(cancelled->abortPending());
    EXPECT_EQ(first->callbacks_.size(), 1u);
    EXPECT_EQ(second->callbacks_.size(), 1u);
    std::unique_lock<std::mutex>               first_lock(first->mutex_);
    std::unique_lock<std::mutex>               second_lock(second->mutex_);
    std::weak_ptr<LoadAsyncContext>            weak = cancelled;
    block_tree_cache_test::BoundedThread<void> destruction(
        [cancelled = std::move(cancelled)]() mutable { cancelled.reset(); });
    auto status = destruction.waitFor(std::chrono::seconds(1));
    first_lock.unlock();
    second_lock.unlock();
    EXPECT_EQ(status, std::future_status::ready);
    destruction.get();
    EXPECT_TRUE(weak.expired());
    first->complete(true);
    second->complete(true);
    EXPECT_TRUE(waiting->success());
}

TEST_F(SharedLoadOperationTest, CancellationAfterCompletionTakesCallbacksDoesNotRepeatUnsubscription) {
    auto operation = makeLoadOperationForTest();
    auto cancelled = join({operation});
    auto waiting   = join({operation});
    ASSERT_TRUE(waiting->commit());
    auto barrier = std::make_shared<block_tree_cache_test::CallbackBarrier>();
    operation->subscribe(0, [barrier](ErrorInfo) { barrier->enterAndWait(); });
    block_tree_cache_test::BoundedThread<void> completion([operation] { operation->complete(true); });
    if (!barrier->waitUntilEnteredFor(1, std::chrono::seconds(1))) {
        barrier->release();
        completion.get();
        FAIL() << "completion did not take subscriptions";
    }
    EXPECT_TRUE(cancelled->abortPending());
    EXPECT_FALSE(cancelled->success());
    std::unique_lock<std::mutex>               operation_lock(operation->mutex_);
    std::weak_ptr<LoadAsyncContext>            weak = cancelled;
    block_tree_cache_test::BoundedThread<void> destruction(
        [cancelled = std::move(cancelled)]() mutable { cancelled.reset(); });
    auto status = destruction.waitFor(std::chrono::seconds(1));
    operation_lock.unlock();
    barrier->release();
    completion.get();
    destruction.get();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(waiting->remaining_join_count_.load(), 0u);
    EXPECT_TRUE(waiting->success());
}

TEST_F(SharedLoadOperationTest, SubscriberDestructionDoesNotLeaveAReferenceCycle) {
    auto                            operation = makeLoadOperationForTest();
    std::weak_ptr<LoadAsyncContext> weak;
    {
        auto context = join({operation});
        weak         = context;
    }
    EXPECT_TRUE(weak.expired());
    EXPECT_TRUE(operation->callbacks_.empty());
    operation->complete(true);
}

TEST_F(SharedLoadOperationTest, CancelledSubscribersDoNotAccumulateWhileTransfersArePending) {
    auto host    = makeLoadOperationForTest(Tier::HOST);
    auto disk    = makeLoadOperationForTest(Tier::DISK);
    auto waiting = join({host, disk});
    ASSERT_TRUE(waiting->commit());
    std::vector<std::shared_ptr<LoadAsyncContext>> cancelled;
    for (size_t index = 0; index < 64; ++index) {
        auto context = join({host, disk});
        ASSERT_TRUE(context->abortPending());
        cancelled.emplace_back(std::move(context));
        EXPECT_EQ(host->callbacks_.size(), 1u);
        EXPECT_EQ(disk->callbacks_.size(), 1u);
    }
    host->complete(true);
    EXPECT_FALSE(waiting->done());
    disk->complete(true);
    EXPECT_TRUE(waiting->success());
    EXPECT_TRUE(host->callbacks_.empty());
    EXPECT_TRUE(disk->callbacks_.empty());
}

TEST_F(SharedLoadOperationTest, DestroyingCommittedSubscriberRemovesPendingCallback) {
    auto operation = makeLoadOperationForTest();
    auto context   = join({operation});
    ASSERT_TRUE(context->commit());
    EXPECT_EQ(operation->callbacks_.size(), 1u);
    context.reset();
    EXPECT_TRUE(operation->callbacks_.empty());
    EXPECT_FALSE(operation->done_);
    operation->complete(true);
}

TEST_F(SharedLoadOperationTest, FailedCommitRemovesJoinedSubscriptions) {
    coordinator_   = std::make_shared<LoadContextCoordinator>([](const auto&) { return false; }, [](auto&) {});
    auto operation = makeLoadOperationForTest();
    auto context   = join({operation});
    ASSERT_FALSE(context->commit());
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(operation->callbacks_.empty());
    EXPECT_FALSE(operation->done_);
    operation->complete(true);
    EXPECT_FALSE(context->success());
}

TEST_F(SharedLoadOperationTest, CancellationRacingCompletionPreservesOtherSubscribers) {
    for (int iteration = 0; iteration < 32; ++iteration) {
        auto operation = makeLoadOperationForTest();
        auto cancelled = join({operation});
        auto waiting   = join({operation});
        ASSERT_TRUE(waiting->commit());
        std::thread producer([&] { operation->complete(true); });
        EXPECT_TRUE(cancelled->abortPending());
        producer.join();
        EXPECT_FALSE(cancelled->success());
        EXPECT_TRUE(waiting->success());
        EXPECT_TRUE(operation->callbacks_.empty());
    }
}

TEST_F(SharedLoadOperationTest, CompletionBeforeCommitIsRemembered) {
    auto operation = makeLoadOperationForTest();
    operation->complete(true);
    auto context = join({operation});
    EXPECT_FALSE(context->done());
    ASSERT_TRUE(context->commit());
    EXPECT_TRUE(context->success());
}

TEST_F(SharedLoadOperationTest, AggregationPreservesHostFallbackAndWaitsForEveryDependency) {
    auto host    = makeLoadOperationForTest(Tier::HOST);
    auto disk    = makeLoadOperationForTest(Tier::DISK);
    auto context = join({host, disk}, 3, 1, 2);
    ASSERT_TRUE(context->commit());
    disk->complete(false);
    EXPECT_FALSE(context->done());
    host->complete(true);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->deviceReadyBlocks(), 2u);
}

TEST(LoadOperationTest, RollbackReleasesProtectionBeforeDestruction) {
    for (const Tier tier : {Tier::HOST, Tier::DISK}) {
        auto       operation = makeLoadOperationForTest(tier);
        const auto group     = operation->group_set_;
        const auto target    = operation->descriptor().target_blocks.front();
        EXPECT_EQ(group->hostPool()->referencedBlocksNum(BlockTreeRefType::LOAD), tier == Tier::HOST ? 1u : 0u);
        EXPECT_EQ(group->diskPool()->referencedBlocksNum(BlockTreeRefType::LOAD), tier == Tier::DISK ? 1u : 0u);
        EXPECT_EQ(group->devicePools().front()->treeRefCount(target), 1u);
        unreferenceBlocks({group},
                          {{0, tier, {{nullptr, operation->descriptor().source_blocks}}},
                           {0, Tier::DEVICE, {{nullptr, operation->descriptor().target_blocks}}}},
                          BlockTreeRefType::LOAD);
        EXPECT_EQ(group->hostPool()->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
        EXPECT_EQ(group->diskPool()->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
        EXPECT_FALSE(group->devicePools().front()->isAllocated(target));
        const auto reused_source = group->allocateSingleBlock(tier, BlockTreeRefType::LOAD);
        ASSERT_EQ(reused_source, operation->descriptor().source_blocks.front());
        operation.reset();
        EXPECT_EQ(group->hostPool()->referencedBlocksNum(BlockTreeRefType::LOAD), tier == Tier::HOST ? 1u : 0u);
        EXPECT_EQ(group->diskPool()->referencedBlocksNum(BlockTreeRefType::LOAD), tier == Tier::DISK ? 1u : 0u);
        group->unreferenceBlocks(tier, {reused_source}, BlockTreeRefType::LOAD);
    }
}

TEST(LoadOperationTest, SettlementReleasesProtectionWhileRequestStillHoldsOperation) {
    using namespace block_transfer_engine_test;
    const auto config      = makeTestGroupBase();
    const auto kv_bytes    = config.group.kvBlockStrideBytes();
    auto       device_pool = makeTestDevicePool({{kv_bytes, config.group.kvScaleStrideBytes()}}, 2, "load_operation");
    auto       host_pool   = makeHostPool(kv_bytes, 2);
    auto       group       = makeTestGroupSet(0, makeTestTopology({config}), {"group0"}, {device_pool}, host_pool);
    const auto source      = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::LOAD);
    const auto targets     = device_pool->malloc(1);
    ASSERT_TRUE(targets.has_value());
    group->referenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{nullptr, *targets}}}, BlockTreeRefType::LOAD);
    auto operation = std::make_shared<LoadOperation>(TransferDescriptor::hostToDevice(0, source, *targets), group, 1);

    auto joined          = TransferDescriptor::hostToDevice(0, source, {});
    joined.target_blocks = operation->descriptor().target_blocks;
    referenceBlocks({group}, {{0, Tier::DEVICE, {{nullptr, joined.target_blocks}}}});
    EXPECT_EQ(joined.target_blocks, *targets);
    EXPECT_EQ(device_pool->treeRefCount(targets->front()), 1u);
    EXPECT_EQ(host_pool->treeRefCount(source), 1u);
    unreferenceBlocks({group},
                      {{0, Tier::HOST, {{nullptr, operation->descriptor().source_blocks}}},
                       {0, Tier::DEVICE, {{nullptr, operation->descriptor().target_blocks}}}},
                      BlockTreeRefType::LOAD);
    operation->complete(true);
    EXPECT_TRUE(operation->error_.ok());
    EXPECT_EQ(device_pool->treeRefCount(targets->front()), 0u);
    EXPECT_EQ(device_pool->refCount(targets->front()), 1u);
    EXPECT_FALSE(host_pool->isAllocated(source));
    group->unreferenceBlocks(MultiNodeResource{joined.group_set_id, Tier::DEVICE, {{nullptr, joined.target_blocks}}});
    EXPECT_FALSE(device_pool->isAllocated(targets->front()));
    operation.reset();
}

}  // namespace
}  // namespace rtp_llm
