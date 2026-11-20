#include <cstdlib>
#include <exception>
#include <new>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadOperation.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace {
thread_local size_t                                    last_descriptor_block_bytes = 0;
thread_local size_t                                    remaining_descriptor_copies = 0;
thread_local const size_t*                             pending_eviction_count      = nullptr;
thread_local int                                       allocation_countdown        = -1;
thread_local bool                                      allocation_failed           = false;
thread_local rtp_llm::TransferTask::Clock::time_point* descriptor_assembly_time    = nullptr;
thread_local const rtp_llm::GroupSetResource*          first_settlement_resource   = nullptr;
thread_local const rtp_llm::GroupSetResource*          second_settlement_resource  = nullptr;
}  // namespace

// Keep allocation interposition in this dedicated test binary.
[[gnu::noinline]] void* operator new(std::size_t size) {
    if (descriptor_assembly_time) {
        if (remaining_descriptor_copies == 0) {
            // The next allocation follows completion of the final descriptor's two block-list copies.
            *descriptor_assembly_time = rtp_llm::TransferTask::Clock::now();
            descriptor_assembly_time  = nullptr;
        } else if (size == last_descriptor_block_bytes) {
            --remaining_descriptor_copies;
        }
    }
    if (allocation_countdown >= 0 && (!pending_eviction_count || *pending_eviction_count != 0)
        && (!first_settlement_resource
            || (first_settlement_resource->transfer_state == rtp_llm::GroupSetTransferState::IDLE
                && !second_settlement_resource->load_operation))
        && allocation_countdown-- == 0) {
        allocation_failed = true;
        throw std::bad_alloc();
    }
    if (void* pointer = std::malloc(size ? size : 1)) {
        return pointer;
    }
    throw std::bad_alloc();
}
[[gnu::noinline]] void* operator new[](std::size_t size) {
    return ::operator new(size);
}
[[gnu::noinline]] void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

namespace rtp_llm {
namespace {

TEST(LoadTaskTest, DeadlinesStartAfterLastDescriptorIsAssembled) {
    auto coordinator = std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    const BlockIdList last_blocks(257, 2);
    auto              context = coordinator->create(
        {TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, last_blocks)},
        {false, false},
        {},
        2);
    TransferTask::Clock::time_point assembled_at;
    last_descriptor_block_bytes = last_blocks.size() * sizeof(BlockIdxType);
    remaining_descriptor_copies = 2;
    descriptor_assembly_time    = &assembled_at;
    const auto next_trace_id    = TransferTask::next_trace_id_.load();
    auto       task             = LoadTask::create(context, 10, 20);
    descriptor_assembly_time    = nullptr;
    EXPECT_EQ(TransferTask::next_trace_id_.load() - next_trace_id, 2u);
    ASSERT_EQ(remaining_descriptor_copies, 0u);
    ASSERT_NE(assembled_at, TransferTask::Clock::time_point{});
    EXPECT_EQ(task->load_descs.back().target_blocks, last_blocks);
    EXPECT_EQ(task->disk_to_device_task.descriptors().back().target_blocks, last_blocks);
    EXPECT_GE(task->host_to_device_task.deadline() - std::chrono::milliseconds(10), assembled_at);
    EXPECT_GE(task->disk_to_device_task.deadline() - std::chrono::milliseconds(20), assembled_at);
}

TEST(LoadFailureDeathTest, MatchAllocationFailuresTerminateWhileCacheLockIsHeld) {
    bool completed = false;
    for (int failure_index = 0; failure_index < 256 && !completed; ++failure_index) {
        SCOPED_TRACE(failure_index);
        EXPECT_EXIT(([&] {
                        alarm(5);
                        auto                          operation = block_tree_cache_test::makeLoadOperationForTest();
                        BlockTree                     tree({operation->group_set_});
                        BlockTreeTaskPool             pool(1, 1, "load_failure");
                        BlockTreeCacheMetricsReporter metrics(nullptr);
                        std::mutex                    mutex;
                        BlockTreeEvictor              evictor(
                            &tree,
                            EvictionPolicy::LRU,
                            EvictionPolicy::LRU,
                            EvictionPolicy::FIFO,
                            nullptr,
                            &pool,
                            metrics,
                            mutex,
                            10000,
                            10000,
                            8,
                            [](Tier) { return true; },
                            [](bool) {},
                            0);
                        BlockTreeLoader loader(
                            &tree, evictor, nullptr, &pool, metrics, mutex, 10000, 10000, true, nullptr, [](bool) {});
                        GroupSetResource resource;
                        resource.host_block = operation->descriptor().source_blocks.front();
                        const CacheKeysType keys{1, 2};
                        auto                inserted = tree.insertNode(keys, {{resource}, {resource}}, true, false);
                        evictor.onInserted(inserted);
                        auto path = tree.findNode(keys);
                        if (path.size() != 2) {
                            std::_Exit(75);
                        }
                        evictor.suspendCandidate(path.back(), 0, Tier::HOST);
                        auto& joined          = path.back()->group_set_resources[0];
                        joined.transfer_state = GroupSetTransferState::LOADING;
                        joined.load_operation = operation;
                        tree.refreshPublishedState(path.back());
                        std::function<bool(size_t)> valid_prefix;
                        std::vector<bool>           candidate_valid;
                        loader.validMatch(path, candidate_valid, valid_prefix);
                        std::lock_guard<std::mutex> lock(mutex);
                        std::set_terminate([] { std::_Exit(allocation_failed ? 71 : 73); });
                        allocation_countdown = failure_index;
                        try {
                            auto match = loader.createMatchResult(
                                path, candidate_valid, keys, valid_prefix, false, std::nullopt);
                            allocation_countdown = -1;
                            if (!match.async_context || match.async_context->newLoadIndices().size() != 1
                                || match.async_context->joinedLoads() != std::vector<bool>({false, true})) {
                                std::_Exit(75);
                            }
                            std::_Exit(allocation_failed ? 74 : 0);
                        } catch (...) {
                            std::_Exit(72);
                        }
                    }()),
                    [&](int status) {
                        completed = WIFEXITED(status) && WEXITSTATUS(status) == 0;
                        return completed || (WIFEXITED(status) && WEXITSTATUS(status) == 71);
                    },
                    "");
    }
    EXPECT_TRUE(completed);
}

TEST(LoadFailureDeathTest, EvictionSubmissionAllocationFailuresRollBack) {
    bool completed = false;
    for (int failure_index = -1; failure_index < 128 && !completed; ++failure_index) {
        SCOPED_TRACE(failure_index);
        ASSERT_EXIT(
            ([&] {
                alarm(5);
                auto                     device = block_tree_cache_test::makeStructuralDevicePool(0);
                auto                     host   = block_tree_cache_test::makeHostPool(1, 1);
                std::vector<GroupSetPtr> groups{
                    std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device}, host, nullptr)};
                block_tree_cache_test::prepareGroupSetsForTest(groups);
                BlockTree                     tree(groups);
                BlockTreeTaskPool             pool(1, 1, "evict_failure");
                BlockTreeCacheMetricsReporter metrics(nullptr);
                std::mutex                    mutex;
                BlockTreeEvictor              evictor(
                    &tree,
                    EvictionPolicy::LRU,
                    EvictionPolicy::LRU,
                    EvictionPolicy::FIFO,
                    nullptr,
                    &pool,
                    metrics,
                    mutex,
                    10000,
                    10000,
                    8,
                    [](Tier) { return true; },
                    [](bool) {},
                    0);
                GroupSetResource resource;
                resource.device_blocks = {1};
                const CacheKeysType keys{1};
                evictor.onInserted(tree.insertNode(keys, {{resource}}, true, false));
                auto*                                  node = tree.findNode(keys).front();
                block_tree_cache_test::CallbackBarrier barrier;
                if (!pool.start() || !pool.submit(BlockTreeTaskClass::BACKGROUND, [&] { barrier.enterAndWait(); })
                    || !barrier.waitUntilEnteredFor(1, std::chrono::seconds(1))) {
                    std::_Exit(75);
                }
                pending_eviction_count =
                    &evictor.heaps_[0].watermark_rounds[static_cast<size_t>(Tier::DEVICE)].pending_count;
                allocation_countdown = failure_index;
                try {
                    const auto result    = evictor.batchEvictLocked(0, Tier::DEVICE, 1);
                    allocation_countdown = -1;
                    if (!result.async_submitted || result.scheduled_count != 1
                        || node->group_set_resources[0].transfer_state != GroupSetTransferState::EVICTING
                        || host->referencedBlocksNum(BlockTreeRefType::EVICT) != 1
                        || device->referencedBlocksNum(BlockTreeRefType::CACHE) != 1 || *pending_eviction_count != 1
                        || pool.queueSizes().background != 1) {
                        std::_Exit(75);
                    }
                    std::_Exit(allocation_failed ? 76 : 0);
                } catch (const std::bad_alloc&) {
                    allocation_countdown = -1;
                    if (node->group_set_resources[0].transfer_state != GroupSetTransferState::IDLE
                        || node->group_set_resources[0].device_blocks != BlockIdList{1}
                        || device->referencedBlocksNum(BlockTreeRefType::CACHE) != 1
                        || host->referencedBlocksNum(BlockTreeRefType::EVICT) != 0 || host->freeBlocksNum() != 1
                        || !evictor.heaps_[0].pending_timings.empty() || *pending_eviction_count != 0
                        || pool.queueSizes().background != 0) {
                        std::_Exit(75);
                    }
                    for (const auto& pending : evictor.pending_release_counts_) {
                        if (pending.second != 0) {
                            std::_Exit(75);
                        }
                    }
                    std::_Exit(71);
                }
            }()),
            [&](int status) {
                if (failure_index == -1) {
                    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
                }
                completed = WIFEXITED(status) && WEXITSTATUS(status) == 0;
                return completed || (WIFEXITED(status) && (WEXITSTATUS(status) == 71 || WEXITSTATUS(status) == 76));
            },
            "");
    }
    EXPECT_TRUE(completed);
}

TEST(LoadFailureDeathTest, CommitPreparationAllocationFailuresTerminate) {
    bool completed = false;
    for (int failure_index = -1; failure_index < 128 && !completed; ++failure_index) {
        SCOPED_TRACE(failure_index);
        ASSERT_EXIT(
            ([&] {
                alarm(5);
                auto                     device = block_tree_cache_test::makeStructuralDevicePool(0);
                auto                     host   = block_tree_cache_test::makeHostPool(1, 1);
                std::vector<GroupSetPtr> groups{
                    std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device}, host, nullptr)};
                block_tree_cache_test::prepareGroupSetsForTest(groups);
                BlockTree                     tree(groups);
                BlockTreeTaskPool             pool(1, 1, "load_failure");
                BlockTreeCacheMetricsReporter metrics(nullptr);
                std::mutex                    mutex;
                BlockTreeEvictor              evictor(
                    &tree,
                    EvictionPolicy::LRU,
                    EvictionPolicy::LRU,
                    EvictionPolicy::FIFO,
                    nullptr,
                    &pool,
                    metrics,
                    mutex,
                    10000,
                    10000,
                    8,
                    [](Tier) { return true; },
                    [](bool) {},
                    0);
                BlockTreeLoader loader(
                    &tree, evictor, nullptr, &pool, metrics, mutex, 10000, 10000, true, nullptr, [](bool) {});
                GroupSetResource resource;
                resource.host_block = host->malloc(1).value().front();
                const CacheKeysType keys{1};
                evictor.onInserted(tree.insertNode(keys, {{resource}}, true, false));
                std::shared_ptr<LoadAsyncContext> context;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    context = loader.matchLocked(keys, {}, false, std::nullopt).async_context;
                }
                if (!context || context->newLoadIndices().size() != 1) {
                    std::_Exit(75);
                }
                const auto& desc = context->loadDescs().front();
                if (!desc.node || !desc.isCurrentOwner()
                    || desc.node->group_set_resources[0].transfer_state != GroupSetTransferState::LOAD_PENDING) {
                    std::_Exit(75);
                }
                context->setTargetBlocks(0, {1});
                // Hold the worker until process exit so the accepted LOAD stays observable in the queue.
                block_tree_cache_test::CallbackBarrier barrier;
                if (!pool.start() || !pool.submit(BlockTreeTaskClass::BACKGROUND, [&] { barrier.enterAndWait(); })
                    || !barrier.waitUntilEnteredFor(1, std::chrono::seconds(1))) {
                    std::_Exit(75);
                }
                std::set_terminate([] { std::_Exit(allocation_failed ? 71 : 73); });
                allocation_countdown = failure_index;
                try {
                    const bool committed  = context->commit();
                    allocation_countdown  = -1;
                    const auto& published = desc.node->group_set_resources[0];
                    if (!committed || context->done() || pool.queueSizes().load != 1
                        || published.transfer_state != GroupSetTransferState::LOADING || !published.load_operation
                        || published.load_operation != context->loadOperations()[0] || !desc.isCurrentOwner()
                        || device->referencedBlocksNum(BlockTreeRefType::LOAD) != 1
                        || host->referencedBlocksNum(BlockTreeRefType::LOAD) != 1) {
                        std::_Exit(75);
                    }
                    std::_Exit(allocation_failed ? 76 : 0);
                } catch (...) {
                    std::_Exit(72);
                }
            }()),
            [&](int status) {
                const bool submitted = WIFEXITED(status) && WEXITSTATUS(status) == 0;
                completed            = failure_index >= 0 && submitted;
                return submitted
                       || (failure_index >= 0 && WIFEXITED(status)
                           && (WEXITSTATUS(status) == 71 || WEXITSTATUS(status) == 76));
            },
            "");
    }
    EXPECT_TRUE(completed);
}

TEST(LoadFailureDeathTest, TwoNodeSettlementAllocationFailuresCannotLeavePartialState) {
    for (bool transfer_success : {false, true}) {
        SCOPED_TRACE(transfer_success);
        for (bool queued : {false, true}) {
            SCOPED_TRACE(queued);
            bool completed     = false;
            int  failure_index = 0;
            for (; failure_index < 256 && !completed; ++failure_index) {
                SCOPED_TRACE(failure_index);
                EXPECT_EXIT(
                    ([&] {
                        alarm(5);
                        auto                     device = block_tree_cache_test::makeStructuralDevicePool(0);
                        auto                     host   = block_tree_cache_test::makeHostPool(1, 2);
                        std::vector<GroupSetPtr> groups{
                            std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device}, host, nullptr)};
                        block_tree_cache_test::prepareGroupSetsForTest(groups);
                        BlockTree                     tree(groups);
                        BlockTreeTaskPool             pool(1, 4, "settle_failure");
                        BlockTreeCacheMetricsReporter metrics(nullptr);
                        std::mutex                    mutex;
                        BlockTreeEvictor              evictor(
                            &tree,
                            EvictionPolicy::LRU,
                            EvictionPolicy::LRU,
                            EvictionPolicy::FIFO,
                            nullptr,
                            &pool,
                            metrics,
                            mutex,
                            10000,
                            10000,
                            8,
                            [](Tier) { return true; },
                            [](bool) {},
                            0);
                        BlockTreeLoader loader(
                            &tree, evictor, nullptr, &pool, metrics, mutex, 10000, 10000, true, nullptr, [](bool) {});
                        auto                                       sources = host->malloc(2).value();
                        std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
                        resources[0][0].host_block = sources[0];
                        resources[1][0].host_block = sources[1];
                        const CacheKeysType keys{1, 2};
                        evictor.onInserted(tree.insertNode(keys, resources, true, false));
                        std::shared_ptr<LoadAsyncContext> context;
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            context = loader.matchLocked(keys, {}, false, std::nullopt).async_context;
                        }
                        if (!context || context->newLoadIndices().size() != 2) {
                            std::_Exit(75);
                        }
                        context->setTargetBlocks(0, {1});
                        context->setTargetBlocks(1, {2});
                        std::shared_ptr<LoadTask> task;
                        loader.load_context_coordinator_->commit_callback_ = [&](const auto& current) {
                            std::lock_guard<std::mutex> lock(mutex);
                            if (current->newLoadIndices().empty()) {
                                return true;
                            }
                            if (!current->createLoadOperations(tree)) {
                                return false;
                            }
                            task = LoadTask::create(current, 10000, 10000);
                            current->setSettlementReadyCallback([](const auto&) {});
                            return true;
                        };
                        if (!context->commit()) {
                            std::_Exit(75);
                        }
                        std::shared_ptr<LoadAsyncContext> joined;
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            joined = loader.matchLocked(keys, {}, false, std::nullopt).async_context;
                        }
                        if (!joined || !joined->newLoadIndices().empty() || joined->loadDescs().size() != 2
                            || !joined->commit() || !context->completeTransfers(2, transfer_success, Tier::HOST)) {
                            std::_Exit(75);
                        }
                        size_t owner_notifications  = 0;
                        size_t joined_notifications = 0;
                        context->onDone([&](ErrorInfo) { ++owner_notifications; });
                        joined->onDone([&](ErrorInfo) { ++joined_notifications; });
                        auto verify = [&] {
                            allocation_countdown = -1;
                            if (owner_notifications != 1 || joined_notifications != 1 || !context->done()
                                || context->success() != transfer_success || !joined->done()
                                || joined->success() != transfer_success || joined->remaining_transfer_count_ != 0
                                || joined->remaining_join_count_ != 0
                                || device->referencedBlocksNum(BlockTreeRefType::LOAD) != 0
                                || device->referencedBlocksNum(BlockTreeRefType::CACHE) != (transfer_success ? 2u : 0u)
                                || host->referencedBlocksNum(BlockTreeRefType::LOAD) != 0
                                || host->referencedBlocksNum(BlockTreeRefType::CACHE) != (transfer_success ? 0u : 2u)) {
                                std::_Exit(75);
                            }
                            for (const auto& operation : task->operations) {
                                const auto& desc     = operation->descriptor();
                                const auto& resource = desc.node->group_set_resources[desc.group_set_id];
                                if (!operation->done_ || operation->error_.ok() != transfer_success
                                    || resource.transfer_state != GroupSetTransferState::IDLE
                                    || (transfer_success && resource.device_blocks != desc.target_blocks)
                                    || resource.hasTier(Tier::DEVICE) != transfer_success
                                    || resource.hasTier(Tier::HOST) == transfer_success) {
                                    std::_Exit(75);
                                }
                            }
                            // A recovered optional allocation must still leave the whole load settled.
                            std::_Exit(allocation_failed ? 76 : 0);
                        };
                        std::set_terminate([] { std::_Exit(allocation_failed ? 71 : 73); });
                        if (queued) {
                            if (!pool.start() || !pool.submitCompletion([&] {
                                    first_settlement_resource  = &task->load_descs[0].node->group_set_resources[0];
                                    second_settlement_resource = &task->load_descs[1].node->group_set_resources[0];
                                    allocation_countdown       = failure_index;
                                    loader.scheduleContextSettlement(task, context);
                                    if (!pool.submitCompletion(verify)) {
                                        std::_Exit(75);
                                    }
                                })) {
                                std::_Exit(75);
                            }
                            pool.waitForIdle();
                            std::_Exit(75);
                        }
                        allocation_countdown = failure_index;
                        try {
                            loader.scheduleContextSettlement(task, context);
                        } catch (...) {
                            std::_Exit(72);
                        }
                        verify();
                    }()),
                    [&](int status) {
                        completed = WIFEXITED(status) && WEXITSTATUS(status) == 0;
                        return completed
                               || (WIFEXITED(status) && (WEXITSTATUS(status) == 71 || WEXITSTATUS(status) == 76));
                    },
                    "");
            }
            EXPECT_TRUE(completed);
            EXPECT_GT(failure_index, 1);
        }
    }
}

}  // namespace
}  // namespace rtp_llm
