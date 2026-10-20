#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::FullSWAEnvironment;
using block_tree_cache_test::FullSWAEnvironmentOptions;
using block_tree_cache_test::cudaAvailable;
using block_tree_cache_test::insertGroupSetResources;
using block_tree_cache_test::makeHostPool;
using block_tree_cache_test::makeStructuralDevicePool;
using block_tree_cache_test::releaseDeviceBlocks;

TEST(BlockTreeLoaderTest, HostLoadInstallsAllocatorBoundDeviceTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 2;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    auto device_hit = environment->cache->match(environment->keys);
    ASSERT_EQ(device_hit.reuse_time_metrics_snapshots.size(), 4u);
    for (const auto& snapshot : device_hit.reuse_time_metrics_snapshots) {
        EXPECT_EQ(snapshot.tier, Tier::DEVICE);
        EXPECT_TRUE(snapshot.reuse_block_first_hit_idletime_ms.has_value());
    }
    environment->releaseMatch(device_hit);
    environment->demoteAll(Tier::DEVICE);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::HOST));
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::HOST), 1u);

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    EXPECT_EQ(result.matched_device_blocks, 0u);
    ASSERT_EQ(result.reuse_time_metrics_snapshots.size(), 4u);
    for (const auto& snapshot : result.reuse_time_metrics_snapshots) {
        EXPECT_EQ(snapshot.tier, Tier::HOST);
        EXPECT_TRUE(snapshot.reuse_block_first_hit_idletime_ms.has_value());
    }
    std::shared_ptr<LoadAsyncContext> load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), 2u);
    EXPECT_EQ(load_context->matchedBlocks(Tier::HOST), 2u);
    EXPECT_EQ(load_context->matchedBlocks(Tier::DISK), 0u);

    for (const TransferDescriptor& desc : load_context->loadDescs()) {
        const auto source_pool = environment->groups.at(desc.group_set_id)->hostPool();
        ASSERT_NE(source_pool, nullptr);
        for (BlockIdxType block : desc.source_blocks) {
            EXPECT_EQ(source_pool->treeRefCount(block), 2u);
        }
    }
    for (const auto& source_pool : environment->host_pools) {
        EXPECT_EQ(source_pool->referencedBlocksNum(BlockTreeRefType::CACHE), options.path_length);
        EXPECT_EQ(source_pool->referencedBlocksNum(BlockTreeRefType::LOAD), options.path_length);
    }

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        std::vector<BlockIdxType> targets;
        const size_t              group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(group_set_id)->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        load_context->setTargetBlocks(desc_index, std::move(targets));
    }

    ASSERT_TRUE(load_context->commit());
    std::shared_ptr<AsyncContext> context = load_context;
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::HOST), 0u);
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);
    environment->expectPayloads();
    auto device_first_hit = environment->cache->match(environment->keys);
    ASSERT_EQ(device_first_hit.reuse_time_metrics_snapshots.size(), 4u);
    for (const auto& snapshot : device_first_hit.reuse_time_metrics_snapshots) {
        EXPECT_EQ(snapshot.tier, Tier::DEVICE);
        EXPECT_TRUE(snapshot.reuse_block_first_hit_idletime_ms.has_value());
    }
    environment->releaseMatch(device_first_hit);
    auto device_second_hit = environment->cache->match(environment->keys);
    ASSERT_EQ(device_second_hit.reuse_time_metrics_snapshots.size(), 4u);
    for (const auto& snapshot : device_second_hit.reuse_time_metrics_snapshots) {
        EXPECT_FALSE(snapshot.reuse_block_first_hit_idletime_ms.has_value());
    }
    environment->releaseMatch(device_second_hit);

    result.async_context.reset();
    load_context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, HostLoadUsesReservedAdmissionWhenBackgroundQueueIsFull) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::HOST));

    constexpr size_t worker_count = 2;
    constexpr size_t queue_size   = BlockTreeTaskPool::kLoadReservedSlots + 2;
    auto replacement = std::make_unique<BlockTreeTaskPool>(worker_count, queue_size, "load_reserved_admission");
    ASSERT_TRUE(replacement->start());
    environment->cache->task_pool_->shutdown();
    environment->cache->task_pool_          = std::move(replacement);
    environment->cache->evictor_.task_pool_ = environment->cache->task_pool_.get();
    environment->cache->loader_.task_pool_  = environment->cache->task_pool_.get();
    environment->cache->storer_.task_pool_  = environment->cache->task_pool_.get();

    auto release_promise = std::make_shared<std::promise<void>>();
    auto release_future  = release_promise->get_future().share();
    auto release_once    = std::make_shared<std::once_flag>();
    auto release_workers = [release_promise, release_once]() {
        std::call_once(*release_once, [release_promise]() { release_promise->set_value(); });
    };
    [[maybe_unused]] auto release_guard =
        std::shared_ptr<void>(nullptr, [release_workers](void*) { release_workers(); });

    auto  entered_count   = std::make_shared<std::atomic<size_t>>(0);
    auto  entered_promise = std::make_shared<std::promise<void>>();
    auto  entered_future  = entered_promise->get_future();
    auto* task_pool       = environment->cache->task_pool_.get();
    for (size_t worker = 0; worker < worker_count; ++worker) {
        if (!task_pool->submit(BlockTreeTaskClass::BACKGROUND,
                               [release_future, entered_count, entered_promise]() {
                                   if (entered_count->fetch_add(1) + 1 == worker_count) {
                                       entered_promise->set_value();
                                   }
                                   release_future.wait();
                               })) {
            FAIL() << "failed to submit background task";
        }
    }
    ASSERT_EQ(entered_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    ASSERT_TRUE(environment->cache->task_pool_->submit(BlockTreeTaskClass::BACKGROUND, [] {}));
    ASSERT_TRUE(environment->cache->task_pool_->submit(BlockTreeTaskClass::BACKGROUND, [] {}));
    EXPECT_FALSE(environment->cache->task_pool_->submit(BlockTreeTaskClass::BACKGROUND, [] {}));

    BlockTreeMatchResult result       = environment->cache->match(environment->keys);
    auto                 load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        std::vector<BlockIdxType> targets;
        const size_t              group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(group_set_id)->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        load_context->setTargetBlocks(desc_index, std::move(targets));
    }

    auto completion_promise = std::make_shared<std::promise<ErrorInfo>>();
    auto completion_future  = completion_promise->get_future();
    load_context->onDone([completion_promise](ErrorInfo error) { completion_promise->set_value(std::move(error)); });
    ASSERT_TRUE(load_context->commit());
    {
        std::lock_guard<std::mutex> lock(environment->cache->task_pool_->lifecycle_mutex_);
        EXPECT_EQ(environment->cache->task_pool_->load_queue_.size(), 1u);
        EXPECT_EQ(environment->cache->task_pool_->background_queue_.size(), 2u);
    }

    release_workers();
    ASSERT_EQ(completion_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(completion_future.get().ok());
    EXPECT_TRUE(load_context->done());
    EXPECT_TRUE(load_context->success());
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));

    result.async_context.reset();
    load_context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
    EXPECT_EQ(environment->cache->task_pool_->pending_tasks_.load(), 0);
}

class BlockTreeLoaderReuseTimeTest: public ::testing::TestWithParam<size_t> {};

TEST_P(BlockTreeLoaderReuseTimeTest, DevicePrefixAndLogicalSuffixShareReuseAndEvictionHitTime) {
    constexpr size_t path_length          = 4;
    const size_t     device_prefix_length = GetParam();
    auto             full                 = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(0)}, makeHostPool(1, path_length), nullptr);
    auto swa    = std::make_shared<SWAGroupSet>(/*sliding_window_size=*/2,
                                             /*seq_size_per_block=*/1,
                                             std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(1)},
                                             makeHostPool(1, path_length),
                                             nullptr);
    auto linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(2)}, makeHostPool(1, path_length), nullptr);
    std::vector<GroupSetPtr> groups = {full, swa, linear};

    BlockTreeCacheConfig config;
    config.enable_device_cache = true;
    config.enable_host_cache   = true;
    config.enable_disk_cache   = false;
    auto cache                 = block_tree_cache_test::makeBlockTreeCacheForTest(groups, config);
    ASSERT_NE(cache, nullptr);

    const CacheKeysType                        keys = {100, 200, 300, 400};
    std::vector<std::vector<GroupSetResource>> resources(path_length, std::vector<GroupSetResource>(groups.size()));
    for (size_t path_index = 0; path_index < path_length; ++path_index) {
        for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
            if (path_index < device_prefix_length) {
                resources[path_index][group_set_id].device_blocks = {static_cast<BlockIdxType>(10 + path_index)};
            } else {
                resources[path_index][group_set_id].host_block =
                    groups[group_set_id]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
                ASSERT_NE(resources[path_index][group_set_id].host_block, NULL_BLOCK_IDX);
            }
        }
    }
    ASSERT_TRUE(insertGroupSetResources(*cache, keys, resources));

    const std::vector<TreeNode*> path = cache->tree()->findNode(keys);
    ASSERT_EQ(path.size(), path_length);
    std::vector<std::vector<uint64_t>> access_before(path.size());
    for (size_t node_index = 0; node_index < path.size(); ++node_index) {
        access_before[node_index].reserve(groups.size());
        for (const GroupSetResource& resource : path[node_index]->group_set_resources) {
            EXPECT_EQ(resource.candidate_meta.hit_count, 0u);
            access_before[node_index].push_back(resource.candidate_meta.last_access_seq);
        }
    }

    BlockTreeMatchResult result = cache->match(keys);
    EXPECT_EQ(result.matched_device_blocks, device_prefix_length);
    auto load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    if (device_prefix_length < path_length) {
        ASSERT_NE(load_context, nullptr);
        EXPECT_EQ(load_context->matchedBlocks(), path_length);
    } else {
        EXPECT_EQ(load_context, nullptr);
    }

    const size_t expected_samples = device_prefix_length == 2 ? 10 : device_prefix_length == 3 ? 9 : 7;
    ASSERT_EQ(result.reuse_time_metrics_snapshots.size(), expected_samples);
    for (const auto& snapshot : result.reuse_time_metrics_snapshots) {
        EXPECT_TRUE(snapshot.tier == Tier::DEVICE || snapshot.tier == Tier::HOST);
        ASSERT_TRUE(snapshot.reuse_block_first_hit_idletime_ms.has_value());
        EXPECT_EQ(snapshot.reuse_block_hit_idletime_ms, *snapshot.reuse_block_first_hit_idletime_ms);
    }
    const uint64_t            access_seq = path.front()->group_set_resources[0].candidate_meta.last_access_seq;
    const std::vector<size_t> reused_suffix_starts = {0, 2, 3};
    const std::vector<size_t> device_suffix_starts = {0,
                                                      device_prefix_length > 2 ? device_prefix_length - 2 : 0,
                                                      device_prefix_length > 0 ? device_prefix_length - 1 : 0};
    for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
        for (size_t node_index = 0; node_index < path.size(); ++node_index) {
            const CandidateMeta& meta = path[node_index]->group_set_resources[group_set_id].candidate_meta;
            const bool           reused =
                node_index >= reused_suffix_starts[group_set_id]
                || (node_index >= device_suffix_starts[group_set_id] && node_index < device_prefix_length);
            if (reused) {
                EXPECT_GT(meta.last_access_seq, access_before[node_index][group_set_id]);
                EXPECT_EQ(meta.last_access_seq, access_seq);
                EXPECT_EQ(meta.hit_count, 1u);
                EXPECT_GE(meta.tier_last_hit_time_us, meta.tier_enter_time_us);
                EXPECT_EQ(meta.last_access_time_us, meta.tier_last_hit_time_us);
                EXPECT_EQ(EvictionTimingSnapshot(meta).last_access_time_us, meta.tier_last_hit_time_us);
            } else {
                EXPECT_EQ(meta.last_access_seq, access_before[node_index][group_set_id]);
                EXPECT_EQ(meta.hit_count, 0u);
                EXPECT_EQ(meta.tier_last_hit_time_us, 0);
                EXPECT_EQ(meta.last_access_time_us, meta.tier_enter_time_us);
            }
        }
    }

    result.async_context.reset();
    load_context.reset();
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
}

INSTANTIATE_TEST_SUITE_P(DevicePrefixes, BlockTreeLoaderReuseTimeTest, ::testing::Values(0u, 2u, 3u, 4u));

TEST(BlockTreeLoaderTest, DiskTransferFailureInstallsNoLoadTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    {
        auto host_hit = environment->cache->match(environment->keys);
        ASSERT_EQ(host_hit.reuse_time_metrics_snapshots.size(), 2u);
        for (const auto& snapshot : host_hit.reuse_time_metrics_snapshots) {
            EXPECT_EQ(snapshot.tier, Tier::HOST);
            EXPECT_TRUE(snapshot.reuse_block_first_hit_idletime_ms.has_value());
        }
    }
    environment->demoteAll(Tier::HOST);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::DISK));

    environment->scripted_per_rank_transfer_engine->clear();
    environment->scripted_per_rank_transfer_engine->enqueue(true);
    environment->scripted_per_rank_transfer_engine->enqueue(false);

    BlockTreeMatchResult result       = environment->cache->match(environment->keys);
    ASSERT_EQ(result.reuse_time_metrics_snapshots.size(), 2u);
    for (const auto& snapshot : result.reuse_time_metrics_snapshots) {
        EXPECT_EQ(snapshot.tier, Tier::DISK);
        EXPECT_TRUE(snapshot.reuse_block_first_hit_idletime_ms.has_value());
    }
    auto                 load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    ASSERT_EQ(load_context->loadDescs().size(), 2u);
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        const size_t              group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        std::vector<BlockIdxType> targets;
        for (const DeviceBlockPoolPtr& pool : environment->groups[group_set_id]->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            pool->incRef(blocks);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        load_context->setTargetBlocks(desc_index, std::move(targets));
    }

    ASSERT_TRUE(load_context->commit());
    load_context->waitDone();
    EXPECT_FALSE(load_context->success());
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedBatchCount(), 2u);

    const auto resources = environment->resourcesForPathNode(0);
    ASSERT_EQ(resources.size(), 2u);
    for (const GroupSetResource& resource : resources) {
        EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
        EXPECT_TRUE(resource.hasTier(Tier::DISK));
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    }
    for (const auto& [pool, block] : request_targets) {
        EXPECT_EQ(pool->refCount(block), 1u);
        pool->decRef(block);
        EXPECT_EQ(pool->freeBlocksNum(), options.usable_device_blocks);
    }

    {
        auto disk_hit = environment->cache->match(environment->keys);
        ASSERT_EQ(disk_hit.reuse_time_metrics_snapshots.size(), 2u);
        for (const auto& snapshot : disk_hit.reuse_time_metrics_snapshots) {
            EXPECT_EQ(snapshot.tier, Tier::DISK);
            EXPECT_FALSE(snapshot.reuse_block_first_hit_idletime_ms.has_value());
        }
    }
    result.async_context.reset();
    load_context.reset();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
    for (const auto& group : environment->groups) {
        EXPECT_EQ(group->diskPool()->frozenBlocksNum(), 0u);
        for (const auto& pool : group->devicePools()) {
            EXPECT_EQ(pool->frozenBlocksNum(), 0u);
        }
    }
}

TEST(BlockTreeLoaderTest, LoadStateMachineRejectsDuplicateTransitionAndRestoresSource) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    auto find_result = environment->cache->tree()->findNode(environment->keys);
    ASSERT_FALSE(find_result.empty());

    constexpr size_t  group_set_id = 0;
    GroupSetResource& resource     = find_result.back()->group_set_resources[group_set_id];

    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::IDLE, GroupSetTransferState::LOAD_PENDING));
    environment->cache->evictor_.suspendCandidate(find_result.back(), group_set_id, Tier::HOST);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOAD_PENDING);
    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
    EXPECT_FALSE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);

    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE));
    environment->cache->evictor_.admitCandidate(find_result.back(), group_set_id, Tier::HOST);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);

    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, ChangeTransferStateDoesNotOverwriteForeignTransferState) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    auto find_result = environment->cache->tree()->findNode(environment->keys);
    ASSERT_FALSE(find_result.empty());

    GroupSetResource& resource = find_result.back()->group_set_resources[0];
    resource.transfer_state    = GroupSetTransferState::DEMOTING;
    EXPECT_FALSE(environment->cache->loader_.changeTransferState(
        find_result.back(), 0, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);

    resource.transfer_state = GroupSetTransferState::IDLE;
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

}  // namespace
}  // namespace rtp_llm
