#include "rtp_llm/cpp/cache/test/KVCacheManagerWithTierCacheTestBase.h"

namespace rtp_llm::test {
using namespace tier_cache_test_detail;

TEST_P(KVCacheManagerWithTierCacheTest, DSV4FullLowerPoolSkipsDemotionAndRetriesAfterCapacityReturns) {
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    auto       cache          = manager_->blockTreeCache();
    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       seed_opt       = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/1);
    ASSERT_TRUE(seed_opt.has_value());
    auto seed = std::move(*seed_opt);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    std::vector<std::shared_ptr<IBlockPool>> host_pools;
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        appendDevicePools(group_set, device_pools);
        host_pools.push_back(group_set->hostPool());
    }
    const auto device_ratio = oneUsedBlockWatermarkRatio(device_pools);
    ASSERT_TRUE(device_ratio.has_value());

    std::vector<std::vector<BlockIdxType>> host_holds(cache->groupSets().size());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        while (true) {
            const auto block = group_set->hostPool()->malloc();
            if (!block.has_value()) {
                break;
            }
            group_set->hostPool()->incTreeRef(*block, BlockTreeRefType::STORE);
            host_holds[group_set_id].push_back(*block);
        }
        EXPECT_EQ(group_set->hostPool()->freeBlocksNum(), 0u);
        EXPECT_EQ(host_holds[group_set_id].size(), group_set->hostPool()->totalBlocksNum());
    }

    const size_t submits_before_full_host = transfer_engine_->submittedDescriptorCount();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    EXPECT_EQ(transfer_engine_->submittedDescriptorCount(), submits_before_full_host);
    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
        for (const DeviceBlockPoolPtr& pool : group_set->devicePools()) {
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
        }
    }

    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        for (const BlockIdxType block : host_holds[group_set_id]) {
            cache->groupSets()[group_set_id]->hostPool()->decTreeRef(block, BlockTreeRefType::STORE);
        }
        EXPECT_EQ(cache->groupSets()[group_set_id]->hostPool()->usedBlocksNum(), 0u);
    }

    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    auto host_path = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(host_path.has_value());
    ASSERT_EQ(host_path->size(), 1u);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& state = (*host_path)[0][group_set_id];
        EXPECT_EQ(state.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(state.getTopTier(), Tier::HOST);
        EXPECT_EQ(cache->groupSets()[group_set_id]->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }

    if (GetParam() == TierLayout::HOST_DISK) {
        std::vector<std::vector<BlockIdxType>> disk_holds(cache->groupSets().size());
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            while (true) {
                const auto block = group_set->diskPool()->malloc();
                if (!block.has_value()) {
                    break;
                }
                group_set->diskPool()->incTreeRef(*block, BlockTreeRefType::STORE);
                disk_holds[group_set_id].push_back(*block);
            }
            EXPECT_EQ(group_set->diskPool()->freeBlocksNum(), 0u);
            EXPECT_EQ(disk_holds[group_set_id].size(), group_set->diskPool()->totalBlocksNum());
        }

        const auto host_ratio = oneUsedBlockWatermarkRatio(host_pools);
        ASSERT_TRUE(host_ratio.has_value());
        const size_t submits_before_full_disk = transfer_engine_->submittedDescriptorCount();
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
        EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
        EXPECT_EQ(transfer_engine_->submittedDescriptorCount(), submits_before_full_disk);
        auto retained_host_path = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(retained_host_path.has_value());
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& state = (*retained_host_path)[0][group_set_id];
            EXPECT_EQ(state.transfer_state, GroupSetTransferState::IDLE);
            EXPECT_EQ(state.getTopTier(), Tier::HOST);
            EXPECT_EQ(cache->groupSets()[group_set_id]->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION),
                      0u);
            EXPECT_EQ(cache->groupSets()[group_set_id]->diskPool()->referencedBlocksNum(BlockTreeRefType::EVICTION),
                      0u);
        }

        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            for (const BlockIdxType block : disk_holds[group_set_id]) {
                cache->groupSets()[group_set_id]->diskPool()->decTreeRef(block, BlockTreeRefType::STORE);
            }
            EXPECT_EQ(cache->groupSets()[group_set_id]->diskPool()->usedBlocksNum(), 0u);
        }

        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
        auto disk_path = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(disk_path.has_value());
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& state = (*disk_path)[0][group_set_id];
            EXPECT_EQ(state.transfer_state, GroupSetTransferState::IDLE);
            EXPECT_EQ(state.getTopTier(), Tier::DISK);
            EXPECT_EQ(cache->groupSets()[group_set_id]->diskPool()->referencedBlocksNum(BlockTreeRefType::EVICTION),
                      0u);
        }
    }

    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4FullHostPoolSelfDrainsThenDeviceDemotionRetries) {
    if (GetParam() != TierLayout::HOST_ONLY) {
        GTEST_SKIP() << "HOST self-drain coverage requires HostOnly layout";
    }
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/128, /*lower_cache_size_mb=*/1));
    auto       cache          = manager_->blockTreeCache();
    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    std::vector<std::shared_ptr<IBlockPool>> host_pools;
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        appendDevicePools(group_set, device_pools);
        host_pools.push_back(group_set->hostPool());
    }
    const size_t host_capacity = host_pools.front()->totalBlocksNum();
    ASSERT_GT(host_capacity, 0u);
    ASSERT_LT(host_capacity + 1, device_pools.front()->totalBlocksNum());
    for (const auto& host_pool : host_pools) {
        ASSERT_EQ(host_pool->totalBlocksNum(), host_capacity);
    }

    auto full_seed_opt = seedDevicePrefix(
        manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/static_cast<int>(host_capacity));
    ASSERT_TRUE(full_seed_opt.has_value());
    auto full_seed = std::move(*full_seed_opt);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, full_seed));
    ASSERT_NO_FATAL_FAILURE(
        moveAllPathResourcesToTier(cache, full_seed, Tier::DEVICE, Tier::HOST, device_pools, transfer_engine_));

    auto full_host = snapshotPathResources(*cache, full_seed.cache_keys);
    ASSERT_TRUE(full_host.has_value());
    ASSERT_EQ(countPathResourcesAtTier(*full_host, Tier::HOST), host_capacity * cache->groupSets().size());
    std::vector<std::vector<BlockIdxType>> host_blocks(
        full_host->size(), std::vector<BlockIdxType>(cache->groupSets().size(), NULL_BLOCK_IDX));
    for (size_t path_index = 0; path_index < full_host->size(); ++path_index) {
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& resource = (*full_host)[path_index][group_set_id];
            ASSERT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            ASSERT_EQ(resource.getTopTier(), Tier::HOST);
            host_blocks[path_index][group_set_id] = resource.host_block;
            EXPECT_EQ(host_pools[group_set_id]->treeRefCount(resource.host_block), 1u);
        }
    }
    for (const auto& host_pool : host_pools) {
        EXPECT_EQ(host_pool->freeBlocksNum(), 0u);
        EXPECT_EQ(host_pool->usedBlocksNum(), host_capacity);
        EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::CACHE), host_capacity);
        EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }

    auto retry_seed_opt = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/100000, /*cached_blocks=*/1);
    ASSERT_TRUE(retry_seed_opt.has_value());
    auto retry_seed = std::move(*retry_seed_opt);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, retry_seed));
    auto retry_device = snapshotPathResources(*cache, retry_seed.cache_keys);
    ASSERT_TRUE(retry_device.has_value());
    ASSERT_EQ(retry_device->size(), 1u);
    std::vector<BlockIndicesType> retry_device_sources(cache->groupSets().size());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const GroupSetPtr&      group_set = cache->groupSets()[group_set_id];
        const GroupSetResource& resource  = (*retry_device)[0][group_set_id];
        ASSERT_EQ(resource.getTopTier(), Tier::DEVICE);
        retry_device_sources[group_set_id] = groupSetSeedBlocksAt(group_set, retry_seed, /*path_index=*/0);
        ASSERT_EQ(resource.device_blocks, retry_device_sources[group_set_id]);
        ASSERT_EQ(group_set->devicePools().size(), retry_device_sources[group_set_id].size());
        for (size_t member_index = 0; member_index < group_set->devicePools().size(); ++member_index) {
            EXPECT_EQ(
                group_set->devicePools()[member_index]->refCount(retry_device_sources[group_set_id][member_index]), 1u);
        }
    }

    const auto device_ratio = oneUsedBlockWatermarkRatio(device_pools);
    ASSERT_TRUE(device_ratio.has_value());
    const size_t submits_before_full_host = transfer_engine_->submittedDescriptorCount();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << transfer_engine_->submittedDescriptorCount();
    EXPECT_EQ(transfer_engine_->submittedDescriptorCount(), submits_before_full_host);
    auto retained_device = snapshotPathResources(*cache, retry_seed.cache_keys);
    ASSERT_TRUE(retained_device.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const GroupSetPtr&      group_set = cache->groupSets()[group_set_id];
        const GroupSetResource& resource  = (*retained_device)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        ASSERT_EQ(resource.getTopTier(), Tier::DEVICE);
        EXPECT_EQ(resource.device_blocks, retry_device_sources[group_set_id]);
        for (size_t member_index = 0; member_index < group_set->devicePools().size(); ++member_index) {
            EXPECT_EQ(
                group_set->devicePools()[member_index]->refCount(retry_device_sources[group_set_id][member_index]), 1u);
            EXPECT_EQ(group_set->devicePools()[member_index]->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
        }
        EXPECT_EQ(host_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, retry_seed.cache_keys));
    ASSERT_NO_FATAL_FAILURE(
        expectFullTierPathUnchanged(*cache, full_seed.cache_keys, Tier::HOST, host_blocks, host_pools, host_capacity));
    const auto host_delete_baseline = snapshotPathResources(*cache, full_seed.cache_keys);
    ASSERT_TRUE(host_delete_baseline.has_value());

    const auto host_ratio = blockExcessWatermarkRatio(host_pools, /*excess_blocks=*/1);
    ASSERT_TRUE(host_ratio.has_value());
    const size_t submits_before_host_delete = transfer_engine_->submittedDescriptorCount();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << transfer_engine_->submittedDescriptorCount();
    EXPECT_EQ(transfer_engine_->submittedDescriptorCount(), submits_before_host_delete)
        << "HostOnly HOST eviction deletes cache-owned victims without a copy";
    std::vector<size_t> freed_host_blocks(cache->groupSets().size(), 0);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        size_t removed = 0;
        for (size_t path_index = 0; path_index < host_delete_baseline->size(); ++path_index) {
            const BlockIdxType block = (*host_delete_baseline)[path_index][group_set_id].host_block;
            if (!host_pools[group_set_id]->isAllocated(block)) {
                ++removed;
            }
        }
        ASSERT_GT(removed, 0u) << "group_set=" << group_set_id;
        freed_host_blocks[group_set_id] = removed;
        EXPECT_EQ(host_pools[group_set_id]->freeBlocksNum(), removed);
        EXPECT_EQ(host_pools[group_set_id]->usedBlocksNum(), host_capacity - removed);
        EXPECT_EQ(host_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::CACHE), host_capacity - removed);
        EXPECT_EQ(host_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }

    const size_t submits_before_retry = transfer_engine_->submittedDescriptorCount();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << transfer_engine_->submittedDescriptorCount();
    const auto descriptors_after_retry = transfer_engine_->descriptors();
    ASSERT_GT(descriptors_after_retry.size(), submits_before_retry);
    for (size_t index = submits_before_retry; index < descriptors_after_retry.size(); ++index) {
        const auto& descriptor = descriptors_after_retry[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.target_tier, Tier::HOST);
        EXPECT_EQ(descriptor.blocksAt(Tier::DEVICE), retry_device_sources[descriptor.group_set_id]);
    }
    auto retry_host = snapshotPathResources(*cache, retry_seed.cache_keys);
    ASSERT_TRUE(retry_host.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& resource = (*retry_host)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(resource.getTopTier(), Tier::HOST);
        EXPECT_EQ(host_pools[group_set_id]->treeRefCount(resource.host_block), 1u);
        const GroupSetPtr& group_set = cache->groupSets()[group_set_id];
        for (size_t member_index = 0; member_index < group_set->devicePools().size(); ++member_index) {
            EXPECT_FALSE(
                group_set->devicePools()[member_index]->isAllocated(retry_device_sources[group_set_id][member_index]));
        }
        EXPECT_EQ(host_pools[group_set_id]->freeBlocksNum(), freed_host_blocks[group_set_id] - 1);
        EXPECT_EQ(host_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::CACHE),
                  host_capacity - freed_host_blocks[group_set_id] + 1);
        EXPECT_EQ(host_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }

    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4FullDiskPoolEvictsThenHostDemotionRetries) {
    if (GetParam() != TierLayout::HOST_DISK) {
        GTEST_SKIP() << "DISK eviction recovery requires HostDisk layout";
    }
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/128, /*lower_cache_size_mb=*/1));
    auto       cache          = manager_->blockTreeCache();
    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    std::vector<std::shared_ptr<IBlockPool>> host_pools;
    std::vector<std::shared_ptr<IBlockPool>> disk_pools;
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        appendDevicePools(group_set, device_pools);
        host_pools.push_back(group_set->hostPool());
        disk_pools.push_back(group_set->diskPool());
    }
    const size_t disk_capacity = disk_pools.front()->totalBlocksNum();
    ASSERT_GT(disk_capacity, 0u);
    ASSERT_LT(disk_capacity + 1, device_pools.front()->totalBlocksNum());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        ASSERT_EQ(host_pools[group_set_id]->totalBlocksNum(), disk_capacity);
        ASSERT_EQ(disk_pools[group_set_id]->totalBlocksNum(), disk_capacity);
    }

    auto full_seed_opt = seedDevicePrefix(
        manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/static_cast<int>(disk_capacity));
    ASSERT_TRUE(full_seed_opt.has_value());
    auto full_seed = std::move(*full_seed_opt);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, full_seed));
    ASSERT_NO_FATAL_FAILURE(
        moveAllPathResourcesToTier(cache, full_seed, Tier::DEVICE, Tier::HOST, device_pools, transfer_engine_));
    ASSERT_NO_FATAL_FAILURE(
        moveAllPathResourcesToTier(cache, full_seed, Tier::HOST, Tier::DISK, host_pools, transfer_engine_));

    auto full_disk = snapshotPathResources(*cache, full_seed.cache_keys);
    ASSERT_TRUE(full_disk.has_value());
    ASSERT_EQ(countPathResourcesAtTier(*full_disk, Tier::DISK), disk_capacity * cache->groupSets().size());
    std::vector<std::vector<BlockIdxType>> disk_blocks(
        full_disk->size(), std::vector<BlockIdxType>(cache->groupSets().size(), NULL_BLOCK_IDX));
    for (size_t path_index = 0; path_index < full_disk->size(); ++path_index) {
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& resource = (*full_disk)[path_index][group_set_id];
            ASSERT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            ASSERT_EQ(resource.getTopTier(), Tier::DISK);
            disk_blocks[path_index][group_set_id] = resource.disk_slot;
            EXPECT_EQ(disk_pools[group_set_id]->treeRefCount(resource.disk_slot), 1u);
        }
    }
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        EXPECT_EQ(host_pools[group_set_id]->usedBlocksNum(), 0u);
        EXPECT_EQ(disk_pools[group_set_id]->freeBlocksNum(), 0u);
        EXPECT_EQ(disk_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::CACHE), disk_capacity);
        EXPECT_EQ(disk_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }

    auto retry_seed_opt = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/100000, /*cached_blocks=*/1);
    ASSERT_TRUE(retry_seed_opt.has_value());
    auto retry_seed = std::move(*retry_seed_opt);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, retry_seed));
    ASSERT_NO_FATAL_FAILURE(
        moveAllPathResourcesToTier(cache, retry_seed, Tier::DEVICE, Tier::HOST, device_pools, transfer_engine_));
    auto retry_host = snapshotPathResources(*cache, retry_seed.cache_keys);
    ASSERT_TRUE(retry_host.has_value());
    ASSERT_EQ(retry_host->size(), 1u);
    std::vector<BlockIdxType> retry_host_sources(cache->groupSets().size(), NULL_BLOCK_IDX);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& resource = (*retry_host)[0][group_set_id];
        ASSERT_EQ(resource.getTopTier(), Tier::HOST);
        retry_host_sources[group_set_id] = resource.host_block;
        EXPECT_EQ(host_pools[group_set_id]->treeRefCount(resource.host_block), 1u);
    }

    const auto host_ratio = oneUsedBlockWatermarkRatio(host_pools);
    ASSERT_TRUE(host_ratio.has_value());
    const size_t submits_before_full_disk = transfer_engine_->submittedDescriptorCount();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << transfer_engine_->submittedDescriptorCount();
    EXPECT_EQ(transfer_engine_->submittedDescriptorCount(), submits_before_full_disk);
    auto retained_host = snapshotPathResources(*cache, retry_seed.cache_keys);
    ASSERT_TRUE(retained_host.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& resource = (*retained_host)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        ASSERT_EQ(resource.getTopTier(), Tier::HOST);
        EXPECT_EQ(resource.host_block, retry_host_sources[group_set_id]);
        EXPECT_EQ(host_pools[group_set_id]->treeRefCount(retry_host_sources[group_set_id]), 1u);
        EXPECT_EQ(host_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
        EXPECT_EQ(disk_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }
    ASSERT_NO_FATAL_FAILURE(
        expectFullTierPathUnchanged(*cache, full_seed.cache_keys, Tier::DISK, disk_blocks, disk_pools, disk_capacity));
    const auto disk_delete_baseline = snapshotPathResources(*cache, full_seed.cache_keys);
    ASSERT_TRUE(disk_delete_baseline.has_value());

    const auto disk_ratio = blockExcessWatermarkRatio(disk_pools, /*excess_blocks=*/1);
    ASSERT_TRUE(disk_ratio.has_value());
    const size_t submits_before_disk_delete = transfer_engine_->submittedDescriptorCount();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DISK, *disk_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DISK, 0.0);
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << transfer_engine_->submittedDescriptorCount();
    EXPECT_EQ(transfer_engine_->submittedDescriptorCount(), submits_before_disk_delete)
        << "DISK eviction deletes cache-owned victims without a copy";
    std::vector<size_t> freed_disk_blocks(cache->groupSets().size(), 0);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        size_t removed = 0;
        for (size_t path_index = 0; path_index < disk_delete_baseline->size(); ++path_index) {
            const BlockIdxType block = (*disk_delete_baseline)[path_index][group_set_id].disk_slot;
            if (!disk_pools[group_set_id]->isAllocated(block)) {
                ++removed;
            }
        }
        ASSERT_GT(removed, 0u) << "group_set=" << group_set_id;
        freed_disk_blocks[group_set_id] = removed;
        EXPECT_EQ(disk_pools[group_set_id]->freeBlocksNum(), removed);
        EXPECT_EQ(disk_pools[group_set_id]->usedBlocksNum(), disk_capacity - removed);
        EXPECT_EQ(disk_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::CACHE), disk_capacity - removed);
        EXPECT_EQ(disk_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }

    const size_t submits_before_retry = transfer_engine_->submittedDescriptorCount();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << transfer_engine_->submittedDescriptorCount();
    const auto descriptors_after_retry = transfer_engine_->descriptors();
    ASSERT_GT(descriptors_after_retry.size(), submits_before_retry);
    for (size_t index = submits_before_retry; index < descriptors_after_retry.size(); ++index) {
        const auto& descriptor = descriptors_after_retry[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DISK);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), retry_host_sources[descriptor.group_set_id]);
    }
    auto retry_disk = snapshotPathResources(*cache, retry_seed.cache_keys);
    ASSERT_TRUE(retry_disk.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& resource = (*retry_disk)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(resource.getTopTier(), Tier::DISK);
        EXPECT_EQ(disk_pools[group_set_id]->treeRefCount(resource.disk_slot), 1u);
        EXPECT_FALSE(host_pools[group_set_id]->isAllocated(retry_host_sources[group_set_id]));
        EXPECT_EQ(disk_pools[group_set_id]->freeBlocksNum(), freed_disk_blocks[group_set_id] - 1);
        EXPECT_EQ(disk_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::CACHE),
                  disk_capacity - freed_disk_blocks[group_set_id] + 1);
        EXPECT_EQ(disk_pools[group_set_id]->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }

    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

}  // namespace rtp_llm::test
