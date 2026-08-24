#include "rtp_llm/cpp/cache/test/KVCacheManagerWithTierCacheTestBase.h"

namespace rtp_llm::test {
using namespace tier_cache_test_detail;

TEST_P(KVCacheManagerWithTierCacheTest, DSV4DeviceWatermarkDemotesToHostAndLoadsBackMissingResources) {
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    ASSERT_NE(manager_, nullptr);
    auto cache = manager_->blockTreeCache();

    auto pausable_engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_engine);
    transfer_engine_.reset();

    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       maybe_seed     = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/3);
    ASSERT_TRUE(maybe_seed.has_value());
    auto seed = std::move(*maybe_seed);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));
    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));

    struct LiveRequest {
        BatchKVCacheResourcePtr resource;
        CompleteTokenIdsPtr     token_ids;
    };
    std::vector<LiveRequest> eviction_guards;
    const int                seq_size_per_block = static_cast<int>(cache_config_.seq_size_per_block);
    for (const int logical_blocks : {2, 3}) {
        LiveRequest hold;
        hold.resource  = makeResource(cache_config_);
        hold.token_ids = makeTokenIds(
            /*offset=*/0, logical_blocks * seq_size_per_block, logical_blocks * seq_size_per_block, seq_size_per_block);
        MallocInfo malloc_info{hold.resource, hold.token_ids};
        malloc_info.reuse_cache         = true;
        malloc_info.enable_cache_lookup = true;
        const auto result               = manager_->malloc(malloc_info);
        ASSERT_TRUE(result.success);
        EXPECT_EQ(result.reuse_len, (logical_blocks - 1) * seq_size_per_block);
        EXPECT_EQ(result.host_reuse_len, 0);
        EXPECT_EQ(result.disk_reuse_len, 0);
        EXPECT_EQ(result.async_context, nullptr);
        eviction_guards.push_back(std::move(hold));
    }

    // The two live matches pin path positions 0 and 1 for both FULL and SWA.
    // A one-block watermark excess can therefore only choose path position 2.
    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    const auto& reference_pool = cache->groupSets().front()->devicePools().front();
    ASSERT_NE(reference_pool, nullptr);
    const size_t capacity = reference_pool->totalBlocksNum();
    const size_t used     = capacity - reference_pool->freeBlocksNum();
    ASSERT_GT(capacity, 0u);
    ASSERT_GT(used, 0u);
    const double one_block_excess_ratio = (static_cast<double>(used) - 0.25) / static_cast<double>(capacity);
    ASSERT_GT(one_block_excess_ratio, 0.0);
    ASSERT_LT(one_block_excess_ratio, 1.0);
    for (const auto& group_set : cache->groupSets()) {
        for (const auto& pool : group_set->devicePools()) {
            ASSERT_NE(pool, nullptr);
            const size_t pool_capacity  = pool->totalBlocksNum();
            const size_t pool_used      = pool_capacity - pool->freeBlocksNum();
            const size_t pool_threshold = static_cast<size_t>(pool_capacity * one_block_excess_ratio);
            ASSERT_EQ(pool_capacity, capacity);
            ASSERT_EQ(pool_used, used);
            ASSERT_EQ(pool_used - pool_threshold, 1u);
        }
    }

    ASSERT_TRUE(pausable_engine->armPause());
    ScopedTransferRelease demotion_release(pausable_engine);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, one_block_excess_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    const bool demotion_entered = pausable_engine->waitUntilEnteredFor(
        std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
    if (!demotion_entered) {
        pausable_engine->release();
    }
    ASSERT_TRUE(demotion_entered);
    EXPECT_GT(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    auto maybe_demoting = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_demoting.has_value());
    for (size_t path_index = 0; path_index < maybe_demoting->size(); ++path_index) {
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const GroupSetPtr&      group_set = cache->groupSets()[group_set_id];
            const GroupSetResource& resource  = (*maybe_demoting)[path_index][group_set_id];
            EXPECT_EQ(resource.transfer_state,
                      path_index == 2 ? GroupSetTransferState::DEMOTING : GroupSetTransferState::IDLE)
                << "path=" << path_index << " group_set=" << group_set_id;
            ASSERT_TRUE(resource.hasTier(Tier::DEVICE));
            EXPECT_EQ(resource.getTopTier(), Tier::DEVICE);
            EXPECT_EQ(resource.device_blocks, groupSetSeedBlocksAt(group_set, seed, path_index));
        }
    }
    for (const auto& group_set : cache->groupSets()) {
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
        if (GetParam() == TierLayout::HOST_DISK) {
            EXPECT_EQ(group_set->diskPool()->usedBlocksNum(), 0u);
        }
    }

    pausable_engine->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    auto maybe_host = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_host.has_value());
    std::vector<BlockIdxType> host_sources(cache->groupSets().size(), NULL_BLOCK_IDX);
    for (size_t path_index = 0; path_index < maybe_host->size(); ++path_index) {
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*maybe_host)[path_index][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            if (path_index == 2) {
                ASSERT_TRUE(resource.hasTier(Tier::HOST));
                EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
                EXPECT_EQ(resource.getTopTier(), Tier::HOST);
                host_sources[group_set_id] = resource.host_block;
                EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
                EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
                EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
                const BlockIndicesType source_blocks = groupSetSeedBlocksAt(group_set, seed, path_index);
                ASSERT_EQ(group_set->devicePools().size(), source_blocks.size());
                for (size_t member_index = 0; member_index < group_set->devicePools().size(); ++member_index) {
                    EXPECT_FALSE(group_set->devicePools()[member_index]->isAllocated(source_blocks[member_index]));
                }
            } else {
                ASSERT_TRUE(resource.hasTier(Tier::DEVICE));
                EXPECT_EQ(resource.getTopTier(), Tier::DEVICE);
                EXPECT_EQ(resource.device_blocks, groupSetSeedBlocksAt(group_set, seed, path_index));
            }
        }
    }

    const auto demotion_descriptors = pausable_engine->descriptors();
    ASSERT_EQ(demotion_descriptors.size(), cache->groupSets().size());
    std::vector<size_t> demotions_by_group_set(cache->groupSets().size(), 0);
    for (const auto& descriptor : demotion_descriptors) {
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.target_tier, Tier::HOST);
        EXPECT_FALSE(isNullBlockIdx(descriptor.singleBlockAt(Tier::HOST)));
        EXPECT_EQ(descriptor.blocksAt(Tier::DEVICE),
                  groupSetSeedBlocksAt(cache->groupSets()[descriptor.group_set_id], seed, /*path_index=*/2));
        ++demotions_by_group_set[descriptor.group_set_id];
    }
    for (size_t group_set_id = 0; group_set_id < demotions_by_group_set.size(); ++group_set_id) {
        EXPECT_EQ(demotions_by_group_set[group_set_id], 1u);
    }

    for (const auto& hold : eviction_guards) {
        manager_->free(FreeInfo{hold.resource, hold.token_ids});
    }
    eviction_guards.clear();

    ASSERT_TRUE(pausable_engine->armPause());
    ScopedTransferRelease load_release(pausable_engine);
    auto                  load_resource = makeResource(cache_config_);
    auto                  load_token_ids =
        makeTokenIds(/*offset=*/0, 4 * seq_size_per_block, 4 * seq_size_per_block, seq_size_per_block);
    MallocInfo load_info{load_resource, load_token_ids};
    load_info.reuse_cache         = true;
    load_info.enable_cache_lookup = true;
    const auto load_result        = manager_->malloc(load_info);
    ASSERT_TRUE(load_result.success);
    EXPECT_EQ(load_result.reuse_len, 2 * seq_size_per_block);
    EXPECT_EQ(load_result.host_reuse_len, 0);
    EXPECT_EQ(load_result.disk_reuse_len, 0);
    ASSERT_NE(load_result.async_context, nullptr);
    const bool load_entered = pausable_engine->waitUntilEnteredFor(
        std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
    if (!load_entered) {
        pausable_engine->release();
    }
    ASSERT_TRUE(load_entered);
    EXPECT_FALSE(load_result.async_context->done());
    EXPECT_GT(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    auto maybe_loading = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_loading.has_value());
    std::vector<BlockIndicesType> load_targets(cache->groupSets().size());
    for (size_t path_index = 0; path_index < maybe_loading->size(); ++path_index) {
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*maybe_loading)[path_index][group_set_id];
            if (path_index == 2) {
                EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
                ASSERT_TRUE(resource.hasTier(Tier::HOST));
                EXPECT_EQ(resource.host_block, host_sources[group_set_id]);
                EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 2u);
                load_targets[group_set_id] = groupSetRequestBlocksAt(group_set, load_resource, 0, path_index);
                ASSERT_EQ(group_set->devicePools().size(), load_targets[group_set_id].size());
                for (size_t member_index = 0; member_index < group_set->groupIds().size(); ++member_index) {
                    const int          group_id = static_cast<int>(group_set->groupIds()[member_index]);
                    const BlockIdxType block    = load_targets[group_set_id][member_index];
                    ASSERT_FALSE(isNullBlockIdx(block));
                    EXPECT_EQ(group_set->devicePools()[member_index]->refCount(block), 2u);
                    ASSERT_TRUE(
                        fillGroupBlockPayload(manager_, cache_config_, group_id, block, path_index, /*poison=*/true));
                }
            } else {
                EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
                ASSERT_TRUE(resource.hasTier(Tier::DEVICE));
                EXPECT_EQ(resource.device_blocks, groupSetSeedBlocksAt(group_set, seed, path_index));
            }
        }
    }

    pausable_engine->release();
    load_result.async_context->waitDone();
    ASSERT_TRUE(load_result.async_context->done());
    ASSERT_TRUE(load_result.async_context->success()) << load_result.async_context->errorInfo().ToString();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    const auto all_descriptors = pausable_engine->descriptors();
    ASSERT_EQ(all_descriptors.size(), 2 * cache->groupSets().size());
    std::vector<size_t> loads_by_group_set(cache->groupSets().size(), 0);
    for (size_t index = demotion_descriptors.size(); index < all_descriptors.size(); ++index) {
        const auto& descriptor = all_descriptors[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), host_sources[descriptor.group_set_id]);
        EXPECT_EQ(descriptor.blocksAt(Tier::DEVICE), load_targets[descriptor.group_set_id]);
        ++loads_by_group_set[descriptor.group_set_id];
    }
    for (size_t group_set_id = 0; group_set_id < loads_by_group_set.size(); ++group_set_id) {
        EXPECT_EQ(loads_by_group_set[group_set_id], 1u);
    }

    auto maybe_loaded = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_loaded.has_value());
    for (size_t path_index = 0; path_index < maybe_loaded->size(); ++path_index) {
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*maybe_loaded)[path_index][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            ASSERT_TRUE(resource.hasTier(Tier::DEVICE));
            EXPECT_EQ(resource.getTopTier(), Tier::DEVICE);
            const BlockIndicesType expected_blocks =
                path_index == 2 ? load_targets[group_set_id] : groupSetSeedBlocksAt(group_set, seed, path_index);
            EXPECT_EQ(resource.device_blocks, expected_blocks);
        }
    }
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        EXPECT_FALSE(cache->groupSets()[group_set_id]->hostPool()->isAllocated(host_sources[group_set_id]));
    }
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, load_resource, /*logical_reuse_blocks=*/3));

    manager_->free(FreeInfo{load_resource, load_token_ids});
    const size_t submits_before_second_hit = pausable_engine->submittedDescriptorCount();
    auto         hit_resource              = makeResource(cache_config_);
    auto hit_token_ids = makeTokenIds(/*offset=*/0, 4 * seq_size_per_block, 4 * seq_size_per_block, seq_size_per_block);
    MallocInfo hit_info{hit_resource, hit_token_ids};
    hit_info.reuse_cache         = true;
    hit_info.enable_cache_lookup = true;
    const auto hit_result        = manager_->malloc(hit_info);
    ASSERT_TRUE(hit_result.success);
    EXPECT_EQ(hit_result.reuse_len, 3 * seq_size_per_block);
    EXPECT_EQ(hit_result.host_reuse_len, 0);
    EXPECT_EQ(hit_result.disk_reuse_len, 0);
    EXPECT_EQ(hit_result.async_context, nullptr);
    EXPECT_EQ(pausable_engine->submittedDescriptorCount(), submits_before_second_hit);
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, hit_resource, /*logical_reuse_blocks=*/3));

    manager_->free(FreeInfo{hit_resource, hit_token_ids});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4DeviceAndHostWatermarksDemoteToDiskAndLoadBack) {
    if (GetParam() != TierLayout::HOST_DISK) {
        GTEST_SKIP() << "disk round-trip requires the HostDisk layout";
    }

    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/8));
    ASSERT_NE(manager_, nullptr);
    auto cache = manager_->blockTreeCache();

    auto pausable_engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_engine);
    transfer_engine_.reset();

    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       maybe_seed     = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/1);
    ASSERT_TRUE(maybe_seed.has_value());
    auto seed = std::move(*maybe_seed);
    ASSERT_EQ(seed.cache_keys.size(), 1u);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));
    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        appendDevicePools(group_set, device_pools);
    }
    const auto device_ratio = oneUsedBlockWatermarkRatio(device_pools);
    ASSERT_TRUE(device_ratio.has_value());

    ASSERT_TRUE(pausable_engine->armPause());
    ScopedTransferRelease device_demotion_release(pausable_engine);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    const bool device_demotion_entered = pausable_engine->waitUntilEnteredFor(
        std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
    if (!device_demotion_entered) {
        pausable_engine->release();
    }
    ASSERT_TRUE(device_demotion_entered);
    EXPECT_GT(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    auto maybe_device_demoting = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_device_demoting.has_value());
    ASSERT_EQ(maybe_device_demoting->size(), 1u);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*maybe_device_demoting)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);
        ASSERT_TRUE(resource.hasTier(Tier::DEVICE));
        EXPECT_EQ(resource.getTopTier(), Tier::DEVICE);
        EXPECT_EQ(resource.device_blocks, groupSetSeedBlocksAt(group_set, seed, /*path_index=*/0));
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
        EXPECT_EQ(group_set->diskPool()->usedBlocksNum(), 0u);
    }

    pausable_engine->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    auto maybe_host = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_host.has_value());
    std::vector<BlockIdxType> host_sources(cache->groupSets().size(), NULL_BLOCK_IDX);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*maybe_host)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        ASSERT_TRUE(resource.hasTier(Tier::HOST));
        EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
        EXPECT_EQ(resource.getTopTier(), Tier::HOST);
        host_sources[group_set_id] = resource.host_block;
        EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
        const BlockIndicesType source_blocks = groupSetSeedBlocksAt(group_set, seed, /*path_index=*/0);
        ASSERT_EQ(group_set->devicePools().size(), source_blocks.size());
        for (size_t member_index = 0; member_index < group_set->devicePools().size(); ++member_index) {
            EXPECT_FALSE(group_set->devicePools()[member_index]->isAllocated(source_blocks[member_index]));
        }
    }

    auto descriptors = pausable_engine->descriptors();
    ASSERT_EQ(descriptors.size(), cache->groupSets().size());
    std::vector<size_t> device_to_host_by_group_set(cache->groupSets().size(), 0);
    for (const auto& descriptor : descriptors) {
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.target_tier, Tier::HOST);
        EXPECT_EQ(descriptor.blocksAt(Tier::DEVICE),
                  groupSetSeedBlocksAt(cache->groupSets()[descriptor.group_set_id], seed, /*path_index=*/0));
        EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), host_sources[descriptor.group_set_id]);
        ++device_to_host_by_group_set[descriptor.group_set_id];
    }
    for (size_t group_set_id = 0; group_set_id < device_to_host_by_group_set.size(); ++group_set_id) {
        EXPECT_EQ(device_to_host_by_group_set[group_set_id], 1u);
    }

    std::vector<std::shared_ptr<IBlockPool>> host_pools;
    for (const auto& group_set : cache->groupSets()) {
        host_pools.push_back(group_set->hostPool());
    }
    const auto host_ratio = oneUsedBlockWatermarkRatio(host_pools);
    ASSERT_TRUE(host_ratio.has_value());

    ASSERT_TRUE(pausable_engine->armPause());
    ScopedTransferRelease host_demotion_release(pausable_engine);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
    const bool host_demotion_entered = pausable_engine->waitUntilEnteredFor(
        std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
    if (!host_demotion_entered) {
        pausable_engine->release();
    }
    ASSERT_TRUE(host_demotion_entered);
    EXPECT_GT(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    auto maybe_host_demoting = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_host_demoting.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*maybe_host_demoting)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);
        ASSERT_TRUE(resource.hasTier(Tier::HOST));
        EXPECT_EQ(resource.getTopTier(), Tier::HOST);
        EXPECT_EQ(resource.host_block, host_sources[group_set_id]);
        EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
        EXPECT_EQ(group_set->diskPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
        EXPECT_EQ(group_set->diskPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
    }

    pausable_engine->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    auto maybe_disk = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_disk.has_value());
    std::vector<BlockIdxType> disk_sources(cache->groupSets().size(), NULL_BLOCK_IDX);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*maybe_disk)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        ASSERT_TRUE(resource.hasTier(Tier::DISK));
        EXPECT_FALSE(resource.hasTier(Tier::HOST));
        EXPECT_EQ(resource.getTopTier(), Tier::DISK);
        disk_sources[group_set_id] = resource.disk_slot;
        EXPECT_FALSE(group_set->hostPool()->isAllocated(host_sources[group_set_id]));
        EXPECT_EQ(group_set->diskPool()->treeRefCount(resource.disk_slot), 1u);
        EXPECT_EQ(group_set->diskPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
        EXPECT_EQ(group_set->diskPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }

    descriptors = pausable_engine->descriptors();
    ASSERT_EQ(descriptors.size(), 2 * cache->groupSets().size());
    std::vector<size_t> host_to_disk_by_group_set(cache->groupSets().size(), 0);
    for (size_t index = cache->groupSets().size(); index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DISK);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), host_sources[descriptor.group_set_id]);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::DISK), disk_sources[descriptor.group_set_id]);
        ++host_to_disk_by_group_set[descriptor.group_set_id];
    }
    for (size_t group_set_id = 0; group_set_id < host_to_disk_by_group_set.size(); ++group_set_id) {
        EXPECT_EQ(host_to_disk_by_group_set[group_set_id], 1u);
    }

    ASSERT_TRUE(pausable_engine->armPause());
    ScopedTransferRelease disk_load_release(pausable_engine);
    const int             seq_size_per_block = static_cast<int>(cache_config_.seq_size_per_block);
    auto                  load_resource      = makeResource(cache_config_);
    auto                  load_token_ids =
        makeTokenIds(/*offset=*/0, 2 * seq_size_per_block, 2 * seq_size_per_block, seq_size_per_block);
    MallocInfo load_info{load_resource, load_token_ids};
    load_info.reuse_cache         = true;
    load_info.enable_cache_lookup = true;
    const auto load_result        = manager_->malloc(load_info);
    ASSERT_TRUE(load_result.success);
    EXPECT_EQ(load_result.reuse_len, 0);
    EXPECT_EQ(load_result.host_reuse_len, 0);
    EXPECT_EQ(load_result.disk_reuse_len, 0);
    ASSERT_NE(load_result.async_context, nullptr);
    const bool disk_load_entered = pausable_engine->waitUntilEnteredFor(
        std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
    if (!disk_load_entered) {
        pausable_engine->release();
    }
    ASSERT_TRUE(disk_load_entered);
    EXPECT_FALSE(load_result.async_context->done());
    EXPECT_GT(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    auto maybe_loading = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_loading.has_value());
    std::vector<BlockIndicesType> load_targets(cache->groupSets().size());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*maybe_loading)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
        ASSERT_TRUE(resource.hasTier(Tier::DISK));
        EXPECT_EQ(resource.getTopTier(), Tier::DISK);
        EXPECT_EQ(resource.disk_slot, disk_sources[group_set_id]);
        EXPECT_EQ(group_set->diskPool()->treeRefCount(resource.disk_slot), 2u);
        load_targets[group_set_id] = groupSetRequestBlocksAt(group_set, load_resource, 0, /*path_index=*/0);
        ASSERT_EQ(group_set->devicePools().size(), load_targets[group_set_id].size());
        for (size_t member_index = 0; member_index < group_set->groupIds().size(); ++member_index) {
            const int          group_id = static_cast<int>(group_set->groupIds()[member_index]);
            const BlockIdxType block    = load_targets[group_set_id][member_index];
            ASSERT_FALSE(isNullBlockIdx(block));
            EXPECT_EQ(group_set->devicePools()[member_index]->refCount(block), 2u);
            ASSERT_TRUE(
                fillGroupBlockPayload(manager_, cache_config_, group_id, block, /*path_index=*/0, /*poison=*/true));
        }
    }

    pausable_engine->release();
    load_result.async_context->waitDone();
    ASSERT_TRUE(load_result.async_context->done());
    ASSERT_TRUE(load_result.async_context->success()) << load_result.async_context->errorInfo().ToString();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    descriptors = pausable_engine->descriptors();
    ASSERT_EQ(descriptors.size(), 3 * cache->groupSets().size());
    std::vector<size_t> disk_to_device_by_group_set(cache->groupSets().size(), 0);
    for (size_t index = 2 * cache->groupSets().size(); index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::DISK);
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::DISK), disk_sources[descriptor.group_set_id]);
        EXPECT_EQ(descriptor.blocksAt(Tier::DEVICE), load_targets[descriptor.group_set_id]);
        ++disk_to_device_by_group_set[descriptor.group_set_id];
    }
    for (size_t group_set_id = 0; group_set_id < disk_to_device_by_group_set.size(); ++group_set_id) {
        EXPECT_EQ(disk_to_device_by_group_set[group_set_id], 1u);
    }

    auto maybe_loaded = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_loaded.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*maybe_loaded)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        ASSERT_TRUE(resource.hasTier(Tier::DEVICE));
        EXPECT_EQ(resource.getTopTier(), Tier::DEVICE);
        EXPECT_EQ(resource.device_blocks, load_targets[group_set_id]);
        EXPECT_FALSE(group_set->diskPool()->isAllocated(disk_sources[group_set_id]));
        ASSERT_EQ(group_set->devicePools().size(), load_targets[group_set_id].size());
        for (size_t member_index = 0; member_index < group_set->devicePools().size(); ++member_index) {
            EXPECT_EQ(group_set->devicePools()[member_index]->refCount(load_targets[group_set_id][member_index]), 2u);
        }
    }
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, load_resource, /*logical_reuse_blocks=*/1));

    manager_->free(FreeInfo{load_resource, load_token_ids});
    const size_t submits_before_second_hit = pausable_engine->submittedDescriptorCount();
    auto         hit_resource              = makeResource(cache_config_);
    auto hit_token_ids = makeTokenIds(/*offset=*/0, 2 * seq_size_per_block, 2 * seq_size_per_block, seq_size_per_block);
    MallocInfo hit_info{hit_resource, hit_token_ids};
    hit_info.reuse_cache         = true;
    hit_info.enable_cache_lookup = true;
    const auto hit_result        = manager_->malloc(hit_info);
    ASSERT_TRUE(hit_result.success);
    EXPECT_EQ(hit_result.reuse_len, seq_size_per_block);
    EXPECT_EQ(hit_result.host_reuse_len, 0);
    EXPECT_EQ(hit_result.disk_reuse_len, 0);
    EXPECT_EQ(hit_result.async_context, nullptr);
    EXPECT_EQ(pausable_engine->submittedDescriptorCount(), submits_before_second_hit);
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, hit_resource, /*logical_reuse_blocks=*/1));

    manager_->free(FreeInfo{hit_resource, hit_token_ids});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4HostToDiskWatermarkFailureKeepsHostSourceMatchableAndCanRetry) {
    if (GetParam() != TierLayout::HOST_DISK) {
        GTEST_SKIP() << "HOST-to-DISK failure serviceability requires HostDisk layout";
    }
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    ASSERT_NE(manager_, nullptr);
    auto cache = manager_->blockTreeCache();

    auto recording_engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, recording_engine);
    transfer_engine_.reset();

    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       maybe_seed     = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/1);
    ASSERT_TRUE(maybe_seed.has_value());
    auto seed = std::move(*maybe_seed);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));
    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    for (const auto& group_set : cache->groupSets()) {
        device_pools.insert(device_pools.end(), group_set->devicePools().begin(), group_set->devicePools().end());
    }
    const auto device_ratio = oneUsedBlockWatermarkRatio(device_pools);
    ASSERT_TRUE(device_ratio.has_value());

    // Watermark maintenance may schedule more than one eviction plan while the
    // first failed task is settling. Fail the whole maintenance window so a
    // later plan cannot accidentally turn this into a partial-success case.
    for (size_t index = 0; index < cache->groupSets().size() * 4; ++index) {
        recording_engine->enqueueResult(/*success=*/false);
    }
    const size_t failed_submit_count = recording_engine->submittedDescriptorCount();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << recording_engine->submittedDescriptorCount();

    const auto after_device_failure = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(after_device_failure.has_value());
    ASSERT_EQ(after_device_failure->size(), 1u);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*after_device_failure)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        ASSERT_TRUE(resource.hasTier(Tier::DEVICE));
        EXPECT_FALSE(resource.hasTier(Tier::HOST));
        EXPECT_EQ(resource.getTopTier(), Tier::DEVICE);
        EXPECT_EQ(resource.device_blocks, groupSetSeedBlocksAt(group_set, seed, /*path_index=*/0));
        ASSERT_EQ(group_set->devicePools().size(), resource.device_blocks.size());
        for (size_t member_index = 0; member_index < group_set->devicePools().size(); ++member_index) {
            EXPECT_EQ(group_set->devicePools()[member_index]->refCount(resource.device_blocks[member_index]), 1u);
        }
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
        EXPECT_EQ(group_set->hostPool()->usedBlocksNum(), 0u);
        ASSERT_NE(group_set->diskPool(), nullptr);
        EXPECT_EQ(group_set->diskPool()->usedBlocksNum(), 0u);
    }
    EXPECT_GT(recording_engine->submittedDescriptorCount(), failed_submit_count);
    expectPoolSnapshotsEq(initial_lower, snapshotLowerPools(*cache, GetParam()));
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));

    auto       hit_resource = makeResource(cache_config_);
    auto       hit_tokens   = makeTokenIds(/*offset=*/0,
                                   2 * static_cast<int>(cache_config_.seq_size_per_block),
                                   2 * static_cast<int>(cache_config_.seq_size_per_block),
                                   static_cast<int>(cache_config_.seq_size_per_block));
    MallocInfo hit_info{hit_resource, hit_tokens};
    hit_info.reuse_cache            = true;
    hit_info.enable_cache_lookup    = true;
    const size_t submits_before_hit = recording_engine->submittedDescriptorCount();
    const auto   hit_result         = manager_->malloc(hit_info);
    ASSERT_TRUE(hit_result.success);
    EXPECT_EQ(hit_result.reuse_len, static_cast<int>(cache_config_.seq_size_per_block));
    EXPECT_EQ(hit_result.host_reuse_len, 0);
    EXPECT_EQ(hit_result.disk_reuse_len, 0);
    EXPECT_EQ(hit_result.async_context, nullptr);
    EXPECT_EQ(recording_engine->submittedDescriptorCount(), submits_before_hit);
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, hit_resource, /*logical_reuse_blocks=*/1));
    ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
        manager_, *cache, cache_config_, seed.cache_keys, hit_resource, /*logical_reuse_blocks=*/1));
    manager_->free(FreeInfo{hit_resource, hit_tokens});

    recording_engine->clearScriptedResults();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << recording_engine->submittedDescriptorCount();

    auto maybe_host = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_host.has_value());
    ASSERT_EQ(maybe_host->size(), 1u);
    std::vector<BlockIdxType> host_sources_before_failure(cache->groupSets().size(), NULL_BLOCK_IDX);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*maybe_host)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        ASSERT_TRUE(resource.hasTier(Tier::HOST));
        EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
        EXPECT_EQ(resource.getTopTier(), Tier::HOST);
        host_sources_before_failure[group_set_id] = resource.host_block;
        EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
    }

    {
        std::vector<std::shared_ptr<IBlockPool>> host_pools;
        for (const auto& group_set : cache->groupSets()) {
            host_pools.push_back(group_set->hostPool());
        }
        const auto host_ratio = oneUsedBlockWatermarkRatio(host_pools);
        ASSERT_TRUE(host_ratio.has_value());

        const auto lower_before_host_failure   = snapshotLowerPools(*cache, GetParam());
        const auto submits_before_host_failure = recording_engine->submittedDescriptorCount();
        for (size_t index = 0; index < cache->groupSets().size() * 4; ++index) {
            recording_engine->enqueueResult(/*success=*/false);
        }
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
        ASSERT_TRUE(waitForPendingTasksDoneFor(
            *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
            << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
            << " submits=" << recording_engine->submittedDescriptorCount();

        const auto descriptors_after_host_failure = recording_engine->descriptors();
        ASSERT_GT(descriptors_after_host_failure.size(), submits_before_host_failure);
        for (size_t index = submits_before_host_failure; index < descriptors_after_host_failure.size(); ++index) {
            const auto& descriptor = descriptors_after_host_failure[index];
            ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
            EXPECT_EQ(descriptor.source_tier, Tier::HOST);
            EXPECT_EQ(descriptor.target_tier, Tier::DISK);
            EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), host_sources_before_failure[descriptor.group_set_id]);
        }

        auto after_host_failure = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(after_host_failure.has_value());
        ASSERT_EQ(after_host_failure->size(), 1u);
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*after_host_failure)[0][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            ASSERT_TRUE(resource.hasTier(Tier::HOST));
            EXPECT_FALSE(resource.hasTier(Tier::DISK));
            EXPECT_EQ(resource.getTopTier(), Tier::HOST);
            EXPECT_EQ(resource.host_block, host_sources_before_failure[group_set_id]);
            EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
            EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
            EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
            EXPECT_EQ(group_set->diskPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
            EXPECT_EQ(group_set->diskPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
        }
        expectPoolSnapshotsEq(lower_before_host_failure, snapshotLowerPools(*cache, GetParam()));

        // Prove the preserved HOST copy remains manager-serviceable before the
        // demotion retry. A successful load consumes that HOST copy, so the
        // same cached path is demoted back to HOST below before retrying H2Dk.
        recording_engine->clearScriptedResults();
        const size_t submits_before_host_hit = recording_engine->submittedDescriptorCount();
        auto         host_hit_resource       = makeResource(cache_config_);
        auto         host_hit_tokens         = makeTokenIds(
            /*offset=*/0,
            2 * static_cast<int>(cache_config_.seq_size_per_block),
            2 * static_cast<int>(cache_config_.seq_size_per_block),
            static_cast<int>(cache_config_.seq_size_per_block));
        MallocInfo host_hit_info{host_hit_resource, host_hit_tokens};
        host_hit_info.reuse_cache         = true;
        host_hit_info.enable_cache_lookup = true;
        const auto host_hit_result        = manager_->malloc(host_hit_info);
        ASSERT_TRUE(host_hit_result.success);
        EXPECT_EQ(host_hit_result.reuse_len, 0);
        EXPECT_EQ(host_hit_result.host_reuse_len, 0);
        EXPECT_EQ(host_hit_result.disk_reuse_len, 0);
        ASSERT_NE(host_hit_result.async_context, nullptr);
        ASSERT_TRUE(
            waitForAsyncContextDoneFor(host_hit_result.async_context,
                                       std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)));
        host_hit_result.async_context->waitDone();
        ASSERT_TRUE(host_hit_result.async_context->success()) << host_hit_result.async_context->errorInfo().ToString();
        ASSERT_TRUE(waitForPendingTasksDoneFor(
            *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
            << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
            << " submits=" << recording_engine->submittedDescriptorCount();

        const auto descriptors_after_host_hit = recording_engine->descriptors();
        ASSERT_EQ(descriptors_after_host_hit.size(), submits_before_host_hit + cache->groupSets().size());
        for (size_t index = submits_before_host_hit; index < descriptors_after_host_hit.size(); ++index) {
            const auto& descriptor = descriptors_after_host_hit[index];
            ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
            EXPECT_EQ(descriptor.source_tier, Tier::HOST);
            EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
            EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), host_sources_before_failure[descriptor.group_set_id]);
        }
        ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
        ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
        ASSERT_TRUE(requestReusesExpectedPath(
            *cache, cache_config_, seed.cache_keys, host_hit_resource, /*logical_reuse_blocks=*/1));
        ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
            manager_, *cache, cache_config_, seed.cache_keys, host_hit_resource, /*logical_reuse_blocks=*/1));
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            EXPECT_FALSE(
                cache->groupSets()[group_set_id]->hostPool()->isAllocated(host_sources_before_failure[group_set_id]));
        }
        manager_->free(FreeInfo{host_hit_resource, host_hit_tokens});

        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
        ASSERT_TRUE(waitForPendingTasksDoneFor(
            *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
            << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
            << " submits=" << recording_engine->submittedDescriptorCount();
        auto rebuilt_host = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(rebuilt_host.has_value());
        ASSERT_EQ(rebuilt_host->size(), 1u);
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*rebuilt_host)[0][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            EXPECT_EQ(resource.getTopTier(), Tier::HOST);
            EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
            EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
            EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
        }
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
        ASSERT_TRUE(waitForPendingTasksDoneFor(
            *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
            << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
            << " submits=" << recording_engine->submittedDescriptorCount();

        auto after_host_retry = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(after_host_retry.has_value());
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*after_host_retry)[0][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            ASSERT_TRUE(resource.hasTier(Tier::DISK));
            EXPECT_FALSE(resource.hasTier(Tier::HOST));
            EXPECT_EQ(resource.getTopTier(), Tier::DISK);
            EXPECT_EQ(group_set->diskPool()->treeRefCount(resource.disk_slot), 1u);
        }
    }

    auto       load_resource = makeResource(cache_config_);
    auto       load_tokens   = makeTokenIds(/*offset=*/0,
                                    2 * static_cast<int>(cache_config_.seq_size_per_block),
                                    2 * static_cast<int>(cache_config_.seq_size_per_block),
                                    static_cast<int>(cache_config_.seq_size_per_block));
    MallocInfo load_info{load_resource, load_tokens};
    load_info.reuse_cache         = true;
    load_info.enable_cache_lookup = true;
    const auto load_result        = manager_->malloc(load_info);
    ASSERT_TRUE(load_result.success);
    ASSERT_NE(load_result.async_context, nullptr);
    EXPECT_EQ(load_result.reuse_len, 0);
    EXPECT_EQ(load_result.host_reuse_len, 0);
    EXPECT_EQ(load_result.disk_reuse_len, 0);
    ASSERT_TRUE(waitForAsyncContextDoneFor(
        load_result.async_context, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)));
    load_result.async_context->waitDone();
    ASSERT_TRUE(load_result.async_context->success()) << load_result.async_context->errorInfo().ToString();
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << recording_engine->submittedDescriptorCount();
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, load_resource, /*logical_reuse_blocks=*/1));
    ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
        manager_, *cache, cache_config_, seed.cache_keys, load_resource, /*logical_reuse_blocks=*/1));

    manager_->free(FreeInfo{load_resource, load_tokens});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4DemotingDeviceHitIsNotReselected) {
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    ASSERT_NE(manager_, nullptr);
    auto cache  = manager_->blockTreeCache();
    auto engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, engine);
    transfer_engine_.reset();
    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       seed_opt       = seedDevicePrefix(manager_, cache_config_, 0, 1);
    ASSERT_TRUE(seed_opt.has_value());
    auto seed = std::move(*seed_opt);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));
    std::vector<std::shared_ptr<IBlockPool>> pools;
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        appendDevicePools(group_set, pools);
    }
    auto ratio = oneUsedBlockWatermarkRatio(pools);
    ASSERT_TRUE(ratio.has_value());
    ASSERT_TRUE(engine->armPause());
    ScopedTransferRelease release(engine);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    const size_t pending_tasks = static_cast<size_t>(BlockTreeCacheTestPeer::pendingTasksForTest(*cache));
    const size_t worker_count  = static_cast<size_t>(cache->config().task_pool_size);
    const size_t expected_entered_count = std::min(pending_tasks, worker_count);
    ASSERT_GT(expected_entered_count, 0u);
    ASSERT_TRUE(engine->waitUntilEnteredCountFor(
        expected_entered_count, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)));
    auto demoting = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(demoting.has_value());
    ASSERT_EQ(demoting->size(), 1u);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& state     = (*demoting)[0][group_set_id];
        EXPECT_EQ(state.transfer_state, GroupSetTransferState::DEMOTING);
        ASSERT_TRUE(state.hasTier(Tier::DEVICE));
        EXPECT_FALSE(state.hasTier(Tier::HOST))
            << "the demotion target is held by the eviction ticket until settlement";
        EXPECT_EQ(state.device_blocks, groupSetSeedBlocksAt(group_set, seed, /*path_index=*/0));
        ASSERT_EQ(group_set->devicePools().size(), state.device_blocks.size());
        for (size_t member_index = 0; member_index < group_set->devicePools().size(); ++member_index) {
            EXPECT_EQ(group_set->devicePools()[member_index]->refCount(state.device_blocks[member_index]), 1u);
        }
        EXPECT_EQ(group_set->hostPool()->usedBlocksNum(), 1u);
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
    }
    auto       resource = makeResource(cache_config_);
    auto       tokens   = makeTokenIds(0,
                               2 * cache_config_.seq_size_per_block,
                               2 * cache_config_.seq_size_per_block,
                               cache_config_.seq_size_per_block);
    MallocInfo info{resource, tokens};
    info.reuse_cache         = true;
    info.enable_cache_lookup = true;
    auto result              = manager_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    // DEMOTING resources are intentionally not matchable; the in-flight source
    // remains retained by the eviction ticket rather than a new request.
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(result.host_reuse_len, 0);
    EXPECT_EQ(result.disk_reuse_len, 0);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), pending_tasks);
    auto still_demoting = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(still_demoting.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& state     = (*still_demoting)[0][group_set_id];
        EXPECT_EQ(state.transfer_state, GroupSetTransferState::DEMOTING);
        ASSERT_EQ(group_set->devicePools().size(), state.device_blocks.size());
        for (size_t member_index = 0; member_index < group_set->devicePools().size(); ++member_index) {
            EXPECT_EQ(group_set->devicePools()[member_index]->refCount(state.device_blocks[member_index]), 1u)
                << "the miss request must not reference the DEMOTING source";
        }
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
    }
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), pending_tasks);
    engine->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    auto settled_host = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(settled_host.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& state     = (*settled_host)[0][group_set_id];
        EXPECT_EQ(state.transfer_state, GroupSetTransferState::IDLE);
        ASSERT_TRUE(state.hasTier(Tier::HOST));
        EXPECT_FALSE(state.hasTier(Tier::DEVICE));
        EXPECT_EQ(group_set->hostPool()->treeRefCount(state.host_block), 1u);
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }
    manager_->free(FreeInfo{resource, tokens});

    auto       load_resource = makeResource(cache_config_);
    auto       load_tokens   = makeTokenIds(0,
                                    2 * cache_config_.seq_size_per_block,
                                    2 * cache_config_.seq_size_per_block,
                                    cache_config_.seq_size_per_block);
    MallocInfo load_info{load_resource, load_tokens};
    load_info.reuse_cache         = true;
    load_info.enable_cache_lookup = true;
    const auto load_result        = manager_->malloc(load_info);
    ASSERT_TRUE(load_result.success);
    EXPECT_EQ(load_result.reuse_len, 0);
    EXPECT_EQ(load_result.host_reuse_len, 0);
    EXPECT_EQ(load_result.disk_reuse_len, 0);
    ASSERT_NE(load_result.async_context, nullptr);
    load_result.async_context->waitDone();
    ASSERT_TRUE(load_result.async_context->success()) << load_result.async_context->errorInfo().ToString();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, load_resource, /*logical_reuse_blocks=*/1));
    ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
        manager_, *cache, cache_config_, seed.cache_keys, load_resource, /*logical_reuse_blocks=*/1));
    manager_->free(FreeInfo{load_resource, load_tokens});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

}  // namespace rtp_llm::test
