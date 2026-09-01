#include "rtp_llm/cpp/cache/test/KVCacheManagerWithTierCacheTestBase.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm::test {
using namespace tier_cache_test_detail;

TEST_P(KVCacheManagerWithTierCacheTest, DSV4CpCanonicalFullAndSwaRoundTripThroughDisk) {
    if (GetParam() != TierLayout::HOST_DISK) {
        GTEST_SKIP() << "CP disk round-trip requires the HostDisk layout";
    }

    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/8));
    ASSERT_NE(manager_, nullptr);
    auto cache = manager_->blockTreeCache();

    auto cp_mapper = std::make_shared<CPSlotMapper>(
        /*cp_rank=*/0, /*cp_size=*/2, static_cast<int>(cache_config_.seq_size_per_block));
    manager_->cp_slot_mapper_ = cp_mapper;
    manager_->allocator_->setCPSlotMapper(cp_mapper);
    ASSERT_EQ(manager_->cpSlotMapper(), cp_mapper);
    ASSERT_TRUE(cp_mapper->isSharded());

    auto pausable_engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_engine);
    transfer_engine_.reset();

    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       maybe_seed =
        seedCpCanonicalDevicePrefix(manager_, cache_config_, cp_mapper, /*token_offset=*/0, /*logical_blocks=*/2);
    ASSERT_TRUE(maybe_seed.has_value());
    auto seed = std::move(*maybe_seed);

    ASSERT_EQ(seed.full_cache_keys.size(), 2u);
    ASSERT_EQ(seed.cache_keys.size(), 1u);
    EXPECT_EQ(seed.cache_keys, cp_mapper->canonicalCacheKeys(seed.full_cache_keys));
    EXPECT_EQ(seed.cache_keys.front(), seed.full_cache_keys.back());
    EXPECT_TRUE(seed.resource->lastBlockAligned());
    const auto key_snapshot = cache->getKeySnapshot(/*limit=*/2);
    EXPECT_EQ(key_snapshot.keys, seed.cache_keys);
    EXPECT_TRUE(cache->tree()->findNode(seed.full_cache_keys).empty())
        << "tree must be keyed by the CP canonical namespace";

    bool saw_full = false;
    bool saw_swa  = false;
    for (const auto& group_set : cache->groupSets()) {
        ASSERT_NE(group_set, nullptr);
        for (const size_t raw_group_id : group_set->groupIds()) {
            ASSERT_LT(raw_group_id, static_cast<size_t>(cache_config_.groupNums()));
            const auto type = cache_config_.typeForGroup(raw_group_id);
            if (type == CacheGroupType::FULL) {
                saw_full = true;
                EXPECT_TRUE(cp_mapper->blockRoundRobinGroup(cache_config_, raw_group_id));
                EXPECT_FALSE(cp_mapper->compactLastRankGroup(cache_config_, raw_group_id));
            } else {
                ASSERT_EQ(type, CacheGroupType::SWA);
                saw_swa = true;
                EXPECT_FALSE(cp_mapper->blockRoundRobinGroup(cache_config_, raw_group_id));
                EXPECT_TRUE(cp_mapper->compactLastRankGroup(cache_config_, raw_group_id));
            }
            const auto position =
                cpCanonicalBlockPosition(*cp_mapper, cache_config_, static_cast<int>(raw_group_id), 0);
            ASSERT_TRUE(position.has_value());
            ASSERT_LT(*position, seed.blocks_by_group[raw_group_id].size());
            EXPECT_FALSE(isNullBlockIdx(seed.blocks_by_group[raw_group_id][*position]));
        }
    }
    EXPECT_TRUE(saw_full);
    EXPECT_TRUE(saw_swa);
    ASSERT_TRUE(fillCpCanonicalSeedPayload(manager_, cache_config_, *cp_mapper, seed));
    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    for (const auto& group_set : cache->groupSets()) {
        ASSERT_EQ(group_set->devicePools().size(), group_set->groupIds().size());
        device_pools.insert(device_pools.end(), group_set->devicePools().begin(), group_set->devicePools().end());
    }
    const auto device_ratio = oneUsedBlockWatermarkRatio(device_pools);
    ASSERT_TRUE(device_ratio.has_value());
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    auto maybe_host = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_host.has_value());
    ASSERT_EQ(maybe_host->size(), 1u);
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
        EXPECT_EQ(group_set->diskPool()->usedBlocksNum(), 0u);
    }

    auto descriptors = pausable_engine->descriptors();
    ASSERT_EQ(descriptors.size(), cache->groupSets().size());
    std::vector<size_t> device_to_host(cache->groupSets().size(), 0);
    for (const auto& descriptor : descriptors) {
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.target_tier, Tier::HOST);
        const GroupSetPtr& group_set = cache->groupSets()[descriptor.group_set_id];
        BlockIndicesType   expected_blocks;
        for (const size_t group_id : group_set->groupIds()) {
            const std::optional<size_t> position =
                cpCanonicalBlockPosition(*cp_mapper, cache_config_, static_cast<int>(group_id), 0);
            ASSERT_TRUE(position.has_value());
            expected_blocks.push_back(seed.blocks_by_group[group_id][*position]);
        }
        EXPECT_EQ(descriptor.blocksAt(Tier::DEVICE), expected_blocks);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), host_sources[descriptor.group_set_id]);
        ++device_to_host[descriptor.group_set_id];
    }
    for (size_t group_set_id = 0; group_set_id < device_to_host.size(); ++group_set_id) {
        EXPECT_EQ(device_to_host[group_set_id], 1u);
    }

    std::vector<std::shared_ptr<IBlockPool>> host_pools;
    for (const auto& group_set : cache->groupSets()) {
        host_pools.push_back(group_set->hostPool());
    }
    const auto host_ratio = oneUsedBlockWatermarkRatio(host_pools);
    ASSERT_TRUE(host_ratio.has_value());
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
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
        disk_sources[group_set_id] = resource.disk_block;
        EXPECT_FALSE(group_set->hostPool()->isAllocated(host_sources[group_set_id]));
        EXPECT_EQ(group_set->diskPool()->treeRefCount(resource.disk_block), 1u);
        EXPECT_EQ(group_set->diskPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
        EXPECT_EQ(group_set->diskPool()->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    }

    descriptors = pausable_engine->descriptors();
    ASSERT_EQ(descriptors.size(), 2 * cache->groupSets().size());
    std::vector<size_t> host_to_disk(cache->groupSets().size(), 0);
    for (size_t index = cache->groupSets().size(); index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DISK);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), host_sources[descriptor.group_set_id]);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::DISK), disk_sources[descriptor.group_set_id]);
        ++host_to_disk[descriptor.group_set_id];
    }
    for (size_t group_set_id = 0; group_set_id < host_to_disk.size(); ++group_set_id) {
        EXPECT_EQ(host_to_disk[group_set_id], 1u);
    }

    ASSERT_TRUE(pausable_engine->armPause());
    ScopedTransferRelease load_release(pausable_engine);
    const int             seq_size_per_block = static_cast<int>(cache_config_.seq_size_per_block);
    auto                  load_resource      = makeResource(cache_config_);
    auto                  load_token_ids =
        makeTokenIds(/*offset=*/0, 2 * seq_size_per_block + 1, 2 * seq_size_per_block + 1, seq_size_per_block);
    MallocInfo load_info{load_resource, load_token_ids};
    load_info.reuse_cache         = true;
    load_info.enable_cache_lookup = true;
    const auto load_result        = manager_->malloc(load_info);
    ASSERT_TRUE(load_result.success);
    EXPECT_EQ(load_result.reuse_len, 0);
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
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*maybe_loading)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
        ASSERT_TRUE(resource.hasTier(Tier::DISK));
        EXPECT_EQ(resource.getTopTier(), Tier::DISK);
        EXPECT_EQ(resource.disk_block, disk_sources[group_set_id]);
        EXPECT_EQ(group_set->diskPool()->treeRefCount(resource.disk_block), 2u);
        ASSERT_EQ(group_set->groupIds().size(), group_set->devicePools().size());
        for (size_t member_index = 0; member_index < group_set->groupIds().size(); ++member_index) {
            const int                   group_id = static_cast<int>(group_set->groupIds()[member_index]);
            const std::optional<size_t> position = cpCanonicalBlockPosition(*cp_mapper, cache_config_, group_id, 0);
            ASSERT_TRUE(position.has_value());
            const BlockIndicesType& blocks = load_resource->blocks(0, group_id);
            ASSERT_LT(*position, blocks.size());
            ASSERT_FALSE(isNullBlockIdx(blocks[*position]));
            load_targets[group_set_id].push_back(blocks[*position]);
            EXPECT_EQ(group_set->devicePools()[member_index]->refCount(blocks[*position]), 2u);
            ASSERT_TRUE(fillGroupBlockPayload(
                manager_, cache_config_, group_id, blocks[*position], /*path_index=*/0, /*poison=*/true));
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
    std::vector<size_t> disk_to_device(cache->groupSets().size(), 0);
    for (size_t index = 2 * cache->groupSets().size(); index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::DISK);
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::DISK), disk_sources[descriptor.group_set_id]);
        EXPECT_EQ(descriptor.blocksAt(Tier::DEVICE), load_targets[descriptor.group_set_id]);
        ++disk_to_device[descriptor.group_set_id];
    }
    for (size_t group_set_id = 0; group_set_id < disk_to_device.size(); ++group_set_id) {
        EXPECT_EQ(disk_to_device[group_set_id], 1u);
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
    ASSERT_TRUE(requestReusesExpectedCpCanonicalPath(
        *cache, cache_config_, *cp_mapper, seed.cache_keys, load_resource, /*logical_reuse_blocks=*/1));

    manager_->free(FreeInfo{load_resource, load_token_ids});
    const size_t submits_before_second_hit = pausable_engine->submittedDescriptorCount();
    auto         hit_resource              = makeResource(cache_config_);
    auto         hit_token_ids =
        makeTokenIds(/*offset=*/0, 2 * seq_size_per_block + 1, 2 * seq_size_per_block + 1, seq_size_per_block);
    MallocInfo hit_info{hit_resource, hit_token_ids};
    hit_info.reuse_cache         = true;
    hit_info.enable_cache_lookup = true;
    const auto hit_result        = manager_->malloc(hit_info);
    ASSERT_TRUE(hit_result.success);
    EXPECT_EQ(hit_result.reuse_len, 2 * seq_size_per_block);
    EXPECT_EQ(hit_result.host_reuse_len, 0);
    EXPECT_EQ(hit_result.disk_reuse_len, 0);
    EXPECT_EQ(hit_result.async_context, nullptr);
    EXPECT_EQ(pausable_engine->submittedDescriptorCount(), submits_before_second_hit);
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
    ASSERT_TRUE(requestReusesExpectedCpCanonicalPath(
        *cache, cache_config_, *cp_mapper, seed.cache_keys, hit_resource, /*logical_reuse_blocks=*/1));

    manager_->free(FreeInfo{hit_resource, hit_token_ids});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4MixedDeviceHostDiskSegmentsLoadBack) {
    if (GetParam() != TierLayout::HOST_DISK) {
        GTEST_SKIP() << "mixed DEVICE+HOST+DISK segmentation requires HostDisk layout";
    }
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    auto cache  = manager_->blockTreeCache();
    auto engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, engine);
    transfer_engine_.reset();
    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       seed_opt       = seedDevicePrefix(manager_, cache_config_, 0, 3);
    ASSERT_TRUE(seed_opt.has_value());
    auto seed = std::move(*seed_opt);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));

    // Keep the first logical path resident while a two-block excess moves the
    // remaining paths through HOST and DISK.
    auto       guard        = makeResource(cache_config_);
    auto       guard_tokens = makeTokenIds(0,
                                     2 * cache_config_.seq_size_per_block,
                                     2 * cache_config_.seq_size_per_block,
                                     cache_config_.seq_size_per_block);
    MallocInfo guard_info{guard, guard_tokens};
    guard_info.reuse_cache         = true;
    guard_info.enable_cache_lookup = true;
    auto guard_result              = manager_->malloc(guard_info);
    ASSERT_TRUE(guard_result.success);
    ASSERT_EQ(guard_result.reuse_len, static_cast<int>(cache_config_.seq_size_per_block));

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    for (const auto& gs : cache->groupSets()) {
        device_pools.insert(device_pools.end(), gs->devicePools().begin(), gs->devicePools().end());
    }
    ASSERT_FALSE(device_pools.empty());
    const auto device_ratio = blockExcessWatermarkRatio(device_pools, /*excess_blocks=*/2);
    ASSERT_TRUE(device_ratio.has_value());
    // FULL path 1 is not an eviction candidate until its child path has
    // settled at HOST, whereas the SWA candidates can be selected in the first
    // pass. Re-running the same target ratio catches up only the still-excess
    // FULL pools; pools already at the target submit nothing.
    for (int pass = 0; pass < 2; ++pass) {
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    }
    auto staged = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(staged.has_value());
    ASSERT_EQ(staged->size(), 3u);
    for (size_t path = 0; path < staged->size(); ++path) {
        for (size_t gid = 0; gid < cache->groupSets().size(); ++gid) {
            const auto& resource = (*staged)[path][gid];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            const Tier expected = path == 0 ? Tier::DEVICE : Tier::HOST;
            EXPECT_EQ(resource.getTopTier(), expected) << "path=" << path << " group_set=" << gid;
        }
    }

    // First load path 1 from HOST. A lower-tier load-back does not itself make
    // that path hot because BlockTreeMatcher only refreshes ready DEVICE
    // resources during match.
    auto       touch_resource = makeResource(cache_config_);
    auto       touch_tokens   = makeTokenIds(0,
                                     3 * cache_config_.seq_size_per_block,
                                     3 * cache_config_.seq_size_per_block,
                                     cache_config_.seq_size_per_block);
    MallocInfo touch_info{touch_resource, touch_tokens};
    touch_info.reuse_cache         = true;
    touch_info.enable_cache_lookup = true;
    const auto touch_result        = manager_->malloc(touch_info);
    ASSERT_TRUE(touch_result.success);
    EXPECT_EQ(touch_result.reuse_len, static_cast<int>(cache_config_.seq_size_per_block));
    EXPECT_EQ(touch_result.host_reuse_len, 0);
    EXPECT_EQ(touch_result.disk_reuse_len, 0);
    ASSERT_NE(touch_result.async_context, nullptr);
    touch_result.async_context->waitDone();
    ASSERT_TRUE(touch_result.async_context->success()) << touch_result.async_context->errorInfo().ToString();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, touch_resource, /*logical_reuse_blocks=*/2));
    ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
        manager_, *cache, cache_config_, seed.cache_keys, touch_resource, /*logical_reuse_blocks=*/2));
    manager_->free(FreeInfo{touch_resource, touch_tokens});

    // Now both path 0 and path 1 are ready on DEVICE. This synchronous hit is
    // the event that updates path 1's LRU heat before it is returned to HOST.
    auto       heat_resource = makeResource(cache_config_);
    auto       heat_tokens   = makeTokenIds(0,
                                    3 * cache_config_.seq_size_per_block,
                                    3 * cache_config_.seq_size_per_block,
                                    cache_config_.seq_size_per_block);
    MallocInfo heat_info{heat_resource, heat_tokens};
    heat_info.reuse_cache         = true;
    heat_info.enable_cache_lookup = true;
    const auto heat_result        = manager_->malloc(heat_info);
    ASSERT_TRUE(heat_result.success);
    EXPECT_EQ(heat_result.reuse_len, 2 * static_cast<int>(cache_config_.seq_size_per_block));
    EXPECT_EQ(heat_result.host_reuse_len, 0);
    EXPECT_EQ(heat_result.disk_reuse_len, 0);
    EXPECT_EQ(heat_result.async_context, nullptr);
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, heat_resource, /*logical_reuse_blocks=*/2));
    manager_->free(FreeInfo{heat_resource, heat_tokens});

    // Keep the first logical block hotter than the loaded block in every group
    // set. SWA matching may not touch this block, and request refs intentionally
    // no longer pin it against eviction.
    const auto matched_path = cache->tree()->findNode(seed.cache_keys);
    ASSERT_GE(matched_path.size(), 2u);
    BlockTreeCacheTestPeer::markPathMatchedForTest(*cache, {matched_path[0]});

    const auto return_path_one_ratio = blockExcessWatermarkRatio(device_pools, /*excess_blocks=*/1);
    ASSERT_TRUE(return_path_one_ratio.has_value());
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *return_path_one_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    staged = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(staged.has_value());
    for (size_t path = 0; path < staged->size(); ++path) {
        for (size_t gid = 0; gid < cache->groupSets().size(); ++gid) {
            const Tier expected = path == 0 ? Tier::DEVICE : Tier::HOST;
            EXPECT_EQ((*staged)[path][gid].getTopTier(), expected)
                << "post-touch path=" << path << " group_set=" << gid;
        }
    }

    // Select a real HOST victim from one FULL group set. Path 2 is colder than
    // the re-heated path 1, and reverse cascading moves every group set on that
    // tier leaf to DISK in the same plan.
    std::vector<size_t> full_group_set_ids;
    for (const auto& group_set : cache->groupSets()) {
        if (group_set->groupType() == CacheGroupType::FULL) {
            full_group_set_ids.push_back(group_set->groupSetId());
        }
    }
    ASSERT_FALSE(full_group_set_ids.empty());
    ASSERT_TRUE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*cache, full_group_set_ids.front(), Tier::HOST));
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    auto mixed = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(mixed.has_value());
    ASSERT_EQ(mixed->size(), 3u);
    std::vector<BlockIdxType> host_sources(cache->groupSets().size(), NULL_BLOCK_IDX);
    std::vector<BlockIdxType> disk_sources(cache->groupSets().size(), NULL_BLOCK_IDX);
    for (size_t path = 0; path < mixed->size(); ++path) {
        for (size_t gid = 0; gid < cache->groupSets().size(); ++gid) {
            const auto& resource = (*mixed)[path][gid];
            const Tier  expected = path == 0 ? Tier::DEVICE : (path == 1 ? Tier::HOST : Tier::DISK);
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            EXPECT_EQ(resource.getTopTier(), expected) << "path=" << path << " group_set=" << gid;
            if (path == 1) {
                host_sources[gid] = resource.host_block;
            } else if (path == 2) {
                disk_sources[gid] = resource.disk_block;
            }
        }
    }

    manager_->free(FreeInfo{guard, guard_tokens});

    size_t failed_host_loads = 0;
    for (const auto& group_set : cache->groupSets()) {
        const size_t reuse_count = group_set->computeReuseBlockCount(/*matched_blocks=*/3);
        const size_t reuse_begin = 3 - reuse_count;
        failed_host_loads += reuse_begin <= 1 ? 1u : 0u;
    }
    ASSERT_GT(failed_host_loads, 0u);
    for (size_t index = 0; index < failed_host_loads; ++index) {
        engine->enqueueResult(/*success=*/true);
    }
    engine->enqueueResult(/*success=*/false);

    const size_t descriptors_before_failure = engine->submittedDescriptorCount();
    ASSERT_TRUE(engine->armPause());
    ScopedTransferRelease failure_release(engine);

    const int block_size = static_cast<int>(cache_config_.seq_size_per_block);
    auto      input_ids  = torch::empty({4 * block_size}, torch::kInt32);
    auto*     input_data = input_ids.data_ptr<int32_t>();
    for (int index = 0; index < 4 * block_size; ++index) {
        input_data[index] = index;
    }
    auto generate_input                                = std::make_shared<GenerateInput>();
    generate_input->input_ids                          = std::move(input_ids);
    generate_input->generate_config                    = std::make_shared<GenerateConfig>();
    generate_input->generate_config->reuse_cache       = true;
    generate_input->generate_config->enable_host_cache = true;

    ResourceContext resource_context;
    resource_context.cache_manager     = manager_;
    resource_context.reuse_cache       = true;
    resource_context.enable_host_cache = true;
    resource_context.enable_disk_cache = true;
    resource_context.role_type         = RoleType::PREFILL;

    ModelConfig model_config;
    model_config.max_seq_len                  = 2048;
    model_config.attn_config.tokens_per_block = block_size;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 1;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 2048;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PREFILL;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;

    auto prefill_stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);
    auto scheduler = std::make_shared<FIFOScheduler>(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, manager_);
    ASSERT_TRUE(scheduler->enqueue(prefill_stream).ok());
    auto first_schedule = scheduler->schedule();
    ASSERT_TRUE(first_schedule.ok());
    EXPECT_TRUE(first_schedule.value().empty());
    EXPECT_EQ(prefill_stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(prefill_stream->reuseLength(), block_size);
    EXPECT_EQ(prefill_stream->streamCacheResource().kvCache().cacheResource(0).deviceReuseBlockNum(), 1u);

    const bool failure_entered =
        engine->waitUntilEnteredFor(std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
    if (!failure_entered) {
        engine->release();
    }
    ASSERT_TRUE(failure_entered);
    engine->release();
    ASSERT_TRUE(waitForPendingTasksDoneFor(
        *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)));

    auto second_schedule = scheduler->schedule();
    ASSERT_TRUE(second_schedule.ok());
    ASSERT_EQ(second_schedule.value().size(), 1u);
    EXPECT_EQ(second_schedule.value().front(), prefill_stream);
    EXPECT_EQ(prefill_stream->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(prefill_stream->hasError());
    EXPECT_EQ(prefill_stream->reuseLength(), block_size);
    EXPECT_EQ(prefill_stream->initialReuseLength(), block_size);
    EXPECT_EQ(prefill_stream->deviceReuseLength(), block_size);
    EXPECT_EQ(prefill_stream->hostReuseLength(), 0);
    EXPECT_EQ(prefill_stream->diskReuseLength(), 0);
    EXPECT_EQ(prefill_stream->streamCacheResource().kvCache().cacheResource(0).deviceReuseBlockNum(), 1u);

    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto&  group_set   = cache->groupSets()[group_set_id];
        const size_t reuse_count = group_set->computeReuseBlockCount(/*matched_blocks=*/3);
        const size_t reuse_begin = 3 - reuse_count;
        for (const size_t raw_group_id : group_set->groupIds()) {
            const BlockIndicesType& blocks =
                prefill_stream->streamCacheResource().kvCache().blocks(0, static_cast<int>(raw_group_id));
            for (size_t path = reuse_begin; path < 3; ++path) {
                ASSERT_LT(path, blocks.size());
                EXPECT_FALSE(isNullBlockIdx(blocks[path]));
            }
        }
    }

    const auto failure_descriptors = engine->descriptors();
    ASSERT_GE(failure_descriptors.size(), descriptors_before_failure + failed_host_loads + 1);
    for (size_t index = descriptors_before_failure; index < descriptors_before_failure + failed_host_loads; ++index) {
        EXPECT_EQ(failure_descriptors[index].source_tier, Tier::HOST);
        EXPECT_EQ(failure_descriptors[index].target_tier, Tier::DEVICE);
    }
    EXPECT_EQ(failure_descriptors[descriptors_before_failure + failed_host_loads].source_tier, Tier::DISK);
    EXPECT_EQ(failure_descriptors[descriptors_before_failure + failed_host_loads].target_tier, Tier::DEVICE);

    prefill_stream->reportError(ErrorCode::CANCELLED, "test cleanup");
    ASSERT_TRUE(scheduler->schedule().ok());
    EXPECT_TRUE(prefill_stream->streamCacheResource().isResourceReleased());

    const size_t descriptors_before_load = engine->submittedDescriptorCount();
    ASSERT_TRUE(engine->armPause());
    ScopedTransferRelease load_release(engine);
    auto                  load_resource = makeResource(cache_config_);
    auto                  load_tokens   = makeTokenIds(0,
                                    4 * cache_config_.seq_size_per_block,
                                    4 * cache_config_.seq_size_per_block,
                                    cache_config_.seq_size_per_block);
    MallocInfo            load_info{load_resource, load_tokens};
    load_info.reuse_cache         = true;
    load_info.enable_cache_lookup = true;
    const auto load_result        = manager_->malloc(load_info);
    ASSERT_TRUE(load_result.success);
    EXPECT_EQ(load_result.reuse_len, static_cast<int>(cache_config_.seq_size_per_block));
    EXPECT_EQ(load_result.host_reuse_len, 0);
    EXPECT_EQ(load_result.disk_reuse_len, 0);
    ASSERT_NE(load_result.async_context, nullptr);
    const bool entered =
        engine->waitUntilEnteredFor(std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
    if (!entered) {
        engine->release();
    }
    ASSERT_TRUE(entered);
    EXPECT_FALSE(load_result.async_context->done());

    const auto found = cache->tree()->findNode(seed.cache_keys);
    ASSERT_EQ(found.size(), seed.cache_keys.size());
    std::vector<size_t> expected_host_loads(cache->groupSets().size(), 0);
    std::vector<size_t> expected_disk_loads(cache->groupSets().size(), 1);
    size_t              expected_load_descriptors = 0;
    auto                loading                   = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(loading.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto&  group_set   = cache->groupSets()[group_set_id];
        const size_t reuse_count = group_set->computeReuseBlockCount(/*matched_blocks=*/3);
        ASSERT_GT(reuse_count, 0u);
        ASSERT_LE(reuse_count, 3u);
        const size_t reuse_begin          = 3 - reuse_count;
        expected_host_loads[group_set_id] = reuse_begin <= 1 ? 1u : 0u;
        expected_load_descriptors += expected_host_loads[group_set_id] + expected_disk_loads[group_set_id];
        for (size_t path = 1; path < 3; ++path) {
            const bool  reused = path >= reuse_begin;
            const auto& state  = (*loading)[path][group_set_id];
            EXPECT_EQ(state.transfer_state, reused ? GroupSetTransferState::LOADING : GroupSetTransferState::IDLE)
                << "path=" << path << " group_set=" << group_set_id;
            if (!reused) {
                EXPECT_EQ(state.getTopTier(), Tier::HOST);
                EXPECT_EQ(group_set->hostPool()->treeRefCount(state.host_block), 1u);
                continue;
            }
            for (const size_t raw_group_id : group_set->groupIds()) {
                const int               group_id = static_cast<int>(raw_group_id);
                const BlockIndicesType& blocks   = load_resource->blocks(0, group_id);
                ASSERT_GE(blocks.size(), 3u);
                ASSERT_FALSE(isNullBlockIdx(blocks[path]));
                ASSERT_TRUE(
                    fillGroupBlockPayload(manager_, cache_config_, group_id, blocks[path], path, /*poison=*/true));
            }
        }
    }

    engine->release();
    ASSERT_TRUE(waitForAsyncContextDoneFor(
        load_result.async_context, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)));
    load_result.async_context->waitDone();
    ASSERT_TRUE(load_result.async_context->success()) << load_result.async_context->errorInfo().ToString();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    const auto descriptors = engine->descriptors();
    ASSERT_EQ(descriptors.size(), descriptors_before_load + expected_load_descriptors);
    std::vector<size_t> host_loads(cache->groupSets().size(), 0);
    std::vector<size_t> disk_loads(cache->groupSets().size(), 0);
    for (size_t index = descriptors_before_load; index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
        if (descriptor.source_tier == Tier::HOST) {
            EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), host_sources[descriptor.group_set_id]);
            ++host_loads[descriptor.group_set_id];
        } else {
            EXPECT_EQ(descriptor.source_tier, Tier::DISK);
            EXPECT_EQ(descriptor.singleBlockAt(Tier::DISK), disk_sources[descriptor.group_set_id]);
            ++disk_loads[descriptor.group_set_id];
        }
    }
    auto loaded = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(loaded.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto&  group_set   = cache->groupSets()[group_set_id];
        const size_t reuse_count = group_set->computeReuseBlockCount(/*matched_blocks=*/3);
        const size_t reuse_begin = 3 - reuse_count;
        EXPECT_EQ(host_loads[group_set_id], expected_host_loads[group_set_id]);
        EXPECT_EQ(disk_loads[group_set_id], expected_disk_loads[group_set_id]);
        for (size_t path = 0; path < 3; ++path) {
            const auto& state = (*loaded)[path][group_set_id];
            EXPECT_EQ(state.transfer_state, GroupSetTransferState::IDLE);
            const Tier expected = path >= reuse_begin || path == 0 ? Tier::DEVICE : Tier::HOST;
            EXPECT_EQ(state.getTopTier(), expected) << "path=" << path << " group_set=" << group_set_id;
            if (expected == Tier::DEVICE) {
                ASSERT_EQ(state.device_blocks.size(), group_set->groupIds().size());
                for (size_t member_index = 0; member_index < group_set->groupIds().size(); ++member_index) {
                    const int group_id = static_cast<int>(group_set->groupIds()[member_index]);
                    EXPECT_TRUE(groupBlockPayloadMatches(
                        manager_, cache_config_, group_id, state.device_blocks[member_index], path));
                }
            }
        }
    }
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, load_resource, /*logical_reuse_blocks=*/3));
    ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
        manager_, *cache, cache_config_, seed.cache_keys, load_resource, /*logical_reuse_blocks=*/3));
    manager_->free(FreeInfo{load_resource, load_tokens});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4LongDiskRoundTripExceedsStagingCapacity) {
    if (GetParam() != TierLayout::HOST_DISK) {
        GTEST_SKIP() << "long disk round-trip requires HostDisk layout";
    }
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    auto             cache               = manager_->blockTreeCache();
    constexpr size_t staging_block_count = 2;
    auto engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets(), staging_block_count);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, engine);
    transfer_engine_.reset();
    EXPECT_EQ(cache->config().device_disk_staging_block_count, staging_block_count);

    const auto    initial_device = snapshotDevicePools(manager_);
    const auto    initial_lower  = snapshotLowerPools(*cache, GetParam());
    constexpr int logical_blocks = 5;
    auto seed_opt = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/logical_blocks);
    ASSERT_TRUE(seed_opt.has_value());
    auto seed = std::move(*seed_opt);
    ASSERT_EQ(seed.cache_keys.size(), static_cast<size_t>(logical_blocks));
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    std::vector<std::shared_ptr<IBlockPool>> host_pools;
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        appendDevicePools(group_set, device_pools);
        host_pools.push_back(group_set->hostPool());
    }
    const auto device_ratio = zeroTargetWatermarkRatio(device_pools);
    ASSERT_TRUE(device_ratio.has_value());
    size_t remaining_device = countTreeResourcesAtTier(*cache, Tier::DEVICE);
    ASSERT_EQ(remaining_device, static_cast<size_t>(logical_blocks) * cache->groupSets().size());
    for (int round = 0; round < logical_blocks && remaining_device > 0; ++round) {
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
        EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
        const size_t next = countTreeResourcesAtTier(*cache, Tier::DEVICE);
        EXPECT_LT(next, remaining_device) << "round=" << round;
        remaining_device = next;
    }
    ASSERT_EQ(remaining_device, 0u);
    ASSERT_EQ(countTreeResourcesAtTier(*cache, Tier::HOST),
              static_cast<size_t>(logical_blocks) * cache->groupSets().size());

    const size_t device_to_host_descriptors = static_cast<size_t>(logical_blocks) * cache->groupSets().size();
    ASSERT_EQ(engine->submittedDescriptorCount(), device_to_host_descriptors);
    auto descriptors = engine->descriptors();
    for (size_t index = 0; index < device_to_host_descriptors; ++index) {
        EXPECT_EQ(descriptors[index].source_tier, Tier::DEVICE);
        EXPECT_EQ(descriptors[index].target_tier, Tier::HOST);
    }

    const auto host_ratio = zeroTargetWatermarkRatio(host_pools);
    ASSERT_TRUE(host_ratio.has_value());
    size_t remaining_host = countTreeResourcesAtTier(*cache, Tier::HOST);
    for (int round = 0; round < logical_blocks && remaining_host > 0; ++round) {
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
        EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
        const size_t next = countTreeResourcesAtTier(*cache, Tier::HOST);
        EXPECT_LT(next, remaining_host) << "round=" << round;
        remaining_host = next;
    }
    ASSERT_EQ(remaining_host, 0u);
    ASSERT_EQ(countTreeResourcesAtTier(*cache, Tier::DISK),
              static_cast<size_t>(logical_blocks) * cache->groupSets().size());

    const size_t host_to_disk_begin = device_to_host_descriptors;
    const size_t disk_snapshot_begin =
        host_to_disk_begin + static_cast<size_t>(logical_blocks) * cache->groupSets().size();
    ASSERT_EQ(engine->submittedDescriptorCount(), disk_snapshot_begin);
    descriptors = engine->descriptors();
    for (size_t index = host_to_disk_begin; index < disk_snapshot_begin; ++index) {
        EXPECT_EQ(descriptors[index].source_tier, Tier::HOST);
        EXPECT_EQ(descriptors[index].target_tier, Tier::DISK);
    }

    auto disk_path = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(disk_path.has_value());
    ASSERT_EQ(disk_path->size(), static_cast<size_t>(logical_blocks));
    std::vector<std::vector<BlockIdxType>> disk_sources(
        static_cast<size_t>(logical_blocks), std::vector<BlockIdxType>(cache->groupSets().size(), NULL_BLOCK_IDX));
    for (size_t path_index = 0; path_index < disk_path->size(); ++path_index) {
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& state     = (*disk_path)[path_index][group_set_id];
            EXPECT_EQ(state.transfer_state, GroupSetTransferState::IDLE);
            EXPECT_EQ(state.getTopTier(), Tier::DISK);
            disk_sources[path_index][group_set_id] = state.disk_block;
            EXPECT_EQ(group_set->diskPool()->treeRefCount(state.disk_block), 1u);
        }
    }

    ASSERT_TRUE(engine->armPause());
    ScopedTransferRelease release(engine);
    const int             block_size = static_cast<int>(cache_config_.seq_size_per_block);
    auto                  resource   = makeResource(cache_config_);
    auto                  tokens =
        makeTokenIds(/*offset=*/0, (logical_blocks + 1) * block_size, (logical_blocks + 1) * block_size, block_size);
    MallocInfo info{resource, tokens};
    info.reuse_cache         = true;
    info.enable_cache_lookup = true;
    const auto result        = manager_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(result.host_reuse_len, 0);
    EXPECT_EQ(result.disk_reuse_len, 0);
    ASSERT_NE(result.async_context, nullptr);

    const bool entered =
        engine->waitUntilEnteredFor(std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
    if (!entered) {
        engine->release();
    }
    ASSERT_TRUE(entered);
    EXPECT_FALSE(result.async_context->done());
    const auto found = cache->tree()->findNode(seed.cache_keys);
    ASSERT_EQ(found.size(), seed.cache_keys.size());
    size_t expected_load_descriptors = 0;
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto&  group_set   = cache->groupSets()[group_set_id];
        const size_t reuse_count = group_set->computeReuseBlockCount(static_cast<size_t>(logical_blocks));
        ASSERT_GT(reuse_count, 0u);
        ASSERT_LE(reuse_count, static_cast<size_t>(logical_blocks));
        expected_load_descriptors += reuse_count;
        const size_t reuse_begin = static_cast<size_t>(logical_blocks) - reuse_count;
        for (const size_t raw_group_id : group_set->groupIds()) {
            const int               group_id = static_cast<int>(raw_group_id);
            const BlockIndicesType& blocks   = resource->blocks(0, group_id);
            ASSERT_EQ(blocks.size(), static_cast<size_t>(logical_blocks + 1));
            for (size_t path_index = reuse_begin; path_index < static_cast<size_t>(logical_blocks); ++path_index) {
                ASSERT_FALSE(isNullBlockIdx(blocks[path_index]));
                ASSERT_TRUE(fillGroupBlockPayload(
                    manager_, cache_config_, group_id, blocks[path_index], path_index, /*poison=*/true));
            }
        }
    }
    EXPECT_GT(expected_load_descriptors, staging_block_count);

    engine->release();
    ASSERT_TRUE(waitForAsyncContextDoneFor(
        result.async_context, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)));
    result.async_context->waitDone();
    ASSERT_TRUE(result.async_context->done());
    ASSERT_TRUE(result.async_context->success()) << result.async_context->errorInfo().ToString();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    ASSERT_EQ(engine->submittedDescriptorCount(), disk_snapshot_begin + expected_load_descriptors);

    descriptors = engine->descriptors();
    for (size_t index = disk_snapshot_begin; index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::DISK);
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
    }
    ASSERT_TRUE(requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, resource, logical_blocks));
    ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
        manager_, *cache, cache_config_, seed.cache_keys, resource, logical_blocks));

    auto loaded_path = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(loaded_path.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto&  group_set   = cache->groupSets()[group_set_id];
        const size_t reuse_count = group_set->computeReuseBlockCount(static_cast<size_t>(logical_blocks));
        const size_t reuse_begin = static_cast<size_t>(logical_blocks) - reuse_count;
        for (size_t path_index = 0; path_index < static_cast<size_t>(logical_blocks); ++path_index) {
            const auto& state = (*loaded_path)[path_index][group_set_id];
            EXPECT_EQ(state.transfer_state, GroupSetTransferState::IDLE);
            if (path_index < reuse_begin) {
                EXPECT_EQ(state.getTopTier(), Tier::DISK);
                EXPECT_EQ(state.disk_block, disk_sources[path_index][group_set_id]);
            } else {
                EXPECT_EQ(state.getTopTier(), Tier::DEVICE);
                EXPECT_FALSE(group_set->diskPool()->isAllocated(disk_sources[path_index][group_set_id]));
            }
        }
    }

    manager_->free(FreeInfo{resource, tokens});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

}  // namespace rtp_llm::test
