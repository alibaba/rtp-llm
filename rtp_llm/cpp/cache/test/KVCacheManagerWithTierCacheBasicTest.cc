#include "rtp_llm/cpp/cache/test/KVCacheManagerWithTierCacheTestBase.h"

namespace rtp_llm::test {
using namespace tier_cache_test_detail;

TEST_P(KVCacheManagerWithTierCacheTest, DSV4DevicePrefixHitKeepsLowerTiersUntouched) {
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    ASSERT_NE(manager_, nullptr);
    auto cache = manager_->blockTreeCache();

    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       maybe_seed     = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/3);
    ASSERT_TRUE(maybe_seed.has_value());
    auto seed = std::move(*maybe_seed);
    ASSERT_EQ(seed.cache_keys.size(), 3u);
    ASSERT_EQ(seed.blocks_by_group.size(), static_cast<size_t>(kDsv4GroupCount));
    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    const auto lower_after_seed = snapshotLowerPools(*cache, GetParam());
    expectPoolSnapshotsEq(initial_lower, lower_after_seed);
    ASSERT_EQ(transfer_engine_->submittedDescriptorCount(), 0u);

    auto       resource  = makeResource(cache_config_);
    auto       token_ids = makeTokenIds(/*offset=*/0,
                                  /*seq_len=*/3 * static_cast<int>(cache_config_.seq_size_per_block),
                                  /*max_seq_len=*/3 * static_cast<int>(cache_config_.seq_size_per_block),
                                  static_cast<int>(cache_config_.seq_size_per_block));
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_cache_lookup = true;
    const auto result               = manager_->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 2 * static_cast<int>(cache_config_.seq_size_per_block));
    EXPECT_EQ(result.host_reuse_len, 0);
    EXPECT_EQ(result.disk_reuse_len, 0);
    EXPECT_EQ(result.async_context, nullptr);

    for (int group_id = 0; group_id < cache_config_.groupNums(); ++group_id) {
        const auto& blocks = resource->blocks(0, group_id);
        ASSERT_EQ(blocks.size(), 3u) << "group=" << group_id;
        if (!isReusableGroup(cache_config_, group_id)) {
            EXPECT_EQ(cache_config_.tagForGroup(static_cast<size_t>(group_id)), "hca_state");
            EXPECT_FALSE(isNullBlockIdx(blocks.back()));
            continue;
        }
        if (isFullGroup(cache_config_, group_id)) {
            EXPECT_EQ(blocks[0], seed.blocks_by_group[static_cast<size_t>(group_id)][0]);
            EXPECT_EQ(blocks[1], seed.blocks_by_group[static_cast<size_t>(group_id)][1]);
        } else {
            EXPECT_EQ(blocks[1], seed.blocks_by_group[static_cast<size_t>(group_id)][1]);
        }
    }

    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    expectPoolSnapshotsEq(lower_after_seed, snapshotLowerPools(*cache, GetParam()));
    EXPECT_EQ(transfer_engine_->submittedDescriptorCount(), 0u);

    manager_->free(FreeInfo{resource, token_ids});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4AllocatorPressureUsesDirectDropNotDemotion) {
    // Eleven physical blocks leave ten usable device blocks per independent
    // pool: three 3-block cached prefixes coexist, while the next 3-block
    // request must evict exactly part of the old cache.
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/11));
    ASSERT_NE(manager_, nullptr);
    auto cache = manager_->blockTreeCache();

    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());

    std::vector<SeededPrefix> seeds;
    for (int request = 0; request < 3; ++request) {
        auto maybe_seed =
            seedDevicePrefix(manager_, cache_config_, /*token_offset=*/request * 10000, /*cached_blocks=*/3);
        ASSERT_TRUE(maybe_seed.has_value()) << "request=" << request;
        seeds.push_back(std::move(*maybe_seed));
        ASSERT_EQ(seeds.back().cache_keys.size(), 3u);
        ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seeds.back().cache_keys));
    }
    ASSERT_NO_FATAL_FAILURE(expectAllTreeResourcesIdleAtDevice(*cache));
    const size_t device_resources_before = countTreeResourcesAtTier(*cache, Tier::DEVICE);
    ASSERT_GT(device_resources_before, 0u);
    const auto lower_before_pressure = snapshotLowerPools(*cache, GetParam());
    expectPoolSnapshotsEq(initial_lower, lower_before_pressure);
    ASSERT_EQ(transfer_engine_->submittedDescriptorCount(), 0u);

    const int  seq_len                = 3 * static_cast<int>(cache_config_.seq_size_per_block);
    const auto device_before_pressure = snapshotDevicePools(manager_);
    ASSERT_EQ(device_before_pressure.size(), static_cast<size_t>(kDsv4GroupCount));
    for (int group_id = 0; group_id < kDsv4GroupCount; ++group_id) {
        if (!isReusableGroup(cache_config_, group_id)) {
            continue;
        }
        EXPECT_EQ(device_before_pressure[static_cast<size_t>(group_id)].request_refs, 0u);
        EXPECT_GT(device_before_pressure[static_cast<size_t>(group_id)].cache_refs, 0u);
        ASSERT_LT(device_before_pressure[static_cast<size_t>(group_id)].free_blocks, 3u)
            << "group=" << group_id << " must require allocator-pressure eviction";
    }
    auto maybe_old_slots = snapshotDeviceSlots(*cache, seeds);
    ASSERT_TRUE(maybe_old_slots.has_value());
    const auto old_slots = std::move(*maybe_old_slots);
    ASSERT_FALSE(old_slots.empty());

    auto resource = makeResource(cache_config_);
    auto token_ids =
        makeTokenIds(/*offset=*/30000, seq_len, seq_len, static_cast<int>(cache_config_.seq_size_per_block));
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_cache_lookup = false;
    const auto result               = manager_->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.async_context, nullptr);

    const size_t device_resources_after = countTreeResourcesAtTier(*cache, Tier::DEVICE);
    EXPECT_LT(device_resources_after, device_resources_before);
    ASSERT_NO_FATAL_FAILURE(expectAllTreeResourcesIdleAtDevice(*cache));
    expectPoolSnapshotsEq(lower_before_pressure, snapshotLowerPools(*cache, GetParam()));
    EXPECT_EQ(transfer_engine_->submittedDescriptorCount(), 0u);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    const auto device_after_pressure = snapshotDevicePools(manager_);
    ASSERT_EQ(device_after_pressure.size(), device_before_pressure.size());
    for (int group_id = 0; group_id < kDsv4GroupCount; ++group_id) {
        size_t allocated_for_request = 0;
        for (const auto block : resource->blocks(0, group_id)) {
            allocated_for_request += !isNullBlockIdx(block);
        }
        EXPECT_EQ(device_after_pressure[static_cast<size_t>(group_id)].request_refs, allocated_for_request)
            << "group=" << group_id;
        if (isReusableGroup(cache_config_, group_id)) {
            EXPECT_EQ(allocated_for_request, 3u) << "linear_step=1 group=" << group_id;
            EXPECT_LT(device_after_pressure[static_cast<size_t>(group_id)].cache_refs,
                      device_before_pressure[static_cast<size_t>(group_id)].cache_refs)
                << "group=" << group_id;
        }
    }

    std::vector<size_t> evicted_per_group_set(cache->groupSets().size(), 0);
    std::vector<size_t> survivors_per_group_set(cache->groupSets().size(), 0);
    for (const auto& snapshot : old_slots) {
        const auto found = cache->tree()->findNode(snapshot.path_keys);
        if (found.size() != snapshot.path_keys.size()) {
            ++evicted_per_group_set[snapshot.group_set_id];
            continue;
        }
        const auto& slot = found.back()->group_set_resources[snapshot.group_set_id];
        EXPECT_EQ(slot.transfer_state, GroupSetTransferState::IDLE);
        if (!slot.hasTier(Tier::DEVICE)) {
            EXPECT_TRUE(slot.is_empty()) << "direct drop must not create a lower-tier copy";
            ++evicted_per_group_set[snapshot.group_set_id];
            continue;
        }
        EXPECT_EQ(slot.getTopTier(), Tier::DEVICE);
        EXPECT_EQ(slot.device_blocks, snapshot.device_blocks)
            << "surviving slot must retain identity, group_set=" << snapshot.group_set_id;
        ++survivors_per_group_set[snapshot.group_set_id];
    }
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        EXPECT_GT(evicted_per_group_set[group_set_id], 0u) << "group_set=" << group_set_id;
        EXPECT_GT(survivors_per_group_set[group_set_id], 0u) << "group_set=" << group_set_id;
    }

    manager_->free(FreeInfo{resource, token_ids});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4ReuseCacheFalsePressureDoesNotDisturbInFlightLoad) {
    if (GetParam() != TierLayout::HOST_ONLY) {
        GTEST_SKIP() << "the in-flight device-pressure case isolates a HostOnly source";
    }

    // Five physical blocks expose four usable blocks per independent device pool.
    // The first request consumes all four while its 3-block prefix is loading:
    // three load targets plus one incremental/tail block.
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/5));
    ASSERT_NE(manager_, nullptr);
    auto cache = manager_->blockTreeCache();

    auto pausable_engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_engine);
    transfer_engine_.reset();

    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    ASSERT_EQ(initial_device.size(), static_cast<size_t>(kDsv4GroupCount));
    for (const auto& pool : initial_device) {
        ASSERT_EQ(pool.free_blocks, 4u) << pool.pool->poolName();
    }

    auto maybe_seed = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/3);
    ASSERT_TRUE(maybe_seed.has_value());
    auto seed = std::move(*maybe_seed);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));
    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    for (const auto& group_set : cache->groupSets()) {
        device_pools.insert(device_pools.end(), group_set->devicePools().begin(), group_set->devicePools().end());
    }
    ASSERT_NO_FATAL_FAILURE(
        moveAllPathResourcesToTier(cache, seed, Tier::DEVICE, Tier::HOST, device_pools, pausable_engine));

    auto maybe_host = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_host.has_value());

    ASSERT_TRUE(pausable_engine->armPause());
    ScopedTransferRelease first_load_release(pausable_engine);
    const int             seq_size_per_block = static_cast<int>(cache_config_.seq_size_per_block);
    auto                  first_resource     = makeResource(cache_config_);
    auto                  first_token_ids =
        makeTokenIds(/*offset=*/0, 4 * seq_size_per_block, 4 * seq_size_per_block, seq_size_per_block);
    MallocInfo first_info{first_resource, first_token_ids};
    first_info.reuse_cache                = true;
    first_info.enable_cache_lookup        = true;
    const size_t first_load_submits_begin = pausable_engine->submittedDescriptorCount();
    const auto   first_result             = manager_->malloc(first_info);
    ASSERT_TRUE(first_result.success);
    EXPECT_EQ(first_result.reuse_len, 0);
    EXPECT_EQ(first_result.host_reuse_len, 0);
    EXPECT_EQ(first_result.disk_reuse_len, 0);
    ASSERT_NE(first_result.async_context, nullptr);
    const bool first_load_entered = pausable_engine->waitUntilEnteredFor(
        std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
    if (!first_load_entered) {
        pausable_engine->release();
    }
    ASSERT_TRUE(first_load_entered);
    EXPECT_FALSE(first_result.async_context->done());
    const int pending_before_second = BlockTreeCacheTestPeer::pendingTasksForTest(*cache);
    EXPECT_GT(pending_before_second, 0);
    EXPECT_EQ(countTreeResourcesAtTier(*cache, Tier::DEVICE), 0u);

    PathResourcesSnapshot loading_snapshot;
    ASSERT_NO_FATAL_FAILURE(expectPausedHostLoadState(cache, seed, *maybe_host, first_resource, &loading_snapshot));

    const auto device_before_second = snapshotDevicePools(manager_);
    const auto lower_before_second  = snapshotLowerPools(*cache, GetParam());
    for (int group_id = 0; group_id < cache_config_.groupNums(); ++group_id) {
        const auto& pool = device_before_second[static_cast<size_t>(group_id)];
        if (isReusableGroup(cache_config_, group_id) && isFullGroup(cache_config_, group_id)) {
            EXPECT_EQ(pool.free_blocks, 0u) << pool.pool->poolName();
            EXPECT_EQ(pool.used_blocks, 4u) << pool.pool->poolName();
        }
        EXPECT_GT(pool.used_blocks, 0u) << pool.pool->poolName();
    }
    ASSERT_EQ(cache_config_.typeForGroup(/*group_id=*/0), CacheGroupType::SWA);
    ASSERT_GE(device_before_second[0].free_blocks, 1u) << "the second request must first allocate in group0 SWA";
    ASSERT_EQ(cache_config_.typeForGroup(/*group_id=*/1), CacheGroupType::FULL);
    ASSERT_EQ(device_before_second[1].free_blocks, 0u) << "the second request must fail after reaching group1 FULL";
    const size_t submits_before_second = pausable_engine->submittedDescriptorCount();

    auto second_resource  = makeResource(cache_config_);
    auto second_token_ids = makeTokenIds(/*offset=*/10000, seq_size_per_block, seq_size_per_block, seq_size_per_block);
    MallocInfo second_info{second_resource, second_token_ids};
    second_info.reuse_cache         = false;
    second_info.enable_cache_lookup = true;
    const auto second_result        = manager_->malloc(second_info);
    EXPECT_FALSE(second_result.success);
    EXPECT_EQ(second_result.async_context, nullptr);
    EXPECT_EQ(second_result.reuse_len, 0);
    EXPECT_EQ(second_result.host_reuse_len, 0);
    EXPECT_EQ(second_result.disk_reuse_len, 0);
    for (int group_id = 0; group_id < cache_config_.groupNums(); ++group_id) {
        EXPECT_EQ(second_resource->blocksNum(0, group_id), 0u) << "group=" << group_id;
    }

    expectPoolSnapshotsEq(device_before_second, snapshotDevicePools(manager_));
    expectPoolSnapshotsEq(lower_before_second, snapshotLowerPools(*cache, GetParam()));
    EXPECT_EQ(pausable_engine->submittedDescriptorCount(), submits_before_second);
    EXPECT_FALSE(first_result.async_context->done());
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), pending_before_second)
        << "reuse_cache=false pressure must not enqueue or settle the first load";

    ASSERT_NO_FATAL_FAILURE(
        expectPausedHostLoadState(cache, seed, *maybe_host, first_resource, nullptr, &loading_snapshot));

    pausable_engine->release();
    ASSERT_TRUE(waitForAsyncContextDoneFor(first_result.async_context,
                                           std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << pausable_engine->submittedDescriptorCount();
    first_result.async_context->waitDone();
    ASSERT_TRUE(first_result.async_context->done());
    ASSERT_TRUE(first_result.async_context->success()) << first_result.async_context->errorInfo().ToString();
    ASSERT_TRUE(
        waitForPendingTasksDoneFor(*cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
        << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
        << " submits=" << pausable_engine->submittedDescriptorCount();
    const auto descriptors_after_first_load = pausable_engine->descriptors();
    ASSERT_GT(descriptors_after_first_load.size(), first_load_submits_begin);
    for (size_t index = first_load_submits_begin; index < descriptors_after_first_load.size(); ++index) {
        const auto& descriptor = descriptors_after_first_load[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
        EXPECT_TRUE(pathSnapshotContainsBlock(
            *maybe_host, descriptor.group_set_id, Tier::HOST, descriptor.singleBlockAt(Tier::HOST)));
        const int   group_id = static_cast<int>(cache->groupSets()[descriptor.group_set_id]->groupIds().front());
        const auto& blocks   = first_resource->blocks(0, group_id);
        EXPECT_NE(std::find(blocks.begin(), blocks.end(), descriptor.singleBlockAt(Tier::DEVICE)), blocks.end());
    }

    ASSERT_NO_FATAL_FAILURE(expectHostLoadSettledAtDevice(cache, seed, *maybe_host, first_resource));
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, first_resource, /*logical_reuse_blocks=*/3));
    ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
        manager_, *cache, cache_config_, seed.cache_keys, first_resource, /*logical_reuse_blocks=*/3));

    manager_->free(FreeInfo{first_resource, first_token_ids});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4ReuseCacheFalseStillMatchesResidentPrefix) {
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    auto       cache          = manager_->blockTreeCache();
    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       seed_opt       = seedDevicePrefix(manager_, cache_config_, 0, 1);
    ASSERT_TRUE(seed_opt.has_value());
    auto       seed     = std::move(*seed_opt);
    auto       resource = makeResource(cache_config_);
    auto       tokens   = makeTokenIds(0,
                               2 * cache_config_.seq_size_per_block,
                               2 * cache_config_.seq_size_per_block,
                               cache_config_.seq_size_per_block);
    MallocInfo info{resource, tokens};
    info.reuse_cache         = false;
    info.enable_cache_lookup = true;
    const auto result        = manager_->malloc(info);
    ASSERT_TRUE(result.success);
    // reuse_cache controls how the per-group allocator preserves and extends
    // blocks. Device-tree lookup is independently gated by
    // enable_device_cache, so a resident prefix remains reusable here.
    EXPECT_EQ(result.reuse_len, static_cast<int>(cache_config_.seq_size_per_block));
    EXPECT_EQ(result.host_reuse_len, 0);
    EXPECT_EQ(result.disk_reuse_len, 0);
    EXPECT_EQ(result.async_context, nullptr);
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, resource, /*logical_reuse_blocks=*/1));
    manager_->free(FreeInfo{resource, tokens});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4DuplicateInsertIsIdempotent) {
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    auto       cache          = manager_->blockTreeCache();
    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       seed_opt       = seedDevicePrefix(manager_, cache_config_, 0, 1);
    ASSERT_TRUE(seed_opt.has_value());
    auto       seed     = std::move(*seed_opt);
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
    ASSERT_EQ(result.async_context, nullptr);

    // The first insert may legitimately publish the request's longer prefix.
    // Idempotence is the second insertion of that exact resource/key mapping.
    manager_->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
    const auto after_first_stats  = cache->getStats();
    const auto after_first_device = snapshotDevicePools(manager_);
    const auto after_first_lower  = snapshotLowerPools(*cache, GetParam());
    manager_->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
    const auto after_second_stats = cache->getStats();
    EXPECT_EQ(after_second_stats.tree_node_count, after_first_stats.tree_node_count);
    EXPECT_EQ(after_second_stats.device_heap_total_size, after_first_stats.device_heap_total_size);
    EXPECT_EQ(after_second_stats.host_heap_total_size, after_first_stats.host_heap_total_size);
    EXPECT_EQ(after_second_stats.disk_heap_total_size, after_first_stats.disk_heap_total_size);
    expectPoolSnapshotsEq(after_first_device, snapshotDevicePools(manager_));
    expectPoolSnapshotsEq(after_first_lower, snapshotLowerPools(*cache, GetParam()));
    manager_->free(FreeInfo{resource, tokens});
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
    (void)seed;
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4LowerTierMatchPublishesAsyncContext) {
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    auto       cache          = manager_->blockTreeCache();
    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       seed_opt       = seedDevicePrefix(manager_, cache_config_, 0, 1);
    ASSERT_TRUE(seed_opt.has_value());
    auto seed = std::move(*seed_opt);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));
    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    for (const auto& gs : cache->groupSets()) {
        device_pools.insert(device_pools.end(), gs->devicePools().begin(), gs->devicePools().end());
    }
    const auto ratio = oneUsedBlockWatermarkRatio(device_pools);
    ASSERT_TRUE(ratio.has_value());
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    auto lower = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(lower.has_value());
    ASSERT_EQ(lower->size(), 1u);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& state = (*lower)[0][group_set_id];
        EXPECT_EQ(state.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(state.getTopTier(), Tier::HOST);
        EXPECT_EQ(cache->groupSets()[group_set_id]->hostPool()->refCount(state.host_block), 1u);
    }

    const size_t batches_before_load     = transfer_engine_->submittedBatchCount();
    const size_t descriptors_before_load = transfer_engine_->submittedDescriptorCount();
    auto         resource                = makeResource(cache_config_);
    auto         tokens              = makeTokenIds(0,
                               2 * cache_config_.seq_size_per_block,
                               2 * cache_config_.seq_size_per_block,
                               cache_config_.seq_size_per_block);
    MallocInfo   info{resource, tokens};
    info.reuse_cache         = true;
    info.enable_cache_lookup = true;
    auto result              = manager_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(result.host_reuse_len, 0);
    EXPECT_EQ(result.disk_reuse_len, 0);
    ASSERT_NE(result.async_context, nullptr);
    result.async_context->waitDone();
    ASSERT_TRUE(result.async_context->success()) << result.async_context->errorInfo().ToString();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(transfer_engine_->submittedBatchCount(), batches_before_load + cache->groupSets().size());
    EXPECT_EQ(transfer_engine_->submittedDescriptorCount(), descriptors_before_load + cache->groupSets().size());
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, resource, /*logical_reuse_blocks=*/1));
    ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
        manager_, *cache, cache_config_, seed.cache_keys, resource, /*logical_reuse_blocks=*/1));
    manager_->free(FreeInfo{resource, tokens});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
    (void)seed;
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4BatchCommonLowerHitSharesOneLoadedTarget) {
    if (GetParam() != TierLayout::HOST_ONLY) {
        GTEST_SKIP() << "batch common-prefix load coverage isolates a HostOnly source";
    }
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
    auto cache  = manager_->blockTreeCache();
    auto engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, engine);
    transfer_engine_.reset();

    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    auto       seed_opt       = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/1);
    ASSERT_TRUE(seed_opt.has_value());
    auto seed = std::move(*seed_opt);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    for (const auto& group_set : cache->groupSets()) {
        ASSERT_EQ(group_set->devicePools().size(), 1u);
        device_pools.push_back(group_set->devicePools().front());
    }
    const auto device_ratio = oneUsedBlockWatermarkRatio(device_pools);
    ASSERT_TRUE(device_ratio.has_value());
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto host_path = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(host_path.has_value());
    ASSERT_EQ(host_path->size(), 1u);
    std::vector<BlockIdxType> host_sources(cache->groupSets().size(), NULL_BLOCK_IDX);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& state     = (*host_path)[0][group_set_id];
        EXPECT_EQ(state.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(state.getTopTier(), Tier::HOST);
        host_sources[group_set_id] = state.host_block;
        EXPECT_EQ(group_set->hostPool()->refCount(state.host_block), 1u);
    }

    const size_t descriptors_before_load = engine->submittedDescriptorCount();
    ASSERT_TRUE(engine->armPause());
    ScopedTransferRelease release(engine);

    constexpr int batch_size = 2;
    const int     block_size = static_cast<int>(cache_config_.seq_size_per_block);
    auto          resource   = makeResource(cache_config_, batch_size);
    auto          tokens     = makeBatchedTokenIdsWithCommonPrefix(
        /*offset=*/0, batch_size, /*common_seq_len=*/block_size, /*seq_len=*/2 * block_size, block_size);
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

    auto loading_path = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(loading_path.has_value());
    ASSERT_EQ(loading_path->size(), 1u);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& state     = (*loading_path)[0][group_set_id];
        EXPECT_EQ(state.transfer_state, GroupSetTransferState::LOADING);
        EXPECT_EQ(state.host_block, host_sources[group_set_id]);
        EXPECT_EQ(group_set->hostPool()->refCount(state.host_block), 2u);
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockRefType::REQUEST), 1u);

        ASSERT_EQ(group_set->groupIds().size(), 1u);
        const int   group_id = static_cast<int>(group_set->groupIds().front());
        const auto& batch0   = resource->blocks(0, group_id);
        const auto& batch1   = resource->blocks(1, group_id);
        ASSERT_EQ(batch0.size(), 2u);
        ASSERT_EQ(batch1.size(), 2u);
        ASSERT_FALSE(isNullBlockIdx(batch0[0]));
        ASSERT_FALSE(isNullBlockIdx(batch0[1]));
        ASSERT_FALSE(isNullBlockIdx(batch1[1]));
        EXPECT_EQ(batch0[0], batch1[0]);
        EXPECT_NE(batch0[1], batch1[1]);
        EXPECT_EQ(group_set->devicePools().front()->refCount(batch0[0]), 3u);
        EXPECT_EQ(group_set->devicePools().front()->referencedBlocksNum(BlockRefType::REQUEST), 3u);
        ASSERT_TRUE(
            fillGroupBlockPayload(manager_, cache_config_, group_id, batch0[0], /*path_index=*/0, /*poison=*/true));
    }

    // The non-reusable typed region is not part of a transfer GroupSet, but
    // common-prefix allocation still shares its common block across the batch.
    for (int group_id = 0; group_id < cache_config_.groupNums(); ++group_id) {
        const auto& batch0 = resource->blocks(0, group_id);
        const auto& batch1 = resource->blocks(1, group_id);
        ASSERT_EQ(batch0.size(), 2u);
        ASSERT_EQ(batch1.size(), 2u);
        EXPECT_EQ(batch0[0], batch1[0]) << "group=" << group_id;
        EXPECT_NE(batch0[1], batch1[1]) << "group=" << group_id;
    }

    engine->release();
    result.async_context->waitDone();
    ASSERT_TRUE(result.async_context->done());
    ASSERT_TRUE(result.async_context->success()) << result.async_context->errorInfo().ToString();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    EXPECT_EQ(engine->submittedDescriptorCount(), descriptors_before_load + cache->groupSets().size());

    const auto descriptors = engine->descriptors();
    ASSERT_EQ(descriptors.size(), engine->submittedDescriptorCount());
    for (size_t index = descriptors_before_load; index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.singleBlockAt(Tier::HOST), host_sources[descriptor.group_set_id]);
    }

    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
    ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
    ASSERT_TRUE(
        requestReusesExpectedPath(*cache, cache_config_, seed.cache_keys, resource, /*logical_reuse_blocks=*/1));
    ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
        manager_, *cache, cache_config_, seed.cache_keys, resource, /*logical_reuse_blocks=*/1));

    const auto found = cache->tree()->findNode(seed.cache_keys);
    ASSERT_EQ(found.size(), 1u);
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const int   group_id  = static_cast<int>(group_set->groupIds().front());
        const auto& tree      = found[0]->group_set_resources[group_set_id];
        const auto& batch0    = resource->blocks(0, group_id);
        const auto& batch1    = resource->blocks(1, group_id);
        ASSERT_EQ(tree.device_blocks.size(), 1u);
        EXPECT_EQ(batch0[0], tree.device_blocks[0]);
        EXPECT_EQ(batch1[0], tree.device_blocks[0]);
        EXPECT_TRUE(groupBlockPayloadMatches(manager_, cache_config_, group_id, batch1[0], /*path_index=*/0));
        EXPECT_EQ(group_set->devicePools().front()->refCount(tree.device_blocks[0]), 3u);
        EXPECT_EQ(group_set->devicePools().front()->referencedBlocksNum(BlockRefType::REQUEST), 3u);
        EXPECT_EQ(group_set->devicePools().front()->referencedBlocksNum(BlockRefType::BLOCK_CACHE), 1u);
        EXPECT_FALSE(group_set->hostPool()->isAllocated(host_sources[group_set_id]));
    }

    manager_->free(FreeInfo{resource, tokens});
    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

}  // namespace rtp_llm::test
