#include "rtp_llm/cpp/cache/test/KVCacheManagerWithTierCacheTestBase.h"

namespace rtp_llm::test {
using namespace tier_cache_test_detail;

TEST_P(KVCacheManagerWithTierCacheTest, DSV4LowerTierLoadFailureReleasesRefsAndCanRetry) {
    const auto source = GetParam() == TierLayout::HOST_ONLY ? LoadFailureSource::HOST : LoadFailureSource::DISK;
    ASSERT_NO_FATAL_FAILURE(runLowerTierLoadFailureScenario(source));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4ConcurrentLowerHitJoinsLoading) {
    if (GetParam() != TierLayout::HOST_ONLY) {
        GTEST_SKIP() << "manager join coverage uses the HostOnly lower source";
    }
    ASSERT_NO_FATAL_FAILURE(runConcurrentLowerHitJoinScenario(Tier::HOST, /*transfer_success=*/true));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4ConcurrentLowerHitFailureSettlesAllJoinersAndCanRetry) {
    if (GetParam() != TierLayout::HOST_ONLY) {
        GTEST_SKIP() << "manager join failure coverage uses the HostOnly lower source";
    }
    ASSERT_NO_FATAL_FAILURE(runConcurrentLowerHitJoinScenario(Tier::HOST, /*transfer_success=*/false));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4ConcurrentDiskLowerHitJoinsLoading) {
    if (GetParam() != TierLayout::HOST_DISK) {
        GTEST_SKIP() << "DISK join coverage requires the HostDisk layout";
    }
    ASSERT_NO_FATAL_FAILURE(runConcurrentLowerHitJoinScenario(Tier::DISK, /*transfer_success=*/true));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4ConcurrentDiskLowerHitFailureSettlesAllJoinersAndCanRetry) {
    if (GetParam() != TierLayout::HOST_DISK) {
        GTEST_SKIP() << "DISK join failure coverage requires the HostDisk layout";
    }
    ASSERT_NO_FATAL_FAILURE(runConcurrentLowerHitJoinScenario(Tier::DISK, /*transfer_success=*/false));
}

TEST_P(KVCacheManagerWithTierCacheTest, DSV4LowerHitOuterIncrFailureAbortsBeforeCommit) {
    if (GetParam() != TierLayout::HOST_ONLY) {
        GTEST_SKIP() << "pre-commit allocation rollback isolates a HostOnly source";
    }

    // Two physical blocks expose exactly one usable block in each independent
    // device pool. The lower hit can materialize its common target, but the
    // second logical block cannot be allocated before LoadTicket::commit().
    ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/2));
    ASSERT_NE(manager_, nullptr);
    auto cache = manager_->blockTreeCache();

    auto recording_engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, recording_engine);
    transfer_engine_.reset();

    const auto initial_device = snapshotDevicePools(manager_);
    const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
    ASSERT_EQ(initial_device.size(), static_cast<size_t>(kDsv4GroupCount));
    for (const auto& pool : initial_device) {
        ASSERT_EQ(pool.free_blocks, 1u) << pool.pool->poolName();
    }

    auto maybe_seed = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/1);
    ASSERT_TRUE(maybe_seed.has_value());
    auto seed = std::move(*maybe_seed);
    ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));
    ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));

    std::vector<std::shared_ptr<IBlockPool>> device_pools;
    for (const auto& group_set : cache->groupSets()) {
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
        EXPECT_EQ(group_set->hostPool()->refCount(resource.host_block), 1u);
    }

    const auto device_before = snapshotDevicePools(manager_);
    const auto lower_before  = snapshotLowerPools(*cache, GetParam());
    const auto stats_before  = cache->getStats();
    for (const auto& pool : device_before) {
        ASSERT_EQ(pool.free_blocks, 1u) << pool.pool->poolName();
        ASSERT_EQ(pool.used_blocks, 0u) << pool.pool->poolName();
    }
    const size_t submits_before = recording_engine->submittedDescriptorCount();

    const int     seq_size_per_block = static_cast<int>(cache_config_.seq_size_per_block);
    constexpr int batch_size         = 2;
    auto          failed_resource    = makeResource(cache_config_, batch_size);
    auto          failed_token_ids   = makeBatchedTokenIdsWithCommonPrefix(/*offset=*/0,
                                                                batch_size,
                                                                /*common_seq_len=*/seq_size_per_block,
                                                                /*seq_len=*/2 * seq_size_per_block,
                                                                seq_size_per_block);
    ASSERT_EQ(failed_resource->batchSize(), batch_size);
    ASSERT_EQ(failed_token_ids->batchSize(), batch_size);
    ASSERT_EQ(failed_token_ids->commonSeqLength(), seq_size_per_block);
    ASSERT_EQ(failed_token_ids->seqLength(), 2 * seq_size_per_block);

    const auto allocator_groups = manager_->allocator_->cacheGroups();
    ASSERT_EQ(allocator_groups.size(), static_cast<size_t>(kDsv4GroupCount));
    ASSERT_NE(allocator_groups[0], nullptr);
    bool   outer_incr_evict_observed = false;
    size_t outer_incr_evict_calls    = 0;
    allocator_groups[0]->setEvictCallback([&](size_t need_blocks) {
        outer_incr_evict_observed = true;
        ++outer_incr_evict_calls;
        EXPECT_EQ(need_blocks, 1u);

        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            for (int group_id = 0; group_id < cache_config_.groupNums(); ++group_id) {
                EXPECT_EQ(failed_resource->blocksNum(batch_id, group_id), 1)
                    << "batch=" << batch_id << " group=" << group_id;
                const auto& blocks = failed_resource->blocks(batch_id, group_id);
                if (blocks.size() == 1u) {
                    EXPECT_FALSE(isNullBlockIdx(blocks[0])) << "batch=" << batch_id << " group=" << group_id;
                }
            }
        }
        for (const auto& group : allocator_groups) {
            const auto pool = group->blockPool();
            EXPECT_EQ(pool->usedBlocksNum(), 1u) << pool->poolName();
            EXPECT_EQ(pool->freeBlocksNum(), 0u) << pool->poolName();
            EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::REQUEST), 1u) << pool->poolName();
            EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::BLOCK_CACHE), 0u) << pool->poolName();
            EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::EVICTION), 0u) << pool->poolName();
        }

        auto pending_path = snapshotPathResources(*cache, seed.cache_keys);
        EXPECT_TRUE(pending_path.has_value());
        if (pending_path.has_value() && pending_path->size() == 1u) {
            for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
                const auto& group_set = cache->groupSets()[group_set_id];
                const auto& resource  = (*pending_path)[0][group_set_id];
                EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOAD_PENDING) << "group_set=" << group_set_id;
                EXPECT_TRUE(resource.hasTier(Tier::HOST)) << "group_set=" << group_set_id;
                EXPECT_FALSE(resource.hasTier(Tier::DEVICE)) << "group_set=" << group_set_id;
                EXPECT_EQ(resource.host_block, host_sources[group_set_id]) << "group_set=" << group_set_id;
                EXPECT_EQ(group_set->hostPool()->refCount(resource.host_block), 2u) << "group_set=" << group_set_id;
                EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockRefType::REQUEST), 1u)
                    << "group_set=" << group_set_id;
                EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockRefType::BLOCK_CACHE), 1u)
                    << "group_set=" << group_set_id;
                EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockRefType::EVICTION), 0u)
                    << "group_set=" << group_set_id;
            }
        }

        const int reclaimed = cache->evictForGroup(/*group_id=*/0, need_blocks);
        EXPECT_EQ(reclaimed, 0);
        return reclaimed > 0 ? static_cast<size_t>(reclaimed) : 0u;
    });

    MallocInfo failed_info{failed_resource, failed_token_ids};
    failed_info.reuse_cache         = true;
    failed_info.enable_cache_lookup = true;
    const auto failed_result        = manager_->malloc(failed_info);
    allocator_groups[0]->setEvictCallback([cache](size_t need_blocks) {
        const int reclaimed = cache->evictForGroup(/*group_id=*/0, need_blocks);
        return reclaimed > 0 ? static_cast<size_t>(reclaimed) : 0u;
    });
    EXPECT_TRUE(outer_incr_evict_observed);
    EXPECT_EQ(outer_incr_evict_calls, 1u);
    EXPECT_FALSE(failed_result.success);
    EXPECT_EQ(failed_result.async_context, nullptr);
    EXPECT_EQ(failed_result.reuse_len, 0);
    EXPECT_EQ(failed_result.host_reuse_len, 0);
    EXPECT_EQ(failed_result.disk_reuse_len, 0);
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        for (int group_id = 0; group_id < cache_config_.groupNums(); ++group_id) {
            EXPECT_EQ(failed_resource->blocksNum(batch_id, group_id), 0u)
                << "batch=" << batch_id << " group=" << group_id;
        }
    }

    EXPECT_EQ(recording_engine->submittedDescriptorCount(), submits_before)
        << "pre-commit failure must not submit a lower-tier copy";
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    expectPoolSnapshotsEq(device_before, snapshotDevicePools(manager_));
    expectPoolSnapshotsEq(lower_before, snapshotLowerPools(*cache, GetParam()));

    auto maybe_after = snapshotPathResources(*cache, seed.cache_keys);
    ASSERT_TRUE(maybe_after.has_value());
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        const auto& resource  = (*maybe_after)[0][group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        ASSERT_TRUE(resource.hasTier(Tier::HOST));
        EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
        EXPECT_EQ(resource.host_block, host_sources[group_set_id]);
        EXPECT_EQ(group_set->hostPool()->refCount(resource.host_block), 1u);
        EXPECT_EQ(group_set->hostPool()->referencedBlocksNum(BlockRefType::REQUEST), 0u);
    }
    const auto stats_after = cache->getStats();
    EXPECT_EQ(stats_after.tree_node_count, stats_before.tree_node_count);
    EXPECT_EQ(stats_after.device_heap_total_size, stats_before.device_heap_total_size);
    EXPECT_EQ(stats_after.host_heap_total_size, stats_before.host_heap_total_size);
    EXPECT_EQ(stats_after.disk_heap_total_size, stats_before.disk_heap_total_size);

    ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
}

}  // namespace rtp_llm::test
