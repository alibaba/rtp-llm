#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/smoke/CacheSmokeTestUtils.h"

namespace rtp_llm::test {

class CacheLifecycleSmokeTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        initCacheSmokeRuntime();
    }
};

TEST_F(CacheLifecycleSmokeTest, PoolInitAllocateCopyRecycle) {
    const auto     config = makeCacheSmokeConfig();
    KVCacheManager manager(config);
    ASSERT_TRUE(manager.init());

    const size_t baseline_free = manager.freeBlocksNum();
    EXPECT_EQ(manager.totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(baseline_free, config.block_num - 1);

    const auto layout = manager.allLayerCacheBase();
    ASSERT_EQ(layout.groups().size(), 1u);
    const auto& group_layout = layout.group("default");
    ASSERT_EQ(group_layout.size(), config.layer_num);
    for (const auto& layer : group_layout.layers()) {
        EXPECT_TRUE(layer.kv_addr.defined());
        EXPECT_GT(layer.kv_addr.numel(), 0);
        EXPECT_TRUE(layer.kv_scale_addr.defined());
        EXPECT_GT(layer.kv_scale_addr.numel(), 0);
    }

    auto       token_ids = makeCacheSmokeTokenIds(makeTokenRange(/*begin=*/1, /*count=*/8));
    auto       resource  = makeCacheSmokeResource(config, makeCacheKeys(/*begin=*/100, /*count=*/2));
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(manager.malloc(malloc_info).success);
    ASSERT_EQ(resource->blocksNum(0, 0), 2);

    const auto released_blocks = resource->blocks(0, 0);
    ASSERT_EQ(std::set<BlockIdxType>(released_blocks.begin(), released_blocks.end()).size(), 2u);
    const auto src_block = released_blocks[0];
    const auto dst_block = released_blocks[1];
    for (int layer_id = 0; layer_id < static_cast<int>(config.layer_num); ++layer_id) {
        fillBlockInfos(managerBlockInfos(manager, layer_id, src_block), static_cast<uint8_t>(41 + layer_id * 19));
        fillBlockInfos(managerBlockInfos(manager, layer_id, dst_block), /*seed=*/0);
    }

    manager.blockCopy(src_block, dst_block);
    synchronizeCacheSmokeDevice();
    for (int layer_id = 0; layer_id < static_cast<int>(config.layer_num); ++layer_id) {
        const auto src_infos = managerBlockInfos(manager, layer_id, src_block);
        const auto dst_infos = managerBlockInfos(manager, layer_id, dst_block);
        ASSERT_EQ(src_infos.size(), dst_infos.size());
        for (size_t buffer_id = 0; buffer_id < src_infos.size(); ++buffer_id) {
            EXPECT_EQ(readBlockInfoBytes(src_infos[buffer_id]), readBlockInfoBytes(dst_infos[buffer_id]));
        }
    }

    manager.free(FreeInfo{resource, token_ids});
    EXPECT_EQ(manager.freeBlocksNum(), baseline_free);
    EXPECT_EQ(resource->curBlocksNum(), 0);

    auto       next_tokens   = makeCacheSmokeTokenIds(makeTokenRange(/*begin=*/1000, /*count=*/4));
    auto       next_resource = makeCacheSmokeResource(config, makeCacheKeys(/*begin=*/1000, /*count=*/1));
    MallocInfo next_malloc{next_resource, next_tokens};
    next_malloc.reuse_cache         = false;
    next_malloc.enable_device_cache = false;
    ASSERT_TRUE(manager.malloc(next_malloc).success);
    ASSERT_EQ(next_resource->blocksNum(0, 0), 1);
    EXPECT_NE(std::find(released_blocks.begin(), released_blocks.end(), next_resource->blocks(0, 0)[0]),
              released_blocks.end());
    manager.free(FreeInfo{next_resource, next_tokens});
    EXPECT_EQ(manager.freeBlocksNum(), baseline_free);
}

TEST_F(CacheLifecycleSmokeTest, PrefixReusePreservesBlockIdentityAndPayload) {
    const auto config    = makeCacheSmokeConfig();
    auto       allocator = makeCacheSmokeAllocator(config, /*enable_prefix_cache=*/true);
    ASSERT_TRUE(allocator->init());
    const size_t baseline_free = allocator->freeBlocksNum();

    const auto seed_keys     = makeCacheKeys(/*begin=*/200, /*count=*/4);
    auto       seed_tokens   = makeCacheSmokeTokenIds(makeTokenRange(/*begin=*/1, /*count=*/16));
    auto       seed_resource = makeCacheSmokeResource(config, seed_keys);
    ASSERT_TRUE(allocateCacheSmokeResource(allocator, seed_resource, seed_tokens).success);
    fillAllocatorResource(*allocator, seed_resource->cacheResource(0), /*seed=*/17);

    const auto                                     seed_blocks = seed_resource->blocks(0, 0);
    std::vector<std::vector<std::vector<uint8_t>>> expected_payload(config.layer_num);
    for (int layer_id = 0; layer_id < static_cast<int>(config.layer_num); ++layer_id) {
        expected_payload[static_cast<size_t>(layer_id)].resize(seed_blocks.size());
        for (size_t block_pos = 0; block_pos < seed_blocks.size(); ++block_pos) {
            const auto infos = allocatorBlockInfos(*allocator, layer_id, seed_blocks[block_pos]);
            for (const auto& info : infos) {
                auto bytes = readBlockInfoBytes(info);
                expected_payload[static_cast<size_t>(layer_id)][block_pos].insert(
                    expected_payload[static_cast<size_t>(layer_id)][block_pos].end(), bytes.begin(), bytes.end());
            }
        }
    }

    allocator->insertIntoCache(InsertInfo{seed_resource, seed_tokens, /*is_resident=*/false});
    allocator->free(FreeInfo{seed_resource, seed_tokens});

    auto hit_keys = seed_keys;
    hit_keys.push_back(999);
    auto hit_tokens   = makeCacheSmokeTokenIds(makeTokenRange(/*begin=*/1, /*count=*/20));
    auto hit_resource = makeCacheSmokeResource(config, hit_keys);
    auto hit_result   = allocateCacheSmokeResource(allocator, hit_resource, hit_tokens, /*enable_device_cache=*/true);
    ASSERT_TRUE(hit_result.success);
    EXPECT_EQ(hit_result.reuse_len, 16);
    EXPECT_EQ(hit_resource->cacheResource(0).deviceReuseBlockNum(), seed_blocks.size());
    ASSERT_EQ(hit_resource->blocksNum(0, 0), 5);
    EXPECT_TRUE(std::equal(seed_blocks.begin(), seed_blocks.end(), hit_resource->blocks(0, 0).begin()));

    for (int layer_id = 0; layer_id < static_cast<int>(config.layer_num); ++layer_id) {
        for (size_t block_pos = 0; block_pos < seed_blocks.size(); ++block_pos) {
            std::vector<uint8_t> actual;
            for (const auto& info : allocatorBlockInfos(*allocator, layer_id, hit_resource->blocks(0, 0)[block_pos])) {
                auto bytes = readBlockInfoBytes(info);
                actual.insert(actual.end(), bytes.begin(), bytes.end());
            }
            EXPECT_EQ(actual, expected_payload[static_cast<size_t>(layer_id)][block_pos]);
        }
    }

    allocator->free(FreeInfo{hit_resource, hit_tokens});
    drainAllocatorCache(allocator);
    EXPECT_EQ(allocator->freeBlocksNum(), baseline_free);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
}

TEST_F(CacheLifecycleSmokeTest, PressureEvictsOldestPrefixAndRecoversPool) {
    const auto config    = makeCacheSmokeConfig(/*block_num=*/8);
    auto       allocator = makeCacheSmokeAllocator(config, /*enable_prefix_cache=*/true);
    ASSERT_TRUE(allocator->init());
    const size_t baseline_free = allocator->freeBlocksNum();
    const auto   cache         = allocator->sharedBlockCache();
    ASSERT_NE(cache, nullptr);

    auto cache_request = [&](CacheKeyType key_begin, int32_t token_begin) {
        auto keys     = makeCacheKeys(key_begin, /*count=*/3);
        auto tokens   = makeCacheSmokeTokenIds(makeTokenRange(token_begin, /*count=*/12));
        auto resource = makeCacheSmokeResource(config, keys);
        EXPECT_TRUE(allocateCacheSmokeResource(allocator, resource, tokens).success);
        allocator->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
        allocator->free(FreeInfo{resource, tokens});
    };

    cache_request(/*key_begin=*/10, /*token_begin=*/1);
    cache_request(/*key_begin=*/20, /*token_begin=*/100);
    EXPECT_EQ(allocator->freeBlocksNum(), 1u);
    ASSERT_TRUE(cache->contains(10));
    ASSERT_TRUE(cache->contains(20));

    auto pressure_tokens   = makeCacheSmokeTokenIds(makeTokenRange(/*begin=*/1000, /*count=*/12));
    auto pressure_resource = makeCacheSmokeResource(config, makeCacheKeys(/*begin=*/30, /*count=*/3));
    ASSERT_TRUE(allocateCacheSmokeResource(allocator, pressure_resource, pressure_tokens).success);

    EXPECT_FALSE(cache->contains(10));
    EXPECT_TRUE(cache->contains(20));
    allocator->free(FreeInfo{pressure_resource, pressure_tokens});
    drainAllocatorCache(allocator);

    EXPECT_EQ(allocator->freeBlocksNum(), baseline_free);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
}

TEST_F(CacheLifecycleSmokeTest, FailedAllocationRollsBackAllReferences) {
    const auto config    = makeCacheSmokeConfig(/*block_num=*/5);
    auto       allocator = makeCacheSmokeAllocator(config, /*enable_prefix_cache=*/true);
    ASSERT_TRUE(allocator->init());

    const size_t baseline_free      = allocator->freeBlocksNum();
    const size_t baseline_available = allocator->availableBlocksNum();
    auto         tokens             = makeCacheSmokeTokenIds(makeTokenRange(/*begin=*/1, /*count=*/20));
    auto         resource           = makeCacheSmokeResource(config, makeCacheKeys(/*begin=*/500, /*count=*/5));

    EXPECT_FALSE(allocateCacheSmokeResource(allocator, resource, tokens).success);
    EXPECT_EQ(resource->curBlocksNum(), 0);
    EXPECT_EQ(allocator->freeBlocksNum(), baseline_free);
    EXPECT_EQ(allocator->availableBlocksNum(), baseline_available);
    EXPECT_EQ(allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(allocator->blockCacheRefBlocksNum(), 0u);
}

TEST_F(CacheLifecycleSmokeTest, MultiTypeGroupsAllocateTaggedCopyRecycle) {
    const auto config    = makeMultiGroupCacheSmokeConfig();
    auto       allocator = makeCacheSmokeAllocatorForConfig(config, /*enable_prefix_cache=*/false);
    ASSERT_TRUE(allocator->init());
    auto* hybrid = dynamic_cast<HybridPoolKVCacheAllocator*>(allocator.get());
    ASSERT_NE(hybrid, nullptr);
    ASSERT_EQ(hybrid->groupBlockPools().size(), 5u);
    const auto baseline = snapshotCacheSmokePools(*hybrid);

    auto tokens   = makeCacheSmokeTokenIds(makeTokenRange(1, 20));
    auto resource = makeCacheSmokeResource(config, makeCacheKeys(800, 5));
    ASSERT_TRUE(allocateCacheSmokeResource(allocator, resource, tokens, /*enable_device_cache=*/true).success);
    auto& batch_resource = resource->cacheResource(0);
    ASSERT_EQ(batch_resource.layerNum(), 3);
    EXPECT_EQ(batch_resource.groupTagsForLayer(0), (std::vector<std::string>{"shared_full", "layer0_linear"}));
    EXPECT_EQ(batch_resource.groupTagsForLayer(1),
              (std::vector<std::string>{"shared_full", "layer1_swa", "layer1_full"}));
    EXPECT_EQ(batch_resource.groupTagsForLayer(2), (std::vector<std::string>{"shared_full", "layer2_linear"}));
    for (int layer = 0; layer < 3; ++layer) {
        for (const auto& tag : batch_resource.groupTagsForLayer(layer)) {
            const auto& blocks = resource->blocksForLayer(0, layer, tag);
            EXPECT_EQ(blocks.size(), 5u) << "layer=" << layer << " tag=" << tag;
            const auto valid =
                std::count_if(blocks.begin(), blocks.end(), [](BlockIdxType b) { return !isNullBlockIdx(b); });
            if (tag == "layer1_swa") {
                EXPECT_GE(valid, 2);
            } else {
                EXPECT_EQ(valid, 5);
            }
        }
    }

    fillAllocatorResource(*allocator, resource->cacheResource(0), 37);
    const auto shared0 = resource->blocksForLayer(0, 0, "shared_full");
    EXPECT_EQ(shared0, resource->blocksForLayer(0, 1, "shared_full"));
    auto       source    = resource->blocksForLayer(0, 0, "shared_full")[0];
    auto       target    = resource->blocksForLayer(0, 0, "shared_full")[1];
    const auto untouched = readBlockInfoBytes(allocatorBlockInfos(*allocator, 0, "layer0_linear", target)[0]);
    allocator->blockBatchCopyByTag({TaggedBlockIdPair{"shared_full", source, target}});
    synchronizeCacheSmokeDevice();
    EXPECT_EQ(readBlockInfoBytes(allocatorBlockInfos(*allocator, 0, "shared_full", source)[0]),
              readBlockInfoBytes(allocatorBlockInfos(*allocator, 0, "shared_full", target)[0]));
    EXPECT_EQ(untouched, readBlockInfoBytes(allocatorBlockInfos(*allocator, 0, "layer0_linear", target)[0]));

    allocator->free(FreeInfo{resource, tokens});
    expectCacheSmokePoolsEqual(*hybrid, baseline);
}

}  // namespace rtp_llm::test
