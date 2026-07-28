#include <gtest/gtest.h>

#include <string>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {
namespace test {

namespace {

GroupBase makeResourceGroup(std::string tag, CacheGroupType type) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = tag;
    spec->seq_size_per_block = 8;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(type);
    group.layer_ids                 = {0};
    group.block_num                 = 16;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = type == CacheGroupType::FULL ? 2 : 8;
    return group;
}

}  // namespace

TEST(BlockIdsTest, NonFull_MirrorsKernelBlocks) {
    BlockIds ids(/*kernel_blocks_per_kv_block=*/1);

    ids.add(BlockIndicesType{1, 2, 3});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{1, 2, 3}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{1, 2, 3}));

    ids.remove(std::vector<size_t>{1});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{1, NULL_BLOCK_IDX, 3}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{1, NULL_BLOCK_IDX, 3}));

    ids.swap(0, 2);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{3, NULL_BLOCK_IDX, 1}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{3, NULL_BLOCK_IDX, 1}));

    ids.setAt(1, 9);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{3, 9, 1}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{3, 9, 1}));
}

TEST(BlockIdsTest, Full_ExpandsKernelBlocks) {
    BlockIds ids(/*kernel_blocks_per_kv_block=*/2);

    ids.add(BlockIndicesType{5, 7});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{5, 7}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{10, 11, 14, 15}));

    ids.remove(std::vector<size_t>{0});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 7}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 14, 15}));

    ids.setAt(1, 3);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 3}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 6, 7}));

    ids.resize(3, 2);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 3, 2}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 6, 7, 4, 5}));

    ids.swap(1, 2);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 2, 3}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 4, 5, 6, 7}));

    const auto popped = ids.popBack();
    ASSERT_EQ(popped, 3);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{NULL_BLOCK_IDX, 2}));
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 4, 5}));
}

TEST(KVCacheResourceTest, InitGroups_RespectsGroupTypesAndBlocksPerKvBlock) {
    KVCacheResource resource;
    resource.initGroups(makeTestCacheTopology(
        /*group_num=*/2,
        /*layer_num=*/3,
        /*layer_group_ids=*/{{0}, {1}, {0}},
        /*kernel_blocks_per_kv_block=*/4,
        /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR}));

    ASSERT_EQ(resource.groupNums(), 2);
    auto multi_group_layer_blocks = resource.layerBlocks();
    ASSERT_EQ(multi_group_layer_blocks.size(), 3u);
    EXPECT_EQ(multi_group_layer_blocks[0], resource.groupBlocks()[0]);
    EXPECT_EQ(multi_group_layer_blocks[1], resource.groupBlocks()[1]);
    EXPECT_EQ(multi_group_layer_blocks[2], resource.groupBlocks()[0]);
    ASSERT_EQ(resource.layerGroupBlocks().size(), 3u);

    KVCacheResource single_group_resource;
    single_group_resource.initGroups(makeTestCacheTopology(/*group_num=*/1,
                                                           /*layer_num=*/3,
                                                           /*layer_group_ids=*/{{0}, {0}, {0}},
                                                           /*kernel_blocks_per_kv_block=*/4,
                                                           /*group_types=*/{CacheGroupType::FULL}));
    auto layer_blocks = single_group_resource.layerBlocks();
    ASSERT_EQ(layer_blocks.size(), 3u);
    ASSERT_EQ(layer_blocks[0], single_group_resource.groupBlocks()[0]);
    ASSERT_EQ(layer_blocks[1], single_group_resource.groupBlocks()[0]);
    ASSERT_EQ(layer_blocks[2], single_group_resource.groupBlocks()[0]);

    auto& g0 = resource.mutableBlockIds(0);
    auto& g1 = resource.mutableBlockIds(1);

    ASSERT_EQ(g0.kernelBlocksPerKvBlock(), 4u);
    ASSERT_EQ(g1.kernelBlocksPerKvBlock(), 1u);

    g0.add(BlockIndicesType{1});
    g1.add(BlockIndicesType{1});

    ASSERT_EQ(resource.blocks(0), (BlockIndicesType{1}));
    ASSERT_EQ(resource.kernelBlocks(0), (BlockIndicesType{4, 5, 6, 7}));

    ASSERT_EQ(resource.blocks(1), (BlockIndicesType{1}));
    ASSERT_EQ(resource.kernelBlocks(1), (BlockIndicesType{1}));
}

TEST(KVCacheResourceTest, LayerBlocksRejectsMultipleGroupsForOneLayer) {
    KVCacheResource resource;
    resource.initGroups(makeTestCacheTopology(/*group_num=*/2,
                                              /*layer_num=*/1,
                                              /*layer_group_ids=*/{{0, 1}},
                                              /*kernel_blocks_per_kv_block=*/1,
                                              /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR}));

    EXPECT_THROW(resource.layerBlocks(), std::exception);
}

TEST(KVCacheResourceTest, TagAccessKeepsSameLayerGroupsIndependent) {
    auto topology = CacheTopology::create(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{0, {"full", "linear"}}});
    KVCacheResource resource;
    resource.initGroups(topology);

    resource.mutableBlockIdsForLayer(0, "full").add(BlockIndicesType{1, 2});
    resource.mutableBlockIdsForLayer(0, "linear").add(BlockIndicesType{7});

    EXPECT_EQ(resource.blocksForLayer(0, "full"), (BlockIndicesType{1, 2}));
    EXPECT_EQ(resource.kernelBlocksForLayer(0, "full"), (BlockIndicesType{4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(resource.blocksForLayer(0, "linear"), (BlockIndicesType{7}));
    EXPECT_EQ(resource.kernelBlocksForLayer(0, "linear"), (BlockIndicesType{7}));
    EXPECT_NE(&resource.blockIds("full"), &resource.blockIds("linear"));
    EXPECT_ANY_THROW(resource.layerBlocks());
}

TEST(KVCacheResourceTest, InitializationDoesNotRetainTopology) {
    auto                               topology      = makeTestCacheTopology(/*group_num=*/1, /*layer_num=*/1, {{0}});
    std::weak_ptr<const CacheTopology> weak_topology = topology;

    KVCacheResource resource;
    resource.initGroups(topology);
    topology.reset();

    EXPECT_TRUE(weak_topology.expired());
    EXPECT_EQ(resource.soleGroupTagForLayer(0), "group0");
    resource.mutableBlockIdsForLayer(0, "group0").add(BlockIndicesType{3});
    EXPECT_EQ(resource.blocksForLayer(0, "group0"), (BlockIndicesType{3}));
}

TEST(PrefillCPConfigTest, ToStringIncludesShardingFields) {
    PrefillCPConfig config;
    config.kv_cache_sharded = true;
    config.prefill_cp_size  = 2;

    const auto text = config.to_string();
    EXPECT_NE(text.find("kv_cache_sharded: 1"), std::string::npos);
    EXPECT_NE(text.find("prefill_cp_size: 2"), std::string::npos);
}

TEST(KVCacheResourceTest, CacheKeysMaintainLinearDependencies) {
    KVCacheResource resource;
    resource.setCacheKeys(CacheKeysType{10, 20, 30});

    ASSERT_EQ(resource.blockDependencies().size(), 3u);
    EXPECT_FALSE(resource.blockDependencies()[0].has_parent);
    EXPECT_EQ(resource.blockDependencies()[0].ordinal, 0u);
    EXPECT_TRUE(resource.blockDependencies()[1].has_parent);
    EXPECT_EQ(resource.blockDependencies()[1].parent_key, 10);
    EXPECT_EQ(resource.blockDependencies()[1].ordinal, 1u);
    EXPECT_TRUE(resource.blockDependencies()[2].has_parent);
    EXPECT_EQ(resource.blockDependencies()[2].parent_key, 20);
    EXPECT_EQ(resource.blockDependencies()[2].ordinal, 2u);

    BlockDependenciesType custom = {
        BlockDependency{false, 0, 7},
        BlockDependency{true, 100, 8},
    };
    resource.setCacheKeys(CacheKeysType{100, 200});
    resource.setBlockDependencies(custom);
    resource.ensureLinearBlockDependencies();
    ASSERT_EQ(resource.blockDependencies().size(), 2u);
    EXPECT_FALSE(resource.blockDependencies()[0].has_parent);
    EXPECT_EQ(resource.blockDependencies()[0].ordinal, 0u);
    EXPECT_TRUE(resource.blockDependencies()[1].has_parent);
    EXPECT_EQ(resource.blockDependencies()[1].parent_key, 100);
    EXPECT_EQ(resource.blockDependencies()[1].ordinal, 1u);

    resource.cacheKeys().push_back(300);
    resource.ensureLinearBlockDependencies();
    ASSERT_EQ(resource.blockDependencies().size(), 3u);
    EXPECT_EQ(resource.blockDependencies()[2].parent_key, 200);
    EXPECT_EQ(resource.blockDependencies()[2].ordinal, 2u);
}

TEST(CacheConfigTest, KernelBlocksPerKvBlockSafeByDefault) {
    CacheConfig config;
    config.seq_size_per_block        = 1;
    config.kernel_seq_size_per_block = 0;
    ASSERT_EQ(config.kernelBlocksPerKvBlock(), 1u);

    config.seq_size_per_block        = 8;
    config.kernel_seq_size_per_block = 2;
    ASSERT_EQ(config.kernelBlocksPerKvBlock(), 4u);
}

TEST(BatchKVCacheResourceTest, BasicBatchOperations_WorkAsExpected) {
    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(makeTestCacheTopology(/*group_num=*/2,
                                           /*layer_num=*/3,
                                           /*layer_group_ids=*/{{0}, {1}, {0}},
                                           /*kernel_blocks_per_kv_block=*/4,
                                           /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR}));

    ASSERT_EQ(batch.batchSize(), 2);
    ASSERT_EQ(batch.groupNums(), 2);

    batch.setBatchBlocks(/*batch_id=*/0, /*group_id=*/0, BlockIndicesType{1, 2});
    ASSERT_EQ(batch.blocks(0, 0), (BlockIndicesType{1, 2}));
    ASSERT_EQ(batch.kernelBlocks(0, 0), (BlockIndicesType{4, 5, 6, 7, 8, 9, 10, 11}));

    batch.setBatchBlocks(/*batch_id=*/0, /*group_id=*/1, BlockIndicesType{9, 10});
    ASSERT_EQ(batch.blocks(0, 1), (BlockIndicesType{9, 10}));
    ASSERT_EQ(batch.kernelBlocks(0, 1), (BlockIndicesType{9, 10}));

    auto all_g0 = batch.getAllBatchBlocks(/*group_id=*/0);
    ASSERT_EQ(all_g0.size(), 2u);
    ASSERT_EQ(all_g0[0], (BlockIndicesType{1, 2}));

    batch.pushBackCacheKey(0, 100);
    batch.pushBackCacheKey(1, 200);
    ASSERT_TRUE(batch.hasCacheKeys());

    batch.popBackAllBatchCacheKeys();
    ASSERT_EQ(batch.cacheKeys(0).size(), 0u);
    ASSERT_EQ(batch.cacheKeys(1).size(), 0u);
    ASSERT_FALSE(batch.hasCacheKeys());

    batch.setLastBlockAligned(true);
    ASSERT_TRUE(batch.lastBlockAligned());
    batch.cacheResource(1).setLastBlockAligned(false);
    ASSERT_FALSE(batch.lastBlockAligned());

    std::vector<KVCacheResource> old_resources;
    batch.resetAndReturnOldResources(/*new_batch_size=*/1, old_resources);
    ASSERT_EQ(old_resources.size(), 2u);
    ASSERT_EQ(batch.batchSize(), 1);

    KVCacheResource moved;
    moved.initGroups(makeTestCacheTopology(/*group_num=*/1,
                                           /*layer_num=*/1,
                                           /*layer_group_ids=*/{{0}},
                                           /*kernel_blocks_per_kv_block=*/2,
                                           /*group_types=*/{CacheGroupType::FULL}));
    moved.mutableBlockIds(0).add(BlockIndicesType{3});
    batch.moveBatchResource(0, std::move(moved));
    ASSERT_EQ(batch.cacheResource(0).groupNums(), 1);
    ASSERT_EQ(batch.cacheResource(0).kernelBlocks(0), (BlockIndicesType{6, 7}));
}

}  // namespace test
}  // namespace rtp_llm
