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
    EXPECT_EQ(resource.groupTagsForLayer(0), std::vector<std::string>{"group0"});
    EXPECT_EQ(&resource.blockIdsForLayer(0, "group0"), &resource.blockIds("group0"));
    EXPECT_EQ(resource.groupTagsForLayer(1), std::vector<std::string>{"group1"});
    EXPECT_EQ(&resource.blockIdsForLayer(1, "group1"), &resource.blockIds("group1"));
    EXPECT_EQ(resource.groupTagsForLayer(2), std::vector<std::string>{"group0"});
    EXPECT_EQ(&resource.blockIdsForLayer(2, "group0"), &resource.blockIds("group0"));

    KVCacheResource single_group_resource;
    single_group_resource.initGroups(makeTestCacheTopology(/*group_num=*/1,
                                                           /*layer_num=*/3,
                                                           /*layer_group_ids=*/{{0}, {0}, {0}},
                                                           /*kernel_blocks_per_kv_block=*/4,
                                                           /*group_types=*/{CacheGroupType::FULL}));
    for (int layer_id = 0; layer_id < 3; ++layer_id) {
        EXPECT_EQ(single_group_resource.groupTagsForLayer(layer_id), std::vector<std::string>{"group0"});
        EXPECT_EQ(&single_group_resource.blockIdsForLayer(layer_id, "group0"),
                  &single_group_resource.blockIds("group0"));
    }

    auto& g0 = resource.mutableBlockIds("group0");
    auto& g1 = resource.mutableBlockIds("group1");

    ASSERT_EQ(g0.kernelBlocksPerKvBlock(), 4u);
    ASSERT_EQ(g1.kernelBlocksPerKvBlock(), 1u);

    g0.add(BlockIndicesType{1});
    g1.add(BlockIndicesType{1});

    ASSERT_EQ(resource.blocks("group0"), (BlockIndicesType{1}));
    ASSERT_EQ(resource.kernelBlocks("group0"), (BlockIndicesType{4, 5, 6, 7}));

    ASSERT_EQ(resource.blocks("group1"), (BlockIndicesType{1}));
    ASSERT_EQ(resource.kernelBlocks("group1"), (BlockIndicesType{1}));
}

TEST(KVCacheResourceTest, LayerTagEnumerationReturnsAllGroupsForOneLayer) {
    KVCacheResource resource;
    resource.initGroups(makeTestCacheTopology(/*group_num=*/2,
                                              /*layer_num=*/1,
                                              /*layer_group_ids=*/{{0, 1}},
                                              /*kernel_blocks_per_kv_block=*/1,
                                              /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR}));

    EXPECT_EQ(resource.groupTagsForLayer(0), (std::vector<std::string>{"group0", "group1"}));
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
    const auto& tags = resource.groupTagsForLayer(0);
    ASSERT_EQ(tags.size(), 2u);
    EXPECT_EQ(&resource.blockIdsForLayer(0, tags.front()), &resource.blockIds(tags.front()));
}

TEST(KVCacheResourceTest, InitializationRetainsTopologyForLayerMembership) {
    auto                               topology      = makeTestCacheTopology(/*group_num=*/1, /*layer_num=*/1, {{0}});
    std::weak_ptr<const CacheTopology> weak_topology = topology;

    KVCacheResource resource;
    resource.initGroups(topology);
    topology.reset();

    EXPECT_FALSE(weak_topology.expired());
    EXPECT_EQ(resource.groupTagsForLayer(0), std::vector<std::string>{"group0"});
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
    resource.initGroups(makeTestCacheTopology(/*group_num=*/1, /*layer_num=*/1, {{0}}));
    constexpr std::string_view tag = "group0";
    resource.setCacheKeys(tag, CacheKeysType{10, 20, 30});

    ASSERT_EQ(resource.blockDependencies(tag).size(), 3u);
    EXPECT_FALSE(resource.blockDependencies(tag)[0].has_parent);
    EXPECT_EQ(resource.blockDependencies(tag)[0].ordinal, 0u);
    EXPECT_TRUE(resource.blockDependencies(tag)[1].has_parent);
    EXPECT_EQ(resource.blockDependencies(tag)[1].parent_key, 10);
    EXPECT_EQ(resource.blockDependencies(tag)[1].ordinal, 1u);
    EXPECT_TRUE(resource.blockDependencies(tag)[2].has_parent);
    EXPECT_EQ(resource.blockDependencies(tag)[2].parent_key, 20);
    EXPECT_EQ(resource.blockDependencies(tag)[2].ordinal, 2u);

    BlockDependenciesType custom = {
        BlockDependency{false, 0, 7},
        BlockDependency{true, 100, 8},
    };
    resource.setCacheKeys(tag, CacheKeysType{100, 200});
    resource.setBlockDependencies(tag, custom);
    resource.ensureLinearBlockDependencies(tag);
    ASSERT_EQ(resource.blockDependencies(tag).size(), 2u);
    EXPECT_FALSE(resource.blockDependencies(tag)[0].has_parent);
    EXPECT_EQ(resource.blockDependencies(tag)[0].ordinal, 7u);
    EXPECT_TRUE(resource.blockDependencies(tag)[1].has_parent);
    EXPECT_EQ(resource.blockDependencies(tag)[1].parent_key, 100);
    EXPECT_EQ(resource.blockDependencies(tag)[1].ordinal, 8u);

    resource.cacheKeys(tag).push_back(300);
    resource.ensureLinearBlockDependencies(tag);
    ASSERT_EQ(resource.blockDependencies(tag).size(), 3u);
    EXPECT_EQ(resource.blockDependencies(tag)[2].parent_key, 200);
    EXPECT_EQ(resource.blockDependencies(tag)[2].ordinal, 2u);
}

TEST(CacheConfigTest, KernelBlocksPerKvBlockIsTagLocal) {
    auto config = makeSimpleMhaCacheConfig(1, 2, 8, DataType::TYPE_FP16);
    ASSERT_EQ(config.kernelBlocksPerKvBlockForGroup("default"), 1u);

    auto                   groups = config.topology().groups();
    std::vector<GroupBase> updated_groups(groups.begin(), groups.end());
    updated_groups.front().kernel_seq_size_per_block = 2;
    config.setTopology(std::move(updated_groups), config.topology().layers());
    ASSERT_EQ(config.kernelBlocksPerKvBlockForGroup("default"), 4u);
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

    batch.setBatchBlocks(/*batch_id=*/0, "group0", BlockIndicesType{1, 2});
    ASSERT_EQ(batch.blocks(0, "group0"), (BlockIndicesType{1, 2}));
    ASSERT_EQ(batch.kernelBlocks(0, "group0"), (BlockIndicesType{4, 5, 6, 7, 8, 9, 10, 11}));

    batch.setBatchBlocks(/*batch_id=*/0, "group1", BlockIndicesType{9, 10});
    ASSERT_EQ(batch.blocks(0, "group1"), (BlockIndicesType{9, 10}));
    ASSERT_EQ(batch.kernelBlocks(0, "group1"), (BlockIndicesType{9, 10}));

    auto all_g0 = batch.getAllBatchBlocks("group0");
    ASSERT_EQ(all_g0.size(), 2u);
    ASSERT_EQ(all_g0[0], (BlockIndicesType{1, 2}));

    batch.pushBackCacheKey(0, "group0", 100);
    batch.pushBackCacheKey(1, "group0", 200);
    ASSERT_TRUE(batch.hasCacheKeys("group0"));

    batch.popBackAllBatchCacheKeys("group0");
    ASSERT_EQ(batch.cacheKeys(0, "group0").size(), 0u);
    ASSERT_EQ(batch.cacheKeys(1, "group0").size(), 0u);
    ASSERT_FALSE(batch.hasCacheKeys("group0"));

    batch.setLastBlockAligned("group0", true);
    ASSERT_TRUE(batch.lastBlockAligned("group0"));
    batch.cacheResource(1).setLastBlockAligned("group0", false);
    ASSERT_FALSE(batch.lastBlockAligned("group0"));

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
    moved.mutableBlockIds("group0").add(BlockIndicesType{3});
    batch.moveBatchResource(0, std::move(moved));
    ASSERT_EQ(batch.cacheResource(0).groupNums(), 1);
    ASSERT_EQ(batch.cacheResource(0).kernelBlocks("group0"), (BlockIndicesType{6, 7}));
}

}  // namespace test
}  // namespace rtp_llm
