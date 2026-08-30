#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {
namespace test {

namespace {

CacheGroup makeResourceGroup(std::string tag, CacheGroupType type) {
    auto spec                       = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block        = 8;
    spec->kernel_seq_size_per_block = type == CacheGroupType::FULL ? 2 : 8;

    CacheGroup group;
    group.tag       = std::move(tag);
    group.spec      = std::move(spec);
    group.policy    = defaultCacheGroupPolicy(type);
    group.block_num = 16;
    return group;
}

CacheConfig makeResourceConfig(std::vector<CacheGroup> groups, std::vector<CacheLayer> layers) {
    const auto  layer_num = static_cast<uint32_t>(layers.size());
    CacheConfig config(std::move(groups), std::move(layers), layer_num);
    return config;
}

}  // namespace

TEST(BlockIdsTest, MutationsPreservePositionsAndHoles) {
    BlockIds ids;

    ids.add(BlockIndicesType{1, 2, 3});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{1, 2, 3}));

    ids.remove(std::vector<size_t>{1});
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{1, NULL_BLOCK_IDX, 3}));

    ids.swap(0, 2);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{3, NULL_BLOCK_IDX, 1}));

    ids.setAt(1, 9);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{3, 9, 1}));
}

TEST(BlockIdsTest, ResizeAndPopPreserveSafeDummyZeroAndRemainingPositions) {
    BlockIds ids;
    ids.assign(BlockIndicesType{0, 7});

    ids.resize(4);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{0, 7, NULL_BLOCK_IDX, NULL_BLOCK_IDX}));

    ids.resize(5, 0);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{0, 7, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 0}));

    EXPECT_EQ(ids.popBack(), 0);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{0, 7, NULL_BLOCK_IDX, NULL_BLOCK_IDX}));

    ids.resize(1);
    ASSERT_EQ(ids.blocks(), (BlockIndicesType{0}));
    EXPECT_EQ(ids.popBack(), 0);
    EXPECT_TRUE(ids.blocks().empty());
}

TEST(BlockIdsTest, KernelBlocksAreProjectedWithoutPersistentKernelStorage) {
    BlockIds ids(/*kernel_blocks_per_block=*/2);

    ids.assign(BlockIndicesType{5, 7});
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{10, 11, 14, 15}));

    ids.assign(BlockIndicesType{NULL_BLOCK_IDX, 3});
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{0, 0, 6, 7}));
    ids.assign(BlockIndicesType{0});
    ASSERT_EQ(ids.kernelBlocks(), (BlockIndicesType{0, 1}));
}

TEST(BlockIdsTest, WriteKernelBlocksUsesCallerBufferAndPreservesProjectionSemantics) {
    BlockIds ids(/*kernel_blocks_per_block=*/2);
    ids.assign(BlockIndicesType{5, NULL_BLOCK_IDX, 7});

    BlockIdxType output[6] = {};
    EXPECT_EQ(ids.writeKernelBlocks(output, 6u), 6u);
    EXPECT_EQ(BlockIndicesType(output, output + 6), (BlockIndicesType{10, 11, 0, 0, 14, 15}));

    EXPECT_THROW(ids.writeKernelBlocks(output, 5u), std::exception);
}

TEST(BlockIdsTest, WriteKernelBlocksRejectsInvalidPhysicalBlock) {
    BlockIds ids(/*kernel_blocks_per_block=*/2);
    ids.assign(BlockIndicesType{-2});
    BlockIdxType output[2] = {};
    EXPECT_THROW(ids.writeKernelBlocks(output, 2u), std::exception);
}

TEST(KVCacheResourceTest, InitGroups_RespectsGroupTypesAndBlocksPerKvBlock) {
    const auto config = makeTestCacheConfigByTag(
        /*group_num=*/2,
        /*layer_num=*/3,
        /*layer_group_tags=*/{{"group0"}, {"group1"}, {"group0"}},
        /*kernel_blocks_per_kv_block=*/4,
        /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR});

    KVCacheResource resource;
    resource.initGroups(config);

    ASSERT_EQ(resource.groupNums(), 2);
    ASSERT_EQ(resource.blocksByGroup().size(), 2u);
    EXPECT_EQ(&resource.blockIdsForLayer(0, "group0"), &resource.blockIds("group0"));
    EXPECT_EQ(&resource.blockIdsForLayer(1, "group1"), &resource.blockIds("group1"));
    EXPECT_EQ(&resource.blockIdsForLayer(2, "group0"), &resource.blockIds("group0"));

    auto& g0 = resource.mutableBlockIds("group0");
    auto& g1 = resource.mutableBlockIds("group1");

    ASSERT_EQ(config.group("group0").kernelBlocksPerKvBlock(), 4u);
    ASSERT_EQ(config.group("group1").kernelBlocksPerKvBlock(), 4u);
    ASSERT_EQ(config.group("group0").storedKernelBlocksPerKvBlock(), 4u);
    ASSERT_EQ(config.group("group1").storedKernelBlocksPerKvBlock(), 1u);
    g0.add(BlockIndicesType{1});
    g1.add(BlockIndicesType{1});

    ASSERT_EQ(resource.blocks("group0"), (BlockIndicesType{1}));
    ASSERT_EQ(resource.blocks("group1"), (BlockIndicesType{1}));
    ASSERT_EQ(resource.blockIds("group0").kernelBlocks(), (BlockIndicesType{4, 5, 6, 7}));
    ASSERT_EQ(resource.blockIds("group1").kernelBlocks(), (BlockIndicesType{1}));
}

TEST(KVCacheResourceTest, LayerBlocksRejectsMultipleGroupsForOneLayer) {
    KVCacheResource resource;
    resource.initGroups(makeTestCacheConfigByTag(
        /*group_num=*/2,
        /*layer_num=*/1,
        /*layer_group_tags=*/{{"group0", "group1"}},
        /*kernel_blocks_per_kv_block=*/1,
        /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR}));

    EXPECT_THROW(resource.blockIdsForLayer(0, "unknown"), std::exception);
}

TEST(KVCacheResourceTest, TagAccessKeepsSameLayerGroupsIndependent) {
    KVCacheResource resource;
    auto            config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});
    resource.initGroups(config);

    resource.mutableBlockIdsForLayer(0, "full").add(BlockIndicesType{1, 2});
    resource.mutableBlockIdsForLayer(0, "linear").add(BlockIndicesType{7});

    EXPECT_EQ(resource.blocksForLayer(0, "full"), (BlockIndicesType{1, 2}));
    EXPECT_EQ(resource.blocksForLayer(0, "linear"), (BlockIndicesType{7}));
    EXPECT_NE(&resource.blockIds("full"), &resource.blockIds("linear"));
}

TEST(KVCacheResourceTest, BlocksByGroupOwnsOneBlockTablePerTag) {
    auto linear = makeResourceGroup("linear", CacheGroupType::LINEAR);
    auto full   = makeResourceGroup("full", CacheGroupType::FULL);
    auto config = makeResourceConfig({std::move(linear), std::move(full)}, {{"full", "linear"}});

    KVCacheResource resource;
    resource.initGroups(config);
    resource.mutableBlockIds("full").add({11});
    resource.mutableBlockIds("linear").add({22});

    const auto& blocks_by_group = resource.blocksByGroup();
    ASSERT_EQ(blocks_by_group.size(), 2u);
    EXPECT_EQ(blocks_by_group.begin()->first, "full");
    EXPECT_EQ(blocks_by_group.at("full").blocks(), (BlockIndicesType{11}));
    EXPECT_EQ(blocks_by_group.at("linear").blocks(), (BlockIndicesType{22}));

    EXPECT_EQ(resource.blockIds("full").blocks(), (BlockIndicesType{11}));
    EXPECT_EQ(resource.blockIds("linear").blocks(), (BlockIndicesType{22}));
    EXPECT_EQ(resource.blockIdsForLayer(0, "full").blocks(), (BlockIndicesType{11}));
    EXPECT_EQ(resource.blockIdsForLayer(0, "linear").blocks(), (BlockIndicesType{22}));
}

TEST(BatchKVCacheResourceTest, CheckValidatesEveryTagAcrossBatches) {
    auto config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});

    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(config);
    batch.setBatchBlocks(0, "full", {1, 2});
    batch.setBatchBlocks(1, "full", {3, 4});
    batch.setBatchBlocks(0, "linear", {5});
    batch.setBatchBlocks(1, "linear", {6, 7});

    EXPECT_THROW(batch.check(), std::exception);
}

TEST(BatchKVCacheResourceTest, CheckAllowsDifferentBlockCountsBetweenTags) {
    auto config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});

    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(config);
    batch.setBatchBlocks(0, "full", {1, 2});
    batch.setBatchBlocks(1, "full", {3, 4});
    batch.setBatchBlocks(0, "linear", {5});
    batch.setBatchBlocks(1, "linear", {6});

    EXPECT_NO_THROW(batch.check());
}

TEST(BatchKVCacheResourceTest, CheckRejectsDifferentTagSetsWithTheSameSize) {
    auto expected_config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});
    auto different_config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("state", CacheGroupType::LINEAR)},
        {{"full", "state"}});

    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(expected_config);
    batch.setBatchBlocks(0, "full", {1});
    batch.setBatchBlocks(0, "linear", {2});

    KVCacheResource different_resource;
    different_resource.initGroups(different_config);
    different_resource.mutableBlockIds("full").assign({3});
    different_resource.mutableBlockIds("state").assign({4});
    batch.moveBatchResource(1, std::move(different_resource));

    EXPECT_THROW(batch.check(), std::exception);
}

TEST(KVCacheResourceTest, UnknownTagIsRejectedWithoutMutatingStorage) {
    auto config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});

    KVCacheResource missing;
    missing.initGroups(config);
    EXPECT_THROW(missing.blockIds("other"), std::exception);
    EXPECT_EQ(missing.blocksByGroup().size(), 2u);
    EXPECT_EQ(missing.blocksByGroup().count("full"), 1u);
    EXPECT_EQ(missing.blocksByGroup().count("linear"), 1u);
}

TEST(KVCacheResourceTest, TaggedStorageHasOneRecordPerConfigGroupNotPerLayer) {
    auto full   = makeResourceGroup("full", CacheGroupType::FULL);
    auto config = makeResourceConfig({std::move(full)}, {{"full"}, {"full"}, {"full"}});

    KVCacheResource resource;
    resource.initGroups(config);

    ASSERT_EQ(resource.blocksByGroup().size(), 1u);
    EXPECT_EQ(resource.blocksByGroup().count("full"), 1u);
    EXPECT_EQ(&resource.blockIdsForLayer(0, "full"), &resource.blockIds("full"));
    EXPECT_EQ(&resource.blockIdsForLayer(1, "full"), &resource.blockIds("full"));
    EXPECT_EQ(&resource.blockIdsForLayer(2, "full"), &resource.blockIds("full"));
}

TEST(KVCacheResourceTest, InitializationDoesNotRetainConfig) {
    KVCacheResource                  resource;
    std::weak_ptr<const CacheConfig> weak_config;
    {
        auto config = std::make_shared<const CacheConfig>(
            makeTestCacheConfigByTag(/*group_num=*/1, /*layer_num=*/1, {{"group0"}}));
        weak_config = config;
        resource.initGroups(*config);
    }

    EXPECT_TRUE(weak_config.expired());
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

TEST(KVCacheResourceTest, ExplicitTimelineRejectsMismatchedLengthsWithoutChangingState) {
    KVCacheResource resource;
    resource.setCacheKeys(CacheKeysType{10});

    const BlockDependenciesType dependencies = {
        BlockDependency{false, 0, 7},
    };
    EXPECT_THROW(resource.setCacheKeysAndBlockDependencies(CacheKeysType{20, 30}, dependencies), std::exception);

    EXPECT_EQ(resource.cacheKeys(), (CacheKeysType{10}));
    ASSERT_EQ(resource.blockDependencies().size(), 1u);
    EXPECT_FALSE(resource.blockDependencies()[0].has_parent);
    EXPECT_EQ(resource.blockDependencies()[0].ordinal, 0u);
}

TEST(KVCacheResourceTest, ExplicitTimelinePreservesDependencyOrdinalAndParent) {
    KVCacheResource             resource;
    const BlockDependenciesType dependencies = {
        BlockDependency{true, 17, 7},
        BlockDependency{true, 29, 11},
    };

    resource.setCacheKeysAndBlockDependencies(CacheKeysType{100, 200}, dependencies);

    EXPECT_EQ(resource.cacheKeys(), (CacheKeysType{100, 200}));
    ASSERT_EQ(resource.blockDependencies().size(), 2u);
    EXPECT_TRUE(resource.blockDependencies()[0].has_parent);
    EXPECT_EQ(resource.blockDependencies()[0].parent_key, 17);
    EXPECT_EQ(resource.blockDependencies()[0].ordinal, 7u);
    EXPECT_TRUE(resource.blockDependencies()[1].has_parent);
    EXPECT_EQ(resource.blockDependencies()[1].parent_key, 29);
    EXPECT_EQ(resource.blockDependencies()[1].ordinal, 11u);
}

TEST(KVCacheResourceTest, AppendPopAndClearKeepTimelineAligned) {
    KVCacheResource resource;
    resource.setCacheKeys(CacheKeysType{10, 20});
    resource.appendCacheKey(30);

    ASSERT_EQ(resource.blockDependencies().size(), 3u);
    EXPECT_FALSE(resource.blockDependencies()[0].has_parent);
    EXPECT_EQ(resource.blockDependencies()[0].ordinal, 0u);
    EXPECT_TRUE(resource.blockDependencies()[1].has_parent);
    EXPECT_EQ(resource.blockDependencies()[1].parent_key, 10);
    EXPECT_EQ(resource.blockDependencies()[1].ordinal, 1u);
    EXPECT_TRUE(resource.blockDependencies()[2].has_parent);
    EXPECT_EQ(resource.blockDependencies()[2].parent_key, 20);
    EXPECT_EQ(resource.blockDependencies()[2].ordinal, 2u);

    resource.popBackCacheKey();
    EXPECT_EQ(resource.cacheKeys(), (CacheKeysType{10, 20}));
    ASSERT_EQ(resource.blockDependencies().size(), 2u);

    resource.clearCacheKeys();
    EXPECT_TRUE(resource.cacheKeys().empty());
    EXPECT_TRUE(resource.blockDependencies().empty());
}

TEST(BatchKVCacheResourceTest, BasicBatchOperations_WorkAsExpected) {
    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(makeTestCacheConfigByTag(
        /*group_num=*/2,
        /*layer_num=*/3,
        /*layer_group_tags=*/{{"group0"}, {"group1"}, {"group0"}},
        /*kernel_blocks_per_kv_block=*/4,
        /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR}));

    ASSERT_EQ(batch.batchSize(), 2);
    ASSERT_EQ(batch.groupNums(), 2);

    batch.setBatchBlocks(/*batch_id=*/0, "group0", BlockIndicesType{1, 2});
    ASSERT_EQ(batch.blocks(0, "group0"), (BlockIndicesType{1, 2}));

    batch.setBatchBlocks(/*batch_id=*/0, "group1", BlockIndicesType{9, 10});
    ASSERT_EQ(batch.blocks(0, "group1"), (BlockIndicesType{9, 10}));

    auto all_g0 = batch.getAllBatchBlocks("group0");
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
    moved.initGroups(makeTestCacheConfigByTag(/*group_num=*/1,
                                              /*layer_num=*/1,
                                              /*layer_group_tags=*/{{"group0"}},
                                              /*kernel_blocks_per_kv_block=*/2,
                                              /*group_types=*/{CacheGroupType::FULL}));
    moved.mutableBlockIds("group0").add(BlockIndicesType{3});
    batch.moveBatchResource(0, std::move(moved));
    ASSERT_EQ(batch.cacheResource(0).groupNums(), 1);
    ASSERT_EQ(batch.cacheResource(0).blocks("group0"), (BlockIndicesType{3}));
}

TEST(BatchKVCacheResourceTest, CopyOwnsTagMappedBlocksWhileMoveAndTimelineStateStayIntact) {
    auto config = makeResourceConfig({makeResourceGroup("full", CacheGroupType::FULL)}, {{"full"}});
    ASSERT_EQ(config.group("full").storedKernelBlocksPerKvBlock(), 4u);

    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(config);
    batch.setBatchBlocks(0, "full", {3, 4});
    batch.swapBlocks(0, "full", 0, 1);
    EXPECT_EQ(batch.blocks(0, "full"), (BlockIndicesType{4, 3}));
    batch.swapBlocks(0, "full", 0, 1);
    batch.cacheResource(0).setCacheKeysAndBlockDependencies({101, 202}, {{true, 7, 9}, {true, 101, 12}});
    batch.cacheResource(0).setCacheKeysAreCpCanonical(true);

    BatchKVCacheResource copied = batch;
    copied.mutableBlockIds(0, "full").setAt(1, 8);
    EXPECT_EQ(batch.blocks(0, "full"), (BlockIndicesType{3, 4}));
    EXPECT_EQ(batch.cacheResource(0).blockIds("full").kernelBlocks(),
              (BlockIndicesType{12, 13, 14, 15, 16, 17, 18, 19}));
    EXPECT_EQ(copied.cacheResource(0).blockIds("full").kernelBlocks(),
              (BlockIndicesType{12, 13, 14, 15, 32, 33, 34, 35}));
    EXPECT_EQ(copied.cacheKeys(0), (CacheKeysType{101, 202}));
    ASSERT_EQ(copied.cacheResource(0).blockDependencies().size(), 2u);
    EXPECT_EQ(copied.cacheResource(0).blockDependencies()[0].parent_key, 7);
    EXPECT_EQ(copied.cacheResource(0).blockDependencies()[0].ordinal, 9u);
    EXPECT_EQ(copied.cacheResource(0).blockDependencies()[1].parent_key, 101);
    EXPECT_EQ(copied.cacheResource(0).blockDependencies()[1].ordinal, 12u);
    EXPECT_TRUE(copied.cacheResource(0).cacheKeysAreCpCanonical());

    std::vector<KVCacheResource> old_resources;
    copied.resetAndReturnOldResources(/*new_batch_size=*/3, old_resources);
    ASSERT_EQ(old_resources.size(), 2u);
    EXPECT_EQ(old_resources[0].blockIds("full").kernelBlocks(), (BlockIndicesType{12, 13, 14, 15, 32, 33, 34, 35}));
    copied.initGroups(config);
    EXPECT_TRUE(copied.cacheResource(0).blockIds("full").kernelBlocks().empty());
    copied.moveBatchResource(2, std::move(old_resources[0]));

    EXPECT_EQ(copied.batchSize(), 3);
    EXPECT_EQ(copied.blocks(2, "full"), (BlockIndicesType{3, 8}));
    EXPECT_EQ(copied.cacheResource(2).blockIds("full").kernelBlocks(),
              (BlockIndicesType{12, 13, 14, 15, 32, 33, 34, 35}));
    EXPECT_EQ(copied.cacheKeys(2), (CacheKeysType{101, 202}));
    ASSERT_EQ(copied.cacheResource(2).blockDependencies().size(), 2u);
    EXPECT_EQ(copied.cacheResource(2).blockDependencies()[0].ordinal, 9u);
    EXPECT_EQ(copied.cacheResource(2).blockDependencies()[1].ordinal, 12u);
    EXPECT_TRUE(copied.cacheResource(2).cacheKeysAreCpCanonical());
    ASSERT_EQ(copied.cacheResource(0).blocksByGroup().size(), 1u);
    ASSERT_EQ(copied.cacheResource(1).blocksByGroup().size(), 1u);
    ASSERT_EQ(copied.cacheResource(2).blocksByGroup().size(), 1u);
    EXPECT_EQ(copied.cacheResource(2).blocksByGroup().count("full"), 1u);
}

}  // namespace test
}  // namespace rtp_llm
