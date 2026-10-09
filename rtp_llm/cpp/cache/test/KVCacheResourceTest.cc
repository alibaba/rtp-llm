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

GroupBase makeResourceGroup(std::string tag, CacheGroupType type) {
    auto spec                       = std::make_shared<MHAKVCacheSpec>();
    spec->tag                       = tag;
    spec->seq_size_per_block        = 8;
    spec->kernel_seq_size_per_block = type == CacheGroupType::FULL ? 2 : 8;

    GroupBase group;
    group.tag       = std::move(tag);
    group.spec      = std::move(spec);
    group.policy    = defaultCacheGroupPolicy(type);
    group.block_num = 16;
    return group;
}

}  // namespace

TEST(GroupBlockIdsTest, CopyOwnsIdentityAndSharesHolders) {
    GroupBlockIds copy;
    {
        GroupBlockIds original;
        original.rows_         = {std::make_shared<BlockIds>(4), std::make_shared<BlockIds>()};
        original.tag_to_index_ = {{"alpha", 1}, {"zeta", 0}};
        original.mutableBlockIds("zeta").assign({2, NULL_BLOCK_IDX});
        original.validate();
        copy = original;
        EXPECT_NE(&copy.tag_to_index_, &original.tag_to_index_);
        EXPECT_NE(copy.rows_.data(), original.rows_.data());
        EXPECT_EQ(copy.rows_[0], original.rows_[0]);
        copy.mutableBlockIds("zeta").setAt(0, 3);
        EXPECT_EQ(original.blocks("zeta"), (BlockIndicesType{3, NULL_BLOCK_IDX}));
        // Replace the whole temporary identity object; the copied map/vector survive.
        original = GroupBlockIds{};
        EXPECT_EQ(copy.orderedTags(), (std::vector<std::string>{"zeta", "alpha"}));
    }
    EXPECT_EQ(copy.size(), 2u);
    EXPECT_EQ(copy.blocksNum("alpha"), 0);
    EXPECT_EQ(copy.kernelBlocks("zeta"), (BlockIndicesType{12, 13, 14, 15, -1, -1, -1, -1}));
    EXPECT_ANY_THROW(copy.blockIds("missing"));
}

TEST(GroupBlockIdsTest, ValidationRejectsIncompleteOrAmbiguousIdentity) {
    GroupBlockIds empty;
    EXPECT_NO_THROW(empty.validate());
    EXPECT_TRUE(empty.orderedTags().empty());
    GroupBlockIds valid;
    valid.rows_         = {std::make_shared<BlockIds>(), std::make_shared<BlockIds>()};
    valid.tag_to_index_ = {{"first", 0}, {"second", 1}};
    EXPECT_NO_THROW(valid.validate());

    auto invalid = valid;
    invalid.tag_to_index_.erase("second");
    EXPECT_ANY_THROW(invalid.validate());
    invalid                         = valid;
    invalid.tag_to_index_["second"] = 0;
    EXPECT_ANY_THROW(invalid.validate());
    EXPECT_ANY_THROW(invalid.orderedTags());
    invalid                         = valid;
    invalid.tag_to_index_["second"] = 2;
    EXPECT_ANY_THROW(invalid.validate());
    EXPECT_ANY_THROW(invalid.blocks("second"));
    invalid = valid;
    invalid.tag_to_index_.erase("second");
    invalid.tag_to_index_[""] = 1;
    EXPECT_ANY_THROW(invalid.validate());
    invalid = valid;
    invalid.rows_[1].reset();
    EXPECT_ANY_THROW(invalid.validate());
    EXPECT_ANY_THROW(invalid.kernelBlocks("second"));
}

TEST(KVCacheResourceTest, CompleteObjectCopyAndLayerViewsShareHoldersAfterResourceDestruction) {
    GroupBlockIds copy;
    LayerBlockIds layers;
    {
        KVCacheResource resource;
        resource.initGroups(CacheTopology::create(
            {makeResourceGroup("zeta", CacheGroupType::FULL), makeResourceGroup("alpha", CacheGroupType::SWA)},
            {{0, {"alpha"}}, {1, {"zeta"}}, {2, {"alpha"}}}));
        const KVCacheResource& view = resource;
        view.mutableBlockIds("alpha").assign({7});
        copy   = view.groupBlockIds();
        layers = resource.layerBlocks();
        EXPECT_EQ(layers[0], layers[2]);
        EXPECT_EQ(layers[0].get(), &copy.blockIds("alpha"));
        copy.mutableBlockIds("alpha").setAt(0, 9);
        EXPECT_EQ(resource.blocksForLayer(2, "alpha"), (BlockIndicesType{9}));
        EXPECT_ANY_THROW(resource.blocksForLayer(0, "zeta"));
    }
    EXPECT_EQ(copy.orderedTags(), (std::vector<std::string>{"zeta", "alpha"}));
    EXPECT_EQ(layers[0]->blocks(), (BlockIndicesType{9}));
    copy.mutableBlockIds("alpha").setAt(0, 11);
    EXPECT_EQ(layers[2]->blocks(), (BlockIndicesType{11}));
}

TEST(KVCacheResourceTest, FailedReinitializationDoesNotPublishPartialRowsOrLayers) {
    KVCacheResource resource;
    resource.initGroups(CacheTopology::create({makeResourceGroup("original", CacheGroupType::FULL)},
                                              {{0, {"original"}}, {1, {"original"}}}));
    resource.mutableBlockIds("original").assign({5});
    resource.setCacheKeys({10, 11});
    resource.setDeviceReuseBlockNum(3);
    const auto old_holder = resource.groupBlockIds("original");
    const auto old_layers = resource.layerBlocks();
    const auto good       = makeResourceGroup("first", CacheGroupType::FULL);
    auto       bad        = makeResourceGroup("second", CacheGroupType::FULL);
    auto       bad_spec = std::make_shared<MHAKVCacheSpec>(*std::dynamic_pointer_cast<const MHAKVCacheSpec>(bad.spec));
    bad.spec            = bad_spec;
    const auto replacement = CacheTopology::create({good, bad}, {{0, {"first", "second"}}});
    // Inject a failure in the second row after the first temporary holder was built.
    bad_spec->kernel_seq_size_per_block = 0;
    EXPECT_ANY_THROW(resource.initGroups(replacement));
    EXPECT_ANY_THROW(resource.initGroups(nullptr));
    EXPECT_EQ(resource.groupTags(), (std::vector<std::string>{"original"}));
    EXPECT_EQ(resource.groupBlockIds("original"), old_holder);
    EXPECT_EQ(resource.layerBlocks(), old_layers);
    EXPECT_EQ(resource.blocks("original"), (BlockIndicesType{5}));
    EXPECT_EQ(resource.cacheKeys(), (CacheKeysType{10, 11}));
    ASSERT_EQ(resource.blockDependencies().size(), 2u);
    EXPECT_EQ(resource.blockDependencies()[1].parent_key, 10);
    EXPECT_EQ(resource.deviceReuseBlockNum(), 3u);
}

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

TEST(KVCacheResourceTest, SwapBlocksUsesTagAcrossDifferentGroupOrders) {
    for (const bool reversed : {false, true}) {
        std::vector<GroupBase> groups{makeResourceGroup("full", CacheGroupType::FULL),
                                      makeResourceGroup("linear", CacheGroupType::LINEAR)};
        if (reversed) {
            std::reverse(groups.begin(), groups.end());
        }
        const auto           topology = CacheTopology::create(std::move(groups), {{0, {"full"}}, {1, {"linear"}}});
        BatchKVCacheResource resource;
        resource.resetBatchSize(2);
        resource.initGroups(topology);
        for (int batch = 0; batch < 2; ++batch) {
            resource.mutableBlockIds(batch, "full").assign({2, 5});
            resource.mutableBlockIds(batch, "linear").assign({3, 7});
        }
        resource.swapBlocks(1, "linear", 0, 1);
        EXPECT_EQ(resource.blocksForLayer(1, 1, "linear"), (BlockIndicesType{7, 3}));
        EXPECT_EQ(resource.blocksForLayer(0, 1, "linear"), (BlockIndicesType{3, 7}));
        EXPECT_EQ(resource.blocksForLayer(1, 0, "full"), (BlockIndicesType{2, 5}));
        resource.swapBlocks(1, "full", 0, 1);
        EXPECT_EQ(resource.kernelBlocksForLayer(1, 0, "full"), (BlockIndicesType{20, 21, 22, 23, 8, 9, 10, 11}));
        EXPECT_ANY_THROW(resource.swapBlocks(1, "missing", 0, 1));
        EXPECT_ANY_THROW(resource.swapBlocks(2, "linear", 0, 1));
        EXPECT_ANY_THROW(resource.blocksForLayer(1, 0, "linear"));
    }
}

TEST(KVCacheResourceTest, OrderedTagsAndAggregateCountsUseOwnedRows) {
    KVCacheResource resource;
    EXPECT_TRUE(resource.groupTags().empty());
    EXPECT_EQ(resource.maxBlocksNum(), 0);
    EXPECT_EQ(resource.firstNonEmptyBlocksNum(), 0u);
    resource.initGroups(CacheTopology::create(
        {makeResourceGroup("linear", CacheGroupType::LINEAR), makeResourceGroup("full", CacheGroupType::FULL)},
        {{0, {"full"}}, {1, {"linear"}}}));
    EXPECT_EQ(resource.groupTags(), (std::vector<std::string>{"linear", "full"}));
    resource.mutableBlockIds("full").assign({2, NULL_BLOCK_IDX, 4});
    EXPECT_EQ(resource.firstNonEmptyBlocksNum(), 3u);
    resource.mutableBlockIds("linear").assign({7});
    EXPECT_EQ(resource.firstNonEmptyBlocksNum(), 1u);
    EXPECT_EQ(resource.maxBlocksNum(), 3);
    EXPECT_EQ(resource.groupBlockIds("full"), resource.layerBlocks()[0]);
    EXPECT_ANY_THROW(resource.blocks("missing"));
    EXPECT_ANY_THROW(resource.mutableBlockIdsForLayer(1, "full"));
    EXPECT_ANY_THROW(resource.kernelBlocksForLayer(-1, "full"));
}

TEST(BatchKVCacheResourceTest, CheckMatchesTagsAcrossMixedResourceOrders) {
    BatchKVCacheResource batches;
    batches.resetBatchSize(2);
    for (int batch = 0; batch < 2; ++batch) {
        std::vector<GroupBase> groups{makeResourceGroup("full", CacheGroupType::FULL),
                                      makeResourceGroup("linear", CacheGroupType::LINEAR)};
        if (batch == 1) {
            std::reverse(groups.begin(), groups.end());
        }
        batches.cacheResource(batch).initGroups(CacheTopology::create(std::move(groups), {{0, {"full", "linear"}}}));
        batches.setBatchBlocks(batch, "full", {2, 3});
        batches.setBatchBlocks(batch, "linear", {7});
    }
    EXPECT_NO_THROW(batches.check());
    EXPECT_EQ(batches.curBlocksNum(), 2);
    EXPECT_EQ(batches.getAllBatchBlocks("full"), (std::vector<BlockIndicesType>{{2, 3}, {2, 3}}));
    batches.setBatchBlocks(1, "linear", {7, 8, 9});
    EXPECT_ANY_THROW(batches.check());
    EXPECT_EQ(batches.curBlocksNum(), 2);

    batches.cacheResource(1).initGroups(CacheTopology::create(
        {makeResourceGroup("other", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{0, {"other", "linear"}}}));
    batches.setBatchBlocks(1, "other", {2, 3});
    batches.setBatchBlocks(1, "linear", {7});
    EXPECT_ANY_THROW(batches.check());
    batches.cacheResource(1).initGroups(makeTestCacheTopology(1, 1, {{0}}));
    EXPECT_ANY_THROW(batches.check());
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
    EXPECT_EQ(multi_group_layer_blocks[0], resource.groupBlockIds("group0"));
    EXPECT_EQ(multi_group_layer_blocks[1], resource.groupBlockIds("group1"));
    EXPECT_EQ(multi_group_layer_blocks[2], resource.groupBlockIds("group0"));
    ASSERT_EQ(resource.layerNum(), 3);

    KVCacheResource single_group_resource;
    single_group_resource.initGroups(makeTestCacheTopology(/*group_num=*/1,
                                                           /*layer_num=*/3,
                                                           /*layer_group_ids=*/{{0}, {0}, {0}},
                                                           /*kernel_blocks_per_kv_block=*/4,
                                                           /*group_types=*/{CacheGroupType::FULL}));
    auto layer_blocks = single_group_resource.layerBlocks();
    ASSERT_EQ(layer_blocks.size(), 3u);
    ASSERT_EQ(layer_blocks[0], single_group_resource.groupBlockIds("group0"));
    ASSERT_EQ(layer_blocks[1], single_group_resource.groupBlockIds("group0"));
    ASSERT_EQ(layer_blocks[2], single_group_resource.groupBlockIds("group0"));

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

TEST(CacheTopologyTest, GroupDerivesKernelBlocksFromSpec) {
    auto spec                       = makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, 8, "full");
    spec->kernel_seq_size_per_block = 2;
    GroupBase group;
    group.tag    = "full";
    group.spec   = std::move(spec);
    group.policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    ASSERT_EQ(group.kernelBlocksPerKvBlock(), 4u);
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
    moved.mutableBlockIds("group0").add(BlockIndicesType{3});
    batch.moveBatchResource(0, std::move(moved));
    ASSERT_EQ(batch.cacheResource(0).groupNums(), 1);
    ASSERT_EQ(batch.cacheResource(0).kernelBlocks("group0"), (BlockIndicesType{6, 7}));
}

}  // namespace test
}  // namespace rtp_llm
