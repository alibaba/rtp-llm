#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestDevicePool;
using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::makeTestCacheConfig;
using block_transfer_engine_test::makeHostPool;

block_transfer_engine_test::TestGroupConfig makeGroupBase(std::vector<int> layer_ids, bool reusable = true) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = reusable;
    return makeTestGroupBase(std::move(policy), std::move(layer_ids), 64, 16);
}

MultiNodeResource makeTwoNodeDeviceResource(BlockIdxType first, BlockIdxType second) {
    return MultiNodeResource{0, Tier::DEVICE, {{nullptr, {first}}, {nullptr, {second}}}};
}

CacheModuleConfig makeModuleConfig(block_transfer_engine_test::TestGroupConfig group_config, std::string tag) {
    const auto topology = makeTestTopology({std::move(group_config)});
    auto       groups   = topology->groups();
    auto       layers   = topology->layers();
    const auto old_tag  = groups.front().tag;
    auto       spec     = groups.front().spec->clone();
    spec->tag           = tag;
    groups.front().tag  = tag;
    groups.front().spec = std::move(spec);
    for (auto& layer : layers) {
        for (auto& layer_tag : layer.group_tags) {
            if (layer_tag == old_tag) {
                layer_tag = tag;
            }
        }
    }
    CacheModuleConfig module;
    module.setTopology(std::move(groups), std::move(layers));
    return module;
}

std::shared_ptr<const CacheConfig> makeTwoModuleConfig(CacheGroupPolicy main_policy,
                                                       CacheGroupPolicy draft_policy,
                                                       size_t           main_kv_bytes  = 64,
                                                       size_t           draft_kv_bytes = 128) {
    CacheConfig config;
    config.main_config =
        makeModuleConfig(makeTestGroupBase(std::move(main_policy), {0}, main_kv_bytes, 16, 64, 16), "main");
    config.mtp_sub_configs.push_back(
        makeModuleConfig(makeTestGroupBase(std::move(draft_policy), {0}, draft_kv_bytes, 32, 64, 16), "draft"));
    return std::make_shared<CacheConfig>(std::move(config));
}

TEST(GroupSetTest, StoresOrderedTagMembershipAndLogicalPayload) {
    const auto topology = makeTestTopology({makeGroupBase({0, 1}), makeGroupBase({0})});
    const auto pool_a   = makeTestDevicePool({{80, 16}, {96, 24}}, 4, "group_set_a");
    const auto pool_b   = makeTestDevicePool({{128, 32}}, 4, "group_set_b");

    const auto group_set = makeTestGroupSet(3, topology, {"group1", "group0"}, {pool_b, pool_a});

    EXPECT_EQ(group_set->groupSetId(), 3u);
    EXPECT_EQ(group_set->groupTags(), (std::vector<std::string>{"group1", "group0"}));
    EXPECT_EQ(group_set->group("group1").tag, topology->group("group1").tag);
    EXPECT_EQ(group_set->group("group0").tag, topology->group("group0").tag);
    EXPECT_EQ(group_set->devicePools(), (std::vector<DeviceBlockPoolPtr>{pool_b, pool_a}));
    EXPECT_EQ(group_set->payloadBytes(), 3u * (64u + 16u));
    EXPECT_EQ(group_set->groupType(), CacheGroupType::FULL);
}

TEST(GroupSetTest, ConfigMembersShareOneTreeWithOrderedIndependentPoolsAfterSetup) {
    auto       config     = makeTwoModuleConfig(defaultCacheGroupPolicy(CacheGroupType::FULL),
                                      defaultCacheGroupPolicy(CacheGroupType::FULL));
    const auto main_pool  = makeTestDevicePool({{64, 16}}, 4, "directory_main");
    const auto draft_pool = makeTestDevicePool({{128, 32}}, 4, "directory_draft");
    const auto host_pool  = makeHostPool(240, 2);
    auto       group_set =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{draft_pool, main_pool}, host_pool, nullptr);

    group_set->initialize(7, config, {"draft", "main"}, BackingLayout{240, 240, 240, false});
    const auto*                      draft_spec  = config->groupByTagGlobal("draft").spec.get();
    std::weak_ptr<const CacheConfig> weak_config = config;
    config.reset();

    ASSERT_FALSE(weak_config.expired());

    EXPECT_EQ(group_set->groupSetId(), 7u);
    EXPECT_EQ(group_set->groupTags(), (std::vector<std::string>{"draft", "main"}));
    EXPECT_EQ(group_set->devicePools(), (std::vector<DeviceBlockPoolPtr>{draft_pool, main_pool}));
    EXPECT_EQ(group_set->localLayerIdsForMember("draft"), (std::vector<int>{0}));
    EXPECT_EQ(group_set->localLayerIdsForMember("main"), (std::vector<int>{0}));
    EXPECT_EQ(&group_set->groupSpec("draft"), draft_spec);
    EXPECT_EQ(group_set->payloadBytes(), 240u);
    EXPECT_EQ(group_set->storageBytes(), 240u);
    EXPECT_EQ(host_pool->payloadBytes(), 240u);

    BlockTree   tree({group_set});
    const auto* draft_location = tree.reusableGroupLocation("draft");
    const auto* main_location  = tree.reusableGroupLocation("main");
    ASSERT_NE(draft_location, nullptr);
    ASSERT_NE(main_location, nullptr);
    EXPECT_EQ(draft_location->group_set_id, 0u);
    EXPECT_EQ(draft_location->member_group_id, 0u);
    EXPECT_EQ(main_location->group_set_id, 0u);
    EXPECT_EQ(main_location->member_group_id, 1u);

    const auto draft_block = draft_pool->malloc();
    const auto main_block  = main_pool->malloc();
    ASSERT_TRUE(draft_block.has_value());
    ASSERT_TRUE(main_block.has_value());
    GroupSetResource resource;
    resource.device_blocks = {*draft_block, *main_block};
    const auto insert      = tree.insertNode({101}, {{resource}}, true, false);
    ASSERT_EQ(insert.path.size(), 1u);
    EXPECT_EQ(insert.accepted_resource_count, 1u);
    EXPECT_EQ(draft_pool->treeRefCount(*draft_block), 1u);
    EXPECT_EQ(main_pool->treeRefCount(*main_block), 1u);
    auto validator = group_set->createMatchValidator();
    EXPECT_TRUE(validator->validate(insert.path.front()->group_set_resources.front()));

    insert.path.front()->group_set_resources.front().evictFromTier(Tier::DEVICE);
    group_set->unreferenceBlocks(
        MultiNodeResource{0, Tier::DEVICE, {{insert.path.front(), {*draft_block, *main_block}}}},
        BlockTreeRefType::CACHE);
    EXPECT_FALSE(draft_pool->isAllocated(*draft_block));
    EXPECT_FALSE(main_pool->isAllocated(*main_block));

    const auto host_blocks = group_set->allocateBlocks(2, Tier::HOST, BlockTreeRefType::EVICTION);
    ASSERT_TRUE(host_blocks.has_value());
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 2u);
    group_set->releaseBlocks(Tier::HOST, *host_blocks, BlockTreeRefType::EVICTION);
    EXPECT_EQ(host_pool->freeBlocksNum(), 2u);
}

TEST(GroupSetTest, ConfigInitializationRejectsIncompatibleOrDuplicateMembersAtomically) {
    const auto expect_rejected = [](CacheGroupPolicy main_policy, CacheGroupPolicy draft_policy) {
        const auto  config     = makeTwoModuleConfig(std::move(main_policy), std::move(draft_policy));
        const auto  main_pool  = makeTestDevicePool({{64, 16}}, 4, "reject_main");
        const auto  draft_pool = makeTestDevicePool({{128, 32}}, 4, "reject_draft");
        GroupSetPtr group_set;
        switch (config->groupByTagGlobal("main").policy.group_type) {
            case CacheGroupType::FULL:
                group_set = std::make_shared<FullGroupSet>(
                    std::vector<DeviceBlockPoolPtr>{main_pool, draft_pool}, nullptr, nullptr);
                break;
            case CacheGroupType::SWA:
                group_set = std::make_shared<SWAGroupSet>(
                    128, 16, std::vector<DeviceBlockPoolPtr>{main_pool, draft_pool}, nullptr, nullptr);
                break;
            case CacheGroupType::LINEAR:
                group_set = std::make_shared<LinearGroupSet>(
                    std::vector<DeviceBlockPoolPtr>{main_pool, draft_pool}, nullptr, nullptr);
                break;
        }
        ASSERT_NE(group_set, nullptr);
        const auto main_free  = main_pool->freeBlocksNum();
        const auto draft_free = draft_pool->freeBlocksNum();

        EXPECT_ANY_THROW(group_set->initialize(9, config, {"main", "draft"}));

        EXPECT_TRUE(group_set->groupTags().empty());
        EXPECT_EQ(main_pool->freeBlocksNum(), main_free);
        EXPECT_EQ(draft_pool->freeBlocksNum(), draft_free);
    };

    auto main_full       = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto draft_full      = main_full;
    main_full.cp_mapping = CpBlockMappingMode::NONE;
    expect_rejected(main_full, draft_full);

    auto main_swa                 = defaultCacheGroupPolicy(CacheGroupType::SWA);
    main_swa.sliding_window_size  = 128;
    main_swa.active_tail_blocks   = 2;
    auto draft_swa                = main_swa;
    draft_swa.sliding_window_size = 64;
    expect_rejected(main_swa, draft_swa);

    auto main_linear                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    main_linear.active_tail_blocks  = 1;
    auto draft_linear               = main_linear;
    draft_linear.active_tail_blocks = 2;
    expect_rejected(main_linear, draft_linear);

    const auto config      = makeTwoModuleConfig(defaultCacheGroupPolicy(CacheGroupType::FULL),
                                            defaultCacheGroupPolicy(CacheGroupType::FULL));
    const auto first_pool  = makeTestDevicePool({{64, 16}}, 4, "duplicate_first");
    const auto second_pool = makeTestDevicePool({{64, 16}}, 4, "duplicate_second");
    auto       duplicate_group =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{first_pool, second_pool}, nullptr, nullptr);
    EXPECT_ANY_THROW(duplicate_group->initialize(9, config, {"main", "main"}));
    EXPECT_ANY_THROW(duplicate_group->initialize(9, config, {"main", "unknown"}));
    EXPECT_TRUE(duplicate_group->groupTags().empty());
}

TEST(GroupSetTest, UsesPlannedBackingLayoutWithoutRecomputingStorage) {
    const auto topology = makeTestTopology({makeGroupBase({0, 1})});
    auto       group    = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, nullptr, nullptr);
    group->initialize(0, makeTestCacheConfig(*topology), {"group0"}, BackingLayout{4096, 4112, 8192, true});
    EXPECT_TRUE(group->usesPhysicalPayloadGeometry());
    EXPECT_TRUE(group->crcEnabled());
    EXPECT_EQ(group->payloadBytes(), 4096u);
    EXPECT_EQ(group->storageBytes(), 4112u);

    group->initialize(0, makeTestCacheConfig(*topology), {"group0"}, BackingLayout{0, 176, 4096, true});
    EXPECT_FALSE(group->usesPhysicalPayloadGeometry());
    EXPECT_TRUE(group->crcEnabled());
    EXPECT_EQ(group->payloadBytes(), 160u);
    EXPECT_EQ(group->storageBytes(), 176u);

    group->initialize(0, makeTestCacheConfig(*topology), {"group0"});
    EXPECT_FALSE(group->usesPhysicalPayloadGeometry());
    EXPECT_FALSE(group->crcEnabled());
    EXPECT_EQ(group->payloadBytes(), 160u);
    EXPECT_EQ(group->storageBytes(), 160u);

    group->initialize(0, makeTestCacheConfig(*topology), {"group0"}, BackingLayout{4096, 4096, 4096, false});
    EXPECT_TRUE(group->usesPhysicalPayloadGeometry());
    EXPECT_FALSE(group->crcEnabled());
    EXPECT_EQ(group->storageBytes(), 4096u);
}

TEST(GroupSetTest, CopyGeometryPreservesFullLogicalBlocksAndPhysicalLayers) {
    const auto topology      = makeTestTopology({makeGroupBase({0, 2}), makeGroupBase({1})});
    const auto pool          = makeTestDevicePool({{64, 16}, {64, 16}}, 4, "group_set_copy_geometry");
    auto       group         = makeTestGroupSet(0, topology, {"group0"}, {pool});
    const auto logical_tiles = getDeviceHostCopyTiles(*group, {1});
    ASSERT_EQ(logical_tiles.size(), 4u);
    const std::vector<size_t> logical_sizes{64, 16, 64, 16};
    const std::vector<size_t> logical_offsets{0, 64, 80, 144};
    for (size_t i = 0; i < logical_tiles.size(); ++i) {
        const auto& tile = logical_tiles[i];
        EXPECT_EQ(tile.buffer.size_bytes, logical_sizes[i]);
        EXPECT_EQ(tile.host_offset, logical_offsets[i]);
        EXPECT_EQ(tile.member_group_id, 0u);
        EXPECT_EQ(tile.local_layer_index, i / 2);
        EXPECT_EQ(tile.buffer.addr, pool->convertIndexToBuffer(i / 2, 1)[i % 2].addr);
    }

    const auto physical_pool  = makeTestDevicePool({{80, 16}, {96, 24}}, 4, "group_set_physical_copy_geometry");
    const auto physical_group = makeTestGroupSet(0, topology, {"group0"}, {physical_pool}, nullptr, nullptr, true, 216);
    const auto physical_tiles = getDeviceHostCopyTiles(*physical_group, {1});
    ASSERT_EQ(physical_tiles.size(), 4u);
    const std::vector<size_t> physical_sizes{80, 16, 96, 24};
    const std::vector<size_t> physical_offsets{0, 80, 96, 192};
    for (size_t i = 0; i < physical_tiles.size(); ++i) {
        const auto& tile = physical_tiles[i];
        EXPECT_EQ(tile.buffer.size_bytes, physical_sizes[i]);
        EXPECT_EQ(tile.host_offset, physical_offsets[i]);
        EXPECT_EQ(tile.member_group_id, 0u);
        EXPECT_EQ(tile.local_layer_index, i / 2);
        EXPECT_EQ(tile.buffer.addr, physical_pool->convertIndexToBuffer(i / 2, 1)[i % 2].addr);
    }
}

TEST(GroupSetTest, KeepsTopologyAliveAfterCallerReleasesOwnership) {
    auto                               topology      = makeTestTopology({makeGroupBase({0})});
    const auto                         pool          = makeTestDevicePool({{64, 16}}, 4, "group_set_topology_lifetime");
    auto                               group         = makeTestGroupSet(0, topology, {"group0"}, {pool});
    const auto group_tag = topology->groupTags().front();

    topology.reset();

    // After the CacheConfig migration, GroupSet retains a CacheConfig (which
    // owns its own internal topology) rather than the caller's topology ptr.
    ASSERT_NE(group->cacheConfig(), nullptr);
    EXPECT_NE(group->cacheConfig()->topologyPtr(), nullptr);
    EXPECT_EQ(group->groupTags(), (std::vector<std::string>{group_tag}));
}

TEST(GroupSetTest, ForwardsTreeReferencesToDevicePools) {
    const auto topology = makeTestTopology({makeGroupBase({0}), makeGroupBase({1})});
    const auto pool_a   = makeTestDevicePool({{64, 16}}, 4, "group_set_tree_ref_a");
    const auto pool_b   = makeTestDevicePool({{64, 16}}, 4, "group_set_tree_ref_b");
    const auto group    = makeTestGroupSet(0, topology, {"group0", "group1"}, {pool_a, pool_b});
    const auto block_a  = pool_a->malloc();
    const auto block_b  = pool_b->malloc();
    ASSERT_TRUE(block_a.has_value());
    ASSERT_TRUE(block_b.has_value());

    MultiNodeResource resource;
    resource.tier = Tier::DEVICE;
    resource.node_blocks.push_back({nullptr, {*block_a, *block_b}});

    group->referenceBlocks(resource, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool_a->refCount(*block_a), 1u);
    EXPECT_EQ(pool_b->refCount(*block_b), 1u);
    EXPECT_EQ(pool_a->treeRefCount(*block_a), 1u);
    EXPECT_EQ(pool_b->treeRefCount(*block_b), 1u);

    group->unreferenceBlocks(resource, BlockTreeRefType::LOAD);
    EXPECT_FALSE(pool_a->isAllocated(*block_a));
    EXPECT_FALSE(pool_b->isAllocated(*block_b));
}

TEST(GroupSetTest, RejectsUnknownDuplicateAndMisalignedTagsBeforeInitialization) {
    const auto topology = makeTestTopology({makeGroupBase({0}), makeGroupBase({1})});
    const auto pool_a   = makeTestDevicePool({{64, 16}}, 4, "group_set_validate_a");
    const auto pool_b   = makeTestDevicePool({{64, 16}}, 4, "group_set_validate_b");

    EXPECT_ANY_THROW(std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool_a}, nullptr, nullptr)
                         ->initialize(0, makeTestCacheConfig(*topology), {"missing"}));
    EXPECT_ANY_THROW(std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool_a, pool_b}, nullptr, nullptr)
                         ->initialize(0, makeTestCacheConfig(*topology), {"group0", "group0"}));
    EXPECT_ANY_THROW(std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool_a}, nullptr, nullptr)
                         ->initialize(0, makeTestCacheConfig(*topology), topology->groupTags()));
}

TEST(GroupSetTest, ReleasesLowerTierTreeReferencesInBatch) {
    const auto topology    = makeTestTopology({makeGroupBase({0})});
    const auto device_pool = makeTestDevicePool({{64, 16}}, 2, "group_set_batch_release_device");
    const auto host_pool   = makeHostPool(/*payload_bytes=*/80, /*usable_count=*/2);
    const auto group       = makeTestGroupSet(0, topology, {"group0"}, {device_pool}, host_pool);
    const auto blocks      = group->allocateBlocks(2, Tier::HOST, BlockTreeRefType::EVICTION);
    ASSERT_TRUE(blocks.has_value());
    ASSERT_EQ(blocks->size(), 2u);
    ASSERT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 2u);

    group->releaseBlocks(Tier::HOST, *blocks, BlockTreeRefType::EVICTION);

    EXPECT_EQ(host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
}

TEST(GroupSetTest, BatchOuterReferenceRejectsInvalidTailWithoutMutatingPrefix) {
    const auto topology = makeTestTopology({makeGroupBase({0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_batch_outer_inc");
    const auto group    = makeTestGroupSet(0, topology, {"group0"}, {pool});
    const auto blocks   = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    pool->incRef(blocks->back());
    pool->decRef(blocks->back());

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(group->referenceBlocks(makeTwoNodeDeviceResource(blocks->front(), blocks->back())));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

    EXPECT_EQ(pool->refCount(blocks->front()), 0u);
    EXPECT_EQ(pool->referencedBlocksNum(), 0u);
    if (pool->refCount(blocks->front()) == 0) {
        pool->incRef(blocks->front());
        pool->decRef(blocks->front());
    } else {
        pool->decRef(blocks->front());
    }
}

TEST(GroupSetTest, BatchOuterUnreferenceRejectsInvalidTailWithoutMutatingPrefix) {
    const auto topology = makeTestTopology({makeGroupBase({0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_batch_outer_dec");
    const auto group    = makeTestGroupSet(0, topology, {"group0"}, {pool});
    const auto blocks   = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    pool->incRef(blocks->front());

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(group->unreferenceBlocks(makeTwoNodeDeviceResource(blocks->front(), blocks->back())));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

    EXPECT_TRUE(pool->isAllocated(blocks->front()));
    if (pool->isAllocated(blocks->front())) {
        EXPECT_EQ(pool->refCount(blocks->front()), 1u);
        pool->decRef(blocks->front());
    }
    pool->incRef(blocks->back());
    pool->decRef(blocks->back());
}

TEST(GroupSetTest, BatchTreeReferenceRejectsInvalidTailWithoutMutatingPrefix) {
    const auto topology = makeTestTopology({makeGroupBase({0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_batch_tree_inc");
    const auto group    = makeTestGroupSet(0, topology, {"group0"}, {pool});
    const auto blocks   = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    pool->incRef(blocks->back());
    pool->decRef(blocks->back());

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(
        group->referenceBlocks(makeTwoNodeDeviceResource(blocks->front(), blocks->back()), BlockTreeRefType::CACHE));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

    EXPECT_EQ(pool->refCount(blocks->front()), 0u);
    EXPECT_EQ(pool->treeRefCount(blocks->front()), 0u);
    if (pool->treeRefCount(blocks->front()) == 0) {
        pool->incRef(blocks->front());
        pool->decRef(blocks->front());
    } else {
        pool->decTreeRef(blocks->front(), BlockTreeRefType::CACHE);
    }
}

TEST(GroupSetTest, BatchTreeUnreferenceRejectsInvalidTailWithoutMutatingPrefix) {
    const auto topology = makeTestTopology({makeGroupBase({0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_batch_tree_dec");
    const auto group    = makeTestGroupSet(0, topology, {"group0"}, {pool});
    const auto blocks   = pool->malloc(2);
    ASSERT_TRUE(blocks.has_value());
    pool->incTreeRef(blocks->front(), BlockTreeRefType::CACHE);
    pool->incTreeRef(blocks->back(), BlockTreeRefType::LOAD);

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(
        group->unreferenceBlocks(makeTwoNodeDeviceResource(blocks->front(), blocks->back()), BlockTreeRefType::CACHE));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

    EXPECT_TRUE(pool->isAllocated(blocks->front()));
    if (pool->isAllocated(blocks->front())) {
        EXPECT_EQ(pool->treeRefCount(blocks->front()), 1u);
        pool->decTreeRef(blocks->front(), BlockTreeRefType::CACHE);
    }
    pool->decTreeRef(blocks->back(), BlockTreeRefType::LOAD);
}

}  // namespace
}  // namespace rtp_llm
