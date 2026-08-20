#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestDevicePool;
using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;

GroupBase makeGroupBase(std::vector<int> layer_ids, bool reusable = true) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = reusable;
    return makeTestGroupBase(std::move(policy), std::move(layer_ids), 64, 16);
}

MultiNodeResource makeTwoNodeDeviceResource(BlockIdxType first, BlockIdxType second) {
    return MultiNodeResource{0, Tier::DEVICE, {{nullptr, {first}}, {nullptr, {second}}}};
}

TEST(GroupSetTest, StoresOrderedTopologyMembershipAndLogicalPayload) {
    const auto topology = makeTestTopology({makeGroupBase({0, 1}), makeGroupBase({0})});
    const auto pool_a   = makeTestDevicePool({{80, 16}, {96, 24}}, 4, "group_set_a");
    const auto pool_b   = makeTestDevicePool({{128, 32}}, 4, "group_set_b");

    const auto group_set = makeTestGroupSet(3, topology, {1, 0}, {pool_b, pool_a});

    EXPECT_EQ(group_set->groupSetId(), 3u);
    EXPECT_EQ(group_set->groupIds(), (std::vector<size_t>{1, 0}));
    EXPECT_EQ(&group_set->groupAt(0), &topology->groupById(1));
    EXPECT_EQ(&group_set->groupAt(1), &topology->groupById(0));
    EXPECT_EQ(group_set->devicePools(), (std::vector<DeviceBlockPoolPtr>{pool_b, pool_a}));
    EXPECT_EQ(group_set->payloadBytes(), 3u * (64u + 16u));
    EXPECT_EQ(group_set->groupType(), CacheGroupType::FULL);
}

TEST(GroupSetTest, KeepsTopologyAliveAfterCallerReleasesOwnership) {
    auto                               topology      = makeTestTopology({makeGroupBase({0})});
    std::weak_ptr<const CacheTopology> weak_topology = topology;
    const auto                         pool          = makeTestDevicePool({{64, 16}}, 4, "group_set_topology_lifetime");
    auto                               group         = makeTestGroupSet(0, topology, {0}, {pool});
    const GroupBase*                   expected_group = &topology->groupById(0);

    topology.reset();

    EXPECT_FALSE(weak_topology.expired());
    EXPECT_EQ(&group->groupAt(0), expected_group);
}

TEST(GroupSetTest, ForwardsTreeReferencesToDevicePools) {
    const auto topology = makeTestTopology({makeGroupBase({0})});
    const auto pool_a   = makeTestDevicePool({{64, 16}}, 4, "group_set_tree_ref_a");
    const auto pool_b   = makeTestDevicePool({{64, 16}}, 4, "group_set_tree_ref_b");
    const auto group    = makeTestGroupSet(0, topology, {0}, {pool_a, pool_b});
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

TEST(GroupSetTest, BatchOuterReferenceRejectsInvalidTailWithoutMutatingPrefix) {
    const auto topology = makeTestTopology({makeGroupBase({0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_batch_outer_inc");
    const auto group    = makeTestGroupSet(0, topology, {0}, {pool});
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
    const auto group    = makeTestGroupSet(0, topology, {0}, {pool});
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
    const auto group    = makeTestGroupSet(0, topology, {0}, {pool});
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
    const auto group    = makeTestGroupSet(0, topology, {0}, {pool});
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
