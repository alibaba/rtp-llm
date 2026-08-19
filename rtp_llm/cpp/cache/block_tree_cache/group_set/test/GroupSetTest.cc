#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

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

}  // namespace
}  // namespace rtp_llm
