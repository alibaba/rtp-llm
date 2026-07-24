#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::makeTestDevicePool;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::makeDiskPool;
using block_transfer_engine_test::makeHostPool;
using block_transfer_engine_test::TempDirGuard;
using block_transfer_engine_test::TestGroupSpec;

TestGroupSpec groupSpec(const std::string& tag, std::vector<int> layer_ids, bool reusable = true) {
    TestGroupSpec spec;
    spec.tag                        = tag;
    spec.policy                     = defaultCacheGroupPolicy(CacheGroupType::FULL);
    spec.policy.enable_prefix_reuse = reusable;
    spec.layer_ids                  = std::move(layer_ids);
    spec.kv_block_stride_bytes      = 64;
    spec.kv_scale_stride_bytes      = 16;
    return spec;
}

TEST(GroupSetTest, StoresOrderedTopologyMembershipAndLogicalPayload) {
    const auto topology = makeTestTopology({groupSpec("a", {0, 1}), groupSpec("b", {0})});
    const auto pool_a   = makeTestDevicePool({{80, 16}, {96, 24}}, 4, "group_set_a");
    const auto pool_b   = makeTestDevicePool({{128, 32}}, 4, "group_set_b");

    const auto group_set = makeTestGroupSet(3, topology, {1, 0}, {pool_b, pool_a});

    EXPECT_EQ(group_set->groupSetId(), 3u);
    EXPECT_EQ(group_set->topology(), topology);
    EXPECT_EQ(group_set->groupIds(), (std::vector<size_t>{1, 0}));
    EXPECT_EQ(group_set->groupAt(0).tag, "b");
    EXPECT_EQ(group_set->groupAt(1).tag, "a");
    EXPECT_EQ(group_set->groupTags(), (std::vector<std::string>{"b", "a"}));
    EXPECT_EQ(group_set->devicePools(), (std::vector<DeviceBlockPoolPtr>{pool_b, pool_a}));
    EXPECT_EQ(group_set->payloadBytes(), 3u * (64u + 16u));
    EXPECT_EQ(group_set->groupType(), CacheGroupType::FULL);
    EXPECT_EQ(group_set->evictPolicy(), CacheEvictPolicy::CHAIN);
}

TEST(GroupSetTest, RejectsDuplicateMembership) {
    const auto topology = makeTestTopology({groupSpec("a", {0})});
    const auto pool_a   = makeTestDevicePool({{64, 16}}, 4, "group_set_duplicate_a");
    const auto pool_b   = makeTestDevicePool({{64, 16}}, 4, "group_set_duplicate_b");
    auto       group    = std::make_shared<FullGroupSet>();
    EXPECT_THROW(group->initialize(0, topology, {0, 0}, {pool_a, pool_b}), std::runtime_error);
}

TEST(GroupSetTest, RejectsEmptyOrOutOfRangeMembership) {
    const auto topology = makeTestTopology({groupSpec("a", {0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_membership");

    auto empty_group = std::make_shared<FullGroupSet>();
    EXPECT_THROW(empty_group->initialize(0, topology, {}, {}), std::runtime_error);

    auto out_of_range_group = std::make_shared<FullGroupSet>();
    EXPECT_THROW(out_of_range_group->initialize(0, topology, {1}, {pool}), std::runtime_error);
}

TEST(GroupSetTest, RejectsNullTopology) {
    const auto pool = makeTestDevicePool({{64, 16}}, 4, "group_set_null_topology");
    auto       group = std::make_shared<FullGroupSet>();

    EXPECT_THROW(group->initialize(0, nullptr, {0}, {pool}), std::runtime_error);
}

TEST(GroupSetTest, RejectsNullPoolAndInvalidLogicalPayload) {
    const auto topology = makeTestTopology({groupSpec("a", {0})});
    auto       null_pool_group = std::make_shared<FullGroupSet>();
    EXPECT_THROW(null_pool_group->initialize(0, topology, {0}, {nullptr}), std::runtime_error);

    auto zero_payload_spec                  = groupSpec("zero", {0});
    zero_payload_spec.kv_block_stride_bytes = 0;
    zero_payload_spec.kv_scale_stride_bytes = 0;
    const auto zero_topology                = makeTestTopology({std::move(zero_payload_spec)});
    const auto pool                         = makeTestDevicePool({{64, 16}}, 4, "group_set_zero_payload");
    auto       zero_payload_group           = std::make_shared<FullGroupSet>();
    EXPECT_THROW(zero_payload_group->initialize(0, zero_topology, {0}, {pool}), std::runtime_error);

    auto overflow_spec                  = groupSpec("overflow", {0});
    overflow_spec.kv_block_stride_bytes = std::numeric_limits<size_t>::max();
    overflow_spec.kv_scale_stride_bytes = 1;
    const auto overflow_topology        = makeTestTopology({std::move(overflow_spec)});
    auto       overflow_group           = std::make_shared<FullGroupSet>();
    EXPECT_THROW(overflow_group->initialize(0, overflow_topology, {0}, {pool}), std::runtime_error);
}

TEST(GroupSetTest, RejectsNonReusableMembership) {
    const auto topology = makeTestTopology({groupSpec("a", {0}, false)});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_non_reusable");
    auto       group    = std::make_shared<FullGroupSet>();
    EXPECT_THROW(group->initialize(0, topology, {0}, {pool}), std::runtime_error);
}

TEST(GroupSetTest, RejectsIncompatibleMemberPolicy) {
    auto first  = groupSpec("a", {0});
    auto second = groupSpec("b", {0});
    second.policy.reservable = false;
    const auto topology = makeTestTopology({std::move(first), std::move(second)});
    const auto pool_a   = makeTestDevicePool({{64, 16}}, 4, "group_set_policy_a");
    const auto pool_b   = makeTestDevicePool({{64, 16}}, 4, "group_set_policy_b");
    auto       group    = std::make_shared<FullGroupSet>();
    EXPECT_THROW(group->initialize(0, topology, {0, 1}, {pool_a, pool_b}), std::runtime_error);
}

TEST(GroupSetTest, MembershipIsImmutableAfterInitialization) {
    const auto topology = makeTestTopology({groupSpec("a", {0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_immutable");
    auto       group    = makeTestGroupSet(0, topology, {0}, {pool});

    EXPECT_THROW(group->initialize(0, topology, {0}, {pool}), std::runtime_error);
}

TEST(GroupSetTest, RejectsMisalignedResourceTreeNodes) {
    const auto topology = makeTestTopology({groupSpec("a", {0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_resource_alignment");
    auto       group    = makeTestGroupSet(0, topology, {0}, {pool});

    MultiNodeResource resource{0, Tier::DEVICE, {{NULL_BLOCK_IDX}}};
    resource.tree_nodes = {nullptr, nullptr};
    EXPECT_THROW(group->unreferenceBlocks(resource, BlockRefType::REQUEST), std::runtime_error);
}

TEST(GroupSetTest, LowerTierAllocationFailureReleasesAllocatedPrefix) {
    const auto topology = makeTestTopology({groupSpec("a", {0})});
    const auto pool     = makeTestDevicePool({{64, 16}}, 4, "group_set_lower_tier_rollback");
    auto       group    = makeTestGroupSet(0, topology, {0}, {pool});

    auto host_pool = makeHostPool(/*payload_bytes=*/80, /*usable_count=*/1, /*enable_pinned=*/false);
    ASSERT_NE(host_pool, nullptr);
    group->setHostPool(host_pool);

    const MultiNodeResource host_result = group->allocateBlocks(Tier::HOST, 2, BlockRefType::REQUEST);
    EXPECT_TRUE(host_result.per_node.empty());
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(host_pool->totalRefCount(BlockRefType::REQUEST), 0u);

    TempDirGuard temp_dir("group_set_lower_tier_rollback");
    auto disk_pool = makeDiskPool(/*payload_bytes=*/80, /*usable_count=*/1, temp_dir.path);
    ASSERT_NE(disk_pool, nullptr);
    group->setDiskPool(disk_pool);

    const MultiNodeResource disk_result = group->allocateBlocks(Tier::DISK, 2, BlockRefType::REQUEST);
    EXPECT_TRUE(disk_result.per_node.empty());
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(disk_pool->totalRefCount(BlockRefType::REQUEST), 0u);
}

TEST(GroupSetTest, KeepsTopologyAliveAfterCallerReleasesOwnership) {
    auto topology = makeTestTopology({groupSpec("a", {0})});
    std::weak_ptr<const CacheTopology> weak_topology = topology;
    const auto pool = makeTestDevicePool({{64, 16}}, 4, "group_set_topology_lifetime");
    auto       group = makeTestGroupSet(0, topology, {0}, {pool});

    topology.reset();

    EXPECT_FALSE(weak_topology.expired());
    EXPECT_EQ(group->groupAt(0).tag, "a");
}

}  // namespace
}  // namespace rtp_llm
