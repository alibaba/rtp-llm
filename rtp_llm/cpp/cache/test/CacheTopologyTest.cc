#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

GroupBase makeGroup(std::string tag, CacheGroupType type = CacheGroupType::FULL) {
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

TEST(CacheTopologyTest, SupportsSingleGlobalGroupAsNEqualsOne) {
    auto topology = CacheTopology::create({makeGroup("full")}, {{0, {"full"}}, {1, {"full"}}});

    EXPECT_TRUE(topology->hasSingleGlobalGroup());
    EXPECT_TRUE(topology->hasOneGroupPerLayer());
    EXPECT_EQ(topology->soleGroupForLayer(0).tag, "full");
    EXPECT_EQ(topology->groupsForLayer(1).front().get().tag, "full");
}

TEST(CacheTopologyTest, SpecConstructorValidatesGeometryAndHeadCount) {
    MHAKVCacheSpec spec("full", 8, 2, 4);
    EXPECT_EQ(spec.tag, "full");
    EXPECT_EQ(spec.seq_size_per_block, 8u);         // tokens/physical block
    EXPECT_EQ(spec.kernel_seq_size_per_block, 2u);  // tokens/kernel page
    EXPECT_EQ(spec.local_kv_head_num, 4u);          // heads/rank
    EXPECT_ANY_THROW(MHAKVCacheSpec("full", 0, 2, 4));
    EXPECT_ANY_THROW(MHAKVCacheSpec("full", 8, 0, 4));
    EXPECT_ANY_THROW(MHAKVCacheSpec("full", 8, 3, 4));
    EXPECT_ANY_THROW(MHAKVCacheSpec("full", 8, 2, 0));
}

TEST(CacheTopologyTest, MaximumKernelExpansionDoesNotDependOnGroupOrder) {
    auto full     = makeGroup("full");
    auto swa      = makeGroup("swa", CacheGroupType::SWA);
    auto topology = CacheTopology::create({swa, full}, {{0, {"swa"}}, {1, {"full"}}});
    EXPECT_EQ(topology->groupById(0).kernelBlocksPerKvBlock(), 1u);
    EXPECT_EQ(topology->maxKernelBlocksPerKvBlock(), 4u);
    auto reversed = CacheTopology::create({full, swa}, {{0, {"swa"}}, {1, {"full"}}});
    EXPECT_EQ(reversed->maxKernelBlocksPerKvBlock(), 4u);
}

TEST(CacheTopologyTest, SupportsDistinctOneToOneGroupsAndOneToManyLayers) {
    auto topology = CacheTopology::create({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)},
                                          {{0, {"full"}}, {1, {"linear"}}, {2, {"full", "linear"}}});

    EXPECT_FALSE(topology->hasSingleGlobalGroup());
    EXPECT_FALSE(topology->hasOneGroupPerLayer());
    EXPECT_EQ(topology->groupForLayer(2, "linear").policy.group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(topology->groupsForLayer(2).size(), 2u);
    EXPECT_ANY_THROW(topology->soleGroupForLayer(2));
}

TEST(CacheTopologyTest, CompatibilitySnapshotsAreIndependentValues) {
    auto topology = CacheTopology::create({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});

    auto        tags_first  = topology->groupTagsSnapshot();
    const auto& tags_second = topology->groupTagsSnapshot();
    EXPECT_EQ(tags_first, (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(topology->groupTypesSnapshot(),
              (std::vector<CacheGroupType>{CacheGroupType::FULL, CacheGroupType::LINEAR}));
    EXPECT_EQ(topology->layerGroupIdsSnapshot(), (std::vector<std::vector<int>>{{0, 1}}));
    tags_first.clear();
    EXPECT_EQ(tags_second.size(), 2u);
    EXPECT_EQ(topology->groupTagsSnapshot().size(), 2u);
}

TEST(CacheTopologyTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first    = CacheTopology::create({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)},
                                       {{0, {"full", "linear"}}});
    auto reversed = CacheTopology::create({makeGroup("linear", CacheGroupType::LINEAR), makeGroup("full")},
                                          {{0, {"full", "linear"}}});

    EXPECT_NE(first->groupIdForTag("full"), reversed->groupIdForTag("full"));
    EXPECT_EQ(first->groupIdForTag("full"), 0u);
    EXPECT_EQ(reversed->groupIdForTag("full"), 1u);
    EXPECT_EQ(&first->group("full"), &first->groupById(first->groupIdForTag("full")));
    EXPECT_ANY_THROW(first->groupIdForTag("missing"));
    EXPECT_EQ(first->group("full").policy.group_type, reversed->group("full").policy.group_type);
    EXPECT_EQ(first->group("linear").policy.group_type, reversed->group("linear").policy.group_type);
    EXPECT_EQ(first->groupForLayer(0, "full").tag, reversed->groupForLayer(0, "full").tag);
    EXPECT_EQ(first->groupForLayer(0, "linear").tag, reversed->groupForLayer(0, "linear").tag);
}

TEST(CacheTopologyTest, DerivesReverseMembershipFromLayers) {
    auto topology = CacheTopology::create({makeGroup("full")}, {{0, {"full"}}, {1, {"full"}}});
    EXPECT_EQ(topology->layerIdsForGroup(0), (std::vector<int>{0, 1}));
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full")}, {{0, {"missing"}}}));
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full")}, {{0, {"full", "full"}}}));
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full"), makeGroup("full")}, {{0, {"full"}}}));
}

}  // namespace
}  // namespace rtp_llm
