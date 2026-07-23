#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

GroupBase makeGroup(std::string tag, std::vector<int> layer_ids, CacheGroupType type = CacheGroupType::FULL) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = tag;
    spec->seq_size_per_block = 8;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(type);
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = 16;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = type == CacheGroupType::FULL ? 2 : 8;
    return group;
}

TEST(CacheTopologyTest, SupportsSingleGlobalGroupAsNEqualsOne) {
    auto topology = CacheTopology::create({makeGroup("full", {0, 1})}, {{0, {"full"}}, {1, {"full"}}});

    EXPECT_TRUE(topology->hasSingleGlobalGroup());
    EXPECT_TRUE(topology->hasOneGroupPerLayer());
    EXPECT_EQ(topology->soleGroupForLayer(0).tag, "full");
    EXPECT_EQ(topology->groupsForLayer(1).front().get().tag, "full");
}

TEST(CacheTopologyTest, SupportsDistinctOneToOneGroupsAndOneToManyLayers) {
    auto topology =
        CacheTopology::create({makeGroup("full", {0, 2}), makeGroup("linear", {1, 2}, CacheGroupType::LINEAR)},
                              {{0, {"full"}}, {1, {"linear"}}, {2, {"full", "linear"}}});

    EXPECT_FALSE(topology->hasSingleGlobalGroup());
    EXPECT_FALSE(topology->hasOneGroupPerLayer());
    EXPECT_EQ(topology->groupForLayer(2, "linear").policy.group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(topology->groupsForLayer(2).size(), 2u);
    EXPECT_ANY_THROW(topology->soleGroupForLayer(2));
}

TEST(CacheTopologyTest, OrderedSnapshotsAreLazyStableAndReadOnly) {
    auto topology = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});

    const auto& tags_first  = topology->groupTagsSnapshot();
    const auto& tags_second = topology->groupTagsSnapshot();
    const auto& spec_types  = topology->groupSpecTypesSnapshot();
    EXPECT_EQ(&tags_first, &tags_second);
    EXPECT_EQ(tags_first, (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(spec_types,
              (std::vector<KVCacheSpecType>{KVCacheSpecType::MultiHeadAttention, KVCacheSpecType::MultiHeadAttention}));
    EXPECT_EQ(topology->layer(0).group_tags, (std::vector<std::string>{"full", "linear"}));
}

TEST(CacheTopologyTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first    = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});
    auto reversed = CacheTopology::create({makeGroup("linear", {0}, CacheGroupType::LINEAR), makeGroup("full", {0})},
                                          {{0, {"full", "linear"}}});

    EXPECT_EQ(first->groupIndex("full"), 0u);
    EXPECT_EQ(first->groupIndex("linear"), 1u);
    EXPECT_EQ(reversed->groupIndex("linear"), 0u);
    EXPECT_EQ(reversed->groupIndex("full"), 1u);
    EXPECT_EQ(first->group("full").policy.group_type, reversed->group("full").policy.group_type);
    EXPECT_EQ(first->group("linear").policy.group_type, reversed->group("linear").policy.group_type);
    EXPECT_EQ(first->groupForLayer(0, "full").tag, reversed->groupForLayer(0, "full").tag);
    EXPECT_EQ(first->groupForLayer(0, "linear").tag, reversed->groupForLayer(0, "linear").tag);
}

TEST(CacheTopologyTest, RejectsInconsistentReverseMembership) {
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full", {0})}, {{0, {"full"}}, {1, {"full"}}}));
}

}  // namespace
}  // namespace rtp_llm
