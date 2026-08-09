#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

GroupBase makeGroup(std::string tag, std::vector<int> layer_ids, CacheGroupType type = CacheGroupType::FULL) {
    auto spec = std::make_shared<MHAKVCacheSpec>(8, type == CacheGroupType::FULL ? 2 : 8);
    spec->tag = tag;

    GroupBase group;
    group.tag       = std::move(tag);
    group.spec      = std::move(spec);
    group.policy    = defaultCacheGroupPolicy(type);
    group.layer_ids = std::move(layer_ids);
    return group;
}

TEST(CacheTopologyTest, SupportsSingleGlobalGroupAsNEqualsOne) {
    auto topology = CacheTopology::create({makeGroup("full", {0, 1})}, {{0, {"full"}}, {1, {"full"}}});

    EXPECT_TRUE(topology->hasSingleGlobalGroup());
    const auto groups = topology->groupsForLayer(0);
    ASSERT_EQ(groups.size(), 1u);
    for (const auto& group : groups) {
        EXPECT_EQ(group.get().tag, "full");
    }
}

TEST(CacheTopologyTest, SupportsDistinctOneToOneGroupsAndOneToManyLayers) {
    auto topology =
        CacheTopology::create({makeGroup("full", {0, 2}), makeGroup("linear", {1, 2}, CacheGroupType::LINEAR)},
                              {{0, {"full"}}, {1, {"linear"}}, {2, {"full", "linear"}}});

    EXPECT_FALSE(topology->hasSingleGlobalGroup());
    EXPECT_EQ(topology->groupForLayer(2, "linear").policy.group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(topology->groupsForLayer(2).size(), 2u);
}

TEST(CacheTopologyTest, IterationEntriesAreSelfDescribingAndUnknownTagsFail) {
    auto topology = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});

    std::vector<std::string> tags;
    for (const auto& group : topology->groups()) {
        tags.push_back(group.tag);
        EXPECT_EQ(group.spec->tag, group.tag);
    }
    EXPECT_EQ(tags, (std::vector<std::string>{"full", "linear"}));
    EXPECT_ANY_THROW(topology->group("missing"));
    EXPECT_ANY_THROW(topology->groupForLayer(0, "missing"));
}

TEST(CacheTopologyTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first    = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});
    auto reversed = CacheTopology::create({makeGroup("linear", {0}, CacheGroupType::LINEAR), makeGroup("full", {0})},
                                          {{0, {"full", "linear"}}});

    EXPECT_EQ(first->group("full").policy.group_type, reversed->group("full").policy.group_type);
    EXPECT_EQ(first->group("linear").policy.group_type, reversed->group("linear").policy.group_type);
    EXPECT_EQ(first->groupForLayer(0, "full").tag, reversed->groupForLayer(0, "full").tag);
    EXPECT_EQ(first->groupForLayer(0, "linear").tag, reversed->groupForLayer(0, "linear").tag);
}

TEST(CacheTopologyTest, RejectsDuplicateAndEmptyTags) {
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full", {0}), makeGroup("full", {0})}, {{0, {"full"}}}));
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("", {0})}, {{0, {""}}}));
}

TEST(CacheTopologyTest, RejectsLayerWithoutAGroup) {
    try {
        CacheTopology::create({makeGroup("full", {1})}, {{0, {}}, {1, {"full"}}});
        FAIL() << "expected empty layer group membership to be rejected";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("layer_id=0"), std::string::npos);
    }
}

TEST(CacheTopologyTest, RejectsInconsistentReverseMembership) {
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full", {0})}, {{0, {"full"}}, {1, {"full"}}}));
}

TEST(CacheTopologyTest, RejectsZeroKernelBlockSize) {
    EXPECT_ANY_THROW(std::make_shared<MHAKVCacheSpec>(8, 0));
}

}  // namespace
}  // namespace rtp_llm
