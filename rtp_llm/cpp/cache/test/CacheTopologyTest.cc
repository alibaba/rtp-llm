#include <set>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

GroupBase makeGroup(std::string tag, std::vector<int> layer_ids, CacheGroupType type = CacheGroupType::FULL) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
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

CacheConfig makeConfig(std::vector<GroupBase> groups, std::vector<LayerBase> layers) {
    CacheConfig config;
    config.layer_num     = static_cast<uint32_t>(layers.size());
    config.layer_all_num = config.layer_num;
    config.setTopology(std::move(groups), std::move(layers));
    return config;
}

// Build a group with a fully explicit physical layout so tag lookup can prove
// that each group retains its own complete record.
GroupBase makeSigGroup(std::string      tag,
                       std::vector<int> layer_ids,
                       uint32_t         heads,
                       uint32_t         physical_b,
                       uint32_t         kernel_b,
                       size_t           kv_stride,
                       size_t           scale_stride,
                       KVCacheSpecType  spec_type = KVCacheSpecType::MultiHeadAttention,
                       CacheGroupType   type      = CacheGroupType::FULL) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = physical_b;
    spec->type               = spec_type;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(type);
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = 16;
    group.local_kv_head_num         = heads;
    group.seq_size_per_block        = physical_b;
    group.kernel_seq_size_per_block = kernel_b;
    group.kv_block_stride_bytes     = kv_stride;
    group.kv_scale_stride_bytes     = scale_stride;
    return group;
}

TEST(CacheTopologyTest, TwoTagsOnOneLayerCarryDistinctPhysicalRecords) {
    // Two FULL groups share layer 0 but have different heads, B, K and strides.
    // No global scalar or group-parallel projection can describe layer 0; every
    // consumer must read the exact group record by tag.
    auto config = makeConfig({makeSigGroup("a", {0}, /*heads=*/2, /*B=*/8, /*kernel=*/2, /*kv=*/64, /*scale=*/8),
                              makeSigGroup("b", {0}, /*heads=*/5, /*B=*/4, /*kernel=*/4, /*kv=*/48, /*scale=*/0)},
                             {{0, {"a", "b"}}});

    const auto& a = config.groupForLayer(0, "a");
    const auto& b = config.groupForLayer(0, "b");
    EXPECT_NE(a.local_kv_head_num, b.local_kv_head_num);
    EXPECT_NE(a.spec->seq_size_per_block, b.spec->seq_size_per_block);
    EXPECT_NE(a.kernel_seq_size_per_block, b.kernel_seq_size_per_block);
    EXPECT_NE(a.kv_block_stride_bytes, b.kv_block_stride_bytes);
    EXPECT_NE(a.kv_scale_stride_bytes, b.kv_scale_stride_bytes);

    EXPECT_EQ(a.local_kv_head_num, 2u);
    EXPECT_EQ(b.local_kv_head_num, 5u);
    EXPECT_EQ(a.kv_block_stride_bytes, 64u);
    EXPECT_EQ(b.kv_block_stride_bytes, 48u);
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

TEST(CacheTopologyTest, CacheConfigPublishesTagLookupAndLayerMembership) {
    auto config = makeConfig({makeGroup("full", {0, 2}), makeGroup("linear", {1, 2}, CacheGroupType::LINEAR)},
                             {{0, {"full"}}, {1, {"linear"}}, {2, {"full", "linear"}}});

    const std::string_view full_tag   = "full";
    const std::string_view linear_tag = "linear";
    EXPECT_EQ(config.group(full_tag).tag, "full");
    EXPECT_EQ(config.groupsForLayer(2), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(config.groupForLayer(2, linear_tag).policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(config.soleGroupForLayer(0).tag, "full");
    EXPECT_ANY_THROW(config.groupForLayer(0, linear_tag));
}

TEST(CacheTopologyTest, TaggedGroupRecordsAreStableAndReadOnly) {
    auto topology = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});

    // The tagged group records are the only published view: repeated reads return
    // the same storage and every group stays reachable by tag.
    EXPECT_EQ(&topology->group("full"), &topology->group("full"));
    std::set<std::string> tags;
    for (const auto& group : topology->groups()) {
        tags.insert(group.tag);
    }
    EXPECT_EQ(tags, (std::set<std::string>{"full", "linear"}));
    EXPECT_EQ(topology->group("full").spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(topology->group("linear").spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(topology->layer(0).group_tags, (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(topology->groupForLayer(0, "linear").tag, "linear");
}

TEST(CacheTopologyTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first    = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});
    auto reversed = CacheTopology::create({makeGroup("linear", {0}, CacheGroupType::LINEAR), makeGroup("full", {0})},
                                          {{0, {"full", "linear"}}});

    // The two topologies declare the same tags in opposite storage order.
    EXPECT_NE(first->groups().front().tag, reversed->groups().front().tag);
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
