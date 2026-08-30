#include <set>
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm {
namespace {

static_assert(std::is_copy_constructible_v<CacheConfig>);
static_assert(std::is_copy_assignable_v<CacheConfig>);
static_assert(std::is_move_constructible_v<CacheConfig>);
static_assert(std::is_move_assignable_v<CacheConfig>);

CacheGroup makeGroup(std::string tag, CacheGroupType type = CacheGroupType::FULL) {
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

CacheConfig makeConfig(std::vector<CacheGroup> groups, std::vector<CacheLayer> layers) {
    const auto  layer_num = static_cast<uint32_t>(layers.size());
    CacheConfig config(std::move(groups), std::move(layers), layer_num);
    return config;
}

// Build a group with a fully explicit physical layout so tag lookup can prove
// that each group retains its own complete record.
CacheGroup makeSigGroup(std::string     tag,
                        uint32_t        heads,
                        uint32_t        physical_b,
                        uint32_t        kernel_b,
                        size_t          kv_stride,
                        size_t          scale_stride,
                        KVCacheSpecType spec_type = KVCacheSpecType::MultiHeadAttention,
                        CacheGroupType  type      = CacheGroupType::FULL) {
    auto spec                       = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block        = physical_b;
    spec->kernel_seq_size_per_block = kernel_b;
    spec->type                      = spec_type;

    CacheGroup group;
    group.tag                   = std::move(tag);
    group.spec                  = std::move(spec);
    group.policy                = defaultCacheGroupPolicy(type);
    group.block_num             = 16;
    group.local_kv_head_num     = heads;
    group.kv_block_stride_bytes = kv_stride;
    group.kv_scale_stride_bytes = scale_stride;
    return group;
}

TEST(CacheConfigOwnershipTest, TwoTagsOnOneLayerCarryDistinctPhysicalRecords) {
    // Two FULL groups share layer 0 but have different heads, B, K and strides.
    // No global scalar or group-parallel projection can describe layer 0; every
    // consumer must read the exact group record by tag.
    auto config = makeConfig({makeSigGroup("a", /*heads=*/2, /*B=*/8, /*kernel=*/2, /*kv=*/64, /*scale=*/8),
                              makeSigGroup("b", /*heads=*/5, /*B=*/4, /*kernel=*/4, /*kv=*/48, /*scale=*/0)},
                             {{"a", "b"}});

    const auto& a = config.groupForLayer(0, "a");
    const auto& b = config.groupForLayer(0, "b");
    EXPECT_NE(a.local_kv_head_num, b.local_kv_head_num);
    EXPECT_NE(a.spec->seq_size_per_block, b.spec->seq_size_per_block);
    EXPECT_NE(a.kernelSeqSizePerBlock(), b.kernelSeqSizePerBlock());
    EXPECT_NE(a.kv_block_stride_bytes, b.kv_block_stride_bytes);
    EXPECT_NE(a.kv_scale_stride_bytes, b.kv_scale_stride_bytes);

    EXPECT_EQ(a.local_kv_head_num, 2u);
    EXPECT_EQ(b.local_kv_head_num, 5u);
    EXPECT_EQ(a.kv_block_stride_bytes, 64u);
    EXPECT_EQ(b.kv_block_stride_bytes, 48u);
}

TEST(CacheConfigOwnershipTest, SupportsSingleGlobalGroupAsNEqualsOne) {
    auto topology = makeConfig({makeGroup("full")}, {{"full"}, {"full"}});

    EXPECT_TRUE(topology.hasSingleGlobalGroup());
    EXPECT_TRUE(topology.hasOneGroupPerLayer());
    EXPECT_EQ(topology.soleGroupForLayer(0).tag, "full");
    EXPECT_EQ(topology.groupsForLayer(1).front(), "full");
}

TEST(CacheConfigOwnershipTest, SupportsDistinctOneToOneGroupsAndOneToManyLayers) {
    auto topology = makeConfig({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)},
                               {{"full"}, {"linear"}, {"full", "linear"}});

    EXPECT_FALSE(topology.hasSingleGlobalGroup());
    EXPECT_FALSE(topology.hasOneGroupPerLayer());
    EXPECT_EQ(topology.groupForLayer(2, "linear").policy.group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(topology.groupsForLayer(2).size(), 2u);
    EXPECT_ANY_THROW(topology.soleGroupForLayer(2));
}

TEST(CacheConfigOwnershipTest, CacheConfigPublishesTagLookupAndLayerMembership) {
    auto config = makeConfig({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)},
                             {{"full"}, {"linear"}, {"full", "linear"}});

    const std::string_view full_tag   = "full";
    const std::string_view linear_tag = "linear";
    EXPECT_EQ(config.group(full_tag).tag, "full");
    EXPECT_EQ(config.groupsForLayer(2), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(config.groupForLayer(2, linear_tag).policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(config.soleGroupForLayer(0).tag, "full");
    EXPECT_ANY_THROW(config.groupForLayer(0, linear_tag));
}

TEST(CacheConfigOwnershipTest, TaggedGroupRecordsAreStableAndReadOnly) {
    auto topology = makeConfig({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)}, {{"full", "linear"}});

    // The tagged group records are the only published view: repeated reads return
    // the same storage and every group stays reachable by tag.
    EXPECT_EQ(&topology.group("full"), &topology.group("full"));
    std::set<std::string> tags;
    for (const auto& group : topology.groups()) {
        tags.insert(group.tag);
    }
    EXPECT_EQ(tags, (std::set<std::string>{"full", "linear"}));
    EXPECT_EQ(topology.group("full").spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(topology.group("linear").spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(topology.groupsForLayer(0), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(topology.groupForLayer(0, "linear").tag, "linear");
}

TEST(CacheConfigOwnershipTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first    = makeConfig({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)}, {{"full", "linear"}});
    auto reversed = makeConfig({makeGroup("linear", CacheGroupType::LINEAR), makeGroup("full")}, {{"full", "linear"}});

    // The two topologies declare the same tags in opposite storage order.
    EXPECT_NE(first.groups().front().tag, reversed.groups().front().tag);
    EXPECT_EQ(first.group("full").policy.group_type, reversed.group("full").policy.group_type);
    EXPECT_EQ(first.group("linear").policy.group_type, reversed.group("linear").policy.group_type);
    EXPECT_EQ(first.groupForLayer(0, "full").tag, reversed.groupForLayer(0, "full").tag);
    EXPECT_EQ(first.groupForLayer(0, "linear").tag, reversed.groupForLayer(0, "linear").tag);
}

TEST(CacheConfigOwnershipTest, MoveAssignmentTransfersTopologyAndLayerMetadata) {
    auto source                    = makeConfig({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)},
                                                {{"full"}, {"linear"}, {"full", "linear"}});
    source.block_num               = 23;
    source.seq_size_per_block      = 8;
    source.enable_hybrid_attention = true;

    auto target = makeConfig({makeGroup("old")}, {{"old"}});
    target      = std::move(source);

    EXPECT_EQ(target.layer_num, 3u);
    EXPECT_EQ(target.layer_all_num, 3u);
    EXPECT_EQ(target.block_num, 23u);
    EXPECT_EQ(target.seq_size_per_block, 8u);
    EXPECT_TRUE(target.enable_hybrid_attention);
    EXPECT_EQ(target.group("full").tag, "full");
    EXPECT_EQ(target.groupForLayer(1, "linear").policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(target.groupsForLayer(2), (std::vector<std::string>{"full", "linear"}));
}

TEST(CacheConfigOwnershipTest, RejectsUnknownLayerTag) {
    EXPECT_ANY_THROW(makeConfig({makeGroup("full")}, {{"full"}, {"missing"}}));
}

TEST(CacheConfigOwnershipTest, FinalizeBlockNumsIsRepeatableAndRecomputesEveryGroup) {
    auto full                                = makeGroup("full");
    auto swa                                 = makeGroup("swa");
    swa.policy.group_type                    = CacheGroupType::SWA;
    auto explicit_group                      = makeGroup("explicit");
    explicit_group.policy.explicit_block_num = 7;

    auto config =
        makeConfig({std::move(full), std::move(swa), std::move(explicit_group)}, {{"full", "swa", "explicit"}});
    config.linear_step = 4;

    config.finalizeBlockNums(10, RuntimeConfig{});
    EXPECT_EQ(config.block_num, 10u);
    EXPECT_EQ(config.group("full").block_num, 10u);
    EXPECT_EQ(config.group("swa").block_num, 3u);
    EXPECT_EQ(config.group("explicit").block_num, 7u);

    config.finalizeBlockNums(5, RuntimeConfig{});
    EXPECT_EQ(config.block_num, 5u);
    EXPECT_EQ(config.group("full").block_num, 5u);
    EXPECT_EQ(config.group("swa").block_num, 2u);
    EXPECT_EQ(config.group("explicit").block_num, 7u);
}

TEST(CacheConfigOwnershipTest, FinalizeBlockNumsRejectsZeroGlobalBlocks) {
    auto config = makeConfig({makeGroup("full")}, {{"full"}});
    EXPECT_THROW(config.finalizeBlockNums(0, RuntimeConfig{}), std::runtime_error);
}

TEST(CacheConfigOwnershipTest, TransitionalBlockLayoutPreservesCountsUntilFinalization) {
    auto config = makeConfig({makeSigGroup("a", 2, 8, 2, 64, 8), makeSigGroup("b", 3, 8, 2, 48, 0)}, {{"a", "b"}});

    rtp_llm::test::setGroupBlockLayout(config, {17, 9}, {64, 48}, {8, 0});
    EXPECT_EQ(config.group("a").block_num, 17u);
    EXPECT_EQ(config.group("b").block_num, 9u);

    config.finalizeBlockNums(5, RuntimeConfig{});
    EXPECT_EQ(config.group("a").block_num, 5u);
    EXPECT_EQ(config.group("b").block_num, 5u);
}

TEST(CacheConfigOwnershipTest, FinalizeBlockNumsValidatesMtpGeometryAndCounts) {
    auto config = makeConfig({makeGroup("full")}, {{"full"}});
    auto sub    = makeConfig({makeGroup("full")}, {{"full"}});
    config.mtp_sub_configs.push_back(std::make_shared<CacheConfig>(std::move(sub)));

    EXPECT_NO_THROW(config.finalizeBlockNums(8, RuntimeConfig{}));
    EXPECT_EQ(config.mtp_sub_configs.front()->group("full").block_num, 8u);

    auto count_mismatch                      = makeConfig({makeGroup("full")}, {{"full"}});
    auto explicit_group                      = makeGroup("full");
    explicit_group.policy.explicit_block_num = 7;
    count_mismatch.mtp_sub_configs.push_back(
        std::make_shared<CacheConfig>(makeConfig({std::move(explicit_group)}, {{"full"}})));
    EXPECT_THROW(count_mismatch.finalizeBlockNums(8, RuntimeConfig{}), std::runtime_error);

    auto geometry_mismatch  = makeConfig({makeGroup("full")}, {{"full"}});
    auto different_geometry = makeSigGroup("full", 1, 8, 4, 64, 0);
    geometry_mismatch.mtp_sub_configs.push_back(
        std::make_shared<CacheConfig>(makeConfig({std::move(different_geometry)}, {{"full"}})));
    EXPECT_THROW(geometry_mismatch.finalizeBlockNums(8, RuntimeConfig{}), std::runtime_error);
}

TEST(CacheConfigOwnershipTest, ConstructorFailuresLeavePublishedStateUnchanged) {
    auto config = makeConfig({makeGroup("full")}, {{"full"}});
    config.finalizeBlockNums(13, RuntimeConfig{});
    const auto before = config.debugString();

    auto duplicate = config.groups();
    duplicate.push_back(duplicate.front());
    EXPECT_THROW(config = CacheConfig(std::move(duplicate), {{"full"}}, /*main_layer_num=*/1), std::runtime_error);
    EXPECT_EQ(config.debugString(), before);

    EXPECT_THROW(config = CacheConfig(config.groups(), {{"missing"}}, /*main_layer_num=*/1), std::runtime_error);
    EXPECT_EQ(config.debugString(), before);

    auto invalid_geometry                            = config.groups();
    auto invalid_geometry_spec                       = invalid_geometry.front().spec->clone();
    invalid_geometry_spec->kernel_seq_size_per_block = 3;
    invalid_geometry.front().spec                    = std::move(invalid_geometry_spec);
    EXPECT_THROW(config = CacheConfig(std::move(invalid_geometry), {{"full"}}, /*main_layer_num=*/1),
                 std::runtime_error);
    EXPECT_EQ(config.debugString(), before);
}

}  // namespace
}  // namespace rtp_llm
