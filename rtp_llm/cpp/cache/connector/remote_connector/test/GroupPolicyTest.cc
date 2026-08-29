#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/GroupPolicy.h"

namespace rtp_llm::remote_connector {
namespace {

GroupBase makeGroup(std::string tag, int layer_id, CacheGroupType type) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = tag;
    spec->seq_size_per_block = 8;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(type);
    group.layer_ids                 = {layer_id};
    group.block_num                 = 16;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = type == CacheGroupType::FULL ? 2 : 8;
    group.kv_block_stride_bytes     = 16;
    return group;
}

StorageBackend::BufferResolver unusedResolver() {
    return [](int, int, int) { return std::vector<BlockInfo>{}; };
}

class InvalidAggregatePolicy: public FullLayerGroupPolicy {
public:
    using FullLayerGroupPolicy::FullLayerGroupPolicy;

    std::vector<uint64_t> reachableAggregateMasks() const override {
        return {uint64_t{1} << 63};
    }
};

TEST(GroupPolicyTest, FullAggregateUsesCanonicalNameOrderIndependentOfNumericGroupIds) {
    auto topology = CacheTopology::create(
        {makeGroup("z_group", 0, CacheGroupType::FULL), makeGroup("a_group", 1, CacheGroupType::FULL)},
        {{0, {"z_group"}}, {1, {"a_group"}}});
    FullLayerGroupPolicy policy(*topology, unusedResolver(), /*full_group_ids=*/{0, 1}, /*other_group_ids=*/{});
    ASSERT_TRUE(policy.init());

    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));
    ASSERT_EQ(groups.count("Fz_groupFa_group"), 0u);
    ASSERT_EQ(groups.at("Fa_groupFz_group"),
              (std::vector<std::string>{"tp0_Fa_group", "tp1_Fa_group", "tp0_Fz_group", "tp1_Fz_group"}));

    const auto first_groups    = groups;
    const auto first_spec_info = policy.spec_info_map();
    groups                     = {{"stale", {"stale_spec"}}};
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));
    EXPECT_EQ(groups, first_groups);
    ASSERT_EQ(policy.spec_info_map().size(), first_spec_info.size());
    for (const auto& [spec_name, expected] : first_spec_info) {
        const auto actual = policy.spec_info_map().find(spec_name);
        ASSERT_NE(actual, policy.spec_info_map().end());
        EXPECT_EQ(actual->second.group_id, expected.group_id);
        EXPECT_EQ(actual->second.tp_rank, expected.tp_rank);
        EXPECT_EQ(actual->second.tag, expected.tag);
    }
}

TEST(GroupPolicyTest, FullLinearPolicyPreservesTheSameCanonicalFullIdentityAndSortsCombinedSpecs) {
    auto                       topology = CacheTopology::create({makeGroup("z_group", 0, CacheGroupType::FULL),
                                                                 makeGroup("a_group", 1, CacheGroupType::FULL),
                                                                 makeGroup("z_linear", 2, CacheGroupType::LINEAR),
                                                                 makeGroup("a_linear", 3, CacheGroupType::LINEAR)},
                                                                {{0, {"z_group"}}, {1, {"a_group"}}, {2, {"z_linear"}}, {3, {"a_linear"}}});
    FullLinearLayerGroupPolicy policy(
        *topology, unusedResolver(), /*full_group_ids=*/{0, 1}, /*other_group_ids=*/{2, 3}, /*write_interval=*/1);
    ASSERT_TRUE(policy.init());

    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));
    EXPECT_EQ(groups.at("Fa_groupFz_group"),
              (std::vector<std::string>{"tp0_Fa_group", "tp1_Fa_group", "tp0_Fz_group", "tp1_Fz_group"}));
    EXPECT_EQ(groups.at("Fa_groupFz_groupLa_linearLz_linear"),
              (std::vector<std::string>{"tp0_Fa_group",
                                        "tp1_Fa_group",
                                        "tp0_Fz_group",
                                        "tp1_Fz_group",
                                        "tp0_La_linear",
                                        "tp1_La_linear",
                                        "tp0_Lz_linear",
                                        "tp1_Lz_linear"}));
}

TEST(GroupPolicyTest, InvalidAggregateBuildDoesNotPublishPartialState) {
    auto topology = CacheTopology::create({makeGroup("full", 0, CacheGroupType::FULL)}, {{0, {"full"}}});
    InvalidAggregatePolicy policy(*topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{});
    ASSERT_TRUE(policy.init());

    GroupPolicy::LocationSpecGroups groups{{"existing", {"existing_spec"}}};
    EXPECT_FALSE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));
    EXPECT_EQ(groups, (GroupPolicy::LocationSpecGroups{{"existing", {"existing_spec"}}}));
    EXPECT_TRUE(policy.spec_info_map().empty());
}

}  // namespace
}  // namespace rtp_llm::remote_connector
