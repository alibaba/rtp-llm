#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"

namespace rtp_llm::kvcm {
namespace {

GroupBase makeGroup(std::string tag, int layer_id, CacheGroupType type, size_t kv_block_stride_bytes = 16) {
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
    group.kv_block_stride_bytes     = kv_block_stride_bytes;
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

kv_cache_manager::Location makeLocation(std::initializer_list<std::string> specs) {
    kv_cache_manager::Location location;
    for (const std::string& spec : specs) {
        location.emplace_back(kv_cache_manager::LocationSpecUnit{spec, "uri_" + spec});
    }
    return location;
}

StorageRequest makeStorageRequest(std::vector<std::vector<StorageBlockHandle>> handles) {
    std::vector<CacheKeyType> keys(handles.size());
    for (size_t index = 0; index < keys.size(); ++index) {
        keys[index] = static_cast<CacheKeyType>(index + 1);
    }
    return {std::make_shared<const CacheKeysType>(std::move(keys)), std::move(handles), 0};
}

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

TEST(GroupPolicyTest, FullAndLinearTP2LoadsOnlyTheNewestLinearLocation) {
    auto topology = CacheTopology::create(
        {makeGroup("full", 0, CacheGroupType::FULL), makeGroup("linear", 1, CacheGroupType::LINEAR)},
        {{0, {"full"}}, {1, {"linear"}}});
    FullLinearLayerGroupPolicy policy(
        *topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{1}, /*write_interval=*/0);
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));

    kv_cache_manager::Locations locations{
        makeLocation({"tp0_Ffull", "tp1_Ffull"}),
        makeLocation({"tp0_Ffull", "tp1_Ffull", "tp0_Llinear", "tp1_Llinear"}),
        makeLocation({"tp0_Ffull", "tp1_Ffull"}),
        makeLocation({"tp0_Ffull", "tp1_Ffull", "tp0_Llinear", "tp1_Llinear"}),
    };
    LocationsView view;
    ASSERT_TRUE(policy.filterNeedLoadLocations(locations, view));
    ASSERT_EQ(view.size(), 4u);
    EXPECT_EQ(view[0].size(), 2u);
    EXPECT_EQ(view[1].size(), 2u);
    EXPECT_EQ(view[2].size(), 2u);
    EXPECT_EQ(view[3].size(), 4u);
    EXPECT_EQ(view[1][0].spec_name, "tp0_Ffull");
    EXPECT_EQ(view[1][1].spec_name, "tp1_Ffull");
}

TEST(GroupPolicyTest, FullAndLinearTP2ReturnsNoRemotePrefixWithoutLinearState) {
    auto topology = CacheTopology::create(
        {makeGroup("full", 0, CacheGroupType::FULL), makeGroup("linear", 1, CacheGroupType::LINEAR)},
        {{0, {"full"}}, {1, {"linear"}}});
    FullLinearLayerGroupPolicy policy(
        *topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{1}, /*write_interval=*/0);
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));

    const kv_cache_manager::Locations locations{
        makeLocation({"tp0_Ffull", "tp1_Ffull"}),
        makeLocation({"tp0_Ffull", "tp1_Ffull"}),
    };
    LocationsView view;
    EXPECT_TRUE(policy.filterNeedLoadLocations(locations, view));
    EXPECT_TRUE(view.empty());
}

TEST(GroupPolicyTest, FullAndLinearTP2RejectsIncompleteRankSet) {
    auto topology = CacheTopology::create(
        {makeGroup("full", 0, CacheGroupType::FULL), makeGroup("linear", 1, CacheGroupType::LINEAR)},
        {{0, {"full"}}, {1, {"linear"}}});
    FullLinearLayerGroupPolicy policy(
        *topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{1}, /*write_interval=*/0);
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));

    LocationsView view;
    EXPECT_FALSE(policy.filterNeedLoadLocations(
        kv_cache_manager::Locations{makeLocation({"tp0_Ffull", "tp0_Llinear", "tp1_Llinear"})}, view));
}

TEST(GroupPolicyTest, FullOnlyTP2RejectsMissingAndDuplicateRanks) {
    auto topology = CacheTopology::create({makeGroup("full", 0, CacheGroupType::FULL)}, {{0, {"full"}}});
    FullLayerGroupPolicy policy(*topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{});
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));

    LocationsView view;
    EXPECT_FALSE(policy.filterNeedLoadLocations(kv_cache_manager::Locations{makeLocation({"tp0_Ffull"})}, view));
    EXPECT_TRUE(view.empty());
    EXPECT_FALSE(
        policy.filterNeedLoadLocations(kv_cache_manager::Locations{makeLocation({"tp0_Ffull", "tp0_Ffull"})}, view));
    EXPECT_TRUE(view.empty());

    const kv_cache_manager::Locations reversed{
        makeLocation({"tp1_Ffull", "tp0_Ffull"}),
    };
    ASSERT_TRUE(policy.filterNeedLoadLocations(reversed, view));
    ASSERT_EQ(view.size(), 1u);
    EXPECT_EQ(view.front().size(), 2u);
}

TEST(GroupPolicyTest, FullAndLinearTP2RejectsDuplicateFullOrLinearSpecs) {
    auto topology = CacheTopology::create(
        {makeGroup("full", 0, CacheGroupType::FULL), makeGroup("linear", 1, CacheGroupType::LINEAR)},
        {{0, {"full"}}, {1, {"linear"}}});
    FullLinearLayerGroupPolicy policy(
        *topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{1}, /*write_interval=*/0);
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));

    LocationsView view;
    EXPECT_FALSE(
        policy.filterNeedLoadLocations(kv_cache_manager::Locations{makeLocation({"tp0_Ffull", "tp0_Ffull"})}, view));
    EXPECT_TRUE(view.empty());
    EXPECT_FALSE(policy.filterNeedLoadLocations(
        kv_cache_manager::Locations{makeLocation({"tp0_Ffull", "tp0_Ffull", "tp0_Llinear", "tp1_Llinear"})}, view));
    EXPECT_TRUE(view.empty());
    EXPECT_FALSE(policy.filterNeedLoadLocations(
        kv_cache_manager::Locations{makeLocation({"tp0_Ffull", "tp1_Ffull", "tp0_Llinear", "tp0_Llinear"})}, view));
    EXPECT_TRUE(view.empty());
}

TEST(GroupPolicyTest, FullAndLinearWriteSelectionPreservesLegacyIntervals) {
    auto topology = CacheTopology::create(
        {makeGroup("full", 0, CacheGroupType::FULL), makeGroup("linear", 1, CacheGroupType::LINEAR)},
        {{0, {"full"}}, {1, {"linear"}}});
    const auto handles = std::vector<std::vector<StorageBlockHandle>>{
        {{0, 10}, {1, 20}}, {{0, 11}}, {{0, 12}, {1, 22}}, {{0, 13}, {1, 23}}};

    FullLinearLayerGroupPolicy last_only(
        *topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{1}, /*write_interval=*/0);
    ASSERT_TRUE(last_only.init());
    GroupPolicy::LocationSpecGroups last_only_groups;
    ASSERT_TRUE(last_only.buildLocationSpecGroups(/*tp_size=*/2, last_only_groups));
    std::vector<std::string> selected;
    ASSERT_TRUE(last_only.getNeedWriteGroups(makeStorageRequest(handles), handles.size(), selected));
    EXPECT_EQ(selected, (std::vector<std::string>{"Ffull", "Ffull", "Ffull", "FfullLlinear"}));

    FullLinearLayerGroupPolicy every_two(
        *topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{1}, /*write_interval=*/2);
    ASSERT_TRUE(every_two.init());
    GroupPolicy::LocationSpecGroups every_two_groups;
    ASSERT_TRUE(every_two.buildLocationSpecGroups(/*tp_size=*/2, every_two_groups));
    selected.clear();
    ASSERT_TRUE(every_two.getNeedWriteGroups(makeStorageRequest(handles), handles.size(), selected));
    EXPECT_EQ(selected, (std::vector<std::string>{"FfullLlinear", "Ffull", "Ffull", "FfullLlinear"}));
}

TEST(GroupPolicyTest, FullAndLinearWriteIntervalOnePreservesLegacyDefaultAndPartialSelection) {
    auto topology = CacheTopology::create(
        {makeGroup("full", 0, CacheGroupType::FULL), makeGroup("linear", 1, CacheGroupType::LINEAR)},
        {{0, {"full"}}, {1, {"linear"}}});
    FullLinearLayerGroupPolicy policy(
        *topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{1}, /*write_interval=*/1);
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));

    std::vector<std::string> selected;
    const auto               all_groups = std::vector<std::vector<StorageBlockHandle>>{
        {{0, 10}, {1, 20}}, {{0, 11}, {1, 21}}, {{0, 12}, {1, 22}}, {{0, 13}, {1, 23}}};
    ASSERT_TRUE(policy.getNeedWriteGroups(makeStorageRequest(all_groups), all_groups.size(), selected));
    EXPECT_TRUE(selected.empty());

    const auto partial_groups = std::vector<std::vector<StorageBlockHandle>>{
        {{0, 10}, {1, 20}}, {{0, 11}, {1, 21}}, {{0, 12}}, {{0, 13}, {1, 23}}};
    ASSERT_TRUE(policy.getNeedWriteGroups(makeStorageRequest(partial_groups), partial_groups.size(), selected));
    EXPECT_EQ(selected, (std::vector<std::string>{"FfullLlinear", "FfullLlinear", "Ffull", "FfullLlinear"}));
}

TEST(GroupPolicyTest, FullOnlyTP2PreservesOffsetAndCanonicalMultiGroupWrites) {
    auto topology = CacheTopology::create(
        {makeGroup("z_full", 0, CacheGroupType::FULL), makeGroup("a_full", 1, CacheGroupType::FULL)},
        {{0, {"z_full"}}, {1, {"a_full"}}});
    FullLayerGroupPolicy policy(*topology, unusedResolver(), /*full_group_ids=*/{0, 1}, /*other_group_ids=*/{});
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));

    const kv_cache_manager::Locations locations{
        makeLocation({"tp0_Fa_full", "tp1_Fa_full", "tp0_Fz_full", "tp1_Fz_full"}),
        makeLocation({"tp0_Fa_full", "tp1_Fa_full", "tp0_Fz_full", "tp1_Fz_full"}),
        makeLocation({"tp0_Fa_full", "tp1_Fa_full", "tp0_Fz_full", "tp1_Fz_full"}),
    };
    LocationsView view;
    ASSERT_TRUE(policy.filterNeedLoadLocations(locations, view, /*block_mask=*/1));
    ASSERT_EQ(view.size(), 3u);
    EXPECT_TRUE(view[0].empty());
    EXPECT_EQ(view[1].size(), 4u);
    EXPECT_EQ(view[2].size(), 4u);

    const auto handles =
        std::vector<std::vector<StorageBlockHandle>>{{{0, 10}, {1, 20}}, {{0, 11}, {1, 21}}, {{0, 12}, {1, 22}}};
    std::vector<std::string> selected;
    ASSERT_TRUE(policy.getNeedWriteGroups(makeStorageRequest(handles), handles.size(), selected));
    EXPECT_EQ(selected, (std::vector<std::string>{"Fa_fullFz_full", "Fa_fullFz_full", "Fa_fullFz_full"}));
}

TEST(GroupPolicyTest, RejectsUnsupportedWriteMasksAndUnknownGroups) {
    auto topology = CacheTopology::create(
        {makeGroup("full", 0, CacheGroupType::FULL), makeGroup("linear", 1, CacheGroupType::LINEAR)},
        {{0, {"full"}}, {1, {"linear"}}});
    FullLinearLayerGroupPolicy policy(
        *topology, unusedResolver(), /*full_group_ids=*/{0}, /*other_group_ids=*/{1}, /*write_interval=*/2);
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, groups));

    std::vector<std::string> selected;
    const auto               linear_only = std::vector<std::vector<StorageBlockHandle>>{{{1, 20}}};
    EXPECT_FALSE(policy.getNeedWriteGroups(makeStorageRequest(linear_only), linear_only.size(), selected));
    const auto unknown = std::vector<std::vector<StorageBlockHandle>>{{{0, 10}, {2, 30}}};
    EXPECT_FALSE(policy.getNeedWriteGroups(makeStorageRequest(unknown), unknown.size(), selected));
}

TEST(GroupPolicyTest, PreservesHeterogeneousBlockSizesAndScalesLocationSpecsLinearly) {
    constexpr int          group_count = 19;
    std::vector<GroupBase> groups;
    std::vector<LayerBase> layers;
    std::vector<int32_t>   full_group_ids;
    groups.reserve(group_count);
    layers.reserve(group_count);
    full_group_ids.reserve(group_count);
    for (int group_id = 0; group_id < group_count; ++group_id) {
        const std::string tag = "group_" + std::to_string(group_id);
        groups.push_back(makeGroup(tag, group_id, CacheGroupType::FULL, static_cast<size_t>(group_id + 1)));
        layers.push_back(LayerBase{group_id, {tag}});
        full_group_ids.push_back(group_id);
    }
    auto                 topology = CacheTopology::create(std::move(groups), std::move(layers));
    FullLayerGroupPolicy policy(*topology, unusedResolver(), full_group_ids, /*other_group_ids=*/{});
    ASSERT_TRUE(policy.init());
    ASSERT_EQ(policy.groups().size(), group_count);
    EXPECT_EQ(policy.groups().at(0).block_size_bytes, 1u);
    EXPECT_EQ(policy.groups().at(group_count - 1).block_size_bytes, static_cast<size_t>(group_count));

    GroupPolicy::LocationSpecGroups location_groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(/*tp_size=*/2, location_groups));
    EXPECT_EQ(policy.spec_info_map().size(), static_cast<size_t>(group_count * 2));
    // One singleton per group and one reachable all-full aggregate.
    EXPECT_EQ(location_groups.size(), static_cast<size_t>(group_count + 1));
}

TEST(GroupPolicyTest, RejectsOverlappingMembershipAndMoreThanSixtyFourGroups) {
    auto one_group = CacheTopology::create({makeGroup("full", 0, CacheGroupType::FULL)}, {{0, {"full"}}});
    EXPECT_FALSE(DefaultLayerGroupPolicy(*one_group, unusedResolver(), {0}, {0}).init());

    std::vector<GroupBase> groups;
    std::vector<LayerBase> layers;
    std::vector<int32_t>   full_group_ids;
    for (int group_id = 0; group_id < 65; ++group_id) {
        const std::string tag = "group_" + std::to_string(group_id);
        groups.push_back(makeGroup(tag, group_id, CacheGroupType::FULL));
        layers.push_back(LayerBase{group_id, {tag}});
        full_group_ids.push_back(group_id);
    }
    auto too_many = CacheTopology::create(std::move(groups), std::move(layers));
    EXPECT_FALSE(FullLayerGroupPolicy(*too_many, unusedResolver(), full_group_ids, {}).init());
}

TEST(GroupPolicyTest, BufferRoutingUsesStableTagsAndValidatesAggregateSize) {
    std::array<char, 16> storage{};
    auto                 topology =
        CacheTopology::create({makeGroup("semantic_tag", 0, CacheGroupType::FULL)}, {{0, {"semantic_tag"}}});
    auto resolver = [&storage](int layer_id, int group_id, int block_id) {
        EXPECT_EQ(layer_id, 0);
        EXPECT_EQ(group_id, 0);
        EXPECT_EQ(block_id, 7);
        BlockInfo info;
        info.is_cuda    = true;
        info.addr       = storage.data();
        info.size_bytes = storage.size();
        return std::vector<BlockInfo>{info};
    };
    FullLayerGroupPolicy policy(*topology, resolver, /*full_group_ids=*/{0}, /*other_group_ids=*/{});
    ASSERT_TRUE(policy.init());

    kv_cache_manager::BlockBuffers buffers;
    ASSERT_TRUE(policy.genBlockBuffersByTag({"semantic_tag"}, {7}, buffers));
    ASSERT_EQ(buffers.size(), 1u);
    EXPECT_EQ(buffers.front().iovs.size(), 1u);

    EXPECT_THROW(policy.genBlockBuffersByTag({"missing"}, {7}, buffers), std::runtime_error);
}

TEST(GroupPolicyTest, RejectsInvalidGroupModeInputs) {
    auto topology = CacheTopology::create(
        {makeGroup("full", 0, CacheGroupType::FULL), makeGroup("linear", 1, CacheGroupType::LINEAR)},
        {{0, {"full"}}, {1, {"linear"}}});
    EXPECT_FALSE(FullLayerGroupPolicy(*topology, unusedResolver(), {}, {}).init());
    EXPECT_FALSE(FullLayerGroupPolicy(*topology, unusedResolver(), {0}, {1}).init());
    EXPECT_FALSE(FullLinearLayerGroupPolicy(*topology, unusedResolver(), {}, {1}, 0).init());
    EXPECT_FALSE(FullLinearLayerGroupPolicy(*topology, unusedResolver(), {0}, {}, 0).init());
}

}  // namespace
}  // namespace rtp_llm::kvcm
