#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
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

TEST(CacheTopologyTest, CompatibilitySnapshotsAreLazyStableAndReadOnly) {
    auto topology = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});

    const auto& tags_first  = topology->groupTagsSnapshot();
    const auto& tags_second = topology->groupTagsSnapshot();
    const auto& spec_types  = topology->groupSpecTypesSnapshot();
    EXPECT_EQ(&tags_first, &tags_second);
    EXPECT_EQ(tags_first, (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(spec_types,
              (std::vector<KVCacheSpecType>{KVCacheSpecType::MultiHeadAttention, KVCacheSpecType::MultiHeadAttention}));
    EXPECT_EQ(topology->layerGroupIdsSnapshot(), (std::vector<std::vector<int>>{{0, 1}}));
    EXPECT_EQ(topology->layerTagToGroupIdSnapshot().front().at("linear"), 1);
}

TEST(CacheTopologyTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first    = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});
    auto reversed = CacheTopology::create({makeGroup("linear", {0}, CacheGroupType::LINEAR), makeGroup("full", {0})},
                                          {{0, {"full", "linear"}}});

    EXPECT_NE(first->groupIdForTag("full"), reversed->groupIdForTag("full"));
    EXPECT_EQ(first->group("full").policy.group_type, reversed->group("full").policy.group_type);
    EXPECT_EQ(first->group("linear").policy.group_type, reversed->group("linear").policy.group_type);
    EXPECT_EQ(first->groupForLayer(0, "full").tag, reversed->groupForLayer(0, "full").tag);
    EXPECT_EQ(first->groupForLayer(0, "linear").tag, reversed->groupForLayer(0, "linear").tag);
}

TEST(CacheTopologyTest, RejectsInconsistentReverseMembership) {
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full", {0})}, {{0, {"full"}}, {1, {"full"}}}));
}

TEST(CacheTopologyTest, ActiveModelTopologyWithoutDraftPreservesAllocation) {
    CacheConfig config;
    config.layer_num = 2;
    config.setTopology({makeGroup("full", {0, 1})}, {{0, {"full"}}, {1, {"full"}}});
    EXPECT_EQ(config.activeModelTopologyPtr(), config.topologyPtr());
}

TEST(CacheTopologyTest, ActiveModelTopologyExcludesUnusedMtpModules) {
    CacheConfig config;
    config.layer_num     = 2;
    config.layer_all_num = 5;
    auto full_group = makeGroup("full", {0, 1});
    auto swa_group  = makeGroup("swa", {0, 1, 2, 3, 4}, CacheGroupType::SWA);
    swa_group.kv_block_stride_bytes = 256;
    swa_group.policy.cp_slice      = CpBlockSliceMode::EQUAL_BYTES;
    config.setTopology({full_group, swa_group},
                       {{0, {"full", "swa"}}, {1, {"full", "swa"}}, {2, {"swa"}}, {3, {"swa"}}, {4, {"swa"}}});
    for (size_t module_index = 0; module_index < 3; ++module_index) {
        auto module_config       = std::make_shared<CacheConfig>();
        module_config->layer_num = 1;
        config.mtp_sub_configs.push_back(std::move(module_config));
    }

    const auto active = config.activeModelTopologyPtr();
    ASSERT_EQ(active->layers().size(), 3u);
    EXPECT_EQ(active->group("full").layer_ids, (std::vector<int>{0, 1}));
    EXPECT_EQ(active->group("swa").layer_ids, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(active->groupIdForTag("swa"), config.topology().groupIdForTag("swa"));
    EXPECT_EQ(active->group("swa").spec, config.group("swa").spec);
    EXPECT_EQ(active->group("swa").kv_block_stride_bytes, 256u);
    EXPECT_EQ(active->group("swa").policy.cp_slice, CpBlockSliceMode::EQUAL_BYTES);
    EXPECT_EQ(active->groupForLayer(2, "swa").tag, "swa");
    EXPECT_EQ(config.topology().layers().size(), 5u);
    EXPECT_EQ(config.group("swa").layer_ids, (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(CacheTopologyTest, ActiveModelTopologyPreservesMultiLayerDraftModule) {
    CacheConfig config;
    config.layer_num     = 2;
    config.layer_all_num = 5;
    config.setTopology({makeGroup("full", {0, 1, 2, 3, 4})},
                       {{0, {"full"}}, {1, {"full"}}, {2, {"full"}}, {3, {"full"}}, {4, {"full"}}});
    auto module_config       = std::make_shared<CacheConfig>();
    module_config->layer_num = 3;
    config.mtp_sub_configs.push_back(std::move(module_config));
    EXPECT_EQ(config.activeModelTopologyPtr(), config.topologyPtr());
}

TEST(CacheTopologyTest, ActiveModelTopologyRejectsInvalidDraftConfig) {
    CacheConfig config;
    config.layer_num = 1;
    config.setTopology({makeGroup("full", {0})}, {{0, {"full"}}});
    config.mtp_sub_configs.push_back(nullptr);
    EXPECT_ANY_THROW(config.activeModelTopologyPtr());
    config.mtp_sub_configs.front() = std::make_shared<CacheConfig>();
    EXPECT_ANY_THROW(config.activeModelTopologyPtr());
    config.mtp_sub_configs.front()->layer_num = 1;
    EXPECT_ANY_THROW(config.activeModelTopologyPtr());
}

}  // namespace
}  // namespace rtp_llm
