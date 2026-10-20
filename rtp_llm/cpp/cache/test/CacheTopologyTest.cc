#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
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

GroupBase makeLayoutGroup(const std::string& tag, int head_dim) {
    AttentionConfigs attention;
    attention.kv_head_num   = 1;
    attention.size_per_head = head_dim;
    ParallelismConfig parallelism;
    KVCacheSpecDesc   desc;
    desc.tag        = tag;
    desc.dtype      = DataType::TYPE_INT8;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    SpecBuildContext context;
    context.attn_config             = &attention;
    context.parallelism_config      = &parallelism;
    context.seq_size_per_block      = 8;
    context.kernel_tokens_per_block = 2;
    auto group                      = makeGroup(tag);
    group.spec                      = MHAKVCacheSpec::build(desc, context);
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
    EXPECT_EQ(topology->group("swa").kernelBlocksPerKvBlock(), 1u);
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

TEST(CacheTopologyTest, GroupPropertiesAndLayerMembershipUseCanonicalIdentity) {
    auto topology = CacheTopology::create({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});

    ASSERT_EQ(topology->groups().size(), 2u);
    EXPECT_EQ(topology->groups()[0].tag, "full");
    EXPECT_EQ(topology->groups()[1].tag, "linear");
    EXPECT_EQ(topology->group("full").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(topology->group("linear").policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(topology->layer(0).group_tags, (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(&topology->groupForLayer(0, "full"), &topology->group("full"));
    EXPECT_EQ(&topology->groupForLayer(0, "linear"), &topology->group("linear"));
}

TEST(CacheTopologyTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first    = CacheTopology::create({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});
    auto reversed = CacheTopology::create({makeGroup("linear", CacheGroupType::LINEAR), makeGroup("full")},
                                          {{0, {"full", "linear"}}});

    EXPECT_EQ(first->groupTags(), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(reversed->groupTags(), (std::vector<std::string>{"linear", "full"}));
    EXPECT_ANY_THROW(first->group("missing"));
    EXPECT_EQ(first->group("full").policy.group_type, reversed->group("full").policy.group_type);
    EXPECT_EQ(first->group("linear").policy.group_type, reversed->group("linear").policy.group_type);
    EXPECT_EQ(first->groupForLayer(0, "full").tag, reversed->groupForLayer(0, "full").tag);
    EXPECT_EQ(first->groupForLayer(0, "linear").tag, reversed->groupForLayer(0, "linear").tag);
}

TEST(CacheTopologyTest, PhysicalGroupResolvesMultipleMtpModuleLayerRanges) {
    CacheConfig config;
    config.layer_num = 1;
    config.setTopology({makeGroup("full")}, {{0, {"full"}}});

    for (int module = 0; module < 2; ++module) {
        CacheConfig draft;
        draft.layer_num = 2;
        draft.setTopology({makeGroup("full")}, {{0, {"full"}}, {1, {"full"}}});
        config.mtp_sub_configs.push_back(config.mergeMTPModule(draft, module, config.layer_num));
    }

    EXPECT_EQ(&config.physicalGroupForLayer(0, "full"), &config.group("full"));
    for (int module = 0; module < 2; ++module) {
        const auto& group = config.mtp_sub_configs[module]->group("full");
        EXPECT_EQ(&config.physicalGroupForLayer(1 + module * 2, "full"), &group);
        EXPECT_EQ(&config.physicalGroupForLayer(2 + module * 2, "full"), &group);
    }
    EXPECT_ANY_THROW((void)config.physicalGroupForLayer(5, "full"));
}

TEST(CacheTopologyTest, GroupTagsKeepConstructionOrderAndStableStorage) {
    auto        topology = CacheTopology::create({makeGroup("zeta"), makeGroup("alpha")}, {{0, {"alpha", "zeta"}}});
    const auto& tags     = topology->groupTags();
    EXPECT_EQ(tags, (std::vector<std::string>{"zeta", "alpha"}));
    EXPECT_EQ(&tags, &topology->groupTags());
    EXPECT_EQ(tags.data(), topology->groupTags().data());
    for (size_t row = 0; row < tags.size(); ++row) {
        EXPECT_EQ(tags[row], topology->groups()[row].tag);
    }
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("alpha"), makeGroup("alpha")}, {{0, {"alpha"}}}));
}

TEST(CacheTopologyTest, TaggedLayoutPreservesOrderAndRetainedTopology) {
    auto        first  = makeLayoutGroup("zeta", 2);
    auto        second = makeLayoutGroup("alpha", 3);
    CacheConfig config;
    config.layer_num = 1;
    config.setTopology({first, second}, {{0, {"zeta", "alpha"}}});
    const auto  retained      = config.topologyPtr();
    const auto& retained_tags = retained->groupTags();
    EXPECT_EQ(&config.groupTags(), &retained_tags);

    config.setGroupBlockLayout({"alpha", "zeta"},
                               {23, 31},
                               {second.kvBlockStrideBytes(), first.kvBlockStrideBytes()},
                               {second.kvScaleStrideBytes(), first.kvScaleStrideBytes()});
    EXPECT_EQ(config.groupTags(), (std::vector<std::string>{"zeta", "alpha"}));
    EXPECT_EQ(config.group("alpha").block_num, 23u);
    EXPECT_EQ(config.group("zeta").block_num, 31u);
    EXPECT_EQ(config.group("alpha").kvBlockStrideBytes(), second.kvBlockStrideBytes());
    EXPECT_EQ(retained->group("alpha").block_num, 16u);
    EXPECT_EQ(retained_tags, (std::vector<std::string>{"zeta", "alpha"}));
    EXPECT_EQ(&retained_tags, &retained->groupTags());
    EXPECT_NE(config.topologyPtr(), retained);
}

TEST(CacheTopologyTest, TaggedLayoutKeepsMtpPhysicalGeometry) {
    CacheConfig config;
    config.layer_num = 1;
    config.setTopology({makeLayoutGroup("full", 2)}, {{0, {"full"}}});
    CacheConfig draft;
    draft.layer_num = 1;
    draft.setTopology({makeLayoutGroup("full", 5)}, {{0, {"full"}}});
    const auto child = config.mergeMTPModule(draft, 0, 1);
    config.mtp_sub_configs.push_back(child);
    const auto main_stride  = config.group("full").kvBlockStrideBytes();
    const auto draft_stride = config.physicalGroupForLayer(1, "full").kvBlockStrideBytes();
    const auto total_bytes  = config.blockSizeBytesForGroup("full");
    ASSERT_NE(main_stride, draft_stride);
    config.setGroupBlockLayout({"full"}, {23}, {main_stride}, {config.group("full").kvScaleStrideBytes()});
    EXPECT_EQ(config.group("full").block_num, 23u);
    EXPECT_EQ(config.physicalGroupForLayer(1, "full").kvBlockStrideBytes(), draft_stride);
    EXPECT_EQ(config.blockSizeBytesForGroup("full"), total_bytes);
    EXPECT_EQ(config.mtp_sub_configs.front(), child);
}

TEST(CacheTopologyTest, InvalidTaggedLayoutsNeverPublishPartialUpdates) {
    CacheConfig config;
    config.layer_num = 1;
    config.setTopology({makeLayoutGroup("first", 2), makeLayoutGroup("second", 2)}, {{0, {"first", "second"}}});
    const auto original = config.topologyPtr();
    const auto kv       = config.group("first").kvBlockStrideBytes();
    const auto scale    = config.group("first").kvScaleStrideBytes();
    const auto rejects  = [&](const std::vector<std::string>& tags,
                             const std::vector<uint32_t>&    blocks,
                             const std::vector<size_t>&      strides,
                             const std::vector<size_t>&      scales) {
        EXPECT_ANY_THROW(config.setGroupBlockLayout(tags, blocks, strides, scales));
        EXPECT_EQ(config.topologyPtr(), original);
        EXPECT_EQ(config.group("first").block_num, 16u);
        EXPECT_EQ(config.group("second").block_num, 16u);
    };
    rejects({"first"}, {21}, {kv}, {scale});
    rejects({"first", "second"}, {21}, {kv, kv}, {scale, scale});
    rejects({"first", "second"}, {21, 22}, {kv}, {scale, scale});
    rejects({"first", "second"}, {21, 22}, {kv, kv}, {scale});
    rejects({"first", "first"}, {21, 22}, {kv, kv}, {scale, scale});
    rejects({"first", "unknown"}, {21, 22}, {kv, kv}, {scale, scale});
    rejects({"first", ""}, {21, 22}, {kv, kv}, {scale, scale});
    rejects({"first", "second"}, {21, 22}, {kv, kv + 1}, {scale, scale});
    rejects({"first", "second"}, {21, 22}, {kv, kv}, {scale, scale + 1});
}

TEST(CacheTopologyTest, DerivesReverseMembershipFromLayers) {
    auto topology = CacheTopology::create({makeGroup("full")}, {{0, {"full"}}, {1, {"full"}}});
    EXPECT_EQ(topology->layerIdsForGroup("full"), (std::vector<int>{0, 1}));
    EXPECT_ANY_THROW(topology->layerIdsForGroup("missing"));
    EXPECT_ANY_THROW(topology->blockSizeBytesForGroup("missing"));
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full")}, {{0, {"missing"}}}));
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full")}, {{0, {"full", "full"}}}));
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full"), makeGroup("full")}, {{0, {"full"}}}));
}

}  // namespace
}  // namespace rtp_llm
