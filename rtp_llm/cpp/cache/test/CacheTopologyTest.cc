#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/test/CacheConfigTestAccess.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"

namespace rtp_llm {
namespace {

// The draft module declares its own module-scoped tags; nothing is inferred.
std::shared_ptr<const CacheTopology> declareModuleScopedTags(const CacheTopology& topology) {
    auto groups = topology.groups();
    auto layers = topology.layers();
    for (auto& group : groups) {
        const std::string new_tag = "mtp_" + group.tag;
        auto              spec    = group.spec->clone();
        spec->tag                 = new_tag;
        group.spec                = std::move(spec);
        group.tag                 = new_tag;
    }
    for (auto& layer : layers) {
        for (auto& tag : layer.group_tags) {
            tag = "mtp_" + tag;
        }
    }
    return CacheTopology::create(std::move(groups), std::move(layers));
}

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

KVCacheSpecPtr makeLayoutSpec(const std::string& tag, int head_dim) {
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
    return MHAKVCacheSpec::build(desc, context);
}

GroupBase makeLayoutGroup(const std::string& tag, int head_dim) {
    auto group = makeGroup(tag);
    group.spec = makeLayoutSpec(tag, head_dim);
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

TEST(CacheTopologyTest, ResolvesLinearTransferFromLocalGeometryAndPreservesItOnClone) {
    ParallelismConfig parallelism;
    parallelism.tp_size = 2;
    LinearAttentionConfig linear;
    linear.linear_num_key_heads   = 4;
    linear.linear_num_value_heads = 8;
    linear.linear_key_head_dim    = 16;
    linear.linear_value_head_dim  = 16;
    linear.linear_conv_kernel_dim = 4;
    linear.ssm_state_dtype        = DataType::TYPE_FP32;
    linear.conv_state_dtype       = DataType::TYPE_BF16;
    SpecBuildContext context;
    context.parallelism_config      = &parallelism;
    context.linear_attention_config = &linear;
    KVCacheSpecDesc desc;
    desc.tag              = "arbitrary_state_tag";
    desc.cache_type       = KVCacheSpecType::LinearAttention;
    desc.dtype            = DataType::TYPE_BF16;
    desc.tp_transfer_mode = TpCacheTransferMode::LINEAR_QKV;
    const auto  spec      = SpecBuilder::build(desc, context);
    const auto& layout    = spec->tp_transfer;
    EXPECT_EQ(layout.key_heads, 2u);
    EXPECT_EQ(layout.value_heads, 4u);
    EXPECT_EQ(layout.q_row_bytes, 64u);
    EXPECT_EQ(layout.k_row_bytes, 64u);
    EXPECT_EQ(layout.v_row_bytes, 128u);
    EXPECT_EQ(layout.history_rows, 3u);
    EXPECT_EQ(spec->k_block_size_bytes(), 4096u);
    EXPECT_EQ(spec->v_block_size_bytes(), 768u);
    auto plan = layout.plan(spec->k_block_size_bytes(), spec->v_block_size_bytes(), 4, {2, 2, 1, 1}, 1);
    plan.validateSlices();
    EXPECT_EQ(plan.source_bytes, 2432u);
    EXPECT_EQ(plan.destination_bytes, 4864u);
    const auto clone = spec->clone();
    EXPECT_EQ(clone->tp_transfer.mode, TpCacheTransferMode::LINEAR_QKV);
    EXPECT_EQ(clone->fingerprint(), spec->fingerprint());
    clone->tp_transfer.mode = TpCacheTransferMode::LEGACY;
    EXPECT_NE(clone->fingerprint(), spec->fingerprint());
    desc.tp_transfer_mode = TpCacheTransferMode::LEGACY;
    const auto legacy     = SpecBuilder::build(desc, context);
    EXPECT_EQ(legacy->tp_transfer.q_row_bytes, 0u);
    desc.tp_transfer_mode = TpCacheTransferMode::MHA_HEADS;
    EXPECT_ANY_THROW(SpecBuilder::build(desc, context));
    desc.tp_transfer_mode = TpCacheTransferMode::REPLICATED;
    EXPECT_ANY_THROW(SpecBuilder::build(desc, context));
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

TEST(CacheTopologyTest, GroupOrderIsCanonicalRegardlessOfConstructionOrder) {
    auto first    = CacheTopology::create({makeGroup("full"), makeGroup("linear", CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});
    auto reversed = CacheTopology::create({makeGroup("linear", CacheGroupType::LINEAR), makeGroup("full")},
                                          {{0, {"full", "linear"}}});

    EXPECT_EQ(first->groupTags(), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(reversed->groupTags(), first->groupTags());
    EXPECT_EQ(reversed->groupIdForTag("full"), 0u);
    EXPECT_EQ(reversed->groupIdForTag("linear"), 1u);
    EXPECT_ANY_THROW(first->group("missing"));
    EXPECT_EQ(first->group("full").policy.group_type, reversed->group("full").policy.group_type);
    EXPECT_EQ(first->group("linear").policy.group_type, reversed->group("linear").policy.group_type);
    EXPECT_EQ(first->groupForLayer(0, "full").tag, reversed->groupForLayer(0, "full").tag);
    EXPECT_EQ(first->groupForLayer(0, "linear").tag, reversed->groupForLayer(0, "linear").tag);
}

TEST(CacheTopologyTest, LinearReplayFollowsTagWhenGroupRowChanges) {
    CacheConfig config;
    EXPECT_TRUE(config.allGroupTags().empty());
    const auto linear                = makeGroup("linear", CacheGroupType::LINEAR);
    config.setTopology(0, {linear}, {{0, {"linear"}}});
    config.linear_replay.group_tags = {"linear"};
    ASSERT_EQ(config.topology().groupIdForTag("linear"), 0u);

    config.setTopology(0, {linear, makeGroup("full")}, {{0, {"linear", "full"}}});
    ASSERT_EQ(config.topology().groupIdForTag("linear"), 1u);
    EXPECT_TRUE(config.isLinearReplayGroup("linear"));
    EXPECT_FALSE(config.isLegacyLinearGroup("linear"));
    EXPECT_FALSE(config.isLinearReplayGroup("full"));
}

TEST(CacheTopologyTest, PhysicalGroupResolvesSingleMtpModuleLayerRange) {
    CacheConfig config;
    config.setTopology(0, {makeGroup("full")}, {{0, {"full"}}});

    CacheConfig draft;
    draft.setTopology(0, {makeGroup("full")}, {{0, {"full"}}, {1, {"full"}}});
    ::rtp_llm::test::CacheConfigTestAccess::addMtpModule(
        config, declareModuleScopedTags(draft.topology(0)), draft.moduleMetadata(0));

    EXPECT_EQ(&config.topology(0).groupForLayer(0, "full"), &config.topology(0).group("full"));
    const auto& mtp_group = config.topology(1).group("mtp_full");
    EXPECT_EQ(&config.topology(1).groupForLayer(0, "mtp_full"), &mtp_group);
    EXPECT_EQ(&config.topology(1).groupForLayer(1, "mtp_full"), &mtp_group);
    EXPECT_ANY_THROW((void)config.topology(1).groupForLayer(2, "mtp_full"));
}

TEST(CacheTopologyTest, GroupTagsUseCanonicalOrderAndStableStorage) {
    auto        topology = CacheTopology::create({makeGroup("zeta"), makeGroup("alpha")}, {{0, {"zeta", "alpha"}}});
    const auto& tags     = topology->groupTags();
    EXPECT_EQ(tags, (std::vector<std::string>{"alpha", "zeta"}));
    EXPECT_EQ(topology->layer(0).group_tags, (std::vector<std::string>{"zeta", "alpha"}));
    EXPECT_EQ(&tags, &topology->groupTags());
    EXPECT_EQ(tags.data(), topology->groupTags().data());
    for (size_t row = 0; row < tags.size(); ++row) {
        EXPECT_EQ(tags[row], topology->groups()[row].tag);
    }
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("alpha"), makeGroup("alpha")}, {{0, {"alpha"}}}));
}

TEST(CacheTopologyTest, GroupedSpecsKeepGeometryPoliciesAndMembershipWithCanonicalTags) {
    const auto  zeta  = makeLayoutGroup("zeta", 2);
    const auto  alpha = makeLayoutGroup("alpha", 3);
    CacheConfig config;
    config.fromGroupedSpecs(0,
                            {zeta.spec->clone(), alpha.spec->clone()},
                            /*layer_num=*/2,
                            {{1}, {0}},
                            {CacheGroupType::SWA, CacheGroupType::FULL},
                            {"zeta", "alpha"});

    EXPECT_EQ(config.groupTags(), (std::vector<std::string>{"alpha", "zeta"}));
    EXPECT_EQ(config.topology().groupIdForTag("alpha"), 0u);
    EXPECT_EQ(config.topology().groupIdForTag("zeta"), 1u);
    EXPECT_EQ(config.group("alpha").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group("zeta").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("alpha").kvBlockStrideBytes(), alpha.kvBlockStrideBytes());
    EXPECT_EQ(config.group("zeta").kvBlockStrideBytes(), zeta.kvBlockStrideBytes());
    EXPECT_EQ(config.groupForLayer(0, "alpha").tag, "alpha");
    EXPECT_EQ(config.groupForLayer(1, "zeta").tag, "zeta");
}

TEST(CacheTopologyTest, TaggedLayoutPreservesOrderAndRetainedTopology) {
    auto        first  = makeLayoutGroup("zeta", 2);
    auto        second = makeLayoutGroup("alpha", 3);
    CacheConfig config;
    config.setTopology(0, {first, second}, {{0, {"zeta", "alpha"}}});
    const auto  retained      = config.topologyPtr();
    const auto& retained_tags = retained->groupTags();
    EXPECT_EQ(&config.groupTags(), &retained_tags);

    config.setGroupBlockLayout({"zeta", "alpha"},
                               {31, 23},
                               {first.kvBlockStrideBytes(), second.kvBlockStrideBytes()},
                               {first.kvScaleStrideBytes(), second.kvScaleStrideBytes()});
    EXPECT_EQ(config.groupTags(), (std::vector<std::string>{"alpha", "zeta"}));
    EXPECT_EQ(config.group("alpha").block_num, 23u);
    EXPECT_EQ(config.group("zeta").block_num, 31u);
    EXPECT_EQ(config.group("alpha").kvBlockStrideBytes(), second.kvBlockStrideBytes());
    EXPECT_EQ(retained->group("alpha").block_num, 16u);
    EXPECT_EQ(retained_tags, (std::vector<std::string>{"alpha", "zeta"}));
    EXPECT_EQ(&retained_tags, &retained->groupTags());
    EXPECT_NE(config.topologyPtr(), retained);
}

TEST(CacheTopologyTest, TaggedLayoutKeepsMtpPhysicalGeometry) {
    CacheConfig config;
    config.setTopology(0, {makeLayoutGroup("full", 2)}, {{0, {"full"}}});
    CacheConfig draft;
    draft.setTopology(0, {makeLayoutGroup("full", 5)}, {{0, {"full"}}});
    const auto child = declareModuleScopedTags(draft.topology(0));
    ::rtp_llm::test::CacheConfigTestAccess::addMtpModule(config, child, draft.moduleMetadata(0));
    const auto main_stride  = config.topology(0).group("full").kvBlockStrideBytes();
    const auto draft_stride = config.topology(1).group("mtp_full").kvBlockStrideBytes();
    const auto main_bytes   = config.topology(0).blockSizeBytesForGroup("full");
    const auto draft_bytes  = config.topology(1).blockSizeBytesForGroup("mtp_full");
    ASSERT_NE(main_stride, draft_stride);
    config.setGroupBlockLayout({"full"}, {23}, {main_stride}, {config.group("full").kvScaleStrideBytes()});
    EXPECT_EQ(config.group("full").block_num, 23u);
    EXPECT_EQ(config.topology(1).group("mtp_full").kvBlockStrideBytes(), draft_stride);
    EXPECT_EQ(config.topology(0).blockSizeBytesForGroup("full"), main_bytes);
    EXPECT_EQ(config.topology(1).blockSizeBytesForGroup("mtp_full"), draft_bytes);
    EXPECT_EQ(config.topologyPtr(1), child);
}

TEST(CacheTopologyTest, InvalidTaggedLayoutsNeverPublishPartialUpdates) {
    CacheConfig config;
    config.setTopology(0, {makeLayoutGroup("first", 2), makeLayoutGroup("second", 2)}, {{0, {"first", "second"}}});
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

TEST(CacheTopologyTest, PublicationFreezesMutableSpecAlias) {
    auto    spec  = std::dynamic_pointer_cast<MHAKVCacheSpec>(makeLayoutSpec("full", 2));
    GroupBase group = makeGroup("full");
    group.spec      = spec;
    CacheConfig config;
    config.setTopology(0, {group}, {{0, {"full"}}});
    const auto published_stride = config.group("full").kvBlockStrideBytes();

    spec->per_token_k_elems = spec->per_token_k_elems * 3;
    EXPECT_EQ(config.group("full").kvBlockStrideBytes(), published_stride);
}

TEST(CacheTopologyTest, CapacityUpdateReusesFrozenSpecPointers) {
    auto first  = makeLayoutGroup("first", 2);
    auto second = makeLayoutGroup("second", 3);
    CacheConfig config;
    config.setTopology(0, {first, second}, {{0, {"first", "second"}}});
    const auto before          = config.topologyPtr();
    const auto first_spec      = before->group("first").spec;
    const auto second_spec     = before->group("second").spec;
    const auto first_stride    = config.group("first").kvBlockStrideBytes();
    const auto second_stride   = config.group("second").kvBlockStrideBytes();
    const auto first_scale     = config.group("first").kvScaleStrideBytes();
    const auto second_scale    = config.group("second").kvScaleStrideBytes();

    config.setGroupBlockLayout({"first", "second"}, {21, 22}, {first_stride, second_stride},
                               {first_scale, second_scale});
    const auto after = config.topologyPtr();
    ASSERT_NE(after, before);
    EXPECT_EQ(after->group("first").spec, first_spec);
    EXPECT_EQ(after->group("second").spec, second_spec);
    EXPECT_EQ(after->group("first").block_num, 21u);
    EXPECT_EQ(after->group("second").block_num, 22u);
    EXPECT_EQ(before->group("first").block_num, 16u);
}

}  // namespace
}  // namespace rtp_llm
