#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

namespace {

using TestDSV4HybridTypeAllocator = test::BlockTreeCacheTestAllocator<HybridTypeKVCacheAllocator>;
using TestDSV4HybridPoolAllocator = test::BlockTreeCacheTestAllocator<HybridPoolKVCacheAllocator>;

constexpr int                  kDsv4PoolNum                = 7;
constexpr uint32_t             kDsv4TokensPerBlock         = 128;
constexpr uint32_t             kDsv4KvEntryBytes           = 1024;
constexpr uint32_t             kDsv4IndexerEntryBytes      = 256;
constexpr uint32_t             kDsv4Fp8KvEntryBytes        = 584;
constexpr uint32_t             kDsv4IndexerStateEntryBytes = 512 * 4;
constexpr uint32_t             kDsv4CsaStateEntryBytes     = 2048 * 4;
constexpr uint32_t             kDsv4HcaStateEntryBytes     = 1024 * 4;
const std::vector<std::string> kDsv4FlashFirstSeenTags     = {
    "swa_kv", "csa_kv", "indexer_kv", "indexer_state", "csa_state", "hca_kv", "hca_state"};
const std::vector<std::string> kDsv4ProFirstSeenTags = {
    "hca_kv", "hca_state", "swa_kv", "csa_kv", "indexer_kv", "indexer_state", "csa_state"};

std::shared_ptr<CompressedKVCacheSpec> buildCompressedSpec(const std::string& tag,
                                                           uint32_t           entry_elems,
                                                           uint32_t           entries_per_block,
                                                           DataType           dtype,
                                                           uint32_t           compression_ratio          = 1,
                                                           size_t             block_size_bytes_alignment = 0) {
    KVCacheSpecDesc desc;
    desc.tag                          = tag;
    desc.cache_type                   = KVCacheSpecType::OpaqueKV;
    desc.dtype                        = dtype;
    desc.entry_elems                  = entry_elems;
    desc.entry_dtype                  = dtype;
    desc.compression_ratio            = compression_ratio;
    desc.block_stride_bytes_alignment = block_size_bytes_alignment;
    desc.entry_count_mode             = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    desc.is_state_cache               = false;
    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = kDsv4TokensPerBlock;
    ctx.kernel_tokens_per_block = entries_per_block * compression_ratio;
    return std::dynamic_pointer_cast<CompressedKVCacheSpec>(SpecBuilder::build(desc, ctx));
}

std::shared_ptr<FixedStateCacheSpec> buildFixedStateSpec(const std::string& tag,
                                                         uint32_t           entry_elems,
                                                         uint32_t           entries_per_block,
                                                         DataType           dtype,
                                                         size_t             block_size_bytes_override        = 0,
                                                         size_t             block_size_bytes_alignment       = 0,
                                                         uint32_t           block_size_alignment_min_entries = 0) {
    KVCacheSpecDesc desc;
    desc.tag                                = tag;
    desc.cache_type                         = KVCacheSpecType::OpaqueState;
    desc.dtype                              = dtype;
    desc.entry_elems                        = entry_elems;
    desc.explicit_entry_count               = entries_per_block;
    desc.entry_dtype                        = dtype;
    desc.block_stride_bytes_override        = block_size_bytes_override;
    desc.block_stride_bytes_alignment       = block_size_bytes_alignment;
    desc.block_stride_alignment_min_entries = block_size_alignment_min_entries;
    desc.is_state_cache                     = true;
    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = kDsv4TokensPerBlock;
    return std::dynamic_pointer_cast<FixedStateCacheSpec>(SpecBuilder::build(desc, ctx));
}

static size_t gidForTag(const CacheConfig& config, const std::string& tag) {
    return static_cast<size_t>(config.groupIdForTag(tag));
}

static size_t opaqueEntriesPerBlock(const OpaqueKVCacheSpec& spec, size_t entry_bytes) {
    RTP_LLM_CHECK_WITH_INFO(entry_bytes > 0, "entry_bytes must be > 0");
    RTP_LLM_CHECK_WITH_INFO(spec.block_payload_bytes() % entry_bytes == 0,
                            "opaque payload bytes %zu must be divisible by entry bytes %zu",
                            spec.block_payload_bytes(),
                            entry_bytes);
    return spec.block_payload_bytes() / entry_bytes;
}

static size_t stateEntryBytesForTag(std::string_view tag) {
    if (tag == "indexer_state") {
        return kDsv4IndexerStateEntryBytes;
    }
    if (tag == "csa_state") {
        return kDsv4CsaStateEntryBytes;
    }
    if (tag == "hca_state") {
        return kDsv4HcaStateEntryBytes;
    }
    RTP_LLM_FAIL("unexpected DSV4 state tag: %s", std::string(tag).c_str());
    return 0;
}

static CacheConfig makeSingleStateCpConfig(const KVCacheSpec& spec, int cp_size) {
    CacheConfig config;
    config.seq_size_per_block = std::max<size_t>(1, spec.seq_size_per_block / static_cast<size_t>(cp_size));
    config.layer_num          = 1;
    config.layer_all_num      = 1;
    GroupBase group;
    group.tag                        = spec.tag;
    group.spec                       = spec.clone();
    group.policy                     = defaultCacheGroupPolicy(CacheGroupType::SWA);
    group.policy.enable_prefix_reuse = true;
    group.policy.cp_slice = spec.block_size_bytes() == spec.block_payload_bytes() ? CpBlockSliceMode::PAYLOAD_BYTES :
                                                                                    CpBlockSliceMode::EQUAL_BYTES;
    group.layer_ids.push_back(0);
    LayerBase layer;
    layer.layer_id   = 0;
    layer.group_tags = {spec.tag};
    config.setTopology({std::move(group)}, {std::move(layer)});
    return config;
}

static std::vector<BlockInfo>
sliceStateBlockForPeer(const KVCacheSpec& spec, std::vector<BlockInfo> parts, int cp_size, size_t peer_idx) {
    auto         config = makeSingleStateCpConfig(spec, cp_size);
    CPSlotMapper mapper(0, cp_size, static_cast<int>(config.seq_size_per_block));
    return mapper.sliceBlockForPeer(config, /*gid=*/0, std::move(parts), peer_idx);
}

static std::vector<CacheStoreBlockPair>
buildSwaStorePlan(size_t total_logical_blocks, size_t reuse_block_size, bool use_hybrid, int cp_size) {
    auto         spec   = makeResolvedOpaqueSpec(/*state_cache=*/true, "swa", DataType::TYPE_UINT8, 2, 1);
    auto         config = makeSingleStateCpConfig(*spec, cp_size);
    CPSlotMapper mapper(/*cp_rank=*/0, cp_size, static_cast<int>(config.seq_size_per_block));
    return mapper.buildStorePlan(config, /*gid=*/0, total_logical_blocks, reuse_block_size, use_hybrid);
}

class DSV4CacheTestEnvironment: public ::testing::Environment {
public:
    void SetUp() override {
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
    }

    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
    }

private:
    bool old_core_dump_on_exception_{false};
};

[[maybe_unused]] auto* const dsv4_cache_test_env = ::testing::AddGlobalTestEnvironment(new DSV4CacheTestEnvironment());

}  // namespace

static void setGroupBlockNumsForTest(CacheConfig& config, const std::vector<uint32_t>& block_nums) {
    std::vector<size_t> kv_strides;
    std::vector<size_t> scale_strides;
    kv_strides.reserve(static_cast<size_t>(config.groupNums()));
    scale_strides.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        kv_strides.push_back(config.kvBlockStrideBytesForGroup(gid));
        scale_strides.push_back(config.kvScaleStrideBytesForGroup(gid));
    }
    config.setGroupBlockLayout(block_nums, kv_strides, scale_strides);
}

static void initDsv4BatchGroups(BatchKVCacheResource& batch_res, const CacheConfig& config) {
    batch_res.initGroups(config.topologyPtr());
}

static ModelConfig makeProModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 61;
    mc.hidden_size                  = 7168;
    mc.attn_config.head_num         = 128;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 1024;
    mc.attn_config.o_groups         = 16;
    mc.attn_config.o_lora_rank      = 1024;
    mc.attn_config.tokens_per_block = kDsv4TokensPerBlock;
    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < 61; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    mc.attn_config.layer_compress_ratios                         = ratios;
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(mc);
    return mc;
}

static ModelConfig makeFlashModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 43;
    mc.hidden_size                  = 4096;
    mc.attn_config.head_num         = 64;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 512;
    mc.attn_config.o_groups         = 8;
    mc.attn_config.o_lora_rank      = 1024;
    mc.attn_config.tokens_per_block = kDsv4TokensPerBlock;
    std::vector<int> ratios         = {0, 0};
    for (int i = 2; i < 43; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    mc.attn_config.layer_compress_ratios                         = ratios;
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(mc);
    return mc;
}

static ModelConfig makeFlashMtpModelConfig() {
    ModelConfig mc                       = makeFlashModelConfig();
    mc.num_layers                        = 1;
    mc.attn_config.layer_compress_ratios = {0};
    setDsv4KvCacheSpecs(mc);
    return mc;
}

static ModelConfig makeHybridAttentionModelConfig(bool independent_pool) {
    ModelConfig mc;
    mc.num_layers                                                = 4;
    mc.hidden_size                                               = 128;
    mc.attn_config.head_num                                      = 4;
    mc.attn_config.kv_head_num                                   = 2;
    mc.attn_config.size_per_head                                 = independent_pool ? 16 : 32;
    mc.attn_config.tokens_per_block                              = 8;
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = independent_pool;
    mc.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    mc.linear_attention_config.linear_conv_kernel_dim = 4;
    mc.linear_attention_config.linear_key_head_dim    = 16;
    mc.linear_attention_config.linear_value_head_dim  = 16;
    mc.linear_attention_config.linear_num_key_heads   = 2;
    mc.linear_attention_config.linear_num_value_heads = 2;
    setHybridAttentionKvCacheSpecs(mc);
    return mc;
}

// ============================================================
// Layer classification
// ============================================================

TEST(HybridPoolConfigCreatorTest, ProLayerClassification) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, false, 0);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_EQ(config.groupTagsSnapshot(), kDsv4ProFirstSeenTags);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 31u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 61u);
}

TEST(HybridPoolConfigCreatorTest, FlashLayerClassification) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, false, 0);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(config.groupTagsSnapshot(), kDsv4FlashFirstSeenTags);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 43u);
}

TEST(HybridPoolConfigCreatorTest, ProAndFlashPagedBytesUseEachGroupsLayerOwnership) {
    for (bool use_flash : {false, true}) {
        ParallelismConfig pc;
        auto              config = CacheConfigCreator::createBasicConfig(
            use_flash ? makeFlashModelConfig() : makeProModelConfig(), pc, false, 0);

        size_t expected_paged_bytes = 0;
        for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
            const size_t expected_group_bytes =
                config.layerIdsForGroup(gid).size()
                * (config.kvBlockStrideBytesForGroup(gid) + config.kvScaleStrideBytesForGroup(gid));
            EXPECT_EQ(config.blockSizeBytesForGroup(gid), expected_group_bytes)
                << "use_flash=" << use_flash << " gid=" << gid;
            if (!config.usesExplicitIndependentBlocks(gid)
                && (config.typeForGroup(gid) == CacheGroupType::FULL
                    || config.typeForGroup(gid) == CacheGroupType::LINEAR)) {
                expected_paged_bytes += expected_group_bytes;
            }
        }
        EXPECT_EQ(config.block_size_bytes, expected_paged_bytes) << "use_flash=" << use_flash;
    }
}

TEST(HybridPoolConfigCreatorTest, MtpSwaOnlyLayerIsNotStripped) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeFlashMtpModelConfig(), pc, true, 0);

    EXPECT_EQ(config.layer_num, 1u);
    EXPECT_EQ(config.block_size_bytes, 1u);
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 1u);
    ASSERT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")), std::vector<int>({0}));
    ASSERT_EQ(config.layerGroupIdsSnapshot().size(), 1u);
    EXPECT_EQ(config.layerGroupIdsSnapshot()[0], std::vector<int>({0}));
    EXPECT_EQ(config.tagForGroup(0), "swa_kv");
    EXPECT_EQ(config.groupIdForLayerTag(0, "swa_kv"), 0);
}

TEST(HybridPoolConfigCreatorTest, Dsv4SpecOrderControlsFirstSeenGroupOrder) {
    auto mc = makeFlashModelConfig();
    for (auto& layer_descs : mc.kv_cache_spec_descs) {
        std::reverse(layer_descs.begin(), layer_descs.end());
    }

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    const std::vector<std::string> expected_tags = {
        "swa_kv", "csa_state", "indexer_state", "indexer_kv", "csa_kv", "hca_state", "hca_kv"};
    EXPECT_EQ(config.groupTagsSnapshot(), expected_tags);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), expected_tags.size());
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), expected_tags.size());
    for (size_t gid = 0; gid < expected_tags.size(); ++gid) {
        ASSERT_NE(config.specForGroup(gid), nullptr);
        EXPECT_EQ(config.specForGroup(gid)->tag, expected_tags[gid]) << "gid=" << gid;
    }

    EXPECT_EQ(config.groupIdForLayerTag(2, "csa_kv"), config.groupIdForTag("csa_kv"));
    EXPECT_EQ(config.groupIdForLayerTag(3, "hca_kv"), config.groupIdForTag("hca_kv"));
    EXPECT_EQ(config.groupIdForLayerTag(0, "swa_kv"), config.groupIdForTag("swa_kv"));
}

static GroupBase makeTestGroup(const KVCacheSpecPtr& spec, CacheGroupType type, std::vector<int> layer_ids) {
    GroupBase group;
    group.tag       = spec->tag;
    group.spec      = spec;
    group.policy    = defaultCacheGroupPolicy(type);
    group.layer_ids = std::move(layer_ids);
    return group;
}

TEST(CacheConfigTest, SetTopologyInstallsTagAndGroupTopology) {
    CacheConfig config;
    config.layer_num     = 3;
    config.layer_all_num = 3;

    auto swa_spec =
        std::dynamic_pointer_cast<FixedStateCacheSpec>(makeResolvedOpaqueSpec(true, "swa", DataType::TYPE_UINT8, 2, 1));
    auto csa_spec = std::dynamic_pointer_cast<CompressedKVCacheSpec>(
        makeResolvedOpaqueSpec(false, "csa", DataType::TYPE_UINT8, 2, 1));

    std::vector<LayerBase> layers = {{0, {"swa"}}, {1, {"swa", "csa"}}, {2, {"swa"}}};

    config.setTopology(
        {makeTestGroup(swa_spec, CacheGroupType::SWA, {0, 1, 2}), makeTestGroup(csa_spec, CacheGroupType::FULL, {1})},
        std::move(layers));

    EXPECT_EQ(config.groupTagsSnapshot(), std::vector<std::string>({"swa", "csa"}));
    EXPECT_EQ(config.groupIdForLayerTag(1, "swa"), 0);
    EXPECT_EQ(config.groupIdForLayerTag(1, "csa"), 1);
    EXPECT_THROW((void)config.groupIdFor(1), std::exception);
    EXPECT_EQ(config.layerGroupIdsSnapshot()[1], std::vector<int>({0, 1}));
}

TEST(CacheConfigTest, TopologyRemainsTheSingleSourceAcrossSupportedUpdates) {
    CacheConfig config;
    config.layer_num     = 2;
    config.layer_all_num = 2;

    auto full_spec   = std::make_shared<MHAKVCacheSpec>();
    full_spec->tag   = "full";
    auto linear_spec = std::make_shared<LinearKVCacheSpec>();
    linear_spec->tag = "linear";

    config.setTopology(
        {makeTestGroup(full_spec, CacheGroupType::FULL, {0}), makeTestGroup(linear_spec, CacheGroupType::LINEAR, {1})},
        {{0, {"full"}}, {1, {"linear"}}});
    const auto initial_topology = config.topologyPtr();

    auto policies                   = config.groupPoliciesSnapshot();
    policies[0].enable_prefix_reuse = !policies[0].enable_prefix_reuse;
    config.setGroupPolicies(policies);
    const auto policy_topology = config.topologyPtr();

    EXPECT_NE(policy_topology.get(), initial_topology.get());
    EXPECT_EQ(config.policyForGroup(0).enable_prefix_reuse, policies[0].enable_prefix_reuse);
    EXPECT_EQ(config.group("full").policy.enable_prefix_reuse, policies[0].enable_prefix_reuse);
    EXPECT_NE(initial_topology->group("full").policy.enable_prefix_reuse, policies[0].enable_prefix_reuse);

    config.setGroupBlockLayout({17, 9}, {128, 256}, {4, 8});
    const auto layout_topology = config.topologyPtr();

    EXPECT_NE(layout_topology.get(), policy_topology.get());
    EXPECT_EQ(config.groupBlockNumsSnapshot(), (std::vector<uint32_t>{17, 9}));
    EXPECT_EQ(config.groupKvBlockStrideBytesSnapshot(), (std::vector<size_t>{128, 256}));
    EXPECT_EQ(config.groupKvScaleStrideBytesSnapshot(), (std::vector<size_t>{4, 8}));
    EXPECT_EQ(config.group("linear").block_num, 9u);
    EXPECT_EQ(config.group("linear").kv_block_stride_bytes, 256u);
    EXPECT_EQ(policy_topology->group("linear").block_num, 0u);

    config.finalizeBlockNums(/*global_block_num=*/23, RuntimeConfig{});
    const auto finalized_topology = config.topologyPtr();

    EXPECT_NE(finalized_topology.get(), layout_topology.get());
    EXPECT_EQ(config.groupBlockNumsSnapshot(), (std::vector<uint32_t>{23, 23}));
    EXPECT_EQ(config.blockNumForGroup(0), config.group("full").block_num);
    EXPECT_EQ(config.blockNumForGroup(1), config.group("linear").block_num);
    EXPECT_EQ(layout_topology->group("full").block_num, 17u);
    EXPECT_EQ(layout_topology->group("linear").block_num, 9u);
}

TEST(CacheConfigTest, SetTopologyRejectsMissingLayer) {
    CacheConfig config;
    config.layer_num     = 2;
    config.layer_all_num = 2;

    auto spec                     = std::make_shared<MHAKVCacheSpec>();
    spec->tag                     = "default";
    std::vector<LayerBase> layers = {{0, {"default"}}, {1, {}}};
    EXPECT_THROW(config.setTopology({makeTestGroup(spec, CacheGroupType::FULL, {0})}, std::move(layers)),
                 std::exception);
}

TEST(CacheConfigTest, SetTopologyRejectsEmptyTag) {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;

    auto                   spec   = std::make_shared<MHAKVCacheSpec>();
    std::vector<LayerBase> layers = {{0, {""}}};
    EXPECT_THROW(config.setTopology({makeTestGroup(spec, CacheGroupType::FULL, {0})}, std::move(layers)),
                 std::exception);
}

TEST(CacheConfigTest, SetTopologyRejectsDuplicateGroupTag) {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;

    auto spec0 = std::make_shared<MHAKVCacheSpec>();
    spec0->tag = "dup";
    auto spec1 = std::make_shared<MHAKVCacheSpec>();
    spec1->tag = "dup";

    std::vector<LayerBase> layers = {{0, {"dup", "dup"}}};
    EXPECT_THROW(config.setTopology({makeTestGroup(spec0, CacheGroupType::FULL, {0}),
                                     makeTestGroup(spec1, CacheGroupType::LINEAR, {0})},
                                    std::move(layers)),
                 std::exception);
}

TEST(CacheConfigTest, SetTopologyAllowsDifferentLayerTags) {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;

    auto spec0 = std::make_shared<MHAKVCacheSpec>();
    spec0->tag = "full";
    auto spec1 = std::make_shared<MHAKVCacheSpec>();
    spec1->tag = "linear";

    std::vector<LayerBase> layers = {{0, {"full", "linear"}}};
    EXPECT_NO_THROW(config.setTopology(
        {makeTestGroup(spec0, CacheGroupType::FULL, {0}), makeTestGroup(spec1, CacheGroupType::LINEAR, {0})},
        std::move(layers)));
    EXPECT_EQ(config.layerGroupIdsSnapshot()[0].size(), 2u);
}

TEST(HybridPoolConfigCreatorTest, Dsv4ModelProvidedAlignmentPropagatesToCacheSpecs) {
    auto mc = makeFlashModelConfig();
    for (auto& layer_descs : mc.kv_cache_spec_descs) {
        for (auto& desc : layer_descs) {
            if (desc.tag == "csa_kv") {
                desc.block_stride_bytes_alignment = 1024;
            } else if (desc.tag == "swa_kv") {
                desc.block_stride_bytes_alignment       = 2048;
                desc.block_stride_alignment_min_entries = 256;
            }
        }
    }

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    const auto* csa_kv =
        dynamic_cast<const CompressedKVCacheSpec*>(config.specForGroup(gidForTag(config, "csa_kv")).get());
    const auto* swa_kv =
        dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "swa_kv")).get());
    ASSERT_NE(csa_kv, nullptr);
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(csa_kv->block_size_bytes() % 1024u, 0u);
    EXPECT_EQ(swa_kv->block_size_bytes() % 2048u, 0u);
}

TEST(HybridPoolConfigCreatorTest, Dsv4TagRoutesAreConsistent) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, false, 0);

    auto expect_route = [&](int layer_id, const std::string& tag, int expected_gid) {
        EXPECT_EQ(config.groupIdForLayerTag(layer_id, tag), expected_gid) << "layer=" << layer_id << " tag=" << tag;
    };

    // Flash DSV4 test config uses layers 2,4,... as CSA and 3,5,... as HCA; 0/1 are SWA-only.
    expect_route(2, "csa_kv", config.groupIdForTag("csa_kv"));
    expect_route(2, "indexer_kv", config.groupIdForTag("indexer_kv"));
    expect_route(2, "indexer_state", config.groupIdForTag("indexer_state"));
    expect_route(2, "csa_state", config.groupIdForTag("csa_state"));
    expect_route(2, "swa_kv", config.groupIdForTag("swa_kv"));

    expect_route(3, "hca_kv", config.groupIdForTag("hca_kv"));
    expect_route(3, "hca_state", config.groupIdForTag("hca_state"));
    expect_route(3, "swa_kv", config.groupIdForTag("swa_kv"));

    expect_route(0, "swa_kv", config.groupIdForTag("swa_kv"));
    EXPECT_THROW(config.groupIdForLayerTag(0, "csa_kv"), std::exception);
    EXPECT_THROW(config.groupIdForLayerTag(0, "hca_kv"), std::exception);

    auto mtp_config = CacheConfigCreator::createBasicConfig(makeFlashMtpModelConfig(), pc, true, 0);
    ASSERT_EQ(mtp_config.groupIdForLayerTag(0, "swa_kv"), 0);
}

TEST(HybridPoolConfigCreatorTest, Dsv4GroupPoliciesMatchLegacyBehavior) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, false, 0);

    ASSERT_EQ(config.groupPoliciesSnapshot().size(), static_cast<size_t>(config.groupNums()));
    auto expect_policy =
        [&](const std::string& tag, bool enable_prefix_reuse, CacheEvictPolicy evict_policy, int active_tail_blocks) {
            const auto group_tags = config.groupTagsSnapshot();
            auto       it         = std::find(group_tags.begin(), group_tags.end(), tag);
            ASSERT_NE(it, group_tags.end()) << tag;
            const auto gid = static_cast<size_t>(std::distance(group_tags.begin(), it));
            EXPECT_EQ(config.policyForGroup(gid).enable_prefix_reuse, enable_prefix_reuse) << tag;
            EXPECT_EQ(config.policyForGroup(gid).evict_policy, evict_policy) << tag;
            EXPECT_EQ(config.policyForGroup(gid).active_tail_blocks, active_tail_blocks) << tag;
        };

    expect_policy("hca_state", false, CacheEvictPolicy::INDEPENDENT, 1);
    expect_policy("swa_kv", true, CacheEvictPolicy::INDEPENDENT, 2);
    expect_policy("csa_state", true, CacheEvictPolicy::INDEPENDENT, 2);
    expect_policy("csa_kv", true, CacheEvictPolicy::CHAIN, 0);
    expect_policy("hca_kv", true, CacheEvictPolicy::CHAIN, 0);
    expect_policy("indexer_kv", true, CacheEvictPolicy::CHAIN, 0);
}

TEST(HybridPoolConfigCreatorTest, SlidingWindowPolicyPropagatesAndSurvivesAggregation) {
    ParallelismConfig pc;
    auto              config   = CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, false, 0);
    auto              policies = config.groupPoliciesSnapshot();

    ASSERT_EQ(policies.size(), static_cast<size_t>(config.groupNums()));
    bool saw_swa     = false;
    bool saw_non_swa = false;
    for (size_t gid = 0; gid < policies.size(); ++gid) {
        const bool is_swa = policies[gid].group_type == CacheGroupType::SWA;
        EXPECT_EQ(policies[gid].sliding_window_size, is_swa ? 128 : 0) << "gid=" << gid;
        saw_swa |= is_swa;
        saw_non_swa |= !is_swa;
    }
    EXPECT_TRUE(saw_swa);
    EXPECT_TRUE(saw_non_swa);

    auto equal_policy = policies.front();
    EXPECT_TRUE(CacheConfig::samePolicy(policies.front(), equal_policy));
    equal_policy.sliding_window_size += 1;
    EXPECT_FALSE(CacheConfig::samePolicy(policies.front(), equal_policy));

    config.setGroupPolicies(policies);
    const auto aggregated = config.groupPoliciesSnapshot();
    ASSERT_EQ(aggregated.size(), policies.size());
    for (size_t gid = 0; gid < policies.size(); ++gid) {
        EXPECT_TRUE(CacheConfig::samePolicy(aggregated[gid], policies[gid])) << "gid=" << gid;
        EXPECT_EQ(aggregated[gid].sliding_window_size, policies[gid].sliding_window_size) << "gid=" << gid;
    }
}

TEST(HybridPoolConfigCreatorTest, Dsv4SpecsMissingFailsFastWithoutRatioFallback) {
    auto mc = makeFlashModelConfig();
    mc.kv_cache_spec_descs.clear();

    ParallelismConfig pc;
    EXPECT_THROW((void)CacheConfigCreator::createBasicConfig(mc, pc, false, 0), std::exception);
}

// ============================================================
// Pool specs
// ============================================================

TEST(HybridPoolConfigCreatorTest, ProPoolSpecs) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, false, 0);

    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 30u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "csa_kv")), CacheGroupType::FULL);

    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 31u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes(), 1u * kDsv4KvEntryBytes);

    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "indexer_kv")).size(), 30u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_kv"))->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);

    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "indexer_state")).size(), 30u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_state"))->block_size_bytes(), 8u * 512u * 4u);

    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_state")).size(), 30u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_state"))->block_size_bytes(), 8u * 2048u * 4u);

    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_state")).size(), 31u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_state"))->block_size_bytes(), 128u * 1024u * 4u);

    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 61u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(),
              kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(HybridPoolConfigCreatorTest, FlashPoolSpecs) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, false, 0);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 43u);
}

// ============================================================
// Block size bytes
// ============================================================

TEST(HybridPoolConfigCreatorTest, BlockSizeBytes) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, false, 0);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_kv"))->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_state"))->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_state"))->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_state"))->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(),
              kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(HybridPoolConfigCreatorTest, Fp8BlockSizeBytesUsePaddedPhysicalStride) {
    ParallelismConfig pc;
    auto              mc          = makeProModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    setDsv4KvCacheSpecs(mc);
    auto config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    ASSERT_EQ(config.groupKvBlockStrideBytesSnapshot().size(), 7u);

    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes(), 19008u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes(), 1152u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_kv"))->block_size_bytes(), 32u * 132u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(), 74880u);

    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "csa_kv")),
              config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes());
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "hca_kv")),
              config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes());
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "swa_kv")),
              config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes());
}

TEST(HybridPoolConfigCreatorTest, BasicConfigUsesModelDefaultPhysicalAndKernelBlockSize) {
    ParallelismConfig pc;
    auto              mc     = makeProModelConfig();
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);

    EXPECT_EQ(config.seq_size_per_block, kDsv4TokensPerBlock);
    EXPECT_EQ(config.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(config.kernelBlocksPerKvBlock(), 1u);

    const auto* csa_kv =
        dynamic_cast<const CompressedKVCacheSpec*>(config.specForGroup(gidForTag(config, "csa_kv")).get());
    const auto* hca_kv =
        dynamic_cast<const CompressedKVCacheSpec*>(config.specForGroup(gidForTag(config, "hca_kv")).get());
    const auto* idx_kv =
        dynamic_cast<const CompressedKVCacheSpec*>(config.specForGroup(gidForTag(config, "indexer_kv")).get());
    const auto* swa_kv =
        dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "swa_kv")).get());
    ASSERT_NE(csa_kv, nullptr);
    ASSERT_NE(hca_kv, nullptr);
    ASSERT_NE(idx_kv, nullptr);
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(csa_kv->block_size() / kDsv4KvEntryBytes, 32u);
    EXPECT_EQ(hca_kv->block_size() / DSV4_FP8_KV_ENTRY_BYTES, 1u);
    EXPECT_EQ(idx_kv->block_size() / kDsv4IndexerEntryBytes, 32u);
    EXPECT_EQ(opaqueEntriesPerBlock(*swa_kv, kDsv4KvEntryBytes), 128u);

    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "csa_kv")),
              config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes());
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "hca_kv")),
              config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes());
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "indexer_kv")),
              config.specForGroup(gidForTag(config, "indexer_kv"))->block_size_bytes());
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "swa_kv")),
              config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes());

    auto full_pool = DeviceBlockPoolConfigHelper::createConfigForGroup(config, gidForTag(config, "csa_kv"));
    auto swa_pool  = DeviceBlockPoolConfigHelper::createConfigForGroup(config, gidForTag(config, "swa_kv"));
    ASSERT_EQ(full_pool.memory_layouts.size(), 1u);
    ASSERT_EQ(swa_pool.memory_layouts.size(), 1u);
    EXPECT_EQ(full_pool.memory_layouts[0].kernel_blocks_per_kv_block, 1u);
    EXPECT_EQ(swa_pool.memory_layouts[0].kernel_blocks_per_kv_block, 1u);
}

TEST(HybridPoolConfigCreatorTest, PrefillCpShardedSlicesFixedAndSwaPhysicalBlocks) {
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = 4;
    pc.prefill_cp_config.kv_cache_sharded = true;

    auto mc                       = makeProModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    setDsv4KvCacheSpecs(mc);
    auto config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    ASSERT_EQ(config.groupKvBlockStrideBytesSnapshot().size(), 7u);

    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes(), 19008u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes(), 1152u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_kv"))->block_size_bytes(), 32u * 132u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_state"))->block_size_bytes(), 2u * 512u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_state"))->block_size_bytes(), 2u * 2048u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_state"))->block_size_bytes(), 32u * 1024u * 4u);

    // SWA_KV keeps full logical ring entries for byte-sliced CP layout, but
    // each prefill rank stores only one aligned byte slice of the full block.
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(), 18720u);
    for (const auto& tag : {"indexer_state", "csa_state", "hca_state", "swa_kv"}) {
        const auto gid = gidForTag(config, tag);
        EXPECT_EQ(config.kvBlockStrideBytesForGroup(gid), config.specForGroup(gid)->block_size_bytes());
    }

    pc.role_type       = RoleType::DECODE;
    auto decode_config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    EXPECT_EQ(decode_config.specForGroup(gidForTag(decode_config, "indexer_state"))->block_size_bytes(),
              8u * 512u * 4u);
    EXPECT_EQ(decode_config.specForGroup(gidForTag(decode_config, "csa_state"))->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(decode_config.specForGroup(gidForTag(decode_config, "hca_state"))->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(decode_config.specForGroup(gidForTag(decode_config, "swa_kv"))->block_size_bytes(), 74880u);
}

TEST(CPSlotMapperTest, CpCompactSwaUsesCanonicalTailRows) {
    auto plan = buildSwaStorePlan(/*total_logical_blocks=*/8,
                                  /*reuse_block_size=*/0,
                                  /*use_hybrid=*/true,
                                  /*cp_size=*/4);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].key_index, 3);
    EXPECT_EQ(plan[0].offset_index, 0);
    EXPECT_EQ(plan[1].key_index, 7);
    EXPECT_EQ(plan[1].offset_index, 1);
}

TEST(CPSlotMapperTest, CpCompactSwaKeepsPartialTailRows) {
    {
        auto plan = buildSwaStorePlan(/*total_logical_blocks=*/1,
                                      /*reuse_block_size=*/0,
                                      /*use_hybrid=*/true,
                                      /*cp_size=*/2);
        ASSERT_EQ(plan.size(), 1u);
        EXPECT_EQ(plan[0].key_index, 0);
        EXPECT_EQ(plan[0].offset_index, 0);
    }
    {
        auto plan = buildSwaStorePlan(/*total_logical_blocks=*/11,
                                      /*reuse_block_size=*/0,
                                      /*use_hybrid=*/true,
                                      /*cp_size=*/2);
        ASSERT_EQ(plan.size(), 2u);
        EXPECT_EQ(plan[0].key_index, 9);
        EXPECT_EQ(plan[0].offset_index, 4);
        EXPECT_EQ(plan[1].key_index, 10);
        EXPECT_EQ(plan[1].offset_index, 5);
    }
}

// ============================================================
// CacheConfig output
// ============================================================

TEST(HybridPoolConfigCreatorTest, CreateCacheConfig) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    // 7 groups -> groupNums() > 1 -> HybridTypeKVCacheAllocator path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_TRUE(config.is_sparse);
    EXPECT_FALSE(config.use_mla);
}

TEST(HybridPoolConfigCreatorTest, FlashCacheConfig) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 43u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 21u);
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionIndependentPoolUsesHybridPoolConfig) {
    ParallelismConfig pc;
    auto config = CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(true), pc, false, 0);

    EXPECT_TRUE(config.use_independent_block_pools);
    ASSERT_EQ(config.groupNums(), 2);
    const auto group_types = config.groupTypesSnapshot();
    EXPECT_EQ(std::count(group_types.begin(), group_types.end(), CacheGroupType::FULL), 1);
    EXPECT_EQ(std::count(group_types.begin(), group_types.end(), CacheGroupType::LINEAR), 1);
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 2u);
    EXPECT_GT(config.specForGroup(gidForTag(config, "full"))->block_size_bytes(), 0u);
    EXPECT_GT(config.specForGroup(gidForTag(config, "linear"))->block_size_bytes(), 0u);
    EXPECT_NE(config.specForGroup(gidForTag(config, "full"))->block_size_bytes(),
              config.specForGroup(gidForTag(config, "linear"))->block_size_bytes());
    EXPECT_EQ(config.groupBlockNumsSnapshot().size(), 2u);
    EXPECT_EQ(config.groupTagsSnapshot(), std::vector<std::string>({"linear", "full"}));

    const auto linear_gid = gidForTag(config, "linear");
    const auto full_gid   = gidForTag(config, "full");
    EXPECT_EQ(config.block_size_bytes,
              config.blockSizeBytesForGroup(linear_gid) + config.blockSizeBytesForGroup(full_gid));

    RuntimeConfig runtime_config;
    config.linear_step = 4;
    config.finalizeBlockNums(/*global_block_num=*/37, runtime_config);
    EXPECT_EQ(config.blockNumForGroup(linear_gid), 37u);
    EXPECT_EQ(config.blockNumForGroup(full_gid), 37u);
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionIndependentPoolSplitsFullAndSwaSpecs) {
    auto mc                                           = makeHybridAttentionModelConfig(true);
    mc.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE,
                                                         HybridAttentionType::SLIDING_WINDOW,
                                                         HybridAttentionType::LINEAR,
                                                         HybridAttentionType::SLIDING_WINDOW};
    setHybridAttentionKvCacheSpecs(mc);

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    ASSERT_EQ(config.groupNums(), 3);
    EXPECT_EQ(config.groupTypesSnapshot(),
              std::vector<CacheGroupType>({CacheGroupType::FULL, CacheGroupType::SWA, CacheGroupType::LINEAR}));
    EXPECT_EQ(config.groupTagsSnapshot(), std::vector<std::string>({"full", "swa", "linear"}));
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 3u);
    EXPECT_NE(config.specForGroup(0).get(), config.specForGroup(1).get());
    EXPECT_EQ(config.layerIdsForGroup(0), std::vector<int>({0}));
    EXPECT_EQ(config.layerIdsForGroup(1), std::vector<int>({1, 3}));
    EXPECT_EQ(config.layerIdsForGroup(2), std::vector<int>({2}));
    EXPECT_EQ(config.layerIdsForGroup(0).size(), 1u);
    EXPECT_EQ(config.layerIdsForGroup(1).size(), 2u);
    EXPECT_EQ(config.layerIdsForGroup(2).size(), 1u);
    EXPECT_EQ(config.groupIdForLayerTag(1, "swa"), 1);
    EXPECT_EQ(config.groupIdForLayerTag(2, "linear"), 2);

    const auto full_gid   = gidForTag(config, "full");
    const auto swa_gid    = gidForTag(config, "swa");
    const auto linear_gid = gidForTag(config, "linear");
    EXPECT_EQ(config.block_size_bytes,
              config.blockSizeBytesForGroup(full_gid) + config.blockSizeBytesForGroup(linear_gid));

    RuntimeConfig runtime_config;
    config.linear_step = 3;
    config.finalizeBlockNums(/*global_block_num=*/10, runtime_config);
    EXPECT_EQ(config.blockNumForGroup(full_gid), 10u);
    EXPECT_EQ(config.blockNumForGroup(linear_gid), 10u);
    EXPECT_EQ(config.blockNumForGroup(swa_gid), 4u);
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionIndependentPoolBackingFitsBudgetExactly) {
    auto mc                                           = makeHybridAttentionModelConfig(true);
    mc.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE,
                                                         HybridAttentionType::SLIDING_WINDOW,
                                                         HybridAttentionType::LINEAR,
                                                         HybridAttentionType::SLIDING_WINDOW};
    setHybridAttentionKvCacheSpecs(mc);

    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.kv_cache_mem_mb = 1;
    kv_cache_config.linear_step     = 4;

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    size_t paged_bytes = 0;
    size_t swa_bytes   = 0;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        if (config.typeForGroup(gid) == CacheGroupType::SWA) {
            swa_bytes += config.blockSizeBytesForGroup(gid);
            EXPECT_EQ(config.blockNumForGroup(gid), (static_cast<uint32_t>(config.block_num) + 3u) / 4u);
        } else {
            paged_bytes += config.blockSizeBytesForGroup(gid);
            EXPECT_EQ(config.blockNumForGroup(gid), static_cast<uint32_t>(config.block_num));
        }
    }

    const auto backing_bytes = [&](uint32_t block_num) {
        return static_cast<size_t>(block_num) * paged_bytes + static_cast<size_t>((block_num + 3u) / 4u) * swa_bytes;
    };
    constexpr size_t budget_bytes = 1024u * 1024u;
    const auto       block_num    = static_cast<uint32_t>(config.block_num);
    EXPECT_LE(backing_bytes(block_num), budget_bytes);
    EXPECT_GT(backing_bytes(block_num + 1u), budget_bytes);
}

TEST(HybridPoolConfigCreatorTest, LinearValueHeadsMustDivideAttentionTp) {
    auto mc                                           = makeHybridAttentionModelConfig(/*independent_pool=*/true);
    mc.linear_attention_config.linear_num_value_heads = 3;

    ParallelismConfig pc;
    pc.tp_size = 2;

    EXPECT_THROW((void)CacheConfigCreator::createBasicConfig(mc, pc, false, 0), std::exception);

    mc.linear_attention_config.linear_num_value_heads = 4;
    EXPECT_NO_THROW((void)CacheConfigCreator::createBasicConfig(mc, pc, false, 0));
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionWithoutIndependentPoolKeepsSharedHybridConfig) {
    ParallelismConfig pc;
    auto config = CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(false), pc, false, 0);

    EXPECT_FALSE(config.use_independent_block_pools);
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_TRUE(config.groupBlockNumsSnapshot().empty());
}

TEST(HybridConfigCreatorTest, HybridAttentionTypesMustCoverAllLayers) {
    auto mc = makeHybridAttentionModelConfig(false);
    mc.hybrid_attention_config.hybrid_attention_types.pop_back();

    ParallelismConfig pc;
    EXPECT_THROW((void)CacheConfigCreator::createBasicConfig(mc, pc, false, 0), std::exception);
}

// ============================================================
// Generic opaque cache specs
// ============================================================

TEST(GenericOpaqueCacheSpecTest, KVSpecFromPoolSpec) {
    auto spec = buildCompressedSpec(
        "csa_kv", kDsv4Fp8KvEntryBytes, 64, DataType::TYPE_UINT8, 1, DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES);
    ASSERT_NE(spec, nullptr);

    EXPECT_EQ(spec->block_size(), 64u * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec->block_size_bytes(), 37440u);
    EXPECT_EQ(spec->block_size_bytes(), 37440u);
    EXPECT_EQ(spec->tag, "csa_kv");
    EXPECT_EQ(spec->block_size() / kDsv4Fp8KvEntryBytes, 64u);

    auto hca_spec = buildCompressedSpec(
        "hca_kv", kDsv4Fp8KvEntryBytes, 2, DataType::TYPE_UINT8, 1, DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES);
    ASSERT_NE(hca_spec, nullptr);
    EXPECT_EQ(hca_spec->block_size(), 2u * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(hca_spec->block_size_bytes(), 1728u);
}

TEST(GenericOpaqueCacheSpecTest, CompressedKVSpecReportsGenericKindsAndLayout) {
    auto spec = buildCompressedSpec(
        "compressed", kDsv4Fp8KvEntryBytes, 64, DataType::TYPE_UINT8, 4, DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES);
    ASSERT_NE(spec, nullptr);

    EXPECT_EQ(spec->type, KVCacheSpecType::OpaqueKV);
    EXPECT_EQ(spec->block_size(), 64u * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec->block_size_bytes(), 37440u);
    EXPECT_EQ(spec->block_size() / kDsv4Fp8KvEntryBytes, 64u);
    EXPECT_EQ(spec->k_block_payload_bytes() / 64u, static_cast<size_t>(kDsv4Fp8KvEntryBytes));
}

TEST(GenericOpaqueCacheSpecTest, OpaqueKVSpecUsesSingleRegionWithoutKVSplit) {
    auto spec = buildCompressedSpec("odd_kv", 3, 1, DataType::TYPE_UINT8);
    ASSERT_NE(spec, nullptr);
    EXPECT_EQ(spec->k_block_size(), 3u);
    EXPECT_EQ(spec->v_block_size(), 0u);
    EXPECT_EQ(spec->k_block_size_bytes(), 3u);
    EXPECT_EQ(spec->v_block_size_bytes(), 0u);
}

TEST(GenericOpaqueCacheSpecTest, OpaqueKVSpecAllowsStrideLargerThanPayload) {
    KVCacheSpecDesc desc;
    desc.tag                         = "odd_bytes";
    desc.cache_type                  = KVCacheSpecType::OpaqueKV;
    desc.dtype                       = DataType::TYPE_UINT8;
    desc.entry_elems                 = 2;
    desc.entry_dtype                 = DataType::TYPE_UINT8;
    desc.explicit_entry_count        = 1;
    desc.block_stride_bytes_override = 3;
    desc.is_state_cache              = false;
    SpecBuildContext ctx;
    ctx.dtype              = DataType::TYPE_UINT8;
    ctx.seq_size_per_block = kDsv4TokensPerBlock;

    auto spec = SpecBuilder::build(desc, ctx);
    ASSERT_NE(spec, nullptr);
    EXPECT_EQ(spec->block_payload_bytes(), 2u);
    EXPECT_EQ(spec->block_size_bytes(), 3u);
}

TEST(GenericOpaqueCacheSpecTest, FixedStateSpecCloneKeepsResolvedLayout) {
    auto original = buildFixedStateSpec("state", 32, 8, DataType::TYPE_FP32);
    ASSERT_NE(original, nullptr);

    auto cloned = std::dynamic_pointer_cast<FixedStateCacheSpec>(original->clone());
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->block_size(), 8u * 32u);
    EXPECT_EQ(cloned->block_size_bytes(), 8u * 32u * 4u);
    EXPECT_EQ(cloned->tag, "state");
}

TEST(GenericOpaqueCacheSpecTest, FixedStateSpecReportsGenericKindsAndSlicesByEntries) {
    auto spec = buildFixedStateSpec("tail_state", 32, 8, DataType::TYPE_FP32);
    ASSERT_NE(spec, nullptr);
    char      storage[8 * 32 * 4] = {};
    BlockInfo block;
    block.addr       = storage;
    block.size_bytes = sizeof(storage);

    auto sliced = sliceStateBlockForPeer(*spec, {block}, 4, 2);
    ASSERT_EQ(sliced.size(), 1u);
    EXPECT_EQ(spec->type, KVCacheSpecType::OpaqueState);
    EXPECT_EQ(sliced[0].addr, storage + 2 * 2 * 32 * 4);
    EXPECT_EQ(sliced[0].size_bytes, 2u * 32u * 4u);
}

TEST(GenericOpaqueCacheSpecTest, FixedStateSpecSlicesOverrideByBytes) {
    auto spec =
        buildFixedStateSpec("tail_bytes", kDsv4Fp8KvEntryBytes, kDsv4TokensPerBlock, DataType::TYPE_UINT8, 74880);
    ASSERT_NE(spec, nullptr);
    char      storage[74880] = {};
    BlockInfo block;
    block.addr       = storage;
    block.size_bytes = sizeof(storage);

    auto sliced = sliceStateBlockForPeer(*spec, {block}, 4, 3);
    ASSERT_EQ(sliced.size(), 1u);
    EXPECT_EQ(sliced[0].addr, storage + 3 * (sizeof(storage) / 4));
    EXPECT_EQ(sliced[0].size_bytes, sizeof(storage) / 4);

    auto cp_sliced = sliceStateBlockForPeer(*spec, {block}, 4, 3);
    ASSERT_EQ(cp_sliced.size(), 1u);
    EXPECT_EQ(cp_sliced[0].addr, sliced[0].addr);
    EXPECT_EQ(cp_sliced[0].size_bytes, sliced[0].size_bytes);
}

TEST(GenericOpaqueCacheSpecTest, FixedStateSpecSlicesAlignedBlockByPhysicalBytes) {
    auto spec = buildFixedStateSpec("aligned_tail",
                                    kDsv4Fp8KvEntryBytes,
                                    132,
                                    DataType::TYPE_UINT8,
                                    0,
                                    DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES,
                                    DSV4_SWA_WINDOW_ENTRIES);
    ASSERT_NE(spec, nullptr);
    ASSERT_EQ(spec->block_size(), 77088u);
    ASSERT_EQ(spec->block_size_bytes(), 77184u);
    char      storage[77184] = {};
    BlockInfo block;
    block.addr       = storage;
    block.size_bytes = sizeof(storage);

    auto sliced = sliceStateBlockForPeer(*spec, {block}, 2, 1);
    ASSERT_EQ(sliced.size(), 1u);
    EXPECT_EQ(sliced[0].addr, storage + 38592);
    EXPECT_EQ(sliced[0].size_bytes, 38592u);
}

TEST(GenericOpaqueCacheSpecTest, SWAFp8StateSpecUsesPaddedPhysicalBlockSize) {
    auto spec = buildFixedStateSpec("swa_kv",
                                    kDsv4Fp8KvEntryBytes,
                                    kDsv4TokensPerBlock,
                                    DataType::TYPE_UINT8,
                                    0,
                                    DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES,
                                    DSV4_SWA_WINDOW_ENTRIES);
    ASSERT_NE(spec, nullptr);

    EXPECT_EQ(spec->block_size(), kDsv4TokensPerBlock * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec->block_size_bytes(), 74880u);
    EXPECT_EQ(spec->tag, "swa_kv");
}

TEST(GenericOpaqueCacheSpecTest, StateSpecFloat32) {
    auto spec = buildFixedStateSpec("csa_state", 2048, 8, DataType::TYPE_FP32);
    ASSERT_NE(spec, nullptr);

    EXPECT_EQ(spec->block_size(), 8u * 2048u);
    EXPECT_EQ(spec->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(spec->tag, "csa_state");
}

TEST(GenericOpaqueCacheSpecTest, IndexerKVSpec) {
    auto spec = buildCompressedSpec("indexer_kv", 132, 64, DataType::TYPE_UINT8);
    ASSERT_NE(spec, nullptr);

    EXPECT_EQ(spec->block_size(), 64u * 132u);
    EXPECT_EQ(spec->block_size_bytes(), 64u * 132u);
    EXPECT_EQ(spec->tag, "indexer_kv");
}

TEST(GenericOpaqueCacheSpecTest, HCAStateSpec) {
    auto spec = buildFixedStateSpec("hca_state", 1024, 128, DataType::TYPE_FP32);
    ASSERT_NE(spec, nullptr);

    EXPECT_EQ(spec->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(spec->tag, "hca_state");
}

// ============================================================
// Pool 0/1/2 shared properties: same tokens_per_block, same num_blocks
// ============================================================

TEST(HybridPoolConfigCreatorTest, PagedPoolsShareTokensPerBlock) {
    // Pro config
    {
        ParallelismConfig pc;
        auto              config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, false, 0);
        EXPECT_EQ(config.seq_size_per_block, kDsv4TokensPerBlock);
    }
    // Flash config
    {
        ParallelismConfig pc;
        auto              config = CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, false, 0);
        EXPECT_EQ(config.seq_size_per_block, kDsv4TokensPerBlock);
    }
}

TEST(HybridPoolConfigCreatorTest, AllPagedPoolsShareBlockNum) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.block_num         = 100;

    // Paged groups derive their block count from the global block_num; explicit
    // independent groups may override it with per-group fixed block counts.
    EXPECT_EQ(config.groupNums(), 7);
    for (int i = 0; i < 7; i++) {
        EXPECT_GT(config.specForGroup(i)->block_size_bytes(), 0u) << "pool " << i;
    }
}

TEST(HybridPoolConfigCreatorTest, DSV4StateSwaPoolsFollowGlobalBlocks) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 100;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.groupBlockNumsSnapshot().size(), static_cast<size_t>(kDsv4PoolNum));
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.blockNumForGroup(gid), 100u) << "gid=" << gid;
    }
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST(HybridPoolConfigCreatorTest, DSV4HcaStatePoolBlocksOverridesOnlyHcaState) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 100;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 350);
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.groupBlockNumsSnapshot().size(), static_cast<size_t>(kDsv4PoolNum));
    const auto hca_state_gid = gidForTag(config, "hca_state");
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = gid == hca_state_gid ? 350u : 100u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }

    const size_t expected_reserve = 350u * config.blockSizeBytesForGroup(hca_state_gid);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, expected_reserve);
    ASSERT_EQ(config.groupPoliciesSnapshot().size(), static_cast<size_t>(kDsv4PoolNum));
    EXPECT_EQ(config.policyForGroup(hca_state_gid).explicit_block_num, 350u);
    for (size_t gid = 0; gid < config.groupPoliciesSnapshot().size(); ++gid) {
        if (gid != hca_state_gid) {
            EXPECT_EQ(config.policyForGroup(gid).explicit_block_num, 0u) << "gid=" << gid;
        }
    }
}

TEST(CacheConfigTest, DSV4HybridPoolRuntimeConfigAllowsDecoupledPhysicalAndKernelBlockSize) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    auto create_config = [&](int seq_size_per_block, int kernel_seq_size_per_block) {
        KVCacheConfig kv_cache_config;
        kv_cache_config.seq_size_per_block        = seq_size_per_block;
        kv_cache_config.kernel_seq_size_per_block = kernel_seq_size_per_block;
        kv_cache_config.test_block_num            = 100;
        return CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);
    };

    auto old_valid = create_config(128, 128);
    EXPECT_EQ(old_valid.seq_size_per_block, 128u);
    EXPECT_EQ(old_valid.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(old_valid.kernelBlocksPerKvBlock(), 1u);

    auto decoupled = create_config(16384, 128);
    EXPECT_EQ(decoupled.seq_size_per_block, 16384u);
    EXPECT_EQ(decoupled.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(decoupled.kernelBlocksPerKvBlock(), 128u);
}

TEST(CacheConfigTest, DSV4HybridPoolRuntimeConfigRejectsInvalidKernelShape) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    auto create_config = [&](int seq_size_per_block, int kernel_seq_size_per_block) {
        KVCacheConfig kv_cache_config;
        kv_cache_config.seq_size_per_block        = seq_size_per_block;
        kv_cache_config.kernel_seq_size_per_block = kernel_seq_size_per_block;
        kv_cache_config.test_block_num            = 100;
        return CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);
    };

    EXPECT_THROW((void)create_config(16384, 64), std::exception);
    EXPECT_THROW((void)create_config(16384, 384), std::exception);
}

TEST(HybridPoolConfigCreatorTest, DSV4HcaStatePoolBlocksIndependentOfMaxConcurrency) {
    for (uint32_t max_concurrency : {1u, 2u, 8u}) {
        auto              mc = makeProModelConfig();
        ParallelismConfig pc;
        RuntimeConfig     runtime_config;
        KVCacheConfig     kv_cache_config;
        kv_cache_config.seq_size_per_block = 128;
        kv_cache_config.test_block_num     = 100;
        setDsv4ExplicitPoolBlocks(mc, "hca_state", 256);
        runtime_config.max_generate_batch_size                      = max_concurrency;
        runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

        auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

        ASSERT_EQ(config.groupBlockNumsSnapshot().size(), static_cast<size_t>(kDsv4PoolNum));
        const auto hca_state_gid = gidForTag(config, "hca_state");
        for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
            const uint32_t expected = static_cast<size_t>(gid) == hca_state_gid ? 256u : 100u;
            EXPECT_EQ(config.blockNumForGroup(gid), expected)
                << "gid=" << gid << " max_concurrency=" << max_concurrency;
        }
    }
}

TEST(HybridPoolConfigCreatorTest, DSV4HcaStatePoolBlocksCanBeOverriddenByConfig) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 100;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 6);
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.groupBlockNumsSnapshot().size(), static_cast<size_t>(kDsv4PoolNum));
    const auto hca_state_gid = gidForTag(config, "hca_state");
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        const uint32_t expected = static_cast<size_t>(gid) == hca_state_gid ? 6u : 100u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
}

TEST(CacheConfigTest, ModelSpecCloneKeepsExistingConfigStable) {
    ModelConfig model_config;
    model_config.num_layers                   = 2;
    model_config.attn_config.kv_head_num      = 4;
    model_config.attn_config.size_per_head    = 16;
    model_config.attn_config.tokens_per_block = 8;
    setDefaultKvCacheSpec(model_config);

    ParallelismConfig pc_tp1;
    pc_tp1.tp_size  = 1;
    auto config_tp1 = CacheConfigCreator::createBasicConfig(model_config, pc_tp1, false, 0);
    ASSERT_EQ(static_cast<size_t>(config_tp1.groupNums()), 1u);
    EXPECT_EQ(config_tp1.localKvHeadNumForGroup(0), 4);

    ParallelismConfig pc_tp2;
    pc_tp2.tp_size  = 2;
    auto config_tp2 = CacheConfigCreator::createBasicConfig(model_config, pc_tp2, false, 0);
    ASSERT_EQ(static_cast<size_t>(config_tp2.groupNums()), 1u);
    EXPECT_EQ(config_tp2.localKvHeadNumForGroup(0), 2);

    EXPECT_EQ(config_tp1.localKvHeadNumForGroup(0), 4);
    EXPECT_NE(config_tp1.specForGroup(0).get(), config_tp2.specForGroup(0).get());
}

TEST(CacheConfigTest, RuntimeKernelBlockOverrideUpdatesTopology) {
    ModelConfig model_config;
    model_config.num_layers                   = 1;
    model_config.attn_config.use_mla          = true;
    model_config.attn_config.kv_lora_rank     = 512;
    model_config.attn_config.rope_head_dim    = 64;
    model_config.attn_config.tokens_per_block = 512;
    setDefaultKvCacheSpec(model_config);

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.kernel_seq_size_per_block = 64;
    kv_cache_config.test_block_num            = 2;

    auto config = CacheConfigCreator::createConfig(model_config, parallelism_config, runtime_config, kv_cache_config);

    ASSERT_EQ(config.groupNums(), 1);
    EXPECT_EQ(config.seq_size_per_block, 512u);
    EXPECT_EQ(config.kernel_seq_size_per_block, 64u);
    EXPECT_EQ(config.seqSizePerBlockForGroup(0), 512u);
    EXPECT_EQ(config.kernelSeqSizePerBlockForGroup(0), 64u);
    EXPECT_EQ(config.kernelBlocksPerKvBlockForGroup(0), 8u);
}

TEST(CacheConfigTest, SpecBuilderDerivesAttentionSpecsFromContext) {
    AttentionConfigs attn{};
    attn.kv_head_num   = 12;
    attn.size_per_head = 64;
    attn.kv_lora_rank  = 512;
    attn.rope_head_dim = 64;

    LinearAttentionConfig linear{};
    linear.linear_num_key_heads   = 16;
    linear.linear_num_value_heads = 16;
    linear.linear_key_head_dim    = 128;
    linear.linear_value_head_dim  = 128;
    linear.linear_conv_kernel_dim = 4;
    linear.ssm_state_dtype        = DataType::TYPE_BF16;
    linear.conv_state_dtype       = DataType::TYPE_FP16;

    ParallelismConfig parallelism;
    parallelism.tp_size = 8;

    SpecBuildContext ctx;
    ctx.dtype                   = DataType::TYPE_INT8;
    ctx.seq_size_per_block      = 16;
    ctx.attn_config             = &attn;
    ctx.linear_attention_config = &linear;
    ctx.parallelism_config      = &parallelism;

    KVCacheSpecDesc mha_desc;
    mha_desc.tag        = "mha";
    mha_desc.cache_type = KVCacheSpecType::MultiHeadAttention;

    auto mha = std::dynamic_pointer_cast<MHAKVCacheSpec>(SpecBuilder::build(mha_desc, ctx));
    ASSERT_NE(mha, nullptr);
    EXPECT_EQ(mha->seq_size_per_block, 16u);
    EXPECT_EQ(mha->block_size(), 2u * 3u * 64u * 16u);
    EXPECT_EQ(mha->scale_block_size_bytes(), 2u * 3u * 16u * sizeof(float));

    ctx.dtype = DataType::TYPE_BF16;
    KVCacheSpecDesc mla_desc;
    mla_desc.tag        = "mla";
    mla_desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;

    auto mla = std::dynamic_pointer_cast<MLAKVCacheSpec>(SpecBuilder::build(mla_desc, ctx));
    ASSERT_NE(mla, nullptr);
    EXPECT_EQ(mla->k_block_size(), 512u * 16u);
    EXPECT_EQ(mla->v_block_size(), 64u * 16u);
    EXPECT_EQ(mla->seq_size_per_block, 16u);
    EXPECT_EQ(mla->block_size(), (512u + 64u) * 16u);

    KVCacheSpecDesc linear_desc;
    linear_desc.tag        = "linear";
    linear_desc.cache_type = KVCacheSpecType::LinearAttention;

    auto linear_spec = std::dynamic_pointer_cast<LinearKVCacheSpec>(SpecBuilder::build(linear_desc, ctx));
    ASSERT_NE(linear_spec, nullptr);
    EXPECT_EQ(linear_spec->k_block_size(), 2u * 128u * 128u);
    EXPECT_EQ(linear_spec->v_block_size(), 3u * (128u * 2u * 2u + 128u * 2u));
    EXPECT_EQ(linear_spec->k_block_size_bytes(), linear_spec->k_block_size() * getTypeSize(DataType::TYPE_BF16));
    EXPECT_EQ(linear_spec->v_block_size_bytes(), linear_spec->v_block_size() * getTypeSize(DataType::TYPE_FP16));
    EXPECT_EQ(linear_spec->seq_size_per_block, 16u);

    SpecBuildContext missing_linear_ctx        = ctx;
    missing_linear_ctx.linear_attention_config = nullptr;
    EXPECT_THROW((void)SpecBuilder::build(linear_desc, missing_linear_ctx), std::exception);

    SpecBuildContext missing_attn_ctx;
    missing_attn_ctx.seq_size_per_block = 16;
    EXPECT_THROW((void)SpecBuilder::build(mha_desc, missing_attn_ctx), std::exception);

    SpecBuildContext missing_parallelism_ctx   = ctx;
    missing_parallelism_ctx.parallelism_config = nullptr;
    EXPECT_THROW((void)SpecBuilder::build(mha_desc, missing_parallelism_ctx), std::exception);

    AttentionConfigs invalid_attn{};
    SpecBuildContext invalid_ctx;
    invalid_ctx.attn_config        = &invalid_attn;
    invalid_ctx.parallelism_config = &parallelism;
    invalid_ctx.seq_size_per_block = 16;
    EXPECT_THROW((void)SpecBuilder::build(mha_desc, invalid_ctx), std::exception);
}

TEST(CacheConfigTest, LinearPolicyDefaultsPrefixReuseAndExplicitDisableOverrides) {
    KVCacheSpecDesc linear_desc;
    linear_desc.tag        = "linear";
    linear_desc.cache_type = KVCacheSpecType::LinearAttention;

    auto default_policy = SpecBuilder::groupPolicy(linear_desc);
    EXPECT_EQ(default_policy.group_type, CacheGroupType::LINEAR);
    EXPECT_TRUE(default_policy.enable_prefix_reuse);
    EXPECT_EQ(default_policy.active_tail_blocks, 1u);
    EXPECT_EQ(default_policy.cp_mapping, CpBlockMappingMode::NONE);

    linear_desc.reuse                      = CacheReusePolicyDesc{};
    linear_desc.reuse->enable_prefix_reuse = false;
    auto disabled_policy                   = SpecBuilder::groupPolicy(linear_desc);
    EXPECT_FALSE(disabled_policy.enable_prefix_reuse);
    EXPECT_EQ(disabled_policy.active_tail_blocks, 1u);
    EXPECT_EQ(disabled_policy.cp_mapping, CpBlockMappingMode::NONE);
}

TEST(CacheConfigTest, SpecBuilderDerivesHybridPoolRuntimeFieldsFromContext) {
    ParallelismConfig prefill_parallelism;
    prefill_parallelism.role_type                          = RoleType::PREFILL;
    prefill_parallelism.tp_size                            = 2;
    prefill_parallelism.prefill_cp_config.kv_cache_sharded = true;

    SpecBuildContext ctx;
    ctx.dtype                   = DataType::TYPE_BF16;
    ctx.seq_size_per_block      = 128;
    ctx.parallelism_config      = &prefill_parallelism;
    ctx.kernel_tokens_per_block = 128;
    ctx.gen_num_per_cycle       = 3;

    KVCacheSpecDesc compressed_desc;
    compressed_desc.tag                = "compressed";
    compressed_desc.cache_type         = KVCacheSpecType::OpaqueKV;
    compressed_desc.entry_elems        = 16;
    compressed_desc.compression_ratio  = 4;
    compressed_desc.entry_dtype        = DataType::TYPE_UINT8;
    compressed_desc.entry_count_mode   = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    compressed_desc.cp                 = CacheCpPolicyDesc{};
    compressed_desc.cp->scale_seq_size = true;

    auto compressed = std::dynamic_pointer_cast<CompressedKVCacheSpec>(SpecBuilder::build(compressed_desc, ctx));
    ASSERT_NE(compressed, nullptr);
    EXPECT_EQ(compressed->block_size() / compressed_desc.entry_elems, 32u);
    EXPECT_EQ(compressed->seq_size_per_block, 256u);
    EXPECT_EQ(compressed->memoryLayoutDType(), DataType::TYPE_UINT8);

    KVCacheSpecDesc state_desc;
    state_desc.tag                                  = "state";
    state_desc.cache_type                           = KVCacheSpecType::OpaqueState;
    state_desc.entry_elems                          = 32;
    state_desc.entry_dtype                          = DataType::TYPE_FP32;
    state_desc.block_stride_bytes_alignment         = 64;
    state_desc.entry_count_mode                     = OpaqueBlockEntryCountMode::STATE_RING;
    state_desc.compression_ratio                    = 4;
    state_desc.state_ring_overlap                   = 1;
    state_desc.state_ring_include_gen_num_per_cycle = true;
    state_desc.cp                                   = CacheCpPolicyDesc{};
    state_desc.cp->align_payload                    = true;
    state_desc.cp->prefill_slice_layout             = CpPrefillSliceLayout::PAYLOAD;
    state_desc.cp->scale_seq_size                   = true;

    auto prefill_state = std::dynamic_pointer_cast<FixedStateCacheSpec>(SpecBuilder::build(state_desc, ctx));
    ASSERT_NE(prefill_state, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*prefill_state, 32u * getTypeSize(DataType::TYPE_FP32)), 6u);
    EXPECT_EQ(prefill_state->block_size_bytes(), 768u);
    EXPECT_EQ(prefill_state->seq_size_per_block, 256u);

    ParallelismConfig decode_parallelism;
    decode_parallelism.role_type                          = RoleType::DECODE;
    decode_parallelism.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_parallelism.prefill_cp_config.kv_cache_sharded = true;
    decode_parallelism.prefill_cp_config.prefill_cp_size  = 2;
    ctx.parallelism_config                                = &decode_parallelism;
    auto decode_state = std::dynamic_pointer_cast<FixedStateCacheSpec>(SpecBuilder::build(state_desc, ctx));
    ASSERT_NE(decode_state, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*decode_state, 32u * getTypeSize(DataType::TYPE_FP32)), 12u);
    EXPECT_EQ(decode_state->seq_size_per_block, 256u);
}

TEST(CacheConfigTest, ExactBlockBudgetHandlesStepAndRoundingBoundaries) {
    const KVCacheBlockBudget budget{/*explicit_pool_reserve_bytes=*/10,
                                    /*paged_block_bytes=*/3,
                                    /*swa_block_bytes=*/5};

    // step=4: cost(N) = 10 + 3*N + 5*ceil(N/4).
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/17, budget, /*linear_step=*/4), 0u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/18, budget, /*linear_step=*/4), 1u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/27, budget, /*linear_step=*/4), 4u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/34, budget, /*linear_step=*/4), 4u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/35, budget, /*linear_step=*/4), 5u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/44, budget, /*linear_step=*/4), 8u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/51, budget, /*linear_step=*/4), 8u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/52, budget, /*linear_step=*/4), 9u);

    // step<=1: both paged and SWA bytes are charged for every block.
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/17, budget, /*linear_step=*/1), 0u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/18, budget, /*linear_step=*/1), 1u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/34, budget, /*linear_step=*/1), 3u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/34, budget, /*linear_step=*/0), 3u);
}

TEST(CacheConfigTest, FinalizeBlockNumsUpdatesGlobalBlockNumForSharedPools) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 8;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 4;

    ParallelismConfig pc;
    ModelConfig       single_model_config;
    single_model_config.num_layers                   = 1;
    single_model_config.attn_config.kv_head_num      = 1;
    single_model_config.attn_config.size_per_head    = 1;
    single_model_config.attn_config.tokens_per_block = 1;
    setDefaultKvCacheSpec(single_model_config);
    auto single_config = CacheConfigCreator::createBasicConfig(single_model_config, pc, false, 0);
    single_config.finalizeBlockNums(123, runtime_config);
    EXPECT_EQ(single_config.block_num, 123u);
    EXPECT_TRUE(single_config.groupBlockNumsSnapshot().empty());
    EXPECT_EQ(single_config.explicitly_sized_pool_reserve_bytes, 0u);

    auto hybrid_config = CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(false), pc, false, 0);
    hybrid_config.finalizeBlockNums(123, runtime_config);
    EXPECT_EQ(hybrid_config.block_num, 123u);
    EXPECT_FALSE(hybrid_config.use_independent_block_pools);
    EXPECT_TRUE(hybrid_config.groupBlockNumsSnapshot().empty());
    EXPECT_EQ(hybrid_config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, FinalizeBlockNumsAppliesToIndependentPools) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, false, 0);
    config.finalizeBlockNums(100, runtime_config);

    ASSERT_EQ(config.groupBlockNumsSnapshot().size(), static_cast<size_t>(kDsv4PoolNum));
    const auto hca_state_gid = gidForTag(config, "hca_state");
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        const uint32_t expected = static_cast<size_t>(gid) == hca_state_gid ? 256u : 100u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 256u * config.blockSizeBytesForGroup(hca_state_gid));
}

TEST(CacheConfigTest, HcaStateReserveDeductedFromPagedBudget) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    const uint32_t small_hca_state_pool = 32;
    const uint32_t large_hca_state_pool = 256;

    KVCacheConfig kv_cache_config_with;
    kv_cache_config_with.seq_size_per_block = 128;
    kv_cache_config_with.kv_cache_mem_mb    = 65536;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", small_hca_state_pool);
    auto config_with = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config_with);

    KVCacheConfig kv_cache_config_without;
    kv_cache_config_without.seq_size_per_block = 128;
    kv_cache_config_without.kv_cache_mem_mb    = 65536;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", large_hca_state_pool);
    auto config_without = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config_without);

    // More HCA_STATE blocks reserve more HBM and leave fewer blocks for the global pools.
    EXPECT_GT(config_with.block_num, config_without.block_num);
    EXPECT_EQ(config_with.blockNumForGroup(gidForTag(config_with, "hca_kv")),
              static_cast<uint32_t>(config_with.block_num));
    EXPECT_EQ(config_without.blockNumForGroup(gidForTag(config_without, "hca_kv")),
              static_cast<uint32_t>(config_without.block_num));
    EXPECT_EQ(config_with.blockNumForGroup(gidForTag(config_with, "hca_state")), small_hca_state_pool);
    EXPECT_EQ(config_without.blockNumForGroup(gidForTag(config_without, "hca_state")), large_hca_state_pool);
    const size_t expected_reserve = static_cast<size_t>(small_hca_state_pool)
                                    * config_with.blockSizeBytesForGroup(gidForTag(config_with, "hca_state"));
    EXPECT_EQ(config_with.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

TEST(CacheConfigTest, DSV4ExplicitHcaStatePoolBlocksIgnoreLinearStep) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, false, 0);
    config.linear_step       = 4;
    config.finalizeBlockNums(100, runtime_config);

    // The explicit pool keeps its requested capacity. Non-explicit FULL/LINEAR
    // groups keep N, while non-explicit SWA groups use ceil(N / step).
    const auto hca_state_gid = gidForTag(config, "hca_state");
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected =
            gid == hca_state_gid ? 256u : (config.typeForGroup(gid) == CacheGroupType::SWA ? 25u : 100u);
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
    const size_t expected_reserve = 256u * config.blockSizeBytesForGroup(hca_state_gid);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

TEST(CacheConfigTest, DSV4StateSwaPoolsWithoutExplicitBlocksScaleWithLinearStep) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 100;
    kv_cache_config.linear_step    = 4;
    auto mc                        = makeProModelConfig();
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.groupBlockNumsSnapshot().size(), static_cast<size_t>(kDsv4PoolNum));
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        const uint32_t expected = config.typeForGroup(gid) == CacheGroupType::SWA ? 25u : 100u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, DSV4MtpKeepsProposeLayerInSwaPool) {
    auto score_model_config                         = makeFlashModelConfig();
    auto propose_model_config                       = makeFlashMtpModelConfig();
    score_model_config.attn_config.kv_cache_dtype   = KvCacheDataType::FP8;
    propose_model_config.attn_config.kv_cache_dtype = KvCacheDataType::FP8;

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 16384;
    kv_cache_config.kernel_seq_size_per_block = 128;
    kv_cache_config.test_block_num            = 100;
    kv_cache_config.linear_step               = 4;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;

    auto config = CacheConfigCreator::createSpConfig(score_model_config,
                                                     propose_model_config,
                                                     parallelism_config,
                                                     runtime_config,
                                                     kv_cache_config,
                                                     sp_config,
                                                     std::nullopt,
                                                     true,
                                                     false);

    ASSERT_EQ(config.layer_num, 43u);
    ASSERT_EQ(config.layer_all_num, 45u);
    ASSERT_EQ(config.topology().layers().size(), static_cast<size_t>(config.layer_all_num));
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    ASSERT_NE(config.mtp_sub_configs[0], nullptr);
    ASSERT_NE(config.mtp_sub_configs[1], nullptr);

    const auto swa_gid = gidForTag(config, "swa_kv");
    EXPECT_EQ(config.layerGroupIdsSnapshot()[43], std::vector<int>({static_cast<int>(swa_gid)}));
    EXPECT_EQ(config.layerGroupIdsSnapshot()[44], std::vector<int>({static_cast<int>(swa_gid)}));
    EXPECT_EQ(config.groupIdForLayerTag(43, "swa_kv"), static_cast<int>(swa_gid));
    EXPECT_EQ(config.topology().layer(43).group_tags, std::vector<std::string>({"swa_kv"}));
    EXPECT_EQ(config.topology().layer(44).group_tags, std::vector<std::string>({"swa_kv"}));
    EXPECT_EQ(config.groupIdForLayerTag(44, "swa_kv"), static_cast<int>(swa_gid));

    EXPECT_EQ(config.layerIdsForGroup(swa_gid).size(), 45u);

    // MTP sub-configs preserve the target/global group namespace while keeping
    // layer ids draft-local. Current MTP execution passes block tables by gid
    // without a draft-local remap, so unused target groups stay as empty
    // placeholders and the real SWA layer keeps the same gid as the target config.
    EXPECT_EQ(config.mtp_sub_configs[0]->groupTagsSnapshot(), config.groupTagsSnapshot());
    EXPECT_EQ(config.mtp_sub_configs[1]->groupTagsSnapshot(), config.groupTagsSnapshot());
    EXPECT_EQ(config.mtp_sub_configs[0]->groupIdForLayerTag(0, "swa_kv"), static_cast<int>(swa_gid));
    EXPECT_EQ(config.mtp_sub_configs[1]->groupIdForLayerTag(0, "swa_kv"), static_cast<int>(swa_gid));
    EXPECT_EQ(config.mtp_sub_configs[0]->layerIdsForGroup(swa_gid), std::vector<int>({0}));
    EXPECT_EQ(config.mtp_sub_configs[1]->layerIdsForGroup(swa_gid), std::vector<int>({0}));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        if (gid == swa_gid) {
            continue;
        }
        EXPECT_TRUE(config.mtp_sub_configs[0]->layerIdsForGroup(gid).empty()) << config.tagForGroup(gid);
        EXPECT_TRUE(config.mtp_sub_configs[1]->layerIdsForGroup(gid).empty()) << config.tagForGroup(gid);
    }
    EXPECT_EQ(config.seq_size_per_block, 16384u);
    EXPECT_EQ(config.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(config.kernelBlocksPerKvBlock(), 128u);
    EXPECT_EQ(config.mtp_sub_configs[0]->seq_size_per_block, 16384u);
    EXPECT_EQ(config.mtp_sub_configs[0]->kernel_seq_size_per_block, 128u);

    EXPECT_EQ(config.blockNumForGroup(swa_gid), 25u);
    EXPECT_EQ(config.mtp_sub_configs[0]->linear_step, 4);
    EXPECT_EQ(config.mtp_sub_configs[1]->linear_step, 4);
    EXPECT_EQ(config.mtp_sub_configs[0]->blockNumForGroup(swa_gid), 25u);
    EXPECT_EQ(config.mtp_sub_configs[1]->blockNumForGroup(swa_gid), 25u);

    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes,
              256u * config.blockSizeBytesForGroup(gidForTag(config, "hca_state")));
}

TEST(CacheConfigTest, DSV4MtpJointBudgetIncludesScoreAndProposeSwaBacking) {
    auto score_model_config   = makeFlashModelConfig();
    auto propose_model_config = makeFlashMtpModelConfig();

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 128;
    kv_cache_config.kernel_seq_size_per_block = 128;
    kv_cache_config.kv_cache_mem_mb           = 65536;
    kv_cache_config.linear_step               = 4;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;

    auto config = CacheConfigCreator::createSpConfig(score_model_config,
                                                     propose_model_config,
                                                     parallelism_config,
                                                     runtime_config,
                                                     kv_cache_config,
                                                     sp_config,
                                                     std::nullopt,
                                                     true,
                                                     false);

    size_t paged_bytes = 0;
    size_t swa_bytes   = 0;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto explicit_blocks = config.policyForGroup(gid).explicit_block_num;
        if (explicit_blocks > 0) {
            EXPECT_EQ(config.blockNumForGroup(gid), explicit_blocks) << "gid=" << gid;
            continue;
        }
        if (config.typeForGroup(gid) == CacheGroupType::SWA) {
            swa_bytes += config.blockSizeBytesForGroup(gid);
            EXPECT_EQ(config.blockNumForGroup(gid), (static_cast<uint32_t>(config.block_num) + 3u) / 4u)
                << "gid=" << gid;
        } else {
            paged_bytes += config.blockSizeBytesForGroup(gid);
            EXPECT_EQ(config.blockNumForGroup(gid), static_cast<uint32_t>(config.block_num)) << "gid=" << gid;
        }
    }

    const auto backing_bytes = [&](uint32_t block_num) {
        return config.explicitly_sized_pool_reserve_bytes + static_cast<size_t>(block_num) * paged_bytes
               + static_cast<size_t>((block_num + 3u) / 4u) * swa_bytes;
    };
    const size_t budget_bytes = static_cast<size_t>(kv_cache_config.kv_cache_mem_mb) * 1024u * 1024u;
    const auto   block_num    = static_cast<uint32_t>(config.block_num);
    EXPECT_LE(backing_bytes(block_num), budget_bytes);
    EXPECT_GT(backing_bytes(block_num + 1u), budget_bytes);

    const auto swa_gid = gidForTag(config, "swa_kv");
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_NE(sub_config, nullptr);
        EXPECT_EQ(sub_config->linear_step, 4);
        EXPECT_EQ(sub_config->blockNumForGroup(swa_gid), (block_num + 3u) / 4u);
    }
}

TEST(HybridPoolConfigCreatorTest, MtpGenNum2RingEntriesMatch) {
    // gen_num_per_cycle=2 -> CSA/INDEXER R=10, HCA R=130, SWA R=130.
    // Formula: R = ceil_even((1 + overlap) * ratio + gen_num_per_cycle).
    // SWA_KV is sized like the HCA state ring (window 128, overlap 0).
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, /*gen_num_per_cycle=*/2);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    // Pool 3: INDEXER_STATE (ratio=4, overlap=1) → R=10
    auto* indexer_state =
        dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "indexer_state")).get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*indexer_state, kDsv4IndexerStateEntryBytes), 10u);
    // Pool 4: CSA_STATE (ratio=4, overlap=1) → R=10
    auto* csa_state =
        dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "csa_state")).get());
    ASSERT_NE(csa_state, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*csa_state, kDsv4CsaStateEntryBytes), 10u);
    // Pool 5: HCA_STATE (ratio=128, overlap=0) → R=130
    auto* hca_state =
        dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "hca_state")).get());
    ASSERT_NE(hca_state, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*hca_state, kDsv4HcaStateEntryBytes), 130u);
    // Pool 6: SWA_KV (window=128, overlap=0) → R=130, same as HCA_STATE
    auto* swa_kv = dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "swa_kv")).get());
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(swa_kv->tag, "swa_kv");
    EXPECT_EQ(opaqueEntriesPerBlock(*swa_kv, kDsv4KvEntryBytes), 130u);
}

TEST(HybridPoolConfigCreatorTest, PrefillCp8MtpGenNum2PadsStateRingBeforeSlicing) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = 8;
    pc.prefill_cp_config.kv_cache_sharded = true;

    auto config = CacheConfigCreator::createBasicConfig(mc, pc, false, 2);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    auto* indexer_state =
        dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "indexer_state")).get());
    auto* csa_state =
        dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "csa_state")).get());
    auto* hca_state =
        dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "hca_state")).get());
    auto* swa_kv = dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "swa_kv")).get());
    ASSERT_NE(indexer_state, nullptr);
    ASSERT_NE(csa_state, nullptr);
    ASSERT_NE(hca_state, nullptr);
    ASSERT_NE(swa_kv, nullptr);

    // gen_num_per_cycle=2 gives raw INDEXER/CSA R=10, HCA/SWA R=130.
    // Fixed state pools are CP-sliced by entries; SWA_KV keeps full logical
    // entries and slices its packed bytes instead.
    EXPECT_EQ(opaqueEntriesPerBlock(*indexer_state, kDsv4IndexerStateEntryBytes), 2u);
    EXPECT_EQ(opaqueEntriesPerBlock(*csa_state, kDsv4CsaStateEntryBytes), 2u);
    EXPECT_EQ(opaqueEntriesPerBlock(*hca_state, kDsv4HcaStateEntryBytes), 17u);
    EXPECT_EQ(opaqueEntriesPerBlock(*swa_kv, kDsv4KvEntryBytes), 136u);
}

TEST(HybridPoolConfigCreatorTest, DecodePrefillCp8MtpGenNum2ExpandsFixedAndSwaSlices) {
    constexpr uint32_t cp_size = 8;
    auto               mc      = makeFlashModelConfig();

    ParallelismConfig prefill_pc;
    prefill_pc.role_type                          = RoleType::PREFILL;
    prefill_pc.tp_size                            = cp_size;
    prefill_pc.prefill_cp_config.kv_cache_sharded = true;

    ParallelismConfig decode_pc;
    decode_pc.role_type                          = RoleType::DECODE;
    decode_pc.tp_size                            = 1;
    decode_pc.dp_size                            = cp_size;
    decode_pc.world_size                         = cp_size;
    decode_pc.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_pc.prefill_cp_config.kv_cache_sharded = true;
    decode_pc.prefill_cp_config.prefill_cp_size  = cp_size;

    auto prefill_config = CacheConfigCreator::createBasicConfig(mc, prefill_pc, false, 2);
    auto decode_config  = CacheConfigCreator::createBasicConfig(mc, decode_pc, false, 2);

    ASSERT_EQ(static_cast<size_t>(prefill_config.groupNums()), 7u);
    ASSERT_EQ(static_cast<size_t>(decode_config.groupNums()), 7u);

    for (const auto& tag : {"indexer_state", "csa_state", "hca_state"}) {
        const auto prefill_gid = gidForTag(prefill_config, tag);
        const auto decode_gid  = gidForTag(decode_config, tag);
        auto* prefill_spec = dynamic_cast<const FixedStateCacheSpec*>(prefill_config.specForGroup(prefill_gid).get());
        auto* decode_spec  = dynamic_cast<const FixedStateCacheSpec*>(decode_config.specForGroup(decode_gid).get());
        ASSERT_NE(prefill_spec, nullptr) << tag;
        ASSERT_NE(decode_spec, nullptr) << tag;
        EXPECT_EQ(decode_spec->tag, prefill_spec->tag) << tag;
        const auto expected_entries = opaqueEntriesPerBlock(*prefill_spec, stateEntryBytesForTag(tag)) * cp_size;
        EXPECT_EQ(opaqueEntriesPerBlock(*decode_spec, stateEntryBytesForTag(tag)), expected_entries) << tag;
    }
    auto* prefill_swa = dynamic_cast<const FixedStateCacheSpec*>(
        prefill_config.specForGroup(gidForTag(prefill_config, "swa_kv")).get());
    auto* decode_swa =
        dynamic_cast<const FixedStateCacheSpec*>(decode_config.specForGroup(gidForTag(decode_config, "swa_kv")).get());
    ASSERT_NE(prefill_swa, nullptr);
    ASSERT_NE(decode_swa, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*prefill_swa, kDsv4KvEntryBytes), 136u);
    EXPECT_EQ(opaqueEntriesPerBlock(*decode_swa, kDsv4KvEntryBytes),
              opaqueEntriesPerBlock(*prefill_swa, kDsv4KvEntryBytes));

    auto* indexer_state = dynamic_cast<const FixedStateCacheSpec*>(
        decode_config.specForGroup(gidForTag(decode_config, "indexer_state")).get());
    auto* csa_state = dynamic_cast<const FixedStateCacheSpec*>(
        decode_config.specForGroup(gidForTag(decode_config, "csa_state")).get());
    auto* hca_state = dynamic_cast<const FixedStateCacheSpec*>(
        decode_config.specForGroup(gidForTag(decode_config, "hca_state")).get());
    auto* swa_kv =
        dynamic_cast<const FixedStateCacheSpec*>(decode_config.specForGroup(gidForTag(decode_config, "swa_kv")).get());
    ASSERT_NE(indexer_state, nullptr);
    ASSERT_NE(csa_state, nullptr);
    ASSERT_NE(hca_state, nullptr);
    ASSERT_NE(swa_kv, nullptr);

    EXPECT_EQ(opaqueEntriesPerBlock(*indexer_state, kDsv4IndexerStateEntryBytes), 16u);
    EXPECT_EQ(opaqueEntriesPerBlock(*csa_state, kDsv4CsaStateEntryBytes), 16u);
    EXPECT_EQ(opaqueEntriesPerBlock(*hca_state, kDsv4HcaStateEntryBytes), 136u);
    EXPECT_EQ(opaqueEntriesPerBlock(*swa_kv, kDsv4KvEntryBytes), 136u);
}

TEST(HybridPoolConfigCreatorTest, DecodeExplicitPrefillCpSizeHandlesDp16) {
    constexpr uint32_t cp_size = 8;
    auto               mc      = makeFlashModelConfig();

    ParallelismConfig prefill_pc;
    prefill_pc.role_type                          = RoleType::PREFILL;
    prefill_pc.tp_size                            = cp_size;
    prefill_pc.prefill_cp_config.kv_cache_sharded = true;

    ParallelismConfig decode_pc;
    decode_pc.role_type                          = RoleType::DECODE;
    decode_pc.tp_size                            = 1;
    decode_pc.dp_size                            = 16;
    decode_pc.world_size                         = 16;
    decode_pc.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_pc.prefill_cp_config.kv_cache_sharded = true;
    decode_pc.prefill_cp_config.prefill_cp_size  = cp_size;

    auto prefill_config = CacheConfigCreator::createBasicConfig(mc, prefill_pc, false, 2);
    auto decode_config  = CacheConfigCreator::createBasicConfig(mc, decode_pc, false, 2);

    for (const auto& tag : {"indexer_state", "csa_state", "hca_state"}) {
        const auto prefill_gid = gidForTag(prefill_config, tag);
        const auto decode_gid  = gidForTag(decode_config, tag);
        auto* prefill_spec = dynamic_cast<const FixedStateCacheSpec*>(prefill_config.specForGroup(prefill_gid).get());
        auto* decode_spec  = dynamic_cast<const FixedStateCacheSpec*>(decode_config.specForGroup(decode_gid).get());
        ASSERT_NE(prefill_spec, nullptr) << tag;
        ASSERT_NE(decode_spec, nullptr) << tag;
        const auto expected_entries = opaqueEntriesPerBlock(*prefill_spec, stateEntryBytesForTag(tag)) * cp_size;
        EXPECT_EQ(opaqueEntriesPerBlock(*decode_spec, stateEntryBytesForTag(tag)), expected_entries) << tag;
    }
    auto* prefill_swa = dynamic_cast<const FixedStateCacheSpec*>(
        prefill_config.specForGroup(gidForTag(prefill_config, "swa_kv")).get());
    auto* decode_swa =
        dynamic_cast<const FixedStateCacheSpec*>(decode_config.specForGroup(gidForTag(decode_config, "swa_kv")).get());
    ASSERT_NE(prefill_swa, nullptr);
    ASSERT_NE(decode_swa, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*prefill_swa, kDsv4KvEntryBytes), 136u);
    EXPECT_EQ(opaqueEntriesPerBlock(*decode_swa, kDsv4KvEntryBytes),
              opaqueEntriesPerBlock(*prefill_swa, kDsv4KvEntryBytes));
}

TEST(CacheConfigTest, DSV4NonMtpSpConfigDoesNotInflateRing) {
    // SP_TYPE_NONE with default gen_num_per_cycle=1 must NOT inflate state ring.
    // Non-MTP DSV4 ring: R = ceil_even((1+overlap)*ratio + 0) = 8 for CSA.
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     rc;
    rc.max_generate_batch_size                      = 2;
    rc.fifo_scheduler_config.max_context_batch_size = 1;
    KVCacheConfig kvc;
    kvc.seq_size_per_block        = 128;
    kvc.kernel_seq_size_per_block = 128;
    kvc.test_block_num            = 50;
    SpeculativeExecutionConfig sp_none;  // type=SP_TYPE_NONE, gen_num_per_cycle=1
    auto config = CacheConfigCreator::createConfig(mc, pc, rc, kvc, std::nullopt, std::make_optional(sp_none));
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    // CSA_STATE (pool 4): ratio=4, overlap=1, gen_num=0 → R=8
    auto* csa = dynamic_cast<const FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "csa_state")).get());
    ASSERT_NE(csa, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*csa, kDsv4CsaStateEntryBytes), 8u) << "SP_TYPE_NONE should not inflate ring";
}

TEST(HybridPoolConfigCreatorTest, BlockIdConsistencyAcrossGroups) {
    // DSV4 has multiple semantic cache tags per logical layer. The config must expose
    // every tag's group id for the layer so model/runtime code can request the
    // correct group by tag.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    // Verify every layer exposes its complete group ids directly.
    const auto layer_group_ids = config.layerGroupIdsSnapshot();
    EXPECT_EQ(layer_group_ids.size(), 61u);
    for (size_t i = 0; i < layer_group_ids.size(); i++) {
        EXPECT_FALSE(layer_group_ids[i].empty()) << "layer " << i;
    }

    // Verify group layer ids: each group has the correct layer list.
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")),
              config.layerIdsForGroup(gidForTag(config, "indexer_kv")));
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")),
              config.layerIdsForGroup(gidForTag(config, "indexer_state")));
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")),
              config.layerIdsForGroup(gidForTag(config, "csa_state")));
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")),
              config.layerIdsForGroup(gidForTag(config, "hca_state")));
}

// ============================================================
// Helper: build a DSV4 CacheConfig with block_num set for allocator tests
// ============================================================

static CacheConfig makeDSV4AllocatorConfig(bool use_flash = false) {
    auto              mc = use_flash ? makeFlashModelConfig() : makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    // Set enough blocks for tests (7 groups × N blocks each)
    config.finalizeBlockNums(/*global_block_num=*/200, RuntimeConfig{});
    return config;
}

static CacheConfig makeDSV4CpAllocatorConfig(uint32_t cp_size) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = cp_size;
    pc.prefill_cp_config.kv_cache_sharded = true;
    auto config                           = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.block_num                      = 200;
    setGroupBlockNumsForTest(config, std::vector<uint32_t>(static_cast<size_t>(config.groupNums()), config.block_num));
    return config;
}

// ============================================================
// HybridTypeKVCacheAllocator integration tests with DSV4 7-group config
// ============================================================

class DSV4AllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

TEST_F(DSV4AllocatorTest, InitAndBasicProperties) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // 7 groups → HybridTypeKVCacheAllocator path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(allocator->seqSizePerBlock(), static_cast<int>(config.seq_size_per_block));
    EXPECT_EQ(allocator->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator->freeBlocksNum(), config.block_num - 1);
}

TEST_F(DSV4AllocatorTest, CpPageRrFixedAndSwaAllocateOneBlockPerVirtualBlock) {
    constexpr uint32_t cp_size   = 4;
    auto               config    = makeDSV4CpAllocatorConfig(cp_size);
    auto               allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int spb     = allocator->seqSizePerBlock();
    const int seq_len = static_cast<int>(cp_size) * spb;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(0, static_cast<int>(cp_size), spb));

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, seq_len + spb, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(seq_len, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = false;
    info.reuse_cache         = false;

    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    for (int gid = 0; gid < 7; ++gid) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 1u) << "gid=" << gid;
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, FlashInitAndBasicProperties) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(allocator->totalBlocksNum(), config.block_num - 1);
}

TEST_F(DSV4AllocatorTest, AddressLookupAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // Verify address lookup works for a layer in each group
    // Group 0 (CSA KV): csa_layer_ids[0]
    // Group 1 (HCA KV): hca_layer_ids[0]
    // Group 6 (SWA KV): all_layer_ids[0]
    for (int gid = 0; gid < 7; gid++) {
        ASSERT_FALSE(config.layerIdsForGroup(gid).empty()) << "group " << gid << " has no layers";
        int  layer_id = config.layerIdsForGroup(gid)[0];
        auto addr     = allocator->convertIndexToAddr(layer_id, gid, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr) << "null kv_addr for group " << gid << " layer " << layer_id;
    }
}

TEST_F(DSV4AllocatorTest, DeviceBlockPoolCreatedWithCorrectTensors) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    // allLayerCacheBase should return tensors for all 61 layers
    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.topology().layers().size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.topology().layers().size(); ++i) {
        for (const auto& tag : layout.topology().layer(static_cast<int>(i)).group_tags) {
            EXPECT_TRUE(layout.group(tag).hasLayer(i)) << "undefined kv buffer for layer " << i << " tag=" << tag;
        }
    }
}

TEST_F(DSV4AllocatorTest, ConvertIndexToBufferAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // convertIndexToBuffer should work for layers in each of the 7 groups
    for (int gid = 0; gid < 7; gid++) {
        int  layer_id = config.layerIdsForGroup(gid)[0];
        auto buf      = allocator->convertIndexToBuffer(layer_id, gid, /*block_id=*/1);
        ASSERT_FALSE(buf.empty()) << "empty buffer for group " << gid;
        EXPECT_NE(buf[0].addr, nullptr) << "null addr for group " << gid;
    }
}

TEST_F(DSV4AllocatorTest, MallocAndFreeBlocks) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 3u);

    // Direct block pool malloc/free
    auto blocks = block_pool->malloc(3);
    ASSERT_TRUE(blocks.has_value());
    ASSERT_EQ(blocks->size(), 3u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 3);

    block_pool->free(*blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(DSV4AllocatorTest, SevenGroupLayerMapping) {
    auto config = makeDSV4AllocatorConfig();

    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 31u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "indexer_kv")).size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "indexer_state")).size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_state")).size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_state")).size(), 31u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 61u);

    EXPECT_EQ(config.typeForGroup(gidForTag(config, "csa_kv")), CacheGroupType::FULL);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "hca_kv")), CacheGroupType::FULL);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "indexer_kv")), CacheGroupType::FULL);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "indexer_state")), CacheGroupType::SWA);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "csa_state")), CacheGroupType::SWA);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "hca_state")), CacheGroupType::SWA);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "swa_kv")), CacheGroupType::SWA);
}

TEST_F(DSV4AllocatorTest, SpecBlockSizesMatchPoolSpecs) {
    auto config = makeDSV4AllocatorConfig();

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_kv"))->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_state"))->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_state"))->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_state"))->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(),
              kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST_F(DSV4AllocatorTest, KVBlockStrideIsMaxAcrossGroups) {
    auto config = makeDSV4AllocatorConfig();

    // kv_block_stride_bytes should be the max block_size_bytes across all 7 pools
    size_t expected_max = 0;
    for (int i = 0; i < kDsv4PoolNum; i++) {
        expected_max = std::max(expected_max, config.specForGroup(i)->block_size_bytes());
    }
    EXPECT_EQ(config.kv_block_stride_bytes, expected_max);
    // HCA_STATE has the largest per-block bytes (128 entries * 1024 * 4)
    EXPECT_EQ(expected_max, config.specForGroup(gidForTag(config, "hca_state"))->block_size_bytes());
}

TEST_F(DSV4AllocatorTest, HCAStateIsExcludedFromReuseCachePolicy) {
    auto config = makeDSV4AllocatorConfig();
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    ASSERT_EQ(config.groupPoliciesSnapshot().size(), static_cast<size_t>(config.groupNums()));

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        if (config.tagForGroup(gid) == "hca_state") {
            EXPECT_EQ(config.policyForGroup(gid).enable_prefix_reuse, false) << "HCA_STATE should skip reuse cache";
        } else {
            EXPECT_EQ(config.policyForGroup(gid).enable_prefix_reuse, true) << "group " << gid;
        }
    }
}

// ============================================================
// Flash config: allocator integration
// ============================================================

TEST_F(DSV4AllocatorTest, FlashGroupTypes) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);

    // Flash: 21 CSA + 20 HCA + 2 SWA-only = 43 layers
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 43u);

    EXPECT_EQ(config.typeForGroup(gidForTag(config, "csa_kv")), CacheGroupType::FULL);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "hca_kv")), CacheGroupType::FULL);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "indexer_kv")), CacheGroupType::FULL);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "indexer_state")), CacheGroupType::SWA);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "csa_state")), CacheGroupType::SWA);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "hca_state")), CacheGroupType::SWA);
    EXPECT_EQ(config.typeForGroup(gidForTag(config, "swa_kv")), CacheGroupType::SWA);
}

TEST_F(DSV4AllocatorTest, FlashAddressLookupAllGroups) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    for (int gid = 0; gid < 7; gid++) {
        ASSERT_FALSE(config.layerIdsForGroup(gid).empty()) << "Flash group " << gid << " has no layers";
        int  layer_id = config.layerIdsForGroup(gid)[0];
        auto addr     = allocator->convertIndexToAddr(layer_id, gid, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr) << "Flash null kv_addr for group " << gid;
    }
}

TEST_F(DSV4AllocatorTest, FlashBlockPoolTensors) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.topology().layers().size(), 43u);
    for (size_t i = 0; i < layout.topology().layers().size(); ++i) {
        for (const auto& tag : layout.topology().layer(static_cast<int>(i)).group_tags) {
            EXPECT_TRUE(layout.group(tag).hasLayer(i)) << "Flash undefined kv buffer for layer " << i << " tag=" << tag;
        }
    }
}

TEST_F(DSV4AllocatorTest, FlashLayerMapping) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);

    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "indexer_kv")).size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "indexer_state")).size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_state")).size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_state")).size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 43u);
}

TEST_F(DSV4AllocatorTest, FlashSpecBlockSizes) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_kv"))->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(),
              kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST_F(DSV4AllocatorTest, FlashMallocAndFree) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto   block_pool  = allocator->getDeviceBlockPool();
    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 5u);

    auto blocks = block_pool->malloc(5);
    ASSERT_TRUE(blocks.has_value());
    ASSERT_EQ(blocks->size(), 5u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 5);

    block_pool->free(*blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ============================================================
// Prefix cache: insertIntoCache skips HCA_STATE but keeps other groups reusable.
// ============================================================

TEST_F(DSV4AllocatorTest, InsertIntoCacheAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getDeviceBlockPool();

    // Manually set up a BatchKVCacheResource with blocks for all 7 groups
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);

    CacheKeysType keys = {200, 201, 202, 203};
    batch_res->setBatchCacheKeys(0, keys);

    // Allocate 3 blocks per group (simulating 3 full blocks)
    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(3);
        ASSERT_TRUE(blocks.has_value());
        ASSERT_EQ(blocks->size(), 3u);
        block_pool->incRef(*blocks, BlockRefType::REQUEST);
        batch_res->mutableBlockIds(0, gid).assign(*blocks);
    }

    // Create CompleteTokenIds: 3 full blocks * seq_size_per_block tokens + partial
    int  seq_size_per_block         = allocator->seqSizePerBlock();
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    int  total_tokens               = 3 * seq_size_per_block + 1;  // 3 full blocks + 1 partial
    generate_input->input_ids       = torch::arange(total_tokens, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    InsertInfo insert_info{batch_res, complete_token_ids, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    auto match = allocator->blockTreeCacheOwner()->match(CacheKeysType{200, 201, 202});
    EXPECT_EQ(match.matched_blocks, 3u);

    // HCA_STATE is runtime scratch state and must not be part of the declarative tree.
    for (int gid = 0; gid < 7; gid++) {
        const auto& tag = config.tagForGroup(gid);
        const auto  it  = match.group_block_indices.find(tag);
        if (tag == "hca_state") {
            EXPECT_EQ(it, match.group_block_indices.end());
            continue;
        }
        ASSERT_NE(it, match.group_block_indices.end()) << tag;
        EXPECT_EQ(it->second.size(), config.typeForGroup(gid) == CacheGroupType::FULL ? 3u : 1u) << tag;
    }
    allocator->blockTreeCacheOwner()->releaseMatchedBlocks(match.matched_block_sets);

    // Free all blocks
    for (int gid = 0; gid < 7; gid++) {
        const auto& blocks = batch_res->blocks(0, gid);
        block_pool->decRef(blocks, BlockRefType::REQUEST);
    }
}

// ============================================================
// Prefix cache: Flash config insertIntoCache skips HCA_STATE.
// ============================================================

TEST_F(DSV4AllocatorTest, FlashInsertIntoCacheAllGroups) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getDeviceBlockPool();

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);

    CacheKeysType keys = {300, 301, 302, 303};
    batch_res->setBatchCacheKeys(0, keys);

    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(3);
        ASSERT_TRUE(blocks.has_value());
        ASSERT_EQ(blocks->size(), 3u);
        block_pool->incRef(*blocks, BlockRefType::REQUEST);
        batch_res->mutableBlockIds(0, gid).assign(*blocks);
    }

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    int  total_tokens               = 3 * seq_size_per_block + 1;
    generate_input->input_ids       = torch::arange(total_tokens, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    InsertInfo insert_info{batch_res, complete_token_ids, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    auto match = allocator->blockTreeCacheOwner()->match(CacheKeysType{300, 301, 302});
    EXPECT_EQ(match.matched_blocks, 3u);

    for (int gid = 0; gid < 7; gid++) {
        const auto& tag = config.tagForGroup(gid);
        const auto  it  = match.group_block_indices.find(tag);
        if (tag == "hca_state") {
            EXPECT_EQ(it, match.group_block_indices.end());
            continue;
        }
        ASSERT_NE(it, match.group_block_indices.end()) << tag;
        EXPECT_EQ(it->second.size(), config.typeForGroup(gid) == CacheGroupType::FULL ? 3u : 1u) << tag;
    }
    allocator->blockTreeCacheOwner()->releaseMatchedBlocks(match.matched_block_sets);

    for (int gid = 0; gid < 7; gid++) {
        block_pool->decRef(batch_res->blocks(0, gid), BlockRefType::REQUEST);
    }
}

// ============================================================
// Prefix cache: paged FULL groups reuse; reusable SWA/state groups require a matched latest tail block.
// ============================================================

TEST_F(DSV4AllocatorTest, PrefixCacheReusePagedGroupsOnly) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // Pre-populate a physically complete declarative path for all reusable groups.
    constexpr int group_num   = 7;
    CacheKeysType cached_keys = {100, 101, 102};
    const auto    seeded      = seedCompleteBlockTreePath(allocator, cached_keys);
    ASSERT_TRUE(seeded.success);

    // Now do a malloc with reuse enabled — keys {100,101,102,103}
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    int  seq_len                    = 3 * seq_size_per_block + 1;  // 3 full + partial
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::arange(seq_len, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    MallocInfo info{batch_res, complete_token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "Prefix cache reuse should work with paged DSV4 groups";

    for (int gid = 0; gid < group_num; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 3u) << config.tagForGroup(gid);
        if (config.typeForGroup(gid) == CacheGroupType::FULL) {
            const auto& cached_blocks = seeded.blocks_by_tag.at(config.tagForGroup(gid));
            EXPECT_EQ(out_blocks[0], cached_blocks[0]) << config.tagForGroup(gid);
            EXPECT_EQ(out_blocks[1], cached_blocks[1]) << config.tagForGroup(gid);
            continue;
        }
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1])) << config.tagForGroup(gid);
        if (config.tagForGroup(gid) == "hca_state") {
            EXPECT_TRUE(isNullBlockIdx(out_blocks[2])) << "HCA_STATE should not reuse a cached tail block";
            continue;
        }
        EXPECT_EQ(out_blocks[2], seeded.blocks_by_tag.at(config.tagForGroup(gid))[2]) << config.tagForGroup(gid);
    }

    // Clean up
    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, PrefixCacheReuseRequiresSWATailHit) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    CacheKeysType cached_keys = {100, 101, 102};
    const auto    seeded      = seedCompleteBlockTreePath(allocator, cached_keys);
    ASSERT_TRUE(seeded.success);

    size_t evicted_tail_blocks = 0;
    for (const auto& component_group : allocator->blockTreeCacheOwner()->componentGroups()) {
        if (component_group->group_type == CacheGroupType::FULL) {
            continue;
        }
        ASSERT_FALSE(component_group->tags().empty());
        evicted_tail_blocks += static_cast<size_t>(
            allocator->blockTreeCacheOwner()->evictForTag(component_group->tags().front(), cached_keys.size()));
    }
    ASSERT_GT(evicted_tail_blocks, 0u);

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    int  seq_len                    = 3 * seq_size_per_block + 1;
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::arange(seq_len, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    MallocInfo info{batch_res, complete_token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.reuse_len, 0) << "SWA tail miss should veto paged prefix reuse";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, PrefixCacheReuseDoesNotRequireHCAStateHit) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    CacheKeysType cached_keys = {1100, 1101, 1102};
    const auto    seeded      = seedCompleteBlockTreePath(allocator, cached_keys);
    ASSERT_TRUE(seeded.success);

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{1100, 1101, 1102, 1103});

    const int spb       = allocator->seqSizePerBlock();
    auto      cti       = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto      gi        = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(3 * spb + 1, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "HCA_STATE miss should not veto DSV4 prefix reuse";
    const auto hca_state_gid = gidForTag(config, "hca_state");
    const auto swa_gid       = gidForTag(config, "swa_kv");
    EXPECT_TRUE(isNullBlockIdx(batch_res->blocks(0, hca_state_gid).at(2))) << "HCA_STATE should remain non-reused";
    EXPECT_EQ(batch_res->blocks(0, swa_gid).at(2), seeded.blocks_by_tag.at("swa_kv")[2])
        << "SWA_KV tail should still gate reuse";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, PrefixCacheReuseAcceptsSingleLatestSWATailHit) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    CacheKeysType cached_keys = {100, 101, 102};
    const auto    seeded      = seedCompleteBlockTreePath(allocator, cached_keys);
    ASSERT_TRUE(seeded.success);

    // Leave only the latest coordinate resident for each fixed-tail component group.
    for (const auto& component_group : allocator->blockTreeCacheOwner()->componentGroups()) {
        if (component_group->group_type == CacheGroupType::FULL) {
            continue;
        }
        ASSERT_FALSE(component_group->tags().empty());
        EXPECT_EQ(allocator->blockTreeCacheOwner()->evictForTag(component_group->tags().front(), 2), 2);
    }

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});

    const int spb       = allocator->seqSizePerBlock();
    auto      cti       = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto      gi        = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(3 * spb + 1, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "latest SWA tail hit should allow paged prefix reuse";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, FlashPrefixCacheReusePagedGroupsOnly) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    constexpr int group_num   = 7;
    CacheKeysType cached_keys = {500, 501, 502};
    const auto    seeded      = seedCompleteBlockTreePath(allocator, cached_keys);
    ASSERT_TRUE(seeded.success);

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{500, 501, 502, 503});

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    int  seq_len                    = 3 * seq_size_per_block + 1;
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::arange(seq_len, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    MallocInfo info{batch_res, complete_token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "Flash prefix cache reuse should work for paged groups";

    for (int gid = 0; gid < group_num; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 3u) << config.tagForGroup(gid);
        if (config.typeForGroup(gid) == CacheGroupType::FULL) {
            EXPECT_EQ(out_blocks[0], seeded.blocks_by_tag.at(config.tagForGroup(gid))[0]) << config.tagForGroup(gid);
            continue;
        }
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1])) << config.tagForGroup(gid);
        if (config.tagForGroup(gid) == "hca_state") {
            EXPECT_TRUE(isNullBlockIdx(out_blocks[2])) << "Flash HCA_STATE should not reuse a cached tail block";
            continue;
        }
        EXPECT_EQ(out_blocks[2], seeded.blocks_by_tag.at(config.tagForGroup(gid))[2]) << config.tagForGroup(gid);
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, HybridPoolReserveBlocksAreDistributedAcrossGroups) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridPoolAllocator>(
        config, AllocationType::DEVICE, nullptr, /*reserve_block_ratio=*/10);
    ASSERT_TRUE(allocator->init());

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{600, 601});

    const int spb       = allocator->seqSizePerBlock();
    auto      cti       = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto      gi        = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    info.verbose             = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, HybridPoolReserveBlocksDoNotReduceExplicitHcaStateCapacity) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 11);
    auto config      = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.block_num = 40;
    std::vector<uint32_t> block_nums(static_cast<size_t>(config.groupNums()), config.block_num);
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        if (config.tagForGroup(gid) == "hca_state") {
            block_nums[gid] = 11;
        }
    }
    setGroupBlockNumsForTest(config, block_nums);

    auto allocator = std::make_shared<TestDSV4HybridPoolAllocator>(
        config, AllocationType::DEVICE, nullptr, /*reserve_block_ratio=*/50);
    ASSERT_TRUE(allocator->init());

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);

    const int spb       = allocator->seqSizePerBlock();
    const int seq_len   = 10 * spb;
    auto      cti       = std::make_shared<CompleteTokenIds>(1, 1, seq_len + spb, spb);
    auto      gi        = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(seq_len, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    info.verbose             = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// SWA prefix cache: cache entries exist and the matched tail window gates reuse.
// ============================================================

TEST_F(DSV4AllocatorTest, SWAPrefixCacheRestoresTailReuse) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    CacheKeysType cached_keys = {800, 801};
    const auto    seeded      = seedCompleteBlockTreePath(allocator, cached_keys);
    ASSERT_TRUE(seeded.success);

    // Malloc with reuse — keys {800, 801, 802}
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{800, 801, 802});

    int  spb            = allocator->seqSizePerBlock();
    int  seq_len        = 2 * spb + 1;
    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(seq_len, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.reuse_len, 0);

    const auto& swa_out = batch_res->blocks(0, gidForTag(config, "swa_kv"));
    ASSERT_GE(swa_out.size(), 2u);
    EXPECT_TRUE(isNullBlockIdx(swa_out[0])) << "SWA previous matched tail is evicted after new tail allocation";
    EXPECT_EQ(swa_out[1], seeded.blocks_by_tag.at("swa_kv")[1]) << "SWA last matched tail block should remain";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// incrMalloc: decode grows sequence after initial prefill
// ============================================================

TEST_F(DSV4AllocatorTest, IncrMallocDecodeGrowsBlocks) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    int spb = allocator->seqSizePerBlock();

    // Initial malloc: 1 block worth of tokens
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{900, 901});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo init_info{batch_res, cti};
    init_info.enable_device_cache = false;
    auto init_result              = allocator->malloc(init_info);
    ASSERT_TRUE(init_result.success);

    // All 7 groups should have 1 block each
    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 1u) << "group " << gid << " should have 1 block after init";
    }

    size_t free_after_init = allocator->freeBlocksNum();

    // incrMalloc: grow to 2 blocks
    cti->setSeqLength(2 * spb);
    MallocInfo incr_info{batch_res, cti};
    incr_info.enable_device_cache = false;
    auto incr_result              = allocator->malloc(incr_info);
    ASSERT_TRUE(incr_result.success);

    // All 7 groups should now have 2 blocks each
    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 2u) << "group " << gid << " should have 2 blocks after incr";
    }

    // HCA_STATE is not reusable: decode may materialize a new tail, but the
    // skipped old tail is released, so only the other six groups consume a net
    // additional block.
    EXPECT_EQ(allocator->freeBlocksNum(), free_after_init - 6);

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// Free and reallocate: blocks return to pool
// ============================================================

TEST_F(DSV4AllocatorTest, FreeReturnsBlocksToPool) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    size_t free_before = allocator->freeBlocksNum();
    int    spb         = allocator->seqSizePerBlock();

    // Allocate
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{1000, 1001});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    size_t free_after_alloc = allocator->freeBlocksNum();
    EXPECT_LT(free_after_alloc, free_before);

    // Free
    FreeInfo free_info{batch_res};
    allocator->free(free_info);

    // All blocks should be returned
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);

    // Can allocate again
    auto batch_res2 = std::make_shared<BatchKVCacheResource>();
    batch_res2->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res2, config);
    batch_res2->setBatchCacheKeys(0, CacheKeysType{1100, 1101});

    MallocInfo info2{batch_res2, cti};
    info2.enable_device_cache = false;
    auto result2              = allocator->malloc(info2);
    ASSERT_TRUE(result2.success);

    FreeInfo free_info2{batch_res2};
    allocator->free(free_info2);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ============================================================
// Flash: incrMalloc decode path
// ============================================================

TEST_F(DSV4AllocatorTest, FlashIncrMallocDecode) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    int spb = allocator->seqSizePerBlock();

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{1200, 1201});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo init_info{batch_res, cti};
    init_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);

    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 1u) << "Flash group " << gid;
    }

    // Grow to 3 blocks
    cti->setSeqLength(3 * spb);
    MallocInfo incr_info{batch_res, cti};
    incr_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(incr_info).success);

    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 3u) << "Flash group " << gid << " after incr";
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

}  // namespace test
}  // namespace rtp_llm
