#include <gtest/gtest.h>
#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/config_creator/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/config_creator/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/allocator/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/allocator/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/spec/OpaqueKVCacheSpec.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
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

constexpr int      kDsv4PoolNum           = 7;
constexpr uint32_t kDsv4TokensPerBlock    = 128;
constexpr uint32_t kDsv4KvEntryBytes      = 1024;
constexpr uint32_t kDsv4IndexerEntryBytes = 256;
constexpr uint32_t kDsv4Fp8KvEntryBytes   = 584;
const std::vector<std::string> kDsv4FlashFirstSeenTags = {
    "swa_kv", "csa_kv", "indexer_kv", "indexer_state", "csa_state", "hca_kv", "hca_state"};
const std::vector<std::string> kDsv4ProFirstSeenTags = {
    "hca_kv", "hca_state", "swa_kv", "csa_kv", "indexer_kv", "indexer_state", "csa_state"};

static size_t gidForTag(const CacheConfig& config, const std::string& tag) {
    return static_cast<size_t>(config.groupIdForTag(tag));
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

[[maybe_unused]] auto* const dsv4_cache_test_env =
    ::testing::AddGlobalTestEnvironment(new DSV4CacheTestEnvironment());

}  // namespace

static KVCacheConfig makeDsv4KvCacheConfig() {
    KVCacheConfig config;
    config.seq_size_per_block = 128;
    return config;
}

[[maybe_unused]] static void setGroupBlockNumsForTest(CacheConfig& config, const std::vector<uint32_t>& block_nums) {
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
    batch_res.initGroups(config.groupNums(),
                         static_cast<int>(config.layer_all_num),
                         config.layerGroupIdsSnapshot(),
                         config.kernelBlocksPerKvBlock(),
                         config.groupTypesSnapshot());
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
    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < 61; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    mc.attn_config.layer_compress_ratios = ratios;
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
    auto config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_EQ(config.groupTagsSnapshot(), kDsv4ProFirstSeenTags);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 31u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 61u);
}

TEST(HybridPoolConfigCreatorTest, FlashLayerClassification) {
    ParallelismConfig pc;
    auto config = CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(config.groupTagsSnapshot(), kDsv4FlashFirstSeenTags);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 43u);
}

TEST(HybridPoolConfigCreatorTest, MtpSwaOnlyLayerIsNotStripped) {
    ParallelismConfig pc;
    auto              config =
        CacheConfigCreator::createBasicConfig(makeFlashMtpModelConfig(), pc, makeDsv4KvCacheConfig(), true, 0);

    EXPECT_EQ(config.layer_num, 1u);
    EXPECT_GT(config.pagedBlockSizeBytes(), 0u);
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
    auto config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

    const std::vector<std::string> expected_tags = {
        "swa_kv", "csa_state", "indexer_state", "indexer_kv", "csa_kv", "hca_state", "hca_kv"};
    EXPECT_EQ(config.groupTagsSnapshot(), expected_tags);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), expected_tags.size());
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), expected_tags.size());
    for (size_t gid = 0; gid < expected_tags.size(); ++gid) {
        ASSERT_NE(config.specForGroup(gid), nullptr);
        EXPECT_EQ(config.specForGroup(gid)->tag, expected_tags[gid]) << "gid=" << gid;
        EXPECT_EQ(config.specForGroup(gid)->layers, config.layerIdsForGroup(gid)) << "gid=" << gid;
    }

    EXPECT_EQ(config.groupIdForLayerTag(2, "csa_kv"), config.groupIdForTag("csa_kv"));
    EXPECT_EQ(config.groupIdForLayerTag(3, "hca_kv"), config.groupIdForTag("hca_kv"));
    EXPECT_EQ(config.groupIdForLayerTag(0, "swa_kv"), config.groupIdForTag("swa_kv"));
}

static GroupInfo makeTestGroup(const KVCacheSpecPtr& spec, CacheGroupType type, std::vector<int> layer_ids) {
    GroupInfo group;
    group.spec      = spec;
    group.policy    = defaultCacheGroupPolicy(type);
    group.layer_ids = std::move(layer_ids);
    return group;
}

TEST(CacheConfigTest, SetTopologyInstallsTagAndGroupTopology) {
    CacheConfig config;
    config.layer_num     = 3;
    config.layer_all_num = 3;

    auto swa_spec = std::make_shared<FixedStateCacheSpec>();
    swa_spec->tag = "swa";
    swa_spec->state_dim = 1;
    swa_spec->entries_per_block = 1;
    swa_spec->store_dtype = DataType::TYPE_UINT8;
    swa_spec->dtype = DataType::TYPE_UINT8;
    auto csa_spec = std::make_shared<CompressedKVCacheSpec>();
    csa_spec->tag = "csa";
    csa_spec->entry_elems = 1;
    csa_spec->entries_per_block = 1;
    csa_spec->compression_ratio = 1;
    csa_spec->store_dtype = DataType::TYPE_UINT8;
    csa_spec->dtype = DataType::TYPE_UINT8;

    std::vector<LayerInfo> layers(3);
    layers[0].group_ids = {0};
    layers[0].tag_to_gid["swa"] = 0;
    layers[1].group_ids = {0, 1};
    layers[1].tag_to_gid["swa"] = 0;
    layers[1].tag_to_gid["csa"] = 1;
    layers[2].group_ids = {0};
    layers[2].tag_to_gid["swa"] = 0;

    config.setTopology({makeTestGroup(swa_spec, CacheGroupType::SWA, {0, 1, 2}),
                        makeTestGroup(csa_spec, CacheGroupType::FULL, {1})},
                       std::move(layers));

    EXPECT_EQ(config.groupTagsSnapshot(), std::vector<std::string>({"swa", "csa"}));
    EXPECT_EQ(config.groupIdForLayerTag(1, "swa"), 0);
    EXPECT_EQ(config.groupIdForLayerTag(1, "csa"), 1);
    EXPECT_THROW((void)config.groupIdFor(1), std::exception);
    EXPECT_EQ(config.layerGroupIdsSnapshot()[1], std::vector<int>({0, 1}));
}

TEST(CacheConfigTest, SetTopologyRejectsMissingLayer) {
    CacheConfig config;
    config.layer_num     = 2;
    config.layer_all_num = 2;

    auto spec = std::make_shared<MHAKVCacheSpec>();
    spec->tag = "default";
    std::vector<LayerInfo> layers(2);
    layers[0].group_ids = {0};
    layers[0].tag_to_gid["default"] = 0;
    EXPECT_THROW(config.setTopology({makeTestGroup(spec, CacheGroupType::FULL, {0})}, std::move(layers)),
                 std::exception);
}

TEST(CacheConfigTest, SetTopologyRejectsEmptyTag) {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;

    auto spec = std::make_shared<MHAKVCacheSpec>();
    std::vector<LayerInfo> layers(1);
    layers[0].group_ids = {0};
    EXPECT_THROW(config.setTopology({makeTestGroup(spec, CacheGroupType::FULL, {0})}, std::move(layers)),
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

    std::vector<LayerInfo> layers(1);
    layers[0].group_ids = {0, 1};
    layers[0].tag_to_gid["full"] = 0;
    layers[0].tag_to_gid["linear"] = 1;
    EXPECT_NO_THROW(config.setTopology({makeTestGroup(spec0, CacheGroupType::FULL, {0}),
                                        makeTestGroup(spec1, CacheGroupType::LINEAR, {0})},
                                       std::move(layers)));
    EXPECT_EQ(config.layerGroupIdsSnapshot()[0].size(), 2u);
}

TEST(HybridPoolConfigCreatorTest, Dsv4ModelProvidedAlignmentPropagatesToCacheSpecs) {
    auto mc = makeFlashModelConfig();
    for (auto& layer_descs : mc.kv_cache_spec_descs) {
        for (auto& desc : layer_descs) {
            if (desc.tag == "csa_kv") {
                desc.block_size_bytes_alignment = 1024;
            } else if (desc.tag == "swa_kv") {
                desc.block_size_bytes_alignment        = 2048;
                desc.block_size_alignment_min_entries = 256;
            }
        }
    }

    ParallelismConfig pc;
    auto config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

    auto* csa_kv = dynamic_cast<CompressedKVCacheSpec*>(config.specForGroup(gidForTag(config, "csa_kv")).get());
    auto* swa_kv = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "swa_kv")).get());
    ASSERT_NE(csa_kv, nullptr);
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(csa_kv->block_size_bytes_alignment, 1024u);
    EXPECT_EQ(swa_kv->block_size_bytes_alignment, 2048u);
    EXPECT_EQ(swa_kv->block_size_alignment_min_entries, 256u);
}

TEST(HybridPoolConfigCreatorTest, Dsv4TagRoutesAreConsistent) {
    ParallelismConfig pc;
    auto              config =
        CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);

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

    auto mtp_config =
        CacheConfigCreator::createBasicConfig(makeFlashMtpModelConfig(), pc, makeDsv4KvCacheConfig(), true, 0);
    ASSERT_EQ(mtp_config.groupIdForLayerTag(0, "swa_kv"), 0);
}

TEST(HybridPoolConfigCreatorTest, Dsv4GroupPoliciesMatchLegacyBehavior) {
    ParallelismConfig pc;
    auto              config =
        CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);

    ASSERT_EQ(config.groupPoliciesSnapshot().size(), static_cast<size_t>(config.groupNums()));
    auto expect_policy = [&](const std::string& tag,
                             CacheReusePolicy reuse_policy,
                             CacheEvictPolicy evict_policy,
                             int active_tail_blocks) {
        const auto group_tags = config.groupTagsSnapshot();
        auto       it         = std::find(group_tags.begin(), group_tags.end(), tag);
        ASSERT_NE(it, group_tags.end()) << tag;
        const auto gid = static_cast<size_t>(std::distance(group_tags.begin(), it));
        EXPECT_EQ(config.policyForGroup(gid).reuse_policy, reuse_policy) << tag;
        EXPECT_EQ(config.policyForGroup(gid).evict_policy, evict_policy) << tag;
        EXPECT_EQ(config.policyForGroup(gid).active_tail_blocks, active_tail_blocks) << tag;
    };

    expect_policy("hca_state", CacheReusePolicy::NON_REUSABLE, CacheEvictPolicy::INDEPENDENT, 1);
    expect_policy("swa_kv", CacheReusePolicy::REUSABLE, CacheEvictPolicy::INDEPENDENT, 2);
    expect_policy("csa_state", CacheReusePolicy::REUSABLE, CacheEvictPolicy::INDEPENDENT, 2);
    expect_policy("csa_kv", CacheReusePolicy::REUSABLE, CacheEvictPolicy::CHAIN, 0);
    expect_policy("hca_kv", CacheReusePolicy::REUSABLE, CacheEvictPolicy::CHAIN, 0);
    expect_policy("indexer_kv", CacheReusePolicy::REUSABLE, CacheEvictPolicy::CHAIN, 0);
}

TEST(HybridPoolConfigCreatorTest, Dsv4SpecsMissingFailsFastWithoutRatioFallback) {
    auto mc = makeFlashModelConfig();
    mc.kv_cache_spec_descs.clear();

    ParallelismConfig pc;
    EXPECT_THROW((void)CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0),
                 std::exception);
}

// ============================================================
// Pool specs
// ============================================================

TEST(HybridPoolConfigCreatorTest, ProPoolSpecs) {
    ParallelismConfig pc;
    auto config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);

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
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(HybridPoolConfigCreatorTest, FlashPoolSpecs) {
    ParallelismConfig pc;
    auto config = CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "hca_kv")).size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 43u);
}

// ============================================================
// Block size bytes
// ============================================================

TEST(HybridPoolConfigCreatorTest, BlockSizeBytes) {
    ParallelismConfig pc;
    auto config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_kv"))->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "indexer_state"))->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "csa_state"))->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "hca_state"))->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(HybridPoolConfigCreatorTest, Fp8BlockSizeBytesUsePaddedPhysicalStride) {
    ParallelismConfig pc;
    auto              mc          = makeProModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    setDsv4KvCacheSpecs(mc);
    auto config                   = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

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

TEST(HybridPoolConfigCreatorTest, DecoupledPhysicalAndKernelBlockSizeUsesPerGroupBpk) {
    ParallelismConfig pc;
    auto              mc = makeProModelConfig();
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block        = 16384;
    kv_cache_config.kernel_seq_size_per_block = 128;
    auto config = CacheConfigCreator::createBasicConfig(mc, pc, kv_cache_config, false, 0);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    const auto group_seq_size_per_block = config.groupSeqSizePerBlockSnapshot();
    ASSERT_EQ(group_seq_size_per_block.size(), 7u);

    EXPECT_EQ(config.seq_size_per_block, 16384u);
    EXPECT_EQ(config.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(config.kernelBlocksPerKvBlock(), 128u);
    for (size_t gid = 0; gid < group_seq_size_per_block.size(); ++gid) {
        EXPECT_EQ(group_seq_size_per_block[gid], 16384u);
    }

    auto* csa_kv = dynamic_cast<CompressedKVCacheSpec*>(config.specForGroup(gidForTag(config, "csa_kv")).get());
    auto* hca_kv = dynamic_cast<CompressedKVCacheSpec*>(config.specForGroup(gidForTag(config, "hca_kv")).get());
    auto* idx_kv = dynamic_cast<CompressedKVCacheSpec*>(config.specForGroup(gidForTag(config, "indexer_kv")).get());
    auto* swa_kv = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "swa_kv")).get());
    ASSERT_NE(csa_kv, nullptr);
    ASSERT_NE(hca_kv, nullptr);
    ASSERT_NE(idx_kv, nullptr);
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(csa_kv->compression_ratio, 4u);
    EXPECT_EQ(hca_kv->compression_ratio, 128u);
    EXPECT_EQ(idx_kv->compression_ratio, 4u);
    EXPECT_EQ(csa_kv->entries_per_block, 32u);
    EXPECT_EQ(hca_kv->entries_per_block, 1u);
    EXPECT_EQ(idx_kv->entries_per_block, 32u);
    EXPECT_EQ(swa_kv->entries_per_block, 128u);

    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "csa_kv")),
              config.specForGroup(gidForTag(config, "csa_kv"))->block_size_bytes() * 128u);
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "hca_kv")),
              config.specForGroup(gidForTag(config, "hca_kv"))->block_size_bytes() * 128u);
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "indexer_kv")),
              config.specForGroup(gidForTag(config, "indexer_kv"))->block_size_bytes() * 128u);
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(gidForTag(config, "swa_kv")),
              config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes());

    RuntimeConfig runtime_config;
    config.finalizeBlockNums(100, runtime_config);
    auto full_pool = BlockPoolConfigHelper::createConfigForGroup(config, gidForTag(config, "csa_kv"));
    auto swa_pool  = BlockPoolConfigHelper::createConfigForGroup(config, gidForTag(config, "swa_kv"));
    ASSERT_EQ(full_pool.memory_layouts.size(), 1u);
    ASSERT_EQ(swa_pool.memory_layouts.size(), 1u);
    EXPECT_EQ(full_pool.memory_layouts[0].kernel_blocks_per_kv_block, 128u);
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
    auto config                   = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

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
        EXPECT_EQ(config.groupSeqSizePerBlockForGroup(gid), kDsv4TokensPerBlock * 4u) << "tag=" << tag;
    }

    pc.role_type       = RoleType::DECODE;
    auto decode_config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(decode_config.specForGroup(gidForTag(decode_config, "indexer_state"))->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(decode_config.specForGroup(gidForTag(decode_config, "csa_state"))->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(decode_config.specForGroup(gidForTag(decode_config, "hca_state"))->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(decode_config.specForGroup(gidForTag(decode_config, "swa_kv"))->block_size_bytes(), 74880u);
}

TEST(KVCacheTransferPlannerTest, CpCompactSwaUsesCanonicalTailRows) {
    auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/8,
                                         /*reuse_block_size=*/0,
                                         /*use_hybrid=*/true,
                                         CacheGroupType::SWA,
                                         /*cp_rank=*/0,
                                         /*cp_size=*/4);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].key_index, 3);
    EXPECT_EQ(plan[0].offset_index, 0);
    EXPECT_EQ(plan[1].key_index, 7);
    EXPECT_EQ(plan[1].offset_index, 1);
}

TEST(KVCacheTransferPlannerTest, CpCompactSwaKeepsPartialTailRows) {
    {
        auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/1,
                                             /*reuse_block_size=*/0,
                                             /*use_hybrid=*/true,
                                             CacheGroupType::SWA,
                                             /*cp_rank=*/0,
                                             /*cp_size=*/2);
        ASSERT_EQ(plan.size(), 1u);
        EXPECT_EQ(plan[0].key_index, 0);
        EXPECT_EQ(plan[0].offset_index, 0);
    }
    {
        auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/11,
                                             /*reuse_block_size=*/0,
                                             /*use_hybrid=*/true,
                                             CacheGroupType::SWA,
                                             /*cp_rank=*/0,
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
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

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
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "swa_kv")).size(), 43u);
    EXPECT_EQ(config.layerIdsForGroup(gidForTag(config, "csa_kv")).size(), 21u);
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionIndependentPoolUsesHybridPoolConfig) {
    ParallelismConfig pc;
    auto              config =
        CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(true), pc, KVCacheConfig{}, false, 0);

    EXPECT_TRUE(config.use_independent_block_pools);
    ASSERT_EQ(config.groupNums(), 2);
    const auto group_types = config.groupTypesSnapshot();
    EXPECT_EQ(std::count(group_types.begin(), group_types.end(), CacheGroupType::FULL), 1);
    EXPECT_EQ(std::count(group_types.begin(), group_types.end(), CacheGroupType::LINEAR), 1);
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 2u);
    EXPECT_LT(config.specForGroup(gidForTag(config, "full"))->block_size_bytes(),
              config.specForGroup(gidForTag(config, "linear"))->block_size_bytes());
    EXPECT_EQ(config.groupBlockNumsSnapshot().size(), 2u);
    EXPECT_EQ(config.groupTagsSnapshot(), std::vector<std::string>({"linear", "full"}));
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionIndependentPoolSplitsFullAndSwaSpecs) {
    auto mc = makeHybridAttentionModelConfig(true);
    mc.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE,
                                                        HybridAttentionType::SLIDING_WINDOW,
                                                        HybridAttentionType::LINEAR,
                                                        HybridAttentionType::SLIDING_WINDOW};
    setHybridAttentionKvCacheSpecs(mc);

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, KVCacheConfig{}, false, 0);

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
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionWithoutIndependentPoolKeepsSharedHybridConfig) {
    ParallelismConfig pc;
    auto              config =
        CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(false), pc, KVCacheConfig{}, false, 0);

    EXPECT_FALSE(config.use_independent_block_pools);
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_EQ(config.groupBlockNumsSnapshot(), std::vector<uint32_t>({0u, 0u}));
}

TEST(HybridConfigCreatorTest, HybridAttentionTypesMustCoverAllLayers) {
    auto mc = makeHybridAttentionModelConfig(false);
    mc.hybrid_attention_config.hybrid_attention_types.pop_back();

    ParallelismConfig pc;
    EXPECT_THROW((void)CacheConfigCreator::createBasicConfig(mc, pc, KVCacheConfig{}, false, 0),
                 std::exception);
}

// ============================================================
// Generic opaque cache specs
// ============================================================

TEST(GenericOpaqueCacheSpecTest, KVSpecFromPoolSpec) {
    CompressedKVCacheSpec spec("csa_kv",
                               kDsv4Fp8KvEntryBytes,
                               64,
                               DataType::TYPE_UINT8,
                               kDsv4TokensPerBlock,
                               1,
                               DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES);

    EXPECT_EQ(spec.block_size(), 64u * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.natural_block_size_bytes(), 64u * kDsv4Fp8KvEntryBytes * 1u);  // uint8 = 1 byte
    EXPECT_EQ(spec.block_size_bytes(), 37440u);
    EXPECT_EQ(spec.tag, "csa_kv");
    EXPECT_EQ(spec.entry_elems, kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.entries_per_block, 64u);

    CompressedKVCacheSpec hca_spec("hca_kv",
                                    kDsv4Fp8KvEntryBytes,
                                    2,
                                    DataType::TYPE_UINT8,
                                    kDsv4TokensPerBlock,
                                    1,
                                    DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES);
    EXPECT_EQ(hca_spec.natural_block_size_bytes(), 2u * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(hca_spec.block_size_bytes(), 1728u);
}

TEST(GenericOpaqueCacheSpecTest, CompressedKVSpecReportsGenericKindsAndLayout) {
    CompressedKVCacheSpec spec("compressed",
                               kDsv4Fp8KvEntryBytes,
                               64,
                               DataType::TYPE_UINT8,
                               kDsv4TokensPerBlock,
                               4,
                               DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES);

    EXPECT_EQ(spec.type, KVCacheSpecType::OpaqueKV);
    EXPECT_EQ(spec.lifecycle, CacheGroupType::FULL);
    EXPECT_EQ(spec.block_size(), 64u * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.natural_block_size_bytes(), 64u * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.block_size_bytes(), 37440u);
    EXPECT_EQ(spec.compression_ratio, 4u);
    EXPECT_EQ(spec.cpTransferPolicy(), CPTransferPolicy::NONE);
    EXPECT_FALSE(spec.supportsCpSlice());
}

TEST(GenericOpaqueCacheSpecTest, FixedStateSpecReportsGenericKindsAndSlicesByEntries) {
    FixedStateCacheSpec spec("tail_state", 32, 8, DataType::TYPE_FP32, kDsv4TokensPerBlock);
    char                storage[8 * 32 * 4] = {};
    BlockInfo           block;
    block.addr       = storage;
    block.size_bytes = sizeof(storage);

    auto sliced = spec.sliceBlockForPeer({block}, 4, 2);
    ASSERT_EQ(sliced.size(), 1u);
    EXPECT_EQ(spec.type, KVCacheSpecType::OpaqueState);
    EXPECT_EQ(spec.lifecycle, CacheGroupType::SWA);
    EXPECT_EQ(spec.cpTransferPolicy(), CPTransferPolicy::INTRA_BLOCK_SLICE);
    EXPECT_TRUE(spec.supportsCpSlice());
    EXPECT_EQ(sliced[0].addr, storage + 2 * 2 * 32 * 4);
    EXPECT_EQ(sliced[0].size_bytes, 2u * 32u * 4u);
}

TEST(GenericOpaqueCacheSpecTest, FixedStateSpecSlicesOverrideByBytes) {
    FixedStateCacheSpec spec("tail_bytes",
                             kDsv4Fp8KvEntryBytes,
                             kDsv4TokensPerBlock,
                             DataType::TYPE_UINT8,
                             kDsv4TokensPerBlock,
                             74880);
    char                storage[74880] = {};
    BlockInfo           block;
    block.addr       = storage;
    block.size_bytes = sizeof(storage);

    auto sliced = spec.sliceBlockForPeer({block}, 4, 3);
    ASSERT_EQ(sliced.size(), 1u);
    EXPECT_EQ(sliced[0].addr, storage + 3 * (sizeof(storage) / 4));
    EXPECT_EQ(sliced[0].size_bytes, sizeof(storage) / 4);

    auto cp_sliced = spec.cpSliceDestination({block}, 4, 3);
    ASSERT_EQ(cp_sliced.size(), 1u);
    EXPECT_EQ(cp_sliced[0].addr, sliced[0].addr);
    EXPECT_EQ(cp_sliced[0].size_bytes, sliced[0].size_bytes);
}

TEST(GenericOpaqueCacheSpecTest, FixedStateSpecSlicesAlignedBlockByPhysicalBytes) {
    FixedStateCacheSpec spec("aligned_tail",
                             kDsv4Fp8KvEntryBytes,
                             132,
                             DataType::TYPE_UINT8,
                             kDsv4TokensPerBlock,
                             0,
                             DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES,
                             DSV4_SWA_WINDOW_ENTRIES);
    ASSERT_EQ(spec.natural_block_size_bytes(), 77088u);
    ASSERT_EQ(spec.block_size_bytes(), 77184u);
    char      storage[77184] = {};
    BlockInfo block;
    block.addr       = storage;
    block.size_bytes = sizeof(storage);

    auto sliced = spec.sliceBlockForPeer({block}, 2, 1);
    ASSERT_EQ(sliced.size(), 1u);
    EXPECT_EQ(sliced[0].addr, storage + 38592);
    EXPECT_EQ(sliced[0].size_bytes, 38592u);
}

TEST(GenericOpaqueCacheSpecTest, SWAFp8StateSpecUsesPaddedPhysicalBlockSize) {
    FixedStateCacheSpec spec("swa_kv",
                             kDsv4Fp8KvEntryBytes,
                             kDsv4TokensPerBlock,
                             DataType::TYPE_UINT8,
                             kDsv4TokensPerBlock,
                             0,
                             DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES,
                             DSV4_SWA_WINDOW_ENTRIES);

    EXPECT_EQ(spec.block_size(), kDsv4TokensPerBlock * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.natural_block_size_bytes(), kDsv4TokensPerBlock * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.block_size_bytes(), 74880u);
    EXPECT_EQ(spec.tag, "swa_kv");
}

TEST(GenericOpaqueCacheSpecTest, StateSpecFloat32) {
    FixedStateCacheSpec spec("csa_state", 2048, 8, DataType::TYPE_FP32, kDsv4TokensPerBlock);

    EXPECT_EQ(spec.block_size(), 8u * 2048u);
    EXPECT_EQ(spec.block_size_bytes(), 8u * 2048u * 4u);  // float32 = 4 bytes
    EXPECT_EQ(spec.tag, "csa_state");
    EXPECT_EQ(spec.state_dim, 2048u);
}

TEST(GenericOpaqueCacheSpecTest, IndexerKVSpec) {
    CompressedKVCacheSpec spec("indexer_kv", 132, 64, DataType::TYPE_UINT8, kDsv4TokensPerBlock);

    EXPECT_EQ(spec.block_size(), 64u * 132u);
    EXPECT_EQ(spec.block_size_bytes(), 64u * 132u);
    EXPECT_EQ(spec.tag, "indexer_kv");
}

TEST(GenericOpaqueCacheSpecTest, HCAStateSpec) {
    FixedStateCacheSpec spec("hca_state", 1024, 128, DataType::TYPE_FP32, kDsv4TokensPerBlock);

    EXPECT_EQ(spec.block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(spec.tag, "hca_state");
}

// ============================================================
// Pool 0/1/2 shared properties: same tokens_per_block, same num_blocks
// ============================================================

TEST(HybridPoolConfigCreatorTest, PagedPoolsShareTokensPerBlock) {
    // Pro config
    {
        ParallelismConfig pc;
        auto              config =
            CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
        for (const auto& tag : {"csa_kv", "hca_kv", "indexer_kv", "swa_kv"}) {
            EXPECT_EQ(config.groupSeqSizePerBlockForGroup(gidForTag(config, tag)), kDsv4TokensPerBlock) << tag;
        }
    }
    // Flash config
    {
        ParallelismConfig pc;
        auto              config =
            CacheConfigCreator::createBasicConfig(makeFlashModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
        for (const auto& tag : {"csa_kv", "hca_kv", "indexer_kv"}) {
            EXPECT_EQ(config.groupSeqSizePerBlockForGroup(gidForTag(config, tag)), kDsv4TokensPerBlock) << tag;
        }
    }
}

TEST(HybridPoolConfigCreatorTest, AllPagedPoolsShareBlockNum) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);
    RuntimeConfig     runtime_config;
    config.finalizeBlockNums(100, runtime_config);

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
    kv_cache_config.seq_size_per_block                          = 128;
    kv_cache_config.test_block_num                              = 100;
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
    kv_cache_config.seq_size_per_block                          = 128;
    kv_cache_config.test_block_num                              = 100;
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

TEST(CacheConfigTest, DSV4KernelSeqSizeAllowsDecoupledPhysicalBlocks) {
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

TEST(CacheConfigTest, DSV4KernelSeqSizeRejectsInvalidPhysicalKernelShape) {
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
        kv_cache_config.seq_size_per_block                          = 128;
        kv_cache_config.test_block_num                              = 100;
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
    kv_cache_config.seq_size_per_block                          = 128;
    kv_cache_config.test_block_num                              = 100;
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
    pc_tp1.tp_size = 1;
    auto config_tp1 = CacheConfigCreator::createBasicConfig(model_config, pc_tp1, KVCacheConfig{}, false, 0);
    ASSERT_EQ(static_cast<size_t>(config_tp1.groupNums()), 1u);
    EXPECT_EQ(config_tp1.specForGroup(0)->local_head_num_kv, 4u);

    ParallelismConfig pc_tp2;
    pc_tp2.tp_size = 2;
    auto config_tp2 = CacheConfigCreator::createBasicConfig(model_config, pc_tp2, KVCacheConfig{}, false, 0);
    ASSERT_EQ(static_cast<size_t>(config_tp2.groupNums()), 1u);
    EXPECT_EQ(config_tp2.specForGroup(0)->local_head_num_kv, 2u);

    EXPECT_EQ(config_tp1.specForGroup(0)->local_head_num_kv, 4u);
    EXPECT_NE(config_tp1.specForGroup(0).get(), config_tp2.specForGroup(0).get());
}

TEST(CacheConfigTest, SpecBuilderDerivesHybridPoolRuntimeFieldsFromContext) {
    SpecBuildContext ctx;
    ctx.dtype                   = DataType::TYPE_BF16;
    ctx.seq_size_per_block      = 128;
    ctx.attn_tp_size            = 1;
    ctx.kernel_tokens_per_block = 128;
    ctx.gen_num_per_cycle       = 3;
    ctx.cp_size                 = 2;
    ctx.cp_prefill_sliced       = true;

    KVCacheSpecDesc compressed_desc;
    compressed_desc.tag                                      = "compressed";
    compressed_desc.cache_type                               = CacheType::COMPRESSED_KV;
    compressed_desc.entry_elems                              = 16;
    compressed_desc.compression_ratio                        = 4;
    compressed_desc.store_dtype                              = DataType::TYPE_UINT8;
    compressed_desc.extra.derive_entries_from_kernel_block   = true;
    compressed_desc.extra.use_fixed_region_cp_tokens         = true;

    auto compressed = std::dynamic_pointer_cast<CompressedKVCacheSpec>(SpecBuilder::build(compressed_desc, ctx));
    ASSERT_NE(compressed, nullptr);
    EXPECT_EQ(compressed->entries_per_block, 32u);
    EXPECT_EQ(compressed->seq_size_per_block, 256u);
    EXPECT_EQ(compressed->dtype, DataType::TYPE_BF16);

    KVCacheSpecDesc state_desc;
    state_desc.tag                                          = "state";
    state_desc.cache_type                                   = CacheType::FIXED_STATE;
    state_desc.entry_elems                                  = 32;
    state_desc.store_dtype                                  = DataType::TYPE_FP32;
    state_desc.block_size_bytes_alignment                   = 64;
    state_desc.extra.state_ring_compression_ratio           = 4;
    state_desc.extra.state_ring_overlap                     = 1;
    state_desc.extra.state_ring_add_gen_num_per_cycle       = true;
    state_desc.extra.cp_align_entries                       = true;
    state_desc.extra.cp_slice_entries                       = true;
    state_desc.extra.cp_prefill_slice_block_bytes           = true;
    state_desc.extra.use_fixed_region_cp_tokens             = true;

    auto prefill_state = std::dynamic_pointer_cast<FixedStateCacheSpec>(SpecBuilder::build(state_desc, ctx));
    ASSERT_NE(prefill_state, nullptr);
    EXPECT_EQ(prefill_state->entries_per_block, 6u);
    EXPECT_EQ(prefill_state->block_size_bytes_override, 384u);
    EXPECT_EQ(prefill_state->seq_size_per_block, 256u);

    ctx.cp_prefill_sliced = false;
    auto decode_state = std::dynamic_pointer_cast<FixedStateCacheSpec>(SpecBuilder::build(state_desc, ctx));
    ASSERT_NE(decode_state, nullptr);
    EXPECT_EQ(decode_state->entries_per_block, 12u);
    EXPECT_EQ(decode_state->block_size_bytes_override, 0u);
    EXPECT_EQ(decode_state->seq_size_per_block, 256u);
}

TEST(CacheConfigTest, FinalizeBlockNumsIsNoopForSingleAndSharedHybridConfig) {
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
    auto single_config = CacheConfigCreator::createBasicConfig(single_model_config, pc, KVCacheConfig{}, false, 0);
    single_config.finalizeBlockNums(123, runtime_config);
    EXPECT_EQ(single_config.groupBlockNumsSnapshot(), std::vector<uint32_t>({123u}));
    EXPECT_EQ(single_config.explicitly_sized_pool_reserve_bytes, 0u);

    auto hybrid_config =
        CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(false), pc, KVCacheConfig{}, false, 0);
    hybrid_config.finalizeBlockNums(123, runtime_config);
    EXPECT_FALSE(hybrid_config.use_independent_block_pools);
    EXPECT_EQ(hybrid_config.groupBlockNumsSnapshot(), std::vector<uint32_t>({123u, 123u}));
    EXPECT_EQ(hybrid_config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, FinalizeBlockNumsAppliesToIndependentPools) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    ParallelismConfig pc;
    auto config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
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
    kv_cache_config_with.seq_size_per_block         = 128;
    kv_cache_config_with.kv_cache_mem_mb            = 65536;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", small_hca_state_pool);
    auto config_with = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config_with);

    KVCacheConfig kv_cache_config_without;
    kv_cache_config_without.seq_size_per_block         = 128;
    kv_cache_config_without.kv_cache_mem_mb            = 65536;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", large_hca_state_pool);
    auto config_without = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config_without);

    // More HCA_STATE blocks reserve more HBM and leave fewer blocks for the global pools.
    EXPECT_GT(config_with.pagedBlockNum(), config_without.pagedBlockNum());
    EXPECT_EQ(config_with.blockNumForGroup(gidForTag(config_with, "hca_kv")),
              config_with.pagedBlockNum());
    EXPECT_EQ(config_without.blockNumForGroup(gidForTag(config_without, "hca_kv")),
              config_without.pagedBlockNum());
    EXPECT_EQ(config_with.blockNumForGroup(gidForTag(config_with, "hca_state")), small_hca_state_pool);
    EXPECT_EQ(config_without.blockNumForGroup(gidForTag(config_without, "hca_state")), large_hca_state_pool);
    const size_t expected_reserve =
        static_cast<size_t>(small_hca_state_pool) * config_with.blockSizeBytesForGroup(gidForTag(config_with, "hca_state"));
    EXPECT_EQ(config_with.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

TEST(CacheConfigTest, DSV4ExplicitHcaStatePoolBlocksIgnoreLinearStep) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config = makeDsv4KvCacheConfig();
    auto config = CacheConfigCreator::createBasicConfig(makeProModelConfig(), pc, kv_cache_config, false, 0);
    config.linear_step = 4;
    config.finalizeBlockNums(100, runtime_config);

    // FULL groups: unaffected by step, get global_block_num
    const auto hca_state_gid = gidForTag(config, "hca_state");
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = gid == hca_state_gid ? 256u : 100u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
    const size_t expected_reserve = 256u * config.blockSizeBytesForGroup(hca_state_gid);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

TEST(CacheConfigTest, DSV4StateSwaPoolsWithoutExplicitBlocksUseGlobalBlocks) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block = 128;
    kv_cache_config.test_block_num     = 100;
    kv_cache_config.linear_step        = 4;
    auto mc = makeProModelConfig();
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.groupBlockNumsSnapshot().size(), static_cast<size_t>(kDsv4PoolNum));
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.blockNumForGroup(gid), 100u) << "gid=" << gid;
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
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    ASSERT_NE(config.mtp_sub_configs[0], nullptr);
    ASSERT_NE(config.mtp_sub_configs[1], nullptr);

    const auto swa_gid = gidForTag(config, "swa_kv");
    EXPECT_EQ(config.layerGroupIdsSnapshot()[43], std::vector<int>({static_cast<int>(swa_gid)}));
    EXPECT_EQ(config.layerGroupIdsSnapshot()[44], std::vector<int>({static_cast<int>(swa_gid)}));
    EXPECT_EQ(config.groupIdForLayerTag(43, "swa_kv"), static_cast<int>(swa_gid));
    EXPECT_EQ(config.groupIdForLayerTag(44, "swa_kv"), static_cast<int>(swa_gid));

    EXPECT_EQ(config.layerIdsForGroup(swa_gid).size(), 45u);

    // MTP sub-configs preserve the target/global group namespace.  Current
    // MTP execution passes block tables by gid without a draft-local remap, so
    // unused target groups stay as empty placeholders and the real SWA layer
    // keeps the same gid as the target config.
    EXPECT_EQ(config.mtp_sub_configs[0]->groupTagsSnapshot(), config.groupTagsSnapshot());
    EXPECT_EQ(config.mtp_sub_configs[1]->groupTagsSnapshot(), config.groupTagsSnapshot());
    EXPECT_EQ(config.mtp_sub_configs[0]->groupIdForLayerTag(0, "swa_kv"), static_cast<int>(swa_gid));
    EXPECT_EQ(config.mtp_sub_configs[1]->groupIdForLayerTag(0, "swa_kv"), static_cast<int>(swa_gid));
    EXPECT_EQ(config.mtp_sub_configs[0]->layerIdsForGroup(swa_gid), std::vector<int>({43}));
    EXPECT_EQ(config.mtp_sub_configs[1]->layerIdsForGroup(swa_gid), std::vector<int>({44}));
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

    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes,
              256u * config.blockSizeBytesForGroup(gidForTag(config, "hca_state")));
}

TEST(HybridPoolConfigCreatorTest, MtpGenNum2RingEntriesMatch) {
    // gen_num_per_cycle=2 -> CSA/INDEXER R=10, HCA R=130, SWA R=130.
    // Formula: R = ceil_even((1 + overlap) * ratio + gen_num_per_cycle).
    // SWA_KV is sized like the HCA state ring (window 128, overlap 0).
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    auto              config =
        CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, /*gen_num_per_cycle=*/2);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    // Pool 3: INDEXER_STATE (ratio=4, overlap=1) → R=10
    auto* indexer_state = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "indexer_state")).get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(indexer_state->entries_per_block, 10u);
    // Pool 4: CSA_STATE (ratio=4, overlap=1) → R=10
    auto* csa_state = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "csa_state")).get());
    ASSERT_NE(csa_state, nullptr);
    EXPECT_EQ(csa_state->entries_per_block, 10u);
    // Pool 5: HCA_STATE (ratio=128, overlap=0) → R=130
    auto* hca_state = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "hca_state")).get());
    ASSERT_NE(hca_state, nullptr);
    EXPECT_EQ(hca_state->entries_per_block, 130u);
    // Pool 6: SWA_KV (window=128, overlap=0) → R=130, same as HCA_STATE
    auto* swa_kv = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "swa_kv")).get());
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(swa_kv->tag, "swa_kv");
    EXPECT_EQ(swa_kv->entries_per_block, 130u);
}

TEST(HybridPoolConfigCreatorTest, PrefillCp8MtpGenNum2PadsStateRingBeforeSlicing) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = 8;
    pc.prefill_cp_config.kv_cache_sharded = true;

    auto config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 2);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    auto* indexer_state = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "indexer_state")).get());
    auto* csa_state     = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "csa_state")).get());
    auto* hca_state     = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "hca_state")).get());
    auto* swa_kv        = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "swa_kv")).get());
    ASSERT_NE(indexer_state, nullptr);
    ASSERT_NE(csa_state, nullptr);
    ASSERT_NE(hca_state, nullptr);
    ASSERT_NE(swa_kv, nullptr);

    // gen_num_per_cycle=2 gives raw INDEXER/CSA R=10, HCA/SWA R=130.
    // Fixed state pools are CP-sliced by entries; SWA_KV keeps full logical
    // entries and slices its packed bytes instead.
    EXPECT_EQ(indexer_state->entries_per_block, 2u);
    EXPECT_EQ(csa_state->entries_per_block, 2u);
    EXPECT_EQ(hca_state->entries_per_block, 17u);
    EXPECT_EQ(swa_kv->entries_per_block, 136u);
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

    auto prefill_config = CacheConfigCreator::createBasicConfig(mc, prefill_pc, makeDsv4KvCacheConfig(), false, 2);
    auto decode_config  = CacheConfigCreator::createBasicConfig(mc, decode_pc, makeDsv4KvCacheConfig(), false, 2);

    ASSERT_EQ(static_cast<size_t>(prefill_config.groupNums()), 7u);
    ASSERT_EQ(static_cast<size_t>(decode_config.groupNums()), 7u);

    for (const auto& tag : {"indexer_state", "csa_state", "hca_state"}) {
        const auto prefill_gid = gidForTag(prefill_config, tag);
        const auto decode_gid  = gidForTag(decode_config, tag);
        auto* prefill_spec = dynamic_cast<FixedStateCacheSpec*>(prefill_config.specForGroup(prefill_gid).get());
        auto* decode_spec  = dynamic_cast<FixedStateCacheSpec*>(decode_config.specForGroup(decode_gid).get());
        ASSERT_NE(prefill_spec, nullptr) << tag;
        ASSERT_NE(decode_spec, nullptr) << tag;
        EXPECT_EQ(decode_spec->tag, prefill_spec->tag) << tag;
        const auto expected_entries = prefill_spec->entries_per_block * cp_size;
        EXPECT_EQ(decode_spec->entries_per_block, expected_entries) << tag;
    }
    auto* prefill_swa = dynamic_cast<FixedStateCacheSpec*>(
        prefill_config.specForGroup(gidForTag(prefill_config, "swa_kv")).get());
    auto* decode_swa = dynamic_cast<FixedStateCacheSpec*>(
        decode_config.specForGroup(gidForTag(decode_config, "swa_kv")).get());
    ASSERT_NE(prefill_swa, nullptr);
    ASSERT_NE(decode_swa, nullptr);
    EXPECT_EQ(prefill_swa->entries_per_block, 136u);
    EXPECT_EQ(decode_swa->entries_per_block, prefill_swa->entries_per_block);

    auto* indexer_state = dynamic_cast<FixedStateCacheSpec*>(decode_config.specForGroup(gidForTag(decode_config, "indexer_state")).get());
    auto* csa_state     = dynamic_cast<FixedStateCacheSpec*>(decode_config.specForGroup(gidForTag(decode_config, "csa_state")).get());
    auto* hca_state     = dynamic_cast<FixedStateCacheSpec*>(decode_config.specForGroup(gidForTag(decode_config, "hca_state")).get());
    auto* swa_kv        = dynamic_cast<FixedStateCacheSpec*>(decode_config.specForGroup(gidForTag(decode_config, "swa_kv")).get());
    ASSERT_NE(indexer_state, nullptr);
    ASSERT_NE(csa_state, nullptr);
    ASSERT_NE(hca_state, nullptr);
    ASSERT_NE(swa_kv, nullptr);

    EXPECT_EQ(indexer_state->entries_per_block, 16u);
    EXPECT_EQ(csa_state->entries_per_block, 16u);
    EXPECT_EQ(hca_state->entries_per_block, 136u);
    EXPECT_EQ(swa_kv->entries_per_block, 136u);
    for (const auto& tag : {"indexer_state", "csa_state", "hca_state", "swa_kv"}) {
        const auto prefill_gid = gidForTag(prefill_config, tag);
        const auto decode_gid  = gidForTag(decode_config, tag);
        EXPECT_EQ(prefill_config.groupSeqSizePerBlockForGroup(prefill_gid), kDsv4TokensPerBlock * cp_size) << tag;
        EXPECT_EQ(decode_config.groupSeqSizePerBlockForGroup(decode_gid), kDsv4TokensPerBlock * cp_size) << tag;
    }
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

    auto prefill_config = CacheConfigCreator::createBasicConfig(mc, prefill_pc, makeDsv4KvCacheConfig(), false, 2);
    auto decode_config  = CacheConfigCreator::createBasicConfig(mc, decode_pc, makeDsv4KvCacheConfig(), false, 2);

    for (const auto& tag : {"indexer_state", "csa_state", "hca_state"}) {
        const auto prefill_gid = gidForTag(prefill_config, tag);
        const auto decode_gid  = gidForTag(decode_config, tag);
        auto* prefill_spec = dynamic_cast<FixedStateCacheSpec*>(prefill_config.specForGroup(prefill_gid).get());
        auto* decode_spec  = dynamic_cast<FixedStateCacheSpec*>(decode_config.specForGroup(decode_gid).get());
        ASSERT_NE(prefill_spec, nullptr) << tag;
        ASSERT_NE(decode_spec, nullptr) << tag;
        const auto expected_entries = prefill_spec->entries_per_block * cp_size;
        EXPECT_EQ(decode_spec->entries_per_block, expected_entries) << tag;
        EXPECT_EQ(prefill_config.groupSeqSizePerBlockForGroup(prefill_gid), kDsv4TokensPerBlock * cp_size) << tag;
        EXPECT_EQ(decode_config.groupSeqSizePerBlockForGroup(decode_gid), kDsv4TokensPerBlock * cp_size) << tag;
    }
    auto* prefill_swa = dynamic_cast<FixedStateCacheSpec*>(
        prefill_config.specForGroup(gidForTag(prefill_config, "swa_kv")).get());
    auto* decode_swa = dynamic_cast<FixedStateCacheSpec*>(
        decode_config.specForGroup(gidForTag(decode_config, "swa_kv")).get());
    ASSERT_NE(prefill_swa, nullptr);
    ASSERT_NE(decode_swa, nullptr);
    EXPECT_EQ(prefill_swa->entries_per_block, 136u);
    EXPECT_EQ(decode_swa->entries_per_block, prefill_swa->entries_per_block);
    EXPECT_EQ(prefill_config.groupSeqSizePerBlockForGroup(gidForTag(prefill_config, "swa_kv")),
              kDsv4TokensPerBlock * cp_size);
    EXPECT_EQ(decode_config.groupSeqSizePerBlockForGroup(gidForTag(decode_config, "swa_kv")),
              kDsv4TokensPerBlock * cp_size);
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
    auto* csa = dynamic_cast<FixedStateCacheSpec*>(config.specForGroup(gidForTag(config, "csa_state")).get());
    ASSERT_NE(csa, nullptr);
    EXPECT_EQ(csa->entries_per_block, 8u) << "SP_TYPE_NONE should not inflate ring";
}

TEST(HybridPoolConfigCreatorTest, BlockIdConsistencyAcrossGroups) {
    // DSV4 has multiple semantic cache tags per logical layer. The config must expose
    // every tag's group id for the layer so model/runtime code can request the
    // correct group by tag.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

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
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);
    // Set enough blocks for tests.
    RuntimeConfig runtime_config;
    config.finalizeBlockNums(200, runtime_config);
    return config;
}

static CacheConfig makeDSV4CpAllocatorConfig(uint32_t cp_size) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = cp_size;
    pc.prefill_cp_config.kv_cache_sharded = true;
    auto          config = CacheConfigCreator::createBasicConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);
    RuntimeConfig runtime_config;
    config.finalizeBlockNums(200, runtime_config);
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
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // 7 groups → HybridTypeKVCacheAllocator path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(allocator->seqSizePerBlock(), static_cast<int>(config.seq_size_per_block));
    EXPECT_EQ(allocator->totalBlocksNum(), config.pagedBlockNum() - 1);
    EXPECT_EQ(allocator->freeBlocksNum(), config.pagedBlockNum() - 1);
}

TEST_F(DSV4AllocatorTest, CpPageRrFixedAndSwaAllocateOneBlockPerVirtualBlock) {
    constexpr uint32_t cp_size   = 4;
    auto               config    = makeDSV4CpAllocatorConfig(cp_size);
    auto               allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
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
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(allocator->totalBlocksNum(), config.pagedBlockNum() - 1);
}

TEST_F(DSV4AllocatorTest, AddressLookupAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
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

TEST_F(DSV4AllocatorTest, BlockPoolCreatedWithCorrectTensors) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    // allLayerCacheBase should return tensors for all 61 layers
    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined()) << "undefined kv buffer for layer " << i;
    }
}

TEST_F(DSV4AllocatorTest, ConvertIndexToBufferAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
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
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 3u);

    // Direct block pool malloc/free
    auto blocks = block_pool->malloc(3);
    ASSERT_EQ(blocks.size(), 3u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 3);

    block_pool->requestFree(blocks);
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
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST_F(DSV4AllocatorTest, KVBlockStrideIsMaxAcrossGroups) {
    auto config = makeDSV4AllocatorConfig();

    // kv_block_stride_bytes should be the max block_size_bytes across all 7 pools
    size_t expected_max = 0;
    for (int i = 0; i < kDsv4PoolNum; i++) {
        expected_max = std::max(expected_max, config.specForGroup(i)->block_size_bytes());
    }
    EXPECT_EQ(config.maxKvBlockStrideBytes(), expected_max);
    // HCA_STATE has the largest per-block bytes (128 entries * 1024 * 4)
    EXPECT_EQ(expected_max, config.specForGroup(gidForTag(config, "hca_state"))->block_size_bytes());
}

TEST_F(DSV4AllocatorTest, HCAStateIsExcludedFromReuseCachePolicy) {
    auto config = makeDSV4AllocatorConfig();
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    ASSERT_EQ(config.groupPoliciesSnapshot().size(), static_cast<size_t>(config.groupNums()));

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        if (config.tagForGroup(gid) == "hca_state") {
            EXPECT_EQ(config.policyForGroup(gid).reuse_policy, CacheReusePolicy::NON_REUSABLE)
                << "HCA_STATE should skip reuse cache";
        } else {
            EXPECT_EQ(config.policyForGroup(gid).reuse_policy, CacheReusePolicy::REUSABLE) << "group " << gid;
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
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
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
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), 43u);
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined()) << "Flash undefined kv buffer for layer " << i;
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
    EXPECT_EQ(config.specForGroup(gidForTag(config, "swa_kv"))->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST_F(DSV4AllocatorTest, FlashMallocAndFree) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto   block_pool  = allocator->getBlockPool();
    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 5u);

    auto blocks = block_pool->malloc(5);
    ASSERT_EQ(blocks.size(), 5u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 5);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ============================================================
// Prefix cache: insertIntoCache skips HCA_STATE but keeps other groups reusable.
// ============================================================

TEST_F(DSV4AllocatorTest, InsertIntoCacheAllGroups) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    // Manually set up a BatchKVCacheResource with blocks for all 7 groups
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);

    CacheKeysType keys = {200, 201, 202, 203};
    batch_res->setBatchCacheKeys(0, keys);

    // Allocate 3 blocks per group (simulating 3 full blocks)
    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(3);
        ASSERT_EQ(blocks.size(), 3u);
        batch_res->mutableBlockIds(0, gid).assign(BlockIndicesType(blocks.begin(), blocks.end()));
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

    // HCA_STATE is runtime scratch state and must not be persisted as reusable prefix cache.
    for (int gid = 0; gid < 7; gid++) {
        if (config.tagForGroup(gid) == "hca_state") {
            EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(200, gid))) << "HCA_STATE should skip key 200";
            EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(201, gid))) << "HCA_STATE should skip tail key 201";
            EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(202, gid))) << "HCA_STATE should skip tail key 202";
            continue;
        }
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(200, gid))) << config.tagForGroup(gid);
        if (config.typeForGroup(gid) != CacheGroupType::FULL) {
            EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(201, gid))) << config.tagForGroup(gid);
            EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(202, gid))) << config.tagForGroup(gid);
        }
    }

    // Free all blocks
    for (int gid = 0; gid < 7; gid++) {
        const auto& blocks = batch_res->blocks(0, gid);
        block_pool->requestFree(blocks);
    }
}

// ============================================================
// Prefix cache: Flash config insertIntoCache skips HCA_STATE.
// ============================================================

TEST_F(DSV4AllocatorTest, FlashInsertIntoCacheAllGroups) {
    auto config       = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);

    CacheKeysType keys = {300, 301, 302, 303};
    batch_res->setBatchCacheKeys(0, keys);

    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(3);
        ASSERT_EQ(blocks.size(), 3u);
        batch_res->mutableBlockIds(0, gid).assign(BlockIndicesType(blocks.begin(), blocks.end()));
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

    for (int gid = 0; gid < 7; gid++) {
        if (config.tagForGroup(gid) == "hca_state") {
            EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(300, gid))) << "Flash HCA_STATE should skip key 300";
            EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(301, gid))) << "Flash HCA_STATE should skip tail key 301";
            EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(302, gid))) << "Flash HCA_STATE should skip tail key 302";
            continue;
        }
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(300, gid))) << config.tagForGroup(gid);
        if (config.typeForGroup(gid) != CacheGroupType::FULL) {
            EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(301, gid))) << config.tagForGroup(gid);
            EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(302, gid))) << config.tagForGroup(gid);
        }
    }

    for (int gid = 0; gid < 7; gid++) {
        block_pool->requestFree(batch_res->blocks(0, gid));
    }
}

// ============================================================
// Prefix cache: paged FULL groups reuse; reusable SWA/state groups require a matched latest tail block.
// ============================================================

TEST_F(DSV4AllocatorTest, PrefixCacheReusePagedGroupsOnly) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    // Pre-populate cache for ALL 7 groups with keys {100,101,102}
    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {100, 101, 102};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

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
            EXPECT_EQ(out_blocks[0], cached_blocks[gid][0]) << config.tagForGroup(gid);
            EXPECT_EQ(out_blocks[1], cached_blocks[gid][1]) << config.tagForGroup(gid);
            continue;
        }
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1])) << config.tagForGroup(gid);
        if (config.tagForGroup(gid) == "hca_state") {
            EXPECT_TRUE(isNullBlockIdx(out_blocks[2])) << "HCA_STATE should not reuse a cached tail block";
            continue;
        }
        EXPECT_EQ(out_blocks[2], cached_blocks[gid][2]) << config.tagForGroup(gid);
    }

    // Clean up
    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, PrefixCacheReuseRequiresSWATailHit) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {100, 101, 102};
    std::vector<std::vector<BlockIdxType>> cached_blocks(3);
    for (int gid = 0; gid < 3; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

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
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {1100, 1101, 1102};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        if (config.tagForGroup(gid) == "hca_state") {
            continue;
        }
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            if (config.typeForGroup(gid) != CacheGroupType::FULL && i + 1 < cached_keys.size()) {
                continue;
            }
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

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
    EXPECT_EQ(batch_res->blocks(0, swa_gid).at(2), cached_blocks[swa_gid][2]) << "SWA_KV tail should still gate reuse";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, PrefixCacheReuseAcceptsSingleLatestSWATailHit) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int group_num   = 7;
    CacheKeysType cached_keys = {100, 101, 102};
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            if (config.typeForGroup(gid) != CacheGroupType::FULL && i + 1 < cached_keys.size()) {
                continue;
            }
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        block_pool->requestFree(blocks);
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
    auto config       = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {500, 501, 502};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

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
            EXPECT_EQ(out_blocks[0], cached_blocks[gid][0]) << config.tagForGroup(gid);
            continue;
        }
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1])) << config.tagForGroup(gid);
        if (config.tagForGroup(gid) == "hca_state") {
            EXPECT_TRUE(isNullBlockIdx(out_blocks[2])) << "Flash HCA_STATE should not reuse a cached tail block";
            continue;
        }
        EXPECT_EQ(out_blocks[2], cached_blocks[gid][2]) << config.tagForGroup(gid);
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, HybridPoolReserveBlocksAreDistributedAcrossGroups) {
    auto config      = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator   = std::make_shared<HybridPoolKVCacheAllocator>(
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
    auto              kv_config = makeDsv4KvCacheConfig();
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 11);
    auto              config    = CacheConfigCreator::createBasicConfig(mc, pc, kv_config, false, 0);
    RuntimeConfig     runtime_config;
    config.finalizeBlockNums(40, runtime_config);

    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(
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
// SWA (group 6) prefix cache: verify SWA blocks participate in reuse
// ============================================================

TEST_F(DSV4AllocatorTest, SWAGroupParticipatesInPrefixCacheReuse) {
    auto config       = makeDSV4AllocatorConfig();
    RuntimeConfig runtime_config;
    config.finalizeBlockNums(100, runtime_config);
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int group_num = 7;

    // Only populate SWA group (6) and one paged group (0) to verify SWA participates
    CacheKeysType             cached_keys = {700, 701};
    std::vector<BlockIdxType> swa_blocks, csa_blocks;

    // Group 0 (CSA KV)
    {
        auto blocks = block_pool->malloc(2);
        for (size_t i = 0; i < 2; ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[0] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        csa_blocks = blocks;
        block_pool->requestFree(blocks);
    }
    // Group 6 (SWA KV)
    {
        auto blocks = block_pool->malloc(2);
        for (size_t i = 0; i < 2; ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[6] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        swa_blocks = blocks;
        block_pool->requestFree(blocks);
    }

    // Verify both groups have cache entries
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(700, 0)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(700, 6)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(701, 0)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(701, 6)));

    // Groups 1,2,3,4,5 not populated — they will limit reuse to 0
    // But this verifies SWA group 6 IS in the reuse path
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(700, 3)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(700, 4)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(700, 5)));
}

// ============================================================
// SWA prefix cache: cache entries exist and the matched tail window gates reuse.
// ============================================================

TEST_F(DSV4AllocatorTest, SWAPrefixCacheRestoresTailReuse) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    // Populate ALL 7 groups with same keys
    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {800, 801};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(2);
        for (size_t i = 0; i < 2; ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

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

    const auto& swa_out = batch_res->blocks(0, 6);
    ASSERT_GE(swa_out.size(), 2u);
    EXPECT_TRUE(isNullBlockIdx(swa_out[0])) << "SWA previous matched tail is evicted after new tail allocation";
    EXPECT_EQ(swa_out[1], cached_blocks[6][1]) << "SWA last matched tail block should remain";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// incrMalloc: decode grows sequence after initial prefill
// ============================================================

TEST_F(DSV4AllocatorTest, IncrMallocDecodeGrowsBlocks) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
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
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
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
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
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
