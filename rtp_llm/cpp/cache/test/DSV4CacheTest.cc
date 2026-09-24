#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"
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

using TestDSV4HybridTypeAllocator = test::BlockTreeCacheTestAllocator<CoordinatorCacheManager>;
using TestDSV4HybridPoolAllocator = test::BlockTreeCacheTestAllocator<CoordinatorCacheManager>;

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

bool containsReusableGroup(const BlockTreeCache& cache, std::string_view tag) {
    return std::any_of(cache.groupSets().begin(), cache.groupSets().end(), [tag](const GroupSetPtr& group_set) {
        return std::find(group_set->groupTags().begin(), group_set->groupTags().end(), tag)
               != group_set->groupTags().end();
    });
}

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
    ctx.seq_size_per_block      = entries_per_block * compression_ratio;
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
    GroupBase group;
    group.tag                        = spec.tag;
    group.spec                       = spec.clone();
    group.policy                     = defaultCacheGroupPolicy(CacheGroupType::SWA);
    group.policy.enable_prefix_reuse = true;
    group.policy.cp_slice = spec.block_size_bytes() == spec.block_payload_bytes() ? CpBlockSliceMode::PAYLOAD_BYTES :
                                                                                    CpBlockSliceMode::EQUAL_BYTES;
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
    return mapper.sliceBlockForPeer(config, spec.tag, std::move(parts), peer_idx);
}

static std::vector<CacheStoreBlockPair>
buildSwaStorePlan(size_t total_logical_blocks, size_t reuse_block_size, bool use_hybrid, int cp_size) {
    auto         spec   = makeResolvedOpaqueSpec(/*state_cache=*/true, "swa", DataType::TYPE_UINT8, 2, 1);
    auto         config = makeSingleStateCpConfig(*spec, cp_size);
    CPSlotMapper mapper(/*cp_rank=*/0, cp_size, static_cast<int>(config.seq_size_per_block));
    return mapper.buildStorePlan(config, spec->tag, total_logical_blocks, reuse_block_size, use_hybrid);
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

static void setGroupBlockNumsForTest(CacheConfig&                    config,
                                     const std::vector<std::string>& tags,
                                     const std::vector<uint32_t>&    block_nums) {
    std::vector<size_t> kv_strides;
    std::vector<size_t> scale_strides;
    kv_strides.reserve(static_cast<size_t>(config.groupNums()));
    scale_strides.reserve(static_cast<size_t>(config.groupNums()));
    for (const auto& tag : tags) {
        kv_strides.push_back(config.group(tag).kvBlockStrideBytes());
        scale_strides.push_back(config.group(tag).kvScaleStrideBytes());
    }
    config.setGroupBlockLayout(tags, block_nums, kv_strides, scale_strides);
}

static void initDsv4BatchGroups(BatchKVCacheResource& batch_res, const CacheConfig& config) {
    batch_res.initGroups(config.topologyPtr());
}

// DEV's `isDsv4FixedRegion()` covered INDEXER_STATE / CSA_STATE / HCA_STATE / SWA_KV.  On MAIN
// those are exactly the SWA-typed groups of a DSV4 topology, addressed by tag.
static const std::vector<std::string>& dsv4StateSwaTags() {
    static const std::vector<std::string> kTags = {"indexer_state", "csa_state", "hca_state", "swa_kv"};
    return kTags;
}

// DEV expressed "put the fixed pools in pinned host memory" with a single global
// KVCacheConfig::dsv4_fixed_pool_use_memory flag.  MAIN drives residency per pool through the
// spec desc; a host-resident pool must also opt out of the paged HBM budget, otherwise
// checkGroupResidencyBudget() rejects the topology.
static void
setDsv4PoolMemoryPlacement(ModelConfig& model_config, const std::string& tag, CacheMemoryPlacement placement) {
    for (auto& descs : model_config.kv_cache_spec_descs) {
        for (auto& desc : descs) {
            if (desc.tag != tag) {
                continue;
            }
            if (!desc.memory.has_value()) {
                desc.memory = CacheMemoryPolicyDesc{};
            }
            desc.memory->placement = placement;
            if (placement != CacheMemoryPlacement::DEVICE) {
                if (!desc.capacity.has_value()) {
                    desc.capacity = CacheCapacityPolicyDesc{};
                }
                desc.capacity->charge_to_paged_budget = false;
            }
        }
    }
}

static std::vector<int> makeProLayerCompressRatios() {
    std::vector<int> ratios = {128, 128};
    for (int i = 2; i < 61; ++i) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    return ratios;
}

static ModelConfig makeProModelConfig() {
    ModelConfig mc;
    mc.num_layers                                      = 61;
    mc.hidden_size                                     = 7168;
    mc.attn_config.head_num                            = 128;
    mc.attn_config.kv_head_num                         = 1;
    mc.attn_config.size_per_head                       = 512;
    mc.attn_config.rope_head_dim                       = 64;
    mc.attn_config.sliding_window                      = 128;
    mc.attn_config.indexer_head_dim                    = 128;
    mc.attn_config.indexer_head_num                    = 64;
    mc.attn_config.indexer_topk                        = 1024;
    mc.attn_config.o_groups                            = 16;
    mc.attn_config.o_lora_rank                         = 1024;
    mc.attn_config.tokens_per_block                    = kDsv4TokensPerBlock;
    mc.attn_config.layer_compress_ratios               = makeProLayerCompressRatios();
    mc.hybrid_attention_config.enable_hybrid_attention = true;
    setDsv4KvCacheSpecs(mc, mc.attn_config.layer_compress_ratios);
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
    mc.attn_config.layer_compress_ratios               = ratios;
    mc.hybrid_attention_config.enable_hybrid_attention = true;
    setDsv4KvCacheSpecs(mc, mc.attn_config.layer_compress_ratios);
    return mc;
}

static ModelConfig makeFlashMtpModelConfig() {
    ModelConfig mc                       = makeFlashModelConfig();
    mc.num_layers                        = 1;
    mc.attn_config.layer_compress_ratios = {0};
    setDsv4KvCacheSpecs(mc, mc.attn_config.layer_compress_ratios);
    return mc;
}

static ModelConfig makeHybridAttentionModelConfig(int size_per_head = 16) {
    ModelConfig mc;
    mc.num_layers                                      = 4;
    mc.hidden_size                                     = 128;
    mc.attn_config.head_num                            = 4;
    mc.attn_config.kv_head_num                         = 2;
    mc.attn_config.size_per_head                       = size_per_head;
    mc.attn_config.tokens_per_block                    = 8;
    mc.hybrid_attention_config.enable_hybrid_attention = true;
    mc.hybrid_attention_config.hybrid_attention_types  = {
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

TEST(CacheConfigCreatorTest, ProLayerClassification) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeProModelConfig(), pc, 0);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_EQ(publishedGroupTags(config.topology()), kDsv4ProFirstSeenTags);
    EXPECT_EQ(config.layerIdsForGroup("csa_kv").size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup("hca_kv").size(), 31u);
    EXPECT_EQ(config.layerIdsForGroup("swa_kv").size(), 61u);
}

TEST(CacheConfigCreatorTest, FlashLayerClassification) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeFlashModelConfig(), pc, 0);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(publishedGroupTags(config.topology()), kDsv4FlashFirstSeenTags);
    EXPECT_EQ(config.layerIdsForGroup("csa_kv").size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup("hca_kv").size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup("swa_kv").size(), 43u);
}

TEST(CacheConfigCreatorTest, ProAndFlashGroupBytesUseEachGroupsLayerOwnership) {
    for (bool use_flash : {false, true}) {
        ParallelismConfig pc;
        auto              config =
            CacheConfigCreator::createWarmupConfig(use_flash ? makeFlashModelConfig() : makeProModelConfig(), pc, 0);

        size_t expected_total_bytes = 0;
        for (const auto& group : config.groups()) {
            const size_t expected_group_bytes =
                config.layerIdsForGroup(group.tag).size() * (group.kvBlockStrideBytes() + group.kvScaleStrideBytes());
            EXPECT_EQ(config.blockSizeBytesForGroup(group.tag), expected_group_bytes)
                << "use_flash=" << use_flash << " tag=" << group.tag;
            expected_total_bytes += expected_group_bytes;
        }
        EXPECT_EQ(config.totalGroupBlockSizeBytes(), expected_total_bytes) << "use_flash=" << use_flash;
    }
}

TEST(CacheConfigCreatorTest, MtpSwaOnlyLayerIsNotStripped) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeFlashMtpModelConfig(), pc, 0);

    EXPECT_EQ(config.layer_num, 1u);
    EXPECT_GT(config.totalGroupBlockSizeBytes(), 0u);
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 1u);
    ASSERT_EQ(config.layerIdsForGroup("swa_kv"), std::vector<int>({0}));
    ASSERT_EQ(config.topology().layers().size(), 1u);
    EXPECT_EQ(config.topology().layer(0).group_tags, std::vector<std::string>({"swa_kv"}));
    EXPECT_EQ(config.groupTags(), std::vector<std::string>({"swa_kv"}));
    EXPECT_EQ(config.groupForLayer(0, "swa_kv").tag, "swa_kv");
}

TEST(CacheConfigCreatorTest, Dsv4SpecOrderControlsFirstSeenGroupOrder) {
    auto mc = makeFlashModelConfig();
    for (auto& layer_descs : mc.kv_cache_spec_descs) {
        std::reverse(layer_descs.begin(), layer_descs.end());
    }

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    const std::vector<std::string> expected_tags = {
        "swa_kv", "csa_state", "indexer_state", "indexer_kv", "csa_kv", "hca_state", "hca_kv"};
    EXPECT_EQ(publishedGroupTags(config.topology()), expected_tags);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), expected_tags.size());
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), expected_tags.size());
    for (const auto& tag : expected_tags) {
        ASSERT_NE(config.group(tag).spec, nullptr);
        EXPECT_EQ(config.group(tag).spec->tag, tag) << "tag=" << tag;
    }

    EXPECT_EQ(config.groupForLayer(2, "csa_kv").tag, "csa_kv");
    EXPECT_EQ(config.groupForLayer(3, "hca_kv").tag, "hca_kv");
    EXPECT_EQ(config.groupForLayer(0, "swa_kv").tag, "swa_kv");
}

TEST(CacheConfigCreatorTest, SparseIndexerUsesIndependentNaturalStridePool) {
    ModelConfig model_config;
    model_config.num_layers                          = 2;
    model_config.attn_config.use_mla                 = true;
    model_config.attn_config.kv_lora_rank            = 512;
    model_config.attn_config.rope_head_dim           = 64;
    model_config.attn_config.tokens_per_block        = 512;
    model_config.attn_config.kernel_tokens_per_block = 128;

    KVCacheSpecDesc default_desc;
    default_desc.tag        = "default";
    default_desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;

    KVCacheSpecDesc indexer_desc;
    indexer_desc.tag               = "indexer_kv";
    indexer_desc.cache_type        = KVCacheSpecType::OpaqueKV;
    indexer_desc.entry_dtype       = DataType::TYPE_UINT8;
    indexer_desc.entry_elems       = 132;
    indexer_desc.entry_count_mode  = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    indexer_desc.compression_ratio = 1;
    model_config.kv_cache_spec_descs.assign(2, {default_desc, indexer_desc});

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.kernel_seq_size_per_block = 128;
    kv_cache_config.test_block_num            = 4;

    auto           created_config = CacheConfigCreator::createConfig(model_config, parallelism_config, kv_cache_config);
    const uint32_t config_candidate_block_num = CacheConfigCreator::computeLocalBlockNum(
        created_config, model_config, runtime_config, kv_cache_config, parallelism_config);
    const auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

    ASSERT_GT(config.groupNums(), 1);
    for (const auto& group : config.topology().groups()) {
        const auto pool_config = DeviceBlockPoolConfigHelper::createConfigForGroup(config, group);
        ASSERT_FALSE(pool_config.memory_layouts.empty());
        for (const auto& layout : pool_config.memory_layouts) {
            EXPECT_FALSE(layout.enable_hybrid_attention);
        }
    }
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_EQ(publishedGroupTags(config.topology()), (std::vector<std::string>{"default", "indexer_kv"}));

    EXPECT_EQ(config.layerIdsForGroup("default"), (std::vector<int>{0, 1}));
    EXPECT_EQ(config.layerIdsForGroup("indexer_kv"), (std::vector<int>{0, 1}));
    EXPECT_EQ(config.group("default").kernelSeqSizePerBlock(), 128u);
    EXPECT_EQ(config.group("indexer_kv").kernelSeqSizePerBlock(), 128u);
    EXPECT_EQ(config.group("default").kvScaleStrideBytes(), 0u);
    EXPECT_TRUE(config.group("default").policy.enable_prefix_reuse);
    EXPECT_TRUE(config.group("indexer_kv").policy.enable_prefix_reuse);
    EXPECT_EQ(config.group("indexer_kv").kvBlockStrideBytes(), 512u * 132u);
    EXPECT_EQ(config.group("indexer_kv").kvScaleStrideBytes(), 0u);
    EXPECT_EQ(config.blockSizeBytesForGroup("indexer_kv"), 2u * 512u * 132u);
}

static GroupBase makeTestGroup(const KVCacheSpecPtr& spec, CacheGroupType type, std::vector<int> layer_ids) {
    (void)layer_ids;
    GroupBase group;
    group.tag    = spec->tag;
    group.spec   = spec;
    group.policy = defaultCacheGroupPolicy(type);
    return group;
}

TEST(CacheConfigTest, GroupIdentityQueriesPreserveGeometryAcrossTopologyOrders) {
    auto                         full_spec = test::makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 4, 16, "full");
    auto                         swa_spec  = test::makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 8, 8, "swa");
    auto                         full      = makeTestGroup(full_spec, CacheGroupType::FULL, {1, 2});
    auto                         swa       = makeTestGroup(swa_spec, CacheGroupType::SWA, {0, 1});
    const std::vector<LayerBase> layers{{0, {"swa"}}, {1, {"full", "swa"}}, {2, {"full"}}};

    for (const auto& groups : {std::vector<GroupBase>{full, swa}, std::vector<GroupBase>{swa, full}}) {
        CacheConfig config;
        config.layer_num = 3;
        config.setTopology(groups, layers);
        EXPECT_EQ(config.layerIdsForGroup("full"), (std::vector<int>{1, 2}));
        EXPECT_EQ(config.layerIdsForGroup("swa"), (std::vector<int>{0, 1}));
        EXPECT_EQ(config.blockSizeBytesForGroup("full"), 2u * (full.kvBlockStrideBytes() + full.kvScaleStrideBytes()));
        EXPECT_EQ(config.blockSizeBytesForGroup("swa"), 2u * (swa.kvBlockStrideBytes() + swa.kvScaleStrideBytes()));
        EXPECT_ANY_THROW(config.layerIdsForGroup("missing"));
        EXPECT_ANY_THROW(config.blockSizeBytesForGroup("missing"));
    }
}

TEST(CacheConfigTest, SetTopologyInstallsTagAndGroupTopology) {
    CacheConfig config;
    config.layer_num = 3;

    auto swa_spec =
        std::dynamic_pointer_cast<FixedStateCacheSpec>(makeResolvedOpaqueSpec(true, "swa", DataType::TYPE_UINT8, 2, 1));
    auto csa_spec = std::dynamic_pointer_cast<CompressedKVCacheSpec>(
        makeResolvedOpaqueSpec(false, "csa", DataType::TYPE_UINT8, 2, 1));

    std::vector<LayerBase> layers = {{0, {"swa"}}, {1, {"swa", "csa"}}, {2, {"swa"}}};

    config.setTopology(
        {makeTestGroup(swa_spec, CacheGroupType::SWA, {0, 1, 2}), makeTestGroup(csa_spec, CacheGroupType::FULL, {1})},
        std::move(layers));

    EXPECT_EQ(publishedGroupTags(config.topology()), std::vector<std::string>({"swa", "csa"}));
    EXPECT_EQ(config.groupForLayer(1, "swa").tag, "swa");
    EXPECT_EQ(config.groupForLayer(1, "csa").tag, "csa");
    EXPECT_THROW((void)config.soleGroupForLayer(1), std::exception);
    EXPECT_EQ(config.topology().layer(1).group_tags, std::vector<std::string>({"swa", "csa"}));
}

TEST(CacheConfigTest, TopologyRemainsTheSingleSourceAcrossSupportedUpdates) {
    CacheConfig config;
    config.layer_num = 2;

    auto full_spec   = std::make_shared<MHAKVCacheSpec>();
    full_spec->tag   = "full";
    auto linear_spec = std::make_shared<LinearKVCacheSpec>();
    linear_spec->tag = "linear";

    config.setTopology(
        {makeTestGroup(full_spec, CacheGroupType::FULL, {0}), makeTestGroup(linear_spec, CacheGroupType::LINEAR, {1})},
        {{0, {"full"}}, {1, {"linear"}}});
    const auto initial_topology = config.topologyPtr();

    std::vector<CacheGroupPolicy> policies;
    for (const auto& group : config.topology().groups()) {
        policies.push_back(group.policy);
    }
    policies[0].enable_prefix_reuse = !policies[0].enable_prefix_reuse;
    test::setTestGroupPolicies(config, policies);
    const auto policy_topology = config.topologyPtr();

    EXPECT_NE(policy_topology.get(), initial_topology.get());
    EXPECT_EQ(config.group("full").policy.enable_prefix_reuse, policies[0].enable_prefix_reuse);
    EXPECT_NE(initial_topology->group("full").policy.enable_prefix_reuse, policies[0].enable_prefix_reuse);

    // The fixture receives rows independently ordered from the target topology.
    EXPECT_ANY_THROW(test::setGroupBlockLayout(config, {"linear", "missing"}, {9, 17}, {256, 128}, {8, 4}));
    EXPECT_EQ(config.topologyPtr(), policy_topology);
    EXPECT_ANY_THROW(test::setGroupBlockLayout(config, {"linear", "linear"}, {9, 17}, {256, 128}, {8, 4}));
    EXPECT_EQ(config.topologyPtr(), policy_topology);
    test::setGroupBlockLayout(config, {"linear", "full"}, {9, 17}, {256, 128}, {8, 4});
    const auto layout_topology = config.topologyPtr();

    EXPECT_NE(layout_topology.get(), policy_topology.get());
    EXPECT_EQ(config.group("full").block_num, 17u);
    EXPECT_EQ(config.group("linear").block_num, 9u);
    EXPECT_EQ(config.group("full").kvBlockStrideBytes(), 128u);
    EXPECT_EQ(config.group("full").kvScaleStrideBytes(), 4u);
    EXPECT_EQ(config.group("linear").kvScaleStrideBytes(), 8u);
    EXPECT_EQ(config.group("linear").block_num, 9u);
    EXPECT_EQ(config.group("linear").kvBlockStrideBytes(), 256u);
    EXPECT_EQ(policy_topology->group("linear").block_num, 0u);

    config.finalizeBlockNums(/*global_block_num=*/23, RuntimeConfig{});
    const auto finalized_topology = config.topologyPtr();

    EXPECT_NE(finalized_topology.get(), layout_topology.get());
    EXPECT_EQ(config.group("full").block_num, 23u);
    EXPECT_EQ(config.group("linear").block_num, 23u);
    EXPECT_EQ(config.topology().groups()[0].block_num, config.group("full").block_num);
    EXPECT_EQ(config.topology().groups()[1].block_num, config.group("linear").block_num);
    EXPECT_EQ(layout_topology->group("full").block_num, 17u);
    EXPECT_EQ(layout_topology->group("linear").block_num, 9u);
}

TEST(CacheConfigTest, SetTopologyRejectsMissingLayer) {
    CacheConfig config;
    config.layer_num = 2;

    auto spec                     = std::make_shared<MHAKVCacheSpec>();
    spec->tag                     = "default";
    std::vector<LayerBase> layers = {{0, {"default"}}, {1, {}}};
    EXPECT_THROW(config.setTopology({makeTestGroup(spec, CacheGroupType::FULL, {0})}, std::move(layers)),
                 std::exception);
}

TEST(CacheConfigTest, SetTopologyRejectsEmptyTag) {
    CacheConfig config;
    config.layer_num = 1;

    auto                   spec   = std::make_shared<MHAKVCacheSpec>();
    std::vector<LayerBase> layers = {{0, {""}}};
    EXPECT_THROW(config.setTopology({makeTestGroup(spec, CacheGroupType::FULL, {0})}, std::move(layers)),
                 std::exception);
}

TEST(CacheConfigTest, SetTopologyRejectsDuplicateGroupTag) {
    CacheConfig config;
    config.layer_num = 1;

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
    config.layer_num = 1;

    auto spec0 = std::make_shared<MHAKVCacheSpec>();
    spec0->tag = "full";
    auto spec1 = std::make_shared<MHAKVCacheSpec>();
    spec1->tag = "linear";

    std::vector<LayerBase> layers = {{0, {"full", "linear"}}};
    EXPECT_NO_THROW(config.setTopology(
        {makeTestGroup(spec0, CacheGroupType::FULL, {0}), makeTestGroup(spec1, CacheGroupType::LINEAR, {0})},
        std::move(layers)));
    EXPECT_EQ(config.topology().layer(0).group_tags.size(), 2u);
}

TEST(CacheConfigCreatorTest, Dsv4ModelProvidedAlignmentPropagatesToCacheSpecs) {
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
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    const auto* csa_kv = dynamic_cast<const CompressedKVCacheSpec*>(config.group("csa_kv").spec.get());
    const auto* swa_kv = dynamic_cast<const FixedStateCacheSpec*>(config.group("swa_kv").spec.get());
    ASSERT_NE(csa_kv, nullptr);
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(csa_kv->block_size_bytes() % 1024u, 0u);
    EXPECT_EQ(swa_kv->block_size_bytes() % 2048u, 0u);
}

TEST(CacheConfigCreatorTest, Dsv4TagRoutesAreConsistent) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeFlashModelConfig(), pc, 0);

    auto expect_route = [&](int layer_id, const std::string& tag) {
        EXPECT_EQ(&config.groupForLayer(layer_id, tag), &config.group(tag)) << "layer=" << layer_id << " tag=" << tag;
    };

    // Flash DSV4 test config uses layers 2,4,... as CSA and 3,5,... as HCA; 0/1 are SWA-only.
    expect_route(2, "csa_kv");
    expect_route(2, "indexer_kv");
    expect_route(2, "indexer_state");
    expect_route(2, "csa_state");
    expect_route(2, "swa_kv");

    expect_route(3, "hca_kv");
    expect_route(3, "hca_state");
    expect_route(3, "swa_kv");

    expect_route(0, "swa_kv");
    EXPECT_THROW(config.groupForLayer(0, "csa_kv"), std::exception);
    EXPECT_THROW(config.groupForLayer(0, "hca_kv"), std::exception);

    auto mtp_config = CacheConfigCreator::createWarmupConfig(makeFlashMtpModelConfig(), pc, 0);
    ASSERT_EQ(mtp_config.groupForLayer(0, "swa_kv").tag, "swa_kv");
}

TEST(CacheConfigCreatorTest, Dsv4GroupPoliciesMatchLegacyBehavior) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeFlashModelConfig(), pc, 0);

    ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(config.groupNums()));
    auto expect_policy = [&](const std::string& tag, bool enable_prefix_reuse, int active_tail_blocks) {
        const auto& policy = config.group(tag).policy;
        EXPECT_EQ(policy.enable_prefix_reuse, enable_prefix_reuse) << tag;
        EXPECT_EQ(policy.active_tail_blocks, active_tail_blocks) << tag;
    };

    expect_policy("hca_state", false, 1);
    expect_policy("swa_kv", true, 2);
    expect_policy("csa_state", true, 2);
    expect_policy("csa_kv", true, 0);
    expect_policy("hca_kv", true, 0);
    expect_policy("indexer_kv", true, 0);
}

TEST(CacheConfigCreatorTest, SlidingWindowPolicyPropagatesAndSurvivesAggregation) {
    ParallelismConfig             pc;
    auto                          config = CacheConfigCreator::createWarmupConfig(makeFlashModelConfig(), pc, 0);
    std::vector<CacheGroupPolicy> policies;
    for (const auto& group : config.topology().groups()) {
        policies.push_back(group.policy);
    }

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

    test::setTestGroupPolicies(config, policies);
    std::vector<CacheGroupPolicy> aggregated;
    for (const auto& group : config.topology().groups()) {
        aggregated.push_back(group.policy);
    }
    ASSERT_EQ(aggregated.size(), policies.size());
    for (size_t gid = 0; gid < policies.size(); ++gid) {
        EXPECT_TRUE(CacheConfig::samePolicy(aggregated[gid], policies[gid])) << "gid=" << gid;
        EXPECT_EQ(aggregated[gid].sliding_window_size, policies[gid].sliding_window_size) << "gid=" << gid;
    }
}

TEST(CacheConfigCreatorTest, Dsv4SpecsMissingFailsFastWithoutRatioFallback) {
    auto mc = makeFlashModelConfig();
    mc.kv_cache_spec_descs.clear();

    ParallelismConfig pc;
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);
}

// ============================================================
// Pool specs
// ============================================================

TEST(CacheConfigCreatorTest, ProPoolSpecs) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeProModelConfig(), pc, 0);

    EXPECT_EQ(config.layerIdsForGroup("csa_kv").size(), 30u);
    EXPECT_EQ(config.group("csa_kv").spec->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.group("csa_kv").policy.group_type, CacheGroupType::FULL);

    EXPECT_EQ(config.layerIdsForGroup("hca_kv").size(), 31u);
    EXPECT_EQ(config.group("hca_kv").spec->block_size_bytes(), 1u * kDsv4KvEntryBytes);

    EXPECT_EQ(config.layerIdsForGroup("indexer_kv").size(), 30u);
    EXPECT_EQ(config.group("indexer_kv").spec->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);

    EXPECT_EQ(config.layerIdsForGroup("indexer_state").size(), 30u);
    EXPECT_EQ(config.group("indexer_state").spec->block_size_bytes(), 8u * 512u * 4u);

    EXPECT_EQ(config.layerIdsForGroup("csa_state").size(), 30u);
    EXPECT_EQ(config.group("csa_state").spec->block_size_bytes(), 8u * 2048u * 4u);

    EXPECT_EQ(config.layerIdsForGroup("hca_state").size(), 31u);
    EXPECT_EQ(config.group("hca_state").spec->block_size_bytes(), 128u * 1024u * 4u);

    EXPECT_EQ(config.layerIdsForGroup("swa_kv").size(), 61u);
    EXPECT_EQ(config.group("swa_kv").spec->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(CacheConfigCreatorTest, FlashPoolSpecs) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeFlashModelConfig(), pc, 0);
    EXPECT_EQ(config.layerIdsForGroup("csa_kv").size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup("hca_kv").size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup("swa_kv").size(), 43u);
}

// ============================================================
// Block size bytes
// ============================================================

TEST(CacheConfigCreatorTest, BlockSizeBytes) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeProModelConfig(), pc, 0);
    EXPECT_EQ(config.group("csa_kv").spec->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.group("hca_kv").spec->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.group("indexer_kv").spec->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.group("indexer_state").spec->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(config.group("csa_state").spec->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(config.group("hca_state").spec->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(config.group("swa_kv").spec->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(CacheConfigCreatorTest, Fp8BlockSizeBytesUsePaddedPhysicalStride) {
    ParallelismConfig pc;
    auto              mc          = makeProModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    setDsv4KvCacheSpecs(mc, makeProLayerCompressRatios());
    auto config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);

    EXPECT_EQ(config.group("csa_kv").spec->block_size_bytes(), 19008u);
    EXPECT_EQ(config.group("hca_kv").spec->block_size_bytes(), 1152u);
    EXPECT_EQ(config.group("indexer_kv").spec->block_size_bytes(), 32u * 132u);
    EXPECT_EQ(config.group("swa_kv").spec->block_size_bytes(), 74880u);

    EXPECT_EQ(config.group("csa_kv").kvBlockStrideBytes(), config.group("csa_kv").spec->block_size_bytes());
    EXPECT_EQ(config.group("hca_kv").kvBlockStrideBytes(), config.group("hca_kv").spec->block_size_bytes());
    EXPECT_EQ(config.group("swa_kv").kvBlockStrideBytes(), config.group("swa_kv").spec->block_size_bytes());
}

TEST(CacheConfigCreatorTest, BasicConfigUsesModelDefaultPhysicalAndKernelBlockSize) {
    ParallelismConfig pc;
    auto              mc     = makeProModelConfig();
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);

    EXPECT_EQ(config.seq_size_per_block, kDsv4TokensPerBlock);
    EXPECT_EQ(config.group("csa_kv").kernelSeqSizePerBlock(), 128u);
    EXPECT_EQ(config.group("csa_kv").kernelBlocksPerKvBlock(), 1u);

    const auto* csa_kv = dynamic_cast<const CompressedKVCacheSpec*>(config.group("csa_kv").spec.get());
    const auto* hca_kv = dynamic_cast<const CompressedKVCacheSpec*>(config.group("hca_kv").spec.get());
    const auto* idx_kv = dynamic_cast<const CompressedKVCacheSpec*>(config.group("indexer_kv").spec.get());
    const auto* swa_kv = dynamic_cast<const FixedStateCacheSpec*>(config.group("swa_kv").spec.get());
    ASSERT_NE(csa_kv, nullptr);
    ASSERT_NE(hca_kv, nullptr);
    ASSERT_NE(idx_kv, nullptr);
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(csa_kv->block_size() / kDsv4KvEntryBytes, 32u);
    EXPECT_EQ(hca_kv->block_size() / DSV4_FP8_KV_ENTRY_BYTES, 1u);
    EXPECT_EQ(idx_kv->block_size() / kDsv4IndexerEntryBytes, 32u);
    EXPECT_EQ(opaqueEntriesPerBlock(*swa_kv, kDsv4KvEntryBytes), 128u);

    EXPECT_EQ(config.group("csa_kv").kvBlockStrideBytes(), config.group("csa_kv").spec->block_size_bytes());
    EXPECT_EQ(config.group("hca_kv").kvBlockStrideBytes(), config.group("hca_kv").spec->block_size_bytes());
    EXPECT_EQ(config.group("indexer_kv").kvBlockStrideBytes(), config.group("indexer_kv").spec->block_size_bytes());
    EXPECT_EQ(config.group("swa_kv").kvBlockStrideBytes(), config.group("swa_kv").spec->block_size_bytes());

    auto full_pool = DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.group("csa_kv"));
    auto swa_pool  = DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.group("swa_kv"));
    ASSERT_EQ(full_pool.memory_layouts.size(), 1u);
    ASSERT_EQ(swa_pool.memory_layouts.size(), 1u);
    EXPECT_EQ(full_pool.memory_layouts[0].kernel_blocks_per_kv_block, 1u);
    EXPECT_EQ(swa_pool.memory_layouts[0].kernel_blocks_per_kv_block, 1u);
}

// Ported from DEV CacheConfigCreatorTest.DecoupledPhysicalAndKernelBlockSizeUsesPerGroupBpk.
// The test above pins the coupled default (physical == kernel == 128, bpk 1); this one pins the
// decoupled case, where compressed pools stride by bpk kernel blocks while STATE_RING pools keep
// bpk 1 because their kernel block equals their physical block.
TEST(CacheConfigCreatorTest, DecoupledPhysicalAndKernelBlockSizeUsesPerGroupBpk) {
    ParallelismConfig pc;
    auto              mc                   = makeProModelConfig();
    mc.attn_config.tokens_per_block        = 16384;
    mc.attn_config.kernel_tokens_per_block = 128;
    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 16384;
    kv_cache_config.kernel_seq_size_per_block = 128;
    auto config                               = CacheConfigCreator::createWarmupConfig(mc, pc, kv_cache_config, 0);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), static_cast<size_t>(kDsv4PoolNum));
    EXPECT_EQ(config.seq_size_per_block, 16384u);
    EXPECT_EQ(config.group("csa_kv").kernelSeqSizePerBlock(), 128u);
    EXPECT_EQ(config.group("csa_kv").kernelBlocksPerKvBlock(), 128u);
    for (const auto& group : config.groups()) {
        EXPECT_EQ(group.seqSizePerBlock(), 16384u) << "tag=" << group.tag;
    }

    const auto* csa_kv = dynamic_cast<const CompressedKVCacheSpec*>(config.group("csa_kv").spec.get());
    const auto* hca_kv = dynamic_cast<const CompressedKVCacheSpec*>(config.group("hca_kv").spec.get());
    const auto* idx_kv = dynamic_cast<const CompressedKVCacheSpec*>(config.group("indexer_kv").spec.get());
    const auto* swa_kv = dynamic_cast<const FixedStateCacheSpec*>(config.group("swa_kv").spec.get());
    ASSERT_NE(csa_kv, nullptr);
    ASSERT_NE(hca_kv, nullptr);
    ASSERT_NE(idx_kv, nullptr);
    ASSERT_NE(swa_kv, nullptr);
    // Compressed specs aggregate every kernel page in a physical block; state rings stay fixed-size.
    EXPECT_EQ(opaqueEntriesPerBlock(*csa_kv, kDsv4KvEntryBytes), 32u * 128u);
    EXPECT_EQ(opaqueEntriesPerBlock(*hca_kv, kDsv4KvEntryBytes), 128u);
    EXPECT_EQ(opaqueEntriesPerBlock(*idx_kv, kDsv4IndexerEntryBytes), 32u * 128u);
    EXPECT_EQ(opaqueEntriesPerBlock(*swa_kv, kDsv4KvEntryBytes), 128u);

    EXPECT_EQ(config.group("csa_kv").kernelBlocksPerKvBlock(), 128u);
    EXPECT_EQ(config.group("swa_kv").kernelBlocksPerKvBlock(), 1u);
    EXPECT_EQ(config.group("csa_kv").kvBlockStrideBytes(), csa_kv->block_size_bytes());
    EXPECT_EQ(config.group("hca_kv").kvBlockStrideBytes(), hca_kv->block_size_bytes());
    EXPECT_EQ(config.group("indexer_kv").kvBlockStrideBytes(), idx_kv->block_size_bytes());
    EXPECT_EQ(config.group("swa_kv").kvBlockStrideBytes(), swa_kv->block_size_bytes());

    auto full_pool_bpk = DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.group("csa_kv"));
    auto swa_pool_bpk  = DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.group("swa_kv"));
    ASSERT_EQ(full_pool_bpk.memory_layouts.size(), 1u);
    ASSERT_EQ(swa_pool_bpk.memory_layouts.size(), 1u);
    EXPECT_EQ(full_pool_bpk.memory_layouts[0].kernel_blocks_per_kv_block, 128u);
    EXPECT_EQ(swa_pool_bpk.memory_layouts[0].kernel_blocks_per_kv_block, 1u);
}

TEST(CacheConfigCreatorTest, PrefillCpShardedSlicesFixedAndSwaPhysicalBlocks) {
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = 4;
    pc.prefill_cp_config.kv_cache_sharded = true;

    auto mc                       = makeProModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    setDsv4KvCacheSpecs(mc, makeProLayerCompressRatios());
    auto config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);

    EXPECT_EQ(config.group("csa_kv").spec->block_size_bytes(), 19008u);
    EXPECT_EQ(config.group("hca_kv").spec->block_size_bytes(), 1152u);
    EXPECT_EQ(config.group("indexer_kv").spec->block_size_bytes(), 32u * 132u);
    EXPECT_EQ(config.group("indexer_state").spec->block_size_bytes(), 2u * 512u * 4u);
    EXPECT_EQ(config.group("csa_state").spec->block_size_bytes(), 2u * 2048u * 4u);
    EXPECT_EQ(config.group("hca_state").spec->block_size_bytes(), 32u * 1024u * 4u);

    // SWA_KV keeps full logical ring entries for byte-sliced CP layout, but
    // each prefill rank stores only one aligned byte slice of the full block.
    EXPECT_EQ(config.group("swa_kv").spec->block_size_bytes(), 18720u);
    for (const auto& tag : {"indexer_state", "csa_state", "hca_state", "swa_kv"}) {

        EXPECT_EQ(config.group(tag).kvBlockStrideBytes(), config.group(tag).spec->block_size_bytes());
    }

    pc.role_type       = RoleType::DECODE;
    auto decode_config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);
    EXPECT_EQ(decode_config.group("indexer_state").spec->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(decode_config.group("csa_state").spec->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(decode_config.group("hca_state").spec->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(decode_config.group("swa_kv").spec->block_size_bytes(), 74880u);
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

TEST(CacheConfigCreatorTest, CreateCacheConfig) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    // 7 groups -> groupNums() > 1 -> CoordinatorCacheManager path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_TRUE(config.is_sparse);
    EXPECT_FALSE(config.use_mla);
}

TEST(CacheConfigCreatorTest, FlashCacheConfig) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(config.layerIdsForGroup("swa_kv").size(), 43u);
    EXPECT_EQ(config.layerIdsForGroup("csa_kv").size(), 21u);
}

TEST(CacheConfigCreatorTest, HybridAttentionIndependentPoolUsesHybridPoolConfig) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeHybridAttentionModelConfig(), pc, 0);

    EXPECT_GT(config.groupNums(), 1);
    for (const auto& group : config.topology().groups()) {
        const auto pool_config = DeviceBlockPoolConfigHelper::createConfigForGroup(config, group);
        ASSERT_FALSE(pool_config.memory_layouts.empty());
        for (const auto& layout : pool_config.memory_layouts) {
            EXPECT_FALSE(layout.enable_hybrid_attention);
        }
    }
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_EQ(config.group("full").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group("linear").policy.group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 2u);
    EXPECT_GT(config.group("full").spec->block_size_bytes(), 0u);
    EXPECT_GT(config.group("linear").spec->block_size_bytes(), 0u);
    EXPECT_NE(config.group("full").spec->block_size_bytes(), config.group("linear").spec->block_size_bytes());
    EXPECT_EQ(config.topology().groups().size(), 2u);
    EXPECT_EQ(publishedGroupTags(config.topology()), std::vector<std::string>({"linear", "full"}));

    EXPECT_EQ(config.totalGroupBlockSizeBytes(),
              config.blockSizeBytesForGroup("linear") + config.blockSizeBytesForGroup("full"));

    RuntimeConfig runtime_config;
    config.linear_step = 4;
    config.finalizeBlockNums(/*global_block_num=*/37, runtime_config);
    EXPECT_EQ(config.group("linear").block_num, 37u);
    EXPECT_EQ(config.group("full").block_num, 37u);
}

TEST(CacheConfigCreatorTest, HybridAttentionIndependentPoolSplitsFullAndSwaSpecs) {
    auto mc                                           = makeHybridAttentionModelConfig();
    mc.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE,
                                                         HybridAttentionType::SLIDING_WINDOW,
                                                         HybridAttentionType::LINEAR,
                                                         HybridAttentionType::SLIDING_WINDOW};
    setHybridAttentionKvCacheSpecs(mc);

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    ASSERT_EQ(config.groupNums(), 3);
    EXPECT_EQ(config.group("full").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group("swa").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("linear").policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(publishedGroupTags(config.topology()), std::vector<std::string>({"full", "swa", "linear"}));
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 3u);
    EXPECT_NE(config.group("full").spec.get(), config.group("swa").spec.get());
    EXPECT_EQ(config.layerIdsForGroup("full"), std::vector<int>({0}));
    EXPECT_EQ(config.layerIdsForGroup("swa"), std::vector<int>({1, 3}));
    EXPECT_EQ(config.layerIdsForGroup("linear"), std::vector<int>({2}));
    EXPECT_EQ(config.layerIdsForGroup("full").size(), 1u);
    EXPECT_EQ(config.layerIdsForGroup("swa").size(), 2u);
    EXPECT_EQ(config.layerIdsForGroup("linear").size(), 1u);
    EXPECT_EQ(config.groupForLayer(1, "swa").tag, "swa");
    EXPECT_EQ(config.groupForLayer(2, "linear").tag, "linear");

    EXPECT_EQ(config.totalGroupBlockSizeBytes(),
              config.blockSizeBytesForGroup("full") + config.blockSizeBytesForGroup("swa")
                  + config.blockSizeBytesForGroup("linear"));

    RuntimeConfig runtime_config;
    config.linear_step = 3;
    config.finalizeBlockNums(/*global_block_num=*/10, runtime_config);
    EXPECT_EQ(config.group("full").block_num, 10u);
    EXPECT_EQ(config.group("linear").block_num, 10u);
    EXPECT_EQ(config.group("swa").block_num, 4u);
}

TEST(CacheConfigCreatorTest, HybridAttentionIndependentPoolBackingFitsBudgetExactly) {
    auto mc                                           = makeHybridAttentionModelConfig();
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

    auto           created_config = CacheConfigCreator::createConfig(mc, pc, kv_cache_config);
    const uint32_t config_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_config, mc, runtime_config, kv_cache_config, pc);
    auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

    size_t paged_bytes = 0;
    size_t swa_bytes   = 0;
    for (const auto& group : config.groups()) {
        if (group.policy.group_type == CacheGroupType::SWA) {
            swa_bytes += config.blockSizeBytesForGroup(group.tag);
            EXPECT_EQ(group.block_num, (static_cast<uint32_t>(config_candidate_block_num) + 3u) / 4u);
        } else {
            paged_bytes += config.blockSizeBytesForGroup(group.tag);
            EXPECT_EQ(group.block_num, static_cast<uint32_t>(config_candidate_block_num));
        }
    }

    const auto backing_bytes = [&](uint32_t block_num) {
        return static_cast<size_t>(block_num) * paged_bytes + static_cast<size_t>((block_num + 3u) / 4u) * swa_bytes;
    };
    constexpr size_t budget_bytes = 1024u * 1024u;
    const auto       block_num    = static_cast<uint32_t>(config_candidate_block_num);
    EXPECT_LE(backing_bytes(block_num), budget_bytes);
    EXPECT_GT(backing_bytes(block_num + 1u), budget_bytes);
}

TEST(CacheConfigCreatorTest, LinearValueHeadsMustDivideAttentionTp) {
    auto mc                                           = makeHybridAttentionModelConfig();
    mc.linear_attention_config.linear_num_value_heads = 3;

    ParallelismConfig pc;
    pc.tp_size = 2;

    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);

    mc.linear_attention_config.linear_num_value_heads = 4;
    EXPECT_NO_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0));
}

TEST(CacheConfigCreatorTest, HybridAttentionUsesIndependentPerGroupBlockCounts) {
    ParallelismConfig pc;
    auto config = CacheConfigCreator::createWarmupConfig(makeHybridAttentionModelConfig(/*size_per_head=*/32), pc, 0);

    EXPECT_GT(config.groupNums(), 1);
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_EQ(config.group("linear").block_num, 2u);
    EXPECT_EQ(config.group("full").block_num, 2u);
    EXPECT_EQ(publishedGroupTags(config.topology()), std::vector<std::string>({"linear", "full"}));
    EXPECT_EQ(config.groupForLayer(0, "linear").tag, "linear");
    EXPECT_EQ(config.groupForLayer(1, "full").tag, "full");
}

TEST(CacheConfigCreatorTest, HybridAttentionTypesMustCoverAllLayers) {
    auto mc = makeHybridAttentionModelConfig(/*size_per_head=*/32);
    mc.hybrid_attention_config.hybrid_attention_types.pop_back();

    ParallelismConfig pc;
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);
}

TEST(CacheConfigCreatorTest, CanonicalLinearTagRejectsIncompatibleGeometry) {
    auto mc = makeHybridAttentionModelConfig();
    ASSERT_EQ(mc.kv_cache_spec_descs[0][0].tag, "linear");
    ASSERT_EQ(mc.kv_cache_spec_descs[2][0].tag, "linear");
    mc.kv_cache_spec_descs[0][0].dtype = DataType::TYPE_FP16;
    mc.kv_cache_spec_descs[2][0].dtype = DataType::TYPE_BF16;

    ParallelismConfig pc;
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);
}

TEST(CacheConfigCreatorTest, LinearAttentionRequiresMetadataAndMatchingDescriptor) {
    auto              mc = makeHybridAttentionModelConfig();
    ParallelismConfig pc;
    mc.hybrid_attention_config.hybrid_attention_types.clear();
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);

    // A malformed linear descriptor must not bypass validation through missing dimensions.
    mc.linear_attention_config.linear_num_value_heads = 0;
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);

    mc                                                   = makeHybridAttentionModelConfig();
    mc.hybrid_attention_config.hybrid_attention_types[0] = HybridAttentionType::NONE;
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);
}

TEST(CacheConfigCreatorTest, DescriptorRowsMustCoverEveryLayerAndBeNonempty) {
    auto              mc = makeHybridAttentionModelConfig();
    ParallelismConfig pc;
    mc.kv_cache_spec_descs.pop_back();
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);
    mc = makeHybridAttentionModelConfig();
    mc.kv_cache_spec_descs[0].clear();
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);
}

TEST(CacheConfigCreatorTest, OpaqueDescriptorsAcceptOptionalNonLinearAttentionMetadata) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    const auto        without_metadata = CacheConfigCreator::createWarmupConfig(mc, pc, 0);
    mc.hybrid_attention_config.hybrid_attention_types.assign(mc.num_layers, HybridAttentionType::NONE);
    const auto with_metadata = CacheConfigCreator::createWarmupConfig(mc, pc, 0);
    EXPECT_EQ(publishedGroupTags(with_metadata.topology()), publishedGroupTags(without_metadata.topology()));
    for (const auto& group : with_metadata.groups()) {
        EXPECT_EQ(with_metadata.blockSizeBytesForGroup(group.tag), without_metadata.blockSizeBytesForGroup(group.tag));
        EXPECT_EQ(with_metadata.layerIdsForGroup(group.tag), without_metadata.layerIdsForGroup(group.tag));
    }
    mc.hybrid_attention_config.hybrid_attention_types.pop_back();
    EXPECT_THROW((void)CacheConfigCreator::createWarmupConfig(mc, pc, 0), std::exception);
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

TEST(CacheConfigCreatorTest, PagedPoolsShareTokensPerBlock) {
    // Pro config
    {
        ParallelismConfig pc;
        auto              config = CacheConfigCreator::createWarmupConfig(makeProModelConfig(), pc, 0);
        EXPECT_EQ(config.seq_size_per_block, kDsv4TokensPerBlock);
    }
    // Flash config
    {
        ParallelismConfig pc;
        auto              config = CacheConfigCreator::createWarmupConfig(makeFlashModelConfig(), pc, 0);
        EXPECT_EQ(config.seq_size_per_block, kDsv4TokensPerBlock);
    }
}

TEST(CacheConfigCreatorTest, AllPagedPoolsShareBlockNum) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    // Paged groups derive their block count from the confirmed candidate; explicitly
    // sized groups may override it with per-group fixed block counts.
    EXPECT_EQ(config.groupNums(), 7);
    for (const auto& group : config.groups()) {
        EXPECT_GT(group.spec->block_size_bytes(), 0u) << "pool " << group.tag;
    }
}

TEST(CacheConfigCreatorTest, DSV4StateSwaPoolsFollowGlobalBlocks) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 100;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    auto           created_config = CacheConfigCreator::createConfig(mc, pc, kv_cache_config);
    const uint32_t config_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_config, mc, runtime_config, kv_cache_config, pc);
    auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

    ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
    for (const auto& group : config.groups()) {
        EXPECT_EQ(group.block_num, 100u) << "tag=" << group.tag;
    }
    EXPECT_EQ(test::explicitPoolReserveBytes(config), 0u);
}

TEST(CacheConfigCreatorTest, DSV4HcaStatePoolBlocksOverridesOnlyHcaState) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 100;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 350);
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    auto           created_config = CacheConfigCreator::createConfig(mc, pc, kv_cache_config);
    const uint32_t config_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_config, mc, runtime_config, kv_cache_config, pc);
    auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

    ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
    for (const auto& group : config.groups()) {
        const uint32_t expected = group.tag == "hca_state" ? 350u : 100u;
        EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag;
    }

    const size_t expected_reserve = 350u * config.blockSizeBytesForGroup("hca_state");
    EXPECT_EQ(test::explicitPoolReserveBytes(config), expected_reserve);
    ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
    EXPECT_EQ(config.group("hca_state").policy.explicit_block_num, 350u);
    for (const auto& group : config.groups()) {
        if (group.tag != "hca_state") {
            EXPECT_EQ(group.policy.explicit_block_num, 0u) << "tag=" << group.tag;
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
        mc.attn_config.tokens_per_block        = seq_size_per_block;
        mc.attn_config.kernel_tokens_per_block = kernel_seq_size_per_block;
        KVCacheConfig kv_cache_config;
        kv_cache_config.seq_size_per_block        = seq_size_per_block;
        kv_cache_config.kernel_seq_size_per_block = kernel_seq_size_per_block;
        kv_cache_config.test_block_num            = 100;

        auto           created_config = CacheConfigCreator::createConfig(mc, pc, kv_cache_config);
        const uint32_t config_candidate_block_num =
            CacheConfigCreator::computeLocalBlockNum(created_config, mc, runtime_config, kv_cache_config, pc);
        return rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);
    };

    auto old_valid = create_config(128, 128);
    EXPECT_EQ(old_valid.seq_size_per_block, 128u);
    EXPECT_EQ(old_valid.group("csa_kv").kernelSeqSizePerBlock(), 128u);
    EXPECT_EQ(old_valid.group("csa_kv").kernelBlocksPerKvBlock(), 1u);
    EXPECT_EQ(old_valid.topology().group("indexer_kv").kernelSeqSizePerBlock(), 128u);

    auto decoupled = create_config(16384, 128);
    EXPECT_EQ(decoupled.seq_size_per_block, 16384u);
    EXPECT_EQ(decoupled.group("csa_kv").kernelSeqSizePerBlock(), 128u);
    EXPECT_EQ(decoupled.group("csa_kv").kernelBlocksPerKvBlock(), 128u);
}

// Absorbs DEV's CacheConfigTest.DSV4KernelSeqSizeRejectsInvalidPhysicalKernelShape.  DEV enforced
// the 128 kernel-token floor imperatively in validateDsv4KernelSeqSize(); MAIN enforces it
// declaratively in CacheConfigCreator's descriptor validation via
// kCompressedKernelSeqSizeAlignment for every KERNEL_BLOCK_COMPRESSED pool, plus the
// physical >= kernel / divisible-by-kernel check.  Both halves are pinned below.
TEST(CacheConfigTest, DSV4HybridPoolRuntimeConfigRejectsInvalidKernelShape) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    auto create_config = [&](int seq_size_per_block, int kernel_seq_size_per_block) {
        mc.attn_config.tokens_per_block        = seq_size_per_block;
        mc.attn_config.kernel_tokens_per_block = kernel_seq_size_per_block;
        KVCacheConfig kv_cache_config;
        kv_cache_config.seq_size_per_block        = seq_size_per_block;
        kv_cache_config.kernel_seq_size_per_block = kernel_seq_size_per_block;
        kv_cache_config.test_block_num            = 100;

        auto           created_config = CacheConfigCreator::createConfig(mc, pc, kv_cache_config);
        const uint32_t config_candidate_block_num =
            CacheConfigCreator::computeLocalBlockNum(created_config, mc, runtime_config, kv_cache_config, pc);
        return rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);
    };

    // Below the 128-token compression unit, even when it divides the physical block.
    EXPECT_THROW((void)create_config(16384, 32), std::exception);
    EXPECT_THROW((void)create_config(16384, 64), std::exception);
    // A multiple of the compression unit that does not divide the physical block.
    EXPECT_THROW((void)create_config(16384, 384), std::exception);
    // A kernel block larger than the physical block.
    EXPECT_THROW((void)create_config(128, 256), std::exception);
    // Exactly the compression unit is accepted.
    EXPECT_NO_THROW((void)create_config(16384, 128));
}

TEST(CacheConfigCreatorTest, DSV4HcaStatePoolBlocksIndependentOfMaxConcurrency) {
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

        auto           created_config = CacheConfigCreator::createConfig(mc, pc, kv_cache_config);
        const uint32_t config_candidate_block_num =
            CacheConfigCreator::computeLocalBlockNum(created_config, mc, runtime_config, kv_cache_config, pc);
        auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

        ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
        for (const auto& group : config.groups()) {
            const uint32_t expected = group.tag == "hca_state" ? 256u : 100u;
            EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag << " max_concurrency=" << max_concurrency;
        }
    }
}

// DEV's broader variant of the test above: DEV sized *every* fixed region (indexer_state,
// csa_state, hca_state, swa_kv) from one global dsv4_fixed_pool_blocks knob and asserted that
// none of them tracks max_concurrency.  On MAIN each pool carries its own explicit_block_num, so
// set all four and check the invariant still holds pool-by-pool.
TEST(CacheConfigCreatorTest, DSV4FixedPoolBlocksIndependentOfMaxConcurrency) {
    constexpr uint32_t kFixedPoolBlocks = 256;
    for (uint32_t max_concurrency : {1u, 2u, 8u}) {
        auto              mc = makeProModelConfig();
        ParallelismConfig pc;
        RuntimeConfig     runtime_config;
        KVCacheConfig     kv_cache_config;
        kv_cache_config.seq_size_per_block = 128;
        kv_cache_config.test_block_num     = 100;
        for (const auto& tag : dsv4StateSwaTags()) {
            setDsv4ExplicitPoolBlocks(mc, tag, kFixedPoolBlocks);
        }
        runtime_config.max_generate_batch_size                      = max_concurrency;
        runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

        auto           created_config = CacheConfigCreator::createConfig(mc, pc, kv_cache_config);
        const uint32_t config_candidate_block_num =
            CacheConfigCreator::computeLocalBlockNum(created_config, mc, runtime_config, kv_cache_config, pc);
        auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

        ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
        size_t expected_reserve = 0;
        for (const auto& group : config.groups()) {
            const bool     is_fixed = group.policy.group_type == CacheGroupType::SWA;
            const uint32_t expected = is_fixed ? kFixedPoolBlocks : 100u;
            EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag << " max_concurrency=" << max_concurrency;
            if (is_fixed) {
                expected_reserve += static_cast<size_t>(kFixedPoolBlocks) * config.blockSizeBytesForGroup(group.tag);
            }
        }
        // DEV also pinned down that every explicitly-sized pool contributes to the paged-budget
        // reservation, not just hca_state.
        EXPECT_GT(expected_reserve, 0u);
        EXPECT_EQ(test::explicitPoolReserveBytes(config), expected_reserve) << "max_concurrency=" << max_concurrency;
    }
}

TEST(CacheConfigCreatorTest, DSV4HcaStatePoolBlocksCanBeOverriddenByConfig) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 100;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 6);
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    auto           created_config = CacheConfigCreator::createConfig(mc, pc, kv_cache_config);
    const uint32_t config_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_config, mc, runtime_config, kv_cache_config, pc);
    auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

    ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
    for (const auto& group : config.groups()) {
        const uint32_t expected = group.tag == "hca_state" ? 6u : 100u;
        EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag;
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
    auto config_tp1 = CacheConfigCreator::createWarmupConfig(model_config, pc_tp1, 0);
    ASSERT_EQ(static_cast<size_t>(config_tp1.groupNums()), 1u);
    EXPECT_EQ(config_tp1.group("default").localKvHeadNum(), 4);

    ParallelismConfig pc_tp2;
    pc_tp2.tp_size  = 2;
    auto config_tp2 = CacheConfigCreator::createWarmupConfig(model_config, pc_tp2, 0);
    ASSERT_EQ(static_cast<size_t>(config_tp2.groupNums()), 1u);
    EXPECT_EQ(config_tp2.group("default").localKvHeadNum(), 2);

    EXPECT_EQ(config_tp1.group("default").localKvHeadNum(), 4);
    EXPECT_NE(config_tp1.group("default").spec.get(), config_tp2.group("default").spec.get());
}

TEST(CacheConfigCreatorTest, ResolvedModelGeometryPrecedesRawCacheOptions) {
    auto model                                = makeProModelConfig();
    model.attn_config.tokens_per_block        = 512;  // tokens/base cache-key block
    model.attn_config.kernel_tokens_per_block = 128;  // tokens/kernel page
    KVCacheConfig options;
    options.seq_size_per_block        = 64;
    options.kernel_seq_size_per_block = 32;
    const auto config                 = CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0);
    EXPECT_EQ(config.seq_size_per_block, 512u);
    for (const auto& group : config.topology().groups()) {
        EXPECT_EQ(group.seqSizePerBlock(), 512u);
        EXPECT_EQ(group.kernelSeqSizePerBlock(), group.policy.group_type == CacheGroupType::FULL ? 128u : 512u);
    }
    options.seq_size_per_block = -1;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0));
    options.seq_size_per_block        = 64;
    options.kernel_seq_size_per_block = -1;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0));
}

TEST(CacheConfigCreatorTest, UnresolvedModelGeometryUsesRawCacheOptions) {
    ModelConfig model;
    model.num_layers                          = 1;
    model.attn_config.head_num                = 1;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 16;
    model.attn_config.tokens_per_block        = 0;
    model.attn_config.kernel_tokens_per_block = 0;
    setDefaultKvCacheSpec(model);
    KVCacheConfig options;
    EXPECT_EQ(options.seq_size_per_block, 0);
    EXPECT_EQ(options.kernel_seq_size_per_block, 0);
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0));

    options.seq_size_per_block = 64;  // tokens/cache-key block
    const auto fallback        = CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0);
    EXPECT_EQ(fallback.seq_size_per_block, 64u);
    EXPECT_EQ(fallback.group("default").seqSizePerBlock(), 64u);        // tokens/group physical block
    EXPECT_EQ(fallback.group("default").kernelSeqSizePerBlock(), 64u);  // tokens/kernel page

    options.seq_size_per_block        = 64;
    options.kernel_seq_size_per_block = 16;
    const auto config                 = CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0);
    EXPECT_EQ(config.seq_size_per_block, 64u);
    EXPECT_EQ(config.group("default").seqSizePerBlock(), 64u);
    EXPECT_EQ(config.group("default").kernelSeqSizePerBlock(), 16u);

    options.kernel_seq_size_per_block = 128;  // tokens/kernel page: exceeds the physical block
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0));
    options.kernel_seq_size_per_block = 24;  // tokens/kernel page: does not divide the physical block
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0));
}

TEST(CacheConfigCreatorTest, ModelGeometryUsesUint32Range) {
    ModelConfig model;
    model.num_layers                = 1;
    model.attn_config.head_num      = 1;
    model.attn_config.kv_head_num   = 1;
    model.attn_config.size_per_head = 16;
    setDefaultKvCacheSpec(model);
    const KVCacheConfig options;
    const auto          max_span       = std::numeric_limits<uint32_t>::max();
    model.attn_config.tokens_per_block = max_span;  // tokens/cache-key block
    for (const auto kernel_span : {0u, 1u, max_span}) {
        model.attn_config.kernel_tokens_per_block = kernel_span;  // tokens/kernel page; 0 means unspecified
        const auto config = CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0);
        EXPECT_EQ(config.seq_size_per_block, max_span);
        EXPECT_EQ(config.group("default").seqSizePerBlock(), max_span);
        EXPECT_EQ(config.group("default").kernelSeqSizePerBlock(), kernel_span == 0 ? max_span : kernel_span);
    }

    const auto overflow_span                  = static_cast<size_t>(max_span) + 1;
    model.attn_config.tokens_per_block        = overflow_span;
    model.attn_config.kernel_tokens_per_block = 1;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0));
    model.attn_config.tokens_per_block        = max_span;
    model.attn_config.kernel_tokens_per_block = overflow_span;
    EXPECT_ANY_THROW(CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0));
}

TEST(CacheConfigTest, RuntimeKernelBlockOverrideUpdatesTopology) {
    ModelConfig model_config;
    model_config.data_type                    = DataType::TYPE_BF16;
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

    auto           created_config = CacheConfigCreator::createConfig(model_config, parallelism_config, kv_cache_config);
    const uint32_t config_candidate_block_num = CacheConfigCreator::computeLocalBlockNum(
        created_config, model_config, runtime_config, kv_cache_config, parallelism_config);
    auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

    ASSERT_EQ(config.groupNums(), 1);
    EXPECT_EQ(config.seq_size_per_block, 512u);
    EXPECT_EQ(config.group("default").kernelSeqSizePerBlock(), 64u);
    EXPECT_EQ(config.group("default").seqSizePerBlock(), 512u);
    EXPECT_EQ(config.group("default").kernelSeqSizePerBlock(), 64u);
    EXPECT_EQ(config.group("default").kernelBlocksPerKvBlock(), 8u);
    EXPECT_EQ(config.group("default").kvBlockStrideBytes(), 512u * (512u + 64u) * sizeof(at::BFloat16));
}

TEST(CacheConfigCreatorTest, MhaPhysicalDataAndScaleStridesAreNotExpandedTwice) {
    ModelConfig model;
    model.num_layers                   = 1;
    model.attn_config.head_num         = 2;
    model.attn_config.kv_head_num      = 2;
    model.attn_config.size_per_head    = 16;
    model.attn_config.tokens_per_block = 32;
    setDefaultKvCacheSpec(model);
    model.kv_cache_spec_descs[0][0].dtype = DataType::TYPE_INT8;
    KVCacheConfig options;
    options.kernel_seq_size_per_block = 8;
    const auto config                 = CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, options, 0);
    EXPECT_EQ(config.group("default").kernelBlocksPerKvBlock(), 4u);
    EXPECT_EQ(config.group("default").kvBlockStrideBytes(), 2u * 2u * 16u * 32u);
    EXPECT_EQ(config.group("default").kvScaleStrideBytes(), 2u * 2u * 32u * sizeof(float));
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
    EXPECT_EQ(compressed->block_size() / compressed_desc.entry_elems, 64u);
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

TEST(CacheConfigTest, NativeStateRingAlignmentPreservesSpeculativeRollbackWindow) {
    KVCacheSpecDesc desc;
    desc.tag = "indexer_state";
    desc.cache_type = KVCacheSpecType::OpaqueState;
    desc.entry_elems = 512;
    desc.entry_dtype = DataType::TYPE_FP32;
    desc.entry_count_mode = OpaqueBlockEntryCountMode::STATE_RING;
    desc.compression_ratio = 4;
    desc.state_ring_overlap = 1;
    desc.state_ring_include_gen_num_per_cycle = true;
    desc.state_ring_entry_alignment = 4;
    for (uint32_t gamma : {0u, 1u, 3u}) {
        SpecBuildContext ctx;
        ctx.gen_num_per_cycle = gamma;
        ctx.seq_size_per_block = 256;
        ctx.kernel_tokens_per_block = 256;
        auto spec = std::dynamic_pointer_cast<FixedStateCacheSpec>(SpecBuilder::build(desc, ctx));
        ASSERT_NE(spec, nullptr);
        EXPECT_EQ(opaqueEntriesPerBlock(*spec, 512u * sizeof(float)), gamma == 0 ? 8u : 12u);
    }
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

TEST(CacheConfigTest, FinalizeBlockNumsUpdatesHybridPerGroupBlockNums) {
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
    auto single_config = CacheConfigCreator::createWarmupConfig(single_model_config, pc, 0);
    single_config.finalizeBlockNums(123, runtime_config);
    ASSERT_EQ(single_config.groupNums(), 1);
    EXPECT_EQ(single_config.topology().groups().front().block_num, 123u);
    EXPECT_EQ(test::explicitPoolReserveBytes(single_config), 0u);

    auto hybrid_config =
        CacheConfigCreator::createWarmupConfig(makeHybridAttentionModelConfig(/*size_per_head=*/32), pc, 0);
    hybrid_config.finalizeBlockNums(123, runtime_config);
    EXPECT_GT(hybrid_config.groupNums(), 1);
    EXPECT_EQ(hybrid_config.group("linear").block_num, 123u);
    EXPECT_EQ(hybrid_config.group("full").block_num, 123u);
    EXPECT_EQ(test::explicitPoolReserveBytes(hybrid_config), 0u);
}

TEST(CacheConfigTest, FinalizeBlockNumsAppliesToIndependentPools) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeProModelConfig(), pc, 0);
    config.finalizeBlockNums(100, runtime_config);

    ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
    for (const auto& group : config.groups()) {
        const uint32_t expected = group.tag == "hca_state" ? 256u : 100u;
        EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag;
    }
    EXPECT_EQ(test::explicitPoolReserveBytes(config), 256u * config.blockSizeBytesForGroup("hca_state"));
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

    auto           created_config_with = CacheConfigCreator::createConfig(mc, pc, kv_cache_config_with);
    const uint32_t config_with_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_config_with, mc, runtime_config, kv_cache_config_with, pc);
    auto config_with = rtp_llm::test::finalizeCacheConfig(created_config_with, config_with_candidate_block_num);

    KVCacheConfig kv_cache_config_without;
    kv_cache_config_without.seq_size_per_block = 128;
    kv_cache_config_without.kv_cache_mem_mb    = 65536;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", large_hca_state_pool);

    auto           created_config_without = CacheConfigCreator::createConfig(mc, pc, kv_cache_config_without);
    const uint32_t config_without_candidate_block_num = CacheConfigCreator::computeLocalBlockNum(
        created_config_without, mc, runtime_config, kv_cache_config_without, pc);
    auto config_without =
        rtp_llm::test::finalizeCacheConfig(created_config_without, config_without_candidate_block_num);

    // More HCA_STATE blocks reserve more HBM and leave fewer blocks for the global pools.
    EXPECT_GT(config_with_candidate_block_num, config_without_candidate_block_num);
    EXPECT_EQ(config_with.group("hca_kv").block_num, static_cast<uint32_t>(config_with_candidate_block_num));
    EXPECT_EQ(config_without.group("hca_kv").block_num, static_cast<uint32_t>(config_without_candidate_block_num));
    EXPECT_EQ(config_with.group("hca_state").block_num, small_hca_state_pool);
    EXPECT_EQ(config_without.group("hca_state").block_num, large_hca_state_pool);
    const size_t expected_reserve =
        static_cast<size_t>(small_hca_state_pool) * config_with.blockSizeBytesForGroup("hca_state");
    EXPECT_EQ(test::explicitPoolReserveBytes(config_with), expected_reserve);
}

TEST(CacheConfigTest, DSV4ExplicitHcaStatePoolBlocksIgnoreLinearStep) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(makeProModelConfig(), pc, 0);
    config.linear_step       = 4;
    config.finalizeBlockNums(100, runtime_config);

    // The explicit pool keeps its requested capacity. Non-explicit FULL/LINEAR
    // groups keep N, while non-explicit SWA groups use ceil(N / step).
    for (const auto& group : config.groups()) {
        const uint32_t expected =
            group.tag == "hca_state" ? 256u : (group.policy.group_type == CacheGroupType::SWA ? 25u : 100u);
        EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag;
    }
    const size_t expected_reserve = 256u * config.blockSizeBytesForGroup("hca_state");
    EXPECT_EQ(test::explicitPoolReserveBytes(config), expected_reserve);
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

    auto           created_config = CacheConfigCreator::createConfig(mc, pc, kv_cache_config);
    const uint32_t config_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_config, mc, runtime_config, kv_cache_config, pc);
    auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

    ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
    for (const auto& group : config.groups()) {
        const uint32_t expected = group.policy.group_type == CacheGroupType::SWA ? 25u : 100u;
        EXPECT_EQ(group.block_num, expected) << "tag=" << group.tag;
    }
    EXPECT_EQ(test::explicitPoolReserveBytes(config), 0u);
}

// Ported from DEV CacheConfigTest.DSV4PinnedFixedPoolFallbackIsExcludedFromGpuBudget.
// DEV flipped every fixed pool to pinned host memory with KVCacheConfig::dsv4_fixed_pool_use_memory
// and asserted the device pool grew because those pools stopped charging the GPU budget.  On MAIN
// budget exclusion is `charge_to_paged_budget = false`, which blockBudgetForConfig() only honours
// for explicitly-sized pools (non-explicit pools always contribute marginal bytes regardless of
// residency), so the pinned pools are sized explicitly here.
TEST(CacheConfigTest, DSV4PinnedFixedPoolFallbackIsExcludedFromGpuBudget) {
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    constexpr uint32_t kFixedPoolBlocks = 64;

    auto make_kv_config = [] {
        KVCacheConfig kv_cache_config;
        kv_cache_config.seq_size_per_block = 128;
        kv_cache_config.kv_cache_mem_mb    = 65536;
        kv_cache_config.linear_step        = 4;
        return kv_cache_config;
    };

    auto gpu_fixed_mc = makeProModelConfig();
    for (const auto& tag : dsv4StateSwaTags()) {
        setDsv4ExplicitPoolBlocks(gpu_fixed_mc, tag, kFixedPoolBlocks);
    }
    auto pinned_fixed_mc = gpu_fixed_mc;
    for (const auto& tag : dsv4StateSwaTags()) {
        setDsv4PoolMemoryPlacement(pinned_fixed_mc, tag, CacheMemoryPlacement::HOST_PINNED);
    }

    auto           created_gpu_fixed = CacheConfigCreator::createConfig(gpu_fixed_mc, pc, make_kv_config());
    const uint32_t gpu_fixed_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_gpu_fixed, gpu_fixed_mc, runtime_config, make_kv_config(), pc);
    auto gpu_fixed = rtp_llm::test::finalizeCacheConfig(created_gpu_fixed, gpu_fixed_candidate_block_num);

    auto           created_pinned_fixed = CacheConfigCreator::createConfig(pinned_fixed_mc, pc, make_kv_config());
    const uint32_t pinned_fixed_candidate_block_num = CacheConfigCreator::computeLocalBlockNum(
        created_pinned_fixed, pinned_fixed_mc, runtime_config, make_kv_config(), pc);
    auto pinned_fixed = rtp_llm::test::finalizeCacheConfig(created_pinned_fixed, pinned_fixed_candidate_block_num);

    // Device-resident fixed pools reserve HBM; pinned ones do not, so the paged pool is larger.
    EXPECT_GT(test::explicitPoolReserveBytes(gpu_fixed), 0u);
    EXPECT_EQ(test::explicitPoolReserveBytes(pinned_fixed), 0u);
    EXPECT_GT(pinned_fixed_candidate_block_num, gpu_fixed_candidate_block_num);

    for (const auto& group : pinned_fixed.groups()) {
        const auto tag = group.tag;
        if (group.policy.group_type != CacheGroupType::SWA) {
            continue;
        }
        EXPECT_EQ(group.block_num, kFixedPoolBlocks) << "tag=" << tag;
        EXPECT_EQ(group.policy.memory_placement, CacheMemoryPlacement::HOST_PINNED) << "tag=" << tag;
        EXPECT_EQ(gpu_fixed.group(tag).block_num, kFixedPoolBlocks) << "tag=" << tag;
    }
}

// Ported from DEV CacheConfigTest.DSV4PinnedFixedPoolFallbackFollowsExpandedFullPoolWhenStepOne.
// DEV's claim: with linear_step == 1 the non-explicit fixed pools track the full pool 1:1, and
// putting them in pinned host memory does not perturb that rule.  MAIN's ceil(N / step) rule
// degenerates to N at step 1, so both halves are asserted here.  DEV additionally expected the
// pinned variant's block_num to grow; on MAIN residency alone does not change the budget for
// non-explicit pools (see DSV4PinnedFixedPoolFallbackIsExcludedFromGpuBudget above for the
// explicitly-sized case that does), so block_num is asserted equal instead.
TEST(CacheConfigTest, DSV4PinnedFixedPoolFallbackFollowsExpandedFullPoolWhenStepOne) {
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    auto make_kv_config = [] {
        KVCacheConfig kv_cache_config;
        kv_cache_config.seq_size_per_block = 128;
        kv_cache_config.kv_cache_mem_mb    = 65536;
        kv_cache_config.linear_step        = 1;
        return kv_cache_config;
    };

    auto gpu_fixed_mc = makeProModelConfig();
    for (const auto& tag : dsv4StateSwaTags()) {
        setDsv4ExplicitPoolBlocks(gpu_fixed_mc, tag, 0);
    }
    auto pinned_fixed_mc = gpu_fixed_mc;
    for (const auto& tag : dsv4StateSwaTags()) {
        setDsv4PoolMemoryPlacement(pinned_fixed_mc, tag, CacheMemoryPlacement::HOST_PINNED);
    }

    auto           created_gpu_fixed = CacheConfigCreator::createConfig(gpu_fixed_mc, pc, make_kv_config());
    const uint32_t gpu_fixed_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_gpu_fixed, gpu_fixed_mc, runtime_config, make_kv_config(), pc);
    auto gpu_fixed = rtp_llm::test::finalizeCacheConfig(created_gpu_fixed, gpu_fixed_candidate_block_num);

    auto           created_pinned_fixed = CacheConfigCreator::createConfig(pinned_fixed_mc, pc, make_kv_config());
    const uint32_t pinned_fixed_candidate_block_num = CacheConfigCreator::computeLocalBlockNum(
        created_pinned_fixed, pinned_fixed_mc, runtime_config, make_kv_config(), pc);
    auto pinned_fixed = rtp_llm::test::finalizeCacheConfig(created_pinned_fixed, pinned_fixed_candidate_block_num);

    EXPECT_EQ(test::explicitPoolReserveBytes(gpu_fixed), 0u);
    EXPECT_EQ(test::explicitPoolReserveBytes(pinned_fixed), 0u);
    EXPECT_EQ(pinned_fixed_candidate_block_num, gpu_fixed_candidate_block_num);

    ASSERT_EQ(gpu_fixed.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
    ASSERT_EQ(pinned_fixed.topology().groups().size(), static_cast<size_t>(kDsv4PoolNum));
    for (const auto& group : pinned_fixed.groups()) {
        EXPECT_EQ(gpu_fixed.group(group.tag).block_num, static_cast<uint32_t>(gpu_fixed_candidate_block_num))
            << "tag=" << group.tag;
        EXPECT_EQ(group.block_num, static_cast<uint32_t>(pinned_fixed_candidate_block_num)) << "tag=" << group.tag;
    }
}

TEST(CacheConfigTest, DSV4MtpKeepsProposeLayerInSwaPool) {
    auto score_model_config                                  = makeFlashModelConfig();
    auto propose_model_config                                = makeFlashMtpModelConfig();
    score_model_config.attn_config.tokens_per_block          = 16384;
    score_model_config.attn_config.kernel_tokens_per_block   = 128;
    propose_model_config.attn_config.tokens_per_block        = 16384;
    propose_model_config.attn_config.kernel_tokens_per_block = 128;
    score_model_config.attn_config.kv_cache_dtype            = KvCacheDataType::FP8;
    propose_model_config.attn_config.kv_cache_dtype          = KvCacheDataType::FP8;

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

    auto created_config = CacheConfigCreator::createConfig(
        score_model_config, parallelism_config, kv_cache_config, sp_config, &propose_model_config, true, false);
    const uint32_t config_candidate_block_num = CacheConfigCreator::computeLocalBlockNum(created_config,
                                                                                         score_model_config,
                                                                                         runtime_config,
                                                                                         kv_cache_config,
                                                                                         parallelism_config,
                                                                                         std::nullopt,
                                                                                         sp_config);
    auto           config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

    ASSERT_EQ(config.layer_num, 43u);
    ASSERT_EQ(config.layer_all_num(), 45u);
    ASSERT_EQ(config.topology().layers().size(), static_cast<size_t>(config.layer_all_num()));
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    ASSERT_NE(config.mtp_sub_configs[0], nullptr);
    ASSERT_NE(config.mtp_sub_configs[1], nullptr);

    EXPECT_EQ(config.groupForLayer(43, "swa_kv").tag, "swa_kv");
    EXPECT_EQ(config.topology().layer(43).group_tags, std::vector<std::string>({"swa_kv"}));
    EXPECT_EQ(config.topology().layer(44).group_tags, std::vector<std::string>({"swa_kv"}));
    EXPECT_EQ(config.groupForLayer(44, "swa_kv").tag, "swa_kv");

    EXPECT_EQ(config.layerIdsForGroup("swa_kv").size(), 45u);

    // Merge preserves target publication order and placeholder identities;
    // only layer IDs are draft-local. Runtime block tables carry their own tags.
    EXPECT_EQ(publishedGroupTags(config.mtp_sub_configs[0]->topology()), publishedGroupTags(config.topology()));
    EXPECT_EQ(publishedGroupTags(config.mtp_sub_configs[1]->topology()), publishedGroupTags(config.topology()));
    EXPECT_EQ(config.mtp_sub_configs[0]->groupForLayer(0, "swa_kv").tag, "swa_kv");
    EXPECT_EQ(config.mtp_sub_configs[1]->groupForLayer(0, "swa_kv").tag, "swa_kv");
    EXPECT_EQ(config.mtp_sub_configs[0]->layerIdsForGroup(config.mtp_sub_configs[0]->group("swa_kv").tag),
              std::vector<int>({0}));
    EXPECT_EQ(config.mtp_sub_configs[1]->layerIdsForGroup(config.mtp_sub_configs[1]->group("swa_kv").tag),
              std::vector<int>({0}));
    for (const auto& group : config.groups()) {
        if (group.tag == "swa_kv") {
            continue;
        }
        EXPECT_TRUE(config.mtp_sub_configs[0]->layerIdsForGroup(group.tag).empty()) << group.tag;
        EXPECT_TRUE(config.mtp_sub_configs[1]->layerIdsForGroup(group.tag).empty()) << group.tag;
    }
    EXPECT_EQ(config.seq_size_per_block, 16384u);

    EXPECT_EQ(config.group("csa_kv").kernelSeqSizePerBlock(), 128u);
    EXPECT_EQ(config.group("csa_kv").kernelBlocksPerKvBlock(), 128u);
    EXPECT_EQ(config.group("swa_kv").kernelBlocksPerKvBlock(), 1u);
    EXPECT_EQ(config.mtp_sub_configs[0]->seq_size_per_block, 16384u);
    EXPECT_EQ(config.mtp_sub_configs[0]->group("csa_kv").kernelSeqSizePerBlock(), 128u);

    EXPECT_EQ(config.group("swa_kv").block_num, 25u);
    EXPECT_EQ(config.mtp_sub_configs[0]->linear_step, 4);
    EXPECT_EQ(config.mtp_sub_configs[1]->linear_step, 4);
    EXPECT_EQ(config.mtp_sub_configs[0]->group("swa_kv").block_num, 25u);
    EXPECT_EQ(config.mtp_sub_configs[1]->group("swa_kv").block_num, 25u);

    EXPECT_EQ(test::explicitPoolReserveBytes(config), 256u * config.blockSizeBytesForGroup("hca_state"));
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

    auto created_config = CacheConfigCreator::createConfig(
        score_model_config, parallelism_config, kv_cache_config, sp_config, &propose_model_config, true, false);
    const uint32_t config_candidate_block_num = CacheConfigCreator::computeLocalBlockNum(created_config,
                                                                                         score_model_config,
                                                                                         runtime_config,
                                                                                         kv_cache_config,
                                                                                         parallelism_config,
                                                                                         std::nullopt,
                                                                                         sp_config);
    auto           config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);

    size_t paged_bytes = 0;
    size_t swa_bytes   = 0;
    for (const auto& group : config.groups()) {
        const auto explicit_blocks = group.policy.explicit_block_num;
        if (explicit_blocks > 0) {
            EXPECT_EQ(group.block_num, explicit_blocks) << "tag=" << group.tag;
            continue;
        }
        if (group.policy.group_type == CacheGroupType::SWA) {
            swa_bytes += config.blockSizeBytesForGroup(group.tag);
            EXPECT_EQ(group.block_num, (static_cast<uint32_t>(config_candidate_block_num) + 3u) / 4u)
                << "tag=" << group.tag;
        } else {
            paged_bytes += config.blockSizeBytesForGroup(group.tag);
            EXPECT_EQ(group.block_num, static_cast<uint32_t>(config_candidate_block_num)) << "tag=" << group.tag;
        }
    }

    const auto backing_bytes = [&](uint32_t block_num) {
        return test::explicitPoolReserveBytes(config) + static_cast<size_t>(block_num) * paged_bytes
               + static_cast<size_t>((block_num + 3u) / 4u) * swa_bytes;
    };
    const size_t budget_bytes = static_cast<size_t>(kv_cache_config.kv_cache_mem_mb) * 1024u * 1024u;
    const auto   block_num    = static_cast<uint32_t>(config_candidate_block_num);
    EXPECT_LE(backing_bytes(block_num), budget_bytes);
    EXPECT_GT(backing_bytes(block_num + 1u), budget_bytes);

    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_NE(sub_config, nullptr);
        EXPECT_EQ(sub_config->linear_step, 4);
        EXPECT_EQ(sub_config->group("swa_kv").block_num, (block_num + 3u) / 4u);
    }
}

TEST(CacheConfigCreatorTest, MtpGenNum2RingEntriesMatch) {
    // gen_num_per_cycle=2 -> CSA/INDEXER R=10, HCA R=130, SWA R=130.
    // Formula: R = ceil_even((1 + overlap) * ratio + gen_num_per_cycle).
    // SWA_KV is sized like the HCA state ring (window 128, overlap 0).
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, /*gen_num_per_cycle=*/2);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    // Pool 3: INDEXER_STATE (ratio=4, overlap=1) → R=10
    auto* indexer_state = dynamic_cast<const FixedStateCacheSpec*>(config.group("indexer_state").spec.get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*indexer_state, kDsv4IndexerStateEntryBytes), 10u);
    // Pool 4: CSA_STATE (ratio=4, overlap=1) → R=10
    auto* csa_state = dynamic_cast<const FixedStateCacheSpec*>(config.group("csa_state").spec.get());
    ASSERT_NE(csa_state, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*csa_state, kDsv4CsaStateEntryBytes), 10u);
    // Pool 5: HCA_STATE (ratio=128, overlap=0) → R=130
    auto* hca_state = dynamic_cast<const FixedStateCacheSpec*>(config.group("hca_state").spec.get());
    ASSERT_NE(hca_state, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*hca_state, kDsv4HcaStateEntryBytes), 130u);
    // Pool 6: SWA_KV (window=128, overlap=0) → R=130, same as HCA_STATE
    auto* swa_kv = dynamic_cast<const FixedStateCacheSpec*>(config.group("swa_kv").spec.get());
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(swa_kv->tag, "swa_kv");
    EXPECT_EQ(opaqueEntriesPerBlock(*swa_kv, kDsv4KvEntryBytes), 130u);
}

TEST(CacheConfigCreatorTest, PrefillCp8MtpGenNum2PadsStateRingBeforeSlicing) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = 8;
    pc.prefill_cp_config.kv_cache_sharded = true;

    auto config = CacheConfigCreator::createWarmupConfig(mc, pc, 2);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    auto* indexer_state = dynamic_cast<const FixedStateCacheSpec*>(config.group("indexer_state").spec.get());
    auto* csa_state     = dynamic_cast<const FixedStateCacheSpec*>(config.group("csa_state").spec.get());
    auto* hca_state     = dynamic_cast<const FixedStateCacheSpec*>(config.group("hca_state").spec.get());
    auto* swa_kv        = dynamic_cast<const FixedStateCacheSpec*>(config.group("swa_kv").spec.get());
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

TEST(CacheConfigCreatorTest, DecodePrefillCp8MtpGenNum2ExpandsFixedAndSwaSlices) {
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

    auto prefill_config = CacheConfigCreator::createWarmupConfig(mc, prefill_pc, 2);
    auto decode_config  = CacheConfigCreator::createWarmupConfig(mc, decode_pc, 2);

    ASSERT_EQ(static_cast<size_t>(prefill_config.groupNums()), 7u);
    ASSERT_EQ(static_cast<size_t>(decode_config.groupNums()), 7u);

    for (const auto& tag : {"indexer_state", "csa_state", "hca_state"}) {

        auto* prefill_spec = dynamic_cast<const FixedStateCacheSpec*>(prefill_config.group(tag).spec.get());
        auto* decode_spec  = dynamic_cast<const FixedStateCacheSpec*>(decode_config.group(tag).spec.get());
        ASSERT_NE(prefill_spec, nullptr) << tag;
        ASSERT_NE(decode_spec, nullptr) << tag;
        EXPECT_EQ(decode_spec->tag, prefill_spec->tag) << tag;
        const auto expected_entries = opaqueEntriesPerBlock(*prefill_spec, stateEntryBytesForTag(tag)) * cp_size;
        EXPECT_EQ(opaqueEntriesPerBlock(*decode_spec, stateEntryBytesForTag(tag)), expected_entries) << tag;
    }
    auto* prefill_swa = dynamic_cast<const FixedStateCacheSpec*>(prefill_config.group("swa_kv").spec.get());
    auto* decode_swa  = dynamic_cast<const FixedStateCacheSpec*>(decode_config.group("swa_kv").spec.get());
    ASSERT_NE(prefill_swa, nullptr);
    ASSERT_NE(decode_swa, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*prefill_swa, kDsv4KvEntryBytes), 136u);
    EXPECT_EQ(opaqueEntriesPerBlock(*decode_swa, kDsv4KvEntryBytes),
              opaqueEntriesPerBlock(*prefill_swa, kDsv4KvEntryBytes));

    auto* indexer_state = dynamic_cast<const FixedStateCacheSpec*>(decode_config.group("indexer_state").spec.get());
    auto* csa_state     = dynamic_cast<const FixedStateCacheSpec*>(decode_config.group("csa_state").spec.get());
    auto* hca_state     = dynamic_cast<const FixedStateCacheSpec*>(decode_config.group("hca_state").spec.get());
    auto* swa_kv        = dynamic_cast<const FixedStateCacheSpec*>(decode_config.group("swa_kv").spec.get());
    ASSERT_NE(indexer_state, nullptr);
    ASSERT_NE(csa_state, nullptr);
    ASSERT_NE(hca_state, nullptr);
    ASSERT_NE(swa_kv, nullptr);

    EXPECT_EQ(opaqueEntriesPerBlock(*indexer_state, kDsv4IndexerStateEntryBytes), 16u);
    EXPECT_EQ(opaqueEntriesPerBlock(*csa_state, kDsv4CsaStateEntryBytes), 16u);
    EXPECT_EQ(opaqueEntriesPerBlock(*hca_state, kDsv4HcaStateEntryBytes), 136u);
    EXPECT_EQ(opaqueEntriesPerBlock(*swa_kv, kDsv4KvEntryBytes), 136u);
}

TEST(CacheConfigCreatorTest, DecodeExplicitPrefillCpSizeHandlesDp16) {
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

    auto prefill_config = CacheConfigCreator::createWarmupConfig(mc, prefill_pc, 2);
    auto decode_config  = CacheConfigCreator::createWarmupConfig(mc, decode_pc, 2);

    for (const auto& tag : {"indexer_state", "csa_state", "hca_state"}) {

        auto* prefill_spec = dynamic_cast<const FixedStateCacheSpec*>(prefill_config.group(tag).spec.get());
        auto* decode_spec  = dynamic_cast<const FixedStateCacheSpec*>(decode_config.group(tag).spec.get());
        ASSERT_NE(prefill_spec, nullptr) << tag;
        ASSERT_NE(decode_spec, nullptr) << tag;
        const auto expected_entries = opaqueEntriesPerBlock(*prefill_spec, stateEntryBytesForTag(tag)) * cp_size;
        EXPECT_EQ(opaqueEntriesPerBlock(*decode_spec, stateEntryBytesForTag(tag)), expected_entries) << tag;
    }
    auto* prefill_swa = dynamic_cast<const FixedStateCacheSpec*>(prefill_config.group("swa_kv").spec.get());
    auto* decode_swa  = dynamic_cast<const FixedStateCacheSpec*>(decode_config.group("swa_kv").spec.get());
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

    auto           created_config = CacheConfigCreator::createConfig(mc, pc, kvc, std::make_optional(sp_none));
    const uint32_t config_candidate_block_num = CacheConfigCreator::computeLocalBlockNum(
        created_config, mc, rc, kvc, pc, std::nullopt, std::make_optional(sp_none));
    auto config = rtp_llm::test::finalizeCacheConfig(created_config, config_candidate_block_num);
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    // CSA_STATE (pool 4): ratio=4, overlap=1, gen_num=0 → R=8
    auto* csa = dynamic_cast<const FixedStateCacheSpec*>(config.group("csa_state").spec.get());
    ASSERT_NE(csa, nullptr);
    EXPECT_EQ(opaqueEntriesPerBlock(*csa, kDsv4CsaStateEntryBytes), 8u) << "SP_TYPE_NONE should not inflate ring";
}

TEST(CacheConfigCreatorTest, BlockIdConsistencyAcrossGroups) {
    // Every logical layer exposes its complete group membership by identity.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);

    const auto& layers = config.topology().layers();
    EXPECT_EQ(layers.size(), 61u);
    for (const auto& layer : layers) {
        EXPECT_FALSE(layer.group_tags.empty()) << "layer " << layer.layer_id;
    }

    // Verify group layer ids: each group has the correct layer list.
    EXPECT_EQ(config.layerIdsForGroup("csa_kv"), config.layerIdsForGroup("indexer_kv"));
    EXPECT_EQ(config.layerIdsForGroup("csa_kv"), config.layerIdsForGroup("indexer_state"));
    EXPECT_EQ(config.layerIdsForGroup("csa_kv"), config.layerIdsForGroup("csa_state"));
    EXPECT_EQ(config.layerIdsForGroup("hca_kv"), config.layerIdsForGroup("hca_state"));
}

// ============================================================
// Helper: build a DSV4 CacheConfig with block_num set for allocator tests
// ============================================================

static CacheConfig makeDSV4AllocatorConfig(bool use_flash = false) {
    auto              mc = use_flash ? makeFlashModelConfig() : makeProModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);
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
    auto config                           = CacheConfigCreator::createWarmupConfig(mc, pc, 0);
    setGroupBlockNumsForTest(
        config, config.groupTags(), std::vector<uint32_t>(static_cast<size_t>(config.groupNums()), 200u));
    return config;
}

// ============================================================
// CoordinatorCacheManager integration tests with DSV4 7-group config
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

    // 7 groups → CoordinatorCacheManager path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(allocator->seqSizePerBlock(), static_cast<int>(config.seq_size_per_block));
    size_t expected_blocks = 0;
    for (const auto& group : config.topology().groups()) {
        expected_blocks += group.block_num - 1;
    }
    EXPECT_EQ(allocator->totalBlocksNum(), expected_blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_blocks);
}

TEST_F(DSV4AllocatorTest, CompressedBlockCopyIncludesEveryPageAndPadding) {
    constexpr size_t kBlockStrideBytes = 512;
    ModelConfig      model;
    model.num_layers                   = 1;
    model.attn_config.tokens_per_block = 512;
    KVCacheSpecDesc desc;
    desc.tag                          = "compressed";
    desc.cache_type                   = KVCacheSpecType::OpaqueKV;
    desc.entry_dtype                  = DataType::TYPE_UINT8;
    desc.entry_elems                  = 3;
    desc.entry_count_mode             = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    desc.compression_ratio            = 4;
    desc.block_stride_bytes_alignment = 128;
    model.kv_cache_spec_descs         = {{desc}};
    KVCacheConfig cache_options;
    cache_options.kernel_seq_size_per_block = 128;
    auto config = CacheConfigCreator::createWarmupConfig(model, ParallelismConfig{}, cache_options, 0);
    config.finalizeBlockNums(4, RuntimeConfig{});
    CoordinatorCacheManager allocator(config, AllocationType::HOST);
    ASSERT_TRUE(allocator.init());
    ASSERT_EQ(config.group("compressed").kvBlockStrideBytes(), kBlockStrideBytes);
    auto* src  = static_cast<uint8_t*>(allocator.convertIndexToAddr(0, 1).kv_addr);
    auto* dst  = static_cast<uint8_t*>(allocator.convertIndexToAddr(0, 2).kv_addr);
    auto* next = static_cast<uint8_t*>(allocator.convertIndexToAddr(0, 3).kv_addr);
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_NE(next, nullptr);
    for (size_t i = 0; i < kBlockStrideBytes; ++i) {
        src[i]  = static_cast<uint8_t>(i + 1);
        dst[i]  = 0;
        next[i] = 99;
    }
    allocator.blockCopy(1, 2);
    for (size_t i = 0; i < kBlockStrideBytes; ++i) {
        EXPECT_EQ(dst[i], static_cast<uint8_t>(i + 1)) << i;
        EXPECT_EQ(src[i], static_cast<uint8_t>(i + 1)) << i;
        EXPECT_EQ(next[i], 99) << i;
    }
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
    info.enable_cache_lookup = false;
    info.reuse_cache         = false;

    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    for (const auto& group : config.groups()) {
        EXPECT_EQ(batch_res->blocksNum(0, group.tag), 1u) << "tag=" << group.tag;
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
    size_t expected_blocks = 0;
    for (const auto& group : config.topology().groups()) {
        expected_blocks += group.block_num - 1;
    }
    EXPECT_EQ(allocator->totalBlocksNum(), expected_blocks);
}

TEST_F(DSV4AllocatorTest, AddressLookupAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // Verify address lookup works for a layer in each group
    // Group 0 (CSA KV): csa_layer_ids[0]
    // Group 1 (HCA KV): hca_layer_ids[0]
    // Group 6 (SWA KV): all_layer_ids[0]
    for (const auto& group : config.groups()) {
        ASSERT_FALSE(config.layerIdsForGroup(group.tag).empty()) << "group " << group.tag << " has no layers";
        int  layer_id = config.layerIdsForGroup(group.tag)[0];
        auto addr     = allocator->convertIndexToAddr(layer_id, group.tag, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr) << "null kv_addr for group " << group.tag << " layer " << layer_id;
    }
}

TEST_F(DSV4AllocatorTest, DeviceBlockPoolCreatedWithCorrectTensors) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->groupBlockPools().front();
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
    for (const auto& group : config.groups()) {
        int  layer_id = config.layerIdsForGroup(group.tag)[0];
        auto buf      = allocator->convertIndexToBuffer(layer_id, group.tag, /*block_id=*/1);
        ASSERT_FALSE(buf.empty()) << "empty buffer for group " << group.tag;
        EXPECT_NE(buf[0].addr, nullptr) << "null addr for group " << group.tag;
    }
}

TEST_F(DSV4AllocatorTest, MallocAndFreeBlocks) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->groupBlockPools().front();
    ASSERT_NE(block_pool, nullptr);

    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 3u);

    // Direct block pool malloc/free
    auto blocks = block_pool->malloc(3);
    ASSERT_TRUE(blocks.has_value());
    ASSERT_EQ(blocks->size(), 3u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 3);

    block_pool->incRef(*blocks);
    block_pool->decRef(*blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(DSV4AllocatorTest, SevenGroupLayerMapping) {
    auto config = makeDSV4AllocatorConfig();

    EXPECT_EQ(config.layerIdsForGroup("csa_kv").size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup("hca_kv").size(), 31u);
    EXPECT_EQ(config.layerIdsForGroup("indexer_kv").size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup("indexer_state").size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup("csa_state").size(), 30u);
    EXPECT_EQ(config.layerIdsForGroup("hca_state").size(), 31u);
    EXPECT_EQ(config.layerIdsForGroup("swa_kv").size(), 61u);

    EXPECT_EQ(config.group("csa_kv").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group("hca_kv").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group("indexer_kv").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group("indexer_state").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("csa_state").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("hca_state").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("swa_kv").policy.group_type, CacheGroupType::SWA);
}

TEST_F(DSV4AllocatorTest, SpecBlockSizesMatchPoolSpecs) {
    auto config = makeDSV4AllocatorConfig();

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(config.group("csa_kv").spec->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.group("hca_kv").spec->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.group("indexer_kv").spec->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.group("indexer_state").spec->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(config.group("csa_state").spec->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(config.group("hca_state").spec->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(config.group("swa_kv").spec->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST_F(DSV4AllocatorTest, KVBlockStrideIsMaxAcrossGroups) {
    auto config = makeDSV4AllocatorConfig();

    // kv_block_stride_bytes should be the max block_size_bytes across all 7 pools
    size_t expected_max = 0;
    for (const auto& group : config.groups()) {
        expected_max = std::max(expected_max, group.spec->block_size_bytes());
    }
    EXPECT_EQ(config.group("hca_state").kvBlockStrideBytes(), expected_max);
    // HCA_STATE has the largest per-block bytes (128 entries * 1024 * 4)
    EXPECT_EQ(expected_max, config.group("hca_state").spec->block_size_bytes());
}

TEST_F(DSV4AllocatorTest, HCAStateIsExcludedFromReuseCachePolicy) {
    auto config = makeDSV4AllocatorConfig();
    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    ASSERT_EQ(config.topology().groups().size(), static_cast<size_t>(config.groupNums()));

    for (const auto& group : config.groups()) {
        if (group.tag == "hca_state") {
            EXPECT_EQ(group.policy.enable_prefix_reuse, false) << "HCA_STATE should skip reuse cache";
        } else {
            EXPECT_EQ(group.policy.enable_prefix_reuse, true) << "group " << group.tag;
        }
    }
}

// ============================================================
// Flash config: allocator integration
// ============================================================

TEST_F(DSV4AllocatorTest, FlashGroupTypes) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);

    // Flash: 21 CSA + 20 HCA + 2 SWA-only = 43 layers
    EXPECT_EQ(config.layerIdsForGroup("csa_kv").size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup("hca_kv").size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup("swa_kv").size(), 43u);

    EXPECT_EQ(config.group("csa_kv").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group("hca_kv").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group("indexer_kv").policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group("indexer_state").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("csa_state").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("hca_state").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("swa_kv").policy.group_type, CacheGroupType::SWA);
}

TEST_F(DSV4AllocatorTest, FlashAddressLookupAllGroups) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    for (const auto& group : config.groups()) {
        ASSERT_FALSE(config.layerIdsForGroup(group.tag).empty()) << "Flash group " << group.tag << " has no layers";
        int  layer_id = config.layerIdsForGroup(group.tag)[0];
        auto addr     = allocator->convertIndexToAddr(layer_id, group.tag, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr) << "Flash null kv_addr for group " << group.tag;
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

    EXPECT_EQ(config.layerIdsForGroup("csa_kv").size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup("hca_kv").size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup("indexer_kv").size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup("indexer_state").size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup("csa_state").size(), 21u);
    EXPECT_EQ(config.layerIdsForGroup("hca_state").size(), 20u);
    EXPECT_EQ(config.layerIdsForGroup("swa_kv").size(), 43u);
}

TEST_F(DSV4AllocatorTest, FlashSpecBlockSizes) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);

    ASSERT_EQ(static_cast<size_t>(config.groupNums()), 7u);
    EXPECT_EQ(config.group("csa_kv").spec->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.group("hca_kv").spec->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.group("indexer_kv").spec->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.group("swa_kv").spec->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST_F(DSV4AllocatorTest, FlashMallocAndFree) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto   block_pool  = allocator->groupBlockPools().front();
    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 5u);

    auto blocks = block_pool->malloc(5);
    ASSERT_TRUE(blocks.has_value());
    ASSERT_EQ(blocks->size(), 5u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 5);

    block_pool->incRef(*blocks);
    block_pool->decRef(*blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ============================================================
// Prefix cache: insertIntoCache skips HCA_STATE but keeps other groups reusable.
// ============================================================

TEST_F(DSV4AllocatorTest, InsertIntoCacheAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // Manually set up a BatchKVCacheResource with blocks for all 7 groups
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);

    CacheKeysType keys = {200, 201, 202, 203};
    batch_res->setBatchCacheKeys(0, keys);

    // Allocate 3 blocks per group (simulating 3 full blocks)
    // The allocator was constructed from this unchanged config; its full pool rows share these tags.
    for (size_t gid = 0; gid < config.groupTags().size(); ++gid) {
        const auto& block_pool = allocator->groupBlockPools()[gid];
        const auto& tag        = config.groupTags()[gid];
        auto        blocks     = block_pool->malloc(3);
        ASSERT_TRUE(blocks.has_value());
        ASSERT_EQ(blocks->size(), 3u);
        block_pool->incRef(*blocks);
        batch_res->mutableBlockIds(0, tag).assign(*blocks);
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
    {
        size_t resident_prefix_length = 0;
        allocator->insertIntoCache(insert_info, resident_prefix_length);
    }

    auto match = allocator->blockTreeCacheOwner()->match(CacheKeysType{200, 201, 202});
    EXPECT_EQ(match.matched_device_blocks, 3u);

    // HCA_STATE is runtime scratch state and must not be part of the declarative tree.
    for (const auto& group : config.groups()) {
        const auto& tag = group.tag;
        if (tag == "hca_state") {
            EXPECT_FALSE(containsReusableGroup(*allocator->blockTreeCacheOwner(), group.tag));
            continue;
        }
        EXPECT_EQ(allocator->blockTreeCacheOwner()->matchedBlocksForGroup(tag, match.matched_device_resources).size(),
                  group.policy.group_type == CacheGroupType::FULL ? 3u : 1u)
            << tag;
    }
    block_tree_cache_test::releaseRequestRefsForTest(*allocator->blockTreeCacheOwner(), match.matched_device_resources);

    // Free all blocks
    // The allocator was constructed from this unchanged config; its full pool rows share these tags.
    for (size_t gid = 0; gid < config.groupTags().size(); ++gid) {
        const auto& block_pool = allocator->groupBlockPools()[gid];
        const auto& tag        = config.groupTags()[gid];
        const auto& blocks     = batch_res->blocks(0, tag);
        block_pool->decRef(blocks);
    }
}

// ============================================================
// Prefix cache: Flash config insertIntoCache skips HCA_STATE.
// ============================================================

TEST_F(DSV4AllocatorTest, FlashInsertIntoCacheAllGroups) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<TestDSV4HybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    initDsv4BatchGroups(*batch_res, config);

    CacheKeysType keys = {300, 301, 302, 303};
    batch_res->setBatchCacheKeys(0, keys);

    // The allocator was constructed from this unchanged config; its full pool rows share these tags.
    for (size_t gid = 0; gid < config.groupTags().size(); ++gid) {
        const auto& block_pool = allocator->groupBlockPools()[gid];
        const auto& tag        = config.groupTags()[gid];
        auto        blocks     = block_pool->malloc(3);
        ASSERT_TRUE(blocks.has_value());
        ASSERT_EQ(blocks->size(), 3u);
        block_pool->incRef(*blocks);
        batch_res->mutableBlockIds(0, tag).assign(*blocks);
    }

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    int  total_tokens               = 3 * seq_size_per_block + 1;
    generate_input->input_ids       = torch::arange(total_tokens, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    InsertInfo insert_info{batch_res, complete_token_ids, /*is_resident=*/false};
    {
        size_t resident_prefix_length = 0;
        allocator->insertIntoCache(insert_info, resident_prefix_length);
    }

    auto match = allocator->blockTreeCacheOwner()->match(CacheKeysType{300, 301, 302});
    EXPECT_EQ(match.matched_device_blocks, 3u);

    for (const auto& group : config.groups()) {
        const auto& tag = group.tag;
        if (tag == "hca_state") {
            EXPECT_FALSE(containsReusableGroup(*allocator->blockTreeCacheOwner(), group.tag));
            continue;
        }
        EXPECT_EQ(allocator->blockTreeCacheOwner()->matchedBlocksForGroup(tag, match.matched_device_resources).size(),
                  group.policy.group_type == CacheGroupType::FULL ? 3u : 1u)
            << tag;
    }
    block_tree_cache_test::releaseRequestRefsForTest(*allocator->blockTreeCacheOwner(), match.matched_device_resources);

    // The allocator was constructed from this unchanged config; its full pool rows share these tags.
    for (size_t gid = 0; gid < config.groupTags().size(); ++gid) {
        const auto& block_pool = allocator->groupBlockPools()[gid];
        const auto& tag        = config.groupTags()[gid];
        block_pool->decRef(batch_res->blocks(0, tag));
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
    info.enable_cache_lookup = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "Prefix cache reuse should work with paged DSV4 groups";

    for (const auto& group : config.groups()) {
        const auto& out_blocks = batch_res->blocks(0, group.tag);
        ASSERT_GE(out_blocks.size(), 3u) << group.tag;
        if (group.policy.group_type == CacheGroupType::FULL) {
            const auto& cached_blocks = seeded.blocks_by_tag.at(group.tag);
            EXPECT_EQ(out_blocks[0], cached_blocks[0]) << group.tag;
            EXPECT_EQ(out_blocks[1], cached_blocks[1]) << group.tag;
            continue;
        }
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1])) << group.tag;
        if (group.tag == "hca_state") {
            EXPECT_TRUE(isNullBlockIdx(out_blocks[2])) << "HCA_STATE should not reuse a cached tail block";
            continue;
        }
        EXPECT_EQ(out_blocks[2], seeded.blocks_by_tag.at(group.tag)[2]) << group.tag;
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

    for (const auto& group_set : allocator->blockTreeCacheOwner()->groupSets()) {
        if (group_set->groupType() == CacheGroupType::FULL) {
            continue;
        }
        ASSERT_FALSE(group_set->groupTags().empty());
        for (size_t path_index = 0; path_index < cached_keys.size(); ++path_index) {
            ASSERT_GT(allocator->blockTreeCacheOwner()->evictForGroup(group_set->groupTags().front(), 1), 0)
                << "path_index=" << path_index;
        }
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
    info.enable_cache_lookup = true;
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
    info.enable_cache_lookup = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "HCA_STATE miss should not veto DSV4 prefix reuse";

    EXPECT_TRUE(isNullBlockIdx(batch_res->blocks(0, "hca_state").at(2))) << "HCA_STATE should remain non-reused";
    EXPECT_EQ(batch_res->blocks(0, "swa_kv").at(2), seeded.blocks_by_tag.at("swa_kv")[2])
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

    // Leave only the latest coordinate resident for each fixed-tail group set.
    for (const auto& group_set : allocator->blockTreeCacheOwner()->groupSets()) {
        if (group_set->groupType() == CacheGroupType::FULL) {
            continue;
        }
        ASSERT_FALSE(group_set->groupTags().empty());
        for (size_t path_index = 1; path_index < cached_keys.size(); ++path_index) {
            ASSERT_GT(allocator->blockTreeCacheOwner()->evictForGroup(group_set->groupTags().front(), 1), 0)
                << "path_index=" << path_index;
        }
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
    info.enable_cache_lookup = true;
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
    info.enable_cache_lookup = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "Flash prefix cache reuse should work for paged groups";

    for (const auto& group : config.groups()) {
        const auto& out_blocks = batch_res->blocks(0, group.tag);
        ASSERT_GE(out_blocks.size(), 3u) << group.tag;
        if (group.policy.group_type == CacheGroupType::FULL) {
            EXPECT_EQ(out_blocks[0], seeded.blocks_by_tag.at(group.tag)[0]) << group.tag;
            continue;
        }
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1])) << group.tag;
        if (group.tag == "hca_state") {
            EXPECT_TRUE(isNullBlockIdx(out_blocks[2])) << "Flash HCA_STATE should not reuse a cached tail block";
            continue;
        }
        EXPECT_EQ(out_blocks[2], seeded.blocks_by_tag.at(group.tag)[2]) << group.tag;
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
    info.enable_cache_lookup = false;
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
    auto                  config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);
    std::vector<uint32_t> block_nums;
    for (const auto& tag : config.groupTags()) {
        block_nums.push_back(tag == "hca_state" ? 11 : 40u);
    }
    setGroupBlockNumsForTest(config, config.groupTags(), block_nums);

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
    info.enable_cache_lookup = false;
    info.reuse_cache         = false;
    info.verbose             = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// DEV's broader variant of the test above: DEV applied one global fixed-pool block count to every
// fixed region, so the reserve had to leave *all four* explicitly-sized pools at full capacity, not
// just hca_state.  Kept alongside the hca_state-only case because it exercises a different shape:
// four independently explicit pools, including swa_kv which spans every layer.
TEST_F(DSV4AllocatorTest, HybridPoolReserveBlocksDoNotReduceExplicitFixedPoolCapacity) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    for (const auto& tag : dsv4StateSwaTags()) {
        setDsv4ExplicitPoolBlocks(mc, tag, 11);
    }
    auto                  config = CacheConfigCreator::createWarmupConfig(mc, pc, 0);
    std::vector<uint32_t> block_nums;
    for (const auto& tag : config.groupTags()) {
        block_nums.push_back(config.group(tag).policy.group_type == CacheGroupType::SWA ? 11 : 40u);
    }
    setGroupBlockNumsForTest(config, config.groupTags(), block_nums);

    auto allocator =
        std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE, nullptr, /*reserve_block_ratio=*/50);
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
    info.enable_cache_lookup = false;
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
    info.enable_cache_lookup = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.reuse_len, 0);

    const auto& swa_out = batch_res->blocks(0, "swa_kv");
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
    init_info.enable_cache_lookup = false;
    auto init_result              = allocator->malloc(init_info);
    ASSERT_TRUE(init_result.success);

    // All 7 groups should have 1 block each
    for (const auto& group : config.groups()) {
        EXPECT_EQ(batch_res->blocksNum(0, group.tag), 1u) << "group " << group.tag << " should have 1 block after init";
    }

    size_t free_after_init = allocator->freeBlocksNum();

    // incrMalloc: grow to 2 blocks
    cti->setSeqLength(2 * spb);
    MallocInfo incr_info{batch_res, cti};
    incr_info.enable_cache_lookup = false;
    auto incr_result              = allocator->malloc(incr_info);
    ASSERT_TRUE(incr_result.success);

    // All 7 groups should now have 2 blocks each
    for (const auto& group : config.groups()) {
        EXPECT_EQ(batch_res->blocksNum(0, group.tag), 2u)
            << "group " << group.tag << " should have 2 blocks after incr";
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
    info.enable_cache_lookup = false;
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
    info2.enable_cache_lookup = false;
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
    init_info.enable_cache_lookup = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);

    for (const auto& group : config.groups()) {
        EXPECT_EQ(batch_res->blocksNum(0, group.tag), 1u) << "Flash group " << group.tag;
    }

    // Grow to 3 blocks
    cti->setSeqLength(3 * spb);
    MallocInfo incr_info{batch_res, cti};
    incr_info.enable_cache_lookup = false;
    ASSERT_TRUE(allocator->malloc(incr_info).success);

    for (const auto& group : config.groups()) {
        EXPECT_EQ(batch_res->blocksNum(0, group.tag), 3u) << "Flash group " << group.tag << " after incr";
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

namespace {
ModelConfig makeCreationModel(uint32_t layers = 1, uint32_t head_size = 2) {
    ModelConfig model;
    model.num_layers                          = layers;
    model.max_seq_len                         = 64;
    model.data_type                           = DataType::TYPE_FP16;
    model.attn_config.head_num                = 1;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = head_size;
    model.attn_config.tokens_per_block        = 4;
    model.attn_config.kernel_tokens_per_block = 4;
    KVCacheSpecDesc desc;
    desc.tag = "full";
    model.kv_cache_spec_descs.assign(layers, {desc});
    return model;
}

void addCreationFixedPool(ModelConfig& model, bool host, uint32_t blocks = 7) {
    KVCacheSpecDesc desc;
    desc.tag                              = "state";
    desc.cache_type                       = KVCacheSpecType::OpaqueState;
    desc.is_state_cache                   = true;
    desc.entry_elems                      = 3;
    desc.entry_dtype                      = DataType::TYPE_FP16;
    desc.explicit_entry_count             = 1;
    desc.capacity                         = CacheCapacityPolicyDesc{};
    desc.capacity->explicit_block_num     = blocks;
    desc.capacity->charge_to_paged_budget = !host;
    desc.memory                           = CacheMemoryPolicyDesc{};
    desc.memory->placement                = host ? CacheMemoryPlacement::HOST_PINNED : CacheMemoryPlacement::DEVICE;
    for (auto& layer : model.kv_cache_spec_descs) {
        layer.push_back(desc);
    }
}
}  // namespace

TEST(CacheConfigCreatorTest, LayoutAndBudgetCalculationLeaveGroupCapacityUnresolved) {
    KVCacheConfig options;
    options.test_block_num = 13;
    auto config            = CacheConfigCreator::createConfig(makeCreationModel(), {}, options);
    EXPECT_EQ(config.group("full").block_num, 0u);
    const auto*    layout = &config.topology();
    const uint32_t candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(config, makeCreationModel(), {}, options, {});
    EXPECT_EQ(candidate_block_num, 13u);
    EXPECT_EQ(&config.topology(), layout);
    EXPECT_EQ(config.group("full").block_num, 0u);
    const auto spec = config.group("full").spec;
    EXPECT_EQ(publishedGroupTags(config.topology()), std::vector<std::string>({"full"}));
    EXPECT_EQ(config.group("full").seqSizePerBlock(), 4u);
    EXPECT_EQ(config.group("full").kernelSeqSizePerBlock(), 4u);

    config.finalizeBlockNums(7, {});
    EXPECT_EQ(config.group("full").block_num, 7u);
    EXPECT_EQ(candidate_block_num, 13u);
    EXPECT_EQ(config.group("full").spec, spec);
    EXPECT_EQ(config.layerIdsForGroup("full"), std::vector<int>({0}));
    EXPECT_ANY_THROW(config.finalizeBlockNums(0, {}));
}

TEST(CacheConfigCreatorTest, WarmupReturnsCompleteExplicitPoolCapacityWithoutBudgeting) {
    auto model = makeCreationModel();
    addCreationFixedPool(model, true);
    KVCacheConfig options;
    options.test_block_num = 99;
    options.linear_step    = 4;
    auto config            = CacheConfigCreator::createWarmupConfig(model, {}, options);
    EXPECT_EQ(config.group("full").block_num, 2u);
    EXPECT_EQ(config.group("state").block_num, 7u);
    EXPECT_TRUE(config.mtp_sub_configs.empty());
    for (const auto& group : config.groups()) {
        EXPECT_GT(DeviceBlockPoolConfigHelper::createConfigForGroup(config, group).total_size_bytes, 0u);
    }
}

TEST(CacheConfigCreatorTest, MergedBudgetUsesEachPhysicalSegmentOnce) {
    for (const auto group_type : {CacheGroupType::FULL, CacheGroupType::SWA}) {
        for (const bool host : {false, true}) {
            auto target = makeCreationModel(2, 2);
            auto draft  = makeCreationModel(2, 4);
            for (auto* model : {&target, &draft}) {
                for (auto& layer : model->kv_cache_spec_descs) {
                    layer[0].group_type = group_type;
                }
            }
            addCreationFixedPool(target, host);
            KVCacheConfig options;
            options.kv_cache_mem_mb = 1;
            options.linear_step     = 3;
            SpeculativeExecutionConfig sp;
            sp.type              = SP_TYPE_MTP;
            sp.gen_num_per_cycle = 2;

            auto created_config = CacheConfigCreator::createConfig(target, {}, options, sp, &draft, true, false);
            const uint32_t config_candidate_block_num =
                CacheConfigCreator::computeLocalBlockNum(created_config, target, {}, options, {}, std::nullopt, sp);
            auto config = finalizeCacheConfig(created_config, config_candidate_block_num);
            ASSERT_EQ(config.mtp_sub_configs.size(), 2u);

            const auto pool = DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.group("full"));
            ASSERT_EQ(pool.memory_layouts.size(), 3u);
            EXPECT_EQ(pool.memory_layouts[1].kv_block_stride_bytes, 2 * pool.memory_layouts[0].kv_block_stride_bytes);
            for (const auto& sub : config.mtp_sub_configs) {
                EXPECT_TRUE(sub->layerIdsForGroup("state").empty());
                EXPECT_EQ(sub->group("full").block_num, config.group("full").block_num);
            }
            const auto charged_bytes = [&](const CacheConfig& candidate) {
                size_t bytes = 0;
                for (const auto& group : candidate.groups()) {
                    const auto policy = group.policy;
                    if (policy.explicit_block_num == 0 || policy.charge_to_paged_budget) {
                        bytes += DeviceBlockPoolConfigHelper::createConfigForGroup(candidate, group).total_size_bytes;
                    }
                }
                return bytes;
            };
            constexpr size_t budget_bytes = 1024 * 1024;
            EXPECT_LE(charged_bytes(config), budget_bytes);
            const auto baseline_blocks = config_candidate_block_num;
            config.finalizeBlockNums(baseline_blocks + 1, {});
            EXPECT_GT(charged_bytes(config), budget_bytes);
            config.finalizeBlockNums(7, {});
            EXPECT_EQ(config.group("full").block_num, group_type == CacheGroupType::SWA ? 3u : 7u);
            EXPECT_EQ(config.group("state").block_num, 7u);
        }
    }
}

TEST(CacheConfigCreatorTest, MtpModuleCountsPreserveEagleAndDsparkRules) {
    auto          target = makeCreationModel();
    auto          draft  = makeCreationModel(2);
    KVCacheConfig options;
    options.test_block_num = 5;
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_MTP;
    sp.gen_num_per_cycle = 3;

    auto mtp = CacheConfigCreator::createConfig(target, {}, options, sp, &draft, true);
    EXPECT_EQ(mtp.mtp_sub_configs.size(), 3u);
    EXPECT_EQ(mtp.layer_all_num(), 7u);

    auto eagle = CacheConfigCreator::createConfig(target, {}, options, sp, &draft, true, true);
    EXPECT_EQ(eagle.mtp_sub_configs.size(), 1u);
    EXPECT_EQ(eagle.layer_all_num(), 3u);
    sp.type = SP_TYPE_DSPARK;

    auto dspark = CacheConfigCreator::createConfig(target, {}, options, sp, &draft, true);
    EXPECT_EQ(dspark.mtp_sub_configs.size(), 1u);
    EXPECT_EQ(dspark.layer_all_num(), 3u);

    auto ordinary_draft = CacheConfigCreator::createConfig(target, {}, options, sp, &draft);
    EXPECT_EQ(ordinary_draft.mtp_sub_configs.size(), 1u);
}

TEST(CacheConfigCreatorTest, MtpRejectsUnknownDraftTagsButAllowsUnusedTargetGroups) {
    auto target = makeCreationModel();
    auto draft  = target;
    addCreationFixedPool(draft, false);
    KVCacheConfig options;
    options.test_block_num = 5;
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_MTP;
    sp.gen_num_per_cycle = 1;
    try {
        (void)CacheConfigCreator::createConfig(target, {}, options, sp, &draft, true);
        FAIL() << "expected the draft-only state group to be rejected";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("unmapped draft cache group tag=state"), std::string::npos);
    }

    // The reverse relation is valid: target-only pools retain empty child views.
    std::swap(target, draft);

    auto           created_config = CacheConfigCreator::createConfig(target, {}, options, sp, &draft, true);
    const uint32_t config_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_config, target, {}, options, {}, std::nullopt, sp);
    const auto config = finalizeCacheConfig(created_config, config_candidate_block_num);
    ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
    EXPECT_EQ(publishedGroupTags(config.mtp_sub_configs[0]->topology()), publishedGroupTags(config.topology()));

    EXPECT_TRUE(config.mtp_sub_configs[0]->layerIdsForGroup(config.mtp_sub_configs[0]->group("state").tag).empty());
    EXPECT_EQ(DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.group("state")).memory_layouts.size(),
              1u);
}

TEST(CacheConfigCreatorTest, MtpRejectsIncompatiblePoolPolicyAndTokenSpans) {
    auto          target = makeCreationModel();
    KVCacheConfig options;
    options.test_block_num = 5;
    SpeculativeExecutionConfig sp;
    sp.type                                                      = SP_TYPE_MTP;
    sp.gen_num_per_cycle                                         = 1;
    auto draft                                                   = target;
    draft.kv_cache_spec_descs[0][0].capacity                     = CacheCapacityPolicyDesc{};
    draft.kv_cache_spec_descs[0][0].capacity->explicit_block_num = 2;
    EXPECT_ANY_THROW(CacheConfigCreator::createConfig(target, {}, options, sp, &draft, true));
    draft                              = target;
    draft.attn_config.tokens_per_block = 8;
    EXPECT_ANY_THROW(CacheConfigCreator::createConfig(target, {}, options, sp, &draft, true));
    draft                                     = target;
    draft.attn_config.kernel_tokens_per_block = 2;
    EXPECT_ANY_THROW(CacheConfigCreator::createConfig(target, {}, options, sp, &draft, true));
}

TEST(CacheConfigCreatorTest, ExplicitOnlyAutomaticBudgetRejectsUndefinedBaselineButOverrideWorks) {
    auto  model                           = makeCreationModel();
    auto& desc                            = model.kv_cache_spec_descs[0][0];
    desc.capacity                         = CacheCapacityPolicyDesc{};
    desc.capacity->explicit_block_num     = 7;
    desc.capacity->charge_to_paged_budget = true;
    KVCacheConfig options;
    options.kv_cache_mem_mb = 1;
    const auto layout       = CacheConfigCreator::createConfig(model, {}, options);
    EXPECT_ANY_THROW(CacheConfigCreator::computeLocalBlockNum(layout, model, {}, options, {}));
    options.test_block_num = 3;

    auto           created_config = CacheConfigCreator::createConfig(model, {}, options);
    const uint32_t config_candidate_block_num =
        CacheConfigCreator::computeLocalBlockNum(created_config, model, {}, options, {});
    auto config = finalizeCacheConfig(created_config, config_candidate_block_num);
    EXPECT_EQ(config_candidate_block_num, 3u);
    EXPECT_EQ(config.group("full").block_num, 7u);
}

TEST(CacheConfigCreatorTest, RejectsInvalidMtpBudgetSegmentsBeforeIndexing) {
    auto model = makeCreationModel();
    addCreationFixedPool(model, false, 4);
    KVCacheConfig options;
    options.kv_cache_mem_mb = 1;
    auto layout             = CacheConfigCreator::createConfig(model, {}, options);
    ASSERT_EQ(layout.groupNums(), 2);

    layout.mtp_sub_configs.push_back(nullptr);
    EXPECT_ANY_THROW(CacheConfigCreator::computeLocalBlockNum(layout, model, {}, options, {}));

    auto sub = std::make_shared<CacheConfig>();
    sub->layer_num = 1;
    sub->setTopology({layout.group("full")}, {{0, {"full"}}});
    layout.mtp_sub_configs.back() = sub;
    EXPECT_ANY_THROW(CacheConfigCreator::computeLocalBlockNum(layout, model, {}, options, {}));
}

TEST(CacheConfigCreatorTest, ExplicitPoolBudgetMultiplicationOverflowIsRejected) {
    auto model = makeCreationModel();
    addCreationFixedPool(model, false, 4);
    model.kv_cache_spec_descs[0][1].block_stride_bytes_override = std::numeric_limits<size_t>::max() / 2;
    // The byte-stride override itself is valid; only multiplying by pool
    // capacity exceeds size_t. Warmup builds the same Spec without a budget.
    const auto warmup = CacheConfigCreator::createWarmupConfig(model, {});
    EXPECT_EQ(warmup.group("state").kvBlockStrideBytes(), std::numeric_limits<size_t>::max() / 2);
    KVCacheConfig options;
    options.kv_cache_mem_mb = 1;
    try {
        const auto layout = CacheConfigCreator::createConfig(model, {}, options);
        (void)CacheConfigCreator::computeLocalBlockNum(layout, model, {}, options, {});
        FAIL() << "expected explicit-pool byte-budget overflow";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("kv cache budget overflow"), std::string::npos);
    }
}

}  // namespace test
}  // namespace rtp_llm
