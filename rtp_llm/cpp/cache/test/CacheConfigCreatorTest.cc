#include <set>
#include <gtest/gtest.h>

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {
namespace {

constexpr int kTestBlockNum = 8;

ModelConfig makeMhaModel(int64_t layer_num = 2, const std::string& tag = "default") {
    ModelConfig config;
    config.num_layers                   = layer_num;
    config.data_type                    = DataType::TYPE_FP16;
    config.attn_config.head_num         = 4;
    config.attn_config.kv_head_num      = 2;
    config.attn_config.size_per_head    = 8;
    config.attn_config.tokens_per_block = 4;
    config.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;
    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    config.kv_cache_spec_descs.assign(static_cast<size_t>(layer_num), {desc});
    return config;
}

ModelConfig makeMlaModel() {
    auto config                                 = makeMhaModel();
    config.attn_config.use_mla                  = true;
    config.attn_config.kv_lora_rank             = 8;
    config.attn_config.rope_head_dim            = 4;
    config.kv_cache_spec_descs[0][0].tag        = "mla";
    config.kv_cache_spec_descs[0][0].cache_type = KVCacheSpecType::MultiHeadLatentAttention;
    config.kv_cache_spec_descs[1]               = config.kv_cache_spec_descs[0];
    return config;
}

ModelConfig makeSparseMlaModel(int64_t layer_num = 2) {
    auto config                                                      = makeMlaModel();
    config.num_layers                                                = layer_num;
    config.attn_config.is_sparse                                     = true;
    config.attn_config.indexer_head_dim                              = 128;
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    config.kv_cache_spec_descs.assign(static_cast<size_t>(layer_num), {});

    KVCacheSpecDesc default_desc;
    default_desc.tag        = "default";
    default_desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;

    KVCacheSpecDesc indexer_desc;
    indexer_desc.tag                  = "indexer_kv";
    indexer_desc.cache_type           = KVCacheSpecType::OpaqueKV;
    indexer_desc.entry_dtype          = DataType::TYPE_UINT8;
    indexer_desc.entry_elems          = 132;
    indexer_desc.explicit_entry_count = 4;
    for (auto& layer_descs : config.kv_cache_spec_descs) {
        layer_descs = {default_desc, indexer_desc};
    }
    return config;
}

ModelConfig makeKimiModel() {
    auto config                                            = makeMhaModel(/*layer_num=*/4);
    config.hybrid_attention_config.enable_hybrid_attention = true;
    config.hybrid_attention_config.hybrid_attention_types  = {
        HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    config.linear_attention_config.linear_conv_kernel_dim = 4;
    config.linear_attention_config.linear_key_head_dim    = 16;
    config.linear_attention_config.linear_value_head_dim  = 16;
    config.linear_attention_config.linear_num_key_heads   = 2;
    config.linear_attention_config.linear_num_value_heads = 2;
    config.attn_config.size_per_head                      = 32;
    config.attn_config.tokens_per_block                   = 8;
    setHybridAttentionKvCacheSpecs(config);
    return config;
}

ModelConfig makeQwenHybridModel() {
    auto config                                                      = makeKimiModel();
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    // Qwen's retained hybrid descriptor shape is a FULL/LINEAR interleave;
    // tags are business identity and intentionally differ from Kimi's names.
    for (auto& layer_descs : config.kv_cache_spec_descs) {
        for (auto& desc : layer_descs) {
            desc.tag = desc.cache_type == KVCacheSpecType::LinearAttention ? "qwen_linear" : "qwen_full";
        }
    }
    return config;
}

ModelConfig makeDsv4Model() {
    ModelConfig config;
    config.num_layers                                                = 2;
    config.data_type                                                 = DataType::TYPE_FP16;
    config.attn_config.head_num                                      = 128;
    config.attn_config.kv_head_num                                   = 1;
    config.attn_config.size_per_head                                 = 512;
    config.attn_config.indexer_head_dim                              = 128;
    config.attn_config.tokens_per_block                              = 128;
    config.hybrid_attention_config.enable_hybrid_attention           = true;
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(config, {128, 4});
    return config;
}

KVCacheConfig fixedBlockConfig() {
    KVCacheConfig config;
    config.test_block_num = kTestBlockNum;
    return config;
}

CacheConfig createFinalConfig(const ModelConfig& model_config) {
    return CacheConfigCreator::createConfig(model_config, ParallelismConfig{}, RuntimeConfig{}, fixedBlockConfig());
}

std::string runtimeErrorMessage(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

TEST(CacheConfigCreatorTest, ClosedFormBlockBudgetCoversBoundariesAndWideArithmetic) {
    KVCacheBlockBudget budget;
    budget.explicit_pool_reserve_bytes = 7;
    budget.paged_block_bytes           = 3;
    budget.swa_block_bytes             = 5;

    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/6, budget, /*linear_step=*/4), 0u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/24, budget, /*linear_step=*/4), 4u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/14, budget, /*linear_step=*/4), 0u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/15, budget, /*linear_step=*/4), 1u);
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/23, budget, /*linear_step=*/1), 2u);

    KVCacheBlockBudget swa_only;
    swa_only.swa_block_bytes = 2;
    EXPECT_EQ(maxKVCacheBlockNumForBudget(/*total_budget_bytes=*/10, swa_only, /*linear_step=*/4), 20u);

    KVCacheBlockBudget paged_only;
    paged_only.paged_block_bytes = 1;
    EXPECT_EQ(
        maxKVCacheBlockNumForBudget(std::numeric_limits<size_t>::max(), paged_only, std::numeric_limits<int>::max()),
        std::numeric_limits<uint32_t>::max());

    KVCacheBlockBudget wide_inputs;
    wide_inputs.paged_block_bytes = std::numeric_limits<size_t>::max();
    wide_inputs.swa_block_bytes   = std::numeric_limits<size_t>::max();
    EXPECT_EQ(
        maxKVCacheBlockNumForBudget(std::numeric_limits<size_t>::max(), wide_inputs, std::numeric_limits<int>::max()),
        0u);

    EXPECT_THROW(maxKVCacheBlockNumForBudget(1, KVCacheBlockBudget{}, 1), std::exception);
}

TEST(CacheConfigCreatorTest, ExplicitPoolReservationMustLeavePagedBudget) {
    ModelConfig model_config;
    model_config.num_layers = 1;
    model_config.data_type  = DataType::TYPE_FP16;

    KVCacheSpecDesc desc;
    desc.tag                          = "state";
    desc.cache_type                   = KVCacheSpecType::OpaqueState;
    desc.entry_elems                  = 1;
    desc.entry_dtype                  = DataType::TYPE_UINT8;
    desc.explicit_entry_count         = 1;
    desc.block_stride_bytes_override  = 1024 * 1024;
    desc.group_type                   = CacheGroupType::SWA;
    desc.capacity                     = CacheCapacityPolicyDesc{};
    desc.capacity->explicit_block_num = 1;
    model_config.kv_cache_spec_descs  = {{desc}};

    KVCacheConfig kv_cache_config;
    kv_cache_config.kv_cache_mem_mb = 1;

    const auto message = runtimeErrorMessage([&]() {
        (void)CacheConfigCreator::createConfig(model_config, ParallelismConfig{}, RuntimeConfig{}, kv_cache_config);
    });
    EXPECT_NE(message.find("must be greater than explicitly-sized pool reservation"), std::string::npos);
}

TEST(CacheConfigCreatorTest, BasicConfigMaterializesResolvedGeometryAndExplicitReserveWithoutCapacityPrepass) {
    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 256;
    kv_cache_config.kernel_seq_size_per_block = 128;

    const auto config = CacheConfigCreator::createBasicConfig(makeDsv4Model(), ParallelismConfig{}, kv_cache_config, 0);

    EXPECT_EQ(config.block_num, 0);
    EXPECT_EQ(config.seq_size_per_block, 256u);
    EXPECT_GT(config.explicitlySizedPoolReserveBytes(), 0u);
    EXPECT_EQ(config.group("hca_state").block_num, 0u);
    EXPECT_EQ(config.group("hca_state").seqSizePerBlock(), 256u);
    EXPECT_EQ(config.group("hca_state").kernelSeqSizePerBlock(), 128u);
}

TEST(CacheConfigCreatorTest, BasicSingleConfigUsesTheUnifiedPhysicalAndKernelOverrides) {
    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 256;
    kv_cache_config.kernel_seq_size_per_block = 128;

    auto model                                                      = makeMhaModel();
    model.hybrid_attention_config.enable_independent_kv_cache_pools = false;
    const auto config = CacheConfigCreator::createBasicConfig(model, ParallelismConfig{}, kv_cache_config, 0);

    ASSERT_EQ(config.groupNums(), 1);
    EXPECT_EQ(config.seq_size_per_block, 256u);
    EXPECT_EQ(config.group("default").seqSizePerBlock(), 256u);
    EXPECT_EQ(config.group("default").kernelSeqSizePerBlock(), 128u);
    EXPECT_EQ(config.group("default").block_num, 0u);
}

TEST(CacheConfigCreatorTest, CreateConfigLowersOrdinaryMhaToFixedFinalRecord) {
    const CacheSemanticSnapshot expected = {{"default",
                                             KVCacheSpecType::MultiHeadAttention,
                                             CacheGroupType::FULL,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             0,
                                             true,
                                             CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                             CpBlockSliceMode::NONE,
                                             {0, 1},
                                             kTestBlockNum,
                                             4,
                                             4,
                                             512,
                                             256,
                                             0}};

    const auto config = createFinalConfig(makeMhaModel());
    EXPECT_EQ(snapshotCacheConfig(config), expected);
}

TEST(CacheConfigCreatorTest, CreateConfigLowersOrdinaryMlaToFixedFinalRecord) {
    const CacheSemanticSnapshot expected = {{"mla",
                                             KVCacheSpecType::MultiHeadLatentAttention,
                                             CacheGroupType::FULL,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             0,
                                             true,
                                             CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                             CpBlockSliceMode::NONE,
                                             {0, 1},
                                             kTestBlockNum,
                                             4,
                                             4,
                                             192,
                                             96,
                                             0}};

    const auto config = createFinalConfig(makeMlaModel());
    EXPECT_EQ(snapshotCacheConfig(config), expected);
}

TEST(CacheConfigCreatorTest, DescLoweringKeepsTagSeparateFromLayoutFingerprint) {
    auto              model_config = makeMhaModel(/*layer_num=*/1);
    ParallelismConfig parallelism_config;
    SpecBuildContext  ctx;
    ctx.dtype                     = DataType::TYPE_FP16;
    ctx.seq_size_per_block        = 4;
    ctx.kernel_seq_size_per_block = 2;
    ctx.attn_config               = &model_config.attn_config;
    ctx.linear_attention_config   = &model_config.linear_attention_config;
    ctx.parallelism_config        = &parallelism_config;

    KVCacheSpecDesc first_desc  = model_config.kv_cache_spec_descs[0][0];
    KVCacheSpecDesc second_desc = first_desc;
    first_desc.tag              = "first";
    second_desc.tag             = "second";

    const auto first  = SpecBuilder::build(first_desc, ctx);
    const auto second = SpecBuilder::build(second_desc, ctx);

    EXPECT_EQ(first.tag, "first");
    EXPECT_EQ(second.tag, "second");
    EXPECT_EQ(first.spec->fingerprint(), second.spec->fingerprint());
    EXPECT_EQ(first.spec->debugString().find("tag="), std::string::npos);
    EXPECT_EQ(first.spec->seq_size_per_block, 4u);
    EXPECT_EQ(first.spec->kernel_seq_size_per_block, 2u);
}

TEST(CacheConfigCreatorTest, SpecBuilderDerivesPhysicalSpanFromCpMapping) {
    auto              model_config = makeMhaModel(/*layer_num=*/1);
    ParallelismConfig parallelism_config;
    parallelism_config.role_type                          = RoleType::PREFILL;
    parallelism_config.tp_size                            = 2;
    parallelism_config.prefill_cp_config.kv_cache_sharded = true;

    SpecBuildContext ctx;
    ctx.dtype                     = DataType::TYPE_FP16;
    ctx.seq_size_per_block        = 128;
    ctx.kernel_seq_size_per_block = 64;
    ctx.attn_config               = &model_config.attn_config;
    ctx.parallelism_config        = &parallelism_config;

    KVCacheSpecDesc none_desc = model_config.kv_cache_spec_descs[0][0];
    none_desc.tag             = "none";
    none_desc.cp              = CacheCpPolicyDesc{};
    none_desc.cp->mapping     = CpBlockMappingMode::NONE;

    KVCacheSpecDesc round_robin_desc = none_desc;
    round_robin_desc.tag             = "round_robin";
    round_robin_desc.cp->mapping     = CpBlockMappingMode::BLOCK_ROUND_ROBIN;

    KVCacheSpecDesc compact_desc = none_desc;
    compact_desc.tag             = "compact";
    compact_desc.cp->mapping     = CpBlockMappingMode::COMPACT_LAST_RANK;

    const auto none        = SpecBuilder::build(none_desc, ctx).spec;
    const auto round_robin = SpecBuilder::build(round_robin_desc, ctx).spec;
    const auto compact     = SpecBuilder::build(compact_desc, ctx).spec;

    EXPECT_EQ(none->seq_size_per_block, 128u);
    EXPECT_EQ(round_robin->seq_size_per_block, 128u);
    EXPECT_EQ(compact->seq_size_per_block, 256u);
    EXPECT_EQ(none->kernel_seq_size_per_block, 64u);
    EXPECT_EQ(round_robin->kernel_seq_size_per_block, 64u);
    EXPECT_EQ(compact->kernel_seq_size_per_block, 64u);

    parallelism_config.prefill_cp_config.kv_cache_sharded = false;
    const auto inactive_compact                           = SpecBuilder::build(compact_desc, ctx).spec;
    EXPECT_EQ(inactive_compact->seq_size_per_block, 128u);
    EXPECT_EQ(inactive_compact->kernel_seq_size_per_block, 64u);
}

TEST(CacheConfigCreatorTest, SpecFingerprintIncludesKernelGeometry) {
    auto              model_config = makeMhaModel(/*layer_num=*/1);
    ParallelismConfig parallelism_config;
    SpecBuildContext  ctx;
    ctx.dtype                     = DataType::TYPE_FP16;
    ctx.seq_size_per_block        = 128;
    ctx.kernel_seq_size_per_block = 64;
    ctx.attn_config               = &model_config.attn_config;
    ctx.parallelism_config        = &parallelism_config;

    const auto desc               = model_config.kv_cache_spec_descs[0][0];
    const auto first              = SpecBuilder::build(desc, ctx).spec;
    ctx.kernel_seq_size_per_block = 32;
    const auto second             = SpecBuilder::build(desc, ctx).spec;

    EXPECT_EQ(first->seq_size_per_block, second->seq_size_per_block);
    EXPECT_NE(first->kernel_seq_size_per_block, second->kernel_seq_size_per_block);
    EXPECT_NE(first->fingerprint(), second->fingerprint());
    EXPECT_EQ(first->clone()->kernel_seq_size_per_block, first->kernel_seq_size_per_block);
}

TEST(CacheConfigCreatorTest, CreateConfigIsolatesSparseMlaIndexerPoolAndStride) {
    const auto config = createFinalConfig(makeSparseMlaModel());

    ASSERT_EQ(groupTagSet(config), (std::set<std::string>{"default", "indexer_kv"}));
    const std::string default_tag = "default";
    const std::string indexer_tag = "indexer_kv";
    EXPECT_EQ(config.group(default_tag).policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group(indexer_tag).policy.group_type, CacheGroupType::FULL);
    EXPECT_EQ(config.group(default_tag).spec->type, KVCacheSpecType::MultiHeadLatentAttention);
    EXPECT_EQ(config.group(indexer_tag).spec->type, KVCacheSpecType::OpaqueKV);
    EXPECT_EQ(config.group(default_tag).kv_block_stride_bytes, 96u);
    EXPECT_EQ(config.group(default_tag).kv_scale_stride_bytes, 0u);
    EXPECT_EQ(config.group(indexer_tag).kv_block_stride_bytes, 528u);
    EXPECT_EQ(config.group(indexer_tag).kv_scale_stride_bytes, 0u);
    EXPECT_EQ(config.group(default_tag).block_num, kTestBlockNum);
    EXPECT_EQ(config.group(indexer_tag).block_num, kTestBlockNum);
    EXPECT_EQ(config.groupLayerIds(default_tag), std::vector<int>({0, 1}));
    EXPECT_EQ(config.groupLayerIds(indexer_tag), std::vector<int>({0, 1}));
}

TEST(CacheConfigCreatorTest, SparseFlagWithoutIndexerDescriptorDoesNotProjectIndexerIntoDefaultScale) {
    auto model_config                         = makeMlaModel();
    model_config.attn_config.is_sparse        = true;
    model_config.attn_config.indexer_head_dim = 128;
    const auto        config                  = createFinalConfig(model_config);
    const std::string default_tag             = "mla";

    EXPECT_EQ(groupTagSet(config), (std::set<std::string>{"mla"}));
    EXPECT_EQ(config.group(default_tag).kv_scale_stride_bytes, 0u);
}

TEST(CacheConfigCreatorTest, CreateSpConfigAlignsSparseMlaIndexerAcrossTargetAndMtpModules) {
    auto score_config   = makeSparseMlaModel();
    auto propose_config = makeSparseMlaModel(/*layer_num=*/1);

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;
    const auto config           = CacheConfigCreator::createSpConfig(score_config,
                                                           propose_config,
                                                           ParallelismConfig{},
                                                           RuntimeConfig{},
                                                           fixedBlockConfig(),
                                                           sp_config,
                                                           std::nullopt,
                                                           /*is_mtp=*/true,
                                                           /*is_eagle=*/false);

    ASSERT_EQ(groupTagSet(config), (std::set<std::string>{"default", "indexer_kv"}));
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    const std::string target_indexer_tag = "indexer_kv";
    EXPECT_EQ(config.groupLayerIds(target_indexer_tag), std::vector<int>({0, 1, 2, 3}));
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_NE(sub_config, nullptr);
        ASSERT_EQ(groupTagSet(*sub_config), groupTagSet(config));
        const std::string sub_default_tag = "default";
        const std::string sub_indexer_tag = "indexer_kv";
        EXPECT_EQ(sub_config->groupLayerIds(sub_default_tag), std::vector<int>({0}));
        EXPECT_EQ(sub_config->groupLayerIds(sub_indexer_tag), std::vector<int>({0}));
        EXPECT_EQ(sub_config->group(sub_default_tag).kv_block_stride_bytes,
                  config.group("default").kv_block_stride_bytes);
        EXPECT_EQ(sub_config->group(sub_indexer_tag).kv_block_stride_bytes,
                  config.group(target_indexer_tag).kv_block_stride_bytes);
        EXPECT_EQ(sub_config->group(sub_indexer_tag).block_num, kTestBlockNum);
    }
}

TEST(CacheConfigCreatorTest, CreateConfigLowersKimiHybridToFixedFinalRecords) {
    const CacheSemanticSnapshot expected = {{"full",
                                             KVCacheSpecType::MultiHeadAttention,
                                             CacheGroupType::FULL,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             0,
                                             true,
                                             CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                             CpBlockSliceMode::NONE,
                                             {1, 3},
                                             kTestBlockNum,
                                             8,
                                             8,
                                             4096,
                                             2048,
                                             0},
                                            {"linear",
                                             KVCacheSpecType::LinearAttention,
                                             CacheGroupType::LINEAR,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             1,
                                             true,
                                             CpBlockMappingMode::NONE,
                                             CpBlockSliceMode::NONE,
                                             {0, 2},
                                             kTestBlockNum,
                                             8,
                                             8,
                                             3200,
                                             1600,
                                             0}};

    const auto config = createFinalConfig(makeKimiModel());
    EXPECT_EQ(snapshotCacheConfig(config), expected);
}

TEST(CacheConfigCreatorTest, GenericGroupingPreservesDescriptorFirstSeenOrder) {
    auto hybrid = makeKimiModel();
    for (auto& layer_descs : hybrid.kv_cache_spec_descs) {
        for (auto& desc : layer_descs) {
            desc.tag = desc.cache_type == KVCacheSpecType::LinearAttention ? "a_linear" : "z_full";
        }
    }
    const auto hybrid_config = CacheConfigCreator::createBasicConfig(hybrid, ParallelismConfig{}, KVCacheConfig{}, 0);
    ASSERT_EQ(hybrid_config.groups().size(), 2u);
    EXPECT_EQ(hybrid_config.groups()[0].tag, "a_linear");
    EXPECT_EQ(hybrid_config.groups()[0].policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(hybrid_config.groups()[1].tag, "z_full");

    auto independent                          = makeSparseMlaModel(/*layer_num=*/1);
    independent.kv_cache_spec_descs[0][0].tag = "z_first";
    independent.kv_cache_spec_descs[0][1].tag = "a_second";
    const auto independent_config =
        CacheConfigCreator::createBasicConfig(independent, ParallelismConfig{}, KVCacheConfig{}, 0);
    ASSERT_EQ(independent_config.groups().size(), 2u);
    EXPECT_EQ(independent_config.groups()[0].tag, "z_first");
    EXPECT_EQ(independent_config.groups()[1].tag, "a_second");
}

TEST(CacheConfigCreatorTest, GroupRecordsOwnHeterogeneousPhysicalAndKernelGeometry) {
    auto model                                                      = makeMhaModel(/*layer_num=*/1);
    model.attn_config.tokens_per_block                              = 128;
    model.hybrid_attention_config.enable_independent_kv_cache_pools = true;

    KVCacheSpecDesc scaled;
    scaled.tag                  = "scaled";
    scaled.cache_type           = KVCacheSpecType::OpaqueKV;
    scaled.entry_elems          = 16;
    scaled.entry_dtype          = DataType::TYPE_UINT8;
    scaled.explicit_entry_count = 8;
    scaled.cp                   = CacheCpPolicyDesc{};
    scaled.cp->mapping          = CpBlockMappingMode::COMPACT_LAST_RANK;
    model.kv_cache_spec_descs[0].push_back(scaled);

    ParallelismConfig parallelism;
    parallelism.role_type                          = RoleType::PREFILL;
    parallelism.tp_size                            = 2;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    KVCacheConfig kv_cache;
    kv_cache.kernel_seq_size_per_block = 64;
    kv_cache.test_block_num            = kTestBlockNum;

    const auto config = CacheConfigCreator::createConfig(model, parallelism, RuntimeConfig{}, kv_cache);
    EXPECT_EQ(config.group("default").seqSizePerBlock(), 128u);
    EXPECT_EQ(config.group("default").kernelSeqSizePerBlock(), 64u);
    EXPECT_EQ(config.group("default").spec->seq_size_per_block, 128u);
    EXPECT_EQ(config.group("default").spec->kernel_seq_size_per_block, 64u);
    EXPECT_EQ(config.group("scaled").seqSizePerBlock(), 256u);
    EXPECT_EQ(config.group("scaled").kernelSeqSizePerBlock(), 64u);
    EXPECT_EQ(config.group("scaled").spec->seq_size_per_block, 256u);
    EXPECT_EQ(config.group("scaled").spec->kernel_seq_size_per_block, 64u);
    EXPECT_EQ(config.kernelBlocksPerKvBlock("default"), 2u);
    EXPECT_EQ(config.kernelBlocksPerKvBlock("scaled"), 4u);
    EXPECT_EQ(config.seq_size_per_block, 128u);
}

TEST(CacheConfigCreatorTest, HybridGroupingRejectsEarlierFingerprintConflictBeforeLaterCategoryConflict) {
    auto model                                            = makeMhaModel(/*layer_num=*/3, /*tag=*/"full");
    model.hybrid_attention_config.enable_hybrid_attention = true;
    model.hybrid_attention_config.hybrid_attention_types  = {
        HybridAttentionType::NONE, HybridAttentionType::NONE, HybridAttentionType::LINEAR};
    model.kv_cache_spec_descs[1][0].dtype = DataType::TYPE_FP32;

    const auto error = runtimeErrorMessage(
        [&]() { (void)CacheConfigCreator::createBasicConfig(model, ParallelismConfig{}, KVCacheConfig{}, 0); });

    EXPECT_NE(error.find("multiple physical prototypes"), std::string::npos) << error;
    EXPECT_EQ(error.find("does not match attention type"), std::string::npos) << error;
}

TEST(CacheConfigCreatorTest, CreateConfigLowersQwenHybridDescriptorsWithConfiguredCpPolicy) {
    auto config = makeQwenHybridModel();
    for (auto& layer_descs : config.kv_cache_spec_descs) {
        for (auto& desc : layer_descs) {
            if (desc.tag == "qwen_linear") {
                desc.cp                       = CacheCpPolicyDesc{};
                desc.cp->mapping              = CpBlockMappingMode::COMPACT_LAST_RANK;
                desc.cp->slice                = CpBlockSliceMode::PAYLOAD_BYTES;
                desc.tail                     = CacheTailPolicyDesc{};
                desc.tail->active_tail_blocks = 2;
            }
        }
    }

    const auto lowered = createFinalConfig(config);
    ASSERT_EQ(groupTagSet(lowered), (std::set<std::string>{"qwen_full", "qwen_linear"}));
    EXPECT_EQ(lowered.groupLayerIds("qwen_full"), std::vector<int>({1, 3}));
    EXPECT_EQ(lowered.groupLayerIds("qwen_linear"), std::vector<int>({0, 2}));
    EXPECT_EQ(lowered.group("qwen_linear").policy.cp_mapping, CpBlockMappingMode::COMPACT_LAST_RANK);
    EXPECT_EQ(lowered.group("qwen_linear").policy.cp_slice, CpBlockSliceMode::PAYLOAD_BYTES);
    EXPECT_EQ(lowered.group("qwen_linear").policy.active_tail_blocks, 2);
}

TEST(CacheConfigCreatorTest, PublicCreatorWrappersPreserveDescriptorTagsAcrossBasicFinalAndSpeculativeLowering) {
    const auto model = makeMhaModel(/*layer_num=*/1, "wrapper_tag");
    const auto basic = CacheConfigCreator::createBasicConfig(model, ParallelismConfig{}, KVCacheConfig{}, 0);
    const auto final = createFinalConfig(model);
    const auto expect_lowered_descriptor = [](const CacheConfig& config, const std::vector<int>& layer_ids) {
        ASSERT_EQ(groupTagSet(config), (std::set<std::string>{"wrapper_tag"}));
        const auto& group = config.group("wrapper_tag");
        EXPECT_EQ(config.groupLayerIds(group.tag), layer_ids);
        EXPECT_EQ(group.policy.group_type, CacheGroupType::FULL);
        ASSERT_NE(group.spec, nullptr);
        EXPECT_EQ(group.spec->type, KVCacheSpecType::MultiHeadAttention);
    };
    expect_lowered_descriptor(basic, {0});
    expect_lowered_descriptor(final, {0});

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 1;
    const auto speculative      = CacheConfigCreator::createSpConfig(
        model, model, ParallelismConfig{}, RuntimeConfig{}, fixedBlockConfig(), sp_config, std::nullopt, true, false);
    // The target config appends one MTP layer; it must retain the same tagged
    // descriptor while lowering the expected target-layer span.
    expect_lowered_descriptor(speculative, {0, 1});
    ASSERT_EQ(speculative.mtp_sub_configs.size(), 1u);
    expect_lowered_descriptor(*speculative.mtp_sub_configs[0], {0});
}

TEST(CacheConfigCreatorTest, CreateConfigLowersDsv4SevenGroupPoliciesAndExplicitCapacity) {
    const auto snapshot = snapshotCacheConfig(createFinalConfig(makeDsv4Model()));
    ASSERT_EQ(snapshot.size(), 7u);

    const std::vector<std::string> expected_tags = {
        "csa_kv", "csa_state", "hca_kv", "hca_state", "indexer_kv", "indexer_state", "swa_kv"};
    for (size_t index = 0; index < expected_tags.size(); ++index) {
        EXPECT_EQ(snapshot[index].tag, expected_tags[index]);
    }
    EXPECT_EQ(snapshot[0].group_type, CacheGroupType::FULL);
    EXPECT_EQ(snapshot[0].group_layer_ids, std::vector<int>({1}));
    EXPECT_EQ(snapshot[0].block_num, kTestBlockNum);
    EXPECT_EQ(snapshot[0].block_bytes, 32768u);
    EXPECT_EQ(snapshot[1].group_type, CacheGroupType::SWA);
    EXPECT_EQ(snapshot[1].cp_mapping, CpBlockMappingMode::COMPACT_LAST_RANK);
    EXPECT_EQ(snapshot[1].cp_slice, CpBlockSliceMode::PAYLOAD_BYTES);
    EXPECT_EQ(snapshot[1].active_tail_blocks, 2u);
    EXPECT_EQ(snapshot[3].explicit_block_num, 256u);
    EXPECT_EQ(snapshot[3].block_num, 256u);
    EXPECT_FALSE(snapshot[3].enable_prefix_reuse);
    EXPECT_FALSE(snapshot[3].validate_tail_blocks);
    EXPECT_EQ(snapshot[6].cp_slice, CpBlockSliceMode::EQUAL_BYTES);
    EXPECT_EQ(snapshot[6].group_layer_ids, std::vector<int>({0, 1}));
    EXPECT_EQ(snapshot[6].block_num, kTestBlockNum);
    EXPECT_EQ(snapshot[6].block_bytes, 262144u);
}

TEST(CacheConfigCreatorTest, CreateConfigUsesDescriptorCpPolicy) {
    auto config = makeKimiModel();
    for (auto& layer_descs : config.kv_cache_spec_descs) {
        for (auto& desc : layer_descs) {
            if (desc.tag == "full") {
                desc.cp          = CacheCpPolicyDesc{};
                desc.cp->mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
                desc.cp->slice   = CpBlockSliceMode::EQUAL_BYTES;
            }
        }
    }

    const auto        final_config = createFinalConfig(config);
    const std::string full_tag     = "full";
    EXPECT_EQ(final_config.group(full_tag).policy.cp_mapping, CpBlockMappingMode::COMPACT_LAST_RANK);
    EXPECT_EQ(final_config.group(full_tag).policy.cp_slice, CpBlockSliceMode::EQUAL_BYTES);
}

TEST(CacheConfigCreatorTest, DescriptorPolicyControlsActiveCpGeometry) {
    auto config = makeKimiModel();
    for (auto& layer_descs : config.kv_cache_spec_descs) {
        for (auto& desc : layer_descs) {
            if (desc.tag == "full") {
                desc.cp          = CacheCpPolicyDesc{};
                desc.cp->mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
            }
        }
    }

    ParallelismConfig parallelism;
    parallelism.role_type                          = RoleType::PREFILL;
    parallelism.tp_size                            = 2;
    parallelism.prefill_cp_config.kv_cache_sharded = true;

    const auto        cache_config = CacheConfigCreator::createBasicConfig(config, parallelism, KVCacheConfig{}, 0);
    const std::string full_tag     = "full";
    const auto&       full_group   = cache_config.group(full_tag);
    EXPECT_EQ(full_group.policy.cp_mapping, CpBlockMappingMode::COMPACT_LAST_RANK);
    EXPECT_EQ(full_group.seqSizePerBlock(), 16u);
    EXPECT_EQ(full_group.spec->seq_size_per_block, 16u);
    EXPECT_EQ(full_group.kernelSeqSizePerBlock(), 8u);
    EXPECT_EQ(full_group.spec->kernel_seq_size_per_block, 8u);
}

TEST(CacheConfigCreatorTest, CreateConfigPreservesAllLinearExplicitTags) {
    auto config                                                      = makeKimiModel();
    config.num_layers                                                = 2;
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    config.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::LINEAR, HybridAttentionType::LINEAR};
    config.kv_cache_spec_descs.resize(2);
    config.kv_cache_spec_descs[0] = {KVCacheSpecDesc{"recurrent_state", KVCacheSpecType::LinearAttention}};
    config.kv_cache_spec_descs[1] = {KVCacheSpecDesc{"convolution_state", KVCacheSpecType::LinearAttention}};

    const auto final_config = createFinalConfig(config);
    EXPECT_EQ(groupTagSet(final_config), (std::set<std::string>{"recurrent_state", "convolution_state"}));
    EXPECT_EQ(final_config.group("recurrent_state").policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(final_config.group("convolution_state").policy.group_type, CacheGroupType::LINEAR);
}

TEST(CacheConfigCreatorTest, CreateSpConfigPreservesExactAndDefaultMtpMappingsWithPlaceholders) {
    auto score_config                           = makeKimiModel();
    auto propose_config                         = makeMhaModel(/*layer_num=*/1, /*tag=*/"default");
    propose_config.attn_config.size_per_head    = score_config.attn_config.size_per_head;
    propose_config.attn_config.tokens_per_block = score_config.attn_config.tokens_per_block;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;
    const auto config           = CacheConfigCreator::createSpConfig(score_config,
                                                           propose_config,
                                                           ParallelismConfig{},
                                                           RuntimeConfig{},
                                                           fixedBlockConfig(),
                                                           sp_config,
                                                           std::nullopt,
                                                           /*is_mtp=*/true,
                                                           /*is_eagle=*/false);

    EXPECT_EQ(groupTagSet(config), (std::set<std::string>{"full", "linear"}));
    EXPECT_EQ(config.groupLayerIds("full"), std::vector<int>({1, 3, 4, 5}));
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_NE(sub_config, nullptr);
        const std::string full_tag   = "full";
        const std::string linear_tag = "linear";
        EXPECT_EQ(sub_config->group(full_tag).tag, "full");
        EXPECT_EQ(sub_config->groupLayerIds(full_tag), std::vector<int>({0}));
        EXPECT_TRUE(sub_config->groupLayerIds(linear_tag).empty());
        EXPECT_EQ(sub_config->group(full_tag).block_num, kTestBlockNum);
    }
}

TEST(CacheConfigCreatorTest, InvalidInputsKeepDescriptorBoundaries) {
    auto empty_tag_config = makeMhaModel();
    empty_tag_config.kv_cache_spec_descs[0][0].tag.clear();
    empty_tag_config.kv_cache_spec_descs[1][0].tag.clear();
    const auto empty_tag_error = runtimeErrorMessage([&]() { (void)createFinalConfig(empty_tag_config); });
    EXPECT_NE(empty_tag_error.find("tag must not be empty"), std::string::npos);

    auto inconsistent_config                                 = makeKimiModel();
    inconsistent_config.kv_cache_spec_descs[0][0].cache_type = KVCacheSpecType::MultiHeadAttention;
    const auto inconsistent_error = runtimeErrorMessage([&]() { (void)createFinalConfig(inconsistent_config); });
    EXPECT_NE(inconsistent_error.find("multiple physical prototypes"), std::string::npos);
}

TEST(CacheConfigCreatorTest, DuplicateDescTagsFailDuringIndependentGrouping) {
    auto config                                                      = makeMhaModel(/*layer_num=*/1);
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    config.kv_cache_spec_descs[0].push_back(config.kv_cache_spec_descs[0][0]);

    const auto error = runtimeErrorMessage([&]() { (void)createFinalConfig(config); });

    EXPECT_NE(error.find("hybrid-pool layer 0 has duplicate tag=default"), std::string::npos);
}

TEST(CacheConfigCreatorTest, SameTagDifferentLayoutsFailDuringIndependentGrouping) {
    auto config                                                      = makeSparseMlaModel();
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    config.kv_cache_spec_descs[1][1].entry_elems += 1;

    const auto error = runtimeErrorMessage([&]() { (void)createFinalConfig(config); });

    EXPECT_NE(error.find("hybrid-pool tag=indexer_kv has multiple physical prototypes"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm::test
