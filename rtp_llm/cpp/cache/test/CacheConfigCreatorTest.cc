#include <gtest/gtest.h>

#include <functional>
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
    EXPECT_FALSE(config.use_independent_block_pools);
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
    EXPECT_FALSE(config.use_independent_block_pools);
    EXPECT_EQ(snapshotCacheConfig(config), expected);
}

TEST(CacheConfigCreatorTest, CreateConfigIsolatesSparseMlaIndexerPoolAndStride) {
    const auto config = createFinalConfig(makeSparseMlaModel());

    ASSERT_TRUE(config.use_independent_block_pools);
    ASSERT_EQ(config.groupTagsSnapshot(), std::vector<std::string>({"default", "indexer_kv"}));
    const auto default_gid = static_cast<size_t>(config.groupIdForTag("default"));
    const auto indexer_gid = static_cast<size_t>(config.groupIdForTag("indexer_kv"));
    EXPECT_EQ(config.typeForGroup(default_gid), CacheGroupType::FULL);
    EXPECT_EQ(config.typeForGroup(indexer_gid), CacheGroupType::FULL);
    EXPECT_EQ(config.specForGroup(default_gid)->type, KVCacheSpecType::MultiHeadLatentAttention);
    EXPECT_EQ(config.specForGroup(indexer_gid)->type, KVCacheSpecType::OpaqueKV);
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(default_gid), 96u);
    EXPECT_EQ(config.kvScaleStrideBytesForGroup(default_gid), 0u);
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(indexer_gid), 528u);
    EXPECT_EQ(config.kvScaleStrideBytesForGroup(indexer_gid), 0u);
    EXPECT_EQ(config.blockNumForGroup(default_gid), kTestBlockNum);
    EXPECT_EQ(config.blockNumForGroup(indexer_gid), kTestBlockNum);
    EXPECT_EQ(config.layerIdsForGroup(default_gid), std::vector<int>({0, 1}));
    EXPECT_EQ(config.layerIdsForGroup(indexer_gid), std::vector<int>({0, 1}));
}

TEST(CacheConfigCreatorTest, SparseFlagWithoutIndexerDescriptorDoesNotProjectIndexerIntoDefaultScale) {
    auto model_config                         = makeMlaModel();
    model_config.attn_config.is_sparse        = true;
    model_config.attn_config.indexer_head_dim = 128;
    const auto config                         = createFinalConfig(model_config);
    const auto default_gid                    = static_cast<size_t>(config.groupIdForTag("mla"));

    EXPECT_FALSE(config.use_independent_block_pools);
    EXPECT_EQ(config.groupTagsSnapshot(), std::vector<std::string>({"mla"}));
    EXPECT_EQ(config.kvScaleStrideBytesForGroup(default_gid), 0u);
    EXPECT_EQ(config.kv_scale_stride_bytes, 0u);
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

    ASSERT_EQ(config.groupTagsSnapshot(), std::vector<std::string>({"default", "indexer_kv"}));
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    const auto target_indexer_gid = static_cast<size_t>(config.groupIdForTag("indexer_kv"));
    EXPECT_EQ(config.layerIdsForGroup(target_indexer_gid), std::vector<int>({0, 1, 2, 3}));
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_NE(sub_config, nullptr);
        ASSERT_EQ(sub_config->groupTagsSnapshot(), config.groupTagsSnapshot());
        const auto sub_default_gid = static_cast<size_t>(sub_config->groupIdForTag("default"));
        const auto sub_indexer_gid = static_cast<size_t>(sub_config->groupIdForTag("indexer_kv"));
        EXPECT_EQ(sub_config->layerIdsForGroup(sub_default_gid), std::vector<int>({0}));
        EXPECT_EQ(sub_config->layerIdsForGroup(sub_indexer_gid), std::vector<int>({0}));
        EXPECT_EQ(sub_config->kvBlockStrideBytesForGroup(sub_default_gid),
                  config.kvBlockStrideBytesForGroup(static_cast<size_t>(config.groupIdForTag("default"))));
        EXPECT_EQ(sub_config->kvBlockStrideBytesForGroup(sub_indexer_gid),
                  config.kvBlockStrideBytesForGroup(target_indexer_gid));
        EXPECT_EQ(sub_config->blockNumForGroup(sub_indexer_gid), kTestBlockNum);
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
    EXPECT_TRUE(config.use_independent_block_pools);
    EXPECT_EQ(config.group_layer_num, 2);
    EXPECT_EQ(snapshotCacheConfig(config), expected);
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
    EXPECT_EQ(snapshot[0].layer_ids, std::vector<int>({1}));
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
    EXPECT_EQ(snapshot[6].layer_ids, std::vector<int>({0, 1}));
    EXPECT_EQ(snapshot[6].block_num, kTestBlockNum);
    EXPECT_EQ(snapshot[6].block_bytes, 262144u);
}

TEST(CacheConfigCreatorTest, CreateConfigPreservesLegacyHybridDefaultCpPolicy) {
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

    const auto final_config = createFinalConfig(config);
    const auto full_gid     = static_cast<size_t>(final_config.groupIdForTag("full"));
    EXPECT_EQ(final_config.policyForGroup(full_gid).cp_mapping, CpBlockMappingMode::BLOCK_ROUND_ROBIN);
    EXPECT_EQ(final_config.policyForGroup(full_gid).cp_slice, CpBlockSliceMode::NONE);
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
    EXPECT_EQ(final_config.groupTagsSnapshot(), std::vector<std::string>({"recurrent_state", "convolution_state"}));
    EXPECT_EQ(final_config.typeForGroup(0), CacheGroupType::LINEAR);
    EXPECT_EQ(final_config.typeForGroup(1), CacheGroupType::LINEAR);
    EXPECT_TRUE(final_config.use_independent_block_pools);
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

    EXPECT_EQ(config.groupTagsSnapshot(), std::vector<std::string>({"full", "linear"}));
    EXPECT_EQ(config.group_layer_num, 2);
    EXPECT_EQ(config.layerIdsForGroup(static_cast<size_t>(config.groupIdForTag("full"))),
              std::vector<int>({1, 3, 4, 5}));
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_NE(sub_config, nullptr);
        const auto full_gid   = static_cast<size_t>(sub_config->groupIdForTag("full"));
        const auto linear_gid = static_cast<size_t>(sub_config->groupIdForTag("linear"));
        EXPECT_EQ(sub_config->specForGroup(full_gid)->tag, "full");
        EXPECT_EQ(sub_config->layerIdsForGroup(full_gid), std::vector<int>({0}));
        EXPECT_TRUE(sub_config->layerIdsForGroup(linear_gid).empty());
        EXPECT_EQ(sub_config->blockNumForGroup(full_gid), kTestBlockNum);
    }
}

TEST(CacheConfigCreatorTest, InvalidInputsKeepDescriptorAndCategoryBoundaries) {
    auto empty_tag_config = makeMhaModel();
    empty_tag_config.kv_cache_spec_descs[0][0].tag.clear();
    empty_tag_config.kv_cache_spec_descs[1][0].tag.clear();
    const auto empty_tag_error = runtimeErrorMessage([&]() { (void)createFinalConfig(empty_tag_config); });
    EXPECT_NE(empty_tag_error.find("tag must not be empty"), std::string::npos);

    auto duplicate_config                                                      = makeMhaModel(/*layer_num=*/1);
    duplicate_config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    duplicate_config.kv_cache_spec_descs[0].push_back(duplicate_config.kv_cache_spec_descs[0][0]);
    const auto duplicate_error = runtimeErrorMessage([&]() { (void)createFinalConfig(duplicate_config); });
    EXPECT_NE(duplicate_error.find("duplicate tag"), std::string::npos);

    auto category_config                                 = makeKimiModel();
    category_config.kv_cache_spec_descs[0][0].cache_type = KVCacheSpecType::MultiHeadAttention;
    const auto category_error = runtimeErrorMessage([&]() { (void)createFinalConfig(category_config); });
    EXPECT_NE(category_error.find("does not match attention type"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm::test
