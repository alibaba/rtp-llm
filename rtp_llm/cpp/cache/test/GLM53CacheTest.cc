#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {
namespace test {
namespace {

ModelConfig makeGlm53Config() {
    ModelConfig mc;
    mc.num_layers                             = 6;
    mc.hidden_size                            = 1024;
    mc.attn_config.use_mla                    = true;
    mc.attn_config.is_sparse                  = true;
    mc.attn_config.head_num                   = 64;
    mc.attn_config.kv_head_num                = 1;
    mc.attn_config.size_per_head              = 256;
    mc.attn_config.kv_lora_rank               = 512;
    mc.attn_config.nope_head_dim              = 256;
    mc.attn_config.rope_head_dim              = 0;
    mc.attn_config.tokens_per_block           = 128;
    mc.attn_config.kernel_tokens_per_block    = 128;
    mc.attn_config.indexer_head_dim           = 128;
    mc.attn_config.indexer_head_num           = 32;
    mc.attn_config.indexer_topk               = 512;
    mc.attn_config.indexer_compress_ratio     = 4;
    mc.attn_config.indexer_compressor_overlap = 0;
    mc.attn_config.sparse_attention_topk      = 2051;

    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    mc.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::LINEAR,
        HybridAttentionType::NONE,
        HybridAttentionType::NONE,
        HybridAttentionType::LINEAR,
        HybridAttentionType::NONE,
        HybridAttentionType::NONE,
    };
    mc.linear_attention_config.linear_conv_kernel_dim = 4;
    mc.linear_attention_config.linear_key_head_dim    = 64;
    mc.linear_attention_config.linear_value_head_dim  = 64;
    mc.linear_attention_config.linear_num_key_heads   = 4;
    mc.linear_attention_config.linear_num_value_heads = 4;
    mc.linear_attention_config.ssm_state_dtype        = DataType::TYPE_FP32;
    return mc;
}

KVCacheConfig makeKvConfig() {
    KVCacheConfig config;
    config.seq_size_per_block        = 128;
    config.kernel_seq_size_per_block = 128;
    config.dsv4_fixed_pool_blocks    = 32;
    return config;
}

}  // namespace

TEST(GLM53CacheConfigTest, AppendsKPoolRegionsOnlyToMlaLayers) {
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, makeKvConfig(), false, 0);

    ASSERT_EQ(config.cache_specs.size(), 4u);
    EXPECT_EQ(config.group_region_names[0], KVCacheRegionName::DEFAULT);
    EXPECT_EQ(config.group_region_names[1], KVCacheRegionName::DEFAULT);
    EXPECT_EQ(config.group_region_names[2], KVCacheRegionName::INDEXER_KV);
    EXPECT_EQ(config.group_region_names[3], KVCacheRegionName::INDEXER_STATE);
    EXPECT_NE(dynamic_cast<MLAKVCacheSpec*>(config.cache_specs[0].get()), nullptr);
    auto* linear = dynamic_cast<LinearKVCacheSpec*>(config.cache_specs[1].get());
    ASSERT_NE(linear, nullptr);
    EXPECT_EQ(linear->ssm_state_dtype, DataType::TYPE_FP32);

    auto* indexer_kv = dynamic_cast<DSV4KVSpec*>(config.cache_specs[2].get());
    ASSERT_NE(indexer_kv, nullptr);
    EXPECT_EQ(indexer_kv->layer_num, 4u);
    EXPECT_EQ(indexer_kv->entry_elems, 132u);
    EXPECT_EQ(indexer_kv->entries_per_block, 32u);

    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(indexer_state->layer_num, 4u);
    EXPECT_EQ(indexer_state->state_dim, 256u);
    EXPECT_EQ(indexer_state->entries_per_block, 4u);

    const auto kv_region    = static_cast<size_t>(KVCacheRegionName::INDEXER_KV);
    const auto state_region = static_cast<size_t>(KVCacheRegionName::INDEXER_STATE);
    for (int layer : {0, 3}) {
        EXPECT_EQ(config.layer_region_to_group_id[layer][kv_region], -1);
        EXPECT_EQ(config.layer_region_to_group_id[layer][state_region], -1);
    }
    for (int layer : {1, 2, 4, 5}) {
        EXPECT_GE(config.layer_region_to_group_id[layer][kv_region], 0);
        EXPECT_GE(config.layer_region_to_group_id[layer][state_region], 0);
    }
    EXPECT_TRUE(config.use_typed_cache_regions);
    EXPECT_TRUE(config.use_independent_block_pools);
    EXPECT_TRUE(config.use_opaque_kv_cache_store);
}

TEST(GLM53CacheConfigTest, KPoolIsAlwaysFp8AndStateIncludesMtpSlack) {
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, makeKvConfig(), false, 3);
    auto*             indexer_kv = dynamic_cast<DSV4KVSpec*>(config.cache_specs[2].get());
    ASSERT_NE(indexer_kv, nullptr);
    EXPECT_EQ(indexer_kv->store_dtype, DataType::TYPE_UINT8);
    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(indexer_state->entries_per_block, 8u);
}

TEST(GLM53CacheConfigTest, AllMlaMtpDoesNotRequireUnusedLinearConfig) {
    ParallelismConfig pc;
    auto              model = makeGlm53Config();
    model.num_layers        = 1;
    model.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE};
    model.attn_config.indexer_layer_ids                   = {0};
    model.linear_attention_config                         = {};

    auto config = HybridPoolConfigCreator::createConfig(model, pc, makeKvConfig(), true, 3);

    ASSERT_EQ(config.cache_specs.size(), 3u);
    EXPECT_NE(dynamic_cast<MLAKVCacheSpec*>(config.cache_specs[0].get()), nullptr);
    EXPECT_NE(dynamic_cast<DSV4KVSpec*>(config.cache_specs[1].get()), nullptr);
    EXPECT_NE(dynamic_cast<DSV4StateSpec*>(config.cache_specs[2].get()), nullptr);
    for (const auto& spec : config.cache_specs) {
        EXPECT_EQ(dynamic_cast<LinearKVCacheSpec*>(spec.get()), nullptr);
    }
}

TEST(GLM53CacheConfigTest, EagleMtpOwnsIndependentTypedPools) {
    auto score_model   = makeGlm53Config();
    auto propose_model = makeGlm53Config();
    propose_model.num_layers = 1;
    propose_model.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE};
    propose_model.attn_config.indexer_layer_ids                   = {0};
    propose_model.linear_attention_config                         = {};

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_config = makeKvConfig();
    kv_config.test_block_num    = 8;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_EAGLE;
    sp_config.gen_num_per_cycle = 3;

    auto config = CacheConfigCreator::createSpConfig(score_model,
                                                     propose_model,
                                                     parallelism_config,
                                                     runtime_config,
                                                     kv_config,
                                                     sp_config,
                                                     std::nullopt,
                                                     true,
                                                     true);

    ASSERT_EQ(config.layer_all_num, 7u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
    const int mtp_layer = 6;
    const auto default_region = static_cast<size_t>(KVCacheRegionName::DEFAULT);
    const auto kv_region    = static_cast<size_t>(KVCacheRegionName::INDEXER_KV);
    const auto state_region = static_cast<size_t>(KVCacheRegionName::INDEXER_STATE);
    const int default_group = config.layer_region_to_group_id[mtp_layer][default_region];
    const int kv_group      = config.layer_region_to_group_id[mtp_layer][kv_region];
    const int state_group   = config.layer_region_to_group_id[mtp_layer][state_region];

    EXPECT_GE(default_group, 0);
    EXPECT_GE(kv_group, 0);
    EXPECT_GE(state_group, 0);
    EXPECT_EQ(config.layer_to_group_id[mtp_layer], default_group);
    EXPECT_EQ(config.group_region_names[default_group], KVCacheRegionName::DEFAULT);
    EXPECT_NE(kv_group, state_group);
    EXPECT_EQ(config.group_region_names[kv_group], KVCacheRegionName::INDEXER_KV);
    EXPECT_EQ(config.group_region_names[state_group], KVCacheRegionName::INDEXER_STATE);
    EXPECT_EQ(config.global_layer_ids[kv_group], std::vector<int>({mtp_layer}));
    EXPECT_EQ(config.global_layer_ids[state_group], std::vector<int>({mtp_layer}));
    EXPECT_GT(static_cast<size_t>(kv_group), 3u);
    EXPECT_GT(static_cast<size_t>(state_group), 3u);
    EXPECT_EQ(config.mtp_sub_configs[0]->layer_to_group_id, std::vector<int>({0}));
    EXPECT_EQ(config.mtp_sub_configs[0]->local_to_global_layer_ids, std::vector<int>({mtp_layer}));
    const auto& mtp_config         = *config.mtp_sub_configs[0];
    const int   local_default_group = mtp_config.layer_region_to_group_id[0][default_region];
    const int   local_kv_group      = mtp_config.layer_region_to_group_id[0][kv_region];
    const int   local_state_group   = mtp_config.layer_region_to_group_id[0][state_region];
    ASSERT_GE(local_default_group, 0);
    ASSERT_GE(local_kv_group, 0);
    ASSERT_GE(local_state_group, 0);
    ASSERT_LT(static_cast<size_t>(local_default_group), mtp_config.group_region_names.size());
    ASSERT_LT(static_cast<size_t>(local_kv_group), mtp_config.group_region_names.size());
    ASSERT_LT(static_cast<size_t>(local_state_group), mtp_config.group_region_names.size());
    EXPECT_EQ(mtp_config.group_region_names[local_default_group], KVCacheRegionName::DEFAULT);
    EXPECT_EQ(mtp_config.group_region_names[local_kv_group], KVCacheRegionName::INDEXER_KV);
    EXPECT_EQ(mtp_config.group_region_names[local_state_group], KVCacheRegionName::INDEXER_STATE);
    EXPECT_NE(local_default_group, default_group);
    EXPECT_NE(local_kv_group, kv_group);
    EXPECT_NE(local_state_group, state_group);
}

TEST(GLM53CacheConfigTest, SplitPhysicalBlocksScaleOnlyPagedKPool) {
    ParallelismConfig pc;
    auto              kv_config         = makeKvConfig();
    kv_config.seq_size_per_block        = 256;
    kv_config.kernel_seq_size_per_block = 128;
    auto config = HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, kv_config, false, 0);
    EXPECT_EQ(config.group_kv_block_stride_bytes[2], 2u * 32u * 132u);
    auto* state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(config.group_kv_block_stride_bytes[3], state->block_size_bytes());
}

TEST(GLM53CacheConfigTest, RejectsInvalidGeometryOwnershipAndPrefillCp) {
    ParallelismConfig pc;
    auto              kv_config = makeKvConfig();

    auto bad_ratio                               = makeGlm53Config();
    bad_ratio.attn_config.indexer_compress_ratio = 2;
    EXPECT_DEATH(HybridPoolConfigCreator::createConfig(bad_ratio, pc, kv_config, false, 0), "");

    auto bad_topk                              = makeGlm53Config();
    bad_topk.attn_config.sparse_attention_topk = 2048;
    EXPECT_DEATH(HybridPoolConfigCreator::createConfig(bad_topk, pc, kv_config, false, 0), "");

    auto bad_owner                          = makeGlm53Config();
    bad_owner.attn_config.indexer_layer_ids = {0};
    EXPECT_DEATH(HybridPoolConfigCreator::createConfig(bad_owner, pc, kv_config, false, 0), "");

    kv_config.seq_size_per_block        = 64;
    kv_config.kernel_seq_size_per_block = 64;
    EXPECT_DEATH(HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, kv_config, false, 0), "");

    pc.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_DEATH(HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, makeKvConfig(), false, 0), "");
}

}  // namespace test
}  // namespace rtp_llm
