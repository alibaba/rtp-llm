#include <gtest/gtest.h>

#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"

namespace rtp_llm {
namespace test {
namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value): name_(name) {
        const char* old_value = std::getenv(name_);
        if (old_value != nullptr) {
            old_value_ = old_value;
            had_value_ = true;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    std::string old_value_;
    bool        had_value_ = false;
};

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
    mc.linear_attention_config.linear_num_key_heads   = 64;
    mc.linear_attention_config.linear_num_value_heads = 64;
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

TEST(GLM53CacheConfigTest, PinnedMlaLeavesKdaAndKPoolOnDeviceAndBudgetsExpandedTopk) {
    ScopedEnvVar host_budget("RTP_LLM_DSA_MLA_HOST_CACHE_MB", "64");
    ScopedEnvVar resident("RTP_LLM_DSA_MLA_RESIDENT_TOKENS", "0");
    ScopedEnvVar request_cache("ENABLE_LINEAR_ATTN_REQUEST_CACHE", "1");
    auto         model     = makeGlm53Config();
    model.data_type        = DataType::TYPE_BF16;
    auto options           = makeKvConfig();
    options.test_block_num = 256;
    RuntimeConfig runtime;
    runtime.max_generate_batch_size                      = 3;
    runtime.fifo_scheduler_config.max_context_batch_size = 1;
    const auto config = CacheConfigCreator::createConfig(model, ParallelismConfig(), runtime, options, std::nullopt);
    EXPECT_EQ(config.dsa_mla_resident_tokens, 6272u);  // ceil(3 * 2051 / 128) * 128
    EXPECT_GT(config.block_num, config.dsa_mla_hbm_blocks);
    size_t host_bytes = 0;
    for (size_t gid = 0; gid < config.cache_specs.size(); ++gid) {
        const auto  pool   = BlockPoolConfigHelper::createConfigForGroup(config, gid);
        const auto& layout = pool.memory_layouts.front();
        const bool  mla    = dynamic_cast<MLAKVCacheSpec*>(config.cache_specs[gid].get()) != nullptr;
        EXPECT_EQ(pool.mla_tiered_cache, mla);
        if (mla) {
            EXPECT_EQ(layout.mla_hbm_blocks, config.dsa_mla_hbm_blocks);
            EXPECT_EQ(layout.kv_scale_pool_size_bytes, 0u);
            host_bytes += pool.total_size_bytes;
            EXPECT_EQ(layout.mla_hbm_size_bytes,
                      layout.layer_num * layout.kv_block_stride_bytes
                          * (config.dsa_mla_hbm_blocks + config.dsa_mla_resident_tokens / 128));
        } else {
            EXPECT_EQ(layout.mla_resident_tokens, 0u);
            EXPECT_EQ(layout.mla_hbm_size_bytes, 0u);
        }
    }
    EXPECT_GT(host_bytes, 0u);
    EXPECT_LE(host_bytes, 64u * 1024 * 1024);
}

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

TEST(GLM53CacheConfigTest, TpPrefillKeepsHeadShardedComputeAndPageRrStorage) {
    auto model                                           = makeGlm53Config();
    model.linear_attention_config.linear_num_key_heads   = 64;
    model.linear_attention_config.linear_num_value_heads = 64;

    ParallelismConfig pc;
    pc.tp_size                            = 8;
    pc.tp_rank                            = 3;
    pc.prefill_cp_config.method           = CPRotateMethod::DISABLED;
    pc.prefill_cp_config.kv_cache_sharded = true;

    EXPECT_EQ(pc.get_attn_tp_size(), 8);
    EXPECT_EQ(pc.get_attn_tp_rank(), 3);

    auto  config = HybridPoolConfigCreator::createConfig(model, pc, makeKvConfig(), false, 0);
    auto* linear = dynamic_cast<LinearKVCacheSpec*>(config.cache_specs[1].get());
    ASSERT_NE(linear, nullptr);
    EXPECT_EQ(linear->local_num_k_heads, 8u);
    EXPECT_EQ(linear->local_num_v_heads, 8u);
    EXPECT_EQ(linear->ssm_state_size(), 8u * 64u * 64u);

    DeviceResourceConfig resource_config;
    const auto           props = buildExecProperties(pc, resource_config);
    EXPECT_FALSE(props.enable_prefill_cp);
    EXPECT_TRUE(props.prefill_cp_kv_cache_sharded);
    EXPECT_EQ(props.tp_size, 8u);
    EXPECT_EQ(props.tp_rank, 3u);
}

TEST(GLM53CacheConfigTest, RejectsKdaHeadsNotDivisibleByTp) {
    auto model                                           = makeGlm53Config();
    model.linear_attention_config.linear_num_key_heads   = 66;
    model.linear_attention_config.linear_num_value_heads = 66;

    ParallelismConfig pc;
    pc.tp_size                            = 8;
    pc.prefill_cp_config.method           = CPRotateMethod::DISABLED;
    pc.prefill_cp_config.kv_cache_sharded = true;

    EXPECT_DEATH(HybridPoolConfigCreator::createConfig(model, pc, makeKvConfig(), false, 0), "");
}

TEST(GLM53CacheConfigTest, AllMlaMtpDoesNotRequireUnusedLinearConfig) {
    ParallelismConfig pc;
    auto              model                              = makeGlm53Config();
    model.num_layers                                     = 1;
    model.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE};
    model.attn_config.indexer_layer_ids                  = {0};
    model.linear_attention_config                        = {};

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
    auto score_model                                             = makeGlm53Config();
    auto propose_model                                           = makeGlm53Config();
    propose_model.num_layers                                     = 1;
    propose_model.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE};
    propose_model.attn_config.indexer_layer_ids                  = {0};
    propose_model.linear_attention_config                        = {};

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_config = makeKvConfig();
    kv_config.test_block_num    = 8;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_EAGLE;
    sp_config.gen_num_per_cycle = 3;

    auto config = CacheConfigCreator::createSpConfig(
        score_model, propose_model, parallelism_config, runtime_config, kv_config, sp_config, std::nullopt, true, true);

    ASSERT_EQ(config.layer_all_num, 7u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
    const int  mtp_layer      = 6;
    const auto default_region = static_cast<size_t>(KVCacheRegionName::DEFAULT);
    const auto kv_region      = static_cast<size_t>(KVCacheRegionName::INDEXER_KV);
    const auto state_region   = static_cast<size_t>(KVCacheRegionName::INDEXER_STATE);
    const int  default_group  = config.layer_region_to_group_id[mtp_layer][default_region];
    const int  kv_group       = config.layer_region_to_group_id[mtp_layer][kv_region];
    const int  state_group    = config.layer_region_to_group_id[mtp_layer][state_region];

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
    const auto& mtp_config          = *config.mtp_sub_configs[0];
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

TEST(GLM53CacheConfigTest, EagleSmallPagesUseGlmValidationForBothModels) {
    for (auto role : {RoleType::PREFILL, RoleType::DECODE}) {
        auto score                         = makeGlm53Config();
        score.attn_config.tokens_per_block = score.attn_config.kernel_tokens_per_block = 64;
        auto propose                                                                   = score;
        propose.num_layers                                                             = 1;
        propose.hybrid_attention_config.hybrid_attention_types                         = {HybridAttentionType::NONE};
        propose.attn_config.indexer_layer_ids                                          = {0};
        propose.linear_attention_config                                                = {};
        ParallelismConfig pc;
        pc.role_type = role;
        pc.tp_size   = role == RoleType::PREFILL ? 4 : 1;
        pc.dp_size   = role == RoleType::DECODE ? 4 : 1;
        pc.ep_size = pc.world_size            = 4;
        pc.prefill_cp_config.kv_cache_sharded = true;
        pc.prefill_cp_config.prefill_cp_size  = 4;
        pc.prefill_cp_config.method = role == RoleType::DECODE ? CPRotateMethod::PREFILL_CP : CPRotateMethod::DISABLED;
        RuntimeConfig runtime;
        auto          kv      = makeKvConfig();
        kv.seq_size_per_block = kv.kernel_seq_size_per_block = 64;
        kv.test_block_num                                    = 8;
        SpeculativeExecutionConfig sp;
        sp.type              = SP_TYPE_EAGLE;
        sp.gen_num_per_cycle = 3;
        auto config = CacheConfigCreator::createSpConfig(score, propose, pc, runtime, kv, sp, std::nullopt, true, true);
        EXPECT_EQ(config.seq_size_per_block, 64u);
        EXPECT_EQ(config.kernel_seq_size_per_block, 64u);
        ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
        const auto& draft = *config.mtp_sub_configs[0];
        EXPECT_EQ(draft.kernel_seq_size_per_block, 64u);
        const auto region = static_cast<size_t>(KVCacheRegionName::INDEXER_KV);
        const int  group  = draft.layer_region_to_group_id[0][region];
        ASSERT_GE(group, 0);
        auto* indexer = dynamic_cast<DSV4KVSpec*>(draft.cache_specs[group].get());
        ASSERT_NE(indexer, nullptr);
        EXPECT_EQ(indexer->entries_per_block, 16u);
    }
}

TEST(GLM53CacheConfigTest, SplitPhysicalBlocksScaleOnlyPagedKPool) {
    ParallelismConfig pc;
    auto              kv_config         = makeKvConfig();
    kv_config.seq_size_per_block        = 256;
    kv_config.kernel_seq_size_per_block = 128;
    auto config = HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, kv_config, false, 0);

    auto* mla = dynamic_cast<MLAKVCacheSpec*>(config.cache_specs[0].get());
    ASSERT_NE(mla, nullptr);
    EXPECT_EQ(mla->seq_size_per_block, 128u);
    EXPECT_EQ(config.group_kv_block_stride_bytes[0], 2u * mla->block_size_bytes());
    EXPECT_EQ(config.group_kv_block_stride_bytes[2], 2u * 32u * 132u);
    auto* state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(config.group_kv_block_stride_bytes[3], state->block_size_bytes());
}

TEST(GLM53CacheConfigTest, SmallPagesPreserveKPoolAndStateGeometry) {
    ParallelismConfig pc;
    auto              kv_config  = makeKvConfig();
    kv_config.seq_size_per_block = kv_config.kernel_seq_size_per_block = 64;
    auto model                                                         = makeGlm53Config();
    model.attn_config.tokens_per_block = model.attn_config.kernel_tokens_per_block = 64;
    auto  config  = HybridPoolConfigCreator::createConfig(model, pc, kv_config, false, 3);
    auto* indexer = dynamic_cast<DSV4KVSpec*>(config.cache_specs[2].get());
    ASSERT_NE(indexer, nullptr);
    EXPECT_EQ(indexer->entries_per_block, 16u);
    EXPECT_EQ(config.group_kv_block_stride_bytes[2], 16u * 132u);
    auto* state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->entries_per_block, 8u);
}

TEST(GLM53CacheConfigTest, PhysicalMlaSpecIsNotExpandedTwice) {
    ParallelismConfig pc;
    auto              model            = makeGlm53Config();
    auto              kv_config        = makeKvConfig();
    model.attn_config.tokens_per_block = 1024;
    kv_config.seq_size_per_block       = 1024;
    auto config                        = HybridPoolConfigCreator::createConfig(model, pc, kv_config, false, 0);
    EXPECT_EQ(config.group_kv_block_stride_bytes[0], config.cache_specs[0]->block_size_bytes());
    EXPECT_EQ(config.group_kv_block_stride_bytes[2], 8u * 32u * 132u);
}

TEST(GLM53CacheConfigTest, RejectsInvalidGeometryAndOwnership) {
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

    kv_config.seq_size_per_block        = 96;
    kv_config.kernel_seq_size_per_block = 96;
    EXPECT_DEATH(HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, kv_config, false, 0), "");
}

TEST(GLM53CacheConfigTest, PrefillTp8AndDecodeDp8UseMatchingKPoolStateRing) {
    constexpr uint32_t tp_size = 8;

    ParallelismConfig prefill_pc;
    prefill_pc.role_type                          = RoleType::PREFILL;
    prefill_pc.tp_size                            = tp_size;
    prefill_pc.ep_size                            = tp_size;
    prefill_pc.world_size                         = tp_size;
    prefill_pc.prefill_cp_config.method           = CPRotateMethod::DISABLED;
    prefill_pc.prefill_cp_config.kv_cache_sharded = true;

    ParallelismConfig decode_pc;
    decode_pc.role_type                          = RoleType::DECODE;
    decode_pc.tp_size                            = 1;
    decode_pc.dp_size                            = tp_size;
    decode_pc.ep_size                            = tp_size;
    decode_pc.world_size                         = tp_size;
    decode_pc.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_pc.prefill_cp_config.kv_cache_sharded = true;
    decode_pc.prefill_cp_config.prefill_cp_size  = tp_size;

    auto prefill_config =
        HybridPoolConfigCreator::createConfig(makeGlm53Config(), prefill_pc, makeKvConfig(), false, 3);
    auto decode_config = HybridPoolConfigCreator::createConfig(makeGlm53Config(), decode_pc, makeKvConfig(), false, 3);

    auto* prefill_kv = dynamic_cast<DSV4KVSpec*>(prefill_config.cache_specs[2].get());
    auto* decode_kv  = dynamic_cast<DSV4KVSpec*>(decode_config.cache_specs[2].get());
    ASSERT_NE(prefill_kv, nullptr);
    ASSERT_NE(decode_kv, nullptr);
    EXPECT_EQ(prefill_kv->entries_per_block, 32u);
    EXPECT_EQ(decode_kv->entries_per_block, prefill_kv->entries_per_block);

    auto* prefill_state = dynamic_cast<DSV4StateSpec*>(prefill_config.cache_specs[3].get());
    auto* decode_state  = dynamic_cast<DSV4StateSpec*>(decode_config.cache_specs[3].get());
    ASSERT_NE(prefill_state, nullptr);
    ASSERT_NE(decode_state, nullptr);
    EXPECT_EQ(prefill_state->entries_per_block, 1u);
    EXPECT_EQ(decode_state->entries_per_block, tp_size);
    EXPECT_EQ(decode_state->entries_per_block, prefill_state->entries_per_block * tp_size);
    EXPECT_EQ(prefill_state->seq_size_per_block, makeKvConfig().seq_size_per_block * tp_size);
    EXPECT_EQ(decode_state->seq_size_per_block, prefill_state->seq_size_per_block);
}

TEST(GLM53CacheConfigTest, DecodeShardedPrefillCpRequiresExplicitCpSize) {
    ParallelismConfig decode_pc;
    decode_pc.role_type                          = RoleType::DECODE;
    decode_pc.dp_size                            = 8;
    decode_pc.ep_size                            = 8;
    decode_pc.world_size                         = 8;
    decode_pc.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_pc.prefill_cp_config.kv_cache_sharded = true;

    EXPECT_DEATH(HybridPoolConfigCreator::createConfig(makeGlm53Config(), decode_pc, makeKvConfig(), false, 0), "");
}

TEST(GLM53CacheConfigTest, SwitchOffPreservesLinearReusePolicy) {
    ScopedEnvVar      request_cache_mode("ENABLE_LINEAR_ATTN_REQUEST_CACHE", "0");
    ParallelismConfig pc;
    auto              kv_config = makeKvConfig();
    kv_config.linear_step       = 4;
    kv_config.linear_fixed_cap  = 6;

    auto config = HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, kv_config, false, 0);

    EXPECT_FALSE(config.enable_linear_attention_request_cache);
    EXPECT_EQ(config.linear_step, 4);
    EXPECT_EQ(config.linear_fixed_cap, 6);
}

TEST(GLM53CacheConfigTest, LinearPoolIsTokenIndependentAndRoleSized) {
    constexpr uint32_t tp_size = 8;
    ScopedEnvVar       request_cache_mode("ENABLE_LINEAR_ATTN_REQUEST_CACHE", "1");

    ParallelismConfig prefill_pc;
    prefill_pc.role_type                          = RoleType::PREFILL;
    prefill_pc.tp_size                            = tp_size;
    prefill_pc.ep_size                            = tp_size;
    prefill_pc.world_size                         = tp_size;
    prefill_pc.prefill_cp_config.method           = CPRotateMethod::DISABLED;
    prefill_pc.prefill_cp_config.kv_cache_sharded = true;

    ParallelismConfig decode_pc;
    decode_pc.role_type                          = RoleType::DECODE;
    decode_pc.tp_size                            = 1;
    decode_pc.dp_size                            = tp_size;
    decode_pc.ep_size                            = tp_size;
    decode_pc.world_size                         = tp_size;
    decode_pc.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_pc.prefill_cp_config.kv_cache_sharded = true;
    decode_pc.prefill_cp_config.prefill_cp_size  = tp_size;

    RuntimeConfig prefill_runtime;
    prefill_runtime.max_generate_batch_size = 64;
    RuntimeConfig decode_runtime;
    decode_runtime.max_generate_batch_size = 8;

    auto prefill_config =
        HybridPoolConfigCreator::createConfig(makeGlm53Config(), prefill_pc, makeKvConfig(), false, 3);
    auto decode_config = HybridPoolConfigCreator::createConfig(makeGlm53Config(), decode_pc, makeKvConfig(), false, 3);

    ASSERT_EQ(prefill_config.group_types[1], CacheGroupType::LINEAR);
    ASSERT_EQ(decode_config.group_types[1], CacheGroupType::LINEAR);
    EXPECT_EQ(prefill_config.linear_block_size_bytes, prefill_config.group_block_size_bytes[1]);
    EXPECT_EQ(decode_config.linear_block_size_bytes, decode_config.group_block_size_bytes[1]);
    EXPECT_EQ(prefill_config.linear_speculative_reserve_step, 4);
    EXPECT_EQ(decode_config.linear_speculative_reserve_step, 4);

    prefill_config.finalizeBlockNums(10000, prefill_runtime);
    decode_config.finalizeBlockNums(10000, decode_runtime);
    EXPECT_EQ(prefill_config.group_block_nums[1], 64u * 4u + 1u);
    EXPECT_EQ(decode_config.group_block_nums[1], 8u * 6u + 1u);
    EXPECT_GE(prefill_config.fixed_pool_reserve_bytes, (64u * 4u + 1u) * prefill_config.linear_block_size_bytes);
    EXPECT_GE(decode_config.fixed_pool_reserve_bytes, (8u * 6u + 1u) * decode_config.linear_block_size_bytes);

    // Paged MLA/indexer capacity may change with the global token budget, but
    // the LINEAR pool remains bounded only by live-request concurrency.
    prefill_config.finalizeBlockNums(20000, prefill_runtime);
    decode_config.finalizeBlockNums(20000, decode_runtime);
    EXPECT_EQ(prefill_config.group_block_nums[1], 64u * 4u + 1u);
    EXPECT_EQ(decode_config.group_block_nums[1], 8u * 6u + 1u);
}

TEST(GLM53CacheConfigTest, LinearRequestPoolBlockOverrideIsAppliedWithLiveRequestFloor) {
    ScopedEnvVar request_cache_mode("ENABLE_LINEAR_ATTN_REQUEST_CACHE", "1");

    ParallelismConfig pc;
    pc.role_type = RoleType::PREFILL;
    RuntimeConfig runtime;
    runtime.max_generate_batch_size = 64;

    auto kv_config = makeKvConfig();
    kv_config.linear_request_cache_pool_blocks = 384;
    auto config = HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, kv_config, false, 3);
    config.finalizeBlockNums(10000, runtime);
    EXPECT_EQ(config.group_block_nums[1], 384u);

    kv_config.linear_request_cache_pool_blocks = 32;
    config = HybridPoolConfigCreator::createConfig(makeGlm53Config(), pc, kv_config, false, 3);
    config.finalizeBlockNums(10000, runtime);
    EXPECT_EQ(config.group_block_nums[1], 193u);
}

TEST(GLM53CacheConfigTest, CompactStateCapacityAndBudgetUseLocalPageCoordinates) {
    initLogger();
    ScopedEnvVar host_budget("RTP_LLM_DSA_MLA_HOST_CACHE_MB", "0");
    for (bool request_cache : {false, true}) {
        ScopedEnvVar request_cache_mode("ENABLE_LINEAR_ATTN_REQUEST_CACHE", request_cache ? "1" : "0");
        for (int cp_size : {1, 4}) {
            for (int step : {2, 3, 8}) {
                for (bool prefill : {false, true}) {
                    SCOPED_TRACE(testing::Message()
                                 << request_cache << "/" << cp_size << "/" << step << "/" << prefill);
                    auto model                         = makeGlm53Config();
                    model.data_type                    = DataType::TYPE_BF16;
                    model.max_seq_len                  = 4096;
                    model.attn_config.tokens_per_block = model.attn_config.kernel_tokens_per_block = 64;
                    auto kv                                                                        = makeKvConfig();
                    kv.seq_size_per_block = kv.kernel_seq_size_per_block = 64;
                    kv.dsv4_fixed_pool_blocks                            = 0;
                    kv.linear_step                                       = step;
                    kv.kv_cache_mem_mb                                   = 1024;
                    ParallelismConfig pc;
                    pc.role_type = prefill ? RoleType::PREFILL : RoleType::DECODE;
                    pc.tp_size   = prefill ? cp_size : 1;
                    pc.dp_size   = prefill ? 1 : cp_size;
                    pc.ep_size = pc.world_size            = cp_size;
                    pc.prefill_cp_config.kv_cache_sharded = cp_size > 1;
                    if (!prefill && cp_size > 1) {
                        pc.prefill_cp_config.method          = CPRotateMethod::PREFILL_CP;
                        pc.prefill_cp_config.prefill_cp_size = cp_size;
                    }
                    RuntimeConfig runtime;
                    runtime.max_generate_batch_size = 1;
                    auto   config          = CacheConfigCreator::createConfig(model, pc, runtime, kv, std::nullopt);
                    size_t allocated_bytes = 0;
                    for (size_t gid = 0; gid < config.group_block_nums.size(); ++gid) {
                        allocated_bytes +=
                            static_cast<size_t>(config.group_block_nums[gid]) * config.group_block_size_bytes[gid];
                    }
                    EXPECT_LE(allocated_bytes, 1024u * 1024u * 1024u);
                    constexpr uint32_t local_pages = 8192;
                    config.finalizeBlockNums(local_pages, runtime);
                    const size_t interval_tokens = 64u * std::lcm(step, cp_size);
                    const size_t capacity_tokens = local_pages * 64u * (prefill ? cp_size : 1);
                    const size_t expected_states =
                        request_cache ? capacity_tokens / interval_tokens : local_pages / step;
                    ASSERT_EQ(config.group_region_names[3], KVCacheRegionName::INDEXER_STATE);
                    EXPECT_EQ(config.group_block_nums[3], expected_states);
                    // Explicit pool sizing remains an intentional user override.
                    config.dsv4_fixed_pool_blocks = 17;
                    config.finalizeBlockNums(local_pages, runtime);
                    EXPECT_EQ(config.group_block_nums[3], 17u);
                }
            }
        }
    }
}

TEST(GLM53CacheConfigTest, DiskCheckpointsReserveTransientBatchesEvenWithSmallPoolOverride) {
    ScopedEnvVar      request_cache_mode("ENABLE_LINEAR_ATTN_REQUEST_CACHE", "1");
    ParallelismConfig pc;
    pc.role_type = RoleType::PREFILL;
    RuntimeConfig runtime;
    runtime.max_generate_batch_size                      = 4;
    runtime.fifo_scheduler_config.max_context_batch_size = 2;
    auto model                                           = makeGlm53Config();
    model.max_seq_len                                    = 1025;
    auto kv_config                                       = makeKvConfig();
    kv_config.reuse_cache                                = true;
    kv_config.enable_memory_cache                        = true;
    kv_config.enable_memory_cache_disk                   = true;
    for (int step : {1, 3, 4}) {
        kv_config.linear_step                      = step;
        kv_config.linear_request_cache_pool_blocks = 0;
        auto           config      = HybridPoolConfigCreator::createConfig(model, pc, kv_config, false, 0);
        const uint32_t checkpoints = 1024 / (128 * step);
        EXPECT_EQ(config.linear_disk_checkpoint_blocks, checkpoints);
        config.finalizeBlockNums(10000, runtime);
        EXPECT_EQ(config.group_block_nums[1], 13u + 4u * checkpoints);
        kv_config.linear_request_cache_pool_blocks = 1;
        config = HybridPoolConfigCreator::createConfig(model, pc, kv_config, false, 0);
        config.finalizeBlockNums(10000, runtime);
        EXPECT_EQ(config.group_block_nums[1], 13u + 4u * checkpoints);
    }
    pc.role_type       = RoleType::DECODE;
    auto decode_config = HybridPoolConfigCreator::createConfig(model, pc, kv_config, false, 0);
    EXPECT_EQ(decode_config.linear_disk_checkpoint_blocks, 0u);
    decode_config.finalizeBlockNums(10000, runtime);
    EXPECT_EQ(decode_config.group_block_nums[1], 13u);
    pc.role_type                       = RoleType::PREFILL;
    kv_config.enable_memory_cache_disk = false;
    auto config                        = HybridPoolConfigCreator::createConfig(model, pc, kv_config, false, 0);
    EXPECT_EQ(config.linear_disk_checkpoint_blocks, 0u);
    config.finalizeBlockNums(10000, runtime);
    EXPECT_EQ(config.group_block_nums[1], 13u);
}

TEST(GLM53CacheConfigTest, OfficialShapeCacheBytesMatchOneMillionTokenAccounting) {
    auto model       = makeGlm53Config();
    model.num_layers = 45;
    model.hybrid_attention_config.hybrid_attention_types.assign(34, HybridAttentionType::LINEAR);
    model.hybrid_attention_config.hybrid_attention_types.insert(
        model.hybrid_attention_config.hybrid_attention_types.end(), 11, HybridAttentionType::NONE);
    model.linear_attention_config.linear_key_head_dim   = 128;
    model.linear_attention_config.linear_value_head_dim = 128;
    model.attn_config.kv_cache_dtype                    = KvCacheDataType::FP8;

    ParallelismConfig prefill_pc;
    prefill_pc.role_type                          = RoleType::PREFILL;
    prefill_pc.tp_size                            = 8;
    prefill_pc.prefill_cp_config.method           = CPRotateMethod::DISABLED;
    prefill_pc.prefill_cp_config.kv_cache_sharded = true;
    auto prefill_config = HybridPoolConfigCreator::createConfig(model, prefill_pc, makeKvConfig(), false, 3);

    ASSERT_EQ(prefill_config.cache_specs.size(), 4u);
    EXPECT_EQ(prefill_config.global_layer_ids[0].size(), 11u);
    EXPECT_EQ(prefill_config.global_layer_ids[1].size(), 34u);
    EXPECT_EQ(prefill_config.global_layer_ids[2].size(), 11u);
    EXPECT_EQ(prefill_config.global_layer_ids[3].size(), 11u);
    auto* mla_spec = dynamic_cast<MLAKVCacheSpec*>(prefill_config.cache_specs[0].get());
    ASSERT_NE(mla_spec, nullptr);
    EXPECT_EQ(11u * mla_spec->block_size_bytes(), 743424u);
    EXPECT_EQ(11u * mla_spec->scale_block_size_bytes(), 0u);
    EXPECT_EQ(prefill_config.group_block_size_bytes[0], 743424u);
    EXPECT_EQ(prefill_config.group_block_size_bytes[1], 18452480u);
    EXPECT_EQ(prefill_config.group_block_size_bytes[2], 46464u);

    ParallelismConfig decode_pc;
    decode_pc.role_type                          = RoleType::DECODE;
    decode_pc.tp_size                            = 1;
    decode_pc.dp_size                            = 8;
    decode_pc.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_pc.prefill_cp_config.kv_cache_sharded = true;
    decode_pc.prefill_cp_config.prefill_cp_size  = 8;
    auto decode_config = HybridPoolConfigCreator::createConfig(model, decode_pc, makeKvConfig(), false, 3);
    EXPECT_EQ(decode_config.group_block_size_bytes[1], 147619840u);
}

}  // namespace test
}  // namespace rtp_llm
