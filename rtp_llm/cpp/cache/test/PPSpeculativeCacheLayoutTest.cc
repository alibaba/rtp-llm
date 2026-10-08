#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {
namespace {
ModelConfig targetConfig() {
    ModelConfig model;
    model.model_type                                                = "deepseek_v4";
    model.num_layers                                                = 4;
    model.max_seq_len                                               = 32768;
    model.hidden_size                                               = 4096;
    model.data_type                                                 = DataType::TYPE_BF16;
    model.attn_config.head_num                                      = 64;
    model.attn_config.kv_head_num                                   = 1;
    model.attn_config.size_per_head                                 = 512;
    model.attn_config.indexer_head_dim                              = 128;
    model.attn_config.tokens_per_block                              = 256;
    model.attn_config.kv_cache_dtype                                = KvCacheDataType::FP8;
    model.hybrid_attention_config.enable_hybrid_attention           = true;
    model.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(model, {4, 128, 4, 128});
    return model;
}

ParallelismConfig prefillStage(int rank, int cp_size = 2) {
    ParallelismConfig pc;
    pc.pp_size                           = 2;
    pc.pp_rank                           = rank;
    pc.pp_stage_layer_counts             = {2, 2};
    pc.tp_size                           = cp_size;
    pc.ep_size                           = cp_size;
    pc.world_size                        = 2 * cp_size;
    pc.world_rank                        = cp_size * rank;
    pc.role_type                         = RoleType::PREFILL;
    pc.prefill_cp_config.method          = CPRotateMethod::PREFILL_CP;
    pc.prefill_cp_config.prefill_cp_size = cp_size;
    return pc;
}

KVCacheConfig pageConfig() {
    KVCacheConfig kv;
    kv.seq_size_per_block        = 256;
    kv.kernel_seq_size_per_block = 128;
    kv.test_block_num            = 8;
    return kv;
}

size_t stride(const CacheConfig& config, const std::string& tag) {
    return config.kvBlockStrideBytesForGroup(config.groupIdForTag(tag));
}
}  // namespace

TEST(PPSpeculativeCacheLayout, EarlierStageUsesGlobalSpeculativeWidthWithoutDraft) {
    const auto model = targetConfig();
    auto       draft = model;
    draft.num_layers = 3;
    setDsv4KvCacheSpecs(draft, {0, 0, 0});
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_DSPARK;
    sp.gen_num_per_cycle = 3;

    // Real factories used by NormalEngine: earlier stages have no local
    // draft, whereas the last stage owns/merges the draft's cache pools.
    const auto first =
        CacheConfigCreator::createConfig(model, prefillStage(0), RuntimeConfig{}, pageConfig(), std::nullopt, sp);
    const auto last = CacheConfigCreator::createSpConfig(
        model, draft, prefillStage(1), RuntimeConfig{}, pageConfig(), sp, std::nullopt, true, false);
    EXPECT_TRUE(first.mtp_sub_configs.empty());
    ASSERT_EQ(last.mtp_sub_configs.size(), 1u);
    EXPECT_EQ(first.layer_num, 2u);
    EXPECT_EQ(first.global_layer_begin, 0u);
    EXPECT_EQ(last.global_layer_begin, 2u);
    for (const auto& group : first.topology().groups()) {
        EXPECT_EQ(stride(first, group.tag), stride(last, group.tag)) << group.tag;
    }
    // Exact on-wire values from the CEP2PP2 gamma3 failure, not token counts.
    EXPECT_EQ(stride(first, "swa_kv"), 77184u);
    EXPECT_EQ(stride(first, "hca_state"), 540672u);
}

TEST(PPSpeculativeCacheLayout, AbsentSpeculativeConfigKeepsNonSpeculativeLayout) {
    const auto config =
        CacheConfigCreator::createConfig(targetConfig(), prefillStage(0), RuntimeConfig{}, pageConfig());
    EXPECT_EQ(stride(config, "swa_kv"), 74880u);
    EXPECT_EQ(stride(config, "hca_state"), 524288u);
    EXPECT_TRUE(config.mtp_sub_configs.empty());
}

TEST(PPSpeculativeCacheLayout, CEP4StagesMatchRemoteCP4DecodePayloads) {
    const auto model = targetConfig();
    auto       draft = model;
    draft.num_layers = 3;
    setDsv4KvCacheSpecs(draft, {0, 0, 0});
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_DSPARK;
    sp.gen_num_per_cycle = 3;

    const auto first =
        CacheConfigCreator::createConfig(model, prefillStage(0, 4), RuntimeConfig{}, pageConfig(), std::nullopt, sp);
    const auto last = CacheConfigCreator::createSpConfig(
        model, draft, prefillStage(1, 4), RuntimeConfig{}, pageConfig(), sp, std::nullopt, true, false);
    ParallelismConfig decode;
    decode.role_type = RoleType::DECODE;
    decode.dp_size = decode.ep_size = decode.world_size = 8;
    decode.prefill_cp_config.method                     = CPRotateMethod::PREFILL_CP;
    decode.prefill_cp_config.prefill_cp_size            = 4;
    const auto remote                                   = CacheConfigCreator::createSpConfig(
        model, draft, decode, RuntimeConfig{}, pageConfig(), sp, std::nullopt, true, false);
    EXPECT_TRUE(first.mtp_sub_configs.empty());
    ASSERT_EQ(last.mtp_sub_configs.size(), 1u);
    ASSERT_EQ(remote.mtp_sub_configs.size(), 1u);
    for (const auto& group : first.topology().groups()) {
        EXPECT_EQ(stride(first, group.tag), stride(last, group.tag)) << group.tag;
        EXPECT_EQ(stride(first, group.tag), stride(remote, group.tag)) << group.tag;
    }
    for (const auto& group : last.mtp_sub_configs.front()->topology().groups()) {
        EXPECT_EQ(stride(*last.mtp_sub_configs.front(), group.tag), stride(*remote.mtp_sub_configs.front(), group.tag))
            << group.tag;
    }
}

TEST(PPSpeculativeCacheLayout, DisabledSpeculativeConfigIgnoresStaleWidth) {
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_NONE;
    sp.gen_num_per_cycle = 3;
    const auto config    = CacheConfigCreator::createConfig(
        targetConfig(), prefillStage(0), RuntimeConfig{}, pageConfig(), std::nullopt, sp);
    EXPECT_EQ(stride(config, "swa_kv"), 74880u);
    EXPECT_EQ(stride(config, "hca_state"), 524288u);
}

TEST(PPSpeculativeCacheLayout, ZeroWidthKeepsNonSpeculativePayloadSize) {
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_DSPARK;
    sp.gen_num_per_cycle = 0;
    const auto config    = CacheConfigCreator::createConfig(
        targetConfig(), prefillStage(0), RuntimeConfig{}, pageConfig(), std::nullopt, sp);
    EXPECT_EQ(stride(config, "swa_kv"), 74880u);
    EXPECT_EQ(stride(config, "hca_state"), 524288u);
}
}  // namespace rtp_llm::test
