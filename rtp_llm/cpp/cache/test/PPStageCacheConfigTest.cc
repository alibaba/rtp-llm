#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm::test {

namespace {

ModelConfig makeHybridModelConfig(int64_t num_layers) {
    ModelConfig config;
    config.num_layers                   = num_layers;
    config.max_seq_len                  = 128;
    config.hidden_size                  = 64;
    config.vocab_size                   = 1024;
    config.data_type                    = DataType::TYPE_FP16;
    config.attn_config.head_num         = 2;
    config.attn_config.kv_head_num      = 2;
    config.attn_config.size_per_head    = 16;
    config.attn_config.tokens_per_block = 4;
    config.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;

    config.linear_attention_config.linear_conv_kernel_dim = 2;
    config.linear_attention_config.linear_key_head_dim    = 8;
    config.linear_attention_config.linear_value_head_dim  = 8;
    config.linear_attention_config.linear_num_key_heads   = 2;
    config.linear_attention_config.linear_num_value_heads = 2;

    config.hybrid_attention_config.enable_hybrid_attention = true;
    config.hybrid_attention_config.hybrid_attention_types.resize(static_cast<size_t>(num_layers));
    config.kv_cache_spec_descs.assign(static_cast<size_t>(num_layers), {});
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        const bool linear = layer % 4 != 3;
        config.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(layer)] =
            linear ? HybridAttentionType::LINEAR : HybridAttentionType::NONE;
        config.kv_cache_spec_descs[static_cast<size_t>(layer)].push_back(
            linear ? KVCacheSpecDesc{"linear", KVCacheSpecType::LinearAttention} :
                     KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention});
    }
    return config;
}

ModelConfig makeSingleModelConfig(int64_t num_layers) {
    ModelConfig config;
    config.num_layers                   = num_layers;
    config.max_seq_len                  = 128;
    config.hidden_size                  = 64;
    config.vocab_size                   = 1024;
    config.data_type                    = DataType::TYPE_FP16;
    config.attn_config.head_num         = 2;
    config.attn_config.kv_head_num      = 2;
    config.attn_config.size_per_head    = 16;
    config.attn_config.tokens_per_block = 4;
    config.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;
    config.kv_cache_spec_descs.assign(static_cast<size_t>(num_layers),
                                      {KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention}});
    return config;
}

ParallelismConfig makePpConfig(int64_t num_layers, int64_t pp_size, int64_t pp_rank) {
    ParallelismConfig config;
    config.pp_size         = pp_size;
    config.pp_rank         = pp_rank;
    const int64_t base     = num_layers / pp_size;
    const int64_t remainder = num_layers % pp_size;
    for (int64_t stage = 0; stage < pp_size; ++stage) {
        config.pp_stage_layer_counts.push_back(base + (stage < remainder ? 1 : 0));
    }
    return config;
}

size_t countLayersOfType(const CacheConfig& config, CacheGroupType type) {
    size_t count = 0;
    for (const auto& group : config.groups()) {
        if (group.policy.group_type == type) {
            count += config.layerIdsForGroup(group.tag).size();
        }
    }
    return count;
}

KVCacheSpecDesc makeOpaqueStateDesc(const std::string& tag) {
    KVCacheSpecDesc desc;
    desc.tag                  = tag;
    desc.cache_type           = KVCacheSpecType::OpaqueState;
    desc.entry_elems          = 16;
    desc.entry_dtype          = DataType::TYPE_FP32;
    desc.explicit_entry_count = 4;
    return desc;
}

}

TEST(PPStageCacheConfigTest, PpOneKeepsWholeModel) {
    const auto model  = makeHybridModelConfig(8);
    const auto staged = CacheConfigCreator::stageScopedModelConfig(model, ParallelismConfig{});
    EXPECT_EQ(staged.num_layers, 8);
    EXPECT_EQ(staged.global_layer_begin, 0u);
    EXPECT_EQ(staged.kv_cache_spec_descs.size(), 8u);
    EXPECT_EQ(staged.hybrid_attention_config.hybrid_attention_types.size(), 8u);
}

TEST(PPStageCacheConfigTest, StageScopeFollowsRankLayout) {
    const auto model = makeHybridModelConfig(65);
    const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 17}, {17, 33}, {33, 49}, {49, 65}};
    for (int64_t rank = 0; rank < 4; ++rank) {
        const auto staged       = CacheConfigCreator::stageScopedModelConfig(model, makePpConfig(65, 4, rank));
        const auto [begin, end] = expected[static_cast<size_t>(rank)];
        EXPECT_EQ(staged.num_layers, end - begin);
        EXPECT_EQ(staged.global_layer_begin, static_cast<uint32_t>(begin));
        ASSERT_EQ(staged.kv_cache_spec_descs.size(), static_cast<size_t>(end - begin));
        ASSERT_EQ(staged.hybrid_attention_config.hybrid_attention_types.size(), static_cast<size_t>(end - begin));
        for (int64_t local = 0; local < end - begin; ++local) {
            EXPECT_EQ(staged.kv_cache_spec_descs[static_cast<size_t>(local)][0].tag,
                      model.kv_cache_spec_descs[static_cast<size_t>(begin + local)][0].tag);
        }
    }
}

TEST(PPStageCacheConfigTest, StageScopeRejectsInvalidOrEmptyStage) {
    const auto model = makeHybridModelConfig(8);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(model, makePpConfig(8, 2, 2)), std::exception);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(model, makePpConfig(8, 2, -1)), std::exception);
    const auto tiny = makeHybridModelConfig(2);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(tiny, makePpConfig(2, 4, 2)), std::exception);
}

TEST(PPStageCacheConfigTest, IndependentPoolsUseStageLocalGeometry) {
    const auto model = makeHybridModelConfig(8);
    for (int64_t rank = 0; rank < 2; ++rank) {
        const auto config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, rank), KVCacheConfig{});
        EXPECT_EQ(config.layer_num, 4u);
        EXPECT_EQ(config.layer_all_num(), 4u);
        EXPECT_EQ(config.global_layer_begin, static_cast<uint32_t>(rank) * 4u);
        EXPECT_EQ(config.topology().layers().size(), 4u);
        EXPECT_EQ(countLayersOfType(config, CacheGroupType::LINEAR), 3u);
        EXPECT_EQ(countLayersOfType(config, CacheGroupType::FULL), 1u);
    }
}

TEST(PPStageCacheConfigTest, CreateConfigScopesBeforeBuildingTopology) {
    const auto model = makeSingleModelConfig(8);
    KVCacheConfig options;
    const auto config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 1), options);
    EXPECT_EQ(config.layer_num, 4u);
    EXPECT_EQ(config.global_layer_begin, 4u);
    ASSERT_EQ(config.groupNums(), 1);
    EXPECT_EQ(config.layerIdsForGroup("full"), (std::vector<int>{0, 1, 2, 3}));
}

TEST(PPStageCacheConfigTest, WarmupUsesStageLocalTopologyAndSmallFinalizedPools) {
    const auto model  = makeHybridModelConfig(8);
    const auto config = CacheConfigCreator::createWarmupConfig(model, makePpConfig(8, 2, 1), KVCacheConfig{}, 0);
    EXPECT_EQ(config.layer_num, 4u);
    EXPECT_EQ(config.global_layer_begin, 4u);
    EXPECT_EQ(config.linear_step, 1);
    for (const auto& group : config.groups()) {
        EXPECT_EQ(group.block_num, 2u);
    }
}

TEST(PPStageCacheConfigTest, PpAllowsOpaqueStateAndSwaGroups) {
    auto model = makeSingleModelConfig(8);
    model.attn_config.sliding_window = 32;
    model.kv_cache_spec_descs[0]     = {makeOpaqueStateDesc("state")};

    const auto config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 0), KVCacheConfig{});
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_TRUE(config.use_opaque_kv_cache_store);
    EXPECT_EQ(config.group("state").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("state").policy.sliding_window_size, 32);
}

TEST(PPStageCacheConfigTest, TransferPolicySurvivesSingletonStagesAndWarmup) {
    for (bool hybrid : {false, true}) {
        const auto model = hybrid ? makeHybridModelConfig(8) : makeSingleModelConfig(8);
        for (int64_t rank : {0, 1, 2}) {
            auto layout = makePpConfig(8, 3, rank);
            layout.pp_stage_layer_counts = {4, 3, 1};
            const auto config = CacheConfigCreator::createConfig(model, layout, KVCacheConfig{});
            const auto warmup = CacheConfigCreator::createWarmupConfig(model, layout, KVCacheConfig{}, 0);
            EXPECT_EQ(config.groupNums(), hybrid && rank == 0 ? 2 : 1);
            EXPECT_EQ(warmup.groupNums(), config.groupNums());
            EXPECT_EQ(config.usesGroupCacheTransferPolicy(), hybrid);
            EXPECT_EQ(warmup.usesGroupCacheTransferPolicy(), hybrid);
        }
    }
}

TEST(PPStageCacheConfigTest, DraftInheritsWholeModelTransferPolicyOnLastStage) {
    auto layout = makePpConfig(8, 2, 1);
    layout.pp_stage_layer_counts = {7, 1};
    const auto draft = makeSingleModelConfig(1);
    SpeculativeExecutionConfig speculative;
    speculative.type = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;
    for (bool hybrid : {false, true}) {
        const auto target = hybrid ? makeHybridModelConfig(8) : makeSingleModelConfig(8);
        const auto config = CacheConfigCreator::createConfig(
            target, layout, KVCacheConfig{}, speculative, &draft, true, false);
        ASSERT_EQ(config.groupNums(), 1);
        ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
        for (const auto& sub : config.mtp_sub_configs) {
            ASSERT_EQ(sub->groupNums(), 1);
            EXPECT_EQ(sub->usesGroupCacheTransferPolicy(), hybrid);
        }
    }

    auto target = CacheConfigCreator::createConfig(makeHybridModelConfig(8), ParallelismConfig{}, KVCacheConfig{});
    target.model_has_multiple_cache_groups = false;
    ASSERT_EQ(target.groupNums(), 2);
    const auto draft_config = CacheConfigCreator::createConfig(draft, ParallelismConfig{}, KVCacheConfig{});
    const auto sub = target.mergeMTPModule(draft_config, 0, target.layer_num);
    EXPECT_TRUE(sub->model_has_multiple_cache_groups);
}

TEST(PPStageCacheConfigTest, StageZeroMustOwnEveryCrossStageTag) {
    auto model = makeHybridModelConfig(32);
    int64_t linear_seen = 0;
    for (auto& descs : model.kv_cache_spec_descs) {
        if (descs[0].cache_type == KVCacheSpecType::LinearAttention) {
            descs[0].tag = "linear" + std::to_string(linear_seen++ / 8);
        }
    }

    auto first = CacheConfigCreator::createConfig(model, makePpConfig(32, 2, 0), KVCacheConfig{});
    auto last  = CacheConfigCreator::createConfig(model, makePpConfig(32, 2, 1), KVCacheConfig{});
    first.finalizeBlockNums(100, RuntimeConfig{});
    last.finalizeBlockNums(100, RuntimeConfig{});
    const auto result = validatePPTopology({StageCacheSnapshot::fromConfig(first), StageCacheSnapshot::fromConfig(last)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("absent from stage 0"), std::string::npos);
}

TEST(PPStageCacheConfigTest, PerTagOverridesReplaceDerivedCounts) {
    const auto model = makeHybridModelConfig(8);
    auto config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 0), KVCacheConfig{});

    PPBlockNumOverrides overrides{{"full", 60u}, {"linear", 70u}};
    config.finalizeBlockNums(60, RuntimeConfig{}, &overrides);
    EXPECT_EQ(config.group("full").block_num, 60u);
    EXPECT_EQ(config.group("linear").block_num, 70u);
}

TEST(PPStageCacheConfigTest, DraftCacheExistsOnlyOnLastStageAndIsNotSliced) {
    const auto target = makeSingleModelConfig(7);
    const auto draft  = makeSingleModelConfig(2);
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;

    EXPECT_THROW(CacheConfigCreator::createConfig(
                     target, makePpConfig(7, 3, 1), KVCacheConfig{}, speculative, &draft, true, false),
                 std::exception);

    auto last_layout = makePpConfig(7, 3, 2);
    last_layout.pp_stage_layer_counts = {2, 3, 2};
    auto config = CacheConfigCreator::createConfig(
        target, last_layout, KVCacheConfig{}, speculative, &draft, true, false);
    EXPECT_EQ(config.layer_num, 2u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_TRUE(sub_config);
        EXPECT_EQ(sub_config->layer_num, 2u);
        EXPECT_EQ(sub_config->global_layer_begin, 0u);
    }
}

TEST(PPStageCacheConfigTest, PerTagOverridesReachEveryDraftModule) {
    const auto draft  = makeSingleModelConfig(2);
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;
    for (const bool hybrid : {false, true}) {
        SCOPED_TRACE(hybrid);
        const auto target = hybrid ? makeHybridModelConfig(8) : makeSingleModelConfig(4);
        auto config = CacheConfigCreator::createConfig(
            target, makePpConfig(target.num_layers, 2, 1), KVCacheConfig{}, speculative, &draft, true, false);

        /** Distinct tag capacities expose a missing override on the recursive draft path. */
        const uint32_t full_blocks = hybrid ? 32u : 24u;
        PPBlockNumOverrides overrides{{"full", full_blocks}};
        if (hybrid) {
            overrides.emplace("linear", 24u);
        }
        config.finalizeBlockNums(24, RuntimeConfig{}, &overrides);
        EXPECT_EQ(config.group("full").block_num, full_blocks);
        if (hybrid) {
            EXPECT_EQ(config.group("linear").block_num, 24u);
        }
        ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
        for (const auto& sub_config : config.mtp_sub_configs) {
            ASSERT_NE(sub_config, nullptr);
            EXPECT_EQ(sub_config->group("full").block_num, full_blocks);
        }

        NegotiatedCapacity agreed;
        agreed.paged_block_num     = 24;
        agreed.block_num_overrides = overrides;
        EXPECT_NO_THROW(validatePPComposedBlockNums(config, agreed));
        config.mtp_sub_configs.front()->finalizeBlockNums(23, RuntimeConfig{});
        EXPECT_THROW(validatePPComposedBlockNums(config, agreed), std::exception);
    }
}

TEST(PPStageCacheConfigTest, MissingPerTagOverrideFails) {
    const auto model = makeHybridModelConfig(8);
    auto config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 0), KVCacheConfig{});
    PPBlockNumOverrides incomplete{{"full", 60u}};
    EXPECT_THROW(config.finalizeBlockNums(60, RuntimeConfig{}, &incomplete), std::exception);
}

}  // namespace rtp_llm::test
